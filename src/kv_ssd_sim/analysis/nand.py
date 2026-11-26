"""Verify FEMU NAND counters against guest diskstats and same-window block I/O."""

import argparse
import json
from collections import Counter
from itertools import pairwise
from pathlib import Path

from kv_ssd_sim.analysis.aiperf import read_jsonl
from kv_ssd_sim.analysis.block_capture import check_receipt
from kv_ssd_sim.analysis.block_io import check, parse_trace

PROGRAM = (
    "nand-write-pages",
    "gc-write-pages",
    "map-write-pages",
    "pad-write-pages",
    "parity-write-pages",
)
REQUIRED = ("host-write-pages", "nand-write-pages", "gc-write-pages", "block-erases")
PROBE = ("empty", "page", "partial", "read", "filled", "rewritten")


def read(path):
    return json.loads(path.read_text())


def sample(path, label):
    guest = read(path / f"nand-{label}.guest.json")
    qmp = read(path / f"nand-{label}.qmp.json")
    left, right = guest["stat_before"], guest["stat_after"]
    check(guest["label"] == label and guest["before_ns"] < guest["after_ns"], "Wrong guest marker")
    check(
        len(left) >= 17
        and len(right) >= 17
        and left[8] == right[8] == 0
        and all(left[i] == right[i] for i in (0, 2, 4, 6, 11, 13, 15)),
        "I/O crossed NAND snapshot",
    )
    (ns,) = qmp["namespaces"]
    check(qmp["mode"] == ns["mode"] == "bbssd" and ns["nsid"] == 1, "Expected one BBSSD namespace")
    geom, lines, counters = ns["geometry"], ns["lines"], ns["counters"]
    check(all(k in counters for k in REQUIRED), "Missing NAND counter")
    check(
        all(
            type(counters[k]) is int and counters[k] >= 0
            for k in {*REQUIRED, *PROGRAM} & counters.keys()
        ),
        "Invalid NAND counter",
    )
    n = ns["line-counts"]["total"]
    check(
        "next-offset" not in ns
        and ns["offset"] == 0
        and [l["id"] for l in lines] == list(range(n)),
        "Incomplete FTL line snapshot",
    )
    per_line = geom["pages-per-line"]
    blocks_per_line, rem = divmod(per_line, geom["pages-per-block"])
    raw = n * per_line
    check(
        not rem
        and raw > 0
        and raw
        == (
            geom["channels"]
            * geom["luns-per-channel"]
            * geom["planes-per-lun"]
            * geom["blocks-per-plane"]
            * geom["pages-per-block"]
        ),
        "Invalid NAND geometry",
    )
    for line in lines:
        check(
            0 <= line["vpc"]
            and 0 <= line["ipc"]
            and line["vpc"] + line["ipc"] <= per_line
            and 0 <= line["erase-min"] <= line["erase-max"],
            "Invalid line counters",
        )
    low, high = (sum(l[k] for l in lines) * blocks_per_line for k in ("erase-min", "erase-max"))
    check(low <= counters["block-erases"] <= high, "Erase count outside line wear bounds")
    valid = sum(l["vpc"] for l in lines)
    invalid = sum(l["ipc"] for l in lines)
    page = geom["page-size"]
    check(guest["namespace_sectors"] > 0 and page > 0, "Invalid device capacity")
    return {
        **guest,
        "geometry": geom,
        "cumulative": {
            k: counters.get(k, 0) for k in (*PROGRAM, "host-write-pages", "block-erases")
        },
        "raw_pages": raw,
        "valid_pages": valid,
        "invalid_pages": invalid,
        "physical_occupied_fraction": (valid + invalid) / raw,
        "valid_data_over_namespace": valid * page / (guest["namespace_sectors"] * 512),
        "line_counts": ns["line-counts"],
        "min_block_wear": min(l["erase-min"] for l in lines),
        "max_block_wear": max(l["erase-max"] for l in lines),
        "wear_sum_bounds": [low, high],
    }


def interval(a, b):
    check(
        a["after_ns"] <= b["before_ns"]
        and a["geometry"] == b["geometry"]
        and a["namespace_sectors"] == b["namespace_sectors"],
        "Mismatched snapshot boundaries",
    )
    d = {k: v - a["cumulative"][k] for k, v in b["cumulative"].items()}
    disk = {
        k: (b["stat_before"][idx] - a["stat_after"][idx]) * 512
        for k, idx in (("read", 2), ("write", 6))
    }
    check(min(*d.values(), *disk.values()) >= 0, "Counters decreased within window")
    page = a["geometry"]["page-size"]
    programmed = sum(d[k] for k in PROGRAM) * page
    occupied_delta = b["valid_pages"] + b["invalid_pages"] - a["valid_pages"] - a["invalid_pages"]
    # The experiment's default page-mapped BBSSD consumes one free page per program
    # and releases one full block per erase. Verify with separately queried line states.
    check(
        occupied_delta == programmed // page - d["block-erases"] * a["geometry"]["pages-per-block"],
        "Program/erase deltas disagree with occupied pages",
    )
    return {
        "start": a["label"],
        "end": b["label"],
        "host_write_bytes": disk["write"],
        "host_read_bytes": disk["read"],
        "nand_program_bytes": programmed,
        "gc_relocation_bytes": d["gc-write-pages"] * page,
        "block_erases": d["block-erases"],
        "page_deltas": {k: v for k, v in d.items() if k != "block-erases"},
        "byte_waf": programmed / disk["write"] if disk["write"] else None,
    }


def check_probe(samples, windows):
    check(all(v == 0 for v in samples[0]["cumulative"].values()), "Probe requires fresh NAND")
    for w, n in zip(windows, (4096, 512, 0, 48 * 2**20, 72 * 2**20), strict=True):
        check(w["host_write_bytes"] == n, "Probe write bytes differ from authored sequence")
    page, partial, read_only, fill, rewrite = windows
    check(
        page["byte_waf"] == 1
        and partial["byte_waf"] == 8
        and read_only["byte_waf"] is None
        and read_only["host_read_bytes"] == 4096
        and read_only["nand_program_bytes"] == 0,
        "Aligned/partial/read controls failed",
    )
    check(
        all(w["gc_relocation_bytes"] == w["block_erases"] == 0 for w in windows[:4])
        and fill["byte_waf"] == 1,
        "Unexpected GC or extra programs during prefill",
    )
    check(
        rewrite["block_erases"] > 0
        and rewrite["gc_relocation_bytes"] > 0
        and rewrite["byte_waf"] > 1
        and samples[-1]["valid_data_over_namespace"] == 1
        and samples[-1]["wear_sum_bounds"] == [samples[-1]["cumulative"]["block-erases"]] * 2,
        "GC/erase positive control or wear sum failed",
    )


def check_trace(path, samples, window):
    receipt = read(path / "capture.json")
    count = check_receipt(receipt, receipt["device"], receipt["mode"])
    blocks, integrity = parse_trace(
        (path / "block-trace.txt").read_text(), receipt["device"], stream_entries=count
    )
    check(not integrity["error_completions"], "Errors in NVMe completions")
    marks = read_jsonl(path / "diskstats.jsonl")
    begin = min(
        r["start_perf_ns"] for r in read_jsonl(path / "joint/client/profile_export_raw.jsonl")
    )
    end = marks[-1]["after_ns"]
    a, b = samples
    check(
        marks[-1]["label"] == "device_end"
        and a["after_ns"] <= begin < b["before_ns"]
        and b["after_ns"] <= end,
        "Workload and QMP windows disagree",
    )
    # Account for the gaps between the QMP sampling brackets and the HTTP/collector markers.
    check(
        not any(
            a["before_ns"] - 1000 <= e.time <= begin + 1000
            or b["before_ns"] - 1000 <= e.time <= end + 1000
            for e in blocks
        ),
        "Block traffic between NAND and workload boundaries",
    )
    totals = Counter()
    for e in blocks:
        if begin <= e.time < end:
            totals[e.op, e.stage] += e.sectors * 512
    for op, name in (("R", "read"), ("W", "write")):
        check(
            totals[op, "issue"] == totals[op, "complete"] == window[f"host_{name}_bytes"],
            "Block trace and diskstats differ for " + op,
        )
    return {
        "first_http_send_ns": begin,
        "device_end_ns": end,
        "target": receipt["device"],
        "block_issued_write_bytes": totals["W", "issue"],
        "block_completed_write_bytes": totals["W", "complete"],
        "block_issued_read_bytes": totals["R", "issue"],
    }


def verify(path):
    launch = read(path / "launch.json")
    check(
        launch["nand_stats"] and (path / "guest-exit-code.txt").read_text().strip() == "0",
        "Missing NAND instrumentation or failed guest",
    )
    probe = launch["workload"] == "nand-probe"
    labels = PROBE if probe else ("first_send", "device_end")
    samples = [sample(path, k) for k in labels]
    command = launch["command"]
    spec = command[command.index("-device") + 1].split(",")
    dev = dict(part.split("=", 1) for part in spec[1:])
    first = samples[0]
    check(
        spec[0] == "femu"
        and dev["namespaces"] == "1"
        and dev["femu_mode"] == "1"
        and int(dev["devsz_mb"]) * 2**20 == first["namespace_sectors"] * 512
        and int(dev["secsz"]) * int(dev["secs_per_pg"]) == first["geometry"]["page-size"]
        and all(
            int(dev[k]) == first["geometry"][field]
            for k, field in (
                ("pgs_per_blk", "pages-per-block"),
                ("blks_per_pl", "blocks-per-plane"),
                ("nchs", "channels"),
                ("luns_per_ch", "luns-per-channel"),
                ("pls_per_lun", "planes-per-lun"),
            )
        ),
        "Observed capacity or geometry differs from launch",
    )
    windows = [interval(a, b) for a, b in pairwise(samples)]
    report = {"trial": path.name, "samples": samples, "windows": windows}
    if probe:
        check_probe(samples, windows)
    else:
        report["block_reconciliation"] = check_trace(path, samples, windows[0])
    (path / "nand-report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trial", type=Path)
    args = parser.parse_args()
    print(json.dumps(verify(args.trial)["windows"], indent=2))


if __name__ == "__main__":
    main()
