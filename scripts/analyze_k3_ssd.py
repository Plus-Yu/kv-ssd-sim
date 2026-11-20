"""Verify the FEMU K3 online trial and matched C2 repetitions."""

import argparse
import csv
import hashlib
import json
import re
from collections import Counter, defaultdict
from itertools import accumulate
from pathlib import Path
from statistics import median

from analyze_b2 import inspect, matched_settings, read
from measure_ssd import MIB, NS, check
from verify_aiperf import read_jsonl

ROOT = Path(__file__).resolve().parents[1]


def transfer_ledger(events, plan):
    groups = {g["group"]: g for g in plan["groups"]}
    started = {}
    current = peak = 0
    objects = {}
    boundary = defaultdict(set)
    ranks = set()
    for event in events:
        if event["event"] not in ("store_put_begin", "store_put_end"):
            continue
        batch = tuple(event["keys"])
        if event["event"] == "store_put_begin":
            check(batch not in started, "Overlapping identical PUT batches")
            started[batch] = event["bytes"]
            current += event["bytes"]
            peak = max(peak, current)
            continue
        rank = event["logical_rank"]
        check(rank in range(8), "Missing logical worker rank")
        nbytes = started.pop(batch)
        check(nbytes == sum(event["object_bytes"]), "PUT byte ledger differs")
        current -= nbytes
        for k, n, h, rc in zip(
            batch, event["object_bytes"], event["digests"], event["result"], strict=True
        ):
            match = re.search(r"@tp_rank:(\d+)@pcp0@dcp0@pp_rank:0@group:(\d+)@([a-f0-9]+)$", k)
            check(rc == 0 and match and "@store_tp:8@" in k, "Failed or malformed TP8 PUT")
            shard, gid = int(match[1]), int(match[2])
            check(
                n == groups[gid]["logical_bytes_per_object_per_shard"]
                and re.fullmatch("[a-f0-9]{64}", h),
                "Size or digest mismatch",
            )
            check(objects.setdefault(k, (n, h)) == (n, h), "Duplicate key changed")
            if groups[gid]["spec"] == "MLAAttentionSpec":
                check(shard == 0 and "@store_format:attention_head_major@" in k, "MLA not shared")
            else:
                check(shard == rank and "@store_format:mamba_state@" in k, "KDA rank mismatch")
                boundary[match[3]].add((gid, rank))
                ranks.add(rank)
    check(not started and current == 0 and ranks == set(range(8)), "Incomplete transfer lifecycle")
    required = {
        (gid, r) for gid, g in groups.items() if g["spec"] != "MLAAttentionSpec" for r in range(8)
    }
    check(all(shards == required for shards in boundary.values()), "Incomplete KDA snapshot")
    inventory = Counter(n for n, h in objects.values())
    check(len(inventory) == 2, "Missing MLA or KDA")
    return {
        "logical_ranks": sorted(ranks),
        "counts_by_object_bytes": dict(inventory),
        "unique_keys": len(objects),
        "logical_bytes": sum(n for n, h in objects.values()),
        "complete_kda_boundaries": len(boundary),
        "max_inflight_put_bytes": peak,
        "key_content_sha256": hashlib.sha256(
            json.dumps(sorted(objects.items())).encode()
        ).hexdigest(),
    }


def online(path):
    run, bins, issues = inspect(path, "stream")
    events = read_jsonl(path / "joint/server/events.jsonl")
    source = (ROOT / "references/kimi-k3-b300/trace_0504.source.json").read_bytes()
    selection = read(ROOT / "datasets/k3-c2/selection.json")
    stage = read(path / "input/trace.json")
    check(
        hashlib.sha256(source).hexdigest() == selection["sha256"]
        and stage["requests"] == json.loads(source)["requests"][:2],
        "Trace provenance mismatch",
    )
    plan = read(ROOT / "configs/kimi-k3-b300-step2/resolved.json")
    init = next(e for e in events if e["event"] == "hybrid_initialized")
    check(
        init["geometry"] == plan["geometry"]
        and init["layers"] == 93
        and init["logical_ranks"] == 8,
        "K3 model does not match resolved specs",
    )
    check(not read(path / "joint/server/summary.json")["checksum_errors"], "Checksum errors")
    ledger = transfer_ledger(events, plan)
    check(
        sum(v for k, v in run["final_replica_states"].items() if "LOCAL_DISK" in k)
        == ledger["unique_keys"],
        "Final disk replicas incomplete",
    )
    capacities = read_jsonl(path / "joint/server/capacity.jsonl")
    check(len(capacities) > 1, "Missing resource measurements")
    updates = [e for e in events if e["event"] == "compute_begin"]
    checkpoints = sorted(
        {(e["request_id"], e["boundary_tokens"]) for e in events if e["event"] == "k3_checkpoint"}
    )
    cap = {
        "pool_reserved_bytes_per_rank": init["pool_bytes_per_rank"],
        "peak_blocks_per_rank_including_null": max(
            e["active_pool_blocks_per_rank"] for e in updates
        ),
        "peak_digest_bytes": max(e["prepared_digest_bytes"] for e in updates),
        "peak_server_rss_bytes": max(r["server_maxrss_bytes"] for r in capacities),
        "minimum_guest_available_bytes": min(r["mem_available_bytes"] for r in capacities),
        "peak_fs_used_bytes": max(r["fs_used_bytes"] for r in capacities),
        "peak_sampled_store_bytes": max(
            (r["store_segment_used_bytes"] for r in capacities if "store_segment_used_bytes" in r),
            default=None,
        ),
    }
    hotspots = {}
    for op in ("R", "W"):
        by_region = Counter()
        for e in issues:
            if e.op != op:
                continue
            start = e.sector * 512
            stop = start + e.sectors * 512
            for region in range(start // MIB, (stop + MIB - 1) // MIB):
                by_region[region] += min(stop, (region + 1) * MIB) - max(start, region * MIB)
        check(
            sum(by_region.values()) == run["io_first_send_to_device_end"][op]["bytes"],
            "LBA region sum mismatch",
        )
        hotspots[op] = [
            {"start_lba": r * MIB // 512, "bytes": n} for r, n in by_region.most_common(5)
        ]
    run.update(
        top_1MiB_regions=hotspots,
        objects=ledger,
        capacity=cap,
        native_internal_checkpoints=checkpoints,
        peak_write_MiB_s=max(r["write_MiB_s"] for r in bins),
        peak_read_MiB_s=max(r["read_MiB_s"] for r in bins),
    )
    return run, bins, issues


def charts(out, runs, bins, issues):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(
        len(runs), 3, figsize=(14, 3 * len(runs)), squeeze=False, layout="constrained"
    )
    for i, (run, series, data) in enumerate(zip(runs, bins, issues, strict=True)):
        bw, qd, lba = axes[i]
        origin = run["requests"][0]["http_start_ns"]
        edges = [s["start_s"] for s in series] + [run["observed_s"]]
        for op, label, color in (("W", "Write", "#145b8a"), ("R", "Read", "#c06524")):
            bw.stairs(
                [s[label.lower() + "_MiB_s"] for s in series], edges, label=label, color=color
            )
            sub = [e for e in data if e.op == op]
            lba.scatter(
                [(e.time - origin) / NS for e in sub],
                [e.sector * 512 / MIB for e in sub],
                s=1,
                color=color,
            )
        qd.stairs([s["qd_peak"] for s in series], edges, color="#235950")
        for ax in (bw, qd, lba):
            for r in run["requests"]:
                ax.axvspan(
                    (r["http_start_ns"] - origin) / NS,
                    (r["response_end_ns"] - origin) / NS,
                    color="slateblue",
                    alpha=0.10,
                )
            ax.axvline((run["capture"]["sender_drained_ns"] - origin) / NS, color="grey", ls=":")
            ax.set_xlim(0, edges[-1])
        bw.set_ylabel(run["trial"].replace("femu-", ""))
    for ax, label in zip(
        axes[0],
        [
            "Issue bandwidth (MiB/s, 1 s bins)",
            "Peak data I/O in flight",
            "Start address (MiB from LBA 0)",
        ],
        strict=True,
    ):
        ax.set_title(label)
    for ax in axes[-1]:
        ax.set_xlabel("Guest seconds from first HTTP send")
    axes[0, 0].legend()
    fig.savefig(out / "traffic.png", dpi=155)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 3.5), layout="constrained")
    for run in runs:
        for op in ("W", "R"):
            hist = run["io_first_send_to_device_end"][op]["size_histogram"]
            if hist:
                ax.step(
                    [size / 1024 for size in hist],
                    list(accumulate(n / sum(hist.values()) for n in hist.values())),
                    where="post",
                    label=f"{run['trial']} {op}",
                )
    ax.set(
        xlabel="Issued I/O size (KiB, log scale)",
        ylabel="Cumulative fraction of I/Os",
        xscale="log",
        ylim=(0, 1.03),
    )
    ax.legend()
    fig.savefig(out / "sizes.png", dpi=155)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("calibration", type=Path)
    parser.add_argument("--repeats", type=Path, nargs=3)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    settings = matched_settings(args.calibration)
    paths = args.repeats or [args.calibration]
    check(
        len({p.resolve() for p in [args.calibration, *args.repeats]}) == 4
        if args.repeats
        else True,
        "Require independent boots",
    )
    runs = []
    bins = []
    issues = []
    for p in paths:
        check(matched_settings(p) == settings, "Frozen experiment settings differ")
        run, series, rows = online(p)
        runs.append(run)
        bins.append(series)
        issues.append(rows)
    signatures = {tuple(row["wire_prompt_sha256"] for row in r["requests"]) for r in runs}
    check(len(signatures) == 1, "Inputs on the wire differ")
    args.output.mkdir(parents=True, exist_ok=True)
    report = {"settings": settings, "runs": runs}
    if args.repeats:
        report["range"] = {}
        for name, values in {
            "issued_read_bytes": [r["io_first_send_to_device_end"]["R"]["bytes"] for r in runs],
            "issued_write_bytes": [r["io_first_send_to_device_end"]["W"]["bytes"] for r in runs],
            "peak_qd": [r["max_qd"] for r in runs],
            "peak_write_MiB_s": [r["peak_write_MiB_s"] for r in runs],
        }.items():
            report["range"][name] = {
                "min": min(values),
                "median": median(values),
                "max": max(values),
            }
    (args.output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    with (args.output / "timeline.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["trial", *bins[0][0]])
        writer.writeheader()
        for r, series in zip(runs, bins, strict=True):
            writer.writerows({"trial": r["trial"], **s} for s in series)
    charts(args.output, runs, bins, issues)


if __name__ == "__main__":
    main()
