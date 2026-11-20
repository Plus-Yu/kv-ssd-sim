"""Validate Stage A block measurements against the six archived cold-read runs."""

import argparse
import csv
import gzip
import hashlib
import io
import json
import math
import re
import tarfile
import tempfile
from bisect import bisect_right
from collections import Counter, defaultdict, deque
from contextlib import ExitStack
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path
from statistics import mean

NS = 10**9
MIB = 2**20
ROOT = Path(__file__).resolve().parents[1]
# The two tracepoints have different payloads. Preserve RWBS flags and completion status.
LINE = re.compile(
    r"^\s*.+?\s+\[\d+\]\s+\S+\s+(\d+)\.(\d{1,9}): block_rq_(issue|complete): "
    r"(\d+,\d+) (\S+) (?:(\d+) )?\([^)]*\) (\d+) \+ (\d+).* \[(.*)\]$"
)


def check(condition, message):
    if not condition:
        raise ValueError(message)


@dataclass
class Event:
    time: int
    stage: str
    device: str
    flags: str
    sector: int
    sectors: int
    issuer: str
    error: int | None
    ambiguous: bool = False
    latency_ns: int | None = None

    @property
    def op(self):
        # Linux prepends F for REQ_PREFLUSH, e.g. FWS is a data write.
        code = self.flags.lstrip("F")[:1]
        if code in ("R", "W", "D", "E"):
            return {"D": "discard", "E": "erase"}.get(code, code)
        return "flush" if self.flags.startswith("F") and not self.sectors else "other"


def parse_trace(raw, target, *, stream_entries=None):
    header = re.search(r"entries-in-buffer/entries-written:\s*(\d+)/(\d+)", raw)
    if stream_entries is None:
        check(header and header[1] == header[2], "Missing header or overwritten trace entries")
        expected_entries = int(header[1])
    else:
        check(header is None, "Expected a trace_pipe stream")
        expected_entries = stream_entries
    events = []
    pending = defaultdict(deque)
    parsed = 0
    for line in raw.splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        match = LINE.fullmatch(line)
        check(match, f"Unparsed trace record: {line[:160]}")
        sec, frac, stage, dev, flags, nbytes, sector, count, tail = match.groups()
        parsed += 1
        if dev != target:
            continue
        e = Event(
            int(sec) * NS + int(frac.ljust(9, "0")),
            stage,
            dev,
            flags,
            int(sector),
            int(count),
            tail if stage == "issue" else "",
            None if stage == "issue" else int(tail),
        )
        check(not events or e.time >= events[-1].time, "Nonmonotonic block clock")
        q = pending[e.op, e.sector, e.sectors]
        if stage == "issue":
            if e.op in ("R", "W"):
                check(nbytes is not None and int(nbytes) == e.sectors * 512, "Issue byte mismatch")
            if q:
                for outstanding in q:
                    outstanding.ambiguous = True
                e.ambiguous = True
            q.append(e)
        else:
            check(q, "Unmatched completion")
            source = q.popleft()
            e.ambiguous = source.ambiguous
            if not e.ambiguous:
                e.latency_ns = e.time - source.time
        events.append(e)
    check(parsed == expected_entries, "Kernel/parsed event count mismatch")
    check(not any(pending.values()), "Outstanding issue at trace end")
    check(events, "No target device events")
    return events, {
        "buffer_entries": parsed,
        "foreign_device_events": parsed - len(events),
        "ambiguous_completions": sum(e.ambiguous and e.stage == "complete" for e in events),
        "error_completions": sum(bool(e.error) for e in events),
    }


def measurements(events, begin, end, width):
    """Half-open issue-time bins; integrate area under all outstanding data I/O."""
    bins = []
    i = 0
    depth = 0
    data = [e for e in events if e.op in ("R", "W")]
    while i < len(data) and data[i].time < begin:
        depth += 1 if data[i].stage == "issue" else -1
        i += 1
    for lo in range(begin, end, width):
        hi = min(lo + width, end)
        counts = Counter()
        last = lo
        area = busy = 0
        peak = depth
        while i < len(data) and data[i].time < hi:
            e = data[i]
            area += (e.time - last) * depth
            busy += (e.time - last) * bool(depth)
            if e.stage == "issue":
                counts[e.op] += 1
                counts[e.op + "B"] += e.sectors * 512
                depth += 1
            else:
                depth -= 1
            peak = max(peak, depth)
            last = e.time
            i += 1
        area += (hi - last) * depth
        busy += (hi - last) * bool(depth)
        duration = (hi - lo) / NS
        bins.append(
            {
                "start_s": (lo - begin) / NS,
                "duration_s": duration,
                "read_ios": counts["R"],
                "write_ios": counts["W"],
                "read_bytes": counts["RB"],
                "write_bytes": counts["WB"],
                "read_iops": counts["R"] / duration,
                "write_iops": counts["W"] / duration,
                "read_MiB_s": counts["RB"] / MIB / duration,
                "write_MiB_s": counts["WB"] / MIB / duration,
                "qd_mean": area / (hi - lo),
                "qd_peak": peak,
                "busy_ns": busy,
            }
        )
    return bins


def stats(events, begin, end):
    output = {}
    for op in ("R", "W"):
        selected = [e for e in events if begin <= e.time < end and e.op == op]
        issues = [e for e in selected if e.stage == "issue"]
        latencies = sorted(e.latency_ns / 1e6 for e in selected if e.latency_ns is not None)
        transitions = len(issues) - 1
        sizes = Counter(e.sectors * 512 for e in issues)
        # Split requests straddling 1 MiB boundaries before choosing the hottest region.
        regions = Counter()
        for e in issues:
            first, end_byte = e.sector * 512, (e.sector + e.sectors) * 512
            for region in range(first // MIB, (end_byte + MIB - 1) // MIB):
                regions[region] += min(end_byte, (region + 1) * MIB) - max(first, region * MIB)
        total = sum(regions.values())
        check(total == sum(b * n for b, n in sizes.items()), "LBA byte accounting mismatch")
        output[op] = {
            "count": len(issues),
            "bytes": total,
            "size_histogram": dict(sorted(sizes.items())),
            "adjacent_transitions": sum(
                a.sector + a.sectors == b.sector for a, b in pairwise(issues)
            ),
            "transitions": max(0, transitions),
            "hottest_1MiB_byte_share": max(regions.values()) / total if total else None,
            "latency_samples": len(latencies),
            "mean_ms": mean(latencies) if latencies else None,
            "p50_ms": latencies[math.ceil(0.50 * len(latencies)) - 1] if latencies else None,
            "p95_ms": latencies[math.ceil(0.95 * len(latencies)) - 1] if latencies else None,
        }
    for values in output.values():
        for field in ("count", "bytes"):
            denominator = sum(v[field] for v in output.values())
            values[field + "_share"] = values[field] / denominator if denominator else None
        n = values["transitions"]
        values["adjacent_share"] = values["adjacent_transitions"] / n if n else None
    return output


def analyze(archive, reference, writers):
    trial = reference["trial"]

    def read(name):
        return archive.extractfile(f"{trial}/{name}").read().decode()

    manifest = json.loads(read("joint/server/run_manifest.json"))
    device = manifest["storage_device"]
    check(
        device["device"] == "nvme0n1"
        and "FEMU" in device["model"]
        and "/dev/nvme0n1 /mnt/femu " in read("guest-mounts.txt"),
        "Unexpected device",
    )
    proof = json.loads(read("verification.json"))["chains"][0]
    target = proof["first_read_completion"]["device"]
    events, audit = parse_trace(read("block-trace.txt"), target)
    check(not audit["error_completions"], "Block completion error")
    server = [json.loads(l) for l in read("joint/server/events.jsonl").splitlines()]

    def stamps(name):
        return [origin + e["time_ns"] for e in server if e["event"] == name]

    (origin,) = [e["monotonic_origin_ns"] for e in server if e["event"] == "clock_sync"]
    send1, send2 = [
        json.loads(l)["before_native_send_ns"] for l in read("joint/client-send.jsonl").splitlines()
    ]
    response1, response2 = stamps("http_response_finished")
    (cold_begin,) = stamps("cold_prepare_begin")
    (cold_end,) = stamps("cold_prepare_end")
    (get_start,) = stamps("native_get_begin")
    (get_end,) = stamps("store_get_end")
    confirmed = stamps("disk_replicas_confirmed")[-1]
    markers = {
        "turn1": send1,
        "between_turns": response1,
        "cold_prepare": cold_begin,
        "ready_to_send": cold_end,
        "turn2_before_get": send2,
        "get": get_start,
        "turn2_after_get": get_end,
        "after_response": response2,
        "after_confirmation": confirmed,
    }
    boundaries = list(markers.values())
    check(
        events[0].time <= send1
        and boundaries == sorted(boundaries)
        and confirmed <= events[-1].time,
        "Window outside trace or clock/order mismatch",
    )
    all_stats = stats(events, events[0].time, events[-1].time + 1)
    for op, measured in all_stats.items():
        expected = reference["device"][op]
        for field in ("count", "bytes"):
            check(measured[field] == expected[field], f"{trial} {op} {field} mismatch")
        if not audit["ambiguous_completions"]:
            check(math.isclose(measured["mean_ms"], expected["mean_ms"]), "Latency mismatch")
    (sample,) = [e for e in server if e["event"] == "femu_block_io_during_get"]
    subset = [e for e in events if get_start <= e.time <= get_end and e.stage == "complete"]
    check(sample["device"] == device["device"], "Wrong GET counter device")
    for op, direction in (("R", "read"), ("W", "write")):
        during = [e for e in subset if e.op == op]
        n, s = len(during), sum(e.sectors for e in during)
        check(
            (n, s) == (sample["delta"][direction + "s"], sample["delta"][direction + "_sectors"]),
            "Independent GET counter mismatch",
        )
        if op == "R":
            check(
                (n, s * 512)
                == (
                    reference["restore"]["femu_read_completions_during_get"],
                    reference["restore"]["femu_read_bytes_during_get"],
                ),
                "GET mismatch",
            )
    names = ["before_send", *markers]
    for e in events:
        phase = names[bisect_right(boundaries, e.time)]
        writers["events"].writerow(
            [
                trial,
                e.time,
                (e.time - send1) / NS,
                phase,
                e.stage,
                e.device,
                e.flags,
                e.op,
                e.sector,
                e.sectors,
                e.sectors * 512,
                e.issuer,
                e.error,
                int(e.ambiguous),
                e.latency_ns,
            ]
        )
    session = stats(events, send1, confirmed)
    for width in (NS, NS // 10):
        bins = measurements(events, send1, confirmed, width)
        for op, direction in (("R", "read"), ("W", "write")):
            check(
                sum(b[direction + "_bytes"] for b in bins) == session[op]["bytes"]
                and sum(b[direction + "_ios"] for b in bins) == session[op]["count"],
                "Bin totals differ",
            )
        for b in bins:
            writers["bins"].writerow([trial, width // 10**6, *b.values()])
    for scope, values in (("full", all_stats), ("session", session)):
        for op, v in values.items():
            for size, n in v["size_histogram"].items():
                writers["sizes"].writerow([trial, scope, op, size, n, size * n])
    operations = Counter(e.op for e in events if e.stage == "issue")
    return {
        "trial": trial,
        "target": target,
        "nand_read_ns": reference["nand_read_ns"],
        "integrity": audit,
        "issued_operations": dict(operations),
        "full": all_stats,
        "session": session,
        "session_start_ns": send1,
        "markers_relative_s": {k: (v - send1) / NS for k, v in markers.items()},
        "session_busy_s": sum(b["busy_ns"] for b in bins) / NS,
        "session_idle_fraction": 1 - sum(b["busy_ns"] for b in bins) / (confirmed - send1),
        "session_qd_mean": sum(b["qd_mean"] * b["duration_s"] for b in bins)
        / ((confirmed - send1) / NS),
        "session_qd_peak": max(b["qd_peak"] for b in bins),
        "independent_get_diskstats": sample["delta"],
    }


HEADERS = {
    "events": "trial guest_ns relative_s phase stage device rwbs op sector sectors bytes issuer completion_error ambiguous latency_ns",
    "bins": "trial width_ms start_s duration_s read_ios write_ios read_bytes write_bytes read_iops write_iops read_MiB_s write_MiB_s qd_mean qd_peak busy_ns",
    "sizes": "trial scope op bytes_per_io count bytes_total",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "evidence/kimi-k3-step6")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    check(not args.output.exists(), "Choose a new output directory")
    digests = {}
    for line in (args.input / "SHA256SUMS").read_text().splitlines():
        checksum, name = line.split()
        if name in ("trials.tar.gz", "summary.json"):
            check(
                hashlib.sha256((args.input / name).read_bytes()).hexdigest() == checksum,
                name + " SHA256 mismatch",
            )
            digests[name] = checksum
    check(len(digests) == 2, "Missing checksum")
    ref = json.loads((args.input / "summary.json").read_text())
    check(
        ref["status"] == "passed"
        and len({r["trial"] for r in ref["trials"]}) == len(ref["trials"]) == 6,
        "Expected six accepted trials",
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(dir=args.output.parent) as temp:
        dest = Path(temp) / "result"
        dest.mkdir()
        with ExitStack() as stack:
            writers = {}
            for name, header in HEADERS.items():
                fp = stack.enter_context((dest / (name + ".csv.gz")).open("wb"))
                gz = stack.enter_context(gzip.GzipFile(filename="", fileobj=fp, mode="wb", mtime=0))
                writer = csv.writer(
                    stack.enter_context(io.TextIOWrapper(gz, encoding="utf-8", newline=""))
                )
                writer.writerow(header.split())
                writers[name] = writer
            archive = stack.enter_context(tarfile.open(args.input / "trials.tar.gz", "r:gz"))
            rows = [analyze(archive, r, writers) for r in ref["trials"]]
        (dest / "summary.json").write_text(
            json.dumps({"status": "passed", "input_sha256": digests, "trials": rows}, indent=2)
            + "\n"
        )
        dest.rename(args.output)


if __name__ == "__main__":
    main()
