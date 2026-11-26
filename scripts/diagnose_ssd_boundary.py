"""Diagnose a rejected trial by comparing independent completions and in-flight counts."""

import argparse
import json
import sys
from bisect import bisect_left, bisect_right
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from analyze_b2 import read
from analyze_k3_ssd import ROOT, transfer_ledger

from kv_ssd_sim.analysis.aiperf import read_jsonl, verify
from kv_ssd_sim.analysis.block_capture import check_receipt
from kv_ssd_sim.analysis.block_io import NS, check, measurements, parse_trace, stats


def diagnose(path):
    check((path / "guest-exit-code.txt").read_text().strip() == "0", "Workload failed")
    http = verify(path / "joint", staged_trace=path / "input/trace.json")
    receipt = read(path / "capture.json")
    entries = check_receipt(receipt, receipt["device"], "stream")
    b, integrity = parse_trace(
        (path / "block-trace.txt").read_text(), receipt["device"], stream_entries=entries
    )
    t = [e.time for e in b]
    marks = read_jsonl(path / "diskstats.jsonl")
    state = [marks[0]["stat"][k] for k in (0, 2, 4, 6, 8)]
    i = bisect_left(t, marks[0]["before_ns"] - 1000)
    diagnostics = []

    def advance(stop):
        nonlocal i
        while i < stop:
            e = b[i]
            state[4] += 1 if e.stage == "issue" else -1
            if e.stage == "complete" and e.op in ("R", "W"):
                k = 0 if e.op == "R" else 2
                state[k] += 1
                state[k + 1] += e.sectors
            i += 1

    for m in marks:
        lo, hi = bisect_left(t, m["before_ns"] - 1000), bisect_right(t, m["after_ns"] + 1000)
        measured = [m["stat"][k] for k in (0, 2, 4, 6, 8)]
        candidates = []
        for cut in range(lo, hi + 1):
            advance(cut)
            if state[:4] == measured[:4]:
                candidates.append(state[4])
        diagnostics.append(
            {
                "label": m["label"],
                "completions_and_sectors_match": bool(candidates),
                "diskstats_inflight": measured[4],
                "possible_trace_depths": sorted(set(candidates)),
            }
        )
    events = read_jsonl(path / "joint/server/events.jsonl")
    ledger = transfer_ledger(events, read(ROOT / "configs/kimi-k3-b300-step2/resolved.json"))
    begin = min(
        e["start_perf_ns"] for e in read_jsonl(path / "joint/client/profile_export_raw.jsonl")
    )
    end = marks[-1]["after_ns"]
    return {
        "status": "diagnostic_only",
        "trial": path.name,
        "boundaries": diagnostics,
        "http": http,
        "block_trace_integrity": integrity,
        "objects": ledger,
        "io_first_send_to_device_end": stats(b, begin, end),
        "max_qd": max(row["qd_peak"] for row in measurements(b, begin, end, NS)),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trial", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.write_text(json.dumps(diagnose(args.trial), indent=2) + "\n")
