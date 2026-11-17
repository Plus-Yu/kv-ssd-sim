#!/usr/bin/env python3
"""Validate guest block completions -> Store GET -> native scheduler resumption."""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict, deque
from pathlib import Path


def run_path(path):
    launch = json.loads((path / "launch.json").read_text())
    return path / ("joint/server" if launch.get("workload") == "aiperf" else "replay")


def verify(path):
    launch = json.loads((path / "launch.json").read_text())
    run = run_path(path)
    joint = launch.get("workload") == "aiperf"
    if joint:
        from verify_aiperf import verify as verify_http

        http_evidence = verify_http(path / "joint")
        if not http_evidence["femu_verified"]:
            raise AssertionError("HTTP workload did not verify FEMU")
    summary = json.loads((run / "summary.json").read_text())
    manifest = json.loads((run / "run_manifest.json").read_text())
    events = [json.loads(line) for line in (run / "events.jsonl").read_text().splitlines()]
    if not summary["femu_verified"] or manifest["read_delay_ms"] != 0:
        raise AssertionError("Trial did not verify FEMU without injected receiver delay")
    origin = next(e["monotonic_origin_ns"] for e in events if e["event"] == "clock_sync")
    completions = []
    pending = defaultdict(deque)
    trace = (path / "block-trace.txt").read_text()
    counts = re.search(r"entries-in-buffer/entries-written:\s*(\d+)/(\d+)", trace)
    if counts is None or counts[1] != counts[2]:
        raise AssertionError("Block trace is incomplete or overflowed")
    devices = set()
    for line in trace.splitlines():
        match = re.search(r"(\d+\.\d+): block_rq_(issue|complete):\s+(\d+,\d+)\s+(\S+)", line)
        sectors = re.search(r"(\d+) \+ (\d+)", line)
        if match:
            devices.add(match[3])
        if match and sectors and "R" in match[4]:
            key = (match[3], int(sectors[1]), int(sectors[2]))
            timestamp = round(float(match[1]) * 1e9) - origin
            if match[2] == "issue":
                pending[key].append(timestamp)
                continue
            issued = pending[key].popleft() if pending[key] else None
            completions.append(
                {
                    "time_ns": timestamp,
                    "device": match[3],
                    "sector": key[1],
                    "sectors": key[2],
                    "block_latency_ms": (timestamp - issued) / 1e6 if issued is not None else None,
                    "line": line.strip(),
                }
            )
    # The launcher exposes one RAM-backed FEMU and no other block devices.
    if len(devices) != 1:
        raise AssertionError("Expected exactly one traced block device")
    restored = [r for r in summary["request_metrics"] if r["external_load_tokens"] > 0]
    if not restored or summary["checksum_errors"]:
        raise AssertionError("No checksum-verified external prefix restore")
    chains = []
    for row in restored:
        rid = row["request_id"]
        own = [e for e in events if e.get("request_id") == rid]
        begin = next(e for e in own if e["event"] == "native_get_begin")
        end = next(e for e in own if e["event"] == "store_get_end")
        load = next(e for e in own if e["event"] == "load_complete")
        compute = next(
            e for e in events if e["event"] == "compute_begin" and rid in e["request_ids"]
        )
        token = next(e for e in own if e["event"] == "first_token")
        received = next((e for e in own if e["event"] == "http_received"), None)
        admitted = next((e for e in own if e["event"] == "request_admitted"), None)
        emitted = next((e for e in own if e["event"] == "http_token_emitted"), None)
        http_row = None
        if joint:
            if not received or not emitted or token["seq"] >= emitted["seq"]:
                raise AssertionError("Missing scheduler -> HTTP first token ordering")
            if not admitted or not received["seq"] < admitted["seq"] < begin["seq"]:
                raise AssertionError("Missing HTTP -> admission -> GET ordering")
            http_row = next(
                json.loads(line)
                for line in (path / "joint/client/profile_export.jsonl").read_text().splitlines()
                if json.loads(line)["metadata"]["x_request_id"] == received["client_request_id"]
            )
        reads = [c for c in completions if begin["time_ns"] <= c["time_ns"] <= end["time_ns"]]
        if not reads or not (
            begin["seq"] < end["seq"] < load["seq"] < compute["seq"] < token["seq"]
        ):
            raise AssertionError(f"Missing ordered causal chain for {rid}")
        if not all(end["checksum_ok"]):
            raise AssertionError("Restore checksum mismatch")
        latencies = [read["block_latency_ms"] for read in reads]
        if any(value is None or value < 0 for value in latencies):
            raise AssertionError("Read completion has no matching issue event")
        chains.append(
            {
                "request_id": rid,
                "external_tokens": row["external_load_tokens"],
                "get_ms": (end["time_ns"] - begin["time_ns"]) / 1e6,
                "ttft_ms": row["ttft_ms"],
                "compute_budget_ms": row["compute_budget_ms"],
                "compute_window_ms": row["compute_window_ms"],
                "aiperf_ttft_ms": http_row["metrics"]["time_to_first_token"]["value"]
                if http_row
                else None,
                "http_received_to_first_emit_ms": (
                    (emitted["time_ns"] - received["time_ns"]) / 1e6 if joint else None
                ),
                "http_first_emit_seq": emitted["seq"] if emitted else None,
                "prepared_to_admission_ms": (
                    (admitted["time_ns"] - received["time_ns"]) / 1e6 if joint else None
                ),
                "admission_to_http_first_ms": (
                    (emitted["time_ns"] - admitted["time_ns"]) / 1e6 if joint else None
                ),
                "payload_preparation_ms": received.get("preparation_ms") if received else None,
                "source_turn": (
                    [
                        http_row["metadata"]["source_outer_idx"],
                        http_row["metadata"].get("source_inner_idx"),
                    ]
                    if http_row
                    else None
                ),
                "femu_read_completions_during_get": len(reads),
                "femu_read_bytes_during_get": sum(read["sectors"] * 512 for read in reads),
                "block_read_latency_ms": {
                    "min": min(latencies),
                    "mean": sum(latencies) / len(latencies),
                    "max": max(latencies),
                },
                "first_read_completion": reads[0],
                "last_read_completion": reads[-1],
                "ordered_event_sequences": [
                    begin["seq"],
                    end["seq"],
                    load["seq"],
                    compute["seq"],
                    token["seq"],
                ],
            }
        )
    result = {
        "status": "passed",
        "accelerator": launch["accelerator"],
        "nand_read_ns": launch["nand_read_ns"],
        "chains": chains,
        "aiperf": http_evidence if joint else None,
        "scope": "functional closed loop; TCG is not hardware-calibrated inference timing",
    }
    (path / "verification.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trials", type=Path, nargs="+")
    args = parser.parse_args()
    print(json.dumps([verify(path) for path in args.trials], indent=2))
