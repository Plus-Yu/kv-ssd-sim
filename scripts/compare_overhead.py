#!/usr/bin/env python3
"""Compare replay timing without confusing moved preparation with removed work."""

import argparse
import csv
import json
from pathlib import Path

from kv_ssd_sim.events import load_events
from kv_ssd_sim.experiment import validate_trial


def inspect_run(path):
    summary = json.loads((path / "summary.json").read_text())
    manifest = json.loads((path / "run_manifest.json").read_text())
    events = load_events(path / "events.jsonl")
    starts = {}
    rows = {}
    objects = {}
    for event in events:
        if event["event"] == "compute_begin":
            starts[event["step"]] = event
        elif event["event"] == "compute_end":
            begin = starts[event["step"]]
            duration = (event["time_ns"] - begin["time_ns"]) / 1e6
            for rid in begin["request_ids"]:
                row = rows.setdefault(
                    rid, {"budget_ms": 0.0, "compute_window_ms": 0.0, "step_overrun_ms": []}
                )
                row["budget_ms"] += begin["planned_compute_ms"]
                row["compute_window_ms"] += duration
                row["step_overrun_ms"].append(duration - begin["planned_compute_ms"])
        elif event["event"] == "store_put_end":
            for key, code, digest in zip(
                event["keys"], event["result"], event["digests"], strict=True
            ):
                if code != 0:
                    raise AssertionError(f"Failed native PUT in {path}: {key}")
                if key in objects and objects[key] != digest:
                    raise AssertionError(f"Object changed within run: {key}")
                objects[key] = digest
    return summary, manifest, rows, objects


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("optimized", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    before, bm, br, bo = inspect_run(args.baseline)
    after, am, ar, ao = inspect_run(args.optimized)
    for field in (
        "input_sha256",
        "prefill_ms",
        "decode_ms",
        "read_delay_ms",
        "timing",
        "cold_between_turns",
        "max_model_len",
        "num_blocks",
        "versions",
    ):
        if bm[field] != am[field]:
            raise AssertionError(f"Comparison changed {field}")
    if not bo or bo != ao:
        raise AssertionError("Native object keys or payload SHA256 changed")
    validate_trial(args.baseline)
    validate_trial(args.optimized)
    if [r["request_id"] for r in before["request_metrics"]] != [
        r["request_id"] for r in after["request_metrics"]
    ]:
        raise AssertionError("Request identities/order changed")
    comparison = []
    for b, a in zip(before["request_metrics"], after["request_metrics"], strict=True):
        rid = a["request_id"]
        for field in (
            "input_tokens",
            "output_tokens",
            "external_load_tokens",
            "compute_steps",
            "compute_budget_ms",
        ):
            if b[field] != a[field]:
                raise AssertionError(f"Changed {field} for {rid}")
        budget = ar[rid]["budget_ms"]
        excess_before = br[rid]["compute_window_ms"] - budget
        excess_after = ar[rid]["compute_window_ms"] - budget
        overruns = sorted(ar[rid]["step_overrun_ms"])
        comparison.append(
            {
                "request_id": rid,
                "budget_ms": budget,
                "baseline_window_ms": br[rid]["compute_window_ms"],
                "optimized_window_ms": ar[rid]["compute_window_ms"],
                "baseline_excess_ms": excess_before,
                "optimized_excess_ms": excess_after,
                "excess_reduction_percent": 100 * (1 - excess_after / excess_before),
                "optimized_materialize_ms": a["materialize_ms"],
                "optimized_step_overrun_p99_ms": overruns[
                    min(len(overruns) - 1, int(len(overruns) * 0.99))
                ],
                "optimized_step_overrun_max_ms": overruns[-1],
            }
        )
    args.output.mkdir(parents=True, exist_ok=False)
    report = {
        "baseline": str(args.baseline.resolve()),
        "optimized": str(args.optimized.resolve()),
        "identical_store_objects": len(bo),
        "disk_restore_validated": True,
        "optimized_preparation": after["payload_preparation"],
        "requests": comparison,
        "scope": "paired functional measurements, not a confidence interval",
    }
    (args.output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    with (args.output / "comparison.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(comparison[0]))
        writer.writeheader()
        writer.writerows(comparison)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
