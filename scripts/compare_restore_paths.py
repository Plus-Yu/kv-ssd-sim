"""Compare legacy and host-staged recovery on the same compiled input and SSD."""

import argparse
import gzip
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

from compare_volume_replay import load, read_counts, read_identity

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from kv_ssd_sim.analysis.restore import audit as audit_restores


def traffic(folder):
    reads, writes, sizes = Counter(), Counter(), {"read": 0, "write": 0}
    with gzip.open(folder / "ssd-demand.jsonl.gz", "rt") as source:
        for row in map(json.loads, source):
            sizes[row["op"]] += row["bytes"]
            if row["op"] == "read":
                reads[read_identity(row)] += 1
            else:
                writes[row["key"], row["owner"], row["bytes"]] += 1
    full = load(folder / "ssd-summary.json")["scopes"]["full"]
    if any(sizes[op] != full[op + "_bytes"] for op in sizes):
        raise ValueError(f"SSD ledger and summary disagree in {folder}")
    return reads, writes, sizes


def timings(folder):
    end = defaultdict(dict)
    with (folder / "events.jsonl").open() as source:
        for row in map(json.loads, source):
            event = row["event"]
            if event in ("disk_read_complete", "get_complete", "load_complete"):
                end[row["request"]][event] = row["time_ns"]
            elif event == "h2d_complete" and row["tier"] == "disk":
                end[row["request"]]["disk_h2d_complete"] = row["time_ns"]
    return end


def compare(old, new):
    launches = [load(f / "launch.json") for f in (old, new)]
    if [m.get("restore_path", "legacy") for m in launches] != ["legacy", "host-staged"]:
        raise ValueError("Expected legacy first and host-staged second")

    def settings(m):
        return (
            [v["sha256"] for v in m["compiled_inputs"]],
            {
                k: m.get(k)
                for k in (
                    "profile",
                    "geometry_sha256",
                    "root_offsets_s",
                    "native_master",
                    "num_blocks",
                    "max_model_len",
                    "step_timing",
                    "prefill_ms",
                    "decode_ms",
                    "pythonhashseed",
                    "offload_timing_input",
                    "vllm_version",
                    "logical_volume",
                )
            },
            {k: v for k, v in m["store"].items() if k != "h2d_mib_s"},
        )

    if settings(launches[0]) != settings(launches[1]):
        raise ValueError("Inputs, topology or non-restore parameters differ")
    if audit_restores(new) != load(new / "restore-summary.json"):
        raise ValueError("Staged restore audit does not match")
    (old_reads, old_writes, baseline), (new_reads, new_writes, staged) = (
        traffic(f) for f in (old, new)
    )
    old_requests, new_requests = (
        {r["id"]: r for r in load(f / "requests.json")} for f in (old, new)
    )
    if old_requests.keys() != new_requests.keys():
        raise ValueError("Request IDs differ")
    old_times, new_times = timings(old), timings(new)
    return {
        "input_sha256": launches[0]["compiled_inputs"][0]["sha256"],
        "writes_same_key_owner_bytes_multiplicity": old_writes == new_writes,
        "disk_gets_only_legacy": read_counts(old_reads - new_reads),
        "disk_gets_only_staged": read_counts(new_reads - old_reads),
        "legacy_ssd_bytes": baseline,
        "staged_ssd_bytes": staged,
        "request_times": {
            rid: {
                "legacy_last_event_ns": old_times[rid],
                "staged_last_event_ns": new_times[rid],
                "staged_minus_legacy_ns": {
                    k: new_requests[rid][k] - old_requests[rid][k]
                    for k in ("arrival_ns", "admitted_ns", "finished_ns")
                },
            }
            for rid in old_requests
        },
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("legacy", type=Path)
    p.add_argument("staged", type=Path)
    args = p.parse_args()
    print(json.dumps(compare(args.legacy, args.staged), indent=2))
