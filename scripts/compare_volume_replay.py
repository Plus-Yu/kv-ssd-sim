"""Audit a single-volume replay ledger and compare the four warm-window scenarios."""

import argparse
import gzip
import hashlib
import json
import math
from collections import Counter
from itertools import groupby
from pathlib import Path

from check_public_replay import digest, verify

ROOT = Path(__file__).resolve().parents[1]
CASES = ("single-3tb", "volume-1x", "volume-4x", "volume-8x")


def load(path):
    return json.loads(path.read_bytes())


def read_identity(row):
    return row["request"], row["rank"], row["key"], row["bytes"]


def read_counts(reads):
    return {
        "occurrences": reads.total(),
        "unique_keys": len({key for _, _, key, _ in reads}),
        "bytes": sum(k[-1] * n for k, n in reads.items()),
    }


def difference_from_4x(observed, reference):
    return {
        "only_here": read_counts(observed - reference),
        "only_4x": read_counts(reference - observed),
    }


def audit_run(folder):
    """Check one FIFO at bucket granularity; count object payloads at their submit time."""
    checked = verify(folder)
    launch = load(folder / "launch.json")
    store = launch["store"]
    summary = load(folder / "summary.json")
    traffic = load(folder / "ssd-summary.json")
    requests = load(folder / "requests.json")
    anchors = traffic["anchors"]
    if len(anchors) > 1 or launch["offload_timing_input"]:
        raise ValueError(f"Expected at most one measurement anchor and modeled service in {folder}")
    anchor = anchors[0]["time_ns"] if anchors else traffic["scopes"]["full"]["start_ns"]
    last_finish = traffic["last_request_finished_ns"]
    plan = load(ROOT / "configs/kimi-k3-b300-step2/resolved.json")
    kinds = {"MLAAttentionSpec": "MLA", "MambaSpec": "KDA"}
    groups = {g["group"]: kinds[g["spec"]] for g in plan["groups"]}
    rates = {"write": store["device_mib_s"] * 2**20, "read": store["device_read_mib_s"] * 2**20}
    reads = {"full": Counter(), "measurement": Counter()}
    scopes = {}
    for scope in reads:
        scopes[scope] = {
            "by_client": [Counter(read=0, write=0) for _ in range(store["clients"])],
            "by_type": {k: Counter(read=0, write=0) for k in kinds.values()},
        }
    published = {}
    written = Counter()
    changes = Counter()
    outstanding_at_finish = Counter(read=0, write=0)
    after_finish = Counter(read=0, write=0)
    io_counts = Counter(read=0, write=0)
    last_end = 0
    with gzip.open(folder / "ssd-demand.jsonl.gz", "rt") as stream:
        for _, items in groupby(map(json.loads, stream), key=lambda r: r["io_id"]):
            rows = list(items)
            first = rows[0]
            op, owner = first["op"], first["owner"]
            submit, start, end = (first[k] for k in ("submit_ns", "start_ns", "complete_ns"))
            size = sum(r["bytes"] for r in rows)
            if start != max(submit, last_end) or end - start != math.ceil(size * 1e9 / rates[op]):
                raise ValueError(f"Logical-volume service time or queue order differs: {first}")
            last_end = end
            changes[submit] += size
            changes[end] -= size
            io_counts[op] += 1
            if submit <= last_finish < end:
                outstanding_at_finish[op] += size
            if submit > last_finish:
                after_finish[op] += size
            for row in rows:
                for field in ("op", "owner", "bucket_id", "submit_ns", "start_ns", "complete_ns"):
                    if row[field] != first[field]:
                        raise ValueError(f"Objects disagree on their I/O: {row}")
                key = row["key"]
                replica = (owner, row["bucket_id"])
                if op == "write":
                    published[key] = (*replica, end)
                    written[key, owner, row["bytes"]] += 1
                elif (
                    key not in published
                    or published[key][:2] != replica
                    or published[key][2] > submit
                ):
                    raise ValueError(f"Read from a bucket before publication: {row}")
                for scope in ("full", "measurement") if submit >= anchor else ("full",):
                    scopes[scope]["by_client"][owner][op] += row["bytes"]
                    scopes[scope]["by_type"][groups[row["group"]]][op] += row["bytes"]
                    if op == "read":
                        reads[scope][read_identity(row)] += 1
    if reads["full"] != Counter(map(read_identity, checked["disk_read_events"])):
        raise ValueError("File reads and rank-level disk GET identities differ")
    if sum(after_finish.values()) != traffic["submitted_after_last_finish_bytes"] or (
        sum(outstanding_at_finish.values()) + sum(after_finish.values())
        != traffic["completed_after_last_finish_bytes"]
    ):
        raise ValueError("Foreground cut does not reconcile with completions")
    peak = outstanding = 0
    for t in sorted(changes):
        outstanding += changes[t]
        peak = max(outstanding, peak)
    if outstanding:
        raise ValueError("Unfinished bytes at final boundary")
    for name, values in scopes.items():
        data = traffic["scopes"].get(name, traffic["scopes"]["full"])
        for op in rates:
            total = data[op + "_bytes"]
            if total != sum(v[op] for v in values["by_client"]) or total != sum(
                v[op] for v in values["by_type"].values()
            ):
                raise ValueError(f"Client/type accounting differs for {op} in {name}")
        values.update(data, disk_reads=read_counts(reads[name]))
    result = {
        "profile": launch["profile"],
        "client_quota_bytes": store["quota_bytes"],
        "total_quota_bytes": store["clients"] * store["quota_bytes"],
        "effective_bytes_s": rates,
        "scopes": scopes,
        "unique_objects": summary["unique_keys"],
        "write_identity_sha256": hashlib.sha256(
            json.dumps(sorted((*key, count) for key, count in written.items())).encode()
        ).hexdigest(),
        "memory_gets": checked["memory_gets"],
        "evicted_memory_objects": summary["counts"].get("evict_memory", 0),
        "evicted_disk_objects": summary["counts"].get("evict_disk", 0),
        "peak_disk_bytes_by_client": [c["peak_disk_bytes"] for c in summary["clients"]],
        "anchor": anchors[0] if anchors else None,
        "requests_admitted_before_anchor": sum(r["admitted_ns"] < anchor for r in requests),
        "requests_finished_before_anchor": sum(r["finished_ns"] < anchor for r in requests),
        "last_request_finished_ns": last_finish,
        "last_io_complete_ns": last_end,
        "drain_after_last_finish_ns": max(0, last_end - last_finish),
        "submitted_after_last_finish_bytes": after_finish,
        "outstanding_at_last_finish_bytes": outstanding_at_finish,
        "peak_outstanding_bytes": peak,
        "volume_ios": io_counts,
    }
    return result, reads


def compare(folder, window):
    results, all_reads = {}, {}
    common = None
    source = {
        r["id"]: (r["input_tokens"], r["output_tokens"], r["prompt_sha256"])
        for r in window["source_requests"]
    }
    for name in CASES:
        run = folder / name
        launch = load(run / "launch.json")
        store = launch["store"]
        settings = {
            k: launch[k]
            for k in (
                "num_blocks",
                "max_model_len",
                "step_timing",
                "geometry_sha256",
                "root_offsets_s",
                "pythonhashseed",
                "native_master",
                "vllm_version",
                "source_sha256",
            )
        }
        settings["store"] = {
            k: v
            for k, v in store.items()
            if k not in {"quota_bytes", "device_mib_s", "device_read_mib_s"}
        }
        settings["scheduler"] = launch.get("scheduler")
        settings["compute_profile_sha256"] = launch.get("compute_calibration", {}).get("sha256")
        if common is not None and common != settings:
            raise ValueError(f"Non-storage settings differ in {name}")
        common = settings
        inputs = [inp["sha256"] for inp in launch["compiled_inputs"]]
        actual = {
            r["id"]: (r["prompt_tokens"], r["output_tokens"], r["prompt_sha256"])
            for r in load(run / "requests.json")
        }
        if (
            inputs != [window["input_sha256"]]
            or actual != source
            or launch["max_model_len"] != window["context_limit"]
        ):
            raise ValueError(f"Compiled trace or context differs in {name}")
        control = name == "single-3tb"
        factor = 1 if control else int(name.removeprefix("volume-").removesuffix("x"))
        if (
            store["clients"] != 8
            or store["segment_bytes"] != 128_000_000_000
            or store["quota_bytes"] != (375_000_000_000 if control else 3_000_000_000_000)
            or store["device_mib_s"] * 2**20 != factor * 5e9
            or store["device_read_mib_s"] * 2**20 != factor * 6.8e9
        ):
            raise ValueError(f"Wrong client budgets or throughput in {name}")
        if control:
            if launch["profile"] != "reference-single-ssd":
                raise ValueError("Expected the frozen single-drive control")
        else:
            v = launch["logical_volume"]
            members = launch["reference_settings"]["baseline"]["volume"]["member_count"]
            if (
                v["members"],
                v["nominal_bytes"],
                v["kv_quota_bytes"],
                v["fifo_queues"],
                v["service_rate_multiplier"],
            ) != (members, 30_720_000_000_000, 24_000_000_000_000, 1, factor):
                raise ValueError(f"Wrong logical volume in {name}")
        results[name], all_reads[name] = audit_run(run)
        if results[name]["anchor"]["request"] != window["anchor"]:
            raise ValueError(f"Measurement started at a different request in {name}")
    for name, scopes in all_reads.items():
        results[name]["read_identity_difference_vs_4x"] = {
            scope: difference_from_4x(counts, all_reads["volume-4x"][scope])
            for scope, counts in scopes.items()
        }
    return {
        "source_sha256": window["source_sha256"],
        "input_sha256": window["input_sha256"],
        "requests": len(source),
        "sessions": len(window["sessions"]),
        "warmup_source_requests": window["warmup_source_requests"],
        "cases": results,
    }


def compare_pressure(folder):
    results, reads, evicted = {}, {}, {}
    common = None
    for factor in ("1x", "4x", "8x"):
        run = folder / f"probe-{factor}"
        launch = load(run / "launch.json")
        settings = dict(launch["store"])
        for field in ("device_mib_s", "device_read_mib_s"):
            del settings[field]
        signature = (
            [item["sha256"] for item in launch["compiled_inputs"]],
            launch["num_blocks"],
            launch["max_model_len"],
            launch["step_timing"],
            launch["geometry_sha256"],
            launch["pythonhashseed"],
            launch["native_master"],
            launch["root_offsets_s"],
            settings,
        )
        if (common is not None and common != signature) or (
            launch["logical_volume"]["service_scenario"] != factor or settings["clients"] != 1
        ):
            raise ValueError(f"Pressure runs must share their input and single-client setup: {run}")
        common = signature
        results[factor], buckets = audit_run(run)
        reads[factor] = buckets["full"]
        with (run / "events.jsonl").open() as stream:
            events = list(map(json.loads, stream))
        evictions = [e for e in events if e["event"] == "evict_memory"]
        evicted[factor] = {e["key"] for e in evictions}
        results[factor].update(
            first_publish_ns=next(e["time_ns"] for e in events if e["event"] == "disk_publish"),
            eviction_times_ns=sorted({e["time_ns"] for e in evictions}),
            disk_get_keys=sorted({key for _, _, key, _ in reads[factor]}),
        )
    for factor, result in results.items():
        result.update(
            read_identity_difference_vs_4x=difference_from_4x(reads[factor], reads["4x"]),
            evicted_keys_only_here_vs_4x=sorted(evicted[factor] - evicted["4x"]),
            evicted_keys_only_4x=sorted(evicted["4x"] - evicted[factor]),
        )
    identical = {}
    for name in ("events.jsonl", "ssd-summary.json", "requests.json"):
        identical[name] = digest(folder / "probe-8x" / name)
        if identical[name] != digest(folder / "probe-8x-repeat" / name):
            raise ValueError(f"8x pressure repeat differs in {name}")
    return {
        "scope": "one-client 35-second return mechanism test",
        "cases": results,
        "repeat_8x_identical_sha256": identical,
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("matrix", type=Path, help="Directory containing the completed runs")
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--window", type=Path, help="Compiled warm window.json")
    mode.add_argument("--pressure", action="store_true", help="Audit the 35-second pressure probes")
    args = p.parse_args()
    result = (
        compare_pressure(args.matrix) if args.pressure else compare(args.matrix, load(args.window))
    )
    print(json.dumps(result, indent=2))
