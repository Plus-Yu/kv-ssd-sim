"""Audit a staged natural-eviction FEMU replay using the native event evidence."""

import json
import re
from collections import Counter
from itertools import combinations

from kv_ssd_sim.trace import load_weka


def require(ok, message):
    if not ok:
        raise ValueError(message)


def check_pressure(path, events, summary, manifest, chains):
    cfg = manifest["pressure_config"]
    source, restore = cfg["producer"], cfg["restore"]
    incomplete, miss = cfg["incomplete_producer"], cfg["recompute"]
    require(
        not manifest["cold_between_turns"]
        and manifest["timing"] == "causal"
        and manifest["offload_use_uring"]
        and manifest["read_delay_ms"] == 0,
        "Wrong pressure mode",
    )
    forbidden = {
        "local_prefix_cache_reset",
        "guest_page_cache_dropped",
        "memory_replicas_evicted",
        "cold_prepare_begin",
        "load_failure",
    }
    require(not any(e["event"] in forbidden for e in events), "Artificial eviction or load failure")
    require(
        "[UringFile] O_DIRECT mode enabled for /mnt/femu/replay/"
        in (path / "replay.log").read_text(),
        "Missing native direct-read file-open evidence",
    )
    replay = path / "replay"
    command = json.loads((replay / "mooncake-master.json").read_text())["command"]
    store = json.loads((replay / "mooncake.json").read_text())
    require(
        store["global_segment_size"] == cfg["segment_bytes"]
        and store["local_buffer_size"] == manifest["offload_staging_bytes"] == cfg["staging_bytes"]
        and manifest["disk_limit_bytes"] == cfg["disk_bytes"],
        "Store capacity mismatch",
    )
    require(
        f"--default_kv_lease_ttl={cfg.get('lease_ttl_ms', 0)}" in command, "Read lease mismatch"
    )
    for flag, name in (
        ("eviction_ratio", "eviction_ratio"),
        ("eviction_high_watermark_ratio", "high_watermark"),
    ):
        require(f"--{flag}={cfg[name]}" in command, "Master eviction settings mismatch")
    # Pair each native eviction result with its triggering watermark sample.
    triggers = 0
    peak = 0.0
    active = False
    for line in (replay / "mooncake-master.log").read_text().splitlines():
        if "[EVICT-TRIGGER]" in line:
            ratio = re.search(r"memory_ratio=([\d.eE+-]+) high_watermark=([\d.eE+-]+)", line)
            active = bool(ratio and float(ratio[1]) > float(ratio[2]) == cfg["high_watermark"])
            if ratio:
                peak = max(peak, float(ratio[1]))
        if "[EVICT-RESULT]" in line:
            count = int(re.search(r"evicted_count=(\d+)", line)[1])
            triggers += int(active and count > 0)
            active = False
    require(triggers > 0, "No successful native memory watermark eviction")

    def rows(kind, rid=None):
        return [
            e for e in events if e["event"] == kind and (rid is None or e.get("request_id") == rid)
        ]

    def one(kind, rid=None):
        matches = rows(kind, rid)
        require(len(matches) == 1, f"Expected one {kind} for {rid}")
        return matches[0]

    specs = {r.request_id: r for r in load_weka(path / "input/trace.json")}
    metrics = {r["request_id"]: r for r in summary["request_metrics"]}
    require(
        set(specs) == set(metrics)
        and len(specs)
        == summary["requests"]
        == len(rows("request_admitted"))
        == len(rows("request_finished")),
        "Trace request coverage differs",
    )
    for rid, req in specs.items():
        admission = one("request_admitted", rid)
        require(
            admission["dependencies"] == list(req.dependencies)
            and metrics[rid]["output_tokens"] == req.output_tokens
            and admission["input_tokens"] == len(req.prompt_token_ids),
            "Trace semantics differ",
        )
        for dep in req.dependencies:
            require(
                one("request_finished", dep)["seq"] < admission["seq"]
                and metrics[rid]["admitted_s"] >= metrics[dep]["finished_s"] + req.think_time_s,
                "Dependency or think time violated",
            )

    # Track overlapping native jobs, including time spent waiting for the one sender.
    # Count only jobs whose handler actually wrote objects to the native Store.
    def jobs(kind):
        items = rows(kind)
        records = {(e["request_id"], e["job"]): e["seq"] for e in items}
        require(len(records) == len(items), "Duplicate native job event")
        return records

    queued, started, done = (
        jobs(kind) for kind in ("save_job_submitted", "save_job_started", "save_job_finished")
    )
    require(
        queued and queued.keys() == started.keys() == done.keys(), "Unfinished native save jobs"
    )
    writes = {}
    for key, submit in queued.items():
        rid, _ = key
        if any(started[key] < e["seq"] < done[key] for e in rows("store_put_end", rid)):
            writes.setdefault(rid, []).append((submit, done[key]))
    shared = set()
    for step in rows("compute_begin"):
        for a, b in combinations(sorted(step["request_ids"]), 2):
            if specs[a].session_id == specs[b].session_id:
                continue
            if any(
                max(x[0], y[0]) < min(x[1], y[1])
                for x in writes.get(a, [])
                for y in writes.get(b, [])
            ):
                shared.add((a, b))
    require(bool(shared), "Missing concurrent sibling saves")
    require(any(len(spec.dependencies) >= 2 for spec in specs.values()), "Missing parent join")
    puts = rows("store_put_end")
    objects = {}
    for e in puts:
        for key, size, sha, code in zip(
            e["keys"], e["object_bytes"], e["digests"], e["result"], strict=True
        ):
            require(
                code == 0 and objects.setdefault(key, (size, sha)) == (size, sha),
                "Failed or inconsistent PUT",
            )
    probe = one("probe_replicas", restore)
    get = one("store_get_begin", restore)
    native_end = one("store_get_end", restore)
    initial = {
        k for e in rows("producer_replicas", source) for k, v in e["tiers"].items() if "MEMORY" in v
    }
    require(
        set(get["keys"]) <= initial
        and set(get["keys"]) == set(native_end["keys"])
        and all(probe["tiers"][k] == get["tiers"][k] == ["LOCAL_DISK"] for k in get["keys"]),
        "No memory-to-disk-only transition for restored keys",
    )
    require(
        all(
            any(fmt in k for k in get["keys"])
            for fmt in ("@store_format:mamba_state@", "@store_format:attention_head_major@")
        ),
        "Restore did not include both families",
    )
    require(
        all(native_end["checksum_ok"])
        and native_end["result"] == [objects[k][0] for k in native_end["keys"]],
        "Incomplete or corrupt restored objects",
    )
    require(
        probe["seq"] < one("request_admitted", restore)["seq"] < get["seq"],
        "Wrong restore probe ordering",
    )
    require(
        len(chains) == 1
        and chains[0]["request_id"] == restore
        and chains[0]["external_tokens"] == len(specs[source].prompt_token_ids)
        and specs[restore].prompt_token_ids[: chains[0]["external_tokens"]]
        == specs[source].prompt_token_ids,
        "Wrong restored prefix",
    )
    device_sample = one("femu_block_io_during_get", restore)["delta"]
    require(
        device_sample["reads"] > 0
        and device_sample["read_sectors"] * 512 == chains[0]["femu_read_bytes_during_get"],
        "Device read counters and block trace disagree",
    )
    omitted = {k for e in rows("kda_not_submitted", incomplete) for k in e["keys"]}
    partial = one("probe_replicas", miss)
    require(
        partial["seq"] < one("request_admitted", miss)["seq"]
        and omitted
        and all(partial["tiers"][k] == [] and "@store_format:mamba_state@" in k for k in omitted),
        "Missing checkpoint not demonstrated",
    )
    for k in omitted:
        require(
            any(
                v
                and "@store_format:attention_head_major@" in mla
                and mla.rsplit("@", 1)[1] == k.rsplit("@", 1)[1]
                for mla, v in partial["tiers"].items()
            ),
            "Matching attention prefix is absent",
        )
        writers = [e for e in puts if k in e["keys"]]
        require(
            writers and writers[0]["seq"] > partial["seq"] and writers[0]["request_id"] == miss,
            "Checkpoint was not first published by recomputation",
        )
    prefix = specs[incomplete].prompt_token_ids
    require(
        specs[miss].prompt_token_ids[: len(prefix)] == prefix,
        "Missing-state probe used a different prefix",
    )
    require(
        all(e["request_id"] == incomplete for e in rows("kda_not_submitted")),
        "Fault affected other writers",
    )
    scheduled = sum(e["scheduled_tokens"].get(miss, 0) for e in rows("compute_begin"))
    require(
        not rows("store_get_begin", miss)
        and metrics[miss]["external_load_tokens"] == 0
        and scheduled == len(specs[miss].prompt_token_ids) + specs[miss].output_tokens - 1,
        "Missing KDA did not trigger full prefill",
    )
    inventory = one("final_replica_inventory")
    require(
        set(inventory["tiers"]) == set(objects)
        and set(probe["tiers"]) == set(inventory["required_disk_keys"])
        and all("LOCAL_DISK" in inventory["tiers"][k] for k in inventory["required_disk_keys"]),
        "Final required-object durability failed",
    )
    return {
        "requests": len(specs),
        "concurrent_sibling_save_pairs": sorted(shared),
        "objects": len(objects),
        "native_save_jobs": len(queued),
        "watermark_eviction_cycles": triggers,
        "peak_sampled_memory_ratio": peak,
        "restored_tokens": chains[0]["external_tokens"],
        "disk_only_get_objects": len(get["keys"]),
        "object_read_bytes": get["bytes"],
        "device_read_bytes": device_sample["read_sectors"] * 512,
        "missing_kda_keys": sorted(omitted),
        "recomputed_prompt_tokens": len(specs[miss].prompt_token_ids),
        "required_final_disk_objects": len(inventory["required_disk_keys"]),
        "final_tiers": dict(
            Counter("+".join(sorted(t)) or "absent" for t in inventory["tiers"].values())
        ),
    }


def compare_host_controls(high, low):
    """Check paired host observations; device evidence is collected separately in FEMU."""

    def read(run, name):
        return json.loads((run / name).read_text())

    results = []
    comparable = []
    for path in (high, low):
        manifest = read(path, "run_manifest.json")
        cfg = dict(manifest["pressure_config"])
        watermark = cfg.pop("high_watermark")
        command = read(path, "mooncake-master.json")["command"]
        require(
            f"--eviction_high_watermark_ratio={watermark}" in command
            and f"--default_kv_lease_ttl={cfg['lease_ttl_ms']}" in command,
            "Host master flags differ",
        )
        summary = read(path, "summary.json")
        require(
            summary["status"] == "passed"
            and not summary["checksum_errors"]
            and not manifest["cold_between_turns"]
            and not summary["femu_verified"],
            "Invalid host trial",
        )
        events = [json.loads(line) for line in (path / "events.jsonl").read_text().splitlines()]
        (get,) = [
            e
            for e in events
            if e["event"] == "store_get_begin" and e["request_id"] == cfg["restore"]
        ]
        (done,) = [
            e for e in events if e["event"] == "store_get_end" and e["request_id"] == cfg["restore"]
        ]
        require(all(done["checksum_ok"]) and all(n > 0 for n in done["result"]), "Failed host GET")
        counts = [
            int(c)
            for c in re.findall(
                r"\[EVICT-RESULT\] evicted_count=(\d+)", (path / "mooncake-master.log").read_text()
            )
        ]
        objects = {
            k: (n, h)
            for e in events
            if e["event"] == "store_put_end"
            for k, n, h in zip(e["keys"], e["object_bytes"], e["digests"], strict=True)
        }
        comparable.append(
            (
                cfg,
                {
                    k: manifest[k]
                    for k in (
                        "harness_source_sha256",
                        "versions",
                        "hybrid_config",
                        "num_blocks",
                        "max_model_len",
                        "decode_ms",
                        "prefill_ms",
                        "read_delay_ms",
                        "input_sha256",
                        "timing",
                        "offload_use_uring",
                    )
                },
                objects,
                get["keys"],
                done["result"],
            )
        )
        results.append(
            {
                "watermark": watermark,
                "restore_tiers": get["tiers"],
                "positive_eviction_cycles": sum(c > 0 for c in counts),
                "requests": summary["requests"],
                "distinct_objects": len(objects),
            }
        )
    require(comparable[0] == comparable[1], "Host trial workloads or object contents differ")
    h, l = results
    require(
        h["watermark"] > l["watermark"]
        and h["positive_eviction_cycles"] == 0 < l["positive_eviction_cycles"],
        "Watermark intervention did not drive host evictions",
    )
    require(
        all("MEMORY" in t for t in h["restore_tiers"].values())
        and all(t == ["LOCAL_DISK"] for t in l["restore_tiers"].values()),
        "Host tier comparison failed",
    )
    return {
        "status": "passed",
        "lease_ttl_ms": comparable[0][0]["lease_ttl_ms"],
        "high": h,
        "low": l,
    }


if __name__ == "__main__":
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser(description="Compare high and low watermark host controls")
    parser.add_argument("high", type=Path)
    parser.add_argument("low", type=Path)
    args = parser.parse_args()
    print(json.dumps(compare_host_controls(args.high, args.low), indent=2))
