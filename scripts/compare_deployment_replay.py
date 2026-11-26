"""Audit the paired B300 deployment study on one frozen kvlab window."""

import argparse
import hashlib
import json
from collections import Counter
from dataclasses import asdict
from pathlib import Path

from compare_volume_replay import audit_run
from run_kvlab_matrix import DEPLOYMENT_SCENARIOS

from kv_ssd_sim.core.paths import PROJECT_ROOT
from kv_ssd_sim.simulation.timing import ContextTiming


def inspect_steps(folder, launch, requests):
    timing = ContextTiming(**launch["step_timing"]) if "compute_calibration" in launch else None
    bounds = launch["scheduler"]
    batches, copies, beyond_prompt = Counter(), Counter(), Counter()
    copy_wait = longest = max_tokens = outside = 0
    kinds, written = Counter(), Counter()
    with (folder / "events.jsonl").open() as stream:
        for line in stream:
            row = json.loads(line)
            op = row["event"]
            if op == "compute_step":
                batch, fresh = row["batch"], row["fresh_prompt_tokens"]
                scheduled = fresh + row["decode_only_sequences"]
                if batch > bounds["max_num_seqs"] or scheduled > bounds["max_num_batched_tokens"]:
                    raise ValueError("Actual step exceeds the native scheduler bounds")
                batches[batch] += 1
                max_tokens = max(max_tokens, scheduled)
                if timing:
                    contexts = row["context_before_tokens"]
                    work = list(zip(contexts, row["fresh_tokens_by_sequence"], strict=True))
                    if (
                        len(work) != batch
                        or sum(q for _, q in work) != fresh
                        or sum(q == 0 for _, q in work) != row["decode_only_sequences"]
                        or round(timing.step_ms(work) * 1e6) != row["duration_ns"]
                    ):
                        raise ValueError("Context-aware step time or token counts differ")
                    peak = max(h + q for h, q in work)
                    longest = max(peak, longest)
                    outside += peak > timing.decode_context_tokens[-1]
            elif op == "disk_write_submit":
                family = "MLA" if row["group"] == 3 else "KDA"
                kinds[family] += row["bytes"]
                written[family, row["key"], row["bytes"]] += 1
                if row["boundary_tokens"] > requests[row["request"]]["prompt_tokens"]:
                    beyond_prompt[family] += row["bytes"]
            elif op == "save_queued" and row["copy_bytes"]:
                # Legacy ledgers record the start of this job's fixed save overhead.
                start = row.get(
                    "copy_start_ns", row["start_ns"] + round(launch["store"]["save_ms"] * 1e6)
                )
                copy_wait += start - row.get("copy_ready_ns", start)
                copies[start] += 1
                copies[row["complete_ns"]] -= 1
            elif op in ("h2d_start", "h2d_complete"):
                copies[row["time_ns"]] += 1 if op == "h2d_start" else -1
    active = peak_copies = 0
    for ns in sorted(copies):
        active += copies[ns]
        peak_copies = max(peak_copies, active)
    lanes = launch["store"].get("host_copy_lanes")
    if active or (lanes and peak_copies > lanes):
        raise ValueError("Host copies exceeded their shared lane budget or failed to drain")
    return {
        "batch_histogram": dict(sorted(batches.items())),
        "peak_batch": max(batches),
        "peak_scheduled_tokens": max_tokens,
        "max_effective_context": longest if timing else None,
        "steps_above_long_decode_anchor": outside if timing else None,
        "compute_steps": batches.total(),
        "peak_simultaneous_host_copies": peak_copies,
        "d2h_lane_wait_ns_sum": copy_wait,
        "write_bytes_by_type": dict(kinds),
        "write_bytes_beyond_source_prompt": dict(beyond_prompt),
    }, written


def compare(folder, window):
    runs = {}
    policy_writes = {}
    base = json.loads((folder / "base/launch.json").read_bytes())
    config = (PROJECT_ROOT / "configs/compute/b300-tp8-context.json").read_bytes()
    plan = json.loads((PROJECT_ROOT / "configs/kimi-k3-b300-step2/resolved.json").read_bytes())
    model = json.loads((PROJECT_ROOT / "references/kimi-k3-b300/model-config.json").read_bytes())
    expected_requests = {
        r["id"]: (r["input_tokens"], r["output_tokens"], r["prompt_sha256"])
        for r in window["source_requests"]
    }
    for case in DEPLOYMENT_SCENARIOS:
        d = folder / case
        launch = json.loads((d / "launch.json").read_bytes())
        requests = {r["id"]: r for r in json.loads((d / "requests.json").read_bytes())}
        if [x["sha256"] for x in launch["compiled_inputs"]] != [window["input_sha256"]] or {
            rid: (r["prompt_tokens"], r["output_tokens"], r["prompt_sha256"])
            for rid, r in requests.items()
        } != expected_requests:
            raise ValueError(f"Different source window in {case}")
        for field in (
            "geometry_sha256",
            "source_sha256",
            "max_model_len",
            "num_blocks",
            "root_offsets_s",
            "native_master",
            "restore_path",
            "vllm_version",
            "pythonhashseed",
        ):
            if base[field] != launch[field]:
                raise ValueError(f"{field} changed in {case}")
        expected_store = base["store"].copy()
        if case in ("ssd-4x", "ssd-8x"):
            for field in ("device_mib_s", "device_read_mib_s"):
                expected_store[field] *= int(case[4])
        if case == "host-half-rate":
            for field in ("put_mib_s", "memory_mib_s", "h2d_mib_s"):
                expected_store[field] *= 0.5
        if case == "host-4-lanes":
            expected_store["host_copy_lanes"] = 4
        if launch["store"] != expected_store:
            raise ValueError(f"Unexpected storage settings in {case}")
        if launch["profile"] != ("reference" if case == "legacy-clock" else "reference-context"):
            raise ValueError(f"Wrong timing preset in {case}")
        if case != "legacy-clock":
            cost = 2 if case == "context-cost-2x" else 1
            expected = ContextTiming.from_profile(
                json.loads(config), plan, model["text_config"], context_scale=cost
            )
            record = launch["compute_calibration"]
            if (
                record["sha256"] != hashlib.sha256(config).hexdigest()
                or record["parameters"] != json.loads(config)
                or record["context_cost_scale"] != cost
                or launch["step_timing"] != json.loads(json.dumps(asdict(expected)))
            ):
                raise ValueError(f"Unexpected timing anchors in {case}")
        result, _ = audit_run(d)
        if result["anchor"]["request"] != window["anchor"]:
            raise ValueError(f"Different source anchor in {case}")
        scheduler = launch["scheduler"]
        if (
            scheduler["max_num_seqs"] != (int(case[4:]) if case.startswith("seq-") else 8)
            or scheduler["max_num_batched_tokens"]
            != (int(case[7:]) if case.startswith("tokens-") else 4096)
            or scheduler["prefix_cache_retention_input"] != 0
            or scheduler["save_decode_cache"] != (case == "save-decode")
            or scheduler["prefix_cache_retention_interval"]
            != (1536 if case == "save-decode" else 0)
        ):
            raise ValueError(f"Native KDA retention disagrees with the policy in {case}")
        full = result["scopes"]["full"]
        steps, written = inspect_steps(d, launch, requests)
        if sum(steps["write_bytes_by_type"].values()) != full["write_bytes"]:
            raise ValueError(f"Object families do not reconcile in {case}")
        if case in ("base", "save-decode"):
            policy_writes[case] = written
        runs[case] = {
            "profile": launch["profile"],
            "scheduler": scheduler,
            "step_profile_sha256": launch.get("compute_calibration", {}).get("sha256"),
            "context_cost_scale": launch.get("compute_calibration", {}).get("context_cost_scale"),
            "ssd_service": launch["logical_volume"]["service_scenario"],
            "host_copy_lanes": launch["store"].get("host_copy_lanes"),
            "host_copy_scale": launch.get("host_copy_scale", 1),
            "unique_written_keys": result["unique_objects"],
            "write_identity_sha256": result["write_identity_sha256"],
            "file_writes": result["volume_ios"]["write"],
            "read_bytes": full["read_bytes"],
            "write_bytes": full["write_bytes"],
            "measurement_write_bytes": result["scopes"]["measurement"]["write_bytes"],
            "write_peak_10s_bytes": full["write_peak_10s_bytes"],
            "last_response_ns": result["last_request_finished_ns"],
            "last_io_complete_ns": result["last_io_complete_ns"],
            **steps,
        }

    def difference(a, b):
        counts = policy_writes[a] - policy_writes[b]
        return {
            kind: {
                "occurrences": sum(n for (g, _, _), n in counts.items() if g == kind),
                "bytes": sum(size * n for (g, _, size), n in counts.items() if g == kind),
            }
            for kind in ("MLA", "KDA")
        }

    return {
        "source_sha256": window["source_sha256"],
        "input_sha256": window["input_sha256"],
        "sessions": len(window["sessions"]),
        "requests": window["calls"],
        "geometry_sha256": base["geometry_sha256"],
        "vllm_version": base["vllm_version"],
        "runtime_sources_sha256": hashlib.sha256(
            json.dumps(base["source_sha256"], sort_keys=True).encode()
        ).hexdigest(),
        "decode_save_write_difference": {
            "only_enabled": difference("save-decode", "base"),
            "only_disabled": difference("base", "save-decode"),
        },
        "cases": runs,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", type=Path)
    parser.add_argument("--window", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(compare(args.runs, json.loads(args.window.read_bytes())), indent=2))
