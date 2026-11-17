#!/usr/bin/env python3
"""Compare an otherwise matched pair of FEMU NAND-read interventions."""

import argparse
import json
from pathlib import Path

from verify_femu import run_path, verify


def read(path):
    return json.loads(path.read_text())


def objects(path):
    result = {}
    for line in (run_path(path) / "events.jsonl").read_text().splitlines():
        event = json.loads(line)
        if event["event"] == "store_put_end":
            for key, size, digest in zip(
                event["keys"], event["object_bytes"], event["digests"], strict=True
            ):
                result[key] = (size, digest)
    return result


def compare(fast, slow):
    a, b = verify(fast), verify(slow)
    ma, mb = (read(run_path(p) / "run_manifest.json") for p in (fast, slow))
    joint = a["aiperf"] is not None
    if joint != (b["aiperf"] is not None):
        raise AssertionError("Workload types differ")
    fields = [
        "harness_source_sha256",
        "versions",
        "hybrid_config",
        "prefill_ms",
        "decode_ms",
        "read_delay_ms",
        "max_model_len",
        "num_blocks",
    ]
    fields += (
        ["tokenizer_sha256", "cold_between_requests", "max_inflight", "model"]
        if joint
        else [
            "input_sha256",
            "timing",
            "cold_between_turns",
            "cpu_kv_pool_bytes",
            "offload_staging_bytes",
        ]
    )
    for field in fields:
        if ma[field] != mb[field]:
            raise AssertionError(f"Trials differ in {field}")
    if joint:
        ja, jb = (read(p / "joint/launch.json") for p in (fast, slow))
        for field in ("input_sha256", "aiperf_version", "timing", "expected_requests"):
            if ja[field] != jb[field]:
                raise AssertionError(f"AIPerf runs differ in {field}")

        def normalize_http(command, path):
            command = list(command)
            for flag in ("--port", "--url"):
                if flag in command:
                    command[command.index(flag) + 1] = "EPHEMERAL_ENDPOINT"
            return [value.replace(path.name, "TRIAL") for value in command]

        for field in ("server_command", "client_command"):
            if normalize_http(ja[field], fast) != normalize_http(jb[field], slow):
                raise AssertionError(f"AIPerf launch differs in {field}")
    if objects(fast) != objects(slow):
        raise AssertionError("Object keys, sizes or payload digests differ")
    if a["nand_read_ns"] >= b["nand_read_ns"]:
        raise AssertionError("Expected fast NAND setting < slow NAND setting")
    la, lb = (read(p / "launch.json") for p in (fast, slow))
    if la["femu_revision"] != lb["femu_revision"]:
        raise AssertionError("FEMU revisions differ")

    # Only the run name and NAND read parameter may change in the QEMU command.
    def normalized_command(launch, path):
        return [
            str(value)
            .replace(path.name, "TRIAL")
            .replace(f"pg_rd_lat={launch['nand_read_ns']}", "pg_rd_lat=READ_NS")
            for value in launch["command"]
        ]

    if normalized_command(la, fast) != normalized_command(lb, slow):
        raise AssertionError("QEMU launch commands differ beyond the intended intervention")

    # Compare artifact hashes by relative role, ignoring each trial's output path.
    def hashes(launch, path):
        return {
            key.replace(path.name, "TRIAL"): value for key, value in launch["inputs_sha256"].items()
        }

    if hashes(la, fast) != hashes(lb, slow):
        raise AssertionError("Kernel, FEMU binary, initramfs or startup cache contents differ")
    x, y = a["chains"][0], b["chains"][0]
    for field in ("external_tokens", "compute_budget_ms", "femu_read_bytes_during_get"):
        if x[field] != y[field]:
            raise AssertionError(f"Restore workloads differ in {field}")
    increases = {
        "block_mean_ms": y["block_read_latency_ms"]["mean"] - x["block_read_latency_ms"]["mean"],
        "get_ms": y["get_ms"] - x["get_ms"],
        "ttft_ms": y["ttft_ms"] - x["ttft_ms"],
    }
    if joint:
        if x["source_turn"] != y["source_turn"]:
            raise AssertionError("Compared different restored HTTP source turns")
        increases["aiperf_ttft_ms"] = y["aiperf_ttft_ms"] - x["aiperf_ttft_ms"]
    passed = all(value > 0 for value in increases.values())
    return {
        "status": "passed" if passed else "failed",
        "failure_reason": None if passed else "Not all required latency measures increased",
        "checks": {
            "individual_causal_chains": True,
            "matched_workload_and_configuration": True,
            "device_get_native_ttft_increased": all(
                increases[field] > 0 for field in ("block_mean_ms", "get_ms", "ttft_ms")
            ),
            "all_required_latencies_increased": passed,
        },
        "fast": a,
        "slow": b,
        "increases": increases,
        "identical_object_count": len(objects(fast)),
        "scope": "one matched functional pair; not a statistical performance estimate",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("fast", type=Path)
    parser.add_argument("slow", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = compare(args.fast, args.slow)
    except (AssertionError, KeyError, ValueError, FileNotFoundError) as error:
        result = {"status": "failed", "failure_reason": str(error)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["status"] == "passed" else 1)
