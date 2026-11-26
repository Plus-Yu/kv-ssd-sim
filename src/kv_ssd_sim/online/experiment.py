"""Run isolated A/B trials and validate the observable feedback chain."""

from __future__ import annotations

import csv
import json
import math
import subprocess
import sys
from pathlib import Path

from kv_ssd_sim.events import load_events
from kv_ssd_sim.online.runtime import PROJECT_ROOT, native_environment


def validate_trial(path: Path) -> dict:
    summary = json.loads((path / "summary.json").read_text())
    events = load_events(path / "events.jsonl")
    rows = summary["request_metrics"]
    if len(rows) != 2:
        raise AssertionError("The cold-prefix experiment requires exactly two requests")
    rid = rows[1]["request_id"]

    def find(name, *, request=True):
        matches = [
            e for e in events if e["event"] == name and (not request or e.get("request_id") == rid)
        ]
        if not matches:
            raise AssertionError(f"Missing {name} event for {rid}")
        return matches[0]

    requested = find("load_requested")
    get_start = find("store_get_begin")
    native_start = find("native_get_begin")
    get_end = find("store_get_end")
    completed = find("load_complete")
    resumed = next(e for e in events if e["event"] == "compute_begin" and rid in e["request_ids"])
    chain = [requested, get_start, native_start, get_end, completed, resumed]
    if [e["time_ns"] for e in chain] != sorted(e["time_ns"] for e in chain):
        raise AssertionError("Storage completion did not causally precede request compute")
    if not get_end["checksum_ok"] or not all(get_end["checksum_ok"]):
        raise AssertionError("Restored buffers did not pass checksum verification")
    if not get_start["tiers"] or any(
        not tiers or any("MEMORY" in t for t in tiers) or not any("DISK" in t for t in tiers)
        for tiers in get_start["tiers"].values()
    ):
        raise AssertionError("The restore did not use disk-only replicas")
    admitted = find("request_admitted")
    if find("memory_replicas_evicted", request=False)["seq"] >= admitted["seq"]:
        raise AssertionError("Cold eviction must complete before request 2 admission")
    if summary["checksum_errors"] or rows[1]["external_load_tokens"] <= 0:
        raise AssertionError("Native cache restoration failed")
    for row in rows:
        if row["output_tokens"] != row["target_output_tokens"]:
            raise AssertionError("Synthetic output length differs from the trace")
    injected_ms = (native_start["time_ns"] - get_start["time_ns"]) / 1e6
    if injected_ms + 0.2 < summary["read_delay_ms"]:
        raise AssertionError("Receiver delay finished before its configured deadline")
    return {
        "read_delay_ms": summary["read_delay_ms"],
        "observed_delay_ms": injected_ms,
        "native_store_get_ms": (get_end["time_ns"] - native_start["time_ns"]) / 1e6,
        "restore_ms": (completed["time_ns"] - requested["time_ns"]) / 1e6,
        "request_2_ttft_ms": rows[1]["ttft_ms"],
        "request_2_latency_ms": rows[1]["latency_ms"],
        "request_2_compute_budget_ms": rows[1]["compute_budget_ms"],
        "request_2_external_tokens": rows[1]["external_load_tokens"],
        "checksum_verified": True,
        "causal_chain_verified": True,
        "disk_only_verified": True,
    }


def run_demo(output: Path, delays: list[float]) -> dict:
    if len(delays) < 2 or any(not math.isfinite(d) or d < 0 for d in delays):
        raise ValueError("Supply at least two finite, non-negative read delays")
    if len(set(delays)) != len(delays):
        raise ValueError("Read delays must be distinct")
    delays = sorted(delays)
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    comparisons = []
    for index, delay in enumerate(delays):
        trial = output / f"trial-{index:02d}-{delay:g}ms"
        command = [
            sys.executable,
            "-m",
            "kv_ssd_sim",
            "replay",
            str(PROJECT_ROOT / "examples/two_requests.weka.json"),
            "--output",
            str(trial),
            "--read-delay-ms",
            str(delay),
            "--cold-between-turns",
            "--timing",
            "causal",
        ]
        with (output / f"trial-{index:02d}.log").open("w") as log:
            result = subprocess.run(
                command,
                stdout=log,
                stderr=subprocess.STDOUT,
                env=native_environment(),
                timeout=180,
                check=False,
            )
        if result.returncode:
            raise RuntimeError(f"Trial failed; inspect {output / f'trial-{index:02d}.log'}")
        comparisons.append(validate_trial(trial))
    baseline = comparisons[0]
    for row in comparisons[1:]:
        if row["request_2_compute_budget_ms"] != baseline["request_2_compute_budget_ms"]:
            raise AssertionError("Compute budget changed during a storage-only intervention")
        increase = row["read_delay_ms"] - baseline["read_delay_ms"]
        if row["restore_ms"] - baseline["restore_ms"] < increase * 0.75 - 5:
            raise AssertionError("Restore latency did not respond to the injected storage delay")
        if row["request_2_ttft_ms"] - baseline["request_2_ttft_ms"] < increase * 0.5 - 5:
            raise AssertionError("Request TTFT did not respond to the restore delay")
    with (output / "comparison.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(comparisons[0]))
        writer.writeheader()
        writer.writerows(comparisons)
    report = {
        "status": "passed",
        "trials": comparisons,
        "femu_verified": False,
        "scope": "native CPU vLLM + native Mooncake disk offload; receiver delay injection",
    }
    (output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    lines = [
        "# 第一版闭环验收结果",
        "",
        "原生 vLLM 调度器、KV 块管理与 Mooncake Store；CPU 占位 KV；固定计算 step。",
        "存储扰动是 receiver 线程中的延迟注入。本结果没有使用 FEMU，也不代表 NAND/FTL 性能。",
        "",
        "| 注入延迟 ms | 恢复 ms | 第二请求 TTFT ms | 计算预算 ms |",
        "|---:|---:|---:|---:|",
    ]
    for row in comparisons:
        lines.append(
            f"| {row['read_delay_ms']:.1f} | {row['restore_ms']:.2f} | "
            f"{row['request_2_ttft_ms']:.2f} | {row['request_2_compute_budget_ms']:.1f} |"
        )
    lines += [
        "",
        "每组通过磁盘副本、内存淘汰、恢复校验和及加载完成先于计算的事件链检查。",
        "这是功能验收，每个延迟点只运行一次；性能研究需预热并重复测量。",
        "",
    ]
    (output / "report.md").write_text("\n".join(lines))
    return report
