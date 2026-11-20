"""Read-only WEKA inventory and conservative, dependency-complete prefix selection.

No tokens are synthesized and no recorded lengths/times/hashes are changed.
Complex branches are inventoried, but selection currently accepts only an
unambiguous serial parent prefix; AIPerf remains the replay dependency owner.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import statistics
from collections import Counter
from itertools import pairwise
from pathlib import Path


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _integer(value, minimum=0):
    return type(value) is int and value >= minimum


def _time(value):
    return type(value) in (int, float) and math.isfinite(value) and value >= 0


def _stats(values):
    if not values:
        return {"count": 0, "min": None, "median": None, "max": None, "sum": 0}
    return {
        "count": len(values),
        "min": min(values),
        "median": statistics.median(values),
        "max": max(values),
        "sum": sum(values),
    }


def shared_complete_tokens(previous, current, block_size):
    """Only an equal ordered prefix counts; unidentified partial tails do not."""
    count = 0
    for left, right in zip(previous["hash_ids"], current["hash_ids"]):
        if left != right:
            break
        count += 1
    complete = min(previous["in"], current["in"]) // block_size
    return min(count, complete) * block_size


def audit_trace(trace, *, context_limit=131072, match_unit=128, model_layers=93):
    if not isinstance(trace, dict) or not isinstance(trace.get("requests"), list):
        raise TypeError("Expected one WEKA trace object with a requests array")
    if not trace.get("id") or trace.get("hash_id_scope") != "local":
        raise ValueError("A nonempty trace ID and explicit local hash scope are required")
    block_size = trace.get("block_size")
    if not _integer(block_size, 1):
        raise ValueError("block_size must be a positive integer")
    if not all(_integer(v, 1) for v in (context_limit, match_unit, model_layers)):
        raise ValueError("Audit limits must be positive integers")
    records, branches, issues, prefixes = [], [], [], []
    previous_by_stream = {}

    def walk(entries, stream, depth, pointer):
        for index, entry in enumerate(entries):
            path = f"{pointer}/{index}"
            if not isinstance(entry, dict):
                issues.append({"path": path, "reason": "entry is not an object"})
                continue
            kind = entry.get("type")
            if kind == "subagent":
                branches.append(
                    {
                        "path": path,
                        "depth": depth + 1,
                        "agent_id": entry.get("agent_id"),
                        "status": entry.get("status"),
                    }
                )
                children = entry.get("requests")
                if not isinstance(children, list):
                    issues.append({"path": path, "reason": "subagent requests is not an array"})
                else:
                    walk(children, path, depth + 1, path + "/requests")
                continue
            if kind not in {"n", "s"}:
                issues.append({"path": path, "reason": f"unsupported entry type: {kind!r}"})
                continue
            errors = []
            if not _integer(entry.get("in"), 1):
                errors.append("in must be a positive integer")
            if not _integer(entry.get("out"), 1):
                errors.append("out must be positive for the current harness")
            if not _time(entry.get("t")):
                errors.append("t must be finite and nonnegative")
            for name in ("think_time", "api_time"):
                if entry.get(name) is not None and not _time(entry[name]):
                    errors.append(f"{name} must be finite and nonnegative")
            hashes = entry.get("hash_ids")
            if not isinstance(hashes, list) or any(type(h) is not int for h in hashes):
                errors.append("hash_ids must be an integer array")
            elif _integer(entry.get("in"), 1) and len(hashes) not in {
                entry["in"] // block_size,
                math.ceil(entry["in"] / block_size),
            }:
                errors.append("hash count does not cover the recorded input length")
            if not isinstance(entry.get("model"), str) or not entry["model"]:
                errors.append("missing source model")
            if errors:
                issues.extend({"path": path, "reason": error} for error in errors)
                continue
            row = {
                "path": path,
                "stream": stream,
                "depth": depth,
                "type": kind,
                "source_model": entry["model"],
                "input_tokens": entry["in"],
                "output_tokens": entry["out"],
                "context_tokens": entry["in"] + entry["out"],
                "recorded_t_s": entry["t"],
                "think_time_s": entry.get("think_time"),
                "recorded_api_time_s": entry.get("api_time"),
                "hash_count": len(hashes),
                "partial_tail_tokens": entry["in"] % block_size,
                "partial_tail_hash_recorded": len(hashes) == math.ceil(entry["in"] / block_size),
                "context_limit_exceeded": entry["in"] + entry["out"] > context_limit,
            }
            previous = previous_by_stream.get(stream)
            if previous:
                old_path, old = previous
                shared = shared_complete_tokens(old, entry, block_size)
                prefixes.append(
                    {
                        "previous": old_path,
                        "current": path,
                        "shared_complete_tokens": shared,
                        "match_unit_aligned_tokens": shared // match_unit * match_unit,
                        "input_token_reuse_fraction": shared / entry["in"],
                    }
                )
                if entry["t"] < old["t"]:
                    issues.append({"path": path, "reason": "timestamps retreat within stream"})
            previous_by_stream[stream] = (path, entry)
            records.append(row)

    walk(trace["requests"], "main", 0, "/requests")
    # Root timestamps are absolute. Branch timestamp conventions are not
    # guessed here; delegate their timing graph to a pinned AIPerf validation.
    intervals = []
    for row in records:
        if row["depth"] == 0 and row["recorded_api_time_s"]:
            begin = row["recorded_t_s"]
            intervals += [(begin, 1), (begin + row["recorded_api_time_s"], -1)]
    active = peak = 0
    for _, change in sorted(intervals):
        active += change
        peak = max(peak, active)
    parent = [r for r in records if r["depth"] == 0]
    raw_totals = trace.get("totals")
    computed = {
        "input": sum(r["input_tokens"] for r in records),
        "output": sum(r["output_tokens"] for r in records),
    }
    total_mismatch = (
        isinstance(raw_totals, dict)
        and "combined_tokens" in raw_totals
        and raw_totals["combined_tokens"] != computed
    )
    if total_mismatch:
        issues.append(
            {"path": "/totals", "reason": "declared combined totals disagree with records"}
        )
    return {
        "trace_id": trace["id"],
        "scope": "this trace file only; not the entire WEKA corpus",
        "top_level_entries": len(trace["requests"]),
        "valid_request_count": len(records),
        "parent_requests": len(parent),
        "subagent_requests": len(records) - len(parent),
        "explicit_subagents": len(branches),
        "max_explicit_nesting_depth": max((b["depth"] for b in branches), default=0),
        "request_types": dict(Counter(r["type"] for r in records)),
        "source_models": sorted({r["source_model"] for r in records}),
        "block_size": block_size,
        "hash_id_scope": "local",
        "match_unit": match_unit,
        "input_tokens": _stats([r["input_tokens"] for r in records]),
        "output_tokens": _stats([r["output_tokens"] for r in records]),
        "context_tokens": _stats([r["context_tokens"] for r in records]),
        "think_time_s": _stats(
            [r["think_time_s"] for r in records if r["think_time_s"] is not None]
        ),
        "recorded_parent_peak_inflight": peak,
        "concurrency_scope": "observed root request intervals [t, t+api_time); not replay concurrency",
        "context_limit": context_limit,
        "over_context_paths": [r["path"] for r in records if r["context_limit_exceeded"]],
        "prepared_payload_estimate": {
            "layers": model_layers,
            "digest_bytes_per_token_layer": 32,
            "all_requests_bytes": sum(r["context_tokens"] for r in records) * model_layers * 32,
            "largest_request_bytes": max((r["context_tokens"] for r in records), default=0)
            * model_layers
            * 32,
            "limit_bytes": 512 * 1024**2,
            "excludes": "KV buffers, staging, Python/tokenizer memory, and runtime overhead",
        },
        "consecutive_prefixes": prefixes,
        "records": records,
        "branches": branches,
        "issues": issues,
        "structural_validation_passed": not issues,
        "dependency_scope": (
            "serial parent-prefix selection only; no tool-call DAG reconstruction"
            if not branches
            else "branches inventoried; dependency closure requires separate validation"
        ),
    }


def select_parent_prefix(trace, count):
    """Select from the beginning, never omit predecessor turns or clip token counts."""
    if not _integer(count, 1) or count > len(trace["requests"]):
        raise ValueError("Selection count must be in the source request range")
    selected = trace["requests"][:count]
    if any(r.get("type") not in {"n", "s"} for r in selected):
        raise ValueError("Automatic selection only supports a serial parent prefix")
    if any(r.get("think_time") is None for r in selected):
        raise ValueError("Causal selection requires explicit think_time on every selected turn")
    for left, right in pairwise(selected):
        if left.get("api_time") is None:
            raise ValueError("Missing interval duration; cannot establish serial prefix")
        if right["t"] + 1e-6 < left["t"] + left["api_time"]:
            raise ValueError("Recorded requests overlap; cannot assume serial dependency")
        if shared_complete_tokens(left, right, trace["block_size"]) == 0:
            raise ValueError("No shared complete prefix; inspect possible flat agent boundaries")
    result = copy.deepcopy(trace)
    result["requests"] = copy.deepcopy(selected)
    result.pop("totals", None)  # Original full-file aggregate would be stale.
    return result


def write_audit_bundle(
    source,
    output,
    *,
    expected_sha256,
    count=2,
    context_limit=131072,
    match_unit=128,
    model_layers=93,
):
    actual = sha256(source)
    if actual != expected_sha256:
        raise ValueError(f"Source SHA256 mismatch: expected {expected_sha256}, got {actual}")
    trace = json.loads(source.read_text())
    options = {
        "context_limit": context_limit,
        "match_unit": match_unit,
        "model_layers": model_layers,
    }
    full = audit_trace(trace, **options)
    if full["issues"]:
        raise ValueError(f"Source audit failed: {full['issues']}")
    sliced = select_parent_prefix(trace, count)
    audit = audit_trace(sliced, **options)
    if audit["issues"] or audit["over_context_paths"]:
        raise ValueError("Selected prefix is invalid or exceeds the context limit")
    if audit["prepared_payload_estimate"]["all_requests_bytes"] > 512 * 1024**2:
        raise ValueError("Selected prefix exceeds the current all-at-once digest budget")
    output.mkdir(parents=True, exist_ok=False)
    paths = [f"/requests/{i}" for i in range(count)]
    # Provenance is a sidecar: AIPerf forbids unknown fields in its input schema.
    manifest = {
        "schema_version": 1,
        "source_sha256": actual,
        "source_file": source.name,
        "selection": "contiguous parent prefix from index zero",
        "selected_source_paths": paths,
        "omitted_request_count": len(trace["requests"]) - count,
        "record_fields_changed": [],
        "top_level_fields_removed": ["totals"] if "totals" in trace else [],
        "dependency_edges": [
            {"before": a, "after": b, "kind": "serial_parent_turn"} for a, b in pairwise(paths)
        ],
        "dependency_closure": True,
        "dependency_scope": "selected serial parent prefix, not complete tool execution DAG",
        "token_semantics": "source lengths/hash structure via synthetic tokenizer; not K3 retokenization",
        "timing": "causal; wait for predecessor completion then unchanged recorded think_time",
        "recorded_api_time_usage": "audit intervals only; never substitute for K3 compute latency",
        "runtime_validation": "not performed; no scheduler/Store/FEMU executed by this audit",
    }
    for name, value in [
        ("source-audit.json", full),
        ("slice.json", sliced),
        ("slice-audit.json", audit),
    ]:
        (output / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    manifest["slice_sha256"] = sha256(output / "slice.json")
    (output / "selection.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"
    )
    return manifest
