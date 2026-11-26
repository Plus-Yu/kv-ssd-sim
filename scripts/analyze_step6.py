#!/usr/bin/env python3
"""Rebuild the two-turn WEKA acceptance and matched-repeat statistics from FEMU records."""

import argparse
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict, deque
from itertools import takewhile
from pathlib import Path
from statistics import median

from compare_femu import require_matched_trials

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from run_step6 import ORDER, READ_NS, SLICE_SHA

from kv_ssd_sim.analysis.femu import verify


def require(value, message):
    if not value:
        raise ValueError(message)


def read(path):
    return json.loads(path.read_text())


def lines(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def block_totals(path):
    """Guest namespace data I/O throughout the trace, including filesystem metadata."""
    pending = defaultdict(deque)
    samples = {"R": [], "W": []}
    sizes = Counter()
    pattern = re.compile(
        r"(\d+\.\d+): block_rq_(issue|complete): (\d+,\d+) ([RW])\S* .*? (\d+) \+ (\d+)"
    )
    for line in path.read_text().splitlines():
        match = pattern.search(line)
        if not match:
            continue
        stamp, stage, device, op, sector, size = match.groups()
        t = round(float(stamp) * 1e6)
        queue = pending[device, op, sector, size]
        if stage == "issue":
            queue.append(t)
        else:
            require(queue and t >= queue[0], "Unmatched data I/O in trace")
            samples[op].append((t - queue.popleft()) / 1000)
            sizes[op] += int(size) * 512
    require(not any(pending.values()), "Outstanding data I/O at trace end")
    return {
        op: {"count": len(v), "bytes": sizes[op], "mean_ms": sum(v) / len(v) if v else None}
        for op, v in samples.items()
    }


def describe(folder, proof):
    joint = folder / "joint"
    events = lines(joint / "server/events.jsonl")
    metrics = read(joint / "server/summary.json")["request_metrics"]
    traffic = sorted(
        lines(joint / "client/profile_export.jsonl"),
        key=lambda e: e["metadata"]["source_outer_idx"],
    )
    trace = read(folder / "input/trace.json")
    expected = trace["requests"]
    require(
        proof["guest_inputs_sha256"]["trace.json"] == SLICE_SHA
        and len(traffic) == len(metrics) == len(expected) == 2,
        "Wrong trace slice or incomplete coverage",
    )
    require(
        len(proof["chains"]) == 1 and proof["chains"][0]["source_turn"] == [1, None],
        "Missing second-turn restore",
    )
    chain = proof["chains"][0]
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        Path(__file__).resolve().parents[1] / "configs/sim-tokenizer", local_files_only=True
    )
    raw = sorted(
        lines(joint / "client/profile_export_raw.jsonl"),
        key=lambda r: r["metadata"]["source_outer_idx"],
    )
    ids = [
        tokenizer.apply_chat_template(
            r["payload"]["messages"], tokenize=True, add_generation_prompt=True, return_dict=False
        )
        for r in raw
    ]
    shared = sum(1 for _ in takewhile(lambda pair: pair[0] == pair[1], zip(*ids)))
    require(shared >= 71168 and shared // 64 * 64 == 71168, "Wire token prefix differs from trace")
    turns = []
    signatures = []
    published = {}
    for index, (record, req) in enumerate(zip(traffic, expected, strict=True)):
        meta = record["metadata"]
        (receive,) = [
            e
            for e in events
            if e["event"] == "http_received" and e["client_request_id"] == meta["x_request_id"]
        ]
        wire_digest = hashlib.sha256(b"".join(t.to_bytes(4, "big") for t in ids[index])).hexdigest()
        require(
            wire_digest == receive["prompt_sha256"], "Client wire tokens and server prompt differ"
        )
        rid = receive["request_id"]
        (m,) = [row for row in metrics if row["request_id"] == rid]
        own = [e for e in events if e.get("request_id") == rid]
        ops = {verb: Counter() for verb in ("put", "get")}
        stored = {}
        loaded = []
        for e in own:
            if e["event"] == "store_put_end":
                for k, n, h, rc in zip(
                    e["keys"], e["object_bytes"], e["digests"], e["result"], strict=True
                ):
                    require(
                        rc == 0 and stored.setdefault(k, (n, h)) == (n, h),
                        "PUT failed or key changed",
                    )
                    require(
                        published.setdefault(k, (n, h)) == (n, h),
                        "Different bytes published across turns",
                    )
                    group = family(k)
                    ops["put"][group + "_objects"] += 1
                    ops["put"][group + "_bytes"] += n
            if e["event"] == "store_get_end":
                for k, n, ok in zip(e["keys"], e["result"], e["checksum_ok"], strict=True):
                    require(
                        n > 0 and ok and k in published and n == published[k][0],
                        "GET failed or length differs from PUT",
                    )
                    loaded.append((k, n))
                    group = family(k)
                    ops["get"][group + "_objects"] += 1
                    ops["get"][group + "_bytes"] += n
        if index:
            (start,) = [e for e in own if e["event"] == "store_get_begin"]
            require(
                all(t == ["LOCAL_DISK"] for t in start["tiers"].values())
                and set(start["keys"]) == set(start["tiers"]),
                "Restore includes a memory-resident object",
            )
            require(
                ops["get"]["MLA_objects"] > 0 and ops["get"]["KDA_objects"] > 0,
                "Partial hybrid restore",
            )
        steps = [
            e["scheduled_tokens"][rid]
            for e in events
            if e["event"] == "compute_begin" and rid in e["request_ids"]
        ]
        cached = m["external_load_tokens"]
        # Sum includes one decode step per generated token after the first.
        require(
            sum(steps) == req["in"] - cached + req["out"] - 1, "Scheduled tokens do not balance"
        )
        require(
            meta["source_outer_idx"] == index
            and req["in"] == m["input_tokens"]
            and req["out"] == m["output_tokens"],
            "Trace length mismatch",
        )
        gap = (
            None
            if not index
            else (meta["credit_issued_ns"] - traffic[index - 1]["metadata"]["request_end_ns"]) / 1e6
        )
        require(
            gap is None or gap >= 1000 * req["think_time"] - 1,
            "Second turn released before think time",
        )
        turns.append(
            {
                "turn": index,
                "input_tokens": req["in"],
                "output_tokens": req["out"],
                "external_tokens": cached,
                "computed_prompt_tokens": req["in"] - cached,
                "think_gap_ms": gap,
                "client_ttft_ms": record["metrics"]["time_to_first_token"]["value"],
                "client_latency_ms": record["metrics"]["request_latency"]["value"],
                "payload_preparation_ms": receive["preparation_ms"],
                **{
                    k: m[k]
                    for k in (
                        "ttft_ms",
                        "latency_ms",
                        "compute_budget_ms",
                        "compute_window_ms",
                        "deadline_overrun_ms",
                    )
                },
                **{op: dict(counts) for op, counts in ops.items()},
            }
        )
        # Random transport/request IDs have no bearing on logical equality.
        signatures.append((receive["prompt_sha256"], steps, sorted(stored.items()), sorted(loaded)))
    return {
        "trial": folder.name,
        "wire_shared_tokens": shared,
        "unique_published_objects": len(published),
        "unique_published_bytes": sum(n for n, _ in published.values()),
        "nand_read_ns": proof["nand_read_ns"],
        "turns": turns,
        "restore": {
            k: chain[k]
            for k in (
                "external_tokens",
                "cold_prepare_ms",
                "cold_end_to_client_start_ms",
                "prepared_to_admission_ms",
                "client_start_to_http_arrival_ms",
                "get_ms",
                "ttft_ms",
                "aiperf_ttft_ms",
                "femu_read_completions_during_get",
                "femu_read_bytes_during_get",
                "block_read_latency_ms",
            )
        },
        "device": block_totals(folder / "block-trace.txt"),
        "host_usage": read(folder / "host-usage.json")
        if (folder / "host-usage.json").exists()
        else None,
    }, signatures


def family(key):
    if "@store_format:mamba_state@" in key:
        return "KDA"
    if "@store_format:attention_head_major@" in key:
        return "MLA"
    raise ValueError(f"Unknown object family in {key}")


def distribution(values):
    return {"median": median(values), "min": min(values), "max": max(values)}


def summarize(root, prefix):
    runs = []
    for index, suffix in enumerate(ORDER):
        folder = root / f"{prefix}-{suffix}"
        proof = verify(folder)
        require(
            proof["nand_read_ns"] == READ_NS["slow" if suffix.startswith("slow") else "fast"],
            "Wrong device setting",
        )
        profile = read(folder / "input/config.json")
        require(
            (
                profile["num_blocks"],
                profile["max_model_len"],
                profile["prefill_ms"],
                profile["decode_ms"],
            )
            == (4096, 81920, 10, 2),
            "Wrong experiment profile",
        )
        data, signature = describe(folder, proof)
        if index == 0:
            first, first_proof, baseline = folder, proof, signature
        else:
            require_matched_trials(first, folder, first_proof, proof)
            require(signature == baseline, "Prompt IDs, steps or Store payloads differ across runs")
        require(
            data["host_usage"]
            and data["host_usage"]["start_unix_ns"] <= data["host_usage"]["end_unix_ns"],
            "Missing host chronology",
        )
        if runs:
            require(
                runs[-1]["host_usage"]["end_unix_ns"] <= data["host_usage"]["start_unix_ns"],
                "Trial windows overlap or group order differs",
            )
        runs.append(data)
    metrics = {
        "client_ttft_ms": lambda r: r["restore"]["aiperf_ttft_ms"],
        "native_ttft_ms": lambda r: r["restore"]["ttft_ms"],
        "native_get_ms": lambda r: r["restore"]["get_ms"],
        "block_mean_ms": lambda r: r["restore"]["block_read_latency_ms"]["mean"],
        "get_block_bytes": lambda r: r["restore"]["femu_read_bytes_during_get"],
        "cold_prepare_ms": lambda r: r["restore"]["cold_prepare_ms"],
        "request_latency_ms": lambda r: r["turns"][1]["client_latency_ms"],
    }
    stats = {}
    deltas = [{} for _ in range(3)]
    for k, value in metrics.items():
        groups = {
            name: [value(r) for r in runs if r["nand_read_ns"] == ns]
            for name, ns in READ_NS.items()
        }
        stats[k] = {name: distribution(v) for name, v in groups.items()}
        for i, (a, b) in enumerate(zip(groups["fast"], groups["slow"], strict=True)):
            deltas[i][k] = b - a
        stats[k]["paired_difference"] = distribution([row[k] for row in deltas])
    return {
        "status": "passed",
        "scope": "two original WEKA turns, scaled TP1 hybrid, controlled cold cache",
        "input_sha256": SLICE_SHA,
        "order": list(ORDER),
        "trials": runs,
        "statistics": stats,
        "pair_deltas_slow_minus_fast": deltas,
        "all_pairs_latency_increased": all(
            row[k] > 0
            for row in deltas
            for k in ("client_ttft_ms", "native_ttft_ms", "native_get_ms", "block_mean_ms")
        ),
        "all_get_device_bytes_equal": len(
            {r["restore"]["femu_read_bytes_during_get"] for r in runs}
        )
        == 1,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--prefix", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = summarize(args.root, args.prefix)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["statistics"], indent=2))
