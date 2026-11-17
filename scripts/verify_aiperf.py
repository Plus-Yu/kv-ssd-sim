#!/usr/bin/env python3
"""Check trace coverage, wire token counts, prefix identity, and scheduler/SSE order."""

import argparse
import hashlib
import json
from itertools import pairwise
from pathlib import Path


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def verify(root):
    launch = json.loads((root / "launch.json").read_text())
    command = launch["client_command"]
    source = Path(command[command.index("--input-file") + 1])
    assert hashlib.sha256(source.read_bytes()).hexdigest() == launch["input_sha256"]
    trace = json.loads(source.read_text())
    expected = {}
    for outer, entry in enumerate(trace["requests"]):
        if entry["type"] == "subagent":
            for inner, child in enumerate(entry["requests"]):
                expected[(outer, inner)] = child
        else:
            expected[(outer, None)] = entry
    client = json.loads((root / "client/profile_export_aiperf.json").read_text())
    server = json.loads((root / "server/summary.json").read_text())
    config = json.loads((root / "server/run_manifest.json").read_text())
    events = read_jsonl(root / "server/events.jsonl")
    records = read_jsonl(root / "client/profile_export.jsonl")
    raw = read_jsonl(root / "client/profile_export_raw.jsonl")
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(config["tokenizer"], local_files_only=True)
    # Independently count the captured wire payload, even when AIPerf uses
    # server usage counters rather than its newline-joined text estimate.
    wire_counts = {}
    for row in raw:
        token_ids = tokenizer.apply_chat_template(
            row["payload"]["messages"],
            tokenize=True,
            add_generation_prompt=True,
            return_dict=False,
        )
        wire_counts[row["metadata"]["x_request_id"]] = len(token_ids)
    assert client["aiperf_version"] == "0.13.0"
    assert client["is_complete"] and not client["was_cancelled"]
    assert client["request_error_rate"]["avg"] == 0
    assert server["status"] == "passed" and server["checksum_errors"] == []
    femu = launch.get("femu_requested", False)
    assert server["femu_verified"] == config["fidelity"]["femu"] == femu
    if femu:
        assert config["read_delay_ms"] == 0 and server["femu_evidence"]
        assert any(e["event"] == "guest_page_cache_dropped" for e in events)
    assert len(records) == len(raw) == server["requests"] == len(expected)
    assert all(row["status"] == 200 for row in raw)
    received = {e["client_request_id"]: e for e in events if e["event"] == "http_received"}
    native_rows = {r["request_id"]: r for r in server["request_metrics"]}
    observed = {}
    for row in records:
        meta = row["metadata"]
        source_key = (meta["source_outer_idx"], meta.get("source_inner_idx"))
        assert meta["source_trace_id"] == trace["id"]
        assert source_key not in observed, f"Duplicated source turn: {source_key}"
        req = expected[source_key]
        event = received[meta["x_request_id"]]
        assert wire_counts[meta["x_request_id"]] == event["input_tokens"] == req["in"]
        rid = event["request_id"]
        metric = native_rows[rid]
        assert (
            event["input_tokens"] == row["metrics"]["input_sequence_length"]["value"] == req["in"]
        )
        assert (
            metric["output_tokens"]
            == row["metrics"]["output_sequence_length"]["value"]
            == req["out"]
        )
        assert row["metrics"]["usage_prompt_tokens"]["value"] == req["in"]
        assert row["metrics"]["usage_completion_tokens"]["value"] == req["out"]
        own = [e for e in events if e.get("request_id") == rid]
        first = next(e for e in own if e["event"] == "first_token")
        emitted = [e for e in own if e["event"] == "http_token_emitted"]
        assert len(emitted) == req["out"] and first["seq"] < emitted[0]["seq"]
        finished = next(e for e in own if e["event"] == "request_finished")
        response_end = next(e for e in own if e["event"] == "http_response_finished")
        assert finished["seq"] < response_end["seq"]
        observed[source_key] = {"record": row, "event": event, "metric": metric}
    assert observed.keys() == expected.keys(), "Trace coverage mismatch"
    # AIPerf can choose a different synthetic token vocabulary than the legacy
    # driver. Audit prefix equality on actual server tokens, not equal token IDs
    # between the two independent synthesis implementations.
    prefix_pairs = 0
    entries = list(observed.items())
    for i, (key_a, a) in enumerate(entries):
        for key_b, b in entries[i + 1 :]:
            shared = 0
            for ha, hb in zip(expected[key_a]["hash_ids"], expected[key_b]["hash_ids"]):
                if ha != hb:
                    break
                shared += 1
            tokens = min(shared * trace["block_size"], expected[key_a]["in"], expected[key_b]["in"])
            blocks = tokens // 64
            if blocks:
                assert (
                    a["event"]["prefix_block_sha256"][:blocks]
                    == b["event"]["prefix_block_sha256"][:blocks]
                )
                prefix_pairs += 1
    if launch["timing"] == "causal":
        conversations = {}
        for entry in observed.values():
            meta = entry["record"]["metadata"]
            conversations.setdefault(meta["conversation_id"], []).append(meta)
        for turns in conversations.values():
            turns.sort(key=lambda m: m["turn_index"])
            for previous, current in pairwise(turns):
                assert previous["request_end_ns"] <= current["request_start_ns"]
        # Explicit completed subagents ending before the next parent turn must
        # be joined. Async/background branches are not given this assertion.
        branches = []
        previous_parent = None
        for outer, entry in enumerate(trace["requests"]):
            if entry["type"] == "subagent":
                children = [
                    observed[(outer, inner)]["record"]["metadata"]
                    for inner in range(len(entry["requests"]))
                ]
                if previous_parent is not None and children:
                    assert previous_parent["request_end_ns"] <= children[0]["request_start_ns"]
                branches.append((entry, children))
            else:
                parent = observed[(outer, None)]["record"]["metadata"]
                for branch, children in branches:
                    if (
                        branch.get("status") == "completed"
                        and children
                        and branch.get("duration_ms") is not None
                        and branch["t"] + branch["duration_ms"] / 1000 <= entry["t"]
                    ):
                        assert (
                            max(c["request_end_ns"] for c in children) <= parent["request_start_ns"]
                        )
                branches = []
                previous_parent = parent
    restore = [e for e in events if e["event"] == "store_get_end"]
    for get in restore:
        assert all(get["checksum_ok"])
        own = [e for e in events if e.get("request_id") == get["request_id"]]
        loaded = next(e for e in own if e["event"] == "load_complete")
        first = next(e for e in own if e["event"] == "first_token")
        assert get["seq"] < loaded["seq"] < first["seq"]
    if config["cold_between_requests"] and prefix_pairs:
        assert restore, "Cold shared-prefix test did not restore from Store"
        assert any("@store_format:mamba_state@" in key for get in restore for key in get["keys"])
        assert any(
            "@store_format:attention_head_major@" in key for get in restore for key in get["keys"]
        )
    return {
        "status": "passed",
        "requests": len(records),
        "input_tokens": sum(r["in"] for r in expected.values()),
        "output_tokens": sum(r["out"] for r in expected.values()),
        "shared_prefix_pairs_verified": prefix_pairs,
        "external_load_tokens": sum(r["external_load_tokens"] for r in native_rows.values()),
        "restore_batches": len(restore),
        "max_native_batch_requests": max(
            (len(e["request_ids"]) for e in events if e["event"] == "compute_begin"), default=0
        ),
        "aiperf_ttft_avg_ms": client["time_to_first_token"]["avg"],
        "cold_between_requests": config["cold_between_requests"],
        "femu_verified": femu,
        "scope": (
            "AIPerf HTTP native scheduler + hybrid Store; synthetic character tokenizer; "
            + (
                "FEMU device counters verified; block trace verified separately"
                if femu
                else "host filesystem"
            )
        ),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    root = parser.parse_args().output
    try:
        result = verify(root)
    except Exception as error:
        (root / "verification.json").write_text(
            json.dumps({"status": "failed", "error": repr(error)}, indent=2) + "\n"
        )
        raise
    (root / "verification.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
