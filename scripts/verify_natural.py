"""Audit a serial causal AIPerf FEMU replay with untouched cache tiers."""

import json
import re
from collections import Counter
from itertools import takewhile

from measure_ssd import check, parse_trace
from verify_aiperf import read_jsonl


def io_totals(blocks, begin, end):
    counters = Counter()
    for b in blocks:
        if b.stage == "issue" and begin <= b.time < end and b.op in ("R", "W"):
            counters[b.op] += 1
            counters[b.op + "_bytes"] += b.sectors * 512
    return {k: counters[k] for k in ("R", "R_bytes", "W", "W_bytes")}


def restored_batches(events):
    def selected(name):
        return [e for e in events if e["event"] == name]

    batches = []
    for begin, native, io, end in zip(
        selected("store_get_begin"),
        selected("native_get_begin"),
        selected("femu_block_io_during_get"),
        selected("store_get_end"),
        strict=True,
    ):
        check(begin["seq"] < native["seq"] < io["seq"] < end["seq"], "GET event order")
        check(begin["keys"] == native["keys"] == end["keys"] and begin["keys"], "GET key mismatch")
        tiers = Counter()
        for key, size, ok in zip(end["keys"], end["result"], end["checksum_ok"], strict=True):
            check(ok and size > 0, "Invalid restored object")
            tiers["+".join(sorted(begin["tiers"][key])) or "UNKNOWN"] += 1
        check(
            io["device"] == "nvme0n1" and io["delta"]["read_sectors"] >= 0, "Invalid GET counters"
        )
        batches.append(
            {
                "available_replicas_by_object": dict(tiers),
                "logical_bytes": sum(end["result"]),
                "device_read_bytes_in_get": io["delta"]["read_sectors"] * 512,
            }
        )
    return batches


def describe(path, events, manifest, summary, wire_shared=None):
    joint = path / "joint"
    trace = json.loads((path / "input/trace.json").read_text())
    launch = json.loads((joint / "launch.json").read_text())
    check(launch["timing"] == "causal" and not launch["prepare_before_send"], "Invalid client mode")
    check(not manifest["cold_between_requests"], "Natural mode unexpectedly cleared cache")
    evidence = summary["femu_evidence"]
    device = evidence["device"]
    check(
        device == "nvme0n1"
        and "FEMU" in evidence["model"]
        and f"/dev/{device} /mnt/femu " in (path / "guest-mounts.txt").read_text(),
        "Wrong mount",
    )
    # Verified single-namespace guest. Derive the major:minor from its traced events.
    raw_trace = (path / "block-trace.txt").read_text()
    devices = set(re.findall(r"block_rq_(?:issue|complete): (\d+,\d+)", raw_trace))
    check(len(devices) == 1, "Expected a single traced block device")
    capture_mode = json.loads((path / "launch.json").read_text()).get("block_capture")
    target = devices.pop()
    receipt = None
    if capture_mode:
        from verify_block_capture import check_receipt

        receipt = json.loads((path / "capture.json").read_text())
        entries = check_receipt(receipt, target, capture_mode)
    else:
        entries = None
    blocks, integrity = parse_trace(raw_trace, target, stream_entries=entries)
    if capture_mode == "buffer":
        buffered = sum(cpu["entries"] for cpu in receipt["per_cpu_end"].values())
        check(buffered == integrity["buffer_entries"], "Per-CPU count differs from ftrace header")
    check(not integrity["error_completions"], "Unsuccessful block completion")
    raw = {
        r["metadata"]["x_request_id"]: r
        for r in read_jsonl(joint / "client/profile_export_raw.jsonl")
    }
    traffic = sorted(
        read_jsonl(joint / "client/profile_export.jsonl"),
        key=lambda r: r["metadata"]["source_outer_idx"],
    )
    receive = {e["client_request_id"]: e for e in events if e["event"] == "http_received"}
    metrics = {m["request_id"]: m for m in summary["request_metrics"]}
    rows, previous_prompts = [], []
    previous_finish = None
    (origin,) = [e["monotonic_origin_ns"] for e in events if e["event"] == "clock_sync"]
    for idx, (rec, src) in enumerate(zip(traffic, trace["requests"], strict=True)):
        meta = rec["metadata"]
        check(
            meta["source_outer_idx"] == idx and meta.get("source_inner_idx") is None,
            "Expected serial turns",
        )
        cid = meta["x_request_id"]
        rx = receive[cid]
        rid = rx["request_id"]
        own = [e for e in events if e.get("request_id") == rid]
        start = raw[cid]["start_perf_ns"]
        response_end = (
            next(e for e in own if e["event"] == "http_response_finished")["time_ns"] + origin
        )
        check(start <= rx["time_ns"] + origin < response_end, "Invalid HTTP time window")
        think = (
            None if previous_finish is None else (meta["credit_issued_ns"] - previous_finish) / 1e9
        )
        if think is not None:
            check(think + 0.001 >= src["think_time"], "Think time shortened")
        previous_finish = meta["request_end_ns"]
        shared = 64 * max(
            (
                sum(1 for _ in takewhile(lambda t: t[0] == t[1], zip(p, rx["prefix_block_sha256"])))
                for p in previous_prompts
            ),
            default=0,
        )
        if wire_shared is not None:
            shared = wire_shared[cid]
        previous_prompts.append(rx["prefix_block_sha256"])
        m = metrics[rid]
        scheduled = sum(
            e["scheduled_tokens"].get(rid, 0) for e in events if e["event"] == "compute_begin"
        )
        computed = scheduled - (m["output_tokens"] - 1)
        external = m["external_load_tokens"]
        local = m["input_tokens"] - computed - external
        batches = restored_batches(own)
        check(
            0 < computed <= m["input_tokens"] and local >= 0 and external >= 0,
            "Prompt token balance",
        )
        check(bool(external) == bool(batches), "GET does not match external token count")
        check(
            external == sum(e["tokens"] for e in own if e["event"] == "load_requested"),
            "External load count",
        )
        rows.append(
            {
                "turn": idx + 1,
                "prompt_tokens": m["input_tokens"],
                "output_tokens": m["output_tokens"],
                "wire_prompt_sha256": rx["prompt_sha256"],
                "shared_prompt_block_tokens": shared,
                "shared_prompt_tokens_recomputed": max(0, shared - local - external),
                "local_reuse_tokens": local,
                "external_load_tokens": external,
                "computed_prompt_tokens": computed,
                "think_gap_s": think,
                "get_batches": batches,
                "http_start_ns": start,
                "response_end_ns": response_end,
                "device_io_during_request": io_totals(blocks, start, response_end),
                "device_io_since_previous_response": io_totals(
                    blocks, rows[-1]["response_end_ns"], start
                )
                if rows
                else None,
            }
        )
    check(
        sum(len(r["get_batches"]) for r in rows)
        == summary["restore_batches"]
        == len(evidence["get_io_samples"]),
        "Missing or orphaned Store GETs",
    )
    (snapshot,) = [e for e in events if e["event"] == "natural_final_replicas"]
    check(
        snapshot["time_ns"] + origin >= rows[-1]["response_end_ns"], "Snapshot before last response"
    )
    published = set()
    for e in events:
        if e["event"] == "store_put_end":
            for key, rc in zip(e["keys"], e["result"], strict=True):
                check(rc == 0, "Store PUT error")
                published.add(key)
    check(set(snapshot["tiers"]) == published, "Missing final replica states")
    check(
        summary["verified_disk_objects"]
        == sum("LOCAL_DISK" in t for t in snapshot["tiers"].values()),
        "Disk replica count mismatch",
    )
    start, end = rows[0]["http_start_ns"], rows[-1]["response_end_ns"]
    result = {
        "requests": rows,
        "block_trace_integrity": integrity,
        "session_io_first_http_send_to_last_response": io_totals(blocks, start, end),
        "full_trace_io": io_totals(blocks, blocks[0].time, blocks[-1].time + 1),
        "final_replica_states": dict(
            Counter("+".join(sorted(v)) or "ABSENT" for v in snapshot["tiers"].values())
        ),
    }
    if receipt:
        from verify_block_capture import audit_windows

        result["capture"] = audit_windows(
            path,
            blocks,
            rows,
            receive,
            receipt,
            align_counters=(manifest.get("hybrid_config") or {}).get("profile") == "k3-tp8",
        )
    return result
