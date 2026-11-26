"""Validate guest block completions -> Store GET -> native scheduler resumption."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import defaultdict, deque
from pathlib import Path


def run_path(path):
    launch = json.loads((path / "launch.json").read_text())
    return path / ("joint/server" if launch.get("workload") == "aiperf" else "replay")


def verify_inputs(path, launch, manifest):
    directory = path / "input"
    names = ["trace.json", "hybrid.json"]
    if (directory / "pressure.json").exists():
        names.append("pressure.json")
        if manifest.get("pressure_config") != json.loads((directory / "pressure.json").read_text()):
            raise AssertionError("Runtime capacity profile differs")
    actual = {
        name: hashlib.sha256((directory / name).read_bytes()).hexdigest()
        for name in [*names, "config.json"]
    }
    receipt = json.loads((path / "guest-input.json").read_text())
    cfg = json.loads((directory / "config.json").read_text())
    if actual != launch["guest_inputs_sha256"] or actual != receipt["sha256"]:
        raise AssertionError("Host and guest input hashes differ")
    if cfg["files"] != {name: actual[name] for name in names}:
        raise AssertionError("Inconsistent configuration file hashes")
    if cfg["workload"] != launch["workload"] or any(
        manifest[key] != cfg[key]
        for key in ("num_blocks", "max_model_len", "prefill_ms", "decode_ms")
    ):
        raise AssertionError("Runtime did not consume requested settings")
    supplied = json.loads((directory / "hybrid.json").read_text())
    if any(manifest["hybrid_config"].get(key) != value for key, value in supplied.items()):
        raise AssertionError("Runtime hybrid geometry differs from staged config")
    if launch["workload"] == "aiperf":
        client = json.loads((path / "joint/launch.json").read_text())
        source_sha = client["input_sha256"]
        mode = cfg.get("cache_mode", "cold")
        if mode not in ("natural", "cold"):
            raise AssertionError("Invalid cache mode")
        cold = mode == "cold"
        if (
            client.get("prepare_before_send") is not cold
            or manifest["cold_between_requests"] is not cold
        ):
            raise AssertionError("Cache mode differs from staged configuration")
        for argv, flag in (
            (receipt["command"], "--prepare-before-send"),
            (receipt["command"], "--cold-between-requests"),
            (client["server_command"], "--cold-between-requests"),
        ):
            if (flag in argv) != cold:
                raise AssertionError("Effective cache flags differ from staged configuration")
    else:
        source_sha = manifest["input_sha256"]
    if source_sha != actual["trace.json"]:
        raise AssertionError("Runtime consumed a different trace")
    return actual


def verify_cold_windows(root, events, origin):
    def lines(path):
        return [json.loads(line) for line in path.read_text().splitlines()]

    gates = lines(root / "client-send.jsonl")
    raw = lines(root / "client/profile_export_raw.jsonl")
    starts = {row["metadata"]["x_request_id"]: row["start_perf_ns"] for row in raw}
    reported = {
        row["metadata"]["x_request_id"]: row["metrics"]["time_to_first_token"]
        for row in lines(root / "client/profile_export.jsonl")
    }
    if (
        len(starts) != len(raw)
        or len(gates) != len(raw)
        or {gate["request_id"] for gate in gates} != set(starts)
    ):
        raise AssertionError("Incomplete or duplicate client timing evidence")
    for row in raw:
        # This server starts each SSE stream with the first generated content token.
        first = row["responses"][0]
        packet = json.loads(first["packets"][0]["value"])
        if not packet["choices"][0]["delta"]["content"]:
            raise AssertionError("Expected first SSE packet to contain a token")
        metric = reported[row["metadata"]["x_request_id"]]
        if (
            metric["unit"] != "ms"
            or round(metric["value"] * 1e6) != first["perf_ns"] - row["start_perf_ns"]
        ):
            raise AssertionError("Reported client TTFT differs from raw transport timestamps")
    windows = {}
    previous_finish = None
    finishes = {e["request_id"]: e for e in events if e["event"] == "http_response_finished"}
    phases = [
        "disk_replicas_confirmed",
        "memory_replicas_evicted",
        "local_prefix_cache_reset",
        "guest_page_cache_dropped",
    ]
    for index, gate in enumerate(gates):
        cid = gate["request_id"]
        own = [e for e in events if e.get("client_request_id") == cid]

        def single(name, own=own, cid=cid):
            matches = [e for e in own if e["event"] == name]
            if len(matches) != 1:
                raise AssertionError(f"Expected one {name} for {cid}")
            return matches[0]

        arrival = single("http_arrival")
        dispatch = gate["before_native_send_ns"]
        if not dispatch <= starts[cid] <= origin + arrival["time_ns"]:
            raise AssertionError("Client send clock does not precede server arrival")
        finish = finishes.get(single("http_received")["request_id"])
        if finish is None:
            raise AssertionError("Missing HTTP response completion")
        if index == 0:
            if any(e["event"].startswith("cold_prepare_") for e in own):
                raise AssertionError("Unexpected cold preparation before first request")
            previous_finish = finish
            continue
        begin, end = single("cold_prepare_begin"), single("cold_prepare_end")
        if (
            not previous_finish["time_ns"] <= begin["time_ns"] < end["time_ns"]
            or origin + end["time_ns"] > dispatch
        ):
            raise AssertionError("Cold preparation overlaps the measured request")
        previous_finish = finish
        span = [
            item["event"]
            for item in events
            if begin["seq"] < item["seq"] < end["seq"] and item["event"] in phases
        ]
        if span != phases:
            raise AssertionError("Incomplete cold preparation")
        windows[cid] = {
            "cold_prepare_ms": (end["time_ns"] - begin["time_ns"]) / 1e6,
            "cold_end_to_client_start_ms": (starts[cid] - origin - end["time_ns"]) / 1e6,
            "client_start_to_http_arrival_ms": (origin + arrival["time_ns"] - starts[cid]) / 1e6,
        }
    return windows


def verify(path):
    launch = json.loads((path / "launch.json").read_text())
    run = run_path(path)
    joint = launch.get("workload") == "aiperf"
    if joint:
        from kv_ssd_sim.analysis.aiperf import verify as verify_http

        if "guest_inputs_sha256" in launch:
            http_evidence = verify_http(path / "joint", staged_trace=path / "input/trace.json")
        else:
            http_evidence = verify_http(path / "joint")
        if not http_evidence["femu_verified"]:
            raise AssertionError("HTTP workload did not verify FEMU")
    summary = json.loads((run / "summary.json").read_text())
    manifest = json.loads((run / "run_manifest.json").read_text())
    events = [json.loads(line) for line in (run / "events.jsonl").read_text().splitlines()]
    if not summary["femu_verified"] or manifest["read_delay_ms"] != 0:
        raise AssertionError("Trial did not verify FEMU without injected receiver delay")
    origin = next(e["monotonic_origin_ns"] for e in events if e["event"] == "clock_sync")
    staged = (
        "guest_inputs_sha256" in launch
        or (path / "guest-input.json").exists()
        or (path / "input").exists()
    )
    if staged and (path / "guest-exit-code.txt").read_text().strip() != "0":
        raise AssertionError("Guest workload did not finish successfully")
    inputs = verify_inputs(path, launch, manifest) if staged else None
    if joint and inputs and not manifest["cold_between_requests"]:
        from kv_ssd_sim.analysis.natural import describe

        result = {
            "status": "passed",
            "cache_mode": "natural",
            "accelerator": launch["accelerator"],
            "guest_inputs_sha256": inputs,
            "nand_read_ns": launch["nand_read_ns"],
            "aiperf": http_evidence,
            "natural": describe(
                path, events, manifest, summary, http_evidence.get("wire_shared_block_tokens")
            ),
        }
        (path / "verification.json").write_text(json.dumps(result, indent=2) + "\n")
        return result
    windows = verify_cold_windows(path / "joint", events, origin) if joint and inputs else {}
    completions = []
    pending = defaultdict(deque)
    trace = (path / "block-trace.txt").read_text()
    counts = re.search(r"entries-in-buffer/entries-written:\s*(\d+)/(\d+)", trace)
    if counts is None or counts[1] != counts[2]:
        raise AssertionError("Block trace is incomplete or overflowed")
    devices = set()
    for line in trace.splitlines():
        match = re.search(r"(\d+\.\d+): block_rq_(issue|complete):\s+(\d+,\d+)\s+(\S+)", line)
        sectors = re.search(r"(\d+) \+ (\d+)", line)
        if match:
            devices.add(match[3])
        if match and sectors and "R" in match[4]:
            key = (match[3], int(sectors[1]), int(sectors[2]))
            timestamp = round(float(match[1]) * 1e9) - origin
            if match[2] == "issue":
                pending[key].append(timestamp)
                continue
            issued = pending[key].popleft() if pending[key] else None
            completions.append(
                {
                    "time_ns": timestamp,
                    "device": match[3],
                    "sector": key[1],
                    "sectors": key[2],
                    "block_latency_ms": (timestamp - issued) / 1e6 if issued is not None else None,
                    "line": line.strip(),
                }
            )
    # The launcher exposes one RAM-backed FEMU and no other block devices.
    if len(devices) != 1:
        raise AssertionError("Expected exactly one traced block device")
    restored = [r for r in summary["request_metrics"] if r["external_load_tokens"] > 0]
    if not restored or summary["checksum_errors"]:
        raise AssertionError("No checksum-verified external prefix restore")
    chains = []
    for row in restored:
        rid = row["request_id"]
        own = [e for e in events if e.get("request_id") == rid]
        begin = next(e for e in own if e["event"] == "native_get_begin")
        end = next(e for e in own if e["event"] == "store_get_end")
        load = next(e for e in own if e["event"] == "load_complete")
        compute = next(
            e for e in events if e["event"] == "compute_begin" and rid in e["request_ids"]
        )
        token = next(e for e in own if e["event"] == "first_token")
        received = next((e for e in own if e["event"] == "http_received"), None)
        admitted = next((e for e in own if e["event"] == "request_admitted"), None)
        emitted = next((e for e in own if e["event"] == "http_token_emitted"), None)
        http_row = None
        if joint:
            if not received or not emitted or token["seq"] >= emitted["seq"]:
                raise AssertionError("Missing scheduler -> HTTP first token ordering")
            if not admitted or not received["seq"] < admitted["seq"] < begin["seq"]:
                raise AssertionError("Missing HTTP -> admission -> GET ordering")
            http_row = next(
                json.loads(line)
                for line in (path / "joint/client/profile_export.jsonl").read_text().splitlines()
                if json.loads(line)["metadata"]["x_request_id"] == received["client_request_id"]
            )
        reads = [c for c in completions if begin["time_ns"] <= c["time_ns"] <= end["time_ns"]]
        if not reads or not (
            begin["seq"] < end["seq"] < load["seq"] < compute["seq"] < token["seq"]
        ):
            raise AssertionError(f"Missing ordered causal chain for {rid}")
        if not all(end["checksum_ok"]):
            raise AssertionError("Restore checksum mismatch")
        latencies = [read["block_latency_ms"] for read in reads]
        if any(value is None or value < 0 for value in latencies):
            raise AssertionError("Read completion has no matching issue event")
        chains.append(
            {
                "request_id": rid,
                **(windows[received["client_request_id"]] if joint and inputs else {}),
                "external_tokens": row["external_load_tokens"],
                "get_ms": (end["time_ns"] - begin["time_ns"]) / 1e6,
                "ttft_ms": row["ttft_ms"],
                "compute_budget_ms": row["compute_budget_ms"],
                "compute_window_ms": row["compute_window_ms"],
                "aiperf_ttft_ms": http_row["metrics"]["time_to_first_token"]["value"]
                if http_row
                else None,
                "http_received_to_first_emit_ms": (
                    (emitted["time_ns"] - received["time_ns"]) / 1e6 if joint else None
                ),
                "http_first_emit_seq": emitted["seq"] if emitted else None,
                "prepared_to_admission_ms": (
                    (admitted["time_ns"] - received["time_ns"]) / 1e6 if joint else None
                ),
                "admission_to_http_first_ms": (
                    (emitted["time_ns"] - admitted["time_ns"]) / 1e6 if joint else None
                ),
                "payload_preparation_ms": received.get("preparation_ms") if received else None,
                "source_turn": (
                    [
                        http_row["metadata"]["source_outer_idx"],
                        http_row["metadata"].get("source_inner_idx"),
                    ]
                    if http_row
                    else None
                ),
                "femu_read_completions_during_get": len(reads),
                "femu_read_bytes_during_get": sum(read["sectors"] * 512 for read in reads),
                "block_read_latency_ms": {
                    "min": min(latencies),
                    "mean": sum(latencies) / len(latencies),
                    "max": max(latencies),
                },
                "first_read_completion": reads[0],
                "last_read_completion": reads[-1],
                "ordered_event_sequences": [
                    begin["seq"],
                    end["seq"],
                    load["seq"],
                    compute["seq"],
                    token["seq"],
                ],
            }
        )
    result = {
        "status": "passed",
        "accelerator": launch["accelerator"],
        "guest_inputs_sha256": inputs,
        "nand_read_ns": launch["nand_read_ns"],
        "chains": chains,
        "aiperf": http_evidence if joint else None,
        "scope": "functional closed loop; TCG is not hardware-calibrated inference timing",
    }
    if manifest.get("pressure_config"):
        from kv_ssd_sim.analysis.pressure import check_pressure

        result["pressure"] = check_pressure(path, events, summary, manifest, chains)
    (path / "verification.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trials", type=Path, nargs="+")
    args = parser.parse_args()
    print(json.dumps([verify(path) for path in args.trials], indent=2))


if __name__ == "__main__":
    main()
