#!/usr/bin/env python3
"""Run pinned AIPerf WEKA replay against an owned native simulation HTTP server."""

import argparse
import hashlib
import json
import os
import signal
import socket
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def stop(process, *, whole_group=True, timeout=80):
    if process.poll() is None:
        if whole_group:
            os.killpg(process.pid, signal.SIGTERM)
        else:
            process.terminate()
        try:
            process.wait(timeout=600 if os.environ.get("KV_SSD_LAYOUT") else timeout)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=5)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--timing", choices=["causal", "fixed"], default="causal")
    parser.add_argument("--cold-between-requests", action="store_true")
    parser.add_argument("--prepare-before-send", action="store_true")
    parser.add_argument("--read-delay-ms", type=float, default=0)
    parser.add_argument("--prefill-ms", type=float, default=10)
    parser.add_argument("--decode-ms", type=float, default=2)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--num-blocks", type=int, default=128)
    parser.add_argument("--timeout", type=float, default=300)
    parser.add_argument("--startup-timeout", type=float, default=90)
    parser.add_argument("--femu-mount", type=Path)
    parser.add_argument("--hybrid-config", type=Path, default=ROOT / "configs/hybrid-tiny.json")
    args = parser.parse_args()
    if args.prepare_before_send and (
        not args.cold_between_requests or args.concurrency != 1 or args.timing != "causal"
    ):
        parser.error("Pre-send cold validation requires serial causal replay and cold mode")
    trace = json.loads(args.input.read_text())
    records = []
    for entry in trace["requests"]:
        records.extend(entry["requests"] if entry["type"] == "subagent" else [entry])
    if not records or args.concurrency < 1 or min(args.timeout, args.startup_timeout) <= 0:
        parser.error("Require nonempty trace, positive concurrency and timeout")
    if any(r["in"] < 1 or r["out"] < 1 for r in records):
        parser.error(
            "This harness requires positive lengths; it does not rewrite zero-output turns"
        )
    k3 = json.loads(args.hybrid_config.read_text()).get("profile") == "k3-tp8"
    manager_block = 1536 if k3 else 64
    if any(
        r["in"] + r["out"] > min(args.max_model_len, (args.num_blocks - 2) * manager_block)
        for r in records
    ):
        parser.error(
            "Trace exceeds server capacity; increase pool/context, never silently truncate"
        )
    version = subprocess.check_output(
        [
            str(ROOT / ".venv-aiperf/bin/python"),
            "-c",
            "import importlib.metadata; print(importlib.metadata.version('aiperf'))",
        ],
        text=True,
    ).strip()
    if version != "0.13.0":
        raise RuntimeError(f"Require AIPerf 0.13.0, found {version}")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    server_cmd = [
        str(ROOT / ".venv/bin/python"),
        "-m",
        "kv_ssd_sim",
        "serve",
        "--output",
        str(output / "server"),
        "--port",
        str(port),
        "--hybrid-config",
        str(args.hybrid_config.resolve()),
        "--read-delay-ms",
        str(args.read_delay_ms),
        "--prefill-ms",
        str(args.prefill_ms),
        "--decode-ms",
        str(args.decode_ms),
        "--max-model-len",
        str(args.max_model_len),
        "--num-blocks",
        str(args.num_blocks),
    ]
    if args.cold_between_requests:
        server_cmd.append("--cold-between-requests")
    if args.femu_mount:
        if args.read_delay_ms != 0:
            parser.error("FEMU requires zero injected receiver delay")
        server_cmd += ["--femu-mount", str(args.femu_mount.resolve())]
    client_cmd = [
        str(ROOT / ".venv-aiperf/bin/aiperf"),
        "profile",
        "--model",
        "kv-ssd-sim",
        "--tokenizer",
        str(ROOT / "configs/sim-tokenizer"),
        "--url",
        f"http://127.0.0.1:{port}",
        "--endpoint-type",
        "chat",
        "--apply-chat-template",
        "--use-server-token-count",
        "--streaming",
        "--input-file",
        str(args.input.resolve()),
        "--custom-dataset-type",
        "weka_trace",
        "--request-count",
        str(len(records)),
        "--concurrency",
        str(args.concurrency),
        "--max-workers",
        "1",
        "--ui-type",
        "none",
        "--no-gpu-telemetry",
        "--no-server-metrics",
        "--output-artifact-dir",
        str(output / "client"),
        "--export-level",
        "raw",
        "--no-auto-plot",
        "--random-seed",
        "42",
        "--request-timeout-seconds",
        "480" if manager_block == 1536 else "120",
        "--synthesis-max-isl",
        str(args.max_model_len),
        "--synthesis-max-osl",
        str(max(r["out"] for r in records)),
    ]
    client_cmd += (
        ["--no-fixed-schedule", "--use-think-time-only"]
        if args.timing == "causal"
        else ["--fixed-schedule"]
    )
    (output / "launch.json").write_text(
        json.dumps(
            {
                "aiperf_version": version,
                "prepare_before_send": args.prepare_before_send,
                "server_command": server_cmd,
                "client_command": client_cmd,
                "input_sha256": hashlib.sha256(args.input.read_bytes()).hexdigest(),
                "expected_requests": len(records),
                "timing": args.timing,
                "scope": "native HTTP + FEMU guest"
                if args.femu_mount
                else "native HTTP + host filesystem",
                "femu_requested": args.femu_mount is not None,
            },
            indent=2,
        )
        + "\n"
    )
    with (
        (output / "server.log").open("w") as server_log,
        (output / "client.log").open("w") as client_log,
    ):
        server = subprocess.Popen(
            server_cmd,
            cwd=ROOT,
            stdout=server_log,
            env={
                **os.environ,
                **({"VLLM_SSM_CONV_STATE_LAYOUT": "SD"} if manager_block == 1536 else {}),
                "KV_SSD_EXTERNAL_GATE": str(int(args.prepare_before_send)),
            },
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        client = None
        try:
            deadline = time.monotonic() + args.startup_timeout
            while True:
                if server.poll() is not None:
                    raise RuntimeError(f"Server exited: inspect {output / 'server.log'}")
                try:
                    with urllib.request.urlopen(
                        f"http://127.0.0.1:{port}/health", timeout=1
                    ) as response:
                        if json.load(response)["ready"]:
                            break
                except (urllib.error.URLError, TimeoutError):
                    pass
                if time.monotonic() >= deadline:
                    raise TimeoutError("Server startup timed out")
                time.sleep(0.2)
            env = os.environ.copy()
            # AIPerf 0.13's offline branch treats absolute directories as Hub
            # snapshot names. The explicit local tokenizer path needs no download.
            env.pop("HF_HUB_OFFLINE", None)
            env.pop("TRANSFORMERS_OFFLINE", None)
            env["TOKENIZERS_PARALLELISM"] = "false"
            if args.prepare_before_send:
                env["KV_SSD_GATE_URL"] = f"http://127.0.0.1:{port}/internal/prepare"
                env["KV_SSD_GATE_LOG"] = str(output / "client-send.jsonl")
            if args.prepare_before_send or env.get("KV_SSD_CAPTURE_ROOT"):
                env["PYTHONPATH"] = os.pathsep.join(
                    filter(
                        None,
                        [
                            str(ROOT / "scripts/aiperf_gate"),
                            str(ROOT / "src"),
                            env.get("PYTHONPATH"),
                        ],
                    )
                )
            client = subprocess.Popen(
                client_cmd,
                cwd=ROOT,
                env=env,
                stdout=client_log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            if client.wait(timeout=args.timeout) != 0:
                raise RuntimeError(f"AIPerf failed: inspect {output / 'client.log'}")
        finally:
            if client is not None:
                stop(client)
            # K3 can spend 360 s draining senders and 120 s confirming disk replicas.
            stop(server, whole_group=False, timeout=600 if k3 else 80)
    subprocess.run(
        [str(ROOT / ".venv/bin/python"), str(ROOT / "scripts/verify_aiperf.py"), str(output)],
        check=True,
        cwd=ROOT,
    )


if __name__ == "__main__":
    main()
