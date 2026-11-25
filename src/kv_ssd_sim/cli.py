from __future__ import annotations

import argparse
import json
from pathlib import Path

from .runtime import doctor, reexec_with_native_libraries
from .trace import load_weka, save_normalized


def main(argv=None):
    parser = argparse.ArgumentParser(description="CPU-only native KV cache emulation prototype")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("doctor", help="Inspect native runtimes and FEMU capability")
    smoke = commands.add_parser("smoke-store", help="Verify native CPU Mooncake disk restore")
    smoke.add_argument("--output", type=Path, required=True)
    normalize = commands.add_parser("normalize", help="Compile WEKA JSON/JSONL to token-ID records")
    normalize.add_argument("input", type=Path)
    normalize.add_argument("--output", type=Path, required=True)
    replay = commands.add_parser("replay", help="Run the native vLLM/Mooncake CPU harness")
    replay.add_argument("input", type=Path)
    replay.add_argument("--output", type=Path, required=True)
    replay.add_argument("--read-delay-ms", type=float, default=0)
    replay.add_argument("--prefill-ms", type=float, default=10)
    replay.add_argument("--decode-ms", type=float, default=2)
    replay.add_argument("--timing", choices=("causal", "fixed"), default="causal")
    replay.add_argument("--cold-between-turns", action="store_true")
    replay.add_argument("--max-model-len", type=int, default=8192)
    replay.add_argument("--num-blocks", type=int, default=128)
    replay.add_argument("--hybrid-config", type=Path)
    replay.add_argument("--pressure-config", type=Path)
    replay.add_argument("--femu-mount", type=Path)
    serve = commands.add_parser(
        "serve", help="Serve synthetic streaming chat with native scheduling"
    )
    serve.add_argument("--output", type=Path, required=True)
    serve.add_argument("--port", type=int, default=8000)
    serve.add_argument("--model", default="kv-ssd-sim")
    serve.add_argument(
        "--tokenizer",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "configs/sim-tokenizer",
    )
    serve.add_argument("--prefill-ms", type=float, default=10)
    serve.add_argument("--decode-ms", type=float, default=2)
    serve.add_argument("--read-delay-ms", type=float, default=0)
    serve.add_argument("--max-model-len", type=int, default=8192)
    serve.add_argument("--num-blocks", type=int, default=128)
    serve.add_argument("--master-lease-ms", type=int, default=0)
    serve.add_argument("--max-inflight", type=int, default=32)
    serve.add_argument("--hybrid-config", type=Path)
    serve.add_argument("--femu-mount", type=Path)
    serve.add_argument(
        "--cold-between-requests",
        action="store_true",
        help="Serialize and evict memory for explicit disk-restore validation only",
    )
    demo = commands.add_parser("demo", help="Run the cold-prefix latency intervention experiment")
    demo.add_argument("--output", type=Path, required=True)
    demo.add_argument("--delays-ms", type=float, nargs="+", default=[0, 25, 100])
    args = parser.parse_args(argv)
    if args.command == "normalize":
        requests = load_weka(args.input)
        save_normalized(requests, args.output)
        print(json.dumps({"requests": len(requests), "output": str(args.output)}, indent=2))
        return 0
    reexec_with_native_libraries()
    if args.command == "doctor":
        report = doctor()
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0 if all(item["ok"] for item in report["imports"].values()) else 1
    if args.command == "serve":
        from .server import serve

        serve(args)
        return 0
    if args.command == "smoke-store":
        from .smoke import smoke_store

        print(json.dumps(smoke_store(args.output), indent=2))
        return 0
    if args.command == "replay":
        from .engine import run_replay

        report = run_replay(
            args.input,
            args.output,
            read_delay_ms=args.read_delay_ms,
            prefill_ms=args.prefill_ms,
            decode_ms=args.decode_ms,
            timing=args.timing,
            cold_between_turns=args.cold_between_turns,
            max_model_len=args.max_model_len,
            num_blocks=args.num_blocks,
            hybrid_config_path=args.hybrid_config,
            femu_mount=args.femu_mount,
            pressure_config=args.pressure_config,
        )
    else:
        from .experiment import run_demo

        report = run_demo(args.output, args.delays_ms)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0
