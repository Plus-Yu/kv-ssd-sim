from __future__ import annotations

import argparse
import json
import sys
from importlib import import_module
from pathlib import Path

from .core.paths import PROJECT_ROOT
from .online.runtime import doctor, reexec_with_native_libraries
from .workload.weka import load_weka, save_normalized

GROUPS = {
    "workload": (
        "Compile and audit workload inputs",
        {
            "audit": ("workload.weka_audit", "Audit a WEKA input and its selected prefix"),
            "compile-weka": ("workload.weka_compiler", "Compile causal WEKA arrivals with AIPerf"),
            "kvlab": ("workload.prepare", "Audit or compile kvlab windows"),
        },
    ),
    "online": (
        "Run native vLLM and Mooncake on the host clock",
        {
            "replay": (None, "Run a native replay"),
            "serve": (None, "Start the native HTTP service"),
        },
    ),
    "femu": (
        "Run device experiments",
        {
            "run": ("devices.femu.launcher", "Boot a disposable FEMU guest"),
        },
    ),
    "analyze": (
        "Recompute experiment ledgers",
        {
            "traffic": (
                "analysis.traffic",
                "Rebuild SSD traffic and byte reuse from object events",
            ),
            "nand": ("analysis.nand", "Reconcile FEMU snapshots and block events"),
        },
    ),
    "verify": (
        "Verify experiment evidence",
        {
            "restore": ("analysis.restore", "Audit object restores through H2D completion"),
            "femu": ("analysis.femu", "Verify the native and guest evidence of a FEMU run"),
        },
    ),
}


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "simulate":
        from .experiments.runner import main as simulate

        return simulate(argv[1:])
    if argv and argv[0] in GROUPS:
        group = argv[0]
        description, actions = GROUPS[group]
        sub = argparse.ArgumentParser(prog=f"kv-ssd-sim {group}", description=description)
        choices = sub.add_subparsers(dest="action", required=True)
        for name, (_, help_text) in actions.items():
            choices.add_parser(name, help=help_text)
        action = sub.parse_args(argv[1:2]).action
        target, _ = actions[action]
        if group == "online":
            return main([action, *argv[2:]])
        return import_module("." + target, __package__).main(argv[2:])

    parser = argparse.ArgumentParser(description="CPU-only native KV cache emulation prototype")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("simulate", help="Run object-level replay on the virtual clock")
    for name, (description, _) in GROUPS.items():
        commands.add_parser(name, help=description)
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
        default=PROJECT_ROOT / "configs/sim-tokenizer",
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
    reexec_with_native_libraries(argv)
    if args.command == "doctor":
        report = doctor()
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0 if all(item["ok"] for item in report["imports"].values()) else 1
    if args.command == "serve":
        from .online.server import serve

        serve(args)
        return 0
    if args.command == "smoke-store":
        from .online.smoke import smoke_store

        print(json.dumps(smoke_store(args.output), indent=2))
        return 0
    if args.command == "replay":
        from .online.engine import run_replay

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
        from .online.experiment import run_demo

        report = run_demo(args.output, args.delays_ms)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0
