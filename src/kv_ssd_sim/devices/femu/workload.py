"""Stage and consume a bounded, content-identified FEMU input bundle."""

import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path

from kv_ssd_sim.core.paths import PROJECT_ROOT as ROOT


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def stage(directory, args):
    from kv_ssd_sim.models.hybrid import HybridConfig

    source = args.input or ROOT / "examples" / (
        "aiperf-two-requests.weka.json" if args.workload == "aiperf" else "two_requests.weka.json"
    )
    hybrid = HybridConfig.load(args.hybrid_config)
    if not (128 <= args.max_model_len <= 1048576 and 4 <= args.num_blocks <= 32768):
        raise ValueError("Invalid context or pool block count")
    if not all(math.isfinite(ms) and ms >= 0 for ms in (args.prefill_ms, args.decode_ms)):
        raise ValueError("Compute budgets must be finite and nonnegative")
    from kv_ssd_sim.online.k3_online import K3Profile

    k3 = isinstance(hybrid, K3Profile)
    pool = (
        (args.num_blocks * 21233664 * 8)
        if k3
        else (
            args.num_blocks
            * max(hybrid.layers.count(kind) for kind in ("mla", "kda"))
            * hybrid.page_bytes
        )
    )
    trace = json.loads(source.read_bytes())
    entries = trace["requests"]
    pressure_path = getattr(args, "pressure_config", None)
    inputs = {"hybrid.json": args.hybrid_config, "trace.json": source}
    if pressure_path:
        from kv_ssd_sim.online.pressure import PressureConfig
        from kv_ssd_sim.workload.weka import normalize_trace

        pressure = PressureConfig.load(pressure_path)
        if args.workload != "replay" or pressure.disk_bytes >= 384 * 1024**2:
            raise ValueError(
                "Pressure profile requires native replay and a disk budget below the guest namespace"
            )
        entries = [
            child
            for entry in entries
            for child in (entry["requests"] if entry.get("type") == "subagent" else [entry])
        ]
        inputs["pressure.json"] = pressure_path
    if not entries or any(entry.get("type") not in {"s", "n"} for entry in entries):
        raise ValueError("This cold validation profile accepts serial parent turns only")
    limit = min(args.max_model_len, (args.num_blocks - 2) * (1536 if k3 else 64))
    if any(type(r.get(k)) is not int or r[k] <= 0 for r in entries for k in ("in", "out")) or any(
        r["in"] + r["out"] > limit for r in entries
    ):
        raise ValueError("Selected trace exceeds capacity or has invalid lengths; no truncation")
    digest_rows = [r["in"] + r["out"] for r in entries]
    prepared = (
        (max(digest_rows) if args.workload == "aiperf" else sum(digest_rows))
        * 32
        * (93 if k3 else len(hybrid.layers))
    )
    if k3 and (
        args.workload != "aiperf"
        or not args.natural_cache
        or hybrid.quota_mib >= args.namespace_mib
        or pool + ((hybrid.segment_mib + 2 * hybrid.staging_mib + 1800) << 20)
        > args.guest_ram_mib << 20
    ):
        raise ValueError("K3 profile needs natural AIPerf and sufficient RAM and disk")
    if prepared > 512 * 1024**2 or (not k3 and pool > 128 * 1024**2):
        raise ValueError("Input exceeds the bounded 4 GiB guest profile")
    if pressure_path:
        requests = normalize_trace(trace)
        roles = {
            pressure.producer,
            pressure.restore,
            pressure.incomplete_producer,
            pressure.recompute,
        }
        if not roles <= {r.request_id for r in requests}:
            raise ValueError("Capacity profile roles are absent from the trace")
    natural = getattr(args, "natural_cache", False)
    if natural and args.workload != "aiperf":
        raise ValueError("--natural-cache requires the AIPerf workload")
    directory.mkdir()
    for name, path in inputs.items():
        (directory / name).write_bytes(path.read_bytes())
    config = {
        "workload": args.workload,
        "num_blocks": args.num_blocks,
        "max_model_len": args.max_model_len,
        "prefill_ms": args.prefill_ms,
        "decode_ms": args.decode_ms,
        "files": {name: digest(directory / name) for name in inputs},
    }
    if natural:
        config["cache_mode"] = "natural"
    (directory / "config.json").write_text(json.dumps(config, indent=2) + "\n")


def guest(output):
    if not Path("/run/kv-ssd-femu-guest").is_file():
        raise RuntimeError("Guest workload requires disposable FEMU guest")
    bundle = output / "input"
    cfg = json.loads((bundle / "config.json").read_text())
    launch = json.loads((output / "launch.json").read_text())
    names = ["trace.json", "hybrid.json"]
    if (bundle / "pressure.json").is_file():
        names.append("pressure.json")
    actual = {name: digest(bundle / name) for name in [*names, "config.json"]}
    if (
        actual != launch["guest_inputs_sha256"]
        or cfg["workload"] != launch["workload"]
        or cfg["files"] != {name: actual[name] for name in names}
    ):
        raise ValueError("Guest input hash/workload mismatch")
    args = ["--hybrid-config", str(bundle / "hybrid.json"), "--femu-mount", "/mnt/femu"]
    for key in ("num_blocks", "max_model_len", "prefill_ms", "decode_ms"):
        args.extend(["--" + key.replace("_", "-"), str(cfg[key])])
    if cfg["workload"] == "aiperf":
        command = [
            sys.executable,
            str(ROOT / "scripts/run_aiperf.py"),
            "--input",
            str(bundle / "trace.json"),
            "--output",
            str(output / "joint"),
            *(
                []
                if cfg.get("cache_mode", "cold") == "natural"
                else ["--cold-between-requests", "--prepare-before-send"]
            ),
            "--startup-timeout",
            "900",
            "--timeout",
            "900",
            *args,
        ]
    else:
        command = [
            sys.executable,
            "-m",
            "kv_ssd_sim",
            "replay",
            str(bundle / "trace.json"),
            "--output",
            str(output / "replay"),
            *(
                ["--pressure-config", str(bundle / "pressure.json")]
                if "pressure.json" in names
                else ["--cold-between-turns"]
            ),
            *args,
        ]
    (output / "guest-input.json").write_text(
        json.dumps({"sha256": actual, "command": command}, indent=2) + "\n"
    )
    subprocess.run(command, check=True)


def main():
    guest(Path(sys.argv[1]))


if __name__ == "__main__":
    main()
