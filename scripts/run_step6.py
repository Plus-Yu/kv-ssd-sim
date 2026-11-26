#!/usr/bin/env python3
"""Run the bounded, controlled-cold WEKA experiment in six fresh sequential FEMU guests."""

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kv_ssd_sim.analysis.femu import verify

TRACE = ROOT / "datasets/kimi-k3-b300-step1/slice.json"
HYBRID = ROOT / "configs/hybrid-tiny.json"
SLICE_SHA = "40379c1607947b2d84c9de6a37d501fc8f0adafd59b14f0cc7a7d7585cfd3896"
READ_NS = {"fast": 40000, "slow": 2000000}
ORDER = ("fast01", "slow2ms01", "slow2ms02", "fast02", "fast03", "slow2ms03")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--prefix", required=True, help="New femu- trial prefix, e.g. femu-step6-repeat"
    )
    parser.add_argument(
        "--resume", action="store_true", help="Validate and reuse successfully completed trials"
    )
    args = parser.parse_args()
    if not re.fullmatch(r"femu-[a-z0-9-]+", args.prefix):
        parser.error("Expected a femu- prefixed run name")
    if hashlib.sha256(TRACE.read_bytes()).hexdigest() != SLICE_SHA:
        raise ValueError("The audited two-turn trace has changed")
    for suffix in ORDER:
        name = f"{args.prefix}-{suffix}"
        folder = ROOT / "outputs" / name
        read_ns = READ_NS["slow" if suffix.startswith("slow") else "fast"]
        if folder.exists():
            if not args.resume:
                raise FileExistsError(folder)
        else:
            start = time.time_ns()
            command = [
                sys.executable,
                "-m",
                "kv_ssd_sim",
                "femu",
                "run",
                "--trial",
                name,
                "--workload",
                "aiperf",
                "--input",
                str(TRACE),
                "--hybrid-config",
                str(HYBRID),
                "--num-blocks",
                "4096",
                "--max-model-len",
                "81920",
                "--prefill-ms",
                "10",
                "--decode-ms",
                "2",
                "--nand-read-ns",
                str(read_ns),
            ]
            # GNU time reports the maximum RSS of the launcher/children, including QEMU.
            usage = folder.with_suffix(".maxrss-kib")
            env = os.environ.copy()
            env["PYTHONPATH"] = os.pathsep.join(
                filter(None, (str(ROOT / "src"), env.get("PYTHONPATH")))
            )
            subprocess.run(
                ["/usr/bin/time", "-f", "%M", "-o", str(usage), *command], check=True, env=env
            )
            (folder / "host-usage.json").write_text(
                json.dumps(
                    {
                        "window": "launcher start to VM exit",
                        "start_unix_ns": start,
                        "end_unix_ns": time.time_ns(),
                        "max_rss_kib": int(usage.read_text().strip()),
                    },
                    indent=2,
                )
                + "\n"
            )
        result = verify(folder)
        if not (folder / "host-usage.json").exists():
            # The pilot may already have run directly via run_femu.py.
            (folder / "host-usage.json").write_text(
                json.dumps(
                    {
                        "window": "launch manifest to guest exit receipt (filesystem timestamps)",
                        "start_unix_ns": (folder / "launch.json").stat().st_mtime_ns,
                        "end_unix_ns": (folder / "guest-exit-code.txt").stat().st_mtime_ns,
                        "max_rss_kib": None,
                    },
                    indent=2,
                )
                + "\n"
            )
        config = json.loads((folder / "input/config.json").read_text())
        if (
            result["guest_inputs_sha256"]["trace.json"] != SLICE_SHA
            or result["nand_read_ns"] != read_ns
            or (folder / "input/hybrid.json").read_bytes() != HYBRID.read_bytes()
            or (
                config["num_blocks"],
                config["max_model_len"],
                config["prefill_ms"],
                config["decode_ms"],
            )
            != (4096, 81920, 10, 2)
        ):
            raise ValueError(f"Unexpected conditions in {folder}")
        print(name, result["status"], flush=True)


if __name__ == "__main__":
    main()
