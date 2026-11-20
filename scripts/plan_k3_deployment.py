#!/usr/bin/env python3
"""Recompute the step-2 deployment metadata or compare it with a saved baseline."""

import argparse
import json
import os
import sys
from pathlib import Path

from kv_ssd_sim.deployment import convert
from kv_ssd_sim.runtime import PROJECT_ROOT, native_environment


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config-dir", type=Path, default=PROJECT_ROOT / "configs/kimi-k3-b300-step2"
    )
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--output", type=Path, help="Write a new JSON plan (never overwrite)")
    action.add_argument(
        "--check", type=Path, help="Compare a saved plan against a fresh conversion"
    )
    args = parser.parse_args()
    if not os.environ.get("KV_SSD_PLANNER_ENV"):
        env = native_environment()
        env["KV_SSD_PLANNER_ENV"] = "1"
        os.execve(sys.executable, [sys.executable, __file__, *sys.argv[1:]], env)
    plan = convert(args.config_dir)
    data = json.dumps(plan, indent=2, ensure_ascii=False) + "\n"
    if args.check:
        if args.check.read_bytes() != data.encode():
            parser.exit(1, f"Mismatch: {args.check}\n")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as handle:
            handle.write(data)
    print(
        json.dumps(
            {
                "status": "passed",
                "layers": len(plan["layers"]),
                "groups": len(plan["groups"]),
                "ranks": len(plan["ranks"]),
                "pool_blocks_per_rank": plan["ranks"][0]["native_pool_blocks"],
            }
        )
    )


if __name__ == "__main__":
    main()
