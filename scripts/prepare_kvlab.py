"""Audit the complete pinned replay-v5 file, or compile two source-selected windows."""

import argparse
import json
import sqlite3
from pathlib import Path
from tempfile import TemporaryDirectory

from kv_ssd_sim.kvlab import LIMITS, audit, compile_window, dump, select_windows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    scan = commands.add_parser("audit")
    pick = commands.add_parser("windows")
    for sub in (scan, pick):
        sub.add_argument("source", type=Path)
        sub.add_argument("--output", type=Path, required=True)
    pick.add_argument("--audit", type=Path, required=True)
    pick.add_argument("--limit", type=int, choices=LIMITS, default=LIMITS[0])
    pick.add_argument("--max-span-s", type=int, default=300)
    pick.add_argument("--max-requests", type=int, default=24)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Choose a new output directory")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=args.output.parent) as tmp:
        out = Path(tmp) / "results"
        out.mkdir()
        if args.command == "audit":
            audit(args.source, out)
        else:
            if min(args.max_span_s, args.max_requests) < 1:
                parser.error("Window budgets must be positive")
            report = json.loads((args.audit / "audit.json").read_bytes())
            if str(args.limit) not in report["limits"]:
                parser.error(
                    f"Index lacks coverage for {args.limit} tokens. Run audit in a new directory."
                )
            if args.source.stat().st_size != report["source_size"]:
                raise ValueError("Input length differs from the audited source")
            with sqlite3.connect(
                (args.audit / "index.sqlite").resolve().as_uri() + "?mode=ro", uri=True
            ) as db:
                windows, count = select_windows(db, args.limit, args.max_span_s, args.max_requests)
                for label, choice in zip(("calibration", "holdout"), windows):
                    compile_window(
                        args.source, db, choice, out / label, args.limit, report["source_sha256"]
                    )
            dump(
                out / "selection.json",
                {
                    "limit": args.limit,
                    "max_span_s": args.max_span_s,
                    "max_requests": args.max_requests,
                    "candidates_by_quality": {str(k): v for k, v in count.items()},
                    "quality_order": [
                        "return_after_other_session",
                        "multiple_sessions",
                        "overlapping_api",
                        "same_instance_return",
                    ],
                    "windows": windows,
                    "source_sha256": report["source_sha256"],
                },
            )
        out.rename(args.output)


if __name__ == "__main__":
    main()
