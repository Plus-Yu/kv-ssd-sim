"""Audit the pinned replay-v5 file or compile source-selected cold and warm windows."""

import argparse
import json
import sqlite3
from pathlib import Path
from tempfile import TemporaryDirectory

from .kvlab import LIMITS, audit, compile_window, dump, select_warm_window, select_windows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    scan = commands.add_parser("audit")
    pick = commands.add_parser("windows")
    warm = commands.add_parser("warm-window", help="Compile one continuous window with warmup")
    for sub in (scan, pick, warm):
        sub.add_argument("source", type=Path)
        sub.add_argument("--output", type=Path, required=True)
    for sub in (pick, warm):
        sub.add_argument("--audit", type=Path, required=True)
        sub.add_argument("--limit", type=int, choices=LIMITS)
        sub.add_argument("--max-span-s", type=int)
        sub.add_argument("--max-requests", type=int)
    pick.set_defaults(limit=LIMITS[0], max_span_s=300, max_requests=24)
    warm.set_defaults(limit=LIMITS[-1], max_span_s=40 * 86400)
    warm.add_argument("--warmup-requests", type=int, default=32)
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error("Choose a new output directory")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=args.output.parent) as tmp:
        out = Path(tmp) / "results"
        out.mkdir()
        if args.command == "audit":
            audit(args.source, out)
        else:
            report = json.loads((args.audit / "audit.json").read_bytes())
            if args.max_requests is None:
                args.max_requests = report["requests"]
            if min(args.max_span_s, args.max_requests) < 1:
                parser.error("Window budgets must be positive")
            if str(args.limit) not in report["limits"]:
                parser.error(
                    f"Index lacks coverage for {args.limit} tokens. Run audit in a new directory."
                )
            if args.source.stat().st_size != report["source_size"]:
                raise ValueError("Input length differs from the audited source")
            with sqlite3.connect(
                (args.audit / "index.sqlite").resolve().as_uri() + "?mode=ro", uri=True
            ) as db:
                if args.command == "warm-window":
                    choice = select_warm_window(
                        db, args.limit, args.max_span_s, args.max_requests, args.warmup_requests
                    )
                    windows, count, labels = [choice], None, ("warm",)
                else:
                    windows, count = select_windows(
                        db, args.limit, args.max_span_s, args.max_requests
                    )
                    labels = ("calibration", "holdout")
                for label, choice in zip(labels, windows):
                    compile_window(
                        args.source, db, choice, out / label, args.limit, report["source_sha256"]
                    )
            dump(
                out / "selection.json",
                {
                    "limit": args.limit,
                    "max_span_s": args.max_span_s,
                    "max_requests": args.max_requests,
                    **(
                        {"candidates_by_quality": {str(k): v for k, v in count.items()}}
                        if count is not None
                        else {}
                    ),
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
