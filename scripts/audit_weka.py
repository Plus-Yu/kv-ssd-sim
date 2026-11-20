#!/usr/bin/env python3
"""Audit a pinned WEKA file and select a dependency-complete serial prefix."""

import argparse
import json
from pathlib import Path

from kv_ssd_sim.audit import write_audit_bundle

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--source-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--select-first", type=int, default=2)
    parser.add_argument("--context-limit", type=int, default=131072)
    parser.add_argument("--match-unit", type=int, default=128)
    parser.add_argument("--model-layers", type=int, default=93)
    args = parser.parse_args()
    result = write_audit_bundle(
        args.input,
        args.output,
        expected_sha256=args.source_sha256,
        count=args.select_first,
        context_limit=args.context_limit,
        match_unit=args.match_unit,
        model_layers=args.model_layers,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
