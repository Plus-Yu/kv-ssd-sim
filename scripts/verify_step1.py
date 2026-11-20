#!/usr/bin/env python3
"""Verify pinned inputs and reproduce audits; optional installed AIPerf check."""

import argparse
import json
import subprocess
import tempfile
from pathlib import Path

from kv_ssd_sim.audit import sha256, write_audit_bundle


def require(condition, message):
    if not condition:
        raise ValueError(message)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--aiperf", action="store_true", help="Also regenerate offline AIPerf audit"
    )
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    manifest = json.loads((root / "references/kimi-k3-b300/artifacts.lock.json").read_text())
    for name, expected in manifest["files"].items():
        path = (root / name).resolve()
        require(path.is_relative_to(root), f"Path escapes project: {name}")
        require(sha256(path) == expected, f"SHA256 mismatch: {name}")
    source_manifest = json.loads((root / "references/kimi-k3-b300/sources.json").read_text())
    for entry in source_manifest["files"]:
        require(sha256(root / entry["file"]) == entry["sha256"], entry["file"])
    profile = json.loads((root / "configs/kimi-k3-b300-reference.json").read_text())
    model = json.loads((root / profile["model"]["config"]).read_text())["text_config"]
    attention = model["linear_attn_config"]
    layers = attention["full_attn_layers"] + attention["kda_layers"]
    require(sorted(layers) == list(range(1, 94)), "K3 layer coverage/overlap changed")
    require(len(attention["kda_layers"]) == 69, "KDA layer count changed")
    require(profile["parallelism"]["tensor_parallel_size"] == 8, "Reference TP changed")
    bundle = root / "datasets/kimi-k3-b300-step1"
    with tempfile.TemporaryDirectory(prefix="kv-step1-") as temp:
        generated = Path(temp) / "bundle"
        selection = json.loads((bundle / "selection.json").read_text())
        write_audit_bundle(
            root / "references/kimi-k3-b300/trace_0001.source.json",
            generated,
            expected_sha256=selection["source_sha256"],
            context_limit=profile["scheduler"]["max_model_len"],
            match_unit=profile["cache"]["prefix_match_unit"],
            model_layers=model["num_hidden_layers"],
        )
        for path in generated.iterdir():
            require(
                path.read_bytes() == (bundle / path.name).read_bytes(),
                f"Audit reproduction differs: {path.name}",
            )
        if args.aiperf:
            output = Path(temp) / "aiperf.json"
            subprocess.run(
                [
                    str(root / ".venv-aiperf/bin/python"),
                    str(root / "scripts/audit_aiperf_input.py"),
                    "--output",
                    str(output),
                ],
                cwd=root,
                check=True,
            )
            require(
                output.read_bytes() == (bundle / "aiperf-audit.json").read_bytes(),
                "AIPerf preprocessing changed",
            )
    print(
        json.dumps(
            {
                "status": "passed",
                "pinned_files": len(manifest["files"]),
                "source_and_slice_reproduced": True,
                "aiperf_reproduced": args.aiperf,
                "scope": "step 1 only; no K3 runtime or FEMU performance acceptance",
            }
        )
    )


if __name__ == "__main__":
    main()
