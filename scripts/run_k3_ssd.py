"""Run the C1 online acceptance or three identically configured C2 boots."""

import argparse
import json
import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = ROOT / ".venv/bin/python"
ANALYSIS = ROOT / ".venv-aiperf/bin/python"


def boot(trial):
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(filter(None, (str(ROOT / "src"), env.get("PYTHONPATH"))))
    subprocess.run(
        [
            PY,
            "-m",
            "kv_ssd_sim",
            "femu",
            "run",
            "--trial",
            trial,
            "--workload",
            "aiperf",
            "--natural-cache",
            "--block-capture",
            "stream",
            "--input",
            ROOT / "datasets/k3-c2/slice.json",
            "--hybrid-config",
            ROOT / "configs/k3-c1-online.json",
            "--guest-ram-mib",
            "8192",
            "--namespace-mib",
            "3072",
            "--blocks-per-plane",
            "2048",
            "--max-model-len",
            "12288",
            "--num-blocks",
            "22",
            "--timeout",
            "1800",
        ],
        check=True,
        cwd=ROOT,
        env=env,
    )
    return ROOT / "outputs" / trial


def audit(trial, output, repeats=()):
    subprocess.run(
        [
            ANALYSIS,
            ROOT / "scripts/analyze_k3_ssd.py",
            trial,
            *(["--repeats", *repeats] if repeats else []),
            "--output",
            output,
        ],
        check=True,
        cwd=ROOT,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="stage", required=True)
    first = commands.add_parser("c1")
    first.add_argument("--trial", required=True)
    second = commands.add_parser("c2")
    second.add_argument("--calibration", type=Path, required=True)
    second.add_argument("--prefix", required=True, help="New femu-* trial prefix")
    args = parser.parse_args()
    if args.stage == "c1":
        trial = boot(args.trial)
        audit(trial, trial / "analysis")
    else:
        trial = args.calibration.resolve()
        audit(trial, trial / "analysis")
        runs = []
        attempts = 0
        retried = False
        while len(runs) < 3:
            attempts += 1
            name = f"{args.prefix}-{attempts:02d}"
            try:
                run = boot(name)
            except subprocess.CalledProcessError:
                failed = ROOT / "outputs" / name
                counters = failed / "diskstats.jsonl"
                if (
                    retried
                    or not counters.is_file()
                    or not (failed / "guest-exit-code.txt").is_file()
                    or not (failed / "replay.log").is_file()
                    or "Server exited: inspect" not in (failed / "replay.log").read_text()
                    or any(
                        json.loads(line)["label"] == "http_send"
                        for line in counters.read_text().splitlines()
                    )
                ):
                    raise
                print(
                    f"{name}: server exited before the first request; retaining failed boot and retrying",
                    flush=True,
                )
                retried = True
                continue
            audit(run, run / "analysis")
            runs.append(run)
            retried = False
        audit(trial, ROOT / "outputs" / (args.prefix + "-results"), runs)


if __name__ == "__main__":
    main()
