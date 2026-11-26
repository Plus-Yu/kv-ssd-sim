"""Compare volume service scenarios and the single-drive control on identical compiled inputs."""

import argparse
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

SOURCE = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SOURCE))

HISTORICAL_SCENARIOS = {
    "base": [],
    "compute-fast": ["--prefill-ms", "5", "--decode-ms", "1"],
    "compute-slow": ["--prefill-ms", "20", "--decode-ms", "4"],
    "storage-fast": ["--save-ms", "0.5", "--device-mib-s", "2048", "--memory-mib-s", "16384"],
    "storage-slow": ["--save-ms", "2", "--device-mib-s", "512", "--memory-mib-s", "4096"],
}

PUBLIC_SCENARIOS = {
    "base": [],
    "compute-fast": ["--compute-scale", "0.5"],
    "compute-slow": ["--compute-scale", "2"],
    "ssd-half-bandwidth": ["--ssd-time-scale", "2"],
    "ssd-quarter-bandwidth": ["--ssd-time-scale", "4"],
}

DEPLOYMENT_SCENARIOS = {
    "base": [],
    "legacy-clock": ["--no-save-decode-cache"],
    "ssd-4x": ["--volume-service", "4x"],
    "ssd-8x": ["--volume-service", "8x"],
    "host-4-lanes": ["--host-copy-lanes", "4"],
    "host-half-rate": ["--host-copy-scale", "0.5"],
    "save-decode": ["--save-decode-cache"],
    "seq-16": ["--max-num-seqs", "16"],
    "seq-32": ["--max-num-seqs", "32"],
    "tokens-2048": ["--max-num-batched-tokens", "2048"],
    "tokens-8192": ["--max-num-batched-tokens", "8192"],
    "context-cost-2x": ["--context-cost-scale", "2"],
}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("windows", type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--window", choices=("calibration", "holdout", "warm"), action="append")
    p.add_argument("--case", action="append", help="Select individual cases of this matrix")
    p.add_argument("--jobs", type=int, default=1, help="Run this many independent cases at once")
    p.add_argument(
        "--profile",
        choices=(
            "reference",
            "reference-context",
            "reference-volume-20261125",
            "reference-single-ssd",
            "reference-20261124",
        ),
        default="reference",
    )
    args = p.parse_args()
    if args.profile == "reference-context":
        scenarios = DEPLOYMENT_SCENARIOS
    elif args.profile in ("reference", "reference-volume-20261125"):
        from kv_ssd_sim.experiments.profiles import PROFILE_FILES, load_profile

        profile, _ = load_profile(PROFILE_FILES[args.profile])
        scenarios = {
            f"volume-{factor}": ["--volume-service", factor]
            for factor in profile["volume"]["service_rate_multipliers"]
        }
        scenarios["single-3tb"] = ["--max-model-len", str(profile["max_model_len"])]
        if args.profile == "reference":
            scenarios["single-3tb"].extend(["--restore-path", "host-staged"])
    else:
        scenarios = (
            PUBLIC_SCENARIOS if args.profile == "reference-single-ssd" else HISTORICAL_SCENARIOS
        )
    if args.jobs < 1 or (args.case and not set(args.case) <= scenarios.keys()):
        p.error(f"Use positive --jobs and select cases from {', '.join(scenarios)}")
    selected = args.case or list(scenarios)
    default_windows = (
        ("warm",) if (args.windows / "warm/input.json").is_file() else ("calibration", "holdout")
    )
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(filter(None, (str(SOURCE), env.get("PYTHONPATH"))))
    for name in args.window or default_windows:
        folder = args.output / name
        folder.mkdir(parents=True, exist_ok=False)

        def run_case(case, folder=folder, name=name):
            with (folder / (case + ".log")).open("x") as log:
                subprocess.run(
                    [
                        sys.executable,
                        "-m",
                        "kv_ssd_sim",
                        "simulate",
                        str(args.windows / name / "input.json"),
                        "--profile",
                        {"single-3tb": "reference-single-ssd", "legacy-clock": "reference"}.get(
                            case, args.profile
                        ),
                        "--output",
                        str(folder / case),
                        *scenarios[case],
                    ],
                    env=env,
                    check=True,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                )

        with ThreadPoolExecutor(max_workers=args.jobs) as pool:
            list(pool.map(run_case, dict.fromkeys(selected)))


if __name__ == "__main__":
    main()
