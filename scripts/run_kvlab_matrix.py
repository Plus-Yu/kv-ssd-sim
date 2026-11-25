"""Compare volume service scenarios and the single-drive control on identical compiled inputs."""

import argparse
import json
import subprocess
import sys
from pathlib import Path

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


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("windows", type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--window", choices=("calibration", "holdout", "warm"), action="append")
    p.add_argument(
        "--profile",
        choices=(
            "reference",
            "reference-volume-20261125",
            "reference-single-ssd",
            "reference-20261124",
        ),
        default="reference",
    )
    args = p.parse_args()
    if args.profile in ("reference", "reference-volume-20261125"):
        name = (
            "kimi-k3-b300-large-ssd.json"
            if args.profile == "reference"
            else "kimi-k3-b300-logical-volume.json"
        )
        settings = Path(__file__).resolve().parents[1] / "configs" / name
        profile = json.loads(settings.read_bytes())
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
    default_windows = (
        ("warm",) if (args.windows / "warm/input.json").is_file() else ("calibration", "holdout")
    )
    for name in args.window or default_windows:
        folder = args.output / name
        folder.mkdir(parents=True, exist_ok=False)
        for case, flags in scenarios.items():
            with (folder / (case + ".log")).open("x") as log:
                subprocess.run(
                    [
                        sys.executable,
                        str(Path(__file__).with_name("run_object_replay.py")),
                        str(args.windows / name / "input.json"),
                        "--profile",
                        "reference-single-ssd" if case == "single-3tb" else args.profile,
                        "--output",
                        str(folder / case),
                        *flags,
                    ],
                    check=True,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                )


if __name__ == "__main__":
    main()
