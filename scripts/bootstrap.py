#!/usr/bin/env python3
"""Recreate the CPU environment without modifying system Python or libraries.

Run with Python 3.13. Existing local wheels are hash-checked and reused.
Ubuntu 26.04 native libraries are downloaded through apt, then extracted only
under .native. No sudo, apt install, model weights, or CUDA packages are used.
"""

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import venv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run(command, **kwargs):
    subprocess.run([str(x) for x in command], check=True, **kwargs)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index-url", default="https://pypi.org/simple")
    parser.add_argument(
        "--skip-native-libs",
        action="store_true",
        help="Use existing system or project-local native libraries",
    )
    args = parser.parse_args()
    if sys.version_info[:2] != (3, 13) or platform.machine() != "x86_64":
        raise SystemExit("Run this bootstrap using Python 3.13 on Linux x86_64")
    if not sys.platform.startswith("linux") or tuple(
        map(int, platform.libc_ver()[1].split("."))
    ) < (2, 39):
        raise SystemExit("Pinned vLLM CPU wheel requires Linux glibc >= 2.39")
    target = ROOT / ".venv"
    if not (target / "bin/python").exists():
        venv.EnvBuilder(with_pip=True).create(target)
    python = target / "bin/python"
    run([python, "-m", "pip", "install", "--index-url", args.index_url, "uv==0.13.8"])
    native = ROOT / ".native"
    native.mkdir(exist_ok=True)
    manifest = json.loads((ROOT / "environment-manifest.json").read_text())
    overrides = []
    for wheel in manifest["wheels"]:
        path = native / "wheels" / wheel["file"]
        if path.exists():
            with path.open("rb") as stream:
                actual = hashlib.file_digest(stream, "sha256").hexdigest()
            if actual != wheel["sha256"]:
                raise SystemExit(f"Wheel hash mismatch: {path}")
            overrides.append(f"{wheel['name']} @ {path.as_uri()}")
    override_file = native / "bootstrap-overrides.txt"
    override_file.write_text("\n".join(overrides) + "\n")
    env = os.environ.copy()
    env.setdefault("UV_HTTP_TIMEOUT", "600")
    run(
        [
            target / "bin/uv",
            "pip",
            "install",
            "--python",
            python,
            "--index-url",
            args.index_url,
            "--torch-backend",
            "cpu",
            "--override",
            override_file,
            "-r",
            ROOT / "requirements.lock.txt",
            "-e",
            ROOT,
        ],
        env=env,
    )
    if not args.skip_native_libs:
        release = platform.freedesktop_os_release()
        if release.get("ID") != "ubuntu" or release.get("VERSION_ID") != "26.04":
            raise SystemExit(
                "Native deb versions target Ubuntu 26.04; supply suitable libraries "
                "and rerun with --skip-native-libs on other distributions"
            )
        debs = native / "debs"
        debs.mkdir(exist_ok=True)
        packages = [
            "libibverbs1=61.0-2ubuntu3",
            "ibverbs-providers=61.0-2ubuntu3",
            "libnuma1=2.0.19-1build1",
            "libnl-3-200=3.12.0-2",
            "libnl-route-3-200=3.12.0-2",
        ]
        if any(not (debs / d["file"]).exists() for d in manifest["debs"]):
            run(["apt-get", "download", *packages], cwd=debs)
        for item in manifest["debs"]:
            path = debs / item["file"]
            with path.open("rb") as stream:
                if hashlib.file_digest(stream, "sha256").hexdigest() != item["sha256"]:
                    raise SystemExit(f"Native library archive hash mismatch: {path}")
            run(["dpkg-deb", "-x", path, native])
    run([target / "bin/uv", "pip", "check", "--python", python])
    run([python, "-m", "kv_ssd_sim", "doctor"])


if __name__ == "__main__":
    main()
