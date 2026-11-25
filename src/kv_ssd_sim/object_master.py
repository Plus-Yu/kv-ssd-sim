"""Pipe protocol for the isolated, virtual-time Mooncake master."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import tempfile
from contextlib import ExitStack
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REVISION = "e1ea5d31bb5e46ff2df9f90e7c2fbaf52fcdee8c"
STORE_SHA = "19573f3ed46c577e52e62851667266bd3cc8d40781300ae22ce8a73a624814a8"


def checksum(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def build_info():
    folder = Path(os.environ.get("KV_SSD_OBJECT_MASTER_DIR", ROOT / ".object-replay-build"))
    binary = folder / "master"
    if not (folder / "build.json").is_file():
        raise RuntimeError(
            "Build the pinned native master first: python scripts/build_object_master.py"
        )
    info = json.loads((folder / "build.json").read_bytes())
    if (
        info["revision"] != REVISION
        or info["wheel_dso_sha256"] != STORE_SHA
        or checksum(Path(info["wheel_dso"])) != STORE_SHA
        or checksum(Path(info["wheel_dso"]).with_name("mooncake_master"))
        != info["online_master_sha256"]
        or checksum(binary) != info["executable_sha256"]
        or checksum(ROOT / "native/object_replay_master.cpp") != info["driver_source_sha256"]
        or checksum(ROOT / "scripts/build_object_master.py") != info["builder_sha256"]
    ):
        raise RuntimeError(
            "Native master build differs from the pinned sources or wheel. Rebuild it."
        )
    return binary, info


def wire_key(key):
    return key.encode().hex()


def read_key(field):
    return bytes.fromhex(field).decode()


class NativeMaster:
    def __init__(self, config, event, logfile=None):
        binary, self.provenance = build_info()
        self.event = event
        self.resources = ExitStack()
        if logfile:
            self.log = self.resources.enter_context(logfile.open("x+"))
        else:
            self.log = self.resources.enter_context(tempfile.TemporaryFile(mode="w+"))  # noqa: SIM115
        try:
            self.process = subprocess.Popen(
                [
                    str(binary),
                    str(config.clients),
                    str(config.segment_bytes),
                    str(config.quota_bytes),
                    str(int(config.lease_ms)),
                    str(config.high_watermark),
                    str(config.eviction_ratio),
                ],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=self.log,
                text=True,
                bufsize=1,
            )
            if self.process.stdout.readline().strip() != "ready 1":
                raise self.failure("startup")
        except BaseException:
            self.close()
            raise

    def failure(self, operation):
        self.log.seek(0, os.SEEK_END)
        self.log.seek(max(0, self.log.tell() - 2048))
        return RuntimeError(f"Native master failed during {operation}: {self.log.read()}")

    def call(self, op, at, *args):
        p = self.process
        try:
            p.stdin.write(" ".join(map(str, (op, at, *args))) + "\n")
            p.stdin.flush()
        except BrokenPipeError:
            raise self.failure(op) from None
        for line in p.stdout:
            tag, _, payload = line.rstrip("\n").partition(" ")
            if tag == "ok":
                return payload.split()
            if tag not in ("evicted", "round"):
                raise RuntimeError(f"Unexpected native master response: {line}")
            self.event(tag, *payload.split())
        raise self.failure(op)

    def peak_rss_kib(self):
        status = Path(f"/proc/{self.process.pid}/status").read_text()
        return int(
            next(line.split()[1] for line in status.splitlines() if line.startswith("VmHWM:"))
        )

    def close(self):
        p = getattr(self, "process", None)
        if p is not None and p.stdin:
            try:
                p.stdin.close()  # EOF cleanly shuts down the dedicated service.
                p.wait(timeout=10)
            except (BrokenPipeError, subprocess.TimeoutExpired):
                p.kill()
                p.wait()
            finally:
                p.stdout.close()
                self.process = None
        self.resources.close()
