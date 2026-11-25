"""Project-local native-library loading and owned service lifecycle."""

from __future__ import annotations

import importlib.metadata
import json
import os
import platform
import shutil
import socket
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def native_environment() -> dict[str, str]:
    env = os.environ.copy()
    libs = PROJECT_ROOT / ".native/usr/lib/x86_64-linux-gnu"
    if libs.is_dir() and env.get("KV_SSD_USE_SYSTEM_LIBRARIES") != "1":
        env["LD_LIBRARY_PATH"] = str(libs) + (
            ":" + env["LD_LIBRARY_PATH"] if env.get("LD_LIBRARY_PATH") else ""
        )
    env.setdefault("VLLM_HOST_IP", "127.0.0.1")
    env.setdefault("GLOO_SOCKET_IFNAME", "lo")
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("VLLM_NO_USAGE_STATS", "1")
    env.setdefault("HF_HUB_OFFLINE", "1")
    return env


def reexec_with_native_libraries() -> None:
    if os.environ.get("KV_SSD_NATIVE_ENV") == "1":
        return
    env = native_environment()
    env["KV_SSD_NATIVE_ENV"] = "1"
    os.execve(sys.executable, [sys.executable, "-m", "kv_ssd_sim", *sys.argv[1:]], env)


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def doctor() -> dict:
    packages = {}
    for package in ("kv-ssd-sim", "torch", "vllm", "mooncake-transfer-engine-non-cuda"):
        try:
            packages[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            packages[package] = None
    imports = {}
    for label, statement in {
        "mooncake": "from mooncake.store import MooncakeDistributedStore",
        "vllm_scheduler": "from vllm.v1.core.sched.scheduler import Scheduler",
    }.items():
        try:
            result = subprocess.run(
                [sys.executable, "-c", statement],
                env=native_environment(),
                capture_output=True,
                text=True,
                timeout=45,
                check=False,
            )
            imports[label] = {
                "ok": result.returncode == 0,
                "detail": result.stderr[-3000:] if result.returncode else "",
            }
        except subprocess.TimeoutExpired:
            imports[label] = {"ok": False, "detail": "import timed out"}
    return {
        "python": sys.version.split()[0],
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "packages": packages,
        "imports": imports,
        "kvm_available": os.access("/dev/kvm", os.R_OK | os.W_OK),
        "qemu": shutil.which("qemu-system-x86_64")
        or (
            str(PROJECT_ROOT / ".femu/build/qemu-system-x86_64")
            if (PROJECT_ROOT / ".femu/build/qemu-system-x86_64").is_file()
            else None
        ),
        "gpu_required": False,
        "storage_delay_model": "default receiver delay; --femu-mount uses actual FEMU BlackBox",
        "femu_verified": None,
        "femu_validation_scope": "doctor checks availability; it does not run the VM experiment",
        "saved_femu_comparison": (
            str(PROJECT_ROOT / "outputs/femu-comparison.json")
            if (PROJECT_ROOT / "outputs/femu-comparison.json").is_file()
            else None
        ),
    }


@contextmanager
def mooncake_master(output_dir: Path, *, pressure=None, lease_ms=0):
    """Start and stop only the master owned by this experiment."""
    if type(lease_ms) is not int or lease_ms < 0:
        raise ValueError("Expected a nonnegative integer master lease TTL in ms")
    port = free_port()
    binary = Path(sys.prefix) / "bin/mooncake_master"
    log_path = output_dir / "mooncake-master.log"
    config_path = output_dir / "mooncake-master.json"
    command = [
        str(binary),
        f"--rpc_port={port}",
        "--rpc_address=127.0.0.1",
        "--enable_offload=true",
        f"--default_kv_lease_ttl={pressure.lease_ttl_ms if pressure else lease_ms}",
        "--default_kv_soft_pin_ttl=0",
        "--rpc_thread_num=2",
        "--metrics_host=127.0.0.1",
        f"--metrics_port={free_port()}",
        "--logtostderr=true",
    ]
    if pressure:
        command.extend(
            [
                f"--eviction_high_watermark_ratio={pressure.high_watermark}",
                f"--eviction_ratio={pressure.eviction_ratio}",
            ]
        )
    config_path.write_text(json.dumps({"command": command}, indent=2) + "\n")
    with log_path.open("w") as log:
        process = subprocess.Popen(
            command, stdout=log, stderr=subprocess.STDOUT, env=native_environment()
        )
        try:
            deadline = time.monotonic() + 20
            while time.monotonic() < deadline:
                if process.poll() is not None:
                    raise RuntimeError(f"Mooncake master failed; see {log_path}")
                try:
                    with socket.create_connection(("127.0.0.1", port), timeout=0.1):
                        break
                except OSError:
                    time.sleep(0.05)
            else:
                raise TimeoutError(f"Mooncake master did not start; see {log_path}")
            yield f"127.0.0.1:{port}"
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=8)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=3)
