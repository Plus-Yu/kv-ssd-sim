#!/usr/bin/env python3
"""Boot a disposable FEMU NVMe guest, with no host disks or network exposed."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import re
import stat
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / ".femu"
REVISION = "a898d22b2f894042afe0f4ef21867393c5c78a70"


def cpio_entry(name, data, mode):
    encoded = name.encode() + b"\0"
    fields = [0, mode, 0, 0, 1, 0, len(data), 0, 0, 0, 0, len(encoded), 0]
    header = b"070701" + b"".join(f"{value:08x}".encode() for value in fields)
    result = header + encoded
    result += b"\0" * (-len(result) % 4)
    result += data
    result += b"\0" * (-len(data) % 4)
    return result


def initramfs():
    modules = ASSETS / "modloop/modules"
    if not modules.is_dir():
        raise FileNotFoundError("Extract alpine-modloop into .femu/modloop first")
    output = bytearray()
    output.extend(cpio_entry("bootstrap", b"", stat.S_IFDIR | 0o755))
    busybox = ASSETS / "guest-tools/usr/bin/busybox"
    if not busybox.exists():
        busybox = ASSETS / "guest-tools/bin/busybox"
    output.extend(cpio_entry("bootstrap/busybox", busybox.read_bytes(), stat.S_IFREG | 0o755))
    kernel_root = next(p for p in modules.iterdir() if p.is_dir())
    dependencies = {}
    for line in (kernel_root / "modules.dep").read_text().splitlines():
        name, deps = line.split(":", 1)
        dependencies[name] = deps.split()
    needed = set()

    def include(name):
        if name in needed:
            return
        needed.add(name)
        for dependency in dependencies.get(name, []):
            include(dependency)

    for name in dependencies:
        if Path(name).stem in {"virtio_pci", "9pnet_virtio", "9p", "nvme", "ext4"}:
            include(name)
    for path in sorted(modules.rglob("*")):
        if path.suffix == ".ko" and str(path.relative_to(kernel_root)) not in needed:
            continue
        name = "usr/lib/modules/" + str(path.relative_to(modules))
        mode = path.lstat().st_mode
        data = (
            os.readlink(path).encode()
            if path.is_symlink()
            else b""
            if path.is_dir()
            else path.read_bytes()
        )
        output.extend(cpio_entry(name, data, mode))
    output.extend(
        cpio_entry("init", (ROOT / "scripts/femu_guest_init.sh").read_bytes(), stat.S_IFREG | 0o755)
    )
    output.extend(cpio_entry("TRAILER!!!", b"", 0))
    data = (ASSETS / "alpine-initramfs").read_bytes() + gzip.compress(output, mtime=0)
    fingerprint = hashlib.sha256(data).hexdigest()[:16]
    destination = ASSETS / f"guest-initramfs-{fingerprint}.gz"
    if not destination.exists():
        destination.write_bytes(data)
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trial", required=True)
    parser.add_argument("--nand-read-ns", type=int, default=40000)
    parser.add_argument("--accel", choices=("tcg", "kvm"), default="tcg")
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--workload", choices=("replay", "aiperf"), default="replay")
    args = parser.parse_args()
    if not re.fullmatch(r"femu-[a-z0-9-]+", args.trial):
        parser.error("trial must match femu-[a-z0-9-]+")
    if not 0 <= args.nand_read_ns <= 1000000000:
        parser.error("nand-read-ns must be between 0 and 1 second")
    ramdisk = initramfs()
    output = ROOT / "outputs" / args.trial
    output.mkdir(exist_ok=False)
    from kv_ssd_sim.metadata_cache import create

    create(output / "package-distributions.json")
    from kv_ssd_sim.runtime import native_environment

    preparation_env = native_environment()
    preparation_env["VLLM_CACHE_ROOT"] = str(output / "model-cache")
    with (output / "prepare-model-info.log").open("w") as log:
        subprocess.run(
            [
                sys.executable,
                "-c",
                (
                    "import sys; from vllm.config import ModelConfig; "
                    "ModelConfig(model=sys.argv[1], skip_tokenizer_init=True, "
                    "dtype='float16', enforce_eager=True)"
                ),
                str(ROOT / "configs/model"),
            ],
            env=preparation_env,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=True,
            timeout=120,
        )
    binary = ASSETS / "build/qemu-system-x86_64"
    # 512 MiB raw NAND, 384 MiB exposed namespace (25% overprovisioning).
    device = (
        "femu,devsz_mb=384,namespaces=1,femu_mode=1,"
        "secsz=512,secs_per_pg=8,pgs_per_blk=64,blks_per_pl=256,"
        "pls_per_lun=1,luns_per_ch=4,nchs=2,"
        f"pg_rd_lat={args.nand_read_ns},pg_wr_lat=200000,blk_er_lat=2000000,"
        "ch_xfer_lat=0,gc_thres_pcent=75,gc_thres_pcent_high=95"
    )
    command = [
        str(binary),
        "-L",
        str(ASSETS / "FEMU-sparse/pc-bios"),
        "-name",
        args.trial,
        "-machine",
        f"q35,accel={args.accel}",
        "-cpu",
        "max" if args.accel == "tcg" else "host",
        "-smp",
        "2",
        "-m",
        "4096",
        "-nodefaults",
        "-no-reboot",
        "-display",
        "none",
        "-serial",
        "stdio",
        "-monitor",
        "none",
        "-nic",
        "none",
        "-kernel",
        str(ASSETS / "alpine-vmlinuz"),
        "-initrd",
        str(ramdisk),
        "-append",
        f"console=ttyS0 quiet panic=1 rdinit=/init kvtrial={args.trial} kvworkload={args.workload}",
        "-device",
        device,
    ]
    for tag, path, readonly in (
        ("usr", Path("/usr"), True),
        ("conda", Path("/home/plusyu/miniconda3"), True),
        ("project", ROOT, False),
    ):
        command += [
            "-fsdev",
            f"local,id={tag},path={path},security_model=none"
            + (",readonly=on" if readonly else ""),
            "-device",
            f"virtio-9p-pci,fsdev={tag},mount_tag={tag}",
        ]
    manifest = {
        "femu_revision": REVISION,
        "command": command,
        "accelerator": args.accel,
        "nand_read_ns": args.nand_read_ns,
        "workload": args.workload,
        "inputs_sha256": {
            str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (
                ASSETS / "alpine-vmlinuz",
                ramdisk,
                binary,
                output / "package-distributions.json",
                *sorted((output / "model-cache").rglob("*.json")),
            )
        },
        "host_disk_exposed": False,
        "host_network_exposed": False,
    }
    (output / "launch.json").write_text(json.dumps(manifest, indent=2) + "\n")
    if args.prepare_only:
        print(output)
        return 0
    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = str(ASSETS / "toolchain/lib")
    with (output / "serial.log").open("w") as log:
        result = subprocess.run(
            command,
            cwd=output,
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            timeout=args.timeout,
            check=False,
        )
    exit_file = output / "guest-exit-code.txt"
    if result.returncode or not exit_file.exists() or exit_file.read_text().strip() != "0":
        raise RuntimeError(f"FEMU guest failed; inspect {output}/serial.log and replay.log")
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
