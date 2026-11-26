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
import tempfile
from pathlib import Path

from kv_ssd_sim.core.paths import PROJECT_ROOT as ROOT

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
    indexes = sorted(p for p in modules.glob("*/modules.dep") if p.is_file())
    if not indexes:
        raise FileNotFoundError(f"No kernel dependency index matching {modules}/*/modules.dep")
    if len(indexes) > 1:
        raise ValueError(f"Expected one kernel version in {modules}, found: {indexes}")
    index = indexes[0]
    dependencies = {}
    for line in index.read_text().splitlines():
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
    needed_files = {index.parent / name for name in needed}
    for path in sorted(modules.rglob("*")):
        if path.suffix == ".ko" and path not in needed_files:
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


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trial", required=True)
    parser.add_argument("--nand-read-ns", type=int, default=40000)
    parser.add_argument("--accel", choices=("tcg", "kvm"), default="tcg")
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument(
        "--workload", choices=("replay", "aiperf", "k3-probe", "nand-probe"), default="replay"
    )
    parser.add_argument(
        "--natural-cache", action="store_true", help="Use natural AIPerf cache reuse"
    )
    parser.add_argument(
        "--block-capture",
        choices=("stream", "buffer"),
        help="Instrument natural replay with live or control block capture",
    )
    parser.add_argument("--input", type=Path)
    parser.add_argument(
        "--pressure-config", type=Path, help="Natural-capacity replay with concurrent branches"
    )
    parser.add_argument("--hybrid-config", type=Path, default=ROOT / "configs/hybrid-tiny.json")
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--num-blocks", type=int, default=128)
    parser.add_argument("--prefill-ms", type=float, default=10)
    parser.add_argument("--decode-ms", type=float, default=2)
    parser.add_argument("--guest-ram-mib", type=int, default=4096)
    parser.add_argument("--namespace-mib", type=int, default=384)
    parser.add_argument("--blocks-per-plane", type=int, default=256)
    parser.add_argument(
        "--store-segment-mib", type=int, default=768, help="Embedded Store segment for the K3 probe"
    )
    parser.add_argument(
        "--offload-quota-mib", type=int, default=1024, help="Disk quota for the K3 probe"
    )
    parser.add_argument("--bucket-layout", choices=("mixed", "separate"))
    parser.add_argument(
        "--nand-stats", action="store_true", help="Collect same-window QMP NAND counters"
    )
    args = parser.parse_args(argv)
    if args.workload == "nand-probe":
        args.nand_stats = True
        if (args.namespace_mib, args.blocks_per_plane) != (48, 32):
            parser.error("nand-probe calibration uses --namespace-mib 48 --blocks-per-plane 32")
    elif args.nand_stats and not (
        args.block_capture and args.workload == "aiperf" and args.natural_cache
    ):
        parser.error("--nand-stats requires natural AIPerf block capture or nand-probe")
    if args.nand_stats and args.blocks_per_plane > 4096:
        parser.error("Atomic line inventory supports up to 4096 lines")
    if args.bucket_layout and (
        args.block_capture != "stream"
        or not args.natural_cache
        or args.workload != "aiperf"
        or json.loads(args.hybrid_config.read_text()).get("profile") != "k3-tp8"
    ):
        parser.error("Layout measurement requires natural K3 AIPerf with stream capture")

    if not (512 <= args.guest_ram_mib and 0 < args.namespace_mib <= 2 * args.blocks_per_plane):
        parser.error(
            "Namespace must fit the raw NAND geometry (2 MiB per block per plane across 8 LUNs)"
        )
    if args.workload == "k3-probe" and not (
        0 < args.store_segment_mib < args.guest_ram_mib
        and 0 < args.offload_quota_mib < args.namespace_mib
    ):
        parser.error("K3 Store budgets must fit guest RAM and disk")
    if not re.fullmatch(r"femu-[a-z0-9-]+", args.trial):
        parser.error("trial must match femu-[a-z0-9-]+")
    if not 0 <= args.nand_read_ns <= 1000000000:
        parser.error("nand-read-ns must be between 0 and 1 second")
    if args.block_capture and not (args.natural_cache and args.workload == "aiperf"):
        parser.error("Block capture requires a natural AIPerf workload")
    ramdisk = initramfs()
    output = ROOT / "outputs" / args.trial
    output.mkdir(exist_ok=False)
    from kv_ssd_sim.devices.femu.workload import digest, stage

    if args.workload == "k3-probe":
        bundle = output / "input"
        bundle.mkdir()
        (bundle / "resolved.json").write_bytes(
            (ROOT / "configs/kimi-k3-b300-step2/resolved.json").read_bytes()
        )
        (bundle / "config.json").write_text(
            json.dumps(
                {
                    "store_segment_mib": args.store_segment_mib,
                    "offload_quota_mib": args.offload_quota_mib,
                    "namespace_mib": args.namespace_mib,
                    "guest_ram_mib": args.guest_ram_mib,
                },
                indent=2,
            )
            + "\n"
        )
    elif args.workload == "nand-probe":
        (output / "input").mkdir()
    else:
        stage(output / "input", args)
    from kv_ssd_sim.online.metadata_cache import create

    create(output / "package-distributions.json")
    if args.workload != "nand-probe":
        from kv_ssd_sim.online.runtime import native_environment

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
    # Eight LUNs, 256 KiB per erase block.
    device = (
        f"femu,devsz_mb={args.namespace_mib},namespaces=1,femu_mode=1,"
        f"secsz=512,secs_per_pg=8,pgs_per_blk=64,blks_per_pl={args.blocks_per_plane},"
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
        str(args.guest_ram_mib),
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
        f"console=ttyS0 quiet panic=1 rdinit=/init kvtrial={args.trial} kvworkload={args.workload}"
        + (f" kvcapture={args.block_capture}" if args.block_capture else "")
        + (" kvnand=1" if args.nand_stats else "")
        + (f" kvlayout={args.bucket_layout}" if args.bucket_layout else ""),
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
    if args.nand_stats:
        command.extend(["-qmp", "unix:nand-qmp.sock,server=on,wait=off"])
    manifest = {
        "femu_revision": REVISION,
        "command": command,
        "accelerator": args.accel,
        "block_capture": args.block_capture,
        "nand_stats": args.nand_stats,
        "bucket_layout": args.bucket_layout,
        "nand_read_ns": args.nand_read_ns,
        "workload": args.workload,
        "guest_inputs_sha256": {p.name: digest(p) for p in sorted((output / "input").iterdir())},
        "inputs_sha256": {
            str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (
                ASSETS / "alpine-vmlinuz",
                ramdisk,
                binary,
                ROOT / "scripts/femu_workload.py",
                ROOT / "scripts/run_femu.py",
                ROOT / "src/kv_ssd_sim/devices/femu/launcher.py",
                ROOT / "src/kv_ssd_sim/devices/femu/workload.py",
                ROOT / "src/kv_ssd_sim/online/metadata_cache.py",
                ROOT / "src/kv_ssd_sim/core/paths.py",
                *(
                    (
                        ROOT / "scripts/femu_qmp.py",
                        ROOT / "src/kv_ssd_sim/devices/femu/qmp.py",
                        ROOT / "src/kv_ssd_sim/devices/femu/nand.py",
                        ROOT / "src/kv_ssd_sim/devices/femu/block_capture.py",
                        ROOT / "src/kv_ssd_sim/devices/femu/probe.py",
                        ROOT / "src/kv_ssd_sim/analysis/nand.py",
                        ROOT / "src/kv_ssd_sim/analysis/block_capture.py",
                        ROOT / "src/kv_ssd_sim/analysis/block_io.py",
                        ROOT / "src/kv_ssd_sim/analysis/aiperf.py",
                        ROOT / "src/kv_ssd_sim/nand.py",
                        ROOT / "scripts/femu_nand_probe.py",
                        ROOT / "src/kv_ssd_sim/block_capture.py",
                    )
                    if args.nand_stats
                    else ()
                ),
                *(
                    (ROOT / ".native/layout/libbucketlayout.so", ROOT / "native/bucket_layout.cpp")
                    if args.bucket_layout
                    else ()
                ),
                *((ROOT / "scripts/verify_k3_shards.py",) if args.workload == "k3-probe" else ()),
                ROOT / "scripts/aiperf_gate/sitecustomize.py",
                ROOT / "scripts/run_aiperf.py",
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
        if args.nand_stats:
            from kv_ssd_sim.devices.femu.qmp import run

            with tempfile.TemporaryDirectory(prefix="kv-nand-") as private:
                sockpath = Path(private) / "qmp.sock"
                live_command = command.copy()
                live_command[live_command.index("-qmp") + 1] = f"unix:{sockpath},server=on,wait=off"
                manifest["command"] = live_command
                (output / "launch.json").write_text(json.dumps(manifest, indent=2) + "\n")
                rc = run(live_command, output, log, env, args.timeout, sockpath)
        else:
            rc = subprocess.run(
                command,
                cwd=output,
                env=env,
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
                timeout=args.timeout,
                check=False,
            ).returncode
    exit_file = output / "guest-exit-code.txt"
    if rc or not exit_file.exists() or exit_file.read_text().strip() != "0":
        raise RuntimeError(f"FEMU guest failed; inspect {output}/serial.log and replay.log")
    if args.nand_stats:
        from kv_ssd_sim.analysis.nand import verify

        verify(output)
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
