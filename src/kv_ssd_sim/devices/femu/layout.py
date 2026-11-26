"""Off-window object and extent inventory for the single-device layout experiment."""

import ctypes
import fcntl
import hashlib
import json
import os
import re
import struct
import time
from pathlib import Path

from kv_ssd_sim.core.paths import PROJECT_ROOT
from kv_ssd_sim.devices.femu.block_capture import STOP

FIEMAP = 0xC020660B
HEADER = struct.Struct("=QQIIII")
EXTENT = struct.Struct("=QQQQQIIII")
KEY = re.compile(rb"default\x00(k3-online-tp8@[^\x00]{1,250}?@group:[0-3]@[a-f0-9]{64})")
INSTANCE = Path("/sys/kernel/tracing/instances/kv_layout")


def load_policy(mode):
    if mode not in ("mixed", "separate"):
        raise ValueError("Unsupported bucket layout")
    lib = ctypes.CDLL(
        str(PROJECT_ROOT / ".native/layout/libbucketlayout.so"), mode=ctypes.RTLD_GLOBAL
    )
    lib.configure_bucket_layout(mode == "separate")
    for k in ("layout_calls", "layout_mixed_plans", "layout_emitted"):
        getattr(lib, k).restype = ctypes.c_uint64
    return lib


def extent_map(stream):
    start = 0
    rows = []
    while True:
        buf = bytearray(HEADER.size + 128 * EXTENT.size)
        HEADER.pack_into(buf, 0, start, (1 << 64) - 1 - start, 0, 0, 128, 0)
        fcntl.ioctl(stream.fileno(), FIEMAP, buf)
        count = HEADER.unpack_from(buf)[3]
        if not count:
            break
        for i in range(count):
            logical, physical, size, _, _, flags, *_ = EXTENT.unpack_from(
                buf, HEADER.size + i * EXTENT.size
            )
            if not size or logical < start or flags & ~1:
                raise ValueError(f"Unresolved or non-ordinary FIEMAP extent {logical=} {flags=}")
            rows.append([logical, physical, size])
            start = logical + size
        if flags & 1:
            break
    return rows


def inventory(directory, expected):
    files = []
    seen = set()
    for path in sorted(directory.iterdir()):
        with path.open("rb") as f:
            stat = os.fstat(f.fileno())
            if path.suffix not in (".bucket", ".meta") and stat.st_size:
                raise ValueError(f"Unknown nonempty offload file {path.name}")
            row = {
                "path": path.name,
                "inode": stat.st_ino,
                "size": stat.st_size,
                "extents": extent_map(f),
                "objects": [],
            }
            if path.suffix == ".bucket":
                while f.tell() < stat.st_size:
                    start = f.tell()
                    match = KEY.match(f.read(min(320, stat.st_size - start)))
                    if not match:
                        raise ValueError(f"Unknown bucket key at {path}:{start}")
                    key = match[1].decode()
                    size, digest = expected[key]
                    if key in seen:
                        raise ValueError("Duplicate physical object " + key)
                    seen.add(key)
                    begin = start + match.end()
                    f.seek(begin)
                    remaining = size
                    sha = hashlib.sha256()
                    while remaining:
                        part = f.read(min(1 << 20, remaining))
                        if not part:
                            raise ValueError("Truncated bucket")
                        sha.update(part)
                        remaining -= len(part)
                    if sha.hexdigest() != digest:
                        raise ValueError("Bucket payload digest differs")
                    row["objects"].append(
                        {
                            "key": key,
                            "header_offset": start,
                            "file_offset": begin,
                            "bytes": size,
                            "sha256": digest,
                        }
                    )
            files.append(row)
    if seen != set(expected):
        raise ValueError("Bucket inventory differs from published keys")
    return files


def verify_layout(output, device, observed, lib):
    trial = Path(os.environ["KV_SSD_CAPTURE_ROOT"])
    # Start the collector's idle observation only after all disk replicas exist.
    keys = observed.disk_barrier(timeout_s=120)
    STOP.touch()
    deadline = time.monotonic() + 240
    while not (trial / "capture.json").is_file():
        if time.monotonic() > deadline:
            raise TimeoutError("Device measurement did not close")
        time.sleep(0.2)
    INSTANCE.joinpath("tracing_on").write_text("0")
    (trial / "ext4-alloc.trace").write_bytes((INSTANCE / "trace").read_bytes())
    result = {
        "layout": os.environ["KV_SSD_LAYOUT"],
        "native_plans": lib.layout_calls(),
        "mixed_input_plans": lib.layout_mixed_plans(),
        "emitted_buckets": lib.layout_emitted(),
        "validation_start_ns": time.monotonic_ns(),
    }
    if not result["native_plans"]:
        raise RuntimeError("Native bucket planner adapter did not run")
    result["files"] = inventory((output / "disk").resolve(), observed.expected)
    mem = [
        k
        for k, v in observed.native.batch_get_replica_desc(keys).items()
        if any(d.is_memory_replica() for d in v)
    ]
    if mem:
        observed.evict_memory_replicas(mem)
    tiers = observed.native.batch_get_replica_desc(keys)
    if not all(
        any(d.is_local_disk_replica() for d in tiers[k])
        and not any(d.is_memory_replica() for d in tiers[k])
        for k in keys
    ):
        raise ValueError("Recovery requires every key to be disk-only")
    device.clear_guest_page_cache()
    size = max(v[0] for v in observed.expected.values())
    scratch = ctypes.create_string_buffer(size)
    address = ctypes.addressof(scratch)
    if observed.native.register_buffer(address, size) != 0:
        raise RuntimeError("Recovery buffer registration failed")
    before = device.counters()
    checks = {}
    try:
        for key in sorted(keys):
            n, expected = observed.expected[key]
            ctypes.memset(address, 0xA5, n)
            code = observed.native.batch_get_into_multi_buffers([key], [[address]], [[n]])
            actual = hashlib.sha256(memoryview(scratch).cast("B")[:n]).hexdigest()
            if list(code) != [n] or actual != expected:
                raise ValueError("Disk GET differs: " + key)
            checks[key] = actual
    finally:
        observed.native.unregister_buffer(address)
    after = device.counters()
    if after["read_sectors"] <= before["read_sectors"]:
        raise ValueError("No device reads during recovery")
    result.update(
        restored=checks, restore_device_delta={k: after[k] - v for k, v in before.items()}
    )
    (trial / "layout.json").write_text(json.dumps(result, indent=2) + "\n")
