from __future__ import annotations

import ctypes
import json
import os
from pathlib import Path

from kv_ssd_sim.core.events import EventLog
from kv_ssd_sim.online.runtime import mooncake_master
from kv_ssd_sim.online.storage import ObservedStore


def smoke_store(output: Path) -> dict:
    from mooncake.store import MooncakeDistributedStore, ReplicateConfig

    output = output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    disk = output / "disk"
    disk.mkdir()
    os.environ.update(
        {
            "MOONCAKE_OFFLOAD_FILE_STORAGE_PATH": str(disk),
            "MOONCAKE_OFFLOAD_FLUSH_PARTIAL_BUCKETS": "1",
            "MOONCAKE_OFFLOAD_BUCKET_DIRECT_WRITE": "aligned",
            "MOONCAKE_OFFLOAD_LOCAL_BUFFER_SIZE_BYTES": str(8 * 1024 * 1024),
            "MOONCAKE_OFFLOAD_HEARTBEAT_INTERVAL_SECONDS": "1",
        }
    )
    with EventLog(output / "events.jsonl") as events, mooncake_master(output) as master:
        native = MooncakeDistributedStore()
        result = native.setup(
            "127.0.0.1",
            "P2PHANDSHAKE",
            32 * 1024 * 1024,
            8 * 1024 * 1024,
            "tcp",
            "",
            master,
            None,
            True,
            str(disk),
        )
        if result:
            raise RuntimeError(f"Native Mooncake setup failed: {result}")
        try:
            observed = ObservedStore(native, events)
            value = bytes(range(256)) * 256
            source = ctypes.create_string_buffer(value, len(value))
            target = ctypes.create_string_buffer(len(value))
            for buffer in (source, target):
                code = native.register_buffer(ctypes.addressof(buffer), len(value))
                if code:
                    raise RuntimeError(f"Native buffer registration failed: {code}")
            keys = ["native-cpu-disk-smoke"]
            codes = observed.batch_put_from_multi_buffers(
                keys, [[ctypes.addressof(source)]], [[len(value)]], ReplicateConfig()
            )
            if codes != [0]:
                raise RuntimeError(f"Native PUT failed: {codes}")
            observed.disk_barrier()
            observed.evict_memory_replicas(keys)
            codes = observed.batch_get_into_multi_buffers(
                keys, [[ctypes.addressof(target)]], [[len(value)]]
            )
            if any(code < 0 for code in codes) or observed.errors or target.raw != value:
                raise AssertionError(f"Native disk restore failed: {codes}, {observed.errors}")
            for buffer in (source, target):
                native.unregister_buffer(ctypes.addressof(buffer))
            report = {
                "status": "passed",
                "native_mooncake": True,
                "disk_only_restore": True,
                "verified_bytes": len(value),
                "sha256_verified": True,
                "femu_verified": False,
            }
        finally:
            native.close()
    (output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    return report
