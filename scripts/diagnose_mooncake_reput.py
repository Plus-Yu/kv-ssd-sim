#!/usr/bin/env python3
"""Isolate same-key SSD re-offload from vLLM, KDA, and the replay observer."""

import argparse
import hashlib
import importlib.metadata
import json
import os
import sys
import time
from pathlib import Path

from kv_ssd_sim.runtime import free_port, mooncake_master, native_environment
from kv_ssd_sim.storage import replica_types


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--wait-seconds", type=float, default=60)
    args = parser.parse_args()
    if os.environ.get("KV_SSD_REPUT_ENV") != "1":
        env = native_environment()
        env.update(KV_SSD_REPUT_ENV="1", GLOG_v="1", GLOG_logtostderr="1")
        os.execve(sys.executable, [sys.executable, __file__, *sys.argv[1:]], env)
    if args.wait_seconds <= 0:
        parser.error("--wait-seconds must be positive")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    disk = output / "disk"
    disk.mkdir()
    os.environ.update(
        {
            "MOONCAKE_OFFLOAD_FILE_STORAGE_PATH": str(disk),
            "MOONCAKE_OFFLOAD_FLUSH_PARTIAL_BUCKETS": "1",
            "MOONCAKE_OFFLOAD_BUCKET_DIRECT_WRITE": "aligned",
            "MOONCAKE_OFFLOAD_LOCAL_BUFFER_SIZE_BYTES": str(8 * 1024**2),
            "MOONCAKE_OFFLOAD_TOTAL_SIZE_LIMIT_BYTES": str(256 * 1024**2),
            "MOONCAKE_OFFLOAD_HEARTBEAT_INTERVAL_SECONDS": "1",
        }
    )
    from mooncake.store import MooncakeDistributedStore

    report = {
        "version": importlib.metadata.version("mooncake-transfer-engine-non-cuda"),
        "wait_seconds": args.wait_seconds,
        "snapshots": [],
        "operations": [],
    }
    started = time.monotonic()
    with mooncake_master(output) as master:
        store = MooncakeDistributedStore()
        try:
            assert (
                store.setup(
                    f"127.0.0.1:{free_port()}",
                    "P2PHANDSHAKE",
                    32 * 1024**2,
                    8 * 1024**2,
                    "tcp",
                    "",
                    master,
                    None,
                    True,
                    str(disk),
                )
                == 0
            )
            original = {
                "small-same": b"A" * 2624,
                "large-same": b"A" * 8192,
                "small-changed": b"A" * 2624,
                "keep-disk": b"A" * 2624,
            }
            keys = list(original)

            def snapshot(label):
                desc = store.batch_get_replica_desc(keys)
                tiers = {k: replica_types(desc.get(k, [])) for k in keys}
                report["snapshots"].append(
                    {"label": label, "elapsed_s": time.monotonic() - started, "tiers": tiers}
                )
                (output / "diagnosis.json").write_text(json.dumps(report, indent=2) + "\n")
                return tiers

            def put(key, value):
                code = store.put(key, value)
                report["operations"].append(
                    {
                        "operation": "put",
                        "key": key,
                        "result": code,
                        "sha256": hashlib.sha256(value).hexdigest(),
                    }
                )
                assert code == 0

            for key, value in original.items():
                put(key, value)
            deadline = time.monotonic() + 15
            while True:
                tiers = snapshot("initial_offload")
                if all(any("DISK" in t for t in ts) for ts in tiers.values()):
                    break
                if time.monotonic() > deadline:
                    raise TimeoutError("Initial offload failed")
                time.sleep(0.25)
            cleared = list(store.batch_replica_clear(keys, segment_name=store.get_hostname()))
            report["operations"].append({"operation": "clear_memory", "result": cleared})
            tiers = snapshot("memory_evicted")
            assert all(ts and "MEMORY" not in ts for ts in tiers.values())
            for key in keys:
                if key != "keep-disk":
                    code = store.remove(key)
                    report["operations"].append({"operation": "remove", "key": key, "result": code})
                    assert code == 0
            snapshot("removed")
            for key, value in original.items():
                put(key, b"B" * len(value) if key == "small-changed" else value)
            keys.append("fresh-key")
            put("fresh-key", b"A" * 2624)
            deadline = time.monotonic() + args.wait_seconds
            while True:
                snapshot("after_reput")
                if time.monotonic() >= deadline:
                    break
                time.sleep(min(1, max(0, deadline - time.monotonic())))
            snapshot("final")
            print(
                json.dumps(
                    {
                        "final_tiers": report["snapshots"][-1]["tiers"],
                    },
                    indent=2,
                )
            )
        finally:
            store.close()


if __name__ == "__main__":
    main()
