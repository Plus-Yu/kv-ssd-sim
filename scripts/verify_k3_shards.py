"""Exercise actual K3-sized TP8 Store objects with sequential CPU rank contexts."""

import argparse
import hashlib
import json
import os
import sys
import threading
from pathlib import Path

from kv_ssd_sim.core.events import EventLog
from kv_ssd_sim.core.paths import PROJECT_ROOT
from kv_ssd_sim.models.deployment import native_deployment, require
from kv_ssd_sim.online.runtime import mooncake_master, native_environment
from kv_ssd_sim.online.storage import ObservedStore


def run(output, *, mount=None, segment_mib=768, quota_mib=1024):
    import numpy as np
    import torch
    from mooncake.store import MooncakeDistributedStore, ReplicateConfig
    from vllm.distributed.kv_transfer.kv_connector.v1.mooncake_store import worker as upstream
    from vllm.distributed.kv_transfer.kv_connector.v1.mooncake_store.coordinator import (
        MooncakeStoreCoordinator,
    )
    from vllm.model_executor.layers.mamba.checkpoint import compute_mamba_prefill_checkpoints
    from vllm.sampling_params import SamplingParams
    from vllm.utils.hashing import sha256
    from vllm.v1.core.kv_cache_manager import KVCacheManager
    from vllm.v1.core.kv_cache_utils import get_request_block_hasher, init_none_hash
    from vllm.v1.request import Request

    torch.set_num_threads(1)
    folder = PROJECT_ROOT / "configs/kimi-k3-b300-step2"
    plan, allocation = native_deployment(folder)
    require(
        json.loads(json.dumps(plan)) == json.loads((folder / "resolved.json").read_text()),
        "Step 2 baseline changed",
    )
    groups = allocation.kv_cache_groups
    geo = plan["geometry"]
    block, unit = geo["manager_block_tokens"], geo["hash_tokens"]
    length = block + 1
    init_none_hash(sha256)
    request = Request(
        request_id="step3",
        prompt_token_ids=list(range(length)),
        sampling_params=SamplingParams(max_tokens=1),
        pooling_params=None,
        block_hasher=get_request_block_hasher(unit, sha256),
    )
    manager = KVCacheManager(allocation, 131072, block, unit, max_in_flight_tokens=4096)
    require(manager.allocate_slots(request, length) is not None, "Native allocation failed")
    tables = manager.get_blocks("step3").get_block_ids()
    offsets, columns = compute_mamba_prefill_checkpoints(
        [length], [length], unit, block, geo["prefill_checkpoint_alignment_tokens"], False, unit
    )
    require(
        offsets == [block] and columns == [0] and all(row[0] > 0 for row in tables),
        "Missing native checkpoint slot",
    )
    pages = 1 + max(max(row) for row in tables)
    slab = torch.empty(
        (max(len(g.layer_names) for g in groups), pages, geo["common_page_bytes"]),
        dtype=torch.uint8,
    )
    slab_bytes = slab.numel()
    device = None
    if mount:
        from kv_ssd_sim.devices.femu.device import FEMUDevice

        device = FEMUDevice(mount)
    disk = mount / "k3-probe" if mount else output / "disk"
    disk.mkdir()
    os.environ.update(
        {
            "MOONCAKE_OFFLOAD_FILE_STORAGE_PATH": str(disk),
            "MOONCAKE_OFFLOAD_BUCKET_DIRECT_WRITE": "aligned",
            "MOONCAKE_OFFLOAD_FLUSH_PARTIAL_BUCKETS": "1",
            "MOONCAKE_OFFLOAD_LOCAL_BUFFER_SIZE_BYTES": str(64 << 20),
            "MOONCAKE_OFFLOAD_TOTAL_SIZE_LIMIT_BYTES": str(quota_mib << 20),
            "MOONCAKE_OFFLOAD_HEARTBEAT_INTERVAL_SECONDS": "1",
        }
    )
    from kv_ssd_sim.adapters.vllm import metadata_rank_worker

    rank_workers = [
        metadata_rank_worker(groups, unit, rank, "Kimi-K3", "step3-k3") for rank in range(8)
    ]
    reader = rank_workers[0]
    reader.coord = MooncakeStoreCoordinator(groups, block, unit, enable_partial_hash_hits=True)
    reader._capacity_only = False
    reader._selective_lease = None
    reader.kv_connector_stats = upstream.MooncakeStoreConnectorStats()
    reader._kv_connector_stats_lock = threading.Lock()
    objects, restores = {}, []

    def transfer(rank, gid, mode, observed):
        group = groups[gid]
        spec = group.kv_cache_spec
        db = rank_workers[rank].token_dbs[gid]
        layout = db.store_layout
        (shard,) = layout.local_shard_ids
        chunk = db.chunk_size
        is_mla = plan["groups"][gid]["spec"] == "MLAAttentionSpec"
        width = spec.real_page_size_bytes
        views = []
        expected = []
        bid = tables[gid][0]
        hashes = request.block_hashes
        for layer_index, name in enumerate(group.layer_names):
            raw = slab[layer_index, :, :width]
            views.append(raw.view(pages, 1, block, -1) if is_mla else raw.view(pages, 1, 1, -1))
            seed = f"{name}:{shard}:{hashes[-1].hex()}".encode()
            data = hashlib.shake_256(seed).digest(width)
            expected.append(data)
            slab[layer_index, bid].fill_(165)
            if mode == "put":
                raw[bid].numpy()[:] = np.frombuffer(data, dtype=np.uint8)
        layout.register_kv_caches(views, pages)
        for start in range(0, block, chunk):
            end = start + chunk
            key = layout.key_for(shard, hashes[end // unit - 1])
            addresses, sizes, bids = layout.prepare_values([(start, end)], tables[gid], [shard])
            size = sum(sizes[0])
            require(
                bids == [bid] and size == plan["groups"][gid]["logical_bytes_per_object_per_shard"],
                "Descriptor mismatch",
            )
            if mode == "put":
                require(key not in objects, "Unexpected key collision")
                codes = observed.batch_put_from_multi_buffers(
                    [key], addresses, sizes, ReplicateConfig()
                )
                require(codes == [0], "PUT failed")
                objects[key] = {
                    "key": key,
                    "rank": rank,
                    "group": gid,
                    "shard": shard,
                    "end_token": end,
                    "bytes": size,
                    "sha256": observed.expected[key][1],
                }
            else:
                require(key in objects, "Restore key was never written")
                (code,) = observed.batch_get_into_multi_buffers([key], addresses, sizes)
                require(code >= 0 and not observed.errors, "GET/digest failed")
                restores.append({"rank": rank, "key": key, "bytes": size})
        if mode == "get":
            for index, data in enumerate(expected):
                raw = slab[index, bid].numpy()
                require(
                    memoryview(raw[:width]) == data and (raw[width:] == 165).all(),
                    "Wrong restored layer, ordering or page padding",
                )

    with EventLog(output / "events.jsonl") as events, mooncake_master(output) as master:
        store = MooncakeDistributedStore()
        require(
            store.setup(
                "127.0.0.1",
                "P2PHANDSHAKE",
                segment_mib << 20,
                64 << 20,
                "tcp",
                "",
                master,
                None,
                True,
                str(disk),
            )
            == 0,
            "Store setup failed",
        )
        try:
            require(
                store.register_buffer(slab.data_ptr(), slab_bytes) == 0, "CPU registration failed"
            )
            observed = ObservedStore(store, events)
            reader.store = store

            def lookup():
                return reader.lookup(length, request.block_hashes).hit_length

            for gid in range(len(groups)):
                transfer(0, gid, "put", observed)
            observed.disk_barrier(timeout_s=90)
            observed.evict_memory_replicas(list(objects))
            for gid in range(len(groups)):
                transfer(0, gid, "get", observed)
            single_hit = lookup()
            require(single_hit == 0, "One rank reported a full hit")
            for rank in range(1, 8):
                for gid in range(3):
                    if (rank, gid) != (7, 2):
                        transfer(rank, gid, "put", observed)
            require(
                len(objects) == 35
                and all(code == 1 for code in store.batch_is_exist(list(objects))),
                "Missing-shard fixture invalid",
            )
            partial_hit = lookup()
            require(partial_hit == 0, "Incomplete shard set hit")
            transfer(7, 2, "put", observed)
            observed.disk_barrier(timeout_s=90)
            memory_keys = [
                key
                for key, descs in store.batch_get_replica_desc(list(objects)).items()
                if any(d.is_memory_replica() for d in descs)
            ]
            if memory_keys:
                observed.evict_memory_replicas(memory_keys)
            tiers = store.batch_get_replica_desc(list(objects))
            require(
                all(
                    any(d.is_local_disk_replica() for d in tiers.get(key, []))
                    and not any(d.is_memory_replica() for d in tiers[key])
                    for key in objects
                ),
                "Not all keys disk-only",
            )
            if device:
                device.clear_guest_page_cache()
                before = device.counters()
            full_hit = lookup()
            require(full_hit == block, "Complete object set missed")
            restores.clear()
            for rank in range(8):
                for gid in range(len(groups)):
                    transfer(rank, gid, "get", observed)
            if device:
                device.record_get(before, events, request.request_id)
                femu = device.verify()
            require(len(objects) == 36 and len(restores) == 120, "Unexpected object coverage")
            require(store.unregister_buffer(slab.data_ptr()) == 0, "Unregister failed")
        finally:
            store.close()
    return {
        "status": "passed",
        "scope": "one real-size native allocated checkpoint; eight sequential logical TP ranks, one embedded TCP store, "
        + ("FEMU" if device else "host filesystem"),
        **({"femu": femu, "segment_mib": segment_mib, "quota_mib": quota_mib} if device else {}),
        "step2_sha256": hashlib.sha256((folder / "resolved.json").read_bytes()).hexdigest(),
        "request_tokens": length,
        "checkpoint_position": offsets[0],
        "block_tables": tables,
        "allocated_host_buffer_bytes": slab_bytes,
        "lookup_tokens": {
            "rank0_only": single_hit,
            "missing_group2_rank7": partial_hit,
            "all_disk_only": full_hit,
        },
        "unique_object_bytes": sum(obj["bytes"] for obj in objects.values()),
        "objects": list(objects.values()),
        "restores": restores,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--guest-bundle", type=Path)
    args = parser.parse_args()
    if not os.environ.get("KV_SSD_STEP3_ENV"):
        env = native_environment()
        env.update(KV_SSD_STEP3_ENV="1", PYTHONHASHSEED="42", VLLM_SSM_CONV_STATE_LAYOUT="SD")
        os.execve(sys.executable, [sys.executable, __file__, *sys.argv[1:]], env)
    destination = args.output.resolve()
    destination.mkdir(parents=True, exist_ok=False)
    try:
        options = {}
        if args.guest_bundle:
            bundle = args.guest_bundle
            launch = json.loads((bundle.parent / "launch.json").read_text())
            for name in ("resolved.json", "config.json"):
                require(
                    hashlib.sha256((bundle / name).read_bytes()).hexdigest()
                    == launch["guest_inputs_sha256"][name],
                    "Staged input differs",
                )
            require(
                (bundle / "resolved.json").read_bytes()
                == (PROJECT_ROOT / "configs/kimi-k3-b300-step2/resolved.json").read_bytes(),
                "K3 geometry differs",
            )
            cfg = json.loads((bundle / "config.json").read_text())
            require(
                int(Path("/sys/class/block/nvme0n1/size").read_text()) * 512
                == cfg["namespace_mib"] << 20,
                "Namespace size differs",
            )
            options = dict(
                mount=Path("/mnt/femu"),
                segment_mib=cfg["store_segment_mib"],
                quota_mib=cfg["offload_quota_mib"],
            )
        report = run(destination, **options)
    except Exception as exc:
        (destination / "summary.json").write_text(
            json.dumps({"status": "failed", "error": repr(exc)}) + "\n"
        )
        raise
    (destination / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["lookup_tokens"]))
