"""Validate metadata lookup and per-rank GET descriptors against the archived K3 shard probe."""

import argparse
import hashlib
import json
import os
import tarfile
from collections import Counter
from pathlib import Path

from kv_ssd_sim.object_store import MIB, Clock, ObjectStore, StoreConfig
from kv_ssd_sim.runtime import PROJECT_ROOT as ROOT


def check(reference, events_path=None):
    clock = Clock()
    events = []

    def record(event, **data):
        events.append(dict(event=event, time_ns=clock.now, **data))

    with ObjectStore(clock, StoreConfig(1, 768 * MIB, 64 * MIB, 1024 * MIB), record) as store:
        result = exercise(reference, store, clock, events)
    if events_path:
        with events_path.open("x") as target:
            for event in events:
                target.write(json.dumps(event) + "\n")
    return result


def exercise(reference, store, clock, events):
    from vllm.distributed.kv_transfer.kv_connector.v1.mooncake_store.coordinator import (
        MooncakeStoreCoordinator,
    )
    from vllm.distributed.kv_transfer.kv_connector.v1.mooncake_store.data import (
        LoadSpec,
        MooncakeStoreConnectorMetadata,
        ReqMeta,
    )
    from vllm.sampling_params import SamplingParams
    from vllm.utils.hashing import sha256
    from vllm.v1.core.kv_cache_manager import KVCacheManager
    from vllm.v1.core.kv_cache_utils import get_request_block_hasher, init_none_hash
    from vllm.v1.request import Request

    from kv_ssd_sim.deployment import native_deployment
    from kv_ssd_sim.object_worker import MetadataRank, MetadataTransfers

    plan, alloc = native_deployment(ROOT / "configs/kimi-k3-b300-step2")
    geo, groups = plan["geometry"], alloc.kv_cache_groups
    block, unit = geo["manager_block_tokens"], geo["hash_tokens"]
    init_none_hash(sha256)
    request = Request(
        request_id="step3",
        prompt_token_ids=list(range(block + 1)),
        sampling_params=SamplingParams(max_tokens=1),
        pooling_params=None,
        block_hasher=get_request_block_hasher(unit, sha256),
    )
    manager = KVCacheManager(alloc, 131072, block, unit, max_in_flight_tokens=4096)
    if manager.allocate_slots(request, block + 1) is None:
        raise ValueError("Native allocation failed")
    tables = manager.get_blocks(request.request_id).get_block_ids()
    coord = MooncakeStoreCoordinator(groups, block, unit, enable_partial_hash_hits=True)
    ranks = [
        MetadataRank(store, groups, geo, coord, r, 1 + max(map(max, tables)), "Kimi-K3", "step3-k3")
        for r in range(8)
    ]

    def lookup():
        return ranks[0].worker.lookup(block + 1, request.block_hashes).hit_length

    def write_group(rank, gid):
        db = ranks[rank].worker.token_dbs[gid]
        (shard,) = db.store_layout.local_shard_ids
        for start in range(0, block, db.chunk_size):
            end = start + db.chunk_size
            key = db.store_layout.key_for(shard, request.block_hashes[end // unit - 1])
            _, sizes, _ = db.store_layout.prepare_values([(start, end)], tables[gid], [shard])
            if key in store.objects or store.put(
                key,
                sum(sizes[0]),
                {
                    "request": "step3",
                    "rank": rank,
                    "group": gid,
                    "boundary_tokens": end,
                    "job": None,
                },
            ):
                raise ValueError("Invalid controlled write")

    for gid in range(len(groups)):
        write_group(0, gid)
    hits = {"rank0_only": lookup()}
    for rank in range(1, 8):
        for gid, group in enumerate(plan["groups"]):
            if group["spec"] != "MLAAttentionSpec" and (rank, gid) != (7, 2):
                write_group(rank, gid)
    partial_count = len(store.objects)
    hits["missing_group2_rank7"] = lookup()
    write_group(7, 2)
    while clock.events:
        clock.advance(clock.events[0][0])
    store.drop_published_memory({"request": "probe-boundary"})
    hits["all_disk_only"] = lookup()
    meta = ReqMeta(
        "step3",
        block,
        tables,
        request.block_hashes,
        load_spec=LoadSpec(0, hits["all_disk_only"], True),
    )
    transfers = MetadataTransfers(ranks)
    metadata = MooncakeStoreConnectorMetadata({"step3"}, set())
    metadata.requests.append(meta)
    transfers.submit(metadata, set(), clock.now)
    premature_ack = bool(transfers.output().finished_recving)
    while clock.events:
        clock.advance(clock.events[0][0])

    with tarfile.open(reference) as archive:
        actual = json.load(archive.extractfile("femu-c1-probe01/probe/summary.json"))
    writes = [e for e in events if e["event"] == "put_complete"]
    reads = [e for e in events if e["event"] == "get_disk"]

    def rows(data, fields):
        return Counter(tuple(e[k] for k in fields) for e in data)

    keys = ("rank", "key", "bytes")
    checks = {
        "archived_passed": actual["status"] == "passed",
        "missing_one_of_36": partial_count == 35,
        "lookup": hits
        == actual["lookup_tokens"]
        == {"rank0_only": 0, "missing_group2_rank7": 0, "all_disk_only": 1536},
        "writes_by_rank_group_key_boundary_size": rows(writes, (*keys, "group", "boundary_tokens"))
        == rows(
            [dict(o, boundary_tokens=o["end_token"]) for o in actual["objects"]],
            (*keys, "group", "boundary_tokens"),
        ),
        "reads_by_rank_key_size_multiplicity": rows(reads, keys) == rows(actual["restores"], keys),
        "complete_disk_reads": store.counts["get_complete"] == len(reads) == 120
        and not store.counts["get_memory"],
        "all_disk_only": len(store.objects) == 36
        and all(o.disk and not o.memory for o in store.objects.values()),
        "all_ranks_received": not premature_ack
        and transfers.output().finished_recving == {"step3"},
        "two_stage_bytes": store.bytes["get_disk"]
        == store.bytes["disk_read_complete"]
        == store.bytes["h2d_complete"]
        == store.bytes["staging_release"],
        "bounded_restore_staging": store.clients[0].restore_staging_bytes == 0
        and 0 < store.clients[0].peak_restore_staging_bytes <= store.config.staging_bytes,
    }
    return {
        "status": "passed" if all(checks.values()) else "failed",
        "checks": checks,
        "reference_sha256": hashlib.sha256(reference.read_bytes()).hexdigest(),
        "lookup_tokens": hits,
        "unique_objects": len(writes),
        "unique_bytes": store.bytes["put_complete"],
        "rank_gets": len(reads),
        "restore_bytes": store.bytes["get_disk"],
        "restore_staging_capacity_bytes": store.config.staging_bytes,
        "restore_staging_peak_bytes": store.clients[0].peak_restore_staging_bytes,
        "staging_wait_objects": store.counts["staging_wait"],
        "load_complete_ns": next(e["time_ns"] for e in events if e["event"] == "load_complete"),
        "last_disk_complete_ns": max(
            e["time_ns"] for e in events if e["event"] == "disk_read_complete"
        ),
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--reference", type=Path, default=ROOT / "evidence/ssd-c1/femu-c1-probe01.tar.gz"
    )
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--events", type=Path, help="Also retain the raw ordered probe events")
    args = p.parse_args()
    os.environ["PYTHONHASHSEED"] = (
        "42"  # explicit seed passed to the native block hasher by this fixture
    )
    os.environ.setdefault("VLLM_SSM_CONV_STATE_LAYOUT", "SD")
    report = check(args.reference, args.events)
    with args.output.open("x") as stream:
        stream.write(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))
    raise SystemExit(0 if report["status"] == "passed" else 1)
