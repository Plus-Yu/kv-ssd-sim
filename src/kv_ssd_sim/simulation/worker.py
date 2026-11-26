"""Drive the pinned Mooncake connector's save, lookup and load code on tensor metadata."""

from __future__ import annotations

import math
import threading
from collections import Counter, deque
from contextlib import contextmanager
from dataclasses import dataclass, field
from itertools import count
from unittest.mock import patch

import torch
from vllm.config import set_current_vllm_config
from vllm.distributed.kv_transfer.kv_connector.v1.mooncake_store import scheduler as mc_scheduler
from vllm.distributed.kv_transfer.kv_connector.v1.mooncake_store import worker as mc_worker
from vllm.distributed.kv_transfer.kv_connector.v1.mooncake_store.coordinator import (
    MooncakeStoreCoordinator,
)
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.core.single_type_kv_cache_manager import register_all_kvcache_specs
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.outputs import KVConnectorOutput
from vllm.v1.structured_output import StructuredOutputManager

from kv_ssd_sim.adapters.vllm import metadata_rank_worker

from ..adapters.vllm import scheduler_config
from .storage.device import MIB


@dataclass
class SaveBatch:
    writes: list = field(default_factory=list)
    acknowledgements: dict = field(default_factory=dict)


def after_all(pending, done):
    """One acknowledgement for each issued descriptor (or each participating rank)."""

    def acknowledge():
        nonlocal pending
        pending -= 1
        if pending == 0:
            done()
        elif pending < 0:
            raise RuntimeError("Duplicate restore acknowledgement")

    return acknowledge


class RankPort:
    """Descriptor-only Store API. Addresses returned by meta tensors are never dereferenced."""

    def __init__(self, store, rank):
        self.store = store
        self.rank = rank
        self.done_at = 0
        self.saving = None
        self.load_batches = None
        self.load_id = None

    def bind(self, meta):
        self.meta = meta
        self.boundaries = {h.hex(): (i + 1) * self.unit for i, h in enumerate(meta.block_hashes)}
        self.done_at = self.store.clock.now

    def origin(self, key):
        _, group, digest = key.rsplit("@", 2)
        return {
            "request": self.meta.req_id,
            "rank": self.rank,
            "job": self.meta.store_job_id,
            "group": int(group.removeprefix("group:")),
            "boundary_tokens": self.boundaries[digest],
        }

    def batch_is_exist(self, keys):
        return self.store.batch_is_exist(keys)

    def batch_put_from_multi_buffers(self, keys, addresses, sizes, config):
        result = []
        for key, parts in zip(keys, sizes, strict=True):
            origin = self.origin(key)
            code, obj = self.store.begin_put(key, sum(parts), origin)
            if obj is not None:
                self.saving.writes.append((obj, origin))
            result.append(code)
        return result

    def batch_get_into_multi_buffers(self, keys, addresses, sizes):
        batch = [(k, sum(s), self.origin(k)) for k, s in zip(keys, sizes, strict=True)]
        if self.load_batches is None:
            for key, size, origin in batch:
                self.done_at = max(self.done_at, self.store.get(key, size, origin))
        elif batch:
            for _, _, origin in batch:
                origin["load_id"] = self.load_id
                origin["load_batch"] = len(self.load_batches)
            self.load_batches.append(batch)
        return [size for _, size, _ in batch]


class MetadataRank:
    def __init__(self, store, groups, geometry, coord, rank, blocks, model_name, prefix):
        self.store = store
        self.clock = store.clock
        self.port = RankPort(store, rank)
        block, unit = geometry["manager_block_tokens"], geometry["hash_tokens"]
        self.port.unit = unit
        w = self.worker = metadata_rank_worker(groups, unit, rank, model_name, prefix)
        w.store, w.coord, w._capacity_only, w._selective_lease = self.port, coord, False, None
        w.kv_connector_stats = mc_worker.MooncakeStoreConnectorStats()
        w._kv_connector_stats_lock = threading.Lock()
        w.finished_store_req = set()
        # Shape, dtype and stride match K3Buffers. Meta tensors have no payload storage.
        page = geometry["common_page_bytes"]
        for group, db in zip(groups, w.token_dbs, strict=True):
            width = group.kv_cache_spec.real_page_size_bytes
            states = block if db.store_layout.store_format == "attention_head_major" else 1
            tensor = torch.empty((blocks, page), dtype=torch.uint8, device="meta")
            view = tensor[:, :width].view(blocks, 1, states, -1)
            db.store_layout.register_kv_caches([view] * len(group.layer_names), blocks)

        w.kv_send_thread = self.sender = mc_worker.KVCacheStoreSendingThread(
            self.port,
            coord,
            w.token_dbs,
            block,
            rank,
            w._group_tp_replication_factors,
            "kv_both",
            threading.Event(),
        )
        self.receiver = mc_worker.KVCacheStoreRecvingThread(
            self.port,
            coord,
            w.token_dbs,
            block,
            rank,
            threading.Event(),
            disk_offload_buffer_budget_bytes=store.config.staging_bytes,
        )
        self.save_tail = 0
        self.pending_saves = deque()
        self.completed_saves = Counter()
        self.pending_loads = deque()

    def enqueue_save(self, meta, compute_end):
        self.sender.add_request(meta)
        if self.store.config.put_mib_s is not None:
            self.pending_saves.append((meta, compute_end))
            if len(self.pending_saves) == 1:
                self._start_next_save()
            return
        start = max(compute_end, self.save_tail)
        self.save_tail = start + round(self.store.config.save_ms * 1e6)
        self.store.emit(
            "save_queued",
            request=meta.req_id,
            rank=self.port.rank,
            job=meta.store_job_id,
            start_ns=start,
            complete_ns=self.save_tail,
        )
        batch = SaveBatch()
        self.clock.at(
            start + round(self.store.config.put_allocation_ms * 1e6), self._begin_save, meta, batch
        )
        self.clock.at(self.save_tail, self._finish_save, meta, batch)

    def _start_next_save(self):
        meta, fence = self.pending_saves[0]
        start = max(self.clock.now, fence)
        allocation = start + round(self.store.config.put_allocation_ms * 1e6)
        self.clock.at(allocation, self._allocate_timed_save, meta, start)

    def _allocate_timed_save(self, meta, start):
        batch = SaveBatch()
        self._begin_save(meta, batch)
        size = sum(obj.size for obj, _ in batch.writes)
        config = self.store.config
        ready = start + round(config.save_ms * 1e6)
        if config.host_copy_lanes and ready > self.clock.now:
            self.clock.at(ready, self._schedule_copy, meta, batch, start, ready, size)
        else:
            self._schedule_copy(meta, batch, start, ready, size)

    def _schedule_copy(self, meta, batch, start, ready, size):
        config = self.store.config
        details = {}
        if config.host_copy_lanes:
            copy_start, end = self.store.restores.ports.reserve_copy(size, config.put_mib_s)
            details.update(copy_ready_ns=ready, copy_start_ns=copy_start)
        else:
            end = ready + math.ceil(size * 1e9 / (config.put_mib_s * MIB))
        self.store.emit(
            "save_queued",
            request=meta.req_id,
            rank=self.port.rank,
            job=meta.store_job_id,
            start_ns=start,
            complete_ns=end,
            copy_bytes=size,
            **details,
        )
        self.clock.at(end, self._complete_timed_save, meta, batch)

    def _complete_timed_save(self, meta, batch):
        self._finish_save(meta, batch)
        self.pending_saves.popleft()
        if self.pending_saves:
            self._start_next_save()

    def _begin_save(self, meta, batch):
        if self.sender.request_queue.get_nowait() is not meta:
            raise ValueError("Per-rank save queue reordered")
        self.port.bind(meta)
        self.port.saving = batch
        try:
            self.sender._handle_request(meta)
            batch.acknowledgements = self.sender.take_completed_saves()
        finally:
            self.port.saving = None

    def _finish_save(self, meta, batch):
        for obj, origin in batch.writes:
            self.store.finish_put(obj, origin)
        self.completed_saves.update(batch.acknowledgements)
        self.store.emit("save_ack", request=meta.req_id, rank=self.port.rank, job=meta.store_job_id)

    def _describe_load(self, meta):
        meta.load_spec.token_len = meta.load_spec.kvpool_cached_tokens
        self.port.bind(meta)
        self.receiver.add_request(meta)
        self.receiver._handle_request(self.receiver.request_queue.get_nowait())
        if (
            self.receiver.get_and_clear_finished_requests() != {meta.req_id}
            or self.receiver.get_and_clear_block_ids_with_load_errors()
            or self.receiver.get_and_clear_failed_requests()
        ):
            raise ValueError(f"Native load failed for {meta.req_id}")

    def load(self, meta, done=None, load_id=None):
        if self.store.config.h2d_mib_s is None:
            self._describe_load(meta)
            return self.port.done_at
        self.pending_loads.append((meta, done, load_id))
        if len(self.pending_loads) == 1:
            self._start_next_load()

    def _start_next_load(self):
        meta, done, load_id = self.pending_loads[0]
        batches = deque()
        self.port.load_batches, self.port.load_id = batches, load_id
        try:
            self._describe_load(meta)
        finally:
            self.port.load_batches = self.port.load_id = None

        def next_batch():
            if batches:
                batch = batches.popleft()
                acknowledged = after_all(len(batch), next_batch)
                for key, size, origin in batch:
                    self.store.get(key, size, origin, on_complete=acknowledged)
                return
            self.store.emit(
                "rank_load_complete", request=meta.req_id, rank=self.port.rank, load_id=load_id
            )
            self.pending_loads.popleft()
            if self.pending_loads:
                self.clock.at(self.clock.now, self._start_next_load)
            if done:
                done()

        next_batch()


class MetadataTransfers:
    def __init__(self, ranks):
        self.ranks = ranks
        self.receive_acks = set()
        self.load_ids = count(1)

    def _load(self, meta):
        store = self.ranks[0].store
        if store.config.h2d_mib_s is None:
            last = max(rank.load(meta) for rank in self.ranks)
            self.ranks[0].clock.at(last, self.receive_acks.add, meta.req_id)
            return
        load_id = next(self.load_ids)
        store.emit("load_start", request=meta.req_id, load_id=load_id, ranks=len(self.ranks))

        def complete():
            store.emit("load_complete", request=meta.req_id, load_id=load_id)
            self.receive_acks.add(meta.req_id)

        rank_done = after_all(len(self.ranks), complete)
        for rank in self.ranks:
            rank.load(meta, rank_done, load_id)

    def submit(self, metadata, finished, compute_end):
        for rank in self.ranks:
            rank.worker._close_ended_store_requests(finished, metadata)
        external = Counter()
        for meta in metadata.requests:
            if meta.can_save:
                for rank in self.ranks:
                    rank.enqueue_save(meta, compute_end)
            if meta.load_spec is not None and meta.load_spec.can_load:
                external[meta.req_id] += (
                    meta.load_spec.kvpool_cached_tokens - meta.load_spec.vllm_cached_tokens
                )
                self._load(meta)
        return external

    def output(self):
        counts = Counter()
        for rank in self.ranks:
            counts.update(rank.completed_saves)
            rank.completed_saves.clear()
        ready = self.receive_acks.copy()
        self.receive_acks.clear()
        return KVConnectorOutput(
            finished_recving=ready,
            finished_sending=set(),
            kv_connector_worker_meta=mc_worker.MooncakeStoreWorkerMetadata(
                completed_saves=dict(counts)
            ),
        )


class InProcessLookup:
    """Synchronous scheduler lookup with the native rank-0 cross-rank completeness checks."""

    def __init__(self, reader):
        self.reader = reader

    def lookup(self, request_id, num_tokens, block_hashes, *, non_block):
        if non_block:
            raise ValueError("Expected synchronous native prefix lookup")
        result = self.reader.lookup(num_tokens, block_hashes)
        self.reader.store.store.emit(
            "prefix_lookup",
            request=request_id,
            lookup_tokens=num_tokens,
            hit_tokens=result.hit_length,
            tail_key_boundaries=[(b.group_id, b.num_tokens) for b in result.tail_key_boundaries],
        )
        return result

    def discard(self, request_id):
        pass  # No RPC result cache.


@contextmanager
def metadata_scheduler(
    store, plan, groups, blocks, max_length, *, native_retention=False, **scheduler_options
):
    geometry = plan["geometry"]
    block, unit = geometry["manager_block_tokens"], geometry["hash_tokens"]
    config = scheduler_config(max_length, block, k3=True, **scheduler_options)
    cache = config.cache_config
    cache.kv_cache_layout = "LBHNC"
    cache.prefix_match_unit = cache.hash_block_size = cache.cache_hit_alignment_tokens = unit
    cache.num_gpu_blocks = blocks
    config.parallel_config.world_size = 8
    allocation = KVCacheConfig(
        num_blocks=blocks,
        kv_cache_groups=groups,
        kv_cache_tensors=[],
        # Mirror native KV planning before the connector selects decode checkpoints.
        prefix_cache_retention_interval=cache.prefix_cache_retention_interval
        if native_retention
        else None,
    )
    with set_current_vllm_config(config):
        register_all_kvcache_specs(config)
        coord = MooncakeStoreCoordinator.from_kv_cache_config(allocation, config, block, unit)
        ranks = [
            MetadataRank(store, groups, geometry, coord, rank, blocks, "model", "k3-online-tp8")
            for rank in range(8)
        ]
        with patch.object(
            mc_scheduler, "LookupKeyClient", return_value=InProcessLookup(ranks[0].worker)
        ):
            scheduler = Scheduler(
                vllm_config=config,
                kv_cache_config=allocation,
                block_size=block,
                hash_block_size=unit,
                log_stats=False,
                structured_output_manager=StructuredOutputManager(config),
            )
        yield scheduler, MetadataTransfers(ranks)
