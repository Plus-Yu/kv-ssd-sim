"""K3 storage geometry and logical TP8 workers for a single CPU scheduler."""

import copy
import dataclasses
import json
import math
import os
import re
import resource
import threading
import time
import urllib.request
from collections import defaultdict
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

import numpy as np

from .adapters.vllm import metadata_rank_worker
from .deployment import native_deployment, require
from .runtime import PROJECT_ROOT

# Preserve the original metadata planner import used by earlier experiments.
__all__ = [
    "CapacitySamples",
    "K3Buffers",
    "K3Profile",
    "RankConnectors",
    "logical_rank",
    "metadata_rank_worker",
]


@dataclasses.dataclass(frozen=True)
class K3Profile:
    profile: str
    segment_mib: int
    staging_mib: int
    quota_mib: int

    @classmethod
    def load(cls, values):
        cfg = cls(**values)
        require(
            cfg.profile == "k3-tp8"
            and all(
                type(v) is int and v > 0 for v in (cfg.segment_mib, cfg.staging_mib, cfg.quota_mib)
            ),
            "Invalid K3 online budgets",
        )
        return cfg


class K3Buffers:
    def __init__(self, num_blocks):
        import torch

        folder = PROJECT_ROOT / "configs/kimi-k3-b300-step2"
        plan, allocation = native_deployment(folder)
        require(
            json.loads(json.dumps(plan)) == json.loads((folder / "resolved.json").read_text()),
            "Native K3 geometry differs from the frozen plan",
        )
        self.groups = allocation.kv_cache_groups
        self.geometry = plan["geometry"]
        self.names = [layer["name"] for layer in plan["layers"]]
        self.index = {name: i for i, name in enumerate(self.names)}
        self.block = self.geometry["manager_block_tokens"]
        self.unit = self.geometry["hash_tokens"]
        page = self.geometry["common_page_bytes"]
        self.rank_pools, self.rank_caches = [], []
        self.initialized = {}
        for rank in range(8):
            # Fresh anonymous, zero-backed pages. Actual Store views retain full strides.
            pools = [np.zeros((num_blocks, page), dtype=np.uint8) for _ in range(24)]
            caches = {}
            for gid, group in enumerate(self.groups):
                width = group.kv_cache_spec.real_page_size_bytes
                for i, name in enumerate(group.layer_names):
                    raw = torch.from_numpy(pools[i])[:, :width]
                    caches[name] = raw.view(num_blocks, 1, self.block if gid == 3 else 1, -1)
            self.rank_pools.append(pools)
            self.rank_caches.append(caches)
        self.caches = self.rank_caches[0]

    def describe(self):
        return {
            "geometry": self.geometry,
            "logical_ranks": 8,
            "pool_bytes_per_rank": sum(p.nbytes for p in self.rank_pools[0]),
            "layers": len(self.names),
        }

    def copy_blocks(self, copies, events):
        for c in copies or []:
            for pools in self.rank_pools:
                for pool in pools:
                    pool[c.dst_block_id] = pool[c.src_block_id]
            self.initialized[c.dst_block_id] = self.initialized.get(c.src_block_id)
            events.emit("kv_block_copied", source=c.src_block_id, destination=c.dst_block_id)

    def materialize(self, runtime, request, start, end):
        from vllm.model_executor.layers.mamba.checkpoint import compute_mamba_prefill_checkpoints

        tables = runtime.scheduler.kv_cache_manager.get_blocks(request.request_id).get_block_ids()
        patterns = runtime.payloads.patterns[request.request_id]
        offsets, cols = compute_mamba_prefill_checkpoints(
            [end], [end - start], self.unit, self.block, 16, False, self.unit
        )
        for gid, group in enumerate(self.groups):
            width = group.kv_cache_spec.real_page_size_bytes
            if gid == 3:
                pos = start
                while pos < end:
                    col, offset = divmod(pos, self.block)
                    stop = min(end, (col + 1) * self.block)
                    bid = tables[gid][col]
                    require(bid > 0, "Null MLA page")
                    self.initialized[bid] = gid
                    for i, name in enumerate(group.layer_names):
                        stamp = patterns[self.index[name], pos:stop, None, :]
                        for pools in self.rank_pools:
                            pools[i][bid, :width].reshape(self.block, -1, 32)[
                                offset : offset + stop - pos
                            ] = stamp
                    pos = stop
                continue
            slots = [(math.ceil(end / self.block) - 1, end, "running")]
            if offsets[0] and cols[0] >= 0 and tables[gid][cols[0]]:
                slots.append((cols[0], start + offsets[0], "checkpoint"))
            for col, boundary, role in slots:
                bid = tables[gid][col]
                require(bid > 0, "Null KDA page")
                for rank, pools in enumerate(self.rank_pools):
                    for i, name in enumerate(group.layer_names):
                        target = pools[i][bid, :width]
                        # Constant body per layer/shard and prefix-dependent state header.
                        if self.initialized.get(bid) != gid:
                            target.fill((self.index[name] * 8 + rank) % 251)
                        target[:32] = patterns[self.index[name], boundary - 1]
                        target[32] = rank
                self.initialized[bid] = gid
                if role == "checkpoint":
                    runtime.events.emit(
                        "k3_checkpoint",
                        request_id=request.request_id,
                        group=gid,
                        block=bid,
                        boundary_tokens=boundary,
                        logical_ranks=8,
                    )


class RankConnectors:
    """Fan out one scheduler decision and aggregate native per-rank acknowledgements."""

    def __init__(self, connectors):
        self.connectors = connectors
        self.pending = defaultdict(set)

    def bind_connector_metadata(self, metadata):
        for c in self.connectors:
            local = copy.copy(metadata)
            local.requests = [copy.copy(row) for row in metadata.requests]
            c.bind_connector_metadata(local)

    def start_load_kv(self, context):
        for c in self.connectors:
            c.start_load_kv(context)

    def wait_for_save(self):
        for c in self.connectors:
            c.wait_for_save()

    def get_transfer_results(self, finished):
        results = [c.get_transfer_results(finished) for c in self.connectors]
        result = copy.copy(results[0])
        for field in ("finished_sending", "finished_recving"):
            done = set()
            for rank, r in enumerate(results):
                for req in getattr(r, field) or ():
                    ranks = self.pending[field, req]
                    ranks.add(rank)
                    if len(ranks) == len(results):
                        done.add(req)
                        del self.pending[field, req]
            setattr(result, field, done)
        result.failed_recving = set().union(*(r.failed_recving or set() for r in results))
        return result

    def get_block_ids_with_load_errors(self):
        return set().union(*(c.get_block_ids_with_load_errors() for c in self.connectors))

    def build_connector_worker_meta(self):
        combined = None
        for c in self.connectors:
            meta = c.build_connector_worker_meta()
            if meta is not None:
                combined = meta if combined is None else combined.aggregate(meta)
        return combined

    def clear_connector_metadata(self):
        for c in self.connectors:
            c.clear_connector_metadata()

    def shutdown(self):
        # One embedded Store client is shared by the eight logical workers.
        for c in self.connectors[1:]:
            c.connector_worker.native.store = None
        for c in self.connectors:
            c.shutdown()


def logical_rank(config, worker_module, rank, store):
    stack = ExitStack()
    stack.enter_context(patch.object(config.parallel_config, "rank", rank))
    stack.enter_context(
        patch.object(worker_module, "get_tensor_model_parallel_rank", return_value=rank)
    )
    stack.enter_context(
        patch.object(worker_module, "get_tensor_model_parallel_world_size", return_value=8)
    )
    if store is not None:
        # Worker construction imports the Store factory locally.
        import mooncake.store

        stack.enter_context(
            patch.object(mooncake.store, "MooncakeDistributedStore", return_value=store)
        )
    return stack


class CapacitySamples:
    """Sample the actual guest filesystem, host-memory headroom and Store segment."""

    def __init__(self, output, mount):
        self.output = output
        self.mount = mount
        cmd = json.loads((output / "mooncake-master.json").read_text())["command"]
        port = next(a.split("=")[1] for a in cmd if a.startswith("--metrics_port="))
        self.url = f"http://127.0.0.1:{port}/metrics"
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self.run, daemon=True)

    def run(self):
        with (self.output / "capacity.jsonl").open("w", buffering=1) as stream:
            while True:
                disk = os.statvfs(self.mount)
                mem = Path("/proc/meminfo").read_text()
                row = {
                    "time_ns": time.monotonic_ns(),
                    "fs_used_bytes": (disk.f_blocks - disk.f_bfree) * disk.f_frsize,
                    "mem_available_bytes": int(re.search(r"MemAvailable:\s+(\d+)", mem)[1]) * 1024,
                    "server_maxrss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                    * 1024,
                }
                try:
                    with urllib.request.urlopen(self.url, timeout=1) as response:
                        metrics = response.read().decode()
                    gauge = re.search(
                        r"^master_allocated_bytes(?:\{[^}]*\})? ([-+0-9.eE]+)", metrics, re.M
                    )
                    if gauge:
                        row["store_segment_used_bytes"] = int(float(gauge[1]))
                except OSError:
                    pass
                stream.write(json.dumps(row) + "\n")
                if self.stop.wait(1):
                    return

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *exc):
        self.stop.set()
        self.thread.join()
