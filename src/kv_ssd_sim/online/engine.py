"""Real-time harness around the upstream vLLM scheduler and Store connector."""

from __future__ import annotations

import asyncio
import csv
import dataclasses
import hashlib
import importlib.metadata
import json
import math
import os
import time
from contextlib import nullcontext
from pathlib import Path
from unittest.mock import patch

from kv_ssd_sim.core.events import EventLog
from kv_ssd_sim.core.provenance import package_source_hashes
from kv_ssd_sim.online.native import HostFence, NativeRuntime
from kv_ssd_sim.online.runtime import mooncake_master
from kv_ssd_sim.online.timing import wait_until
from kv_ssd_sim.workload.tokens import SYNTHETIC_TOKEN
from kv_ssd_sim.workload.weka import load_weka, save_normalized


class ReplayEngine:
    def __init__(self, runtime, requests, events, *, prefill_ms, decode_ms, timing, cold):
        from vllm.utils.hashing import sha256
        from vllm.v1.core.kv_cache_utils import get_request_block_hasher, init_none_hash

        self.runtime = runtime
        self.events = events
        self.specs = requests
        self.prefill_ms, self.decode_ms = prefill_ms, decode_ms
        self.timing, self.cold = timing, cold
        self.pressure = None
        init_none_hash(sha256)
        hasher = get_request_block_hasher(runtime.hash_unit, sha256)
        self.requests = {}
        self.metrics = {}
        self.hasher = hasher
        for item in requests:
            self.add_request(item)
        self.preparation = runtime.prepare_payloads(requests)
        self.completed = {}
        self.admitted = set()
        self.step_id = 0
        self.origin = time.perf_counter()

    def add_request(self, item):
        from vllm.sampling_params import SamplingParams
        from vllm.v1.request import Request

        if item.request_id in self.requests:
            raise ValueError(f"Duplicate request id: {item.request_id}")
        limit = min(
            self.runtime.config.model_config.max_model_len,
            (self.runtime.num_blocks - 2) * self.runtime.block_size,
        )
        if len(item.prompt_token_ids) + item.output_tokens > limit:
            raise ValueError(
                f"{item.request_id}: current limit is {limit} total tokens; "
                "increase --max-model-len and --num-blocks or select a smaller trace slice"
            )
        params = SamplingParams(max_tokens=item.output_tokens, ignore_eos=True, temperature=0)
        params.update_from_generation_config({}, 2)
        self.requests[item.request_id] = Request(
            request_id=item.request_id,
            prompt_token_ids=list(item.prompt_token_ids),
            sampling_params=params,
            pooling_params=None,
            block_hasher=self.hasher,
        )
        self.metrics[item.request_id] = {
            "request_id": item.request_id,
            "source_model": item.source_model,
            "input_tokens": len(item.prompt_token_ids),
            "target_output_tokens": item.output_tokens,
            "planned_arrival_s": item.arrival_s,
            "admitted_s": None,
            "first_token_s": None,
            "finished_s": None,
            "output_tokens": 0,
            "external_load_tokens": 0,
            "compute_steps": 0,
            "compute_budget_ms": 0.0,
            "materialize_ms": 0.0,
            "compute_window_ms": 0.0,
            "deadline_overrun_ms": 0.0,
        }

    async def step(self):
        from vllm.v1.outputs import KVConnectorOutput, ModelRunnerOutput

        native = self.runtime
        scheduler = native.scheduler
        scheduled = scheduler.schedule()
        self.step_id += 1
        sid = self.step_id
        metadata = scheduled.kv_connector_metadata
        if metadata is None:
            raise RuntimeError("Native scheduler omitted connector metadata")
        native.worker_connector.bind_connector_metadata(metadata)
        for meta in metadata.requests:
            if meta.load_spec is not None and meta.load_spec.can_load:
                count = meta.load_spec.kvpool_cached_tokens - meta.load_spec.vllm_cached_tokens
                self.metrics[meta.req_id]["external_load_tokens"] += count
                self.events.emit("load_requested", request_id=meta.req_id, step=sid, tokens=count)

        # The scheduler advances num_computed_tokens in _update_after_schedule.
        ranges = {
            rid: (
                self.requests[rid].num_computed_tokens - count,
                self.requests[rid].num_computed_tokens,
            )
            for rid, count in scheduled.num_scheduled_tokens.items()
        }
        has_prefill = any(
            start < len(self.requests[rid].prompt_token_ids) for rid, (start, _) in ranges.items()
        )
        has_decode = any(
            start >= len(self.requests[rid].prompt_token_ids) for rid, (start, _) in ranges.items()
        )
        delay_ms = (
            max(self.prefill_ms, self.decode_ms)
            if has_prefill and has_decode
            else self.prefill_ms
            if has_prefill
            else self.decode_ms
            if ranges
            else 0
        )
        fence = HostFence()
        for worker in native.workers:
            worker.fence = fence
        native.worker_connector.start_load_kv(None)
        # Native store jobs retain native block references and wait on this fence.
        native.worker_connector.wait_for_save()
        started = time.perf_counter()
        deadline = started + delay_ms / 1000
        if ranges:
            self.events.emit(
                "compute_begin",
                step=sid,
                request_ids=list(ranges),
                scheduled_tokens=scheduled.num_scheduled_tokens,
                **(
                    {
                        "active_pool_blocks_per_rank": native.num_blocks
                        - scheduler.kv_cache_manager.block_pool.get_num_free_blocks(),
                        "prepared_digest_bytes": native.payloads.nbytes,
                    }
                    if len(native.workers) > 1
                    else {}
                ),
                planned_compute_ms=delay_ms,
                kind="mixed"
                if has_prefill and has_decode
                else "prefill"
                if has_prefill
                else "decode",
            )
        try:
            copy_started = time.perf_counter()
            if native.hybrid:
                native.hybrid.copy_blocks(scheduled.kv_cache_block_copies, self.events)
            for rid, (start, end) in ranges.items():
                native.materialize(self.requests[rid], start, end)
            materialize_ms = (time.perf_counter() - copy_started) * 1000
            if ranges:
                self.events.emit("kv_materialized", step=sid, materialize_ms=materialize_ms)
                # The native sender is already waiting on HostFence. Copying
                # may finish early, but visibility is gated by the deadline.
                await wait_until(deadline)
                self.events.emit("compute_timer_done", step=sid)
            finished = time.perf_counter()
            for rid in ranges:
                self.metrics[rid]["compute_steps"] += 1
                self.metrics[rid]["compute_budget_ms"] += delay_ms
                self.metrics[rid]["materialize_ms"] += materialize_ms
                self.metrics[rid]["compute_window_ms"] += (finished - started) * 1000
                self.metrics[rid]["deadline_overrun_ms"] += max(0, finished - deadline) * 1000
            if ranges:
                self.events.emit(
                    "compute_end",
                    step=sid,
                    materialize_ms=materialize_ms,
                    actual_compute_ms=(finished - started) * 1000,
                    deadline_overrun_ms=max(0, finished - deadline) * 1000,
                )
            fence.release()
        except BaseException as error:
            fence.fail(error)
            raise

        transfers = native.worker_connector.get_transfer_results(scheduled.finished_req_ids)
        for rid in transfers.finished_recving or []:
            self.events.emit("load_complete", request_id=rid, step=sid)
        failed = transfers.failed_recving or set()
        errors = native.worker_connector.get_block_ids_with_load_errors()
        if failed or errors:
            self.events.emit("load_failure", request_ids=list(failed), block_ids=list(errors))
            raise RuntimeError(
                "Native KV restore failed; inspect load_failure and store_get_end events"
            )
        output = ModelRunnerOutput(
            req_ids=list(ranges),
            sampled_token_ids=[
                [SYNTHETIC_TOKEN] if end >= len(self.requests[rid].prompt_token_ids) else []
                for rid, (_, end) in ranges.items()
            ],
            kv_connector_output=KVConnectorOutput(
                finished_sending=transfers.finished_sending,
                finished_recving=transfers.finished_recving,
                failed_recving=failed,
                invalid_block_ids=errors,
                kv_connector_worker_meta=native.worker_connector.build_connector_worker_meta(),
            ),
        )
        scheduler.update_from_output(scheduled, output)
        native.worker_connector.clear_connector_metadata()
        now = time.perf_counter() - self.origin
        for rid in self.admitted - self.completed.keys():
            request = self.requests[rid]
            count = len(request.output_token_ids)
            metric = self.metrics[rid]
            metric["output_tokens"] = count
            if count and metric["first_token_s"] is None:
                metric["first_token_s"] = now
                self.events.emit("first_token", request_id=rid, step=sid)
            if request.is_finished():
                if count != metric["target_output_tokens"]:
                    raise AssertionError(f"Wrong output length for {rid}: {count}")
                self.completed[rid] = now
                metric["finished_s"] = now
                self.events.emit("request_finished", request_id=rid, output_tokens=count)
        if not ranges:
            await asyncio.sleep(0.001)

    async def drain(self):
        deadline = time.perf_counter() + (300 if len(self.runtime.workers) > 1 else 15)
        # At least one empty step forwards the last finished_req_ids to the worker.
        for _ in range(3):
            await self.step()
        while time.perf_counter() < deadline:
            if not any(
                w.native.kv_send_thread and w.native.kv_send_thread.request_queue.unfinished_tasks
                for w in self.runtime.workers
            ):
                await self.step()  # Return completed store-job references to the scheduler.
                return
            await self.step()
        raise TimeoutError("Native Store jobs did not drain")

    async def run(self):
        pending = list(self.specs)
        last_cold_count = 0
        self.origin = time.perf_counter()
        deadline = self.origin + max(120, max((r.arrival_s for r in pending), default=0) + 120)
        while pending or len(self.completed) != len(self.admitted):
            if time.perf_counter() > deadline:
                raise TimeoutError("Replay made no sufficient progress before the deadline")
            if (
                self.cold
                and pending
                and self.admitted
                and len(self.completed) == len(self.admitted)
                and len(self.completed) != last_cold_count
            ):
                await self.drain()
                keys = await asyncio.to_thread(self.runtime.store.disk_barrier)
                await asyncio.to_thread(self.runtime.store.evict_memory_replicas, keys)
                # Reset ONLY the native local block manager. The connector-wide
                # reset calls remove_all and would invalidate the disk experiment.
                if not self.runtime.scheduler.kv_cache_manager.reset_prefix_cache():
                    raise RuntimeError("Local prefix cache still has blocks in use")
                self.events.emit("local_prefix_cache_reset")
                if self.runtime.store.femu_device:
                    await asyncio.to_thread(self.runtime.store.femu_device.clear_guest_page_cache)
                    self.events.emit("guest_page_cache_dropped")
                last_cold_count = len(self.completed)
            now = time.perf_counter() - self.origin
            for item in list(pending):
                if self.cold and len(self.admitted) != len(self.completed):
                    break
                if self.timing == "causal" and item.dependencies:
                    if not all(dep in self.completed for dep in item.dependencies):
                        continue
                    release_at = (
                        max(self.completed[dep] for dep in item.dependencies) + item.think_time_s
                    )
                else:
                    release_at = item.arrival_s
                if now < release_at:
                    continue
                if self.pressure:
                    self.pressure.before_admission(item.request_id)
                self.runtime.scheduler.add_request(self.requests[item.request_id])
                self.admitted.add(item.request_id)
                self.metrics[item.request_id]["admitted_s"] = now
                self.events.emit(
                    "request_admitted",
                    request_id=item.request_id,
                    input_tokens=len(item.prompt_token_ids),
                    dependencies=list(item.dependencies),
                    timing=self.timing,
                    release_at_s=release_at,
                    admission_lateness_s=now - release_at,
                )
                pending.remove(item)
            await self.step()
        await self.drain()
        if self.runtime.store.errors:
            raise AssertionError("; ".join(self.runtime.store.errors))
        rows = list(self.metrics.values())
        for row in rows:
            row["ttft_ms"] = (row["first_token_s"] - row["admitted_s"]) * 1000
            row["latency_ms"] = (row["finished_s"] - row["admitted_s"]) * 1000
        return rows


def run_replay(
    input_path: Path,
    output: Path,
    *,
    read_delay_ms=0.0,
    prefill_ms=10.0,
    decode_ms=2.0,
    timing="causal",
    cold_between_turns=False,
    max_model_len=8192,
    num_blocks=128,
    hybrid_config_path=None,
    femu_mount=None,
    pressure_config=None,
):
    for name, value in {
        "read_delay_ms": read_delay_ms,
        "prefill_ms": prefill_ms,
        "decode_ms": decode_ms,
    }.items():
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be finite and non-negative")
    if timing not in {"causal", "fixed"}:
        raise ValueError("Unknown replay timing mode")
    if not 128 <= max_model_len <= 1048576 or not 4 <= num_blocks <= 32768:
        raise ValueError("Require max_model_len in [128, 1048576], num_blocks in [4, 32768]")
    from kv_ssd_sim.models.hybrid import HybridConfig

    hybrid_config = HybridConfig.load(hybrid_config_path) if hybrid_config_path else None
    requests = load_weka(input_path)
    if not requests:
        raise ValueError("No requests in trace")
    # 2 layers * K/V * 2 heads * 32 elements * FP16 = 512 bytes/token.
    # Cover two receiver batches up to the physical KV pool, including block
    # allocator rounding. Long-prefix restores otherwise overflow Mooncake's
    # native offload staging arena before any disk transfer can finish.
    pool_bytes = (
        num_blocks
        * max(hybrid_config.layers.count(k) for k in ("mla", "kda"))
        * hybrid_config.page_bytes
        if hybrid_config
        else num_blocks * 64 * 512
    )
    staging_bytes = max(8 * 1024 * 1024, pool_bytes * 2)
    disk_limit_bytes = max(256 * 1024 * 1024, pool_bytes * 4)
    pressure = None
    segment_bytes = max(32 * 1024**2, max_model_len * 512 * 2)
    if pressure_config:
        from kv_ssd_sim.online.pressure import PressureConfig

        pressure = PressureConfig.load(pressure_config)
        if cold_between_turns or read_delay_ms or not hybrid_config or timing != "causal":
            raise ValueError(
                "Capacity profile needs causal hybrid replay without artificial cooling or delay"
            )
        roles = {
            pressure.producer,
            pressure.restore,
            pressure.incomplete_producer,
            pressure.recompute,
        }
        if not roles <= {r.request_id for r in requests}:
            raise ValueError("Pressure experiment roles are absent from input trace")
        segment_bytes, staging_bytes, disk_limit_bytes = (
            pressure.segment_bytes,
            pressure.staging_bytes,
            pressure.disk_bytes,
        )
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    disk = output / "disk"
    femu_device = None
    if femu_mount is not None:
        from kv_ssd_sim.devices.femu.device import FEMUDevice

        if read_delay_ms != 0 or not (cold_between_turns or pressure):
            raise ValueError("FEMU verification requires cold-between-turns and no injected delay")
        femu_device = FEMUDevice(femu_mount)
        actual_disk = femu_device.mount / output.name
        actual_disk.mkdir(exist_ok=False)
        disk.symlink_to(actual_disk, target_is_directory=True)
    else:
        disk.mkdir()
    save_normalized(requests, output / "normalized.jsonl")
    manifest = {
        "input": str(input_path.resolve()),
        "input_sha256": hashlib.sha256(input_path.read_bytes()).hexdigest(),
        "harness_source_sha256": package_source_hashes(),
        "versions": {
            name: importlib.metadata.version(name)
            for name in ("vllm", "torch", "mooncake-transfer-engine-non-cuda", "kv-ssd-sim")
        },
        "read_delay_ms": read_delay_ms,
        "prefill_ms": prefill_ms,
        "decode_ms": decode_ms,
        "mixed_step_ms": max(prefill_ms, decode_ms),
        "timing": timing,
        "cold_between_turns": cold_between_turns,
        "max_model_len": max_model_len,
        "num_blocks": num_blocks,
        "hybrid_config": dataclasses.asdict(hybrid_config) if hybrid_config else None,
        "cpu_kv_pool_bytes": pool_bytes,
        "offload_staging_bytes": staging_bytes,
        "disk_limit_bytes": disk_limit_bytes,
        "compute_emulation": {
            "payload": (
                "MLA latent bytes and KDA boundary snapshots from prefix digests"
                if hybrid_config
                else "precomputed compact digests, byte-compatible with v1"
            ),
            "timer": "absolute step deadline including scheduled-slot copying",
            "event_flush_every": 64,
            "startup_package_mapping_snapshot": os.environ.get("KV_SSD_METADATA_CACHE"),
            "store_checksum": "full SHA256, zero-copy host buffer view",
        },
        "fidelity": {
            "native_vllm_scheduler": True,
            "native_kv_cache_manager": True,
            "native_mooncake_store_connector": True,
            "native_mooncake_disk_offload": True,
            "cpu_save_event_adapter": True,
            "explicit_native_offload_setup_adapter": True,
            "model_forward": False,
            "femu": femu_device is not None,
            "nand_ftl_model": femu_device is not None,
            "http_serving_stack": False,
            "mla_kda": hybrid_config is not None,
            "read_delay_injection": "disabled"
            if femu_device
            else "native receiver thread, per GET batch",
        },
    }
    if pressure:
        manifest["pressure_config"] = dataclasses.asdict(pressure)
        manifest["offload_use_uring"] = femu_device is not None
    manifest["storage_device"] = femu_device.describe() if femu_device else None
    (output / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    os.environ.update(
        {
            "MOONCAKE_OFFLOAD_FILE_STORAGE_PATH": str(disk.resolve()),
            "MOONCAKE_OFFLOAD_FLUSH_PARTIAL_BUCKETS": "1",
            "MOONCAKE_OFFLOAD_BUCKET_DIRECT_WRITE": "aligned",
            "MOONCAKE_OFFLOAD_LOCAL_BUFFER_SIZE_BYTES": str(staging_bytes),
            "MOONCAKE_OFFLOAD_TOTAL_SIZE_LIMIT_BYTES": str(disk_limit_bytes),
            "MOONCAKE_OFFLOAD_HEARTBEAT_INTERVAL_SECONDS": "1",
            "VLLM_MOONCAKE_LOAD_RECV_THREADS": "2",
        }
    )
    io_mode = (
        patch.dict(os.environ, {"MOONCAKE_OFFLOAD_USE_URING": "1" if femu_device else "0"})
        if pressure
        else nullcontext()
    )
    with (
        io_mode,
        EventLog(output / "events.jsonl") as events,
        mooncake_master(output, pressure=pressure) as master,
    ):
        events.emit("clock_sync", monotonic_origin_ns=events._start)
        store_config = {
            "mode": "embedded",
            "metadata_server": "P2PHANDSHAKE",
            "master_server_address": master,
            "global_segment_size": segment_bytes,
            "local_buffer_size": staging_bytes,
            "protocol": "tcp",
            "device_name": "",
            "enable_offload": True,
        }
        config_path = output / "mooncake.json"
        config_path.write_text(json.dumps(store_config, indent=2) + "\n")
        os.environ["MOONCAKE_CONFIG_PATH"] = str(config_path)
        runtime = NativeRuntime(
            output,
            events,
            read_delay_ms=read_delay_ms,
            max_model_len=max_model_len,
            num_blocks=num_blocks,
            hybrid_config=hybrid_config,
        )
        runtime.store.femu_device = femu_device
        try:
            engine = ReplayEngine(
                runtime,
                requests,
                events,
                prefill_ms=prefill_ms,
                decode_ms=decode_ms,
                timing=timing,
                cold=cold_between_turns,
            )
            if pressure:
                from kv_ssd_sim.online.pressure import PressureObserver

                engine.pressure = PressureObserver(runtime, pressure, events)
            rows = asyncio.run(engine.run())
            required = engine.pressure.published[pressure.producer] if pressure else None
            disk_keys = runtime.store.disk_barrier(required)
            if pressure:
                events.emit(
                    "final_replica_inventory",
                    tiers=engine.pressure.tiers(runtime.store.expected),
                    required_disk_keys=sorted(required),
                )
            report = {
                "status": "passed",
                "requests": len(rows),
                "restore_batches": runtime.store.restore_batches,
                "verified_disk_objects": len(disk_keys),
                "checksum_errors": runtime.store.errors,
                "read_delay_ms": read_delay_ms,
                "request_metrics": rows,
                "payload_preparation": engine.preparation,
                "hybrid": runtime.hybrid.describe() if runtime.hybrid else None,
                "femu_verified": femu_device is not None,
                "femu_evidence": femu_device.verify() if femu_device else None,
            }
        finally:
            runtime.close()
    with (output / "requests.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    return report
