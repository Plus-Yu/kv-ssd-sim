"""Run K3 object-level predictions on compiled WEKA sessions with no KV payload allocation."""

import argparse
import hashlib
import json
import math
import os
from collections import defaultdict, deque
from dataclasses import asdict
from importlib.metadata import version
from pathlib import Path

from check_staged_restores import audit as audit_restores

from kv_ssd_sim.object_store import MIB, Clock, ObjectStore, OffloadTimeline, StoreConfig
from kv_ssd_sim.object_timing import StepTiming
from kv_ssd_sim.object_traffic import describe_traffic, export_demand
from kv_ssd_sim.runtime import PROJECT_ROOT as ROOT

PUBLIC_PROFILES = {
    "reference": ROOT / "configs/kimi-k3-b300-large-ssd.json",
    "reference-volume-20261125": ROOT / "configs/kimi-k3-b300-logical-volume.json",
    "reference-single-ssd": ROOT / "configs/kimi-k3-b300-object-replay.json",
}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class Admissions:
    """Release an arrival after every predecessor edge and its own delay are satisfied."""

    def __init__(self, inputs, offsets, clock, limit):
        self.clock = clock
        self.rows = {}
        self.followers = defaultdict(list)
        self.gates = {}
        self.release_time = {}
        self.ready = deque()
        trace_ids = set()
        for path, offset in zip(inputs, offsets, strict=True):
            data = json.loads(path.read_bytes())
            if data["schema"] != 1 or data["trace_id"] in trace_ids:
                raise ValueError("Expected compiled inputs with distinct local trace IDs")
            trace_ids.add(data["trace_id"])
            for row in data["requests"]:
                rid = row["id"]
                if rid in self.rows or not row["tokens"] or row["output_tokens"] < 1:
                    raise ValueError(f"Duplicate or empty request {rid}")
                if len(row["tokens"]) + row["output_tokens"] > limit:
                    raise ValueError(f"{rid} exceeds configured context/pool limit {limit}")
                if any(type(t) is not int or not 4 <= t < 32000 for t in row["tokens"]):
                    raise ValueError("Invalid compiled synthetic token IDs")
                digest = hashlib.sha256(
                    b"".join(t.to_bytes(4, "big") for t in row["tokens"])
                ).hexdigest()
                if digest != row["prompt_sha256"]:
                    raise ValueError(f"Token digest mismatch for {rid}")
                self.rows[rid] = row
                edges = row["dependencies"]
                self.gates[rid] = len(edges)
                root_ms = row["root_arrival_ms"]
                if not math.isfinite(root_ms) or root_ms < 0:
                    raise ValueError("Invalid root arrival time")
                self.release_time[rid] = round((offset * 1000 + root_ms) * 1e6)
                for parent, marker, delay in edges:
                    if marker not in ("start", "finish") or not math.isfinite(delay) or delay < 0:
                        raise ValueError("Invalid causal edge")
                    self.followers[parent, marker].append((rid, round(delay * 1e6)))
                if not edges:
                    clock.at(self.release_time[rid], self.arrive, rid)
        if not self.rows or any(parent not in self.rows for parent, _ in self.followers):
            raise ValueError("Empty workload or unresolved predecessor")

    def arrive(self, rid):
        self.ready.append(rid)
        self.notify(rid, "start")

    def notify(self, parent, marker):
        for rid, delay in self.followers.pop((parent, marker), []):
            self.release_time[rid] = max(self.release_time[rid], self.clock.now + delay)
            self.gates[rid] -= 1
            if not self.gates[rid]:
                self.clock.at(self.release_time[rid], self.arrive, rid)


def simulate(admissions, scheduler, transfers, store, hash_unit, timing, record_steps):
    from vllm.sampling_params import SamplingParams
    from vllm.utils.hashing import sha256
    from vllm.v1.core.kv_cache_utils import get_request_block_hasher, init_none_hash
    from vllm.v1.outputs import ModelRunnerOutput
    from vllm.v1.request import Request

    from kv_ssd_sim.payload import SYNTHETIC_TOKEN

    init_none_hash(sha256)
    hasher = get_request_block_hasher(hash_unit, sha256)
    clock = store.clock
    live, metrics = {}, {}
    idle_steps = 0
    while clock.events or admissions.ready or scheduler.has_requests():
        clock.advance(clock.now)
        while admissions.ready:
            rid = admissions.ready.popleft()
            row = admissions.rows[rid]
            params = SamplingParams(max_tokens=row["output_tokens"], ignore_eos=True, temperature=0)
            params.update_from_generation_config({}, 2)
            request = Request(
                request_id=rid,
                prompt_token_ids=row["tokens"],
                sampling_params=params,
                pooling_params=None,
                block_hasher=hasher,
            )
            live[rid] = request
            metrics[rid] = {
                "id": rid,
                "session": row["session"],
                "source_index": row["source_index"],
                "arrival_ns": admissions.release_time[rid],
                "admitted_ns": clock.now,
                "prompt_tokens": len(row["tokens"]),
                "prompt_sha256": row["prompt_sha256"],
                "output_tokens": row["output_tokens"],
                "computed_prompt_tokens": 0,
                "external_load_tokens": 0,
                "local_reuse_tokens": None,
            }
            if row.get("measurement_anchor"):
                cache = scheduler.kv_cache_manager
                store.mark_measurement(
                    rid, cache.usage, len(cache.block_pool.cached_block_hash_to_block)
                )
            scheduler.add_request(request)
            store.emit("request_admitted", request=rid)
            clock.advance(clock.now)
        if scheduler.has_requests():
            scheduled = scheduler.schedule()
            ranges = {
                rid: (live[rid].num_computed_tokens - count, live[rid].num_computed_tokens)
                for rid, count in scheduled.num_scheduled_tokens.items()
            }
            fresh = sum(
                max(0, min(b, live[r].num_prompt_tokens) - a) for r, (a, b) in ranges.items()
            )
            decoding = sum(a >= live[r].num_prompt_tokens for r, (a, _) in ranges.items())
            duration = round(timing.duration_ms(fresh, len(ranges), decoding) * 1e6)
            end_ns = clock.now + duration
            if record_steps and ranges:
                store.emit(
                    "compute_step",
                    batch=len(ranges),
                    fresh_prompt_tokens=fresh,
                    decode_only_sequences=decoding,
                    duration_ns=duration,
                )
            loads = transfers.submit(
                scheduled.kv_connector_metadata, scheduled.finished_req_ids, end_ns
            )
            for rid, count in loads.items():
                metrics[rid]["external_load_tokens"] += count
            for rid, (a, b) in ranges.items():
                m = metrics[rid]
                if m["local_reuse_tokens"] is None:
                    m["local_reuse_tokens"] = a - m["external_load_tokens"]
                m["computed_prompt_tokens"] += max(0, min(b, m["prompt_tokens"]) - a)
            clock.advance(end_ns)
            replies = transfers.output()
            result = ModelRunnerOutput(
                req_ids=list(ranges),
                kv_connector_output=replies,
                sampled_token_ids=[
                    [SYNTHETIC_TOKEN] if b >= live[r].num_prompt_tokens else []
                    for r, (_, b) in ranges.items()
                ],
            )
            scheduler.update_from_output(scheduled, result)
            for rid in ranges:
                if not live[rid].is_finished():
                    continue
                request = live.pop(rid)
                m = metrics[rid]
                if request.num_output_tokens != m["output_tokens"] or (
                    not request.num_preemptions
                    and m["local_reuse_tokens"]
                    + m["external_load_tokens"]
                    + m["computed_prompt_tokens"]
                    != m["prompt_tokens"]
                ):
                    raise ValueError(f"Token accounting failed for {rid}")
                m.update(finished_ns=clock.now, preemptions=request.num_preemptions)
                store.emit("request_finished", request=rid)
                admissions.notify(rid, "finish")
            if (
                ranges
                or loads
                or scheduled.finished_req_ids
                or replies.finished_recving
                or replies.kv_connector_worker_meta.completed_saves
            ):
                idle_steps = 0
                continue
            idle_steps += 1
            if idle_steps == 1:
                continue  # a fence or deferred lookup can require one further native step
        if clock.events:
            clock.advance(clock.events[0][0])
            idle_steps = 0
        elif scheduler.has_requests() or live:
            raise ValueError(f"Scheduler stalled at {clock.now} ns; live requests: {list(live)}")
    if len(metrics) != len(admissions.rows) or any(
        "finished_ns" not in m for m in metrics.values()
    ):
        raise ValueError("Incomplete workload or cyclic dependencies")
    return list(metrics.values())


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("inputs", type=Path, nargs="+")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument(
        "--offset-s", type=float, action="append", help="One root offset per input (seconds)"
    )
    p.add_argument(
        "--profile", choices=("n15", *PUBLIC_PROFILES, "reference-20261124"), default="n15"
    )
    p.add_argument(
        "--volume-service",
        choices=("1x", "4x", "8x"),
        help="Scale the service rate of the same shared SSD FIFO",
    )
    p.add_argument(
        "--offload-timeline",
        type=Path,
        help="JSONL keyed by native client and bucket keys, with source, write start and completion times (ns)",
    )
    p.add_argument("--num-blocks", type=int)
    p.add_argument("--max-model-len", type=int)
    p.add_argument("--clients", type=int, choices=(1, 8))
    for budget in ("segment", "staging", "quota"):
        p.add_argument("--" + budget + "-mib", type=int, help="Override per-client capacity in MiB")
    p.add_argument(
        "--offload-interval-ms",
        "--offload-delay-ms",
        dest="offload_delay_ms",
        type=float,
        help="Virtual client's offload heartbeat interval",
    )
    p.add_argument("--high-watermark", type=float)
    p.add_argument("--eviction-ratio", type=float)
    p.add_argument("--disk-high-watermark", type=float)
    p.add_argument("--disk-low-watermark", type=float)
    p.add_argument(
        "--save-ms", type=float, help="Fixed per-rank save overhead, in addition to copy time"
    )
    p.add_argument("--put-allocation-ms", type=float)
    p.add_argument("--first-offload-ms", type=float)
    p.add_argument(
        "--service-profile", type=Path, help="Frozen timing fit from the n15 calibration window"
    )
    p.add_argument("--lease-ms", type=float)
    p.add_argument(
        "--device-mib-s",
        type=float,
        help="SSD write rate, also used for reads without a separate read rate",
    )
    p.add_argument("--device-read-mib-s", type=float)
    p.add_argument("--memory-mib-s", type=float)
    p.add_argument(
        "--restore-path",
        choices=("host-staged", "legacy"),
        help="Bounded file-owner staging and a destination H2D port, or archived GET timing",
    )
    p.add_argument("--h2d-mib-s", type=float, help="H2D bandwidth per destination rank (MiB/s)")
    p.add_argument("--put-mib-s", type=float, help="Per-rank rate for newly allocated PUT payloads")
    p.add_argument(
        "--prefill-ms", type=float, help="Use a fixed duration instead of a token-rate estimate"
    )
    p.add_argument("--decode-ms", type=float, help="Use a fixed duration for every decode batch")
    p.add_argument("--compute-scale", type=float, default=1, help="Multiply compute time")
    p.add_argument(
        "--ssd-time-scale", type=float, default=1, help="Multiply virtual SSD service time"
    )
    args = p.parse_args()
    if args.volume_service and args.profile not in ("reference", "reference-volume-20261125"):
        p.error("--volume-service requires a reference volume profile")
    fit = None
    if args.service_profile:
        fit = json.loads(args.service_profile.read_bytes())
        expected = {
            "prefill_ms",
            "decode_ms",
            "save_ms",
            "put_allocation_ms",
            "device_mib_s",
            "first_offload_ms",
        }
        if (
            args.profile != "n15"
            or not fit.get("calibration_only")
            or set(fit["flags"]) != expected
        ):
            p.error("Expected a complete n15 calibration profile")
        for field, value in fit["flags"].items():
            setattr(args, field, value)
    os.environ.setdefault("VLLM_SSM_CONV_STATE_LAYOUT", "SD")

    from kv_ssd_sim.deployment import native_deployment
    from kv_ssd_sim.object_worker import metadata_scheduler

    folder = ROOT / "configs/kimi-k3-b300-step2"
    plan, allocation = native_deployment(folder)
    if json.loads(json.dumps(plan)) != json.loads((folder / "resolved.json").read_bytes()):
        raise ValueError("Pinned native K3 geometry changed")
    reference = args.profile != "n15"
    profile_path = PUBLIC_PROFILES.get(args.profile)
    public = json.loads(profile_path.read_bytes()) if profile_path else None
    volume = public.get("volume") if public else None
    online = json.loads((ROOT / "configs/k3-c1-online.json").read_bytes())
    mc = plan["mooncake"]
    budgets = {
        "segment": mc["global_segment_size_bytes_per_client"],
        "staging": mc["disk_staging_size_bytes_per_client"],
        "quota": (volume or plan["ssd"])["offload_quota_bytes_per_client"],
    }
    for name in budgets:
        override = getattr(args, name + "_mib")
        if override is not None:
            budgets[name] = override * MIB
        elif not reference:
            budgets[name] = online[name + "_mib"] * MIB
    store_flags = (
        "offload_delay_ms",
        "device_mib_s",
        "device_read_mib_s",
        "memory_mib_s",
        "lease_ms",
        "save_ms",
        "high_watermark",
        "eviction_ratio",
        "put_allocation_ms",
        "first_offload_ms",
        "put_mib_s",
        "disk_high_watermark",
        "disk_low_watermark",
    )
    store_args = {}
    default_blocks = plan["ranks"][0]["native_pool_blocks"] if reference else 22
    default_len = 131072 if reference else 12288
    prefill, decode, decode8, prefill_rate = 10, 2, 2, None
    if public:
        default_len = public["max_model_len"]
        default_blocks = (
            int(public["kv_gib_per_rank"] * 2**30)
            // plan["geometry"]["shared_pool_bytes_per_block"]
        )
        b = public["bandwidth"]
        rate_multiplier = 1
        if volume:
            scenario = args.volume_service or volume["default_service"]
            rate_multiplier = volume["service_rate_multipliers"][scenario]
        store_args.update(
            public["store"],
            device_mib_s=b["ssd_write_MB_s"] * rate_multiplier * 1e6 / MIB,
            device_read_mib_s=b["ssd_read_MB_s"] * rate_multiplier * 1e6 / MIB,
            memory_mib_s=b["get_h2d_GB_s"] * 1e9 / MIB,
            put_mib_s=b["put_d2h_GB_s"] * 1e9 / MIB,
        )
        compute = public["compute"]
        prefill, decode, decode8, prefill_rate = (
            None,
            compute["decode_ms_batch_1"],
            compute["decode_ms_batch_8"],
            compute["prefill_tokens_s"],
        )
    overrides = {k: getattr(args, k) for k in store_flags if getattr(args, k) is not None}
    store_args.update(overrides)
    default_restore = (public or {}).get(
        "restore_path", "host-staged" if not reference else "legacy"
    )
    restore_path = args.restore_path or default_restore
    if args.h2d_mib_s is not None and restore_path != "host-staged":
        p.error("--h2d-mib-s requires --restore-path host-staged")
    if restore_path == "host-staged":
        reference_h2d = (
            public["bandwidth"]["get_h2d_GB_s"] * 1e9 / MIB
            if public
            else store_args.get("memory_mib_s", StoreConfig.memory_mib_s)
        )
        store_args["h2d_mib_s"] = reference_h2d if args.h2d_mib_s is None else args.h2d_mib_s
    else:
        store_args["h2d_mib_s"] = None
    if not math.isfinite(args.ssd_time_scale) or args.ssd_time_scale <= 0:
        p.error("--ssd-time-scale must be finite and positive")
    if args.ssd_time_scale != 1:
        store_args["device_mib_s"] = (
            store_args.get("device_mib_s", StoreConfig.device_mib_s) / args.ssd_time_scale
        )
        if "device_read_mib_s" in store_args:
            store_args["device_read_mib_s"] /= args.ssd_time_scale
    config = StoreConfig(
        args.clients or (mc["client_count"] if reference else 1),
        budgets["segment"],
        budgets["staging"],
        budgets["quota"],
        **store_args,
    )
    if volume:
        nominal = volume["member_count"] * volume["nominal_bytes_per_member"]
        total_quota = config.clients * config.quota_bytes
        if total_quota > nominal:
            p.error("Sum of client quotas exceeds the nominal logical volume capacity")
    if args.prefill_ms is not None:
        prefill, prefill_rate = args.prefill_ms, None
    if args.decode_ms is not None:
        decode = decode8 = args.decode_ms
    timing = StepTiming(prefill, decode, decode8, prefill_rate, args.compute_scale)
    timeline = OffloadTimeline(args.offload_timeline) if args.offload_timeline else None
    blocks = args.num_blocks if args.num_blocks is not None else default_blocks
    max_len = args.max_model_len if args.max_model_len is not None else default_len
    if blocks < 3 or max_len < 1:
        p.error("Invalid scheduler capacity")
    offsets = args.offset_s or [0] * len(args.inputs)
    if len(offsets) != len(args.inputs) or any(not math.isfinite(s) or s < 0 for s in offsets):
        p.error("Supply one nonnegative, finite offset per input")
    clock = Clock()
    limit = min(max_len, (blocks - 2) * plan["geometry"]["manager_block_tokens"])
    admissions = Admissions(args.inputs, offsets, clock, limit)
    args.output.mkdir(parents=True, exist_ok=False)
    manifest = {
        "profile": args.profile,
        "num_blocks": blocks,
        "max_model_len": max_len,
        "store": asdict(config),
        "prefill_ms": timing.prefill_ms,
        "decode_ms": timing.decode_ms if timing.decode_ms == timing.decode_ms_batch_8 else None,
        "step_timing": asdict(timing),
        "ssd_time_scale": args.ssd_time_scale,
        "restore_path": restore_path,
        "root_offsets_s": offsets,
        "vllm_version": version("vllm"),
        "pythonhashseed": os.environ.get("PYTHONHASHSEED"),
        "geometry_sha256": sha(folder / "resolved.json"),
        "compiled_inputs": [{"file": str(path), "sha256": sha(path)} for path in args.inputs],
        "service_profile": {"sha256": sha(args.service_profile), "calibration": fit}
        if fit
        else None,
    }
    if public:
        override_fields = (
            "clients",
            "num_blocks",
            "max_model_len",
            "segment_mib",
            "staging_mib",
            "quota_mib",
            "prefill_ms",
            "decode_ms",
            "restore_path",
            "h2d_mib_s",
        )
        manifest["reference_settings"] = {
            "file": str(profile_path.relative_to(ROOT)),
            "sha256": sha(profile_path),
            "baseline": public,
            "overrides": {
                **overrides,
                **{k: getattr(args, k) for k in override_fields if getattr(args, k) is not None},
            },
        }
    if volume:
        manifest["logical_volume"] = {
            "members": volume["member_count"],
            "nominal_bytes_per_member": volume["nominal_bytes_per_member"],
            "nominal_bytes": nominal,
            "kv_quota_bytes": total_quota,
            "quota_scope": "per storage client",
            "fifo_queues": 1,
            "service_scenario": scenario,
            "service_rate_multiplier": rate_multiplier,
            "effective_read_bytes_s": config.device_read_mib_s * MIB,
            "effective_write_bytes_s": config.device_mib_s * MIB,
            "io_scope": "logical volume; owner identifies a storage client, not a physical drive",
        }
    if restore_path == "host-staged":
        manifest["restore_model"] = {
            "destination_h2d_ports": 8,
            "h2d_bytes_s_per_rank": config.h2d_mib_s * MIB,
            "staging_scope": "payload bytes at the disk replica owner, reserved before SSD submit through H2D completion",
            "granularity": "object pipeline within a native vLLM GET sub-batch; one sequential receiver per rank",
            "memory_get": "one copy occupies both the owner's memory-read port and the destination H2D port",
            "disk_get": "SSD completion releases the file pin; H2D completion releases staging",
            "request_ack": "all objects in all receiver sub-batches on all TP ranks complete",
        }
    manifest["source_sha256"] = {
        str(p.relative_to(ROOT)): sha(p)
        for p in (
            Path(__file__).resolve(),
            ROOT / "scripts/check_staged_restores.py",
            ROOT / "scripts/prepare_object_trace.py",
            ROOT / "scripts/prepare_kvlab.py",
            ROOT / "src/kv_ssd_sim/kvlab.py",
            ROOT / "scripts/audit_aiperf_input.py",
            *(
                ROOT / "src/kv_ssd_sim" / name
                for name in (
                    "object_traffic.py",
                    "object_store.py",
                    "object_timing.py",
                    "object_master.py",
                    "object_worker.py",
                    "native.py",
                    "k3_online.py",
                    "deployment.py",
                )
            ),
        )
    }
    manifest["service_model"] = (
        "Native offset allocators, writer-preferred segments, global MasterService.BatchEvict, "
        "eager offload with periodic client heartbeats, native bucket grouping and source leases, "
        "one shared virtual FIFO for storage service, native FIFO bucket selection at the virtual payload quota, "
        "one memory read port per client."
    )
    manifest["clock"] = {
        "epoch_unix_ns": 1700000000000000000,
        "same_time_order": "native maintenance, then external events in enqueue order",
        "final_boundary": "first native maintenance boundary strictly after the last event",
    }
    manifest["offload_timing_input"] = (
        {"file": str(args.offload_timeline), "sha256": sha(args.offload_timeline)}
        if args.offload_timeline
        else None
    )

    def write(name, data):
        (args.output / name).write_text(json.dumps(data, indent=2) + "\n")

    with (
        (args.output / "events.jsonl").open("w") as f,
        (args.output / "bucket-timing.jsonl").open("x") as timings,
    ):

        def emit(event, **fields):
            f.write(json.dumps(dict(time_ns=clock.now, event=event, **fields)) + "\n")
            if event == "bucket_timing":
                timings.write(json.dumps(fields) + "\n")

        with ObjectStore(clock, config, emit, args.output / "master.log", timeline) as store:
            manifest["native_master"] = store.master.provenance
            write("launch.json", manifest)
            with metadata_scheduler(store, plan, allocation.kv_cache_groups, blocks, max_len) as (
                scheduler,
                transfers,
            ):
                requests = simulate(
                    admissions,
                    scheduler,
                    transfers,
                    store,
                    plan["geometry"]["hash_tokens"],
                    timing,
                    bool(public),
                )
            clock.advance((clock.now // store.period + 1) * store.period)
            store_summary = store.summary()
    touches = sum(store.counts[x] for x in ("put_complete", "get_memory", "get_disk"))
    traffic, reuses = describe_traffic(
        args.output / "events.jsonl", args.output / "reuse.jsonl", touches
    )
    for row in requests:
        row["store_traffic"] = traffic.get(row["id"], {"counts": {}, "bytes": {}})
    write("requests.json", requests)
    objects = []
    for k, size in sorted(store.inventory.items()):
        obj = store.objects.get(k)
        objects.append(
            {
                "key": k,
                "bytes": size,
                "owner": obj.owner if obj else None,
                "memory": bool(obj and obj.memory),
                "disk": bool(obj and obj.disk),
            }
        )
    write("objects.json", objects)
    demand = export_demand(args.output, args.output / "events.jsonl", clock.now, requests)
    write("ssd-summary.json", demand)
    result = dict(
        status="complete",
        cross_request_store_reuses=reuses,
        virtual_end_ns=clock.now,
        requests=len(requests),
        **store_summary,
    )
    if restore_path == "host-staged":
        write("restore-summary.json", audit_restores(args.output, result))
    write("summary.json", result)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
