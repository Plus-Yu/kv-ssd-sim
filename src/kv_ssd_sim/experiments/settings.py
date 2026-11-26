"""Resolve experiment profiles into effective budgets, timing and recorded settings."""

import json
import math
import os
from argparse import ArgumentTypeError
from dataclasses import asdict, dataclass
from importlib.metadata import version

from ..adapters.mooncake import checksum
from ..core.paths import PROJECT_ROOT as ROOT
from ..core.provenance import package_source_hashes
from ..simulation.storage.config import StoreConfig
from ..simulation.storage.device import MIB
from ..simulation.timing import ContextTiming, StepTiming
from .profiles import PROFILE_FILES, load_profile


@dataclass
class ReplaySettings:
    store: StoreConfig
    timing: StepTiming | ContextTiming
    scheduler: dict
    blocks: int
    max_len: int
    offsets: list[float]
    record_steps: bool
    manifest: dict


def apply_calibration(args):
    if args.volume_service and args.profile not in (
        "reference",
        "reference-context",
        "reference-volume-20261125",
    ):
        raise ArgumentTypeError("--volume-service requires a reference volume profile")
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
            raise ArgumentTypeError("Expected a complete n15 calibration profile")
        for field, value in fit["flags"].items():
            setattr(args, field, value)
    return fit


def resolve_settings(args, plan, folder, fit):
    reference = args.profile != "n15"
    profile_path = PROFILE_FILES.get(
        "reference" if args.profile == "reference-context" else args.profile
    )
    public, profile_inputs = load_profile(profile_path) if profile_path else (None, {})
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
        raise ArgumentTypeError("--h2d-mib-s requires --restore-path host-staged")
    if restore_path == "host-staged":
        reference_h2d = (
            public["bandwidth"]["get_h2d_GB_s"] * 1e9 / MIB
            if public
            else store_args.get("memory_mib_s", StoreConfig.memory_mib_s)
        )
        store_args["h2d_mib_s"] = reference_h2d if args.h2d_mib_s is None else args.h2d_mib_s
    else:
        store_args["h2d_mib_s"] = None
    if not math.isfinite(args.host_copy_scale) or args.host_copy_scale <= 0:
        raise ArgumentTypeError("--host-copy-scale must be finite and positive")
    if args.host_copy_lanes is not None or args.host_copy_scale != 1:
        if restore_path != "host-staged" or store_args.get("put_mib_s") is None:
            raise ArgumentTypeError("Host-copy controls require timed PUTs and host-staged GETs")
        store_args["host_copy_lanes"] = args.host_copy_lanes
        for field in ("put_mib_s", "h2d_mib_s", "memory_mib_s"):
            store_args[field] = (
                store_args.get(field, getattr(StoreConfig, field)) * args.host_copy_scale
            )
    if not math.isfinite(args.ssd_time_scale) or args.ssd_time_scale <= 0:
        raise ArgumentTypeError("--ssd-time-scale must be finite and positive")
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
            raise ArgumentTypeError(
                "Sum of client quotas exceeds the nominal logical volume capacity"
            )
    if args.prefill_ms is not None:
        prefill, prefill_rate = args.prefill_ms, None
    if args.decode_ms is not None:
        decode = decode8 = args.decode_ms
    step_profile = args.step_profile
    if args.profile == "reference-context" and step_profile is None:
        step_profile = ROOT / "configs/compute/b300-tp8-context.json"
    calibration = None
    if step_profile:
        if args.prefill_ms is not None or args.decode_ms is not None or fit:
            raise ArgumentTypeError("The paired context profile supplies both phase timings")
        source = json.loads(step_profile.read_bytes())
        model_file = ROOT / "references/kimi-k3-b300/model-config.json"
        model = json.loads(model_file.read_bytes())
        timing = ContextTiming.from_profile(
            source,
            plan,
            model.get("text_config", model),
            args.compute_scale,
            args.context_cost_scale,
        )
        calibration = {
            "file": str(step_profile),
            "sha256": checksum(step_profile),
            "parameters": source,
            "model_geometry_sha256": checksum(model_file),
            "context_cost_scale": args.context_cost_scale,
        }
    else:
        if args.context_cost_scale != 1:
            raise ArgumentTypeError("--context-cost-scale requires a context timing profile")
        timing = StepTiming(prefill, decode, decode8, prefill_rate, args.compute_scale)
    scheduler = {
        "max_num_seqs": args.max_num_seqs,
        "max_num_batched_tokens": args.max_num_batched_tokens,
        "save_decode_cache": bool(args.save_decode_cache),
        "native_retention": calibration is not None or args.save_decode_cache is not None,
    }
    max_batch = timing.decode_batch[-1] if calibration else 8
    if not 1 <= args.max_num_seqs <= max_batch:
        raise ArgumentTypeError(f"Timing profile supports 1–{max_batch} sequences per step")
    blocks = args.num_blocks if args.num_blocks is not None else default_blocks
    max_len = args.max_model_len if args.max_model_len is not None else default_len
    if blocks < 3 or max_len < 1:
        raise ArgumentTypeError("Invalid scheduler capacity")
    if args.max_num_batched_tokens < max(
        args.max_num_seqs, min(max_len, plan["geometry"]["manager_block_tokens"])
    ):
        raise ArgumentTypeError(
            "Token budget must accommodate native hybrid block alignment and the batch"
        )
    offsets = args.offset_s or [0] * len(args.inputs)
    if len(offsets) != len(args.inputs) or any(not math.isfinite(s) or s < 0 for s in offsets):
        raise ArgumentTypeError("Supply one nonnegative, finite offset per input")
    manifest = {
        "profile": args.profile,
        "num_blocks": blocks,
        "max_model_len": max_len,
        "store": asdict(config),
        "prefill_ms": None if calibration else timing.prefill_ms,
        "decode_ms": None
        if calibration or timing.decode_ms != timing.decode_ms_batch_8
        else timing.decode_ms,
        "step_timing": asdict(timing),
        "ssd_time_scale": args.ssd_time_scale,
        "restore_path": restore_path,
        "root_offsets_s": offsets,
        "vllm_version": version("vllm"),
        "pythonhashseed": os.environ.get("PYTHONHASHSEED"),
        "geometry_sha256": checksum(folder / "resolved.json"),
        "compiled_inputs": [{"file": str(path), "sha256": checksum(path)} for path in args.inputs],
        "service_profile": {"sha256": checksum(args.service_profile), "calibration": fit}
        if fit
        else None,
    }
    if config.host_copy_lanes is None:
        del manifest["store"]["host_copy_lanes"]
    else:
        manifest["host_copy_model"] = {
            "shared_lanes": config.host_copy_lanes,
            "directions": ["D2H", "H2D"],
            "upper_bound_bytes_s": config.host_copy_lanes
            * max(config.put_mib_s, config.h2d_mib_s)
            * MIB,
            "scope": "FIFO reservations shared by rank PUTs and staged GETs, excluding SSD DMA",
        }
    if args.host_copy_scale != 1:
        manifest["host_copy_scale"] = args.host_copy_scale
    if calibration:
        manifest["compute_calibration"] = calibration
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
            "sha256": profile_inputs[str(profile_path.relative_to(ROOT))],
            "inputs_sha256": profile_inputs,
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
    manifest["source_sha256"] = package_source_hashes(relative_to=ROOT)
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
        {"file": str(args.offload_timeline), "sha256": checksum(args.offload_timeline)}
        if args.offload_timeline
        else None
    )

    return ReplaySettings(
        store=config,
        timing=timing,
        scheduler=scheduler,
        blocks=blocks,
        max_len=max_len,
        offsets=offsets,
        record_steps=bool(public) or bool(calibration),
        manifest=manifest,
    )
