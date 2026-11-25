"""Run K3 object-level predictions on compiled WEKA sessions with no KV payload allocation."""

import argparse
import json
import os
from pathlib import Path

from ..analysis.restore import audit as audit_restores
from ..object_store import ObjectStore
from ..object_traffic import describe_traffic, export_demand
from ..runtime import PROJECT_ROOT as ROOT
from .clock import Clock
from .engine import Admissions, simulate
from .settings import PUBLIC_PROFILES, apply_calibration, resolve_settings
from .storage.offload import OffloadTimeline


def build_parser():
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
    return p


def run(args):
    fit = apply_calibration(args)
    os.environ.setdefault("VLLM_SSM_CONV_STATE_LAYOUT", "SD")

    from kv_ssd_sim.deployment import native_deployment
    from kv_ssd_sim.object_worker import metadata_scheduler

    folder = ROOT / "configs/kimi-k3-b300-step2"
    plan, allocation = native_deployment(folder)
    if json.loads(json.dumps(plan)) != json.loads((folder / "resolved.json").read_bytes()):
        raise ValueError("Pinned native K3 geometry changed")
    settings = resolve_settings(args, plan, folder, fit)
    blocks, max_len = settings.blocks, settings.max_len
    manifest = settings.manifest
    timeline = OffloadTimeline(args.offload_timeline) if args.offload_timeline else None
    clock = Clock()
    limit = min(max_len, (blocks - 2) * plan["geometry"]["manager_block_tokens"])
    admissions = Admissions(args.inputs, settings.offsets, clock, limit)
    args.output.mkdir(parents=True, exist_ok=False)

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

        with ObjectStore(
            clock, settings.store, emit, args.output / "master.log", timeline
        ) as store:
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
                    settings.timing,
                    settings.record_steps,
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
    if settings.store.h2d_mib_s is not None:
        write("restore-summary.json", audit_restores(args.output, result))
    write("summary.json", result)
    return result


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        result = run(args)
    except argparse.ArgumentTypeError as exc:
        parser.error(str(exc))
    print(json.dumps(result))


if __name__ == "__main__":
    main()
