"""Compare a frozen object prediction to native server events, independent of event ordering."""

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path


def load(path):
    return json.loads(path.read_bytes())


def digest(data):
    return hashlib.sha256(json.dumps(sorted(data)).encode()).hexdigest()


def compare(prediction, server, retrospective=False):
    reqs = {r["id"]: r for r in load(prediction / "requests.json")}
    model = [json.loads(line) for line in (prediction / "events.jsonl").open()]
    objects = {r["key"]: r for r in load(prediction / "objects.json")}
    native = [json.loads(line) for line in (server / "events.jsonl").open()]
    actual = load(server / "summary.json")
    freeze_path = server.parent / "frozen-prediction.json"
    matches_original_freeze = None
    if freeze_path.exists():
        frozen = load(freeze_path)["sha256"]
        matches_original_freeze = not any(
            hashlib.sha256((prediction / name).read_bytes()).hexdigest() != h
            for name, h in frozen.items()
        )
        if not matches_original_freeze and not retrospective:
            raise ValueError("Prediction changed after it was frozen")
    ids, http, metrics = {}, {}, {}
    for e in native:
        if e["event"] == "http_received":
            rid = e["client_request_id"]
            if rid not in reqs:
                candidates = [
                    r["id"]
                    for r in reqs.values()
                    if r["prompt_sha256"] == e["prompt_sha256"]
                    and r["output_tokens"] == e["target_output_tokens"]
                ]
                if len(candidates) != 1:
                    raise ValueError(
                        "Ambiguous native request mapping; require explicit client IDs"
                    )
                rid = candidates[0]
            if rid in http or e["request_id"] in ids:
                raise ValueError("Duplicate native request mapping")
            ids[e["request_id"]] = rid
            http[rid] = e

    nat_put, nat_get, nat_tier = Counter(), Counter(), Counter()
    pred_put, pred_get, pred_tier = Counter(), Counter(), Counter()
    payloads = {}
    rejected = Counter()
    issued = Counter()
    disk_keys = set()
    final_tiers = None
    restored_ok = True
    failed_gets = Counter()
    for e in native:
        op = e["event"]
        rid = ids.get(e.get("request_id"))
        if op == "store_put_end":
            for k, n, rc, h in zip(
                e["keys"], e["object_bytes"], e["result"], e["digests"], strict=True
            ):
                if rc:
                    rejected[str(rc)] += 1
                    continue
                if payloads.setdefault(k, (n, h)) != (n, h):
                    raise ValueError("Actual identical key changed content")
                nat_put[rid, e["logical_rank"], k, n] += 1
        elif op == "store_get_begin":
            for k in e["keys"]:
                tiers = e["tiers"][k]
                tier = (
                    "memory"
                    if "MEMORY" in tiers
                    else "disk"
                    if "LOCAL_DISK" in tiers
                    else "unknown"
                )
                nat_tier[rid, e.get("logical_rank"), k, tier] += 1
        elif op == "store_get_end":
            for k, rc, ok in zip(e["keys"], e["result"], e["checksum_ok"], strict=True):
                restored_ok &= ok and rc >= 0
                if rc >= 0 and ok:
                    nat_get[rid, e.get("logical_rank"), k, payloads[k][0]] += 1
                else:
                    failed_gets[str(rc)] += 1
        elif op == "compute_begin":
            issued.update(e["scheduled_tokens"])
        elif op == "disk_replicas_confirmed":
            disk_keys.update(e["keys"])
        elif op == "natural_final_replicas":
            final_tiers = e["tiers"]
            disk_keys = {k for k, t in final_tiers.items() if "LOCAL_DISK" in t}

    for m in actual.get("request_metrics", []):
        nr = m["request_id"]
        computed = issued[nr] - (m["output_tokens"] - 1)
        metrics[ids[nr]] = {
            "prompt_tokens": m["input_tokens"],
            "prompt_sha256": http[ids[nr]]["prompt_sha256"],
            "output_tokens": m["output_tokens"],
            "computed_prompt_tokens": computed,
            "external_load_tokens": m["external_load_tokens"],
            "local_reuse_tokens": m["input_tokens"] - computed - m["external_load_tokens"],
        }
    fields = (
        "prompt_tokens",
        "prompt_sha256",
        "output_tokens",
        "computed_prompt_tokens",
        "external_load_tokens",
        "local_reuse_tokens",
    )
    rows = [
        {"id": k, "prediction": {f: r[f] for f in fields}, "native": metrics.get(k)}
        for k, r in reqs.items()
    ]
    queued, acks = defaultdict(Counter), defaultdict(Counter)
    for e in model:
        op = e["event"]
        if op in ("save_queued", "save_ack"):
            target = queued if op == "save_queued" else acks
            target[e["request"], e["job"]][e["rank"]] += 1
        elif op == "put_complete":
            pred_put[e["request"], e["rank"], e["key"], e["bytes"]] += 1
        elif op in ("get_memory", "get_disk"):
            pred_get[e["request"], e["rank"], e["key"], e["bytes"]] += 1
            pred_tier[e["request"], e["rank"], e["key"], op[4:]] += 1
    mp = {k: r["bytes"] for k, r in objects.items()}
    checks = {
        "native_completed": actual["status"] == "passed"
        and not actual.get("checksum_errors")
        and not rejected,
        "requests_and_tokens": len(rows) == len(http) == len(metrics)
        and all(r["prediction"] == r["native"] for r in rows),
        "inventory_key_size": mp == {k: n for k, (n, _) in payloads.items()},
        "put_request_rank_key_bytes": pred_put == nat_put,
        "get_request_rank_key_bytes": pred_get == nat_get and restored_ok,
        "get_available_tier_request_rank_key": pred_tier == nat_tier,
        "final_disk_keys": {k for k, r in objects.items() if r["disk"]} == disk_keys,
        "save_fences_all_ranks": queued == acks
        and all(v == Counter(range(8)) for v in queued.values()),
    }
    if final_tiers is not None:
        checks["final_memory_keys"] = {k for k, r in objects.items() if r["memory"]} == {
            k for k, t in final_tiers.items() if "MEMORY" in t
        }

    def unmatched(left, right, value):
        return [
            {"request": rid, "rank": rank, "key": key, value: item, "occurrences": count}
            for (rid, rank, key, item), count in (left - right).items()
        ]

    differences = {}
    for name, a, b, value in (
        ("put", pred_put, nat_put, "bytes"),
        ("get", pred_get, nat_get, "bytes"),
        ("get_available_tier", pred_tier, nat_tier, "tier"),
    ):
        if a != b:
            differences[name] = {
                "prediction_only": unmatched(a, b, value),
                "native_only": unmatched(b, a, value),
            }
    if final_tiers is not None:
        a = {k for k, r in objects.items() if r["memory"]}
        b = {k for k, tiers in final_tiers.items() if "MEMORY" in tiers}
        if a != b:
            differences["final_memory"] = {
                "prediction_only": sorted(a - b),
                "native_only": sorted(b - a),
            }

    def summ(puts, gets, tiers, inv):
        sizes = dict(sorted(Counter(inv.values()).items()))
        tier_ops = Counter()
        tier_bytes = Counter()
        for (_, _, k, t), count in tiers.items():
            tier_ops[t] += count
            tier_bytes[t] += count * inv.get(k, 0)
        return {
            "unique_keys": len(inv),
            "unique_bytes": sum(inv.values()),
            "counts_by_object_bytes": sizes,
            "puts": puts.total(),
            "gets": gets.total(),
            "get_bytes": sum(row[3] * n for row, n in gets.items()),
            "get_counts_by_available_tier": dict(tier_ops),
            "get_bytes_by_available_tier": dict(tier_bytes),
            "key_size_sha256": digest(inv.items()),
        }

    return {
        "status": "passed" if all(checks.values()) else "failed",
        "comparison": "retrospective"
        if retrospective
        else ("frozen" if freeze_path.exists() else "unfrozen"),
        "matches_original_freeze": matches_original_freeze,
        "checks": checks,
        "differences": differences,
        "requests": rows,
        "prediction": summ(pred_put, pred_get, pred_tier, mp),
        "native": summ(nat_put, nat_get, nat_tier, {k: n for k, (n, _) in payloads.items()}),
        "native_key_content_sha256": digest(payloads.items()),
        "native_error": actual.get("error"),
        "native_rejected_puts": dict(rejected),
        "native_failed_gets": dict(failed_gets),
        "save_jobs": len(queued),
        "final_memory_keys": {
            "prediction": sum(r["memory"] for r in objects.values()),
            "native": None
            if final_tiers is None
            else sum("MEMORY" in t for t in final_tiers.values()),
        },
        "unmatched_tier_gets": (pred_tier - nat_tier).total(),
        "native_events_sha256": hashlib.sha256((server / "events.jsonl").read_bytes()).hexdigest(),
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--prediction", required=True, type=Path)
    p.add_argument("--server", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument(
        "--retrospective",
        action="store_true",
        help="Compare a later model to a historical trial and record its original freeze status",
    )
    args = p.parse_args()
    report = compare(args.prediction, args.server, retrospective=args.retrospective)
    with args.output.open("x") as stream:
        stream.write(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"status": report["status"], "checks": report["checks"]}))
    raise SystemExit(int(report["status"] != "passed"))
