"""Fit n15 timing on one native window, or score an independently frozen prediction."""

import argparse
import csv
import gzip
import hashlib
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path

from compare_object_replay import compare, load

MS = 1_000_000
BIN = 10_000 * MS


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write(path, data):
    with path.open("x") as f:
        json.dump(data, f, indent=2)
        f.write("\n")


def span(values):
    return (
        {
            "n": len(values),
            "min": min(values),
            "median": statistics.median(values),
            "max": max(values),
        }
        if values
        else {"n": 0, "min": None, "median": None, "max": None}
    )


def online(trial):
    ev = [json.loads(line) for line in (trial / "server/events.jsonl").open()]
    origin = next(e["monotonic_origin_ns"] for e in ev if e["event"] == "clock_sync")
    zero = min(e["time_ns"] for e in ev if e["event"] == "request_admitted")
    names = {e["request_id"]: e["client_request_id"] for e in ev if e["event"] == "http_received"}
    # Real calls can duplicate keys and overlap across ranks. Attribute by the call interval.
    calls = defaultdict(list)
    open_calls = {}
    for e in ev:
        tag = e["event"]
        if tag not in ("store_put_begin", "store_put_end", "store_get_begin", "store_get_end"):
            continue
        op = tag.split("_")[1]
        key = (op, e["request_id"], e["logical_rank"], tuple(e["keys"]))
        if tag.endswith("begin"):
            open_calls[key] = e
            continue
        begin = open_calls.pop(key)
        for k, rc in zip(e["keys"], e["result"], strict=True):
            if rc >= 0:
                calls[op, k].append(
                    {
                        "begin": begin["time_ns"] - zero,
                        "end": e["time_ns"] - zero,
                        "request": names[e["request_id"]],
                        "rank": e["logical_rank"],
                        "job": e.get("job"),
                    }
                )
    if open_calls:
        raise ValueError("Unclosed native client call")
    stages = defaultdict(list)
    commits = defaultdict(list)
    pending = {}
    demand = []
    failures = []
    with (trial / "store-boundaries.tsv").open() as ledger:
        boundaries = sorted(csv.reader(ledger, delimiter="\t"), key=lambda r: int(r[0]))
    for stamp, stage, k, b, u, rc in boundaries:
        # FileStorage namespaces its backend keys with tenant_id + NUL. Master RPCs
        # carry the tenant separately. This fixture uses Mooncake's default tenant.
        if "\x00" in k:
            tenant, k = k.split("\x00", 1)
            if tenant != "default":
                raise ValueError(f"Unconfigured Mooncake tenant {tenant!r}")
        t = int(stamp) - origin - zero
        b, u, rc = int(b), int(u), int(rc)
        if rc or (stage == "disk_ack" and b < 0):
            failures.append({"key": k, "stage": stage, "code": rc, "bytes": b})
            continue
        if stage in {"allocated", "memory_ready", "read_submit"}:
            op = "get" if stage == "read_submit" else "put"
            sources = [r for r in calls[op, k] if r["begin"] <= t <= r["end"]]
            if len(sources) != 1:
                raise ValueError(f"Native boundary has {len(sources)} possible {op} callers: {k}")
            caller = sources[0]
            if stage == "memory_ready":
                commits[k].append(caller)
        elif stage == "write_submit":
            if not commits[k]:
                raise ValueError(f"File submission without an attributable ready replica: {k}")
            caller = commits[k][-1]
        else:
            op = "read" if stage == "read_complete" else "write"
            caller = pending[op, k]
        base = {
            "key": k,
            "request": caller["request"],
            "rank": caller["rank"],
            "job": caller["job"],
        }
        stages[stage].append(dict(**base, time_ns=t, bytes=b, bucket_id=u))
        if stage in ("write_submit", "read_submit"):
            op = stage.split("_")[0]
            if (op, k) in pending:
                raise ValueError(f"Overlapping same-key file calls: {k}")
            pending[op, k] = dict(**base, op=op, bytes=b, submit_ns=t, bucket_id=None)
        elif stage in ("write_start", "file_write_end", "disk_ack", "read_complete"):
            op = "read" if stage == "read_complete" else "write"
            row = pending[op, k]
            row[
                {
                    "write_start": "start_ns",
                    "file_write_end": "file_complete_ns",
                    "disk_ack": "complete_ns",
                    "read_complete": "complete_ns",
                }[stage]
            ] = t
            if u >= 0:
                row["bucket_id"] = u
            if stage in ("disk_ack", "read_complete"):
                if row["bytes"] != b:
                    raise ValueError(f"Object length differs at completion: {k}")
                if (
                    op == "write"
                    and not row["submit_ns"] <= row["start_ns"] <= row["file_complete_ns"] <= t
                ):
                    raise ValueError(f"Native write boundaries out of order: {k}")
                demand.append(pending.pop((op, k)))
    final = next(e for e in ev if e["event"] == "natural_final_replicas")
    if pending or final["unresolved_offloads"] or failures:
        raise ValueError(
            f"Incomplete native observation: {len(pending)} open, "
            f"{len(final['unresolved_offloads'])} unacknowledged, {failures[:2]}"
        )
    if {k for op, k in calls if op == "put"} != set(commits):
        raise ValueError("The passive Mooncake observer missed a successful PUT")
    return ev, names, zero, origin, stages, demand


def fit(trial, out):
    ev, names, zero, origin, stages, demand = online(trial)
    kinds = {e["step"]: e["kind"] for e in ev if e["event"] == "compute_begin"}
    compute = defaultdict(list)
    materialize = defaultdict(list)
    for e in ev:
        if e["event"] == "compute_end":
            kind = kinds[e["step"]]
            compute[kind].append(e["actual_compute_ms"])
            materialize[kind].append(e["materialize_ms"])
    begin = {}
    ready = defaultdict(list)
    allocated = defaultdict(list)
    for stage, target in (("allocated", allocated), ("memory_ready", ready)):
        for e in stages[stage]:
            target[e["request"], e["rank"], e["job"]].append(e["time_ns"])
    alloc_ms, ready_ms, worker_ms = [], [], []
    for e in ev:
        if (
            e["event"] not in {"native_worker_begin", "native_worker_end"}
            or e["operation"] != "save"
        ):
            continue
        k = (names[e["request_id"]], e["logical_rank"], e["job"])
        if e["event"] == "native_worker_begin":
            begin[k] = e["time_ns"] - zero
        elif k in ready:
            start = max(begin[k], e["fence_released_mono_ns"] - origin - zero)
            alloc_ms.append((min(allocated[k]) - start) / MS)
            ready_ms.append((max(ready[k]) - start) / MS)
            worker_ms.append((e["time_ns"] - zero - start) / MS)
    writes = [r for r in demand if r["op"] == "write"]
    times = {(r["bucket_id"], r["start_ns"], r["complete_ns"]) for r in writes}
    rate = (
        sum(r["bytes"] for r in writes)
        * 1e9
        / (sum(stop - start for _, start, stop in times) * (1 << 20))
    )
    params = {
        "prefill_ms": statistics.median(compute["prefill"]),
        "decode_ms": statistics.median(compute["decode"]),
        "save_ms": statistics.median(ready_ms),
        "put_allocation_ms": statistics.median(alloc_ms),
        "device_mib_s": rate,
        "first_offload_ms": min(r["submit_ns"] for r in writes) / MS % 1000,
    }
    if not 0 <= params["put_allocation_ms"] <= params["save_ms"]:
        raise ValueError("Calibration fence and client allocation disagree")
    write(
        out,
        {
            "calibration_only": True,
            "inputs": {
                p: sha(trial / p)
                for p in (
                    "server/events.jsonl",
                    "store-boundaries.tsv",
                    "frozen-prediction.json",
                    "native-launch.json",
                )
            },
            "flags": params,
            "rank_nonempty_jobs": len(ready_ms),
            "buckets": len(times),
            "compute_ms": {k: span(v) for k, v in compute.items()},
            "cpu_materialize_ms": {k: span(v) for k, v in materialize.items()},
            "after_fence_ms": {
                "first_allocation": span(alloc_ms),
                "last_ready": span(ready_ms),
                "worker_end": span(worker_ms),
            },
            "memory_get_service": "no positive calibration read; retain working point of 8192 MiB/s",
            "heartbeat_phase": "first native submission modulo 1000 ms from first admission; estimate includes source fetch",
        },
    )


def native_curve(demand, end):
    bins = [[0, 0] for _ in range(end // BIN + 1)]
    for r in demand:
        bins[r["submit_ns"] // BIN][r["op"] == "write"] += r["bytes"]
    return [
        {
            "from_ns": i * BIN,
            "until_ns": min((i + 1) * BIN, end),
            "read_bytes": b[0],
            "write_bytes": b[1],
        }
        for i, b in enumerate(bins)
    ]


def score(pred, trial, out):
    out.mkdir(parents=True, exist_ok=False)
    ev, names, zero, _, stages, demand = online(trial)
    base = compare(pred, trial / "server")
    write(out / "object-comparison.json", base)
    model = [json.loads(l) for l in (pred / "events.jsonl").open()]
    with gzip.open(pred / "ssd-demand.jsonl.gz", "rt") as f:
        modeled = [json.loads(l) for l in f]
    with gzip.open(out / "native-ssd-demand.jsonl.gz", "xt") as f:
        for r in sorted(demand, key=lambda r: r["submit_ns"]):
            f.write(json.dumps(r) + "\n")
    mzero = min(e["time_ns"] for e in model if e["event"] == "request_admitted")
    final = next(e for e in ev if e["event"] == "natural_final_replicas")
    actual_curve = native_curve(demand, final["time_ns"] - zero)
    with (out / "native-10s.csv").open("x", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=actual_curve[0])
        writer.writeheader()
        writer.writerows(actual_curve)

    def accounting(rows):
        return Counter((r["op"], r["request"], r["rank"], r["key"], r["bytes"]) for r in rows)

    a, b = accounting(modeled), accounting(demand)

    def changes(c):
        return [
            {"op": op, "request": rid, "rank": rank, "key": k, "bytes": size, "occurrences": n}
            for (op, rid, rank, k, size), n in sorted(c.items())
        ]

    diff = {"model_only": changes(a - b), "native_only": changes(b - a)}
    write(out / "unmatched-ssd.json", diff)

    def totals(c, op):
        items = [(t[-1], n) for t, n in c.items() if t[0] == op]
        return {"objects": sum(n for _, n in items), "bytes": sum(n * size for size, n in items)}

    counts = {
        op: {
            label: totals(c, op)
            for label, c in (
                ("matched", a & b),
                ("false_positive", a - b),
                ("missed", b - a),
                ("prediction", a),
                ("native", b),
            )
        }
        for op in ("read", "write")
    }
    pairs = {}
    with gzip.open(out / "timing-by-key.csv.gz", "wt", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            (
                "stage",
                "request",
                "rank",
                "key",
                "occurrence",
                "model_ns",
                "native_ns",
                "model_minus_native_ms",
            )
        )
        for v, n in (
            ("put_allocated", "allocated"),
            ("put_complete", "memory_ready"),
            ("disk_write_submit", "write_submit"),
            ("disk_write_start", "write_start"),
            ("disk_publish", "disk_ack"),
            ("disk_read_submit", "read_submit"),
            ("disk_read_complete", "read_complete"),
        ):
            left, right = defaultdict(list), defaultdict(list)
            for rows, tag, target, offset in ((model, v, left, mzero), (stages[n], None, right, 0)):
                for e in rows:
                    if tag is None or e["event"] == tag:
                        target[e["request"], e["rank"], e["key"]].append(e["time_ns"] - offset)
            errors = []
            for k in left.keys() | right.keys():
                for i in range(max(len(left[k]), len(right[k]))):
                    x = left[k][i] if i < len(left[k]) else None
                    y = right[k][i] if i < len(right[k]) else None
                    delta = None if x is None or y is None else (x - y) / MS
                    if delta is not None:
                        errors.append(delta)
                    writer.writerow((n, *k, i, x, y, delta))
            pairs[n] = span(errors)
    end = max(e["time_ns"] for e in ev if e["event"] == "request_finished") - zero
    model_scope = load(pred / "ssd-summary.json")["scopes"]["full"]
    final_tiers = final["tiers"]
    clocks = []
    mr = {r["id"]: r for r in load(pred / "requests.json")}
    native_adm = {
        names[e["request_id"]]: e["time_ns"] - zero for e in ev if e["event"] == "request_admitted"
    }
    for e in ev:
        if e["event"] == "request_finished":
            rid = names[e["request_id"]]
            r = mr[rid]
            clocks.append(
                {
                    "request": rid,
                    "model_start_ns": r["admitted_ns"] - mzero,
                    "native_start_ns": native_adm[rid],
                    "model_finish_ns": r["finished_ns"] - mzero,
                    "native_finish_ns": e["time_ns"] - zero,
                }
            )
    write(
        out / "score.json",
        {
            "freeze": base["matches_original_freeze"],
            "object_checks": base["checks"],
            "ssd_counts": counts,
            "object_sets_match": (a == b),
            "positive_native_disk_read": counts["read"]["native"]["objects"] > 0,
            "boundary_model_minus_native_ms": pairs,
            "foreground": clocks,
            "final_observed_ns": final["time_ns"] - zero,
            "last_request_finished_ns": end,
            "submitted_after_foreground_bytes": sum(
                r["bytes"] for r in demand if r["submit_ns"] > end
            ),
            "peak_10s_bytes": {
                op: {
                    "model": model_scope[op + "_peak_10s_bytes"],
                    "native": max(c[op + "_bytes"] for c in actual_curve),
                }
                for op in ("read", "write")
            },
            "inventory": {
                "memory_keys": sum("MEMORY" in t for t in final_tiers.values()),
                "disk_keys": sum("LOCAL_DISK" in t for t in final_tiers.values()),
                "committed_keys": len({r["key"] for r in stages["memory_ready"]}),
            },
            "clock_source": "Python perf_counter and C++ CLOCK_MONOTONIC aligned at first admission",
            "native_sha256": {
                p: sha(trial / p)
                for p in ("server/events.jsonl", "store-boundaries.tsv", "native-launch.json")
            },
        },
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="mode", required=True)
    f = sub.add_parser("fit")
    s = sub.add_parser("score")
    for parser in (f, s):
        parser.add_argument("--native", required=True, type=Path)
        parser.add_argument("--output", required=True, type=Path)
    s.add_argument("--prediction", required=True, type=Path)
    args = p.parse_args()
    if args.mode == "fit":
        fit(args.native, args.output)
    else:
        score(args.prediction, args.native, args.output)
