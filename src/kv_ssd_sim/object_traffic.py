"""Summarize object demand and compute byte reuse distance in the observed Store access stream."""

import json
from array import array
from collections import Counter, defaultdict


class LastTouches:
    """Fenwick tree of distinct payload bytes at each key's latest access position."""

    def __init__(self, size):
        self.values = array("q", [0]) * (size + 1)

    def add(self, position, delta):
        while position < len(self.values):
            self.values[position] += delta
            position += position & -position

    def prefix(self, position):
        total = 0
        while position:
            total += self.values[position]
            position -= position & -position
        return total


def describe_traffic(path, output, max_touches):
    """Count all rank GETs, and report reuse across requests separately from intra-request GETs."""
    last = {}
    unique_bytes = 0
    reused = 0
    index = 0
    tree = LastTouches(max_touches)
    counts = defaultdict(Counter)
    sizes = defaultdict(Counter)
    with path.open() as source, output.open("x") as target:
        for line in source:
            e = json.loads(line)
            op = e["event"]
            if op not in (
                "put_complete",
                "get_memory",
                "get_disk",
                "disk_publish",
                "put_rejected",
                "offload_rejected",
            ):
                continue
            rid, key, size, now = e["request"], e["key"], e["bytes"], e["time_ns"]
            counts[rid][op] += 1
            sizes[rid][op] += size
            if op not in ("put_complete", "get_memory", "get_disk"):
                continue
            index += 1
            prev = last.get(key)
            if prev:
                pos, previous_request, previous_time = prev
                if previous_request != rid:
                    distance = unique_bytes - tree.prefix(pos)
                    row = {
                        "key": key,
                        "bytes": size,
                        "previous_request": previous_request,
                        "request": rid,
                        "event": op,
                        "time_ns": now,
                        "interval_ns": now - previous_time,
                        "distinct_intervening_bytes": distance,
                    }
                    target.write(json.dumps(row) + "\n")
                    reused += 1
                tree.add(pos, -size)
            else:
                unique_bytes += size
            tree.add(index, size)
            last[key] = (index, rid, now)
    if index != max_touches:
        raise ValueError("Mismatch between event ledger and object counts")
    return {rid: {"counts": dict(counts[rid]), "bytes": dict(sizes[rid])} for rid in counts}, reused


def export_demand(folder, events, end_ns, requests):
    """Join every submitted object to its own dispatch and completion, including the drain."""
    import csv
    import gzip

    pending = {}
    completed = []
    anchors = []
    with events.open() as source:
        for line in source:
            e = json.loads(line)
            name = e["event"]
            if name == "measurement_anchor":
                anchors.append(e)
            if name not in {
                "disk_write_submit",
                "disk_read_submit",
                "disk_read_start",
                "disk_write_start",
                "disk_publish",
                "disk_read_complete",
            }:
                continue
            identity = (e["io_id"], e["key"])
            if name.endswith("_submit"):
                if identity in pending:
                    raise ValueError(f"Duplicate SSD submission {identity}")
                pending[identity] = {
                    "op": "read" if name == "disk_read_submit" else "write",
                    "request": e["request"],
                    "rank": e["rank"],
                    "key": e["key"],
                    "bytes": e["bytes"],
                    "owner": e["owner"],
                    "staging_client": e["staging_client"],
                    "io_id": e["io_id"],
                    "submit_ns": e["time_ns"],
                }
            elif name.endswith("_start"):
                pending[identity]["start_ns"] = e["time_ns"]
            else:
                r = pending.pop(identity)
                r.update(complete_ns=e["time_ns"], bucket_id=e["bucket_id"])
                if not r["submit_ns"] <= r["start_ns"] <= r["complete_ns"] <= end_ns:
                    raise ValueError(f"SSD lifecycle out of order: {identity}")
                completed.append(r)
    if pending:
        raise ValueError("Unfinished SSD submissions at final boundary")
    completed.sort(key=lambda r: (r["submit_ns"], r["io_id"], r["key"]))
    with gzip.open(folder / "ssd-demand.jsonl.gz", "xt") as target:
        for r in completed:
            target.write(json.dumps(r) + "\n")
    width = 10_000_000_000
    begin = min(r["arrival_ns"] for r in requests)
    foreground_end = max(r["finished_ns"] for r in requests)

    def bins(start):
        result = [
            {
                "from_ns": start + i * width,
                "until_ns": min(end_ns, start + (i + 1) * width),
                "read_objects": 0,
                "read_bytes": 0,
                "write_objects": 0,
                "write_bytes": 0,
            }
            for i in range(1 + (end_ns - start) // width)
        ]
        for r in completed:
            if r["submit_ns"] >= start:
                b = result[(r["submit_ns"] - start) // width]
                b[r["op"] + "_objects"] += 1
                b[r["op"] + "_bytes"] += r["bytes"]
        return result

    boundaries = {"full": begin}
    if len(anchors) == 1:
        boundaries["measurement"] = anchors[0]["time_ns"]
    curves = {}
    with (folder / "ssd-10s.csv").open("x", newline="") as target:
        writer = None
        for name, start in boundaries.items():
            series = bins(start)
            if writer is None:
                writer = csv.DictWriter(target, ["scope", *series[0]])
                writer.writeheader()
            for b in series:
                writer.writerow(dict(scope=name, **b))
            curves[name] = dict(
                start_ns=start,
                end_ns=end_ns,
                bins=len(series),
                **{
                    op + "_bytes": sum(b[op + "_bytes"] for b in series) for op in ("read", "write")
                },
                **{
                    op + "_peak_10s_bytes": max(b[op + "_bytes"] for b in series)
                    for op in ("read", "write")
                },
            )
    return {
        "scopes": curves,
        "anchors": anchors,
        "last_request_finished_ns": foreground_end,
        "submitted_after_last_finish_bytes": sum(
            r["bytes"] for r in completed if r["submit_ns"] > foreground_end
        ),
        "completed_after_last_finish_bytes": sum(
            r["bytes"] for r in completed if r["complete_ns"] > foreground_end
        ),
    }
