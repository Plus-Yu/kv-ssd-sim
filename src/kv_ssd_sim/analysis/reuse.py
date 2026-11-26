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
