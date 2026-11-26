"""Release requests on a supplied clock after source dependency edges fire."""

import hashlib
import json
import math
from collections import defaultdict, deque


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
