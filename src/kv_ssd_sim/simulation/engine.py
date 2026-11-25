"""Causal admissions and native vLLM steps on the object-replay clock."""

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
