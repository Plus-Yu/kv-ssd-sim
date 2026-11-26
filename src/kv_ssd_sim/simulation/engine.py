"""Causal admissions and native vLLM steps on the object-replay clock."""


def simulate(admissions, scheduler, transfers, store, hash_unit, timing, record_steps):
    from vllm.sampling_params import SamplingParams
    from vllm.utils.hashing import sha256
    from vllm.v1.core.kv_cache_utils import get_request_block_hasher, init_none_hash
    from vllm.v1.outputs import ModelRunnerOutput
    from vllm.v1.request import Request

    from ..workload.tokens import SYNTHETIC_TOKEN

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
