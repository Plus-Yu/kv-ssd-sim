"""Audit recorded cross-agent overlap using the pinned AIPerf input semantics."""

import argparse
import gzip
import hashlib
import json
import os
from collections import Counter, defaultdict
from importlib.metadata import version
from pathlib import Path

from aiperf.dataset.loader.weka_trace import _sa_end_seconds, _subagent_request_absolute_t
from aiperf.dataset.loader.weka_trace_models import WekaSubagentEntry, WekaTrace


def overlap(intervals):
    changes = defaultdict(Counter)
    for start, stop, actor in intervals:
        if stop > start:
            changes[start][actor] += 1
            changes[stop][actor] -= 1
    active = Counter()
    previous = min(changes)
    peak = 0
    simultaneous = 0
    for t, delta in sorted(changes.items()):
        n = sum(v > 0 for v in active.values())
        peak = max(n, peak)
        if n >= 2:
            simultaneous += t - previous
        active.update(delta)
        previous = t
    return {"peak_distinct_agents": peak, "seconds_with_multiple_agents": round(simultaneous, 6)}


def replay_graph(trace, source, output_dir):
    from aiperf.common import random_generator as rng
    from aiperf.common.tokenizer import Tokenizer
    from aiperf.config.config import BenchmarkConfig
    from aiperf.config.resolution.plan import BenchmarkRun
    from aiperf.dataset.generator.prompt import PromptGenerator
    from aiperf.dataset.loader.weka_trace import WekaTraceLoader

    expected = []
    agents = set()
    for i, entry in enumerate(trace.requests):
        if isinstance(entry, WekaSubagentEntry):
            expected.extend((i, j) for j in range(len(entry.requests)))
            agents.add(f"{trace.id}::sa:{entry.agent_id}")
        else:
            expected.append((i, None))
    # The tokenizer directory is local. AIPerf 0.13 rejects this path with its HF offline flag.
    os.environ.pop("HF_HUB_OFFLINE", None)
    os.environ.pop("TRANSFORMERS_OFFLINE", None)
    tokenizer = str(Path(__file__).resolve().parents[1] / "configs/sim-tokenizer")
    cfg = BenchmarkConfig(
        models=["kv-ssd-sim"],
        endpoint={"urls": ["http://127.0.0.1:8000"]},
        datasets=[
            {
                "name": "main",
                "type": "file",
                "path": str(source),
                "format": "weka_trace",
                "use_think_time_only": True,
            }
        ],
        phases=[
            {
                "name": "profiling",
                "type": "concurrency",
                "requests": len(expected),
                "concurrency": 2,
            }
        ],
        tokenizer={"name": tokenizer},
    )
    rng.init(42)
    run = BenchmarkRun(
        benchmark_id="offline-layout-audit", cfg=cfg, artifact_dir=output_dir, random_seed=42
    )
    pg = PromptGenerator(
        prompts=None, prefix_prompts=None, tokenizer=Tokenizer.from_pretrained(tokenizer)
    )
    loader = WekaTraceLoader(run=run, prompt_generator=pg)
    conversations = loader.convert_to_conversations({trace.id: [trace]})
    seen, children, sessions, branches = [], set(), {}, []
    for c in conversations:
        turns = [t for t in c.turns if not t.no_request]
        sessions[c.session_id] = len(turns)
        seen.extend((t.source_outer_idx, t.source_inner_idx) for t in turns)
        for b in c.branches:
            branches.append(b.model_dump(mode="json"))
            if str(b.mode).lower() == "spawn":
                children.update(b.child_conversation_ids)
    if Counter(seen) != Counter(expected) or not agents.issubset(children):
        raise ValueError("Loader request coverage or explicit SPAWN children differ")
    return {
        "request_count": len(seen),
        "requests_per_session": sessions,
        "branches": branches,
        "scope": "full uncapped offline conversion, no HTTP dispatch",
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("source", type=Path)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    raw = gzip.decompress(args.source.read_bytes())
    if (
        hashlib.sha256(raw).hexdigest()
        != "51a7888f72175765f6c2dd7a72eface5bbe7f3ee37bb6e22781344693dcfbb85"
    ):
        raise ValueError("Source differs from the pinned WEKA trace")
    data = WekaTrace.model_validate_json(raw)
    assert version("aiperf") == "0.13.0"
    assert data.id == "trace_0004"
    branches, calls, rows = [], [], []
    for i, entry in enumerate(data.requests):
        if not isinstance(entry, WekaSubagentEntry):
            continue
        branches.append((entry.t, _sa_end_seconds(entry), entry.agent_id))
        children = []
        for j, r in enumerate(entry.requests):
            start = _subagent_request_absolute_t(entry, r)
            assert r.api_time is not None and r.api_time >= 0
            end = start + r.api_time
            calls.append((start, end, entry.agent_id))
            children.append(
                {
                    "inner_idx": j,
                    "start_s": start,
                    "end_s": end,
                    "input": r.input_length,
                    "output": r.output_length,
                }
            )
        rows.append(
            {
                "outer_idx": i,
                "agent_id": entry.agent_id,
                "status": entry.status,
                "start_s": entry.t,
                "end_s": _sa_end_seconds(entry),
                "requests": children,
            }
        )
    report = {
        "source": "https://github.com/callanjfox/kv-cache-tester/blob/94f6046dfd9c0c8bc0cdb24a5aa6579bac8669bf/traces/trace_0004.json",
        "source_sha256": hashlib.sha256(raw).hexdigest(),
        "aiperf": version("aiperf"),
        "outer_entries": len(data.requests),
        "aiperf_graph": replay_graph(data, args.source, args.output.parent),
        "branches": rows,
        "recorded_branch_lifetimes": overlap(branches),
        "recorded_child_api_intervals": overlap(calls),
    }
    assert report["recorded_child_api_intervals"]["peak_distinct_agents"] >= 2
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
