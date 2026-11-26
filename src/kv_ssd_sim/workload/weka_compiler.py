"""Compile one WEKA trace into causal arrivals and the online simulator's token IDs."""

import argparse
import hashlib
import json
import math
from pathlib import Path

from ..core.paths import PROJECT_ROOT as ROOT
from .aiperf import digest, load_conversations


def compile_trace(source, context_limit, artifacts):
    trace = json.loads(source.read_bytes())
    if not trace.get("id") or trace.get("hash_id_scope") != "local":
        raise ValueError("Require an identified local-hash WEKA trace")
    expected = {}
    for outer, entry in enumerate(trace["requests"]):
        if entry["type"] == "subagent":
            for inner, child in enumerate(entry["requests"]):
                expected[outer, inner] = child
        else:
            expected[outer, None] = entry
    for loc, item in expected.items():
        if (
            item["type"] not in ("n", "s")
            or min(item["in"], item["out"]) < 1
            or item["in"] + item["out"] > context_limit
        ):
            raise ValueError(f"Unsupported turn or context at {loc}")
    tokenizer_dir = ROOT / "configs/sim-tokenizer"
    conversations, tokenizer = load_conversations(
        source,
        tokenizer_dir,
        artifacts,
        context_limit=context_limit,
        request_count=len(expected),
        wire=True,
        synthesis={"max_isl": context_limit, "max_osl": max(v["out"] for v in expected.values())},
    )
    from aiperf.endpoints.openai_chat import ChatEndpoint

    endpoint = ChatEndpoint(model_endpoint=None)
    sessions = {c.session_id: c for c in conversations}
    branches = {b.branch_id: b for c in conversations for b in c.branches}
    rows = {}
    seen = set()
    for conv in conversations:
        history = []
        if conv.context_mode != "deltas_with_responses":
            raise ValueError(f"Unexpected WEKA context mode {conv.context_mode}")
        for index, turn in enumerate(conv.turns):
            loc = (turn.source_outer_idx, turn.source_inner_idx)
            if turn.no_request or loc not in expected or loc in seen:
                raise ValueError(f"Unmapped, synthetic or duplicate AIPerf turn at {loc}")
            seen.add(loc)
            original = expected[loc]
            if turn.reset_context:
                history.clear()
            history.append(turn)
            messages = endpoint.build_messages(history)
            tokens = tokenizer.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True, return_dict=False
            )
            if len(tokens) != original["in"] or turn.max_tokens != original["out"]:
                raise ValueError(f"AIPerf wire lengths differ from source at {loc}")
            rid = f"{conv.session_id}/{index}"
            edges = []
            if index:
                pause = turn.delay or 0
                if original.get("think_time") is not None and not math.isclose(
                    pause, original["think_time"] * 1000, abs_tol=0.001
                ):
                    raise ValueError(f"Think time changed at {loc}")
                edges.append((f"{conv.session_id}/{index - 1}", "finish", pause))
            for ref in turn.replay_predecessors:
                edges.append((f"{ref.conversation_id}/{ref.turn_index}", "finish", 0))
            for gate in turn.prerequisites:
                if gate.kind != "spawn_join":
                    raise ValueError(f"Unsupported prerequisite {gate.kind}")
                for sid in branches[gate.branch_id].child_conversation_ids:
                    edges.append((f"{sid}/{len(sessions[sid].turns) - 1}", "finish", 0))
            rows[rid] = {
                "id": rid,
                "session": conv.session_id,
                "source_index": loc,
                "source_ms": turn.timestamp,
                "dependencies": edges,
                "root_arrival_ms": (turn.timestamp or 0)
                if not index and not conv.parent_conversation_id
                else 0,
                "tokens": tokens,
                "output_tokens": turn.max_tokens,
                "prompt_sha256": hashlib.sha256(
                    b"".join(t.to_bytes(4, "big") for t in tokens)
                ).hexdigest(),
            }

    for conv in conversations:
        for index, turn in enumerate(conv.turns):
            for bid in turn.branch_ids:
                branch = branches[bid]
                if branch.mode != "spawn":
                    raise ValueError(f"Unsupported branch mode {branch.mode}")
                end = turn.timestamp + (turn.api_time_ms or 0)
                overlaps = branch.start_timestamp_ms < end
                origin = turn.timestamp if overlaps else end
                for sid in branch.child_conversation_ids:
                    child = rows[f"{sid}/0"]
                    delay = max(0, child["source_ms"] - origin)
                    child["dependencies"].append(
                        (f"{conv.session_id}/{index}", "start" if overlaps else "finish", delay)
                    )
    if seen != expected.keys():
        raise ValueError(f"Missing source turns: {expected.keys() - seen}")
    return {
        "schema": 1,
        "trace_id": trace["id"],
        "source_sha256": digest(source),
        "aiperf_version": "0.13.0",
        "random_seed": 42,
        "context_limit": context_limit,
        "tokenizer_sha256": {
            p.name: digest(p) for p in sorted(tokenizer_dir.iterdir()) if p.is_file()
        },
        "requests": list(rows.values()),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--context-limit", type=int, default=12288)
    args = parser.parse_args()
    compiled = compile_trace(args.input, args.context_limit, args.output.parent)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as f:
        json.dump(compiled, f, separators=(",", ":"))
        f.write("\n")
    print(f"{args.output}: {len(compiled['requests'])} requests")


if __name__ == "__main__":
    main()
