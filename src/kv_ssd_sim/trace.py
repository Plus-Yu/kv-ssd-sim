"""WEKA normalization without text/tokenizer round trips.

Only observed input prefix identity is reconstructed. Generated output is
synthetic and is not claimed to match the next turn's original input.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ReplayRequest:
    request_id: str
    trace_id: str
    session_id: str
    arrival_s: float
    think_time_s: float
    dependencies: tuple[str, ...]
    prompt_token_ids: tuple[int, ...]
    output_tokens: int
    source_model: str


def synthesize_block(scope: str, identity: str, block_size: int, vocab_size: int) -> list[int]:
    if block_size <= 0 or vocab_size <= 4:
        raise ValueError("Invalid block size or vocabulary size")
    # IDs 0..3 are reserved. Include the trace namespace to prevent cross-trace reuse.
    seed = f"{scope}\0{identity}".encode()
    return [
        4
        + int.from_bytes(hashlib.sha256(seed + i.to_bytes(4, "big")).digest()[:8], "big")
        % (vocab_size - 4)
        for i in range(block_size)
    ]


def _prompt(record: dict, scope: str, request_id: str, block_size: int, vocab: int) -> tuple:
    length = record.get("in")
    hashes = record.get("hash_ids")
    if isinstance(length, bool) or not isinstance(length, int) or length <= 0:
        raise ValueError(f"{request_id}: 'in' must be a positive integer")
    if not isinstance(hashes, list):
        raise TypeError(f"{request_id}: missing hash_ids list")
    if len(hashes) not in {length // block_size, math.ceil(length / block_size)}:
        raise ValueError(f"{request_id}: hash count does not match input token length")
    tokens = []
    for block_hash in hashes:
        tokens.extend(synthesize_block(scope, str(block_hash), block_size, vocab))
    if len(tokens) < length:
        # A missing partial tail has no observable identity. Never share it accidentally.
        tokens.extend(synthesize_block(scope, f"tail:{request_id}", block_size, vocab))
    return tuple(tokens[:length])


def normalize_trace(trace: dict[str, Any], *, vocab_size: int = 32000) -> list[ReplayRequest]:
    trace_id = str(trace.get("id", ""))
    if not trace_id:
        raise ValueError("Every trace needs a non-empty id")
    scope = trace.get("hash_id_scope", "local")
    if scope != "local":
        raise ValueError("This prototype requires explicit per-trace local hash scope")
    block_size = trace.get("block_size", 64)
    if isinstance(block_size, bool) or not isinstance(block_size, int) or block_size <= 0:
        raise ValueError("block_size must be a positive integer")
    entries = trace.get("requests")
    if not isinstance(entries, list):
        raise TypeError("requests must be a list")
    result = []
    previous_parent: str | None = None
    pending_children: list[str] = []

    def add(record, request_id, session_id, dependencies):
        arrival = record.get("t", 0)
        think = record.get("think_time", 0) or 0
        output = record.get("out")
        if not isinstance(arrival, (int, float)) or not math.isfinite(arrival) or arrival < 0:
            raise ValueError(f"{request_id}: t must be finite and non-negative")
        if not isinstance(think, (int, float)) or not math.isfinite(think) or think < 0:
            raise ValueError(f"{request_id}: think_time must be finite and non-negative")
        if isinstance(output, bool) or not isinstance(output, int) or output < 1:
            raise ValueError(f"{request_id}: out must be a positive integer")
        result.append(
            ReplayRequest(
                request_id=request_id,
                trace_id=trace_id,
                session_id=session_id,
                arrival_s=float(arrival),
                think_time_s=float(think),
                dependencies=tuple(dependencies),
                prompt_token_ids=_prompt(record, trace_id, request_id, block_size, vocab_size),
                output_tokens=output,
                source_model=str(record.get("model", "unspecified")),
            )
        )

    for i, entry in enumerate(entries):
        kind = entry.get("type", "n")
        if kind in {"n", "s"}:
            rid = f"{trace_id}/main/{i}"
            deps = ([previous_parent] if previous_parent else []) + pending_children
            add(entry, rid, f"{trace_id}/main", deps)
            previous_parent, pending_children = rid, []
        elif kind == "subagent":
            agent = str(entry.get("agent_id", i))
            session = f"{trace_id}/subagent/{agent}/{i}"
            previous_child = previous_parent
            children = entry.get("requests", [])
            if not isinstance(children, list):
                raise ValueError(f"{session}: requests must be a list")
            for j, child in enumerate(children):
                if child.get("type", "n") not in {"n", "s"}:
                    raise ValueError(f"{session}: nested group format is not supported")
                rid = f"{session}/{j}"
                add(child, rid, session, [previous_child] if previous_child else [])
                previous_child = rid
            if children:
                pending_children.append(previous_child)
        else:
            raise ValueError(f"Unsupported WEKA entry type: {kind!r}")
    # A parent preceding a spawn and the next parent form the observable join
    # convention. Full tool-level dependency reconstruction is not available.
    by_id = {r.request_id: r for r in result}
    for request in result:
        for dep in request.dependencies:
            if by_id[dep].arrival_s > request.arrival_s:
                raise ValueError(f"{request.request_id}: dependency has a later recorded timestamp")
    return result


def load_weka(path: Path, *, vocab_size: int = 32000) -> list[ReplayRequest]:
    raw = path.read_text(encoding="utf-8")
    try:
        data = json.loads(raw)
        traces = data if isinstance(data, list) else [data]
    except json.JSONDecodeError:
        traces = [json.loads(line) for line in raw.splitlines() if line.strip()]
    seen = set()
    result = []
    for trace in traces:
        requests = normalize_trace(trace, vocab_size=vocab_size)
        trace_id = str(trace.get("id", ""))
        if trace_id in seen:
            raise ValueError(f"Duplicate trace id: {trace_id}")
        seen.add(trace_id)
        result.extend(requests)
    return result


def save_normalized(requests: list[ReplayRequest], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for request in requests:
            stream.write(json.dumps(asdict(request), ensure_ascii=False) + "\n")
