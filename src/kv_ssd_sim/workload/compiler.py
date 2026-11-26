"""Compile indexed kvlab sessions into deterministic replay requests."""

from __future__ import annotations

import hashlib
import json

from .corpus import LIMITS, MS, REVISION, block_spec, dump, session_calls


def synthetic_tokens(n, seed):
    # Reversible single-character vocabulary; avoid tokenizer control/special token spellings.
    raw = hashlib.shake_256(seed.encode()).digest(2 * n)
    return [0x3404 + (a * 256 + b) % 6400 for a, b in zip(raw[::2], raw[1::2], strict=True)]


def render(request, rid, lengths=None):
    # One redacted content hash represents one immutable block. Its reported
    # token estimate can drift across calls. Learn lengths in arrival order only.
    if lengths is None:
        lengths = {}
    target = request["input_tokens"]
    tokens = []
    declared = 0
    unknown = 0
    adjusted = drift = 0
    for index, (role, key, measured) in enumerate(block_spec(request)):
        size = lengths.setdefault((role, key), measured) if key else measured
        adjusted += size != measured
        drift += size - measured
        declared += size
        take = min(target - len(tokens), size)
        if not key:
            key = f"unknown/{rid}/{index}"
            unknown += take
        tokens.extend(synthetic_tokens(take, role + "/" + key))
    residual = target - len(tokens)
    tokens.extend(synthetic_tokens(residual, "unobserved-tail/" + rid))
    return tokens, {
        "fingerprinted_tokens": len(tokens) - residual - unknown,
        "unshared_tail_tokens": residual,
        "unidentified_block_tokens": unknown,
        "clipped_representation_tokens": max(0, declared - target),
        "blocks_using_prior_length": adjusted,
        "canonical_minus_reported_tokens": drift,
    }


def common_prefix(left, right):
    return next(
        (i for i, (a, b) in enumerate(zip(left, right)) if a != b), min(len(left), len(right))
    )


def compile_window(source, db, choice, out, limit, source_sha):
    out.mkdir()
    rows = []
    requests = {}
    fingerprint_records = []
    for sid in choice["sessions"]:
        offset, length, checksum = db.execute(
            "SELECT offset,length,sha256 FROM sessions WHERE sid=?", (sid,)
        ).fetchone()
        with source.open("rb") as stream:
            stream.seek(offset)
            line = stream.readline(length + 2)
        if hashlib.sha256(line).hexdigest() != checksum:
            raise ValueError(f"Changed source line {sid}")
        calls = session_calls(json.loads(line))
        for c in calls:
            if c.start < choice["start"] or [c.start, sid, c.ordinal] > choice["last_order"]:
                continue
            r = c.request
            rid = f"{sid}/{r['request_id']}"
            if not c.closed_mask & (1 << LIMITS.index(limit)):
                raise ValueError(f"Uncovered request {rid}")
            requests[rid] = r
            edges = []
            if c.parent is not None:
                parent = calls[c.parent]
                edges.append([f"{sid}/{parent.request['request_id']}", c.marker, c.delay_ns / MS])
            rows.append(
                {
                    "id": rid,
                    "session": sid,
                    "source_index": [c.instance, c.number],
                    "source_request_id": r["request_id"],
                    "source_utc_ns": c.start,
                    "source_end_utc_ns": c.finish,
                    "think_ref": r["think_ref"],
                    "think_ms": r["think_ms"],
                    "total_ms": r["total_ms"],
                    "input_tokens": r["input_tokens"],
                    "source_usage": {
                        k: r.get(k)
                        for k in (
                            "model",
                            "max_tokens",
                            "prefill_tokens",
                            "cache_read_tokens",
                            "cache_creation_tokens",
                        )
                    },
                    "observed_overlap_ns": c.overlap_ns,
                    "dependencies": edges,
                    "root_arrival_ms": (c.start - choice["start"]) / MS if not edges else 0,
                    "source_ms": (c.start - choice["start"]) / MS,
                    "measurement_anchor": rid == choice["anchor"],
                    "tokens": None,
                    "output_tokens": r["output_tokens"],
                    "prompt_sha256": None,
                }
            )
    rows.sort(key=lambda r: (r["source_utc_ns"], r["session"], r["source_index"]))
    lengths = {}
    history = []
    for row in rows:
        rid = row["id"]
        tokens, audit_row = render(requests[rid], rid, lengths)
        row["tokens"] = tokens
        row["prompt_sha256"] = hashlib.sha256(
            b"".join(t.to_bytes(4, "big") for t in tokens)
        ).hexdigest()

        best, other = max(
            ((common_prefix(tokens, r["tokens"]), r["id"]) for r in history), default=(0, None)
        )
        fingerprint_records.append(
            dict(
                id=rid,
                max_previous_shared_tokens=best,
                previous_request=other if best else None,
                **audit_row,
            )
        )
        history.append(row)
    ids = {r["id"] for r in rows}
    if (
        len(ids) != choice["calls"]
        or choice["anchor"] not in ids
        or any(edge[0] not in ids for r in rows for edge in r["dependencies"])
    ):
        raise ValueError("Window does not contain all source calls and predecessors")
    raw = {r["id"]: r for r in rows}
    for row in rows:
        due = choice["start"] + round(row["root_arrival_ms"] * MS)
        for parent, marker, delay in row["dependencies"]:
            base = "source_utc_ns" if marker == "start" else "source_end_utc_ns"
            due = max(due, raw[parent][base] + round(delay * MS))
        if due != row["source_utc_ns"]:
            raise ValueError(f"Source-clock replay differs from UTC for {row['id']}")
    compiled = {
        "schema": 1,
        "trace_id": f"kvlab/{choice['start']}",
        "source_sha256": source_sha,
        "context_limit": limit,
        "requests": rows,
    }
    input_file = out / "input.json"
    input_file.write_text(json.dumps(compiled, separators=(",", ":")) + "\n")
    manifest = {
        **choice,
        "source_sha256": source_sha,
        "revision": REVISION,
        "context_limit": limit,
        "input_sha256": hashlib.sha256(input_file.read_bytes()).hexdigest(),
        "initial_state": "cold at recording idle boundary; complete history of contributing sessions",
        "root_routing": "all recorded arrivals on one simulated server",
        "prefix_method": "kvlab BuildReplayRequestPrefix order; fix each (role, hash) to its first observed length in this window; isolate unobserved tail per request",
        "prefix_accounting": fingerprint_records,
        "recorded_arrival_check": "all arrivals reproduced from roots, edges and source durations at UTC nanosecond precision",
        "source_requests": [{k: v for k, v in r.items() if k != "tokens"} for r in rows],
    }
    dump(out / "window.json", manifest)
    return manifest
