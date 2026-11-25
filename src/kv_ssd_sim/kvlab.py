"""Read the pinned kvlab replay-v5 corpus without expanding the corpus into token IDs."""

from __future__ import annotations

import calendar
import csv
import gzip
import hashlib
import json
import sqlite3
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

REVISION = "5afaa89882fc2c24a3dcb8bcdcd0485770c5021d"
SHA256 = "337a03810637e0d062c2f320bd6a17392a476fa19e196a80e3a7359a47731f61"
LIMITS = (12288, 131072, 1048576)
NS = 1_000_000_000
MS = 1_000_000
BIN = 10 * NS


def utc_ns(value):
    """RFC3339 UTC with nanosecond precision, including Go's optional fractional part."""
    if not value.endswith("Z"):
        raise ValueError(f"Expected UTC timestamp: {value}")
    seconds, _, fraction = value[:-1].partition(".")
    if len(fraction) > 9 or (fraction and not fraction.isdigit()):
        raise ValueError(f"Invalid fraction in {value}")
    dt = datetime.fromisoformat(seconds)
    return calendar.timegm(dt.timetuple()) * NS + int(fraction.ljust(9, "0"))


def dump(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def prefix_parts(request):
    """Ordered redacted fingerprints, following kvlab BuildReplayRequestPrefix."""
    for index, block in enumerate(request.get("system_blocks", [])):
        if index == 0 and block.get("bytes", 0) < 200:
            continue
        yield "system", block
    tools = request.get("tools")
    if tools and tools.get("hash"):
        yield "tools", tools
    for message in request.get("messages", []):
        yield message["role"], message


def block_spec(request):
    parts = []
    for role, block in prefix_parts(request):
        size = block.get("tokens", 0)
        if not isinstance(size, int) or size < 0:
            raise ValueError("Invalid redacted block token count")
        if size:
            parts.append((role, block.get("hash", ""), size))
    return parts


def prefix_signature(parts, input_tokens, unit=1536):
    if input_tokens < unit:
        return None
    prefix = []
    left = unit
    for role, digest, count in parts:
        if not digest:
            break
        take = min(count, left)
        prefix.append((role, digest, take))
        left -= take
        if not left:
            return hashlib.sha256(json.dumps(prefix, separators=(",", ":")).encode()).hexdigest()
    return None


@dataclass
class Call:
    ordinal: int
    instance: int
    number: int
    request: dict
    start: int
    finish: int
    parent: int | None = None
    marker: str | None = None
    delay_ns: int = 0
    overlap_ns: int = 0
    problem: str | None = None
    direct_mask: int = 0
    closed_mask: int = 0


def session_calls(session):
    by_id = defaultdict(list)
    calls = []
    previous = {}
    roots = {}
    start = utc_ns(session["start_ts"])
    for i, instance in enumerate(session["instances"]):
        roots[i] = instance.get("parent_spawn_request_id")
        for j, r in enumerate(instance["requests"]):
            ts = utc_ns(r["ts"])
            call = Call(len(calls), i, j, r, ts, ts + r["total_ms"] * MS)
            if j:
                previous[call.ordinal] = call.ordinal - 1
            by_id[r["request_id"]].append(call.ordinal)
            calls.append(call)
    for c in calls:
        r = c.request
        kind = r.get("think_ref")
        if not r["request_id"] or len(by_id[r["request_id"]]) != 1:
            c.problem = "ambiguous_request_id"
        elif kind == "prev" and c.number:
            c.parent = previous[c.ordinal]
        elif kind == "spawn" and not c.number and len(by_id.get(roots[c.instance], [])) == 1:
            c.parent = by_id[roots[c.instance]][0]
        elif kind != "session_start" or c.number or roots[c.instance]:
            c.problem = "unresolved_reference"
        p = calls[c.parent] if c.parent is not None else None
        ref_end = p.finish if p else start
        gap = c.start - ref_end
        think = r.get("think_ms")
        if r["total_ms"] < 0 or think != max(0, gap // MS):
            c.problem = c.problem or "timing_mismatch"
        if p:
            if gap < 0:
                c.overlap_ns = -gap
                c.delay_ns = c.start - p.start
                c.marker = "start"
            else:
                c.delay_ns = gap
                c.marker = "finish"
        else:
            c.delay_ns = c.start - start
        if c.delay_ns < 0:
            c.problem = c.problem or "backwards_reference"
        it, ot = r.get("input_tokens", 0), r.get("output_tokens", 0)
        if it > 0 and ot > 0:
            for i, limit in enumerate(LIMITS):
                if it + ot <= limit:
                    c.direct_mask |= 1 << i

    # Resolve the graph iteratively. Some captured parent IDs live in later instance rows.
    resolved = {}
    for c in calls:
        path, seen = [], set()
        node = c.ordinal
        while node is not None and node not in resolved:
            if node in seen:
                calls[node].problem = "dependency_cycle"
                break
            seen.add(node)
            path.append(node)
            node = calls[node].parent
        mask = (1 << len(LIMITS)) - 1 if node is None else resolved.get(node, 0)
        for node in reversed(path):
            current = calls[node]
            mask &= current.direct_mask if not current.problem else 0
            current.closed_mask = resolved[node] = mask
    return calls


SCHEMA = """
CREATE TABLE sessions(sid TEXT PRIMARY KEY, offset INTEGER, length INTEGER, sha256 TEXT,
    start INTEGER, finish INTEGER, requests INTEGER);
CREATE TABLE calls(sid TEXT, seq INTEGER, request_id INTEGER, instance INTEGER, at INTEGER,
    finish INTEGER, n_in INTEGER, n_out INTEGER, parent INTEGER, marker TEXT,
    ref TEXT, problem TEXT, direct INTEGER, closed INTEGER, overlap INTEGER, prefix TEXT,
    fingerprint_tokens INTEGER, PRIMARY KEY(sid,seq)) WITHOUT ROWID;
"""


def insert_session(db, raw, offset, session):
    sid = session["session_id"]
    calls = session_calls(session)
    first = utc_ns(session["start_ts"])
    end = max((c.finish for c in calls), default=first)
    db.execute(
        "INSERT INTO sessions VALUES(?,?,?,?,?,?,?)",
        (
            sid,
            offset,
            len(raw.rstrip(b"\r\n")),
            hashlib.sha256(raw).hexdigest(),
            first,
            end,
            len(calls),
        ),
    )

    def records():
        for c in calls:
            r = c.request
            it, ot = r.get("input_tokens", 0), r.get("output_tokens", 0)
            parts = block_spec(r)
            yield (
                sid,
                c.ordinal,
                r["request_id"],
                c.instance,
                c.start,
                c.finish,
                it,
                ot,
                c.parent,
                c.marker,
                r.get("think_ref"),
                c.problem,
                c.direct_mask,
                c.closed_mask,
                c.overlap_ns,
                prefix_signature(parts, it),
                sum(n for _, h, n in parts if h),
            )

    db.executemany("INSERT INTO calls VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", records())


def concurrency(db, out):
    query = """SELECT t, sum(d), sum(starts) FROM
       (SELECT at t, 1 d, 1 starts FROM calls WHERE finish>at UNION ALL
        SELECT finish t,-1 d,0 starts FROM calls WHERE finish>at)
        GROUP BY t ORDER BY t"""
    active = peak = 0
    prev = None
    histogram = Counter()
    with gzip.open(out, "wt", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["utc_ns", "starts", "completions", "in_flight_after"])
        for ts, delta, starts in db.execute(query):
            if prev is not None:
                histogram[active] += ts - prev
            active += delta
            peak = max(peak, active)
            prev = ts
            writer.writerow((ts, starts, starts - delta, active))
    if active:
        raise ValueError("Unbalanced recorded request intervals")
    return {"peak": peak, "duration_ns_by_in_flight": dict(sorted(histogram.items()))}


def coverage(db, index_counts):
    summary = {
        "sessions": index_counts,
        "requests": db.execute("SELECT count(*) FROM calls").fetchone()[0],
    }
    summary["recorded_input_tokens"] = db.execute("SELECT sum(n_in) FROM calls").fetchone()[0]
    summary["nonempty_instances"] = db.execute(
        "SELECT count(*) FROM (SELECT sid, instance FROM calls GROUP BY sid,instance)"
    ).fetchone()[0]
    summary["missing_usage"] = db.execute(
        "SELECT count(*) FROM calls WHERE n_in<=0 OR n_out<=0"
    ).fetchone()[0]
    summary["zero_input"] = db.execute("SELECT count(*) FROM calls WHERE n_in<=0").fetchone()[0]
    summary["zero_output"] = db.execute("SELECT count(*) FROM calls WHERE n_out<=0").fetchone()[0]
    summary["structural_issues"] = dict(
        db.execute("SELECT problem,count(*) FROM calls WHERE problem IS NOT NULL GROUP BY problem")
    )
    summary["observed_overlaps"] = db.execute(
        "SELECT count(*) FROM calls WHERE overlap>0"
    ).fetchone()[0]
    summary["clamped_whole_ms"] = db.execute(
        "SELECT count(*) FROM calls WHERE overlap>=?", (MS,)
    ).fetchone()[0]
    summary["timing_refs"] = dict(db.execute("SELECT ref,count(*) FROM calls GROUP BY ref"))
    summary["limits"] = {}
    for i, limit in enumerate(LIMITS):
        bit = 1 << i

        def totals(condition, arg):
            r = db.execute(
                "SELECT count(*),coalesce(sum(n_in),0) FROM calls WHERE " + condition, (arg,)
            ).fetchone()
            return {"requests": r[0], "input_tokens": r[1]}

        summary["limits"][str(limit)] = {
            "over_context": totals("n_in+n_out>?", limit),
            "direct": totals("direct & ?", bit),
            "dependency_closed": totals("closed & ?", bit),
            "direct_but_unclosed": totals("(direct & ~closed) & ?", bit),
            "complete_sessions": db.execute(
                "SELECT count(*) FROM (SELECT sid FROM calls GROUP BY sid HAVING min(closed & ?)>0)",
                (bit,),
            ).fetchone()[0],
        }
    return summary


def audit(source: Path, out: Path, expected_sha=SHA256):
    h = hashlib.sha256()
    with sqlite3.connect(out / "index.sqlite") as db, source.open("rb") as stream:
        db.executescript(SCHEMA)
        line = stream.readline()
        h.update(line)
        header = json.loads(line)
        if header.get("_schema") != "replay-v5":
            raise ValueError("Expected replay-v5")
        trailer = None
        while raw := stream.readline():
            offset = stream.tell() - len(raw)
            h.update(raw)
            row = json.loads(raw)
            if trailer is not None:
                raise ValueError("Data after index")
            if row.get("_schema") == "replay-index-v1":
                trailer = row["sessions"]
            else:
                insert_session(db, raw, offset, row)
        if h.hexdigest() != expected_sha or trailer is None:
            raise ValueError("Source digest or trailer mismatch")
        actual = list(
            db.execute(
                "SELECT sid,offset,length,requests,start,finish FROM sessions ORDER BY offset"
            )
        )
        expected = [
            (
                r["session_id"],
                r["offset"],
                r["length"],
                r["requests"],
                utc_ns(r["start_ts"]),
                utc_ns(r["end_ts"]),
            )
            for r in trailer
        ]
        if actual != expected:
            raise ValueError(
                f"Session index mismatch: {next(((a, b) for a, b in zip(actual, expected) if a != b), (len(actual), len(expected)))}"
            )
        db.executescript("CREATE INDEX by_time ON calls(at); CREATE INDEX by_end ON calls(finish);")
        report = coverage(db, len(actual))
        if report["clamped_whole_ms"] != header["summary"]["timing"]["think_clamped"]:
            raise ValueError("Signed source gaps disagree with the publisher's clamped count")
        if (report["sessions"], report["requests"]) != (
            header["summary"]["sessions"],
            header["summary"]["requests"],
        ):
            raise ValueError("Header count mismatch")
        report.update(
            source_sha256=h.hexdigest(),
            source_size=stream.tell(),
            header=header,
            source_concurrency=concurrency(db, out / "recorded-concurrency.csv.gz"),
        )
        dump(out / "audit.json", report)
    return report


@dataclass(frozen=True)
class Arrival:
    at: int
    end: int
    sid: str
    seq: int
    rid: int
    session_start: int
    closed: int
    prefix: str | None
    instance: int
    n_in: int


def candidates(db, limit, max_span_s, max_calls):
    """Cold-start prefixes with no left-censored in-flight call or omitted intervening arrival."""
    bit = 1 << LIMITS.index(limit)
    data = [
        Arrival(*r)
        for r in db.execute("""
        SELECT c.at,c.finish,c.sid,c.seq,c.request_id,s.start,c.closed,c.prefix,c.instance,c.n_in
        FROM calls c JOIN sessions s USING(sid) ORDER BY c.at,c.sid,c.seq""")
    ]
    result = []
    prev_end = 0
    for lo, first in enumerate(data):
        quiet = prev_end <= first.at and (not lo or data[lo - 1].at < first.at)
        prev_end = max(prev_end, first.end)
        if not quiet or first.at != first.session_start or not first.closed & bit:
            continue
        sessions = {}
        last_prefix = {}
        latest, anchor, revisit, concurrent = first.at, None, False, False
        has_object = False
        for hi in range(lo, min(len(data), lo + max_calls)):
            row = data[hi]
            latest = max(latest, row.end)
            if (
                row.session_start < first.at
                or not row.closed & bit
                or latest - first.at > max_span_s * NS
            ):
                break
            if any(end > row.at for sid, (_, end) in sessions.items() if sid != row.sid):
                concurrent = True
            if row.prefix:
                key = (row.sid, row.instance, row.prefix)
                if key in last_prefix:
                    revisit = True
                    if anchor is None and any(
                        j > last_prefix[key] for sid, (j, _) in sessions.items() if sid != row.sid
                    ):
                        anchor = hi
                last_prefix[key] = hi
                has_object = True
            sessions[row.sid] = hi, max(row.end, sessions.get(row.sid, (0, 0))[1])
            # Include every source submission through the end of the last modeled request.
            if hi + 1 < len(data) and (latest > data[hi + 1].at or row.at == data[hi + 1].at):
                continue
            if not has_object:
                continue
            # Prefer a real return across a competing session, then concurrent sessions,
            # then a within-session return. Within a class take the earliest shortest prefix.
            level = (int(anchor is not None), int(len(sessions) > 1), int(concurrent), int(revisit))
            metric_row = data[anchor if anchor is not None else hi]
            result.append(
                {
                    "quality": level,
                    "start": first.at,
                    "end": latest,
                    "anchor": f"{metric_row.sid}/{metric_row.rid}",
                    "calls": hi - lo + 1,
                    "sessions": sorted(sessions),
                    "first_index": lo,
                    "last_index": hi,
                    "last_order": [row.at, row.sid, row.seq],
                }
            )
    return sorted(result, key=lambda x: (tuple(-v for v in x["quality"]), x["start"], x["calls"]))


def select_windows(db, limit, max_span_s=300, max_calls=24):
    choices = candidates(db, limit, max_span_s, max_calls)
    # Evaluate both windows without referring to a model prediction or an online result.
    counts = Counter(tuple(w["quality"]) for w in choices)
    for i, first in enumerate(choices):
        for other in choices[i + 1 :]:
            if (other["end"] <= first["start"] or other["start"] >= first["end"]) and set(
                first["sessions"]
            ).isdisjoint(other["sessions"]):
                return sorted((first, other), key=lambda w: w["start"]), counts
    return choices[:1], counts


def select_warm_window(db, limit, max_span_s, max_calls, warmup_requests):
    """Longest complete window with a return after a competing session, on source data alone."""
    if warmup_requests < 1:
        raise ValueError("A warm window requires at least one preceding request")
    choices = candidates(db, limit, max_span_s, max_calls)
    eligible = (w for w in choices if w["quality"][0] and w["calls"] > warmup_requests)
    window = max(eligible, key=lambda w: (w["calls"], -w["start"]), default=None)
    if window is None:
        raise ValueError("No complete interleaved window longer than the requested warmup")
    sid, rid = db.execute(
        "SELECT sid,request_id FROM calls WHERE at>=? ORDER BY at,sid,seq LIMIT 1 OFFSET ?",
        (window["start"], warmup_requests),
    ).fetchone()
    return dict(
        window,
        first_interleaved_return=window["anchor"],
        anchor=f"{sid}/{rid}",
        warmup_source_requests=warmup_requests,
        selection_rule="longest complete interleaved-return window, earliest start on ties",
    )


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
