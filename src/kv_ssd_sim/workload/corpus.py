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
