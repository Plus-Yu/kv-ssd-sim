"""Select closed kvlab measurement and warmup windows."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from .corpus import LIMITS, NS


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
