import csv
import gzip
import hashlib
import json
import sqlite3

from kv_ssd_sim.kvlab import (
    NS,
    SCHEMA,
    candidates,
    compile_window,
    insert_session,
    render,
    select_warm_window,
    select_windows,
    session_calls,
    utc_ns,
)
from kv_ssd_sim.object_traffic import export_demand


def request(rid, seconds, duration, ref, think, length=2200):
    return {
        "request_id": rid,
        "ts": f"2026-09-10T00:00:{seconds}Z",
        "total_ms": duration,
        "think_ref": ref,
        "think_ms": think,
        "input_tokens": length,
        "output_tokens": 10,
        "messages": [{"role": "user", "hash": "shared", "tokens": 2050}],
    }


def session(sid, rows):
    return {"session_id": sid, "start_ts": rows[0]["ts"], "instances": [{"requests": rows}]}


def test_dependency_clocks_overlap_missing_parent_and_context():
    rows = [
        request(1, "01", 3000, "session_start", 0),
        request(2, "03.125000001", 1000, "prev", 0),
        request(3, "04.625456789", 200, "prev", 500),
    ]
    s = session("s", rows)
    s["instances"].append(
        {"parent_spawn_request_id": 999, "requests": [request(4, "04", 1, "spawn", 0)]}
    )
    a, b, c, d = session_calls(s)
    assert a.closed_mask == 0b111
    assert (b.marker, b.delay_ns, b.overlap_ns) == ("start", 2_125_000_001, 874_999_999)
    assert (c.marker, c.delay_ns) == ("finish", 500_456_788)
    assert b.start + b.delay_ns != b.finish
    assert a.start + b.delay_ns == b.start and b.finish + c.delay_ns == c.start
    assert d.problem == "unresolved_reference" and not d.closed_mask
    rows[0]["input_tokens"] = 12280
    a, b, c, _ = session_calls(s)
    assert (
        a.direct_mask == 0b110
        and b.direct_mask == 0b111
        and b.closed_mask == c.closed_mask == 0b110
    )
    rows[0]["input_tokens"] = 131072
    a, b, c, _ = session_calls(s)
    assert a.direct_mask == b.closed_mask == c.closed_mask == 0b100
    rows[0]["input_tokens"] = 1048576
    a, b, c, _ = session_calls(s)
    assert not (a.closed_mask | b.closed_mask | c.closed_mask)
    assert utc_ns("1970-01-01T00:00:01.000000009Z") == NS + 9


def test_joined_window_preserves_competing_root_and_prior_call(tmp_path):
    rows = [
        session(
            "a", [request(1, "01", 1000, "session_start", 0), request(2, "04", 100, "prev", 2000)]
        ),
        session("b", [request(3, "01.25", 100, "session_start", 0)]),
        session("c", [request(4, "12", 100, "session_start", 0)]),
    ]
    source = tmp_path / "source"
    offset = 0
    with sqlite3.connect(":memory:") as db, source.open("wb") as f:
        db.executescript(SCHEMA)
        for s in rows:
            line = json.dumps(s).encode() + b"\n"
            f.write(line)
            insert_session(db, line, offset, s)
            offset += len(line)
        f.flush()
        pair, _ = select_windows(db, 12288)
        assert [p["calls"] for p in pair] == [3, 1]
        choice = pair[0]
        assert choice["quality"] == (1, 1, 1, 1) and choice["sessions"] == ["a", "b"]
        manifest = compile_window(
            source,
            db,
            choice,
            tmp_path / "out",
            12288,
            hashlib.sha256(source.read_bytes()).hexdigest(),
        )
        a, b, c = manifest["source_requests"]
        assert b["root_arrival_ms"] == 250 and c["dependencies"] == [["a/1", "finish", 2000]]
        assert c["measurement_anchor"] and not a["measurement_anchor"]
        warm = select_warm_window(db, 12288, 300, 24, 1)
        assert (warm["calls"], warm["anchor"], warm["first_interleaved_return"]) == (
            4,
            "b/3",
            "a/2",
        )
        assert warm["first_index"] == choice["first_index"]
        db.execute("UPDATE calls SET closed=0 WHERE sid='b'")
        assert all(p["quality"][0] == 0 for p in candidates(db, 12288, 300, 24))


def test_unknown_tail_does_not_create_fake_sharing_or_double_count_cache_tokens():
    r = request(1, "01", 100, "session_start", 0)
    r.update(prefill_tokens=10, cache_read_tokens=2000, cache_creation_tokens=190)
    a, c = render(r, "a")
    b, _ = render(r, "b")
    assert len(a) == r["input_tokens"] and a[:2050] == b[:2050] and a[2050:] != b[2050:]
    assert c["fingerprinted_tokens"] == 2050 and c["unshared_tail_tokens"] == 150
    r.update(input_tokens=1800)
    t, c = render(r, "c")
    assert t == a[:1800] and c["clipped_representation_tokens"] == 250


def test_submit_bins_span_drain_and_only_disk_gets(tmp_path):
    events = []

    def event(t, name, **attrs):
        events.append(dict(event=name, time_ns=t, **attrs))

    fields = {"request": "a", "rank": 2, "key": "same", "bytes": 7, "owner": 0, "group": 3}
    event(9 * NS, "disk_write_submit", **fields, staging_client=0, io_id=1)
    event(10 * NS, "disk_write_start", **fields, staging_client=0, io_id=1)
    event(12 * NS, "disk_publish", **fields, staging_client=0, io_id=1, bucket_id=8)
    event(15 * NS, "measurement_anchor", request="b")
    event(16 * NS, "get_memory", **fields)
    event(30 * NS, "disk_read_submit", **fields, staging_client=2, io_id=2, bucket_id=8)
    event(31 * NS, "disk_read_start", **fields, staging_client=2, io_id=2, bucket_id=8)
    event(32 * NS, "disk_read_complete", **fields, staging_client=2, io_id=2, bucket_id=8)
    path = tmp_path / "events.jsonl"
    path.write_text("".join(json.dumps(e) + "\n" for e in events))
    s = export_demand(tmp_path, path, 33 * NS, [{"arrival_ns": 0, "finished_ns": 29 * NS}])
    assert s["scopes"]["full"]["read_bytes"] == 7 and s["scopes"]["measurement"]["write_bytes"] == 0
    assert s["submitted_after_last_finish_bytes"] == 7
    with (tmp_path / "ssd-10s.csv").open() as f:
        full = [r for r in csv.DictReader(f) if r["scope"] == "full"]
    assert [int(r["write_bytes"]) for r in full] == [7, 0, 0, 0]
    with gzip.open(tmp_path / "ssd-demand.jsonl.gz", "rt") as f:
        assert [
            (r["op"], r["owner"], r["rank"], r["group"], r["bucket_id"]) for r in map(json.loads, f)
        ] == [
            ("write", 0, 2, 3, 8),
            ("read", 0, 2, 3, 8),
        ]


def test_immutable_hash_content_keeps_following_blocks_aligned_when_estimates_drift():
    r = request(1, "01", 100, "session_start", 0, 2500)
    r["messages"] = [
        {"role": "user", "hash": "A", "tokens": 1600},
        {"role": "assistant", "hash": "B", "tokens": 800},
    ]
    lengths = {}
    old, _ = render(r, "old", lengths)
    r["messages"][0]["tokens"] = 1590
    new, audit = render(r, "new", lengths)
    assert old[:2400] == new[:2400]
    assert old[2400:] != new[2400:] and len(new) == 2500
    assert (
        audit["canonical_minus_reported_tokens"] == 10 and audit["blocks_using_prior_length"] == 1
    )
