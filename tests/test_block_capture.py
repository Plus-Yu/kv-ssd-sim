"""Verify stream integrity, completions crossing a window, and time-bracket checks."""

import pytest

from kv_ssd_sim.analysis import block_capture as audit


def snapshot(label, t, reads, secs, depth):
    return {
        "label": label,
        "before_ns": t,
        "after_ns": t + 100,
        "stat": [reads, 0, secs, 0, 0, 0, 0, 0, depth, 0, 0],
    }


def test_completion_window_with_an_inflight_read():
    from kv_ssd_sim.analysis.block_io import Event

    blocks = [
        Event(5000, "issue", "259,0", "R", 0, 8, "x", None),
        Event(10500, "complete", "259,0", "R", 0, 8, "", 0),
    ]
    marks = [
        snapshot("s", 1000, 0, 0, 0),
        snapshot("mid", 8000, 0, 0, 1),
        snapshot("end", 14000, 1, 8, 0),
    ]
    a, b = audit.reconcile(blocks, marks)
    assert a["R"]["issued_bytes"] == 4096 and not a["R"]["completions"]
    assert b["R"]["sectors"] == 8 and not b["R"]["issues"]
    marks[1] = snapshot("mid", 10000, 0, 0, 1)
    with pytest.raises(ValueError, match="bracket"):
        audit.reconcile(blocks, marks)
    marks[1] = snapshot("mid", 8000, 0, 0, 0)
    with pytest.raises(ValueError, match="in-flight mismatch"):
        audit.reconcile(blocks, marks)
    marks[1] = snapshot("mid", 8000, 0, 0, 1)
    marks[-1]["stat"][2] = 0
    with pytest.raises(ValueError, match="diskstats"):
        audit.reconcile(blocks, marks)


def test_stream_receipt():
    counters = {
        "overrun": 0,
        "commit overrun": 0,
        "dropped events": 0,
        "entries": 0,
        "read events": 0,
    }
    receipt = {
        "mode": "stream",
        "device": "259,0",
        "output_device": 1,
        "target_device": 2,
        "per_cpu_start": {"cpu0": counters.copy()},
        "per_cpu_end": {"cpu0": counters.copy()},
    }
    receipt["per_cpu_end"]["cpu0"]["read events"] = 4
    assert audit.check_receipt(receipt, "259,0", "stream") == 4
    receipt["per_cpu_end"]["cpu0"]["overrun"] = 1
    with pytest.raises(ValueError, match="Lost"):
        audit.check_receipt(receipt, "259,0", "stream")


def test_plain_trace_requires_external_kernel_record_count():
    from kv_ssd_sim.analysis.block_io import parse_trace

    text = (
        "worker [001] ..... 1.1: block_rq_issue: 259,0 W 4096 () 0 + 8 [worker]\n"
        "worker [001] ..... 1.2: block_rq_complete: 259,0 W () 0 + 8 [0]\n"
    )
    assert parse_trace(text, "259,0", stream_entries=2)[1]["buffer_entries"] == 2
    with pytest.raises(ValueError, match="event count"):
        parse_trace(text, "259,0", stream_entries=3)
    with pytest.raises(ValueError, match="header"):
        parse_trace(text, "259,0")


def test_resolve_busy_counter_bracket_and_reject_inconsistent_state():
    from kv_ssd_sim.analysis.block_io import Event

    blocks = [
        Event(5000, "issue", "259,0", "R", 0, 8, "x", None),
        Event(9500, "complete", "259,0", "R", 0, 8, "", 0),
    ]
    marks = [snapshot("start", 1000, 0, 0, 0), snapshot("response", 9000, 1, 8, 0)]
    cuts = audit.counter_cuts(blocks, marks, [b.time for b in blocks])
    assert cuts == [0, 2]
    assert audit.reconcile(blocks, marks, cuts=cuts)[0]["R"]["diskstats_sectors"] == 8
    marks[1]["stat"][2] = 7
    with pytest.raises(ValueError, match="0 possible cuts"):
        audit.counter_cuts(blocks, marks, [b.time for b in blocks])
