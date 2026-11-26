"""Check trace loss, ambiguity, and window-boundary accounting on small kernels traces."""

import pytest

from kv_ssd_sim.analysis import block_io as measure


def line(t, stage="issue", op="RA", sector=2047, count=8, error=0, device="259,0"):
    description = f"{count * 512} ()" if stage == "issue" else "()"
    tail = "thread" if stage == "issue" else error
    return f"worker [001] ..... {t}: block_rq_{stage}: {device} {op} {description} {sector} + {count} be,0,4 [{tail}]"


def trace(*lines):
    return f"# entries-in-buffer/entries-written: {len(lines)}/{len(lines)}\n" + "\n".join(lines)


def test_window_edges_spanning_idle_and_hotspot():
    data = trace(line("0.5"), line("1.5", "complete"), line("2.0"), line("2.5", "complete"))
    es, _ = measure.parse_trace(data, "259,0")
    a, b, idle = measure.measurements(es, 1_000_000_000, 3_500_000_000, 1_000_000_000)
    assert (a["read_ios"], a["qd_mean"], a["busy_ns"]) == (0, 0.5, 500_000_000)
    assert (b["read_ios"], b["read_bytes"], idle["duration_s"], idle["qd_peak"]) == (
        1,
        4096,
        0.5,
        0,
    )
    s = measure.stats(es, 0, 3_000_000_000)
    assert s["R"]["hottest_1MiB_byte_share"] == 7 / 8
    assert s["W"]["adjacent_share"] is None


def test_overlapping_identical_requests_and_control_ops():
    data = trace(
        line("1.0"),
        line("1.1"),
        line("1.2", "complete"),
        line("1.3", "complete"),
        line("2.0", op="F", count=0),
        line("2.1", "complete", "F", count=0),
        line("3.0", op="D"),
        line("3.1", "complete", "D", error=-5),
        line("4.0", op="FWS"),
        line("4.1", "complete", "FWS"),
    )
    es, audit = measure.parse_trace(data, "259,0")
    assert audit["ambiguous_completions"] == 2 and audit["error_completions"] == 1
    assert all(e.latency_ns is None for e in es[:4])
    assert [e.op for e in es[::2]] == ["R", "R", "flush", "discard", "W"]


def test_integrity_rejections():
    good = trace(line("1.0"), line("1.2", "complete"))
    for bad in (
        good.replace("2/2", "2/3"),
        good.replace("2/2", "3/3"),
        good.replace("4096 ()", "2048 ()"),
        trace(line("1.0")),
        trace(line("1.2", "complete")),
        good.replace("block_rq_issue", "bad"),
    ):
        with pytest.raises(ValueError):
            measure.parse_trace(bad, "259,0")
