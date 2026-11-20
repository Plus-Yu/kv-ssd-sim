"""Block-layer accounting for bounded WEKA experiments."""

import importlib
from pathlib import Path

import pytest


def test_namespace_bytes_and_completion_latency(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    totals = importlib.import_module("analyze_step6").block_totals
    trace = tmp_path / "block-trace.txt"
    issue = "worker [001] 1.000000: block_rq_issue: 259,0 RA 4096 () 16 + 8 [worker]\n"
    end = "worker [001] 1.001500: block_rq_complete: 259,0 RA () 16 + 8 be,0,4 [0]\n"
    trace.write_text(issue + end)
    assert totals(trace) == {
        "R": {"count": 1, "bytes": 4096, "mean_ms": 1.5},
        "W": {"count": 0, "bytes": 0, "mean_ms": None},
    }
    trace.write_text(end)
    with pytest.raises(ValueError, match="Unmatched"):
        totals(trace)
    trace.write_text(issue)
    with pytest.raises(ValueError, match="Outstanding"):
        totals(trace)
