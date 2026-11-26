import json
from pathlib import Path

import pytest

from kv_ssd_sim.workload.weka import load_weka, normalize_trace

ROOT = Path(__file__).resolve().parents[1]


def test_observed_prefix_is_exact_and_scoped():
    original = json.loads((ROOT / "examples/two_requests.weka.json").read_text())
    a, b = normalize_trace(original)
    assert b.prompt_token_ids[:256] == a.prompt_token_ids
    assert len(b.prompt_token_ids) == 320
    assert all(4 <= t < 32000 for t in b.prompt_token_ids)
    original["id"] = "another-trace"
    other, _ = normalize_trace(original)
    assert other.prompt_token_ids[:64] != a.prompt_token_ids[:64]


def test_subagent_fanout_and_join():
    requests = load_weka(ROOT / "examples/subagents.weka.json")
    parent, child_a, child_a2, child_b, join = requests
    assert child_a.dependencies == child_b.dependencies == (parent.request_id,)
    assert child_a2.dependencies == (child_a.request_id,)
    assert set(join.dependencies) == {parent.request_id, child_a2.request_id, child_b.request_id}


def test_partial_tail_without_hash_is_not_shared():
    trace = {
        "id": "tail",
        "requests": [
            {"t": 0, "in": 65, "out": 1, "hash_ids": [1]},
            {"t": 1, "in": 65, "out": 1, "hash_ids": [1]},
        ],
    }
    a, b = normalize_trace(trace)
    assert a.prompt_token_ids[:64] == b.prompt_token_ids[:64]
    assert a.prompt_token_ids[64:] != b.prompt_token_ids[64:]


@pytest.mark.parametrize("field,value", [("in", 0), ("out", 0), ("t", float("nan"))])
def test_invalid_requests_are_not_silently_dropped(field, value):
    record = {"in": 64, "out": 2, "t": 0, "hash_ids": [1]}
    record[field] = value
    with pytest.raises(ValueError):
        normalize_trace({"id": "invalid", "requests": [record]})


def test_jsonl_duplicate_trace_rejected(tmp_path):
    trace = {"id": "duplicate", "requests": []}
    path = tmp_path / "traces.jsonl"
    path.write_text(json.dumps(trace) + "\n" + json.dumps(trace) + "\n")
    with pytest.raises(ValueError, match="Duplicate"):
        load_weka(path)
