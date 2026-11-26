import copy
import json
from pathlib import Path

import pytest

from kv_ssd_sim.workload.audit import (
    audit_trace,
    select_parent_prefix,
    sha256,
    shared_complete_tokens,
    write_audit_bundle,
)

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "references/kimi-k3-b300/trace_0001.source.json"
SOURCE_SHA = "cfd45b93c58e16b1ae3cb177f91d45bb58590dc592e140e28e4155145746feb3"


@pytest.fixture
def trace():
    return {
        "id": "audit",
        "block_size": 4,
        "hash_id_scope": "local",
        "models": ["test"],
        "requests": [
            {
                "type": "n",
                "model": "test",
                "t": 0,
                "api_time": 1,
                "think_time": 0,
                "in": 10,
                "out": 2,
                "hash_ids": [1, 2, 3],
            },
            {
                "type": "n",
                "model": "test",
                "t": 3,
                "api_time": 1,
                "think_time": 2,
                "in": 14,
                "out": 3,
                "hash_ids": [1, 2, 3, 4],
            },
        ],
    }


def test_public_source_and_reproducible_bundle(tmp_path):
    write_audit_bundle(SOURCE, tmp_path / "bundle", expected_sha256=SOURCE_SHA)
    for name in ["source-audit.json", "slice-audit.json", "selection.json", "slice.json"]:
        assert (tmp_path / "bundle" / name).read_bytes() == (
            ROOT / "datasets/kimi-k3-b300-step1" / name
        ).read_bytes()
    full = json.loads((tmp_path / "bundle/source-audit.json").read_text())
    assert full["valid_request_count"] == 41
    assert len(full["over_context_paths"]) == 13


def test_partial_hash_does_not_imply_full_prefix(trace):
    assert shared_complete_tokens(*trace["requests"], 4) == 8
    trace["requests"][1]["hash_ids"] = [2, 1, 3, 4]
    assert shared_complete_tokens(*trace["requests"], 4) == 0


@pytest.mark.parametrize(
    "field,value",
    [("in", True), ("out", 0), ("t", float("nan")), ("think_time", -1), ("hash_ids", [1])],
)
def test_bad_input_is_reported_not_silently_accepted(trace, field, value):
    trace["requests"][0][field] = value
    report = audit_trace(trace)
    assert not report["structural_validation_passed"]
    assert report["issues"][0]["path"] == "/requests/0"


def test_nested_branches_inventoried_but_not_selected(trace):
    child = copy.deepcopy(trace["requests"][0])
    trace["requests"].insert(
        1,
        {
            "type": "subagent",
            "agent_id": "a",
            "requests": [
                {"type": "subagent", "agent_id": "b", "requests": [child]},
            ],
        },
    )
    report = audit_trace(trace)
    assert report["explicit_subagents"] == 2
    assert report["subagent_requests"] == 1
    assert report["max_explicit_nesting_depth"] == 2
    with pytest.raises(ValueError, match="serial parent"):
        select_parent_prefix(trace, 2)


@pytest.mark.parametrize(
    "change,match", [("overlap", "overlap"), ("no_think", "think_time"), ("no_prefix", "No shared")]
)
def test_ambiguous_causal_slice_is_rejected(trace, change, match):
    if change == "overlap":
        trace["requests"][1]["t"] = 0.5
    elif change == "no_think":
        del trace["requests"][1]["think_time"]
    else:
        trace["requests"][1]["hash_ids"][0] = 99
    with pytest.raises(ValueError, match=match):
        select_parent_prefix(trace, 2)


def test_selection_never_modifies_records_or_source(trace):
    trace["totals"] = {"anything": 100}
    original = copy.deepcopy(trace)
    sliced = select_parent_prefix(trace, 2)
    assert trace == original
    assert sliced["requests"] == trace["requests"]
    assert "totals" not in sliced
    assert set(sliced) == set(trace) - {"totals"}


@pytest.mark.parametrize(
    "options,match",
    [
        ({"expected_sha256": "0" * 64}, "SHA256"),
        ({"context_limit": 100}, "context limit"),
        ({"model_layers": 1000}, "digest budget"),
    ],
)
def test_no_output_after_failed_guard(tmp_path, options, match):
    kwargs = {"expected_sha256": sha256(SOURCE), **options}
    with pytest.raises(ValueError, match=match):
        write_audit_bundle(SOURCE, tmp_path / "bundle", **kwargs)
    assert not (tmp_path / "bundle").exists()


def test_unknown_request_kind_is_visible(trace):
    trace["requests"].append({"type": "future-format"})
    report = audit_trace(trace)
    assert report["top_level_entries"] == 3
    assert report["valid_request_count"] == 2
    assert "unsupported entry" in report["issues"][0]["reason"]
