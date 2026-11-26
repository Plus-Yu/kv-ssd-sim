"""Reject incomplete device/Store/scheduler/HTTP evidence, even with passing summaries."""

import json

import pytest

from kv_ssd_sim.analysis import aiperf, femu


@pytest.fixture
def evidence(tmp_path, monkeypatch):
    # HTTP coverage/token verification is separate; this fixture exercises
    # the final correlation against kernel block completions.
    monkeypatch.setattr(aiperf, "verify", lambda _: {"femu_verified": True})
    server = tmp_path / "joint/server"
    server.mkdir(parents=True)
    client = tmp_path / "joint/client"
    client.mkdir()
    (tmp_path / "launch.json").write_text(
        json.dumps(
            {
                "workload": "aiperf",
                "accelerator": "tcg",
                "nand_read_ns": 40000,
            }
        )
    )
    (server / "summary.json").write_text(
        json.dumps(
            {
                "femu_verified": True,
                "checksum_errors": [],
                "request_metrics": [
                    {
                        "request_id": "r",
                        "external_load_tokens": 64,
                        "ttft_ms": 12,
                        "compute_budget_ms": 10,
                        "compute_window_ms": 10.5,
                    }
                ],
            }
        )
    )
    (server / "run_manifest.json").write_text('{"read_delay_ms": 0}')
    events = [
        {"event": "clock_sync", "monotonic_origin_ns": 1_000_000_000},
        {"event": "http_received", "request_id": "r", "client_request_id": "client-r"},
        {"event": "request_admitted", "request_id": "r"},
        {"event": "native_get_begin", "request_id": "r"},
        {"event": "store_get_end", "request_id": "r", "checksum_ok": [True]},
        {"event": "load_complete", "request_id": "r"},
        {"event": "compute_begin", "request_ids": ["r"]},
        {"event": "first_token", "request_id": "r"},
        {"event": "http_token_emitted", "request_id": "r"},
    ]
    for i, event in enumerate(events):
        event.update(seq=i + 1, time_ns=i * 1_000_000)
    (server / "events.jsonl").write_text("\n".join(map(json.dumps, events)))
    (client / "profile_export.jsonl").write_text(
        json.dumps(
            {
                "metadata": {"x_request_id": "client-r", "source_outer_idx": 1},
                "metrics": {"time_to_first_token": {"value": 15}},
            }
        )
    )
    (tmp_path / "block-trace.txt").write_text(
        "# entries-in-buffer/entries-written: 2/2\n"
        "worker-1 [000] 1.003100: block_rq_issue: 259,0 R 4096 () 16 + 8 [worker]\n"
        "worker-1 [000] 1.003200: block_rq_complete: 259,0 R () 16 + 8 [0]\n"
    )
    return femu, tmp_path


def test_joint_device_to_http_chain(evidence):
    verifier, root = evidence
    result = verifier.verify(root)
    assert result["status"] == "passed"
    chain = result["chains"][0]
    assert chain["femu_read_bytes_during_get"] == 4096
    assert chain["aiperf_ttft_ms"] == 15
    assert chain["http_first_emit_seq"] > chain["ordered_event_sequences"][-1]


@pytest.mark.parametrize(
    "fault", ["overflow", "missing_issue", "outside_get", "bad_checksum", "early_http"]
)
def test_reject_incomplete_or_reordered_evidence(evidence, fault):
    verifier, root = evidence
    trace = root / "block-trace.txt"
    events_file = root / "joint/server/events.jsonl"
    events = [json.loads(line) for line in events_file.read_text().splitlines()]
    if fault == "overflow":
        trace.write_text(trace.read_text().replace("2/2", "2/3"))
    elif fault == "missing_issue":
        trace.write_text(
            "\n".join(
                line for line in trace.read_text().splitlines() if "block_rq_issue" not in line
            )
        )
    elif fault == "outside_get":
        trace.write_text(trace.read_text().replace("1.003", "1.103"))
    elif fault == "bad_checksum":
        events[4]["checksum_ok"] = [False]
    elif fault == "early_http":
        events[-1]["seq"] = 2
    events_file.write_text("\n".join(map(json.dumps, events)))
    with pytest.raises(AssertionError):
        verifier.verify(root)
