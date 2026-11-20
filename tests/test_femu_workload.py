"""Fail closed on an invalid guest workload or a timing window overlapping cold prep."""

import importlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def modules(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "scripts"))
    return importlib.import_module("femu_workload"), importlib.import_module("verify_femu")


def test_staged_copy_not_live_host_input(tmp_path, modules):
    workload, verify = modules
    source = tmp_path / "user.json"
    source.write_bytes((ROOT / "examples/two_requests.weka.json").read_bytes())
    args = SimpleNamespace(
        workload="replay",
        input=source,
        hybrid_config=ROOT / "configs/hybrid-tiny.json",
        max_model_len=1024,
        num_blocks=17,
        prefill_ms=7.0,
        decode_ms=1.0,
    )
    folder = tmp_path / "input"
    workload.stage(folder, args)
    source.write_text("{}")
    actual = {p.name: workload.digest(p) for p in folder.iterdir()}
    config = json.loads((folder / "config.json").read_text())
    manifest = {
        **config,
        "input_sha256": actual["trace.json"],
        "hybrid_config": json.loads(args.hybrid_config.read_text()),
    }
    launch = {"workload": "replay", "guest_inputs_sha256": actual}
    (tmp_path / "guest-input.json").write_text(json.dumps({"sha256": actual}))
    assert verify.verify_inputs(tmp_path, launch, manifest) == actual
    (folder / "trace.json").write_text("{}")
    with pytest.raises(AssertionError, match="hash"):
        verify.verify_inputs(tmp_path, launch, manifest)


@pytest.mark.parametrize(
    ("fixture", "blocks", "reason"),
    [("subagents.weka.json", 96, "serial parent"), ("two_requests.weka.json", 4, "no truncation")],
)
def test_preflight_rejects_unrepresentable_input(tmp_path, modules, fixture, blocks, reason):
    workload, _ = modules
    args = SimpleNamespace(
        workload="aiperf",
        input=ROOT / "examples" / fixture,
        hybrid_config=ROOT / "configs/hybrid-tiny.json",
        max_model_len=8192,
        num_blocks=blocks,
        prefill_ms=10.0,
        decode_ms=2.0,
    )
    with pytest.raises(ValueError, match=reason):
        workload.stage(tmp_path / "input", args)
    assert not (tmp_path / "input").exists()


@pytest.mark.parametrize("fault", [None, "overlap", "missing", "unfinished_previous", "ttft"])
def test_cold_preparation_boundary(tmp_path, modules, fault):
    _, verify = modules
    (tmp_path / "client").mkdir()

    def write(path, rows):
        (tmp_path / path).write_text("".join(json.dumps(row) + "\n" for row in rows))

    write(
        "client-send.jsonl",
        [
            {"request_id": "a", "before_native_send_ns": 100},
            {"request_id": "b", "before_native_send_ns": 205},
        ],
    )
    write(
        "client/profile_export_raw.jsonl",
        [
            {
                "metadata": {"x_request_id": cid},
                "start_perf_ns": ns,
                "responses": [
                    {
                        "perf_ns": ns + 1_000_000,
                        "packets": [
                            {"value": json.dumps({"choices": [{"delta": {"content": "x"}}]})}
                        ],
                    }
                ],
            }
            for cid, ns in [("a", 101), ("b", 208)]
        ],
    )
    write(
        "client/profile_export.jsonl",
        [
            {
                "metadata": {"x_request_id": cid},
                "metrics": {
                    "time_to_first_token": {"value": 2 if fault == "ttft" else 1, "unit": "ms"}
                },
            }
            for cid in ("a", "b")
        ],
    )
    events = [
        {"client_request_id": "a", "event": "http_arrival", "time_ns": 105, "seq": 1},
        {"client_request_id": "a", "event": "http_received", "request_id": "ra", "seq": 2},
        {
            "event": "http_response_finished",
            "request_id": "ra",
            "time_ns": 112 if fault == "unfinished_previous" else 109,
            "seq": 3,
        },
        {"client_request_id": "b", "event": "cold_prepare_begin", "time_ns": 110, "seq": 4},
    ]
    for name in (
        "disk_replicas_confirmed",
        "memory_replicas_evicted",
        "local_prefix_cache_reset",
        "guest_page_cache_dropped",
    ):
        events.append({"seq": len(events) + 1, "event": name})
    events.extend(
        [
            {
                "client_request_id": "b",
                "event": "cold_prepare_end",
                "time_ns": 207 if fault == "overlap" else 200,
                "seq": 9,
            },
            {"client_request_id": "b", "event": "http_arrival", "time_ns": 210, "seq": 10},
            {"client_request_id": "b", "event": "http_received", "request_id": "rb", "seq": 11},
            {"event": "http_response_finished", "request_id": "rb", "time_ns": 230, "seq": 12},
        ]
    )
    if fault == "missing":
        events.pop(5)
    if fault:
        with pytest.raises(AssertionError):
            verify.verify_cold_windows(tmp_path, events, 0)
    else:
        assert verify.verify_cold_windows(tmp_path, events, 0)["b"]["cold_prepare_ms"] == 90 / 1e6
