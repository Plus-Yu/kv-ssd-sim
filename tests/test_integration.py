"""Explicit opt-in: exercises the actual native packages and local sockets."""

import os
import subprocess
import sys
from pathlib import Path

import pytest

from kv_ssd_sim.events import load_events
from kv_ssd_sim.experiment import validate_trial
from kv_ssd_sim.trace import load_weka

ROOT = Path(__file__).resolve().parents[1]
pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        os.getenv("KV_SSD_RUN_NATIVE_TESTS") != "1",
        reason="set KV_SSD_RUN_NATIVE_TESTS=1 for native socket tests",
    ),
]


def replay(tmp_path, fixture, *extra, env=None):
    output = tmp_path / "replay"
    log = tmp_path / "process.log"
    with log.open("w") as stream:
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "kv_ssd_sim",
                "replay",
                str(ROOT / "examples" / fixture),
                "--output",
                str(output),
                *extra,
            ],
            stdout=stream,
            stderr=subprocess.STDOUT,
            timeout=120,
            check=False,
            env=env,
        )
    assert result.returncode == 0, log.read_text()[-10000:]
    return output


def test_disk_restore_releases_native_scheduler(tmp_path):
    output = replay(
        tmp_path, "two_requests.weka.json", "--cold-between-turns", "--read-delay-ms", "30"
    )
    result = validate_trial(output)
    assert result["request_2_external_tokens"] == 256
    assert result["request_2_compute_budget_ms"] == 16
    assert result["observed_delay_ms"] >= 30
    events = load_events(output / "events.jsonl")
    prepared = next(e for e in events if e["event"] == "payloads_prepared")
    first_admit = next(e for e in events if e["event"] == "request_admitted")
    assert prepared["seq"] < first_admit["seq"]
    steps = {}
    for event in events:
        if event["event"] == "compute_begin":
            steps[event["step"]] = event
        elif event["event"] == "compute_end":
            assert event["actual_compute_ms"] >= steps[event["step"]]["planned_compute_ms"]
    first_put = next(e for e in events if e["event"] == "store_put_begin")
    first_end = next(e for e in events if e["event"] == "compute_end")
    assert first_put["seq"] > first_end["seq"]


def test_native_batching_obeys_subagent_join(tmp_path):
    fixture = "subagents.weka.json"
    output = replay(tmp_path, fixture)
    events = load_events(output / "events.jsonl")
    admitted = {e["request_id"]: e for e in events if e["event"] == "request_admitted"}
    finished = {e["request_id"]: e for e in events if e["event"] == "request_finished"}
    specs = load_weka(ROOT / "examples" / fixture)
    for request in specs:
        assert finished[request.request_id]["output_tokens"] == request.output_tokens
        for dep in request.dependencies:
            assert finished[dep]["seq"] < admitted[request.request_id]["seq"]
    assert any(e["event"] == "compute_begin" and len(e["request_ids"]) > 1 for e in events)


@pytest.mark.parametrize("fixture", ["two_requests.weka.json", "hybrid-partial.weka.json"])
def test_hybrid_restores_both_families(tmp_path, fixture):
    import json

    output = replay(
        tmp_path,
        fixture,
        "--cold-between-turns",
        "--hybrid-config",
        str(ROOT / "configs/hybrid-tiny.json"),
    )
    events = load_events(output / "events.jsonl")
    summary = json.loads((output / "summary.json").read_text())
    assert summary["checksum_errors"] == []
    assert summary["request_metrics"][1]["external_load_tokens"] == 256
    restore = next(e for e in events if e["event"] == "store_get_end")
    assert all(restore["checksum_ok"])
    assert any("@store_format:mamba_state@" in key for key in restore["keys"])
    assert any("@store_format:attention_head_major@" in key for key in restore["keys"])
    assert restore["result"].count(8192) == 4
    assert restore["result"].count(2624) == 1
    assert any(e["event"] == "kda_state_materialized" and e["role"] == "checkpoint" for e in events)


def test_hybrid_missing_kda_forces_recompute(tmp_path):

    # Fault injection lives in this subprocess test only.
    (tmp_path / "sitecustomize.py").write_text("""
from kv_ssd_sim.storage import ObservedStore
original = ObservedStore.evict_memory_replicas
def evict_and_remove_kda(self, keys):
    original(self, keys)
    removed = [key for key in keys if "@group:1@" in key]
    assert removed
    for key in removed:
        assert self.native.remove(key) == 0
        self.expected.pop(key)  # The test intentionally deleted this object.
    self.events.emit("test_kda_objects_removed", keys=removed)
ObservedStore.evict_memory_replicas = evict_and_remove_kda
""")
    env = os.environ.copy()
    env["PYTHONPATH"] = str(tmp_path) + os.pathsep + env.get("PYTHONPATH", "")
    # Pinned Mooncake currently fails to re-establish disk replicas after
    # remove(key) + re-PUT of the same key. Assert both correct scheduler fallback
    # and this explicit durability failure; do not turn it into a passing report.
    with pytest.raises(AssertionError, match="Mooncake did not establish all disk replicas"):
        replay(
            tmp_path,
            "two_requests.weka.json",
            "--cold-between-turns",
            "--hybrid-config",
            str(ROOT / "configs/hybrid-tiny.json"),
            env=env,
        )
    events = load_events(tmp_path / "replay/events.jsonl")
    removed = next(e for e in events if e["event"] == "test_kda_objects_removed")
    later = [e for e in events if e["seq"] > removed["seq"]]
    timeout = next(e for e in later if e["event"] == "disk_barrier_timeout")
    assert set(timeout["missing_disk_keys"]) == set(removed["keys"])
    assert all(timeout["tiers"][key] == ["MEMORY"] for key in removed["keys"])
    # New KDA keys still offload: this is specific to deleted then reused keys.
    assert any(
        "@group:1@" in key and key not in removed["keys"] and "LOCAL_DISK" in tiers
        for key, tiers in timeout["tiers"].items()
    )
    assert not any(e["event"] == "load_requested" for e in later)
    assert any(e["event"] == "request_finished" and e["output_tokens"] == 4 for e in later)
    assert sum(e["planned_compute_ms"] for e in later if e["event"] == "compute_begin") > 16


def test_hybrid_configurable_layers_and_state_sizes(tmp_path):
    import json

    geometry = tmp_path / "geometry.json"
    geometry.write_text(
        json.dumps(
            {"layers": ["mla", "kda", "kda", "mla"], "mla_latent_dim": 64, "kda_head_dim": 32}
        )
    )
    output = replay(
        tmp_path, "two_requests.weka.json", "--cold-between-turns", "--hybrid-config", str(geometry)
    )
    events = load_events(output / "events.jsonl")
    restore = next(e for e in events if e["event"] == "store_get_end")
    assert all(restore["checksum_ok"])
    assert restore["result"].count(20480) == 4
    assert restore["result"].count(18688) == 1
