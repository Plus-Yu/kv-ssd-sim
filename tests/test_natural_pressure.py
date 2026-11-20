"""The archived capacity run must fail verification when its distinguishing evidence is lost."""

import importlib
import json
from pathlib import Path

import pytest


@pytest.mark.parametrize("fault", ["cold", "resident", "kda_present", "no_device_reads", "jobs"])
def test_pressure_evidence_rejects_false_positives(monkeypatch, fault):
    project = Path(__file__).resolve().parents[1]
    monkeypatch.syspath_prepend(str(project / "scripts"))
    check = importlib.import_module("check_pressure").check_pressure
    root = project / "evidence/kimi-k3-step5/femu-step5-natural03"
    run = root / "replay"
    manifest = json.loads((run / "run_manifest.json").read_text())
    events = [json.loads(line) for line in (run / "events.jsonl").read_text().splitlines()]
    summary = json.loads((run / "summary.json").read_text())
    chains = json.loads((root / "verification.json").read_text())["chains"]
    cases = {
        "cold": "Artificial eviction",
        "resident": "memory-to-disk",
        "kda_present": "Missing checkpoint",
        "no_device_reads": "Device read counters",
        "jobs": "Unfinished native save jobs",
    }
    cfg = manifest["pressure_config"]
    if fault == "cold":
        events.append({"event": "guest_page_cache_dropped"})
    elif fault == "resident":
        for event in events:
            if event["event"] == "probe_replicas" and event["request_id"] == cfg["restore"]:
                for tiers in event["tiers"].values():
                    tiers.append("MEMORY")
    elif fault == "kda_present":
        for event in events:
            if event["event"] == "probe_replicas" and event["request_id"] == cfg["recompute"]:
                for key in event["tiers"]:
                    if "@store_format:mamba_state@" in key:
                        event["tiers"][key] = ["LOCAL_DISK"]
    elif fault == "no_device_reads":
        next(e for e in events if e["event"] == "femu_block_io_during_get")["delta"]["reads"] = 0
    else:
        events.pop(next(i for i, e in enumerate(events) if e["event"] == "save_job_finished"))
    with pytest.raises(ValueError, match=cases[fault]):
        check(root, events, summary, manifest, chains)
