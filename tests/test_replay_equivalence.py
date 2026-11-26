"""Order within a virtual timestamp matters, gzip container timestamps do not."""

import gzip
import json
import runpy
import shutil
import tarfile
from pathlib import Path

import pytest

from kv_ssd_sim.experiments.profiles import PROFILE_FILES, load_profile, verify_reference
from kv_ssd_sim.simulation.settings import PUBLIC_PROFILES

ROOT = Path(__file__).resolve().parents[1]
compare = runpy.run_path(str(ROOT / "scripts/verify_replay_equivalence.py"))["compare"]


@pytest.mark.parametrize("name", PROFILE_FILES)
def test_composed_profile_reproduces_frozen_snapshot_and_binds_all_inputs(name):
    path = PROFILE_FILES[name].relative_to(ROOT)
    profile, inputs = load_profile(path)
    assert profile == json.loads(PUBLIC_PROFILES[name].read_bytes())
    record = {
        "file": path.as_posix(),
        "sha256": inputs[path.as_posix()],
        "baseline": profile,
        "inputs_sha256": inputs.copy(),
    }
    verify_reference(record)
    del record["inputs_sha256"]["configs/hardware/b300-tp8.json"]
    with pytest.raises(ValueError, match="component hashes"):
        verify_reference(record)


@pytest.mark.parametrize("name", ["events.jsonl", "ssd-demand.jsonl.gz"])
def test_rejects_reordered_records_at_same_time(tmp_path, name):
    reference, replay = tmp_path / "reference", tmp_path / "replay"
    reference.mkdir()
    archive = ROOT / "evidence/large-ssd-staged-20261125/restore-runs.tar.xz"
    with tarfile.open(archive, "r|xz") as source:
        for member in source:
            path = Path(member.name)
            if member.isfile() and path.parent.as_posix() == "runs/base-staged":
                (reference / path.name).write_bytes(source.extractfile(member).read())
    shutil.copytree(reference, replay)
    ledger = replay / "ssd-demand.jsonl.gz"
    payload = gzip.decompress(ledger.read_bytes())
    ledger.write_bytes(gzip.compress(payload, mtime=1))
    assert compare(reference, replay)["status"] == "passed"

    changed = replay / name
    content = payload if changed == ledger else changed.read_bytes()
    first, second, rest = content.split(b"\n", 2)
    assert first != second
    timestamp = "submit_ns" if changed == ledger else "time_ns"
    assert json.loads(first)[timestamp] == json.loads(second)[timestamp]
    reordered = b"\n".join((second, first, rest))
    changed.write_bytes(gzip.compress(reordered, mtime=1) if changed == ledger else reordered)
    with pytest.raises(ValueError, match="Ordered replay output differs"):
        compare(reference, replay)
