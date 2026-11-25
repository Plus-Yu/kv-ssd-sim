"""Order within a virtual timestamp matters, gzip container timestamps do not."""

import gzip
import json
import runpy
import shutil
import tarfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
compare = runpy.run_path(str(ROOT / "scripts/verify_replay_equivalence.py"))["compare"]


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
