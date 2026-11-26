"""Archive the first kvlab milestone without copying the 8.8 GB corpus or KV payload files."""

import argparse
import csv
import hashlib
import json
import platform
import shutil
import subprocess
import tarfile
from collections import Counter
from pathlib import Path
from tempfile import TemporaryDirectory

from fetch_kvlab import BLOB, LAYER_SIZE, NAME, SIZE

from kv_ssd_sim.core.provenance import package_source_hashes
from kv_ssd_sim.workload.corpus import REVISION, SHA256, dump

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def raw_files(trial):
    for name in (
        "native-launch.json",
        "frozen-prediction.json",
        "server.log",
        "client.json",
        "store.json",
        "store-boundaries.tsv",
        "server/events.jsonl",
        "server/summary.json",
        "server/run_manifest.json",
        "server/mooncake-master.log",
        "server/mooncake-master.json",
        "server/mooncake.json",
    ):
        yield trial / name


def control(study, target):
    native = study / "positive-fixture/test_disk_restore_releases_nat0/replay"
    events = [json.loads(l) for l in (native / "events.jsonl").open()]
    expected = Counter(
        (k, n)
        for e in events
        if e["event"] == "store_get_end"
        for k, n, ok in zip(e["keys"], e["result"], e["checksum_ok"], strict=True)
        if ok
    )
    actual = Counter()
    starts = Counter()
    with (study / "controlled-read.tsv").open() as f:
        for _, stage, k, size, _, code in csv.reader(f, delimiter="\t"):
            if stage in ("read_submit", "read_complete") and code == "0":
                (starts if stage == "read_submit" else actual)[
                    k.removeprefix("default\0"), int(size)
                ] += 1
    if not actual or actual != expected or starts != actual:
        raise ValueError("Controlled disk-only GET did not appear in the file observer")
    dump(
        target,
        {
            "fixture": "two_requests.weka.json, --cold-between-turns, --read-delay-ms 30",
            "objects": actual.total(),
            "bytes": sum(k[1] * n for k, n in actual.items()),
            "matches_successful_native_gets_by_key_and_bytes": True,
            "classification": "controlled instrumentation test",
        },
    )
    return native


def archive(study, out):
    if out.exists():
        raise FileExistsError(out)
    out.parent.mkdir(exist_ok=True, parents=True)
    with TemporaryDirectory(dir=out.parent) as temp:
        dest = Path(temp) / "result"
        shutil.copytree(study / "final-results", dest)
        for src in (
            study / "corpus/audit.json",
            study / "corpus/recorded-concurrency.csv.gz",
            study / "calibration.json",
            study / "regression.json",
            study / "tests-final.log",
            study / "controlled-read.log",
            study / "integration.log",
        ):
            shutil.copy2(src, dest / src.name)
        native = control(study, dest / "controlled-read.json")
        # Preserve the original compiler's input for the observed prefix-length
        # correction. Subsequent reproductions use the corrected compiler.
        if (study / "matrix-target/calibration/base").is_dir():
            with tarfile.open(dest / "prefix-diagnostic.tar.gz", "w:gz") as tar:
                for label in ("calibration", "holdout"):
                    for name in ("input.json", "window.json"):
                        tar.add(study / "target" / label / name, arcname=f"{label}/{name}")
                    for name in (
                        "launch.json",
                        "summary.json",
                        "requests.json",
                        "ssd-summary.json",
                    ):
                        tar.add(
                            study / "matrix-target" / label / "base" / name,
                            arcname=f"{label}/prediction/{name}",
                        )
        for name in ("score-cal", "score-1", "score-2", "score-3"):
            shutil.copytree(study / name, dest / "validation" / name)
        with tarfile.open(dest / "compiled-inputs.tar.gz", "w:gz") as tar:
            for group in ("paired", "reference"):
                shutil.copy2(study / group / "selection.json", dest / (group + "-selection.json"))
                for label in ("calibration", "holdout"):
                    dirpath = study / group / label
                    shutil.copy2(dirpath / "window.json", dest / f"{group}-{label}.json")
                    tar.add(dirpath / "input.json", arcname=f"{group}/{label}/input.json")
        with tarfile.open(dest / "predictions.tar.gz", "w:gz", compresslevel=6) as tar:
            for name in (
                "pred-cal",
                "frozen-holdout",
                "regression-0504",
                "demand-reference",
                "demand-paired",
            ):
                tar.add(study / name, arcname=name)
        with tarfile.open(dest / "native-events.tar.gz", "w:gz") as tar:
            for name in ("online-cal", *(f"online-holdout-{n}" for n in (1, 2, 3))):
                for src in raw_files(study / name):
                    tar.add(src, arcname=src.relative_to(study))
            tar.add(study / "controlled-read.tsv", arcname="controlled/observer.tsv")
            for name in ("events.jsonl", "summary.json"):
                tar.add(native / name, arcname="controlled/" + name)
        dependencies = (
            "scripts/build_object_master.py",
            "scripts/build_object_observer.py",
            "scripts/fetch_kvlab.py",
            "scripts/run_object_holdout.py",
            "scripts/analyze_kvlab_native.py",
            "scripts/compare_object_replay.py",
            "scripts/run_kvlab_matrix.py",
            "scripts/summarize_kvlab.py",
            "scripts/archive_kvlab.py",
            "scripts/reproduce_kvlab_n15.sh",
            "native/object_replay_master.cpp",
            "native/object_replay_observer.cpp",
            "tests/test_kvlab.py",
        )
        dump(
            dest / "provenance.json",
            {
                "source": {
                    "hf_revision": REVISION,
                    "sha256": SHA256,
                    "size": SIZE,
                    "hf_url": f"https://huggingface.co/datasets/wekaio/{NAME}/resolve/{REVISION}/{NAME}.jsonl",
                    "retrieved_via": "public Quay image",
                    "quay_layer_sha256": BLOB,
                    "quay_layer_size": LAYER_SIZE,
                    "upstream_kvlab_revision": "3fd1a8411f9f30b6c54fda850a094b809c331d51",
                },
                "runtime_manifests": "launch.json and native-launch.json inside archives",
                "after_run": {
                    "host": platform.node(),
                    "kernel": platform.release(),
                    "git_head": subprocess.check_output(
                        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
                    ).strip(),
                    "working_files_sha256": package_source_hashes(
                        relative_to=ROOT, extra_files=(ROOT / name for name in dependencies)
                    ),
                },
                "raw_trial_dir": str(study.resolve()),
                "0504_native_dir": "outputs/femu-n15-0504-20261120-074735/joint/server",
                "raw_corpus_retrieval": "scripts/fetch_kvlab.py; index regenerated by kv-ssd-sim workload kvlab audit",
            },
        )
        with (dest / "SHA256SUMS").open("x") as f:
            for path in sorted(dest.rglob("*")):
                if path.is_file() and path.name != "SHA256SUMS":
                    f.write(f"{sha(path)}  {path.relative_to(dest)}\n")
        dest.rename(out)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("study", type=Path)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    archive(a.study, a.output)
