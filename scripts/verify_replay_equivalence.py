"""Compare the deterministic outputs and effective settings of two object replays."""

import argparse
import gzip
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from kv_ssd_sim.core.paths import PROJECT_ROOT
from kv_ssd_sim.experiments.profiles import verify_reference


def digest(path):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def settings(folder):
    launch = json.loads((folder / "launch.json").read_bytes())
    sources = launch.pop("source_sha256")
    for compiled in launch["compiled_inputs"]:
        del compiled["file"]
    if launch.get("offload_timing_input"):
        del launch["offload_timing_input"]["file"]
    return launch, sources


def compare(before, after, source_root=None):
    reference, _ = settings(before)
    candidate, sources = settings(after)
    old_profile, new_profile = (m.get("reference_settings") for m in (reference, candidate))
    if source_root and new_profile:
        verify_reference(new_profile, source_root)
    if old_profile and new_profile and old_profile != new_profile:
        # Paths and file layout may change. Both snapshots must bind to their inputs.
        for record in (old_profile, new_profile):
            verify_reference(record, source_root or PROJECT_ROOT)
            for field in ("file", "sha256", "inputs_sha256"):
                record.pop(field, None)
    if reference != candidate:
        differing = reference.keys() ^ candidate.keys()
        differing.update(
            k for k in reference.keys() & candidate.keys() if reference[k] != candidate[k]
        )
        raise ValueError(f"Effective launch settings differ: {sorted(differing)}")
    if source_root:
        if not sources:
            raise ValueError("Candidate has no recorded runtime source hashes")
        for name, expected in sources.items():
            if digest(source_root / name) != expected:
                raise ValueError(f"Recorded runtime source differs from checkout: {name}")

    files = [
        "events.jsonl",
        "requests.json",
        "bucket-timing.jsonl",
        "ssd-demand.jsonl.gz",
        "ssd-10s.csv",
        "ssd-summary.json",
    ]
    if reference["restore_path"] == "host-staged":
        files.append("restore-summary.json")
    # Compact archives omit these two ledgers. Compare them whenever the reference includes them.
    files.extend(n for n in ("objects.json", "reuse.jsonl") if (before / n).is_file())
    hashes = {}
    for name in files:
        expected = digest(before / name)
        if digest(after / name) != expected:
            raise ValueError(f"Ordered replay output differs: {name}")
        hashes[name] = expected

    def logical_summary(folder):
        data = json.loads((folder / "summary.json").read_bytes())
        del data["native_master"]["process_peak_rss_kib"]
        return data

    if logical_summary(before) != logical_summary(after):
        raise ValueError("Replica, resource or service summary differs")
    return {"status": "passed", "source_files": len(sources), "output_sha256": hashes}


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("reference", type=Path)
    p.add_argument("candidate", type=Path)
    p.add_argument(
        "--source-root", type=Path, help="Check candidate runtime hashes against this tree"
    )
    args = p.parse_args()
    print(json.dumps(compare(args.reference, args.candidate, args.source_root), indent=2))
