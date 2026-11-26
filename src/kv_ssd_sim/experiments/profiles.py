"""Load complete historical profiles or assemble a profile from disjoint components."""

import hashlib
import json
from pathlib import Path

from ..core.paths import PROJECT_ROOT

PROFILE_FILES = {
    name: PROJECT_ROOT / "configs/profiles" / (name + ".json")
    for name in ("reference", "reference-volume-20261125", "reference-single-ssd")
}


def load_profile(path, root=PROJECT_ROOT):
    root = Path(root).resolve()
    inputs = {}
    config_dir = (root / "configs").resolve()

    def read(source):
        if not source.resolve().is_relative_to(config_dir):
            raise ValueError(f"Profile input lies outside configs: {source}")
        name = source.relative_to(root).as_posix()
        if name in inputs:
            raise ValueError(f"Duplicate profile input: {name}")
        data = source.read_bytes()
        inputs[name] = hashlib.sha256(data).hexdigest()
        return json.loads(data)

    profile = read(root / path)
    if "components" not in profile:
        return profile, inputs
    settings = {}
    references = {}
    parts = [read(root / "configs" / name) for name in profile["components"]]
    for part in [*parts, profile["settings"]]:
        refs = part.get("sources", {})
        values = {k: v for k, v in part.items() if k != "sources"}
        if settings.keys() & values.keys() or references.keys() & refs.keys():
            raise ValueError(f"Conflicting profile components in {path}")
        settings.update(values)
        references.update(refs)
    settings["sources"] = references
    return settings, inputs


def verify_reference(record, root=PROJECT_ROOT):
    root = root.resolve()
    path = root / record["file"]
    baseline, inputs = load_profile(path, root)
    if record["baseline"] != baseline or inputs[record["file"]] != record["sha256"]:
        raise ValueError(f"Recorded reference profile differs from {path}")
    if (len(inputs) > 1 or "inputs_sha256" in record) and inputs != record.get("inputs_sha256"):
        raise ValueError(f"Recorded component hashes differ from {path}")
