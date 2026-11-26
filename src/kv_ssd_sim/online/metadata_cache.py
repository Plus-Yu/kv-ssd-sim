"""Validated startup-only package mapping cache for the TCG/9p guest.

Transformers calls packages_distributions(), which stats every RECORD entry.
The host can compute the exact map for the same exported venv before boot.
No scheduler, connector, tensors, or request processing are cached here.
"""

from __future__ import annotations

import hashlib
import json
import sys
from importlib import metadata
from pathlib import Path


def environment_signature():
    records = []
    for dist in metadata.distributions():
        digest = hashlib.sha256()
        for name in ("METADATA", "RECORD", "top_level.txt"):
            digest.update(name.encode())
            digest.update((dist.read_text(name) or "").encode())
        records.append((dist.metadata["Name"], dist.version, digest.hexdigest()))
    return {
        "prefix": sys.prefix,
        "cache_tag": sys.implementation.cache_tag,
        "distributions": sorted(records),
    }


def create(path: Path):
    snapshot = {
        "environment": environment_signature(),
        "mapping": metadata.packages_distributions(),
    }
    path.write_text(json.dumps(snapshot, sort_keys=True) + "\n")


def install(path: Path):
    snapshot = json.loads(path.read_text())
    # JSON normalizes tuples to lists; compare the normalized representation.
    actual = json.loads(json.dumps(environment_signature()))
    if snapshot["environment"] != actual:
        raise RuntimeError("Guest package metadata differs from the prepared host snapshot")
    mapping = snapshot["mapping"]
    if not isinstance(mapping, dict) or not all(
        isinstance(key, str)
        and isinstance(value, list)
        and all(isinstance(name, str) for name in value)
        for key, value in mapping.items()
    ):
        raise ValueError("Invalid package distribution mapping")
    metadata.packages_distributions = lambda: {key: list(value) for key, value in mapping.items()}
