"""Fingerprint the package sources and the extra files used by an experiment."""

import hashlib
from itertools import chain
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]


def package_source_hashes(*, relative_to=PACKAGE_ROOT, extra_files=()):
    hashes = {}
    for path in chain(sorted(PACKAGE_ROOT.rglob("*.py")), extra_files):
        with path.open("rb") as stream:
            hashes[path.relative_to(relative_to).as_posix()] = hashlib.file_digest(
                stream, "sha256"
            ).hexdigest()
    return hashes
