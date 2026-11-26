"""Fingerprint the installed experiment package, preserving nested module names."""

import hashlib
from pathlib import Path


def package_source_hashes():
    package = Path(__file__).resolve().parents[1]
    return {
        p.relative_to(package).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(package.rglob("*.py"))
    }
