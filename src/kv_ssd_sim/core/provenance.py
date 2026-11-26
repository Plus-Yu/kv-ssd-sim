"""Fingerprint the installed experiment package, preserving nested module names."""

import hashlib
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]


def package_source_files():
    return sorted(PACKAGE_ROOT.rglob("*.py"))


def package_source_hashes():
    return {
        p.relative_to(PACKAGE_ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in package_source_files()
    }
