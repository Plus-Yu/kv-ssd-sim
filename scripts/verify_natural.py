"""Compatibility entry point for kv_ssd_sim.analysis.natural."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from kv_ssd_sim.analysis import natural as _implementation

sys.modules[__name__] = _implementation
