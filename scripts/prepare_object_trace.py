"""Compatibility entry point for kv_ssd_sim.workload.weka_compiler."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from kv_ssd_sim.workload import weka_compiler as _implementation  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(_implementation.main())

sys.modules[__name__] = _implementation
