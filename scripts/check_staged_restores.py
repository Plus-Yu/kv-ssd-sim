"""Audit a saved host-staged replay."""

import sys
from pathlib import Path

# Allow the historical script commands to run directly from a source checkout.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from kv_ssd_sim.analysis.restore import COMMON, DISK, MEMORY, RESTORE_EVENTS, audit, main

__all__ = ["COMMON", "DISK", "MEMORY", "RESTORE_EVENTS", "audit", "main"]

if __name__ == "__main__":
    main()
