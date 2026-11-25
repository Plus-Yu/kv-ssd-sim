"""Run K3 object-level predictions on compiled WEKA sessions with no KV payload allocation."""

import sys
from pathlib import Path

# Allow the historical script commands to run directly from a source checkout.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from kv_ssd_sim.simulation.engine import Admissions, simulate
from kv_ssd_sim.simulation.runner import main

__all__ = ["Admissions", "main", "simulate"]

if __name__ == "__main__":
    main()
