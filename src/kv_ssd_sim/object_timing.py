"""Compatibility import for kv_ssd_sim.simulation.timing."""

import sys

from kv_ssd_sim.simulation import timing as _implementation

sys.modules[__name__] = _implementation
