"""Compatibility import for kv_ssd_sim.simulation.worker."""

import sys

from kv_ssd_sim.simulation import worker as _implementation

sys.modules[__name__] = _implementation
