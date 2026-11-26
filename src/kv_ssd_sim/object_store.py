"""Compatibility import for kv_ssd_sim.simulation.storage.store."""

import sys

from kv_ssd_sim.simulation.storage import store as _implementation

sys.modules[__name__] = _implementation
