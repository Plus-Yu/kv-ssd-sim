"""Compatibility import for kv_ssd_sim.online.timing."""

import sys

from kv_ssd_sim.online import timing as _implementation

sys.modules[__name__] = _implementation
