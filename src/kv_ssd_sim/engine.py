"""Compatibility import for kv_ssd_sim.online.engine."""

import sys

from kv_ssd_sim.online import engine as _implementation

sys.modules[__name__] = _implementation
