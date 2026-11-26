"""Compatibility import for kv_ssd_sim.online.pressure."""

import sys

from kv_ssd_sim.online import pressure as _implementation

sys.modules[__name__] = _implementation
