"""Compatibility import for kv_ssd_sim.online.payload."""

import sys

from kv_ssd_sim.online import payload as _implementation

sys.modules[__name__] = _implementation
