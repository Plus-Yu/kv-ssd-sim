"""Compatibility import for kv_ssd_sim.online.runtime."""

import sys

from kv_ssd_sim.online import runtime as _implementation

sys.modules[__name__] = _implementation
