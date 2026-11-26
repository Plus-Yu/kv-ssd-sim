"""Compatibility import for kv_ssd_sim.adapters.mooncake."""

import sys

from kv_ssd_sim.adapters import mooncake as _implementation

sys.modules[__name__] = _implementation
