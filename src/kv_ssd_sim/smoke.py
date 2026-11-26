"""Compatibility import for kv_ssd_sim.online.smoke."""

import sys

from kv_ssd_sim.online import smoke as _implementation

sys.modules[__name__] = _implementation
