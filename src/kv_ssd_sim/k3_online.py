"""Compatibility import for kv_ssd_sim.online.k3_online."""

import sys

from kv_ssd_sim.online import k3_online as _implementation

sys.modules[__name__] = _implementation
