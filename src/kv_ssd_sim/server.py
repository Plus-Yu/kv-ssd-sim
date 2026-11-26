"""Compatibility import for kv_ssd_sim.online.server."""

import sys

from kv_ssd_sim.online import server as _implementation

sys.modules[__name__] = _implementation
