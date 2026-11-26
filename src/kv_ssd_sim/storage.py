"""Compatibility import for kv_ssd_sim.online.storage."""

import sys

from kv_ssd_sim.online import storage as _implementation

sys.modules[__name__] = _implementation
