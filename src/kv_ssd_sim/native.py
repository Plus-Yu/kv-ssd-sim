"""Compatibility import for kv_ssd_sim.online.native."""

import sys

from kv_ssd_sim.online import native as _implementation

sys.modules[__name__] = _implementation
