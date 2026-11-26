"""Compatibility import for kv_ssd_sim.devices.femu.layout."""

import sys

from kv_ssd_sim.devices.femu import layout as _implementation

sys.modules[__name__] = _implementation
