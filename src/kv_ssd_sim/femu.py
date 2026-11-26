"""Compatibility import for kv_ssd_sim.devices.femu.device."""

import sys

from kv_ssd_sim.devices.femu import device as _implementation

sys.modules[__name__] = _implementation
