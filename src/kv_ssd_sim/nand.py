"""Compatibility import for kv_ssd_sim.devices.femu.nand."""

import sys

from kv_ssd_sim.devices.femu import nand as _implementation

sys.modules[__name__] = _implementation
