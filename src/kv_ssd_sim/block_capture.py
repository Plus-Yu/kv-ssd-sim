"""Compatibility import for kv_ssd_sim.devices.femu.block_capture."""

import sys

from kv_ssd_sim.devices.femu import block_capture as _implementation

if __name__ == "__main__":
    _implementation.main()
else:
    sys.modules[__name__] = _implementation
