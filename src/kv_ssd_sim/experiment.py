"""Compatibility import for kv_ssd_sim.online.experiment."""

import sys

from kv_ssd_sim.online import experiment as _implementation

sys.modules[__name__] = _implementation
