"""Compatibility import for kv_ssd_sim.workload.kvlab."""

import sys

from kv_ssd_sim.workload import kvlab as _implementation

sys.modules[__name__] = _implementation
