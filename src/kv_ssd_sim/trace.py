"""Compatibility import for kv_ssd_sim.workload.weka."""

import sys

from kv_ssd_sim.workload import weka as _implementation

sys.modules[__name__] = _implementation
