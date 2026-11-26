"""Compatibility import for kv_ssd_sim.workload.audit."""

import sys

from kv_ssd_sim.workload import audit as _implementation

sys.modules[__name__] = _implementation
