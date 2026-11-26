"""Compatibility import for kv_ssd_sim.experiments.provenance."""

import sys

from kv_ssd_sim.experiments import provenance as _implementation

sys.modules[__name__] = _implementation
