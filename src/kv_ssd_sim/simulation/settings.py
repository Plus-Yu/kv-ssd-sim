"""Compatibility import for kv_ssd_sim.experiments.settings."""

import sys

from kv_ssd_sim.experiments import settings as _implementation

sys.modules[__name__] = _implementation
