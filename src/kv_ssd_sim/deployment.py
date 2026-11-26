"""Compatibility import for kv_ssd_sim.models.deployment."""

import sys

from kv_ssd_sim.models import deployment as _implementation

sys.modules[__name__] = _implementation
