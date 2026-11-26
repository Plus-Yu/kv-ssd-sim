"""Compatibility import for kv_ssd_sim.models.hybrid."""

import sys

from kv_ssd_sim.models import hybrid as _implementation

sys.modules[__name__] = _implementation
