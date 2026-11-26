"""Compatibility import for kv_ssd_sim.online.metadata_cache."""

import sys

from kv_ssd_sim.online import metadata_cache as _implementation

sys.modules[__name__] = _implementation
