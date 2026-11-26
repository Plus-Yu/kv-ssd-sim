"""Compatibility import for kv_ssd_sim.experiments.runner."""

import sys

from kv_ssd_sim.experiments import runner as _implementation

if __name__ == "__main__":
    _implementation.main()
else:
    sys.modules[__name__] = _implementation
