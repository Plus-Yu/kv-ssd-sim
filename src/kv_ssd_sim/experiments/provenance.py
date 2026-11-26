"""Source checkout inputs fingerprinted by the object runner and its offline archiver."""

from ..core.paths import PROJECT_ROOT
from ..core.provenance import package_source_files

REPLAY_SOURCES = (
    "scripts/run_object_replay.py",
    "scripts/check_staged_restores.py",
    "scripts/prepare_object_trace.py",
    "scripts/prepare_kvlab.py",
    "scripts/audit_aiperf_input.py",
    "scripts/audit_weka.py",
    *(path.relative_to(PROJECT_ROOT).as_posix() for path in package_source_files()),
)
