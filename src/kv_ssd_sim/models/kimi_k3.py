"""K3 TP8 online capacity profile."""

import dataclasses

from .deployment import require


@dataclasses.dataclass(frozen=True)
class K3Profile:
    profile: str
    segment_mib: int
    staging_mib: int
    quota_mib: int

    @classmethod
    def load(cls, values):
        cfg = cls(**values)
        require(
            cfg.profile == "k3-tp8"
            and all(
                type(v) is int and v > 0 for v in (cfg.segment_mib, cfg.staging_mib, cfg.quota_mib)
            ),
            "Invalid K3 online budgets",
        )
        return cfg
