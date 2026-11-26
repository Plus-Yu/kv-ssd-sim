"""Validated virtual storage budgets and service rates."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class StoreConfig:
    clients: int
    segment_bytes: int
    staging_bytes: int
    quota_bytes: int
    offload_delay_ms: float = 1000
    device_mib_s: float = 1024
    memory_mib_s: float = 8192
    lease_ms: float = 0
    save_ms: float = 1
    high_watermark: float = 0.90
    eviction_ratio: float = 0.05
    put_allocation_ms: float = 0
    first_offload_ms: float | None = None
    device_read_mib_s: float | None = None
    put_mib_s: float | None = None
    disk_high_watermark: float | None = None
    disk_low_watermark: float | None = None
    h2d_mib_s: float | None = 8192  # None explicitly selects the historical completion clock.
    host_copy_lanes: int | None = None

    def __post_init__(self):
        if self.host_copy_lanes is not None and (
            type(self.host_copy_lanes) is not int
            or not 1 <= self.host_copy_lanes <= 16
            or self.h2d_mib_s is None
            or self.put_mib_s is None
        ):
            raise ValueError("Shared copy lanes require staged GET, timed PUT and 1–16 lanes")
        if self.clients not in (1, 8) or any(
            type(v) is not int or v <= 0
            for v in (self.segment_bytes, self.staging_bytes, self.quota_bytes)
        ):
            raise ValueError("Expected 1 or 8 clients with positive per-client byte budgets")
        for v in (self.offload_delay_ms, self.lease_ms, self.save_ms):
            if not math.isfinite(v) or v < 0:
                raise ValueError("Delay must be finite and nonnegative")
        if int(self.lease_ms) != self.lease_ms:
            raise ValueError("Mooncake lease TTL must be a whole number of milliseconds")
        if not 0 <= self.put_allocation_ms <= self.save_ms:
            raise ValueError("Allocation must lie inside the per-rank save service")
        if (
            self.first_offload_ms is not None
            and not 0 <= self.first_offload_ms <= self.offload_delay_ms
        ):
            raise ValueError("Initial heartbeat phase must lie in the first polling interval")
        if not 0 < self.eviction_ratio < self.high_watermark <= 1:
            raise ValueError("Expected 0 < eviction ratio < high watermark <= 1")
        low, high = self.disk_low_watermark, self.disk_high_watermark
        if (low, high) != (None, None) and (low is None or high is None or not 0 < low < high <= 1):
            raise ValueError("Expected both disk watermarks with 0 < low < high <= 1")
        rates = [self.device_mib_s, self.memory_mib_s]
        rates.extend(
            v for v in (self.device_read_mib_s, self.put_mib_s, self.h2d_mib_s) if v is not None
        )
        for v in rates:
            if not math.isfinite(v) or v <= 0:
                raise ValueError("Service rates must be finite and positive")
