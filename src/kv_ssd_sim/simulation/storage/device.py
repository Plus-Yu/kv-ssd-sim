"""A single FIFO shared by object reads and whole-bucket writes."""

import math
from dataclasses import dataclass
from itertools import count
from typing import Literal

MIB = 1 << 20


@dataclass(frozen=True)
class DeviceIO:
    io_id: int
    start_ns: int
    complete_ns: int


class FifoSSD:
    def __init__(self, read_mib_s: float, write_mib_s: float):
        self.rates = {"read": read_mib_s, "write": write_mib_s}
        self.free_at = 0
        self._ids = count(1)

    def submit(
        self,
        operation: Literal["read", "write"],
        size_bytes: int,
        submit_ns: int,
        *,
        observed_write: tuple[int, int] | None = None,
    ) -> DeviceIO:
        """Reserve service once. Object attribution and publication belong to the caller."""
        earliest = max(self.free_at, submit_ns)
        if observed_write is None:
            start = earliest
            end = start + math.ceil(size_bytes * 1e9 / (self.rates[operation] * MIB))
        else:
            start, end = observed_write
            if operation != "write" or start < earliest or end <= start:
                raise ValueError("Recorded write conflicts with an occupied drive or pending read")
        self.free_at = end
        return DeviceIO(next(self._ids), start, end)


def default_device(config):
    return FifoSSD(config.device_read_mib_s or config.device_mib_s, config.device_mib_s)
