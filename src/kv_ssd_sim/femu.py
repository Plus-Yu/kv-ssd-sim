"""Fail-closed identification and observation of a real FEMU-backed filesystem."""

from __future__ import annotations

import os
import re
from pathlib import Path


class FEMUDevice:
    def __init__(self, mount: Path):
        self.mount = mount.resolve(strict=True)
        device = self.mount.stat().st_dev
        self.sysfs = Path(f"/sys/dev/block/{os.major(device)}:{os.minor(device)}").resolve(
            strict=True
        )
        self.name = self.sysfs.name
        if not re.fullmatch(r"nvme\d+n\d+", self.name):
            raise ValueError("FEMU verification requires a directly mounted NVMe namespace")
        model_path = self.sysfs / "device/model"
        self.model = model_path.read_text().strip()
        if "FEMU" not in self.model.upper() or "BLACKBOX" not in self.model.upper():
            raise ValueError(f"Refusing to label {self.model!r} as FEMU BlackBox")
        self.samples = []

    def counters(self):
        values = list(map(int, (self.sysfs / "stat").read_text().split()))
        return {
            "reads": values[0],
            "read_sectors": values[2],
            "read_ms": values[3],
            "writes": values[4],
            "write_sectors": values[6],
            "write_ms": values[7],
        }

    def describe(self):
        return {
            "mount": str(self.mount),
            "device": self.name,
            "model": self.model,
            "sysfs": str(self.sysfs),
            "block_counters": self.counters(),
        }

    def clear_guest_page_cache(self):
        # Drop caches only in the disposable guest created by our launcher.
        marker = Path("/run/kv-ssd-femu-guest")
        if not marker.exists() or marker.read_text().strip() != "disposable-femu-guest":
            raise RuntimeError("Page-cache eviction requires the disposable FEMU guest marker")
        os.sync()
        Path("/proc/sys/vm/drop_caches").write_text("3\n")

    def record_get(self, before, events, request_id):
        after = self.counters()
        delta = {key: after[key] - before[key] for key in before}
        sample = {"request_id": request_id, "device": self.name, "delta": delta}
        self.samples.append(sample)
        events.emit("femu_block_io_during_get", **sample)

    def verify(self):
        if not self.samples or sum(s["delta"]["read_sectors"] for s in self.samples) <= 0:
            raise RuntimeError("No FEMU block reads observed during native Store GET; not verified")
        return {
            **self.describe(),
            "get_io_samples": self.samples,
            "verification": "native disk restore with observed FEMU namespace reads",
        }
