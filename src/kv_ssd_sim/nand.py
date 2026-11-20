"""Guest side of bracketed, read-only FEMU QMP sampling."""

import json
import time
from pathlib import Path

DEVICE = Path("/sys/class/block/nvme0n1")


def snapshot(output, label):
    """The host reads one atomic FTL listing between these guest samples."""
    prefix = output / f"nand-{label}"
    request = prefix.with_suffix(".request")
    reply = prefix.with_suffix(".qmp.json")
    if request.exists() or reply.exists():
        raise ValueError("Duplicate NAND snapshot " + label)
    start = time.monotonic_ns()
    left = list(map(int, (DEVICE / "stat").read_text().split()))
    temp = prefix.with_suffix(".tmp")
    temp.write_text(label)
    temp.replace(request)
    while not reply.exists():
        if time.monotonic_ns() - start > 30_000_000_000:
            raise TimeoutError("No QMP response for " + label)
        time.sleep(0.01)
    right = list(map(int, (DEVICE / "stat").read_text().split()))
    end = time.monotonic_ns()
    row = {
        "label": label,
        "before_ns": start,
        "after_ns": end,
        "stat_before": left,
        "stat_after": right,
        "namespace_sectors": int((DEVICE / "size").read_text()),
    }
    prefix.with_suffix(".guest.json").write_text(json.dumps(row) + "\n")
    if left[8] or right[8] or any(left[i] != right[i] for i in (0, 2, 4, 6, 11, 13, 15)):
        raise ValueError("Device I/O crossed NAND snapshot " + label)


def capture_boundary(output, label):
    if label == "http_send":
        try:
            Path("/run/kv-ssd-nand-first-send").touch(exist_ok=False)
        except FileExistsError:
            return
        snapshot(output, "first_send")
    elif label == "device_end":
        snapshot(output, label)
