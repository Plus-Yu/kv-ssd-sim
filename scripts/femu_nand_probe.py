"""Controlled direct I/O on the fresh, unmounted RAM-backed calibration namespace."""

import mmap
import os
import sys
from pathlib import Path

from kv_ssd_sim.nand import DEVICE, snapshot

MIB = 1024**2


def main():
    output = Path(sys.argv[1])
    dev = Path("/dev/nvme0n1")
    if not (
        "kvworkload=nand-probe" in Path("/proc/cmdline").read_text().split()
        and Path("/run/kv-ssd-femu-guest").is_file()
        and "FEMU" in (DEVICE / "device/model").read_text().upper()
        and int((DEVICE / "size").read_text()) * 512 == 48 * MIB
        and not any(
            row.split()[2] == (DEVICE / "dev").read_text().strip()
            for row in Path("/proc/self/mountinfo").read_text().splitlines()
        )
    ):
        raise RuntimeError("Expected the dedicated 48 MiB unmounted FEMU probe namespace")
    fd = os.open(dev, os.O_RDWR | os.O_DIRECT | os.O_EXCL)
    try:
        with mmap.mmap(-1, MIB) as buf, memoryview(buf) as data:

            def write(offset, size):
                if os.pwrite(fd, data[:size], offset) != size:
                    raise OSError("Short direct write")

            snapshot(output, "empty")
            data[:] = b"A" * MIB
            write(0, 4096)
            snapshot(output, "page")
            data[:512] = b"B" * 512
            write(512, 512)
            snapshot(output, "partial")
            if (
                os.preadv(fd, [data[:4096]], 0) != 4096
                or data[:4096] != b"A" * 512 + b"B" * 512 + b"A" * 3072
            ):
                raise OSError("Partial write or direct read mismatch")
            snapshot(output, "read")
            for start in range(0, 48 * MIB, MIB):
                write(start, MIB)
            snapshot(output, "filled")
            # Overwrite alternate 64 KiB stripes, leaving live pages in victim lines.
            for repeat in range(3):
                for start in range(0, 48 * MIB, 128 * 1024):
                    write(start, 64 * 1024)
            snapshot(output, "rewritten")
    finally:
        os.close(fd)


if __name__ == "__main__":
    main()
