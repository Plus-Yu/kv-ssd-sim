"""Guest-only monotonic diskstats snapshots and off-device ftrace collection."""

import argparse
import json
import os
import select
import time
from pathlib import Path

TRACE = Path("/sys/kernel/tracing")
STAT = Path("/sys/class/block/nvme0n1/stat")
STOP = Path("/run/kv-ssd-capture-stop")
READY = Path("/run/kv-ssd-capture-ready")
QUIET_SECONDS = 5


def mark(label, **fields):
    output = os.environ.get("KV_SSD_CAPTURE_ROOT")
    if not output:
        return
    if os.environ.get("KV_SSD_NAND_STATS"):
        from kv_ssd_sim.devices.femu.nand import capture_boundary

        capture_boundary(Path(output), label)
    before = time.monotonic_ns()
    values = list(map(int, STAT.read_text().split()))
    after = time.monotonic_ns()
    row = dict(label=label, before_ns=before, after_ns=after, stat=values, **fields)
    with (Path(output) / "diskstats.jsonl").open("a") as stream:
        stream.write(json.dumps(row) + "\n")


def per_cpu():
    return {
        p.parent.name: {
            k.strip(): int(v)
            for line in p.read_text().splitlines()
            for k, v in [line.split(":", 1)]
            if v.strip().isdigit()
        }
        for p in sorted(TRACE.glob("per_cpu/cpu*/stats"))
    }


def collect(mode):
    layout_run = bool(os.environ.get("KV_SSD_LAYOUT"))
    quiet_seconds = 65 if layout_run else QUIET_SECONDS
    timeout_seconds = 185 if layout_run else 60
    output = Path(os.environ["KV_SSD_CAPTURE_ROOT"])
    if (
        not Path("/run/kv-ssd-femu-guest").is_file()
        or output.stat().st_dev == Path("/mnt/femu").stat().st_dev
    ):
        raise RuntimeError("Capture requires an off-device guest output directory")
    trace_on = TRACE / "tracing_on"
    trace_on.write_text("0")
    (TRACE / "trace").write_text("\n")
    initial = per_cpu()
    fd = None
    with (output / "block-trace.txt").open("xb", buffering=0) as dest:
        if mode == "stream":
            fd = os.open(TRACE / "trace_pipe", os.O_RDONLY | os.O_NONBLOCK)

        def drain():
            while fd is not None:
                try:
                    chunk = os.read(fd, 65536)
                except BlockingIOError:
                    break
                if not chunk:
                    break
                dest.write(chunk)

        try:
            trace_on.write_text("1")
            mark("capture_start")
            READY.touch()
            stop_time = steady = None
            last = None
            cpu_start = time.process_time()
            while True:
                if fd is None:
                    time.sleep(0.1)
                else:
                    select.select([fd], [], [], 0.1)
                    drain()
                if not STOP.exists():
                    continue
                now = time.monotonic()
                if stop_time is None:
                    stop_time = now
                if now - stop_time > timeout_seconds:
                    raise TimeoutError("No quiet device observation boundary")
                raw = list(map(int, STAT.read_text().split()))
                counts = (raw[0], raw[2], raw[4], raw[6], *raw[11:])
                if last != counts or raw[8] or steady is None:
                    steady = now
                last = counts
                if now - steady >= quiet_seconds:
                    break
            mark("device_end", quiet_seconds=quiet_seconds)
            trace_on.write_text("0")
            drain()
            final = per_cpu()
            if mode == "buffer":
                with (TRACE / "trace").open("rb") as source:
                    import shutil

                    shutil.copyfileobj(source, dest)
            receipt = {
                "mode": mode,
                "device": (STAT.parent / "dev").read_text().strip().replace(":", ","),
                "output_device": output.stat().st_dev,
                "target_device": Path("/mnt/femu").stat().st_dev,
                "per_cpu_start": initial,
                "per_cpu_end": final,
                "collector_cpu_s": time.process_time() - cpu_start,
            }
            (output / "capture.json").write_text(json.dumps(receipt, indent=2) + "\n")
        finally:
            trace_on.write_text("0")
            if fd is not None:
                os.close(fd)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("buffer", "stream"))
    collect(parser.parse_args(argv).mode)


if __name__ == "__main__":
    main()
