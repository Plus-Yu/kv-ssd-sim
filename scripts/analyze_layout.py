"""Join native object offsets, ext4 allocation lifetimes and issued NVMe sectors."""

import argparse
import csv
import gzip
import hashlib
import json
import re
from bisect import bisect_right
from collections import Counter, defaultdict
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path

import numpy as np
from analyze_b2 import matched_settings, read
from analyze_k3_ssd import charts, online
from measure_ssd import check


def fields(path):
    return dict(line.split(": ", 1) for line in path.read_text().splitlines() if ": " in line)


def allocation_state(path, device, block_count):
    """Retain only the last witnessed owner and logical offset of each block."""
    owner = np.zeros(block_count, dtype=np.int64)
    logical = np.zeros_like(owner)
    since = np.zeros_like(owner)
    trace = path.read_text()
    header = re.search(r"entries-in-buffer/entries-written:\s*(\d+)/(\d+)", trace)
    check(header and header[1] == header[2], "Missing or overwritten ext4 events")
    pattern = re.compile(r"(\d+)\.(\d+): ext4_(allocate|free)_blocks: dev (\d+,\d+) ino (\d+) (.*)")
    count = 0
    previous = 0
    for line in trace.splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        m = pattern.search(line)
        check(m and m[4] == device, "Unexpected ext4 event or device")
        t = int(m[1]) * 10**9 + int(m[2].ljust(9, "0"))
        check(t >= previous, "Nonmonotonic ext4 clock")
        previous = t
        vals = dict(re.findall(r"\b(block|lblk|len|count) (\d+)", m[6]))
        start = int(vals["block"])
        length = int(vals["len" if m[3] == "allocate" else "count"])
        check(0 <= start < start + length <= block_count, "Invalid allocation range")
        region = slice(start, start + length)
        if m[3] == "allocate":
            owner[region] = int(m[5])
            logical[region] = np.arange(length) + int(vals["lblk"])
        else:
            owner[region] = 0
        since[region] = t
        count += 1
    check(count == int(header[1]) and count > 0, "Incomplete ext4 trace")
    return owner, logical, since


@dataclass(frozen=True)
class Span:
    start: int
    stop: int
    label: str
    since: int = 0
    key: str = ""
    inode: int = 0
    offset: int = 0


def physical_map(path, layout, device):
    superblock = fields(path / "ext4-super.txt")
    bs = int(superblock["Block size"])
    blocks = int(superblock["Block count"])
    owner, logical, since = allocation_state(path / "ext4-alloc.trace", device, blocks)
    ranges = []

    def fixed(first, end, label):
        ranges.append(Span(first * bs, end * bs, label))

    for line in (path / "ext4-groups.txt").read_text().splitlines():
        if not re.match(r"^\d+:", line):
            continue
        _, _, sup, gdt, bb, ib, it = line.split(":")
        if int(sup) >= 0:
            fixed(int(sup), int(sup) + 1, "ext4_static")
        if gdt != "-1":
            ends = list(map(int, gdt.split("-")))
            fixed(ends[0], ends[-1] + 1, "ext4_static")
        for b in [bb, ib]:
            fixed(int(b), int(b) + 1, "ext4_static")
        fixed(int(it), int(it) + int(superblock["Inode blocks per group"]), "ext4_static")
    journal = sorted(map(int, (path / "ext4-journal.txt").read_text().split()))
    check(len(set(journal)) == int(superblock["Total journal blocks"]), "Incomplete journal")
    for first in journal:
        if ranges and ranges[-1].label == "ext4_journal" and ranges[-1].stop == first * bs:
            prev = ranges.pop()
            ranges.append(Span(prev.start, (first + 1) * bs, prev.label))
        else:
            fixed(first, first + 1, "ext4_journal")

    buckets = []
    objects = {}
    for f in layout["files"]:
        ino = f["inode"]
        regions = []
        kinds = set()
        for obj in f["objects"]:
            key = obj["key"]
            kind = "MLA" if "@store_format:attention_head_major@" in key else "KDA"
            check(kind == "MLA" or "@store_format:mamba_state@" in key, "Unknown object class")
            check(key not in objects and layout["restored"][key] == obj["sha256"], "GET mismatch")
            objects[key] = (obj["bytes"], obj["sha256"])
            kinds.add(kind)
            regions.extend(
                [
                    (obj["header_offset"], obj["file_offset"], "bucket_key", key),
                    (obj["file_offset"], obj["file_offset"] + obj["bytes"], kind, key),
                ]
            )
        if kinds:
            buckets.append("+".join(sorted(kinds)))
        elif f["size"]:
            check(f["path"].endswith(".meta"), "Unknown file")
            regions.append((0, f["size"], "object_index", ""))
        if not f["extents"]:
            check(not f["size"], "Nonempty sparse file")
            continue
        allocated_end = max(l + n for l, p, n in f["extents"])
        check(allocated_end >= f["size"], "FIEMAP does not cover EOF")
        if allocated_end > f["size"]:
            regions.append((f["size"], allocated_end, "file_slack", ""))
        cursor = 0
        for off, phys, n in f["extents"]:
            check(off == cursor and phys % bs == off % bs == n % bs == 0, "FIEMAP alignment/hole")
            cursor += n
            p, l, count = phys // bs, off // bs, n // bs
            sl = slice(p, p + count)
            valid = np.where(
                (owner[sl] == ino) & (logical[sl] == l + np.arange(count)), since[sl], -1
            )
            edges = [0, *(np.flatnonzero(np.diff(valid)) + 1).tolist(), count]
            for a, b in pairwise(edges):
                for low, high, label, key in regions:
                    start, end = max(low, off + a * bs), min(high, off + b * bs)
                    if start < end:
                        ranges.append(
                            Span(
                                phys + start - off,
                                phys + end - off,
                                label,
                                int(valid[a]),
                                key,
                                ino,
                                start,
                            )
                        )
    ranges.sort(key=lambda r: r.start)
    check(all(a.stop <= b.start for a, b in pairwise(ranges)), "Physical ranges overlap")
    check(set(objects) == set(layout["restored"]), "Unexpected recovery keys")
    check(
        sorted(buckets)
        and layout["native_plans"] > 0
        and layout["emitted_buckets"] == len(buckets),
        "Native planner count differs from disk inventory",
    )
    check(bool("KDA+MLA" in buckets) == (layout["layout"] == "mixed"), "Layout did not take effect")
    return ranges, Counter(buckets), objects


def intersect(ranges, ends, start, end, t):
    counts = Counter()
    cursor = start
    idx = bisect_right(ends, start)
    while idx < len(ranges) and ranges[idx].start < end:
        span = ranges[idx]
        low, high = max(cursor, span.start), min(end, span.stop)
        counts["unmapped"] += low - cursor
        label = span.label if 0 <= span.since < t else "lifetime_unverified"
        counts[label] += high - low
        cursor = high
        idx += 1
    counts["unmapped"] += end - cursor
    check(sum(counts.values()) == end - start, "Issue bytes do not reconcile")
    return +counts


def shared_units(ranges, unit):
    # Classes can share a unit even with a key header between the two payloads.
    edge_units = defaultdict(set)
    for span in ranges:
        if span.label in ("MLA", "KDA"):
            for edge in (span.start // unit, (span.stop - 1) // unit):
                edge_units[edge].add(span.label)
    return sum(len(classes) == 2 for classes in edge_units.values())


def analyze(path, out):
    run, bins, issues = online(path)
    layout = read(path / "layout.json")
    receipt = read(path / "capture.json")
    mode = layout["layout"]
    check(mode == read(path / "launch.json")["bucket_layout"], "Wrong planner mode")
    check(
        layout["validation_start_ns"] > run["capture"]["device_end_ns"]
        and layout["restore_device_delta"]["read_sectors"] > 0,
        "Recovery window/device mismatch",
    )
    check(run["capture"]["quiet_seconds"] == 65, "Device quiet interval differs")
    spans, buckets, objects = physical_map(path, layout, receipt["device"])
    digest = hashlib.sha256(json.dumps(sorted(objects.items())).encode()).hexdigest()
    check(digest == run["objects"]["key_content_sha256"], "On-disk bytes differ from PUT ledger")
    ends = [s.stop for s in spans]
    files = {f["inode"]: f["path"] for f in layout["files"]}
    with gzip.open(out / f"{mode}-object-LBA.csv.gz", "wt") as stream:
        w = csv.writer(stream)
        w.writerow(
            [
                "key",
                "class",
                "file",
                "inode",
                "file_offset",
                "lba512",
                "within_sector",
                "bytes",
                "alloc_ns_or_minus_one",
            ]
        )
        for s in spans:
            if s.inode:
                w.writerow(
                    [
                        s.key,
                        s.label,
                        files[s.inode],
                        s.inode,
                        s.offset,
                        s.start // 512,
                        s.start % 512,
                        s.stop - s.start,
                        s.since,
                    ]
                )
    counts = {op: Counter() for op in ("R", "W")}
    both = Counter()
    labels = [
        "MLA",
        "KDA",
        "bucket_key",
        "object_index",
        "file_slack",
        "ext4_static",
        "ext4_journal",
        "lifetime_unverified",
        "unmapped",
    ]
    with gzip.open(out / f"{mode}-io.csv.gz", "wt") as stream:
        w = csv.writer(stream)
        w.writerow(["guest_ns", "op", "lba512", "bytes", *labels])
        for event in issues:
            if event.op not in counts:
                continue
            start = event.sector * 512
            size = event.sectors * 512
            pieces = intersect(spans, ends, start, start + size, event.time)
            counts[event.op].update(pieces)
            both[event.op] += bool(pieces["MLA"] and pieces["KDA"])
            w.writerow([event.time, event.op, event.sector, size, *(pieces[l] for l in labels)])
    for op, c in counts.items():
        check(sum(c.values()) == run["io_first_send_to_device_end"][op]["bytes"], "Lost I/O bytes")
    for key, (n, _) in objects.items():
        check(
            sum(s.stop - s.start for s in spans if s.key == key and s.label in ("KDA", "MLA")) == n,
            "Object LBA map is incomplete",
        )
    coverage = {}
    for op, attributed in counts.items():
        total = sum(attributed.values())
        payload = attributed["KDA"] + attributed["MLA"]
        unknown = attributed["unmapped"] + attributed["lifetime_unverified"]
        coverage[op] = {
            "payload_fraction": payload / total if total else None,
            "explained_fraction": (total - unknown) / total if total else None,
        }
    run.update(
        coverage=coverage,
        bucket_layout=mode,
        bucket_composition=buckets,
        attribution_bytes=counts,
        issues_with_both_payload_classes=both,
        shared_sectors=shared_units(spans, 512),
        shared_fs_blocks=shared_units(spans, 4096),
        validated_objects=len(objects),
        native_planner_calls=layout["native_plans"],
        candidate_mixed_buckets=layout["mixed_input_plans"],
        recovery_counters=layout["restore_device_delta"],
    )
    return run, bins, issues, objects


def plot_layout(out, runs):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 1, sharex=True, sharey=True, figsize=(10, 6), layout="constrained")
    colors = {"KDA": "#21687f", "MLA": "#d66b25", "Both": "#8c4a91", "Other": "#787878"}
    for ax, (run, *_rest) in zip(axes, runs, strict=True):
        mode = run["bucket_layout"]
        origin = run["requests"][0]["http_start_ns"]
        series = defaultdict(list)
        with gzip.open(out / f"{mode}-io.csv.gz", "rt") as stream:
            for row in csv.DictReader(stream):
                if row["op"] != "W":
                    continue
                mla, kda = int(row["MLA"]), int(row["KDA"])
                group = "Both" if mla and kda else "MLA" if mla else "KDA" if kda else "Other"
                series[group].append(
                    ((int(row["guest_ns"]) - origin) / 1e9, int(row["lba512"]) * 512 / 2**20)
                )
        for label, color in colors.items():
            points = series[label]
            ax.scatter(
                [x for x, y in points],
                [y for x, y in points],
                s=10,
                alpha=0.65,
                color=color,
                label=label,
            )
        ax.set_title(mode)
        ax.set_ylabel("Write start (MiB)")
    axes[0].legend(loc="upper left", ncols=4)
    axes[-1].set_xlabel("Seconds from first HTTP send")
    fig.savefig(out / "layout.png", dpi=155)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mixed", type=Path)
    parser.add_argument("separate", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    check(matched_settings(args.mixed) == matched_settings(args.separate), "Frozen settings differ")
    for field in (
        "Filesystem UUID",
        "Directory Hash Seed",
        "Free blocks",
        "Free inodes",
        "Block count",
    ):
        check(
            fields(args.mixed / "ext4-super.txt")[field]
            == fields(args.separate / "ext4-super.txt")[field],
            f"Initial filesystem differs: {field}",
        )
    runs = [analyze(p, args.output) for p in (args.mixed, args.separate)]
    check(
        [r[0]["bucket_layout"] for r in runs] == ["mixed", "separate"] and runs[0][3] == runs[1][3],
        "Paired object keys, sizes or contents differ",
    )
    charts(args.output, *[[r[i] for r in runs] for i in range(3)])
    plot_layout(args.output, runs)
    (args.output / "results.json").write_text(json.dumps([r[0] for r in runs], indent=2) + "\n")
    print("Matched device and object inventories verified")


if __name__ == "__main__":
    main()
