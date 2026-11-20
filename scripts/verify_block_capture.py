"""Independently reconcile continuous block events and bracketed NVMe counters."""

from bisect import bisect_left, bisect_right
from itertools import pairwise

from measure_ssd import check
from verify_aiperf import read_jsonl


def check_receipt(receipt, target, mode):
    check(mode in ("stream", "buffer") and receipt["mode"] == mode, "Capture mode mismatch")
    check(
        receipt["device"] == target and receipt["target_device"] != receipt["output_device"],
        "Wrong device or recorder output",
    )
    first, last = receipt["per_cpu_start"], receipt["per_cpu_end"]
    check(first and first.keys() == last.keys(), "Missing per-CPU stats")
    read = 0
    for cpu, end in last.items():
        start = first[cpu]
        check(start["entries"] == 0, "Nonempty initial ftrace buffer")
        for key in ("overrun", "commit overrun", "dropped events"):
            check(end[key] == start[key] == 0, "Lost ftrace events on " + cpu)
        if mode == "stream":
            check(end["entries"] == 0, "Unconsumed ftrace records")
            read += end["read events"] - start["read events"]
    return read if mode == "stream" else None


def counter_cuts(blocks, markers, times):
    """Locate each sampled counter state inside its own measured time bracket."""
    offsets = (0, 2, 4, 6, 8)
    first = markers[0]
    baseline = [first["stat"][k] for k in offsets]
    index = bisect_left(times, first["before_ns"] - 1000)
    check(index == bisect_right(times, first["after_ns"] + 1000), "Ambiguous starting snapshot")
    state = baseline.copy()
    cuts = []

    def advance(stop):
        nonlocal index
        while index < stop:
            e = blocks[index]
            state[4] += 1 if e.stage == "issue" else -1
            if e.stage == "complete" and e.op in ("R", "W"):
                slot = 0 if e.op == "R" else 2
                state[slot] += 1
                state[slot + 1] += e.sectors
            index += 1

    high = 0
    for m in markers:
        lo, hi = m["before_ns"], m["after_ns"]
        check(high <= lo <= hi, "Nonmonotonic snapshot")
        high = hi
        start, stop = bisect_left(times, lo - 1000), bisect_right(times, hi + 1000)
        advance(start)
        measured = [m["stat"][k] for k in offsets]
        candidates = []
        for cut in range(start, stop + 1):
            advance(cut)
            if measured == state:
                candidates.append(cut)
        check(
            len(candidates) == 1, f"{m['label']}: counter state has {len(candidates)} possible cuts"
        )
        cuts.append(candidates[0])
    return cuts


def reconcile(blocks, markers, *, cuts=None):
    """Diskstats counts completions. Capture separate issue and completion totals."""
    times = [b.time for b in blocks]
    if cuts is None:
        cuts = []
        cut_time = 0
        # ftrace text clock has microsecond resolution. Reject racy samples explicitly.
        for m in markers:
            lo, hi = m["before_ns"], m["after_ns"]
            check(lo <= hi and cut_time <= lo, "Nonmonotonic snapshot")
            cut_time = hi
            cut = bisect_left(times, lo - 1000)
            check(
                cut == len(times) or times[cut] > hi + 1000,
                "I/O crosses diskstats sampling bracket",
            )
            cuts.append(cut)
    depth = 0
    for b in blocks[: cuts[0]]:
        depth += 1 if b.stage == "issue" else -1
    out = []
    for (left, right), (start, stop) in zip(pairwise(markers), pairwise(cuts), strict=True):
        check(left["stat"][8] == depth, "Start in-flight mismatch")
        part = blocks[start:stop]
        depth += sum(1 if b.stage == "issue" else -1 for b in part)
        check(right["stat"][8] == depth, "End in-flight mismatch")
        row = {
            "start": left["label"],
            "end": right["label"],
            "begin_ns": left["after_ns"],
            "end_ns": right["after_ns"],
            "inflight_start": left["stat"][8],
            "inflight_end": depth,
        }
        for op, offset in (("R", 0), ("W", 4)):
            issued = [b for b in part if b.op == op and b.stage == "issue"]
            done = [b for b in part if b.op == op and b.stage == "complete"]
            sectors = sum(b.sectors for b in done)
            delta = [right["stat"][i] - left["stat"][i] for i in (offset, offset + 2)]
            check(
                delta == [len(done), sectors],
                f"{row['start']} -> {row['end']} {op}: diskstats {delta}, trace {[len(done), sectors]}",
            )
            row[op] = {
                "issues": len(issued),
                "issued_bytes": sum(b.sectors * 512 for b in issued),
                "completions": len(done),
                "sectors": sectors,
                "diskstats_ios": delta[0],
                "diskstats_sectors": delta[1],
                "last_complete_ns": done[-1].time if done else None,
            }
        out.append(row)
    return out


def audit_windows(path, blocks, rows, receive, receipt, *, align_counters=False):
    marks = read_jsonl(path / "diskstats.jsonl")
    expected = [
        "capture_start",
        *["http_send", "http_response"] * len(rows),
        "store_sender_drained",
        "device_end",
    ]
    check([m["label"] for m in marks] == expected, "Missing or reordered capture markers")
    times = [b.time for b in blocks]
    cuts = counter_cuts(blocks, marks, times) if align_counters else None
    bridges = []
    auth_times = [None] * len(marks)
    for i, r in enumerate(rows):
        send, response = marks[1 + 2 * i : 3 + 2 * i]
        rid = receive[send["client_id"]]["request_id"]
        check(
            response["request_id"] == rid
            and send["after_ns"] <= r["http_start_ns"]
            and r["response_end_ns"] <= response["before_ns"],
            "Marker/client boundaries disagree",
        )
        auth_times[1 + 2 * i] = r["http_start_ns"]
        auth_times[2 + 2 * i] = r["response_end_ns"]
        if align_counters:
            continue
        for first, last in (
            (send["before_ns"], r["http_start_ns"]),
            (r["response_end_ns"], response["after_ns"]),
        ):
            check(
                not any(first - 1000 <= b.time <= last + 1000 for b in blocks),
                "Actual HTTP boundary differs from diskstats accounting window",
            )
    check(
        marks[-2]["unfinished"] == 0 and marks[-1]["quiet_seconds"] > 0,
        "Incomplete drain or quiet observation",
    )
    windows = reconcile(blocks, marks, cuts=cuts)
    if align_counters:
        # Absolute diskstats at HTTP boundaries = sampled counter + events in
        # the signed gap. At sampling cuts, every field is independently verified.
        aligned = []
        for idx, m in enumerate(marks):
            t = auth_times[idx]
            if t is None:
                aligned.append(m)
                continue
            check(
                not (bisect_right(times, t + 1000) - bisect_left(times, t - 1000)),
                "HTTP time indistinguishable from a block event",
            )
            cut = bisect_left(times, t)
            observed = cuts[idx]
            sign = 1 if cut >= observed else -1
            values = m["stat"].copy()
            between = blocks[min(cut, observed) : max(cut, observed)]
            for e in between:
                values[8] += sign * (1 if e.stage == "issue" else -1)
                if e.stage == "complete" and e.op in ("R", "W"):
                    k = 0 if e.op == "R" else 4
                    values[k] += sign
                    values[k + 2] += sign * e.sectors
            bridges.append(
                {
                    "marker": idx,
                    "label": m["label"],
                    "http_ns": t,
                    "sample_before_ns": m["before_ns"],
                    "sample_after_ns": m["after_ns"],
                    "signed_events": sign * len(between),
                    "delta_stat": [values[k] - m["stat"][k] for k in (0, 2, 4, 6, 8)],
                }
            )
            aligned.append({**m, "before_ns": t, "after_ns": t, "stat": values})
        aligned_cuts = [
            bisect_left(times, t) if t is not None else c
            for t, c in zip(auth_times, cuts, strict=True)
        ]
        http_windows = reconcile(blocks, aligned, cuts=aligned_cuts)
    end = marks[-1]["after_ns"]
    tail = [
        b
        for b in blocks
        if rows[-1]["response_end_ns"] <= b.time < end
        and b.stage == "complete"
        and b.op in ("R", "W")
    ]
    before_end = [b for b in blocks if b.time < end and b.op in ("R", "W")]
    check(
        before_end and end - before_end[-1].time >= marks[-1]["quiet_seconds"] * 1e9,
        "Device observation lacks claimed quiet period",
    )
    return {
        **(
            {"http_boundary_adjustments": bridges, "http_aligned_windows": http_windows}
            if align_counters
            else {}
        ),
        "mode": receipt["mode"],
        "collector_cpu_s": receipt["collector_cpu_s"],
        "windows": windows,
        "last_response_ns": rows[-1]["response_end_ns"],
        "sender_drained_ns": marks[-2]["after_ns"],
        "device_end_ns": end,
        "quiet_seconds": marks[-1]["quiet_seconds"],
        "tail_from_response_s": (tail[-1].time - rows[-1]["response_end_ns"]) / 1e9 if tail else 0,
        "tail_read_bytes": sum(b.sectors * 512 for b in tail if b.op == "R"),
        "tail_write_bytes": sum(b.sectors * 512 for b in tail if b.op == "W"),
        "tail_read_ios": sum(b.op == "R" for b in tail),
        "tail_write_ios": sum(b.op == "W" for b in tail),
    }
