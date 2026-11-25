"""Reconcile SSD reads, file-owner staging, H2D ports and TP acknowledgements from raw events."""

import argparse
import json
import math
from collections import Counter
from pathlib import Path

COMMON = ("h2d_ready", "h2d_start", "h2d_complete")
MEMORY = ("get_memory", *COMMON, "get_complete")
DISK = (
    "get_disk",
    "staging_reserve",
    "disk_read_submit",
    "disk_read_start",
    "disk_read_complete",
    *COMMON,
    "staging_release",
    "get_complete",
)
RESTORE_EVENTS = set(MEMORY + DISK) | {"staging_wait"}


def audit(folder, summary=None):
    def load(name):
        return json.loads((folder / name).read_bytes())

    launch = load("launch.json")
    if summary is None:
        summary = load("summary.json")
    if launch.get("restore_path") != "host-staged":
        raise ValueError("This audit requires the host-staged restore model")
    cfg = launch["store"]
    capacity = cfg["staging_bytes"]
    used = [0] * cfg["clients"]
    peaks = used.copy()
    active, loads, busy_rank, busy_source, last_batch = {}, {}, {}, {}, {}
    remaining = Counter()
    counts, sizes, h2d_ns, queue_ns = Counter(), Counter(), Counter(), Counter()
    peak_ranks = waits = wait_ns = max_disk_lag = disk_lag = 0
    previous_time = last_restore_id = 0
    with (folder / "events.jsonl").open() as stream:
        for line in stream:
            row = json.loads(line)
            event, now = row["event"], row["time_ns"]
            if now < previous_time:
                raise ValueError("Nonmonotonic event clock")
            previous_time = now
            if event == "load_start":
                load_id = row["load_id"]
                if load_id in loads or row["ranks"] != 8:
                    raise ValueError(f"Invalid TP8 load: {row}")
                loads[load_id] = {
                    "request": row["request"],
                    "start_ns": now,
                    "rank_done": {},
                    "last_disk_complete_ns": None,
                }
                continue
            if event in ("rank_load_complete", "load_complete"):
                load_id = row["load_id"]
                job = loads[load_id]
                if job["request"] != row["request"] or "complete_ns" in job:
                    raise ValueError(f"Duplicate or mismatched load acknowledgement: {row}")
                if event == "rank_load_complete":
                    rank = row["rank"]
                    if not 0 <= rank < 8 or remaining[load_id, rank] or rank in job["rank_done"]:
                        raise ValueError(
                            f"Rank acknowledged before its objects or more than once: {row}"
                        )
                    job["rank_done"][rank] = now
                else:
                    if len(job["rank_done"]) != 8 or now != max(job["rank_done"].values()):
                        raise ValueError(f"Load did not complete at the TP barrier: {row}")
                    job["complete_ns"] = now
                continue
            if event not in RESTORE_EVENTS:
                continue
            rid, rank, owner, size = (row[k] for k in ("restore_id", "rank", "owner", "bytes"))
            load_id = row["load_id"]
            batch = row["load_batch"]
            identity = (load_id, batch, row["request"], rank, row["job"], row["key"], size, owner)
            job = loads[load_id]
            if row["request"] != job["request"] or rank in job["rank_done"]:
                raise ValueError(f"Object outside its rank load: {row}")
            if event in ("get_disk", "get_memory"):
                if rid <= last_restore_id or not 0 <= owner < len(used) or not 0 <= rank < 8:
                    raise ValueError(f"Invalid restore identity: {row}")
                last_restore_id = rid
                prev = last_batch.get((load_id, rank), -1)
                if batch != prev and (batch != prev + 1 or remaining[load_id, rank]):
                    raise ValueError(
                        f"Native GET sub-batches overlap or arrived out of order: {row}"
                    )
                last_batch[load_id, rank] = batch
                active[rid] = (identity, DISK if event == "get_disk" else MEMORY, {})
                remaining[load_id, rank] += 1
            stored, sequence, times = active[rid]
            if stored != identity:
                raise ValueError(f"Restore identity changed: {row}")
            if event == "staging_wait":
                if tuple(times) != ("get_disk",) or owner != row["staging_client"]:
                    raise ValueError(f"Invalid restore staging wait: {row}")
                waits += 1
                continue
            if sequence[len(times)] != event:
                raise ValueError(f"Unexpected restore phase after {tuple(times)}: {row}")
            times[event] = now
            tier = "disk" if sequence == DISK else "memory"
            if (event in COMMON and row["tier"] != tier) or (
                event.startswith("disk_read_") and row["staging_client"] != owner
            ):
                raise ValueError(f"Wrong source tier or staging owner: {row}")
            if event in ("staging_reserve", "staging_release"):
                used[owner] += size if event == "staging_reserve" else -size
                peaks[owner] = max(peaks[owner], used[owner])
                if (
                    row["staging_client"] != owner
                    or not 0 <= used[owner] <= capacity
                    or row["used_bytes"] != used[owner]
                ):
                    raise ValueError(f"Staging capacity or owner mismatch: {row}")
                if event == "staging_reserve":
                    wait_ns += now - times["get_disk"]
                elif now != times["h2d_complete"]:
                    raise ValueError(f"Staging not released at H2D completion: {row}")
            elif event == "disk_read_complete":
                rate = cfg["device_read_mib_s"] or cfg["device_mib_s"]
                if now - times["disk_read_start"] != math.ceil(size * 1e9 / (rate * 2**20)):
                    raise ValueError(f"Incorrect SSD read duration: {row}")
                job["last_disk_complete_ns"] = now
                sizes["ssd_read"] += size
            elif event == "h2d_ready":
                ready = times["disk_read_complete" if tier == "disk" else "get_memory"]
                if now != ready:
                    raise ValueError(f"Incorrect H2D readiness: {row}")
            elif event == "h2d_start":
                if rank in busy_rank or (tier == "memory" and owner in busy_source):
                    raise ValueError(f"Overlapping H2D or memory source service: {row}")
                busy_rank[rank] = rid
                if tier == "memory":
                    busy_source[owner] = rid
                peak_ranks = max(peak_ranks, len(busy_rank))
                queue_ns[tier] += now - times["h2d_ready"]
            elif event == "h2d_complete":
                rate = cfg["h2d_mib_s"]
                if tier == "memory":
                    rate = min(rate, cfg["memory_mib_s"])
                    if busy_source.pop(owner) != rid:
                        raise ValueError(f"Wrong memory port completion: {row}")
                duration = now - times["h2d_start"]
                if busy_rank.pop(rank) != rid or duration != math.ceil(size * 1e9 / (rate * 2**20)):
                    raise ValueError(f"Incorrect H2D port or duration: {row}")
                h2d_ns[tier] += duration
            elif event == "get_complete":
                if now != times["h2d_complete"]:
                    raise ValueError(f"GET acknowledged at wrong time: {row}")
                if tier == "disk":
                    lag = now - times["disk_read_complete"]
                    disk_lag += lag
                    max_disk_lag = max(lag, max_disk_lag)
                counts[tier] += 1
                sizes[tier] += size
                remaining[load_id, rank] -= 1
                del active[rid]
    if (
        active
        or busy_rank
        or busy_source
        or any(used)
        or any(remaining.values())
        or any("complete_ns" not in job for job in loads.values())
        or peaks != [c["peak_restore_staging_bytes"] for c in summary["clients"]]
        or used != [c["restore_staging_bytes"] for c in summary["clients"]]
        or sizes["ssd_read"] != load("ssd-summary.json")["scopes"]["full"]["read_bytes"]
        or any(sizes[t] != summary["bytes"].get("get_" + t, 0) for t in ("memory", "disk"))
    ):
        raise ValueError("Unfinished restore or inconsistent SSD/GET/staging totals")
    for job in loads.values():
        job["rank_done_ns"] = [job["rank_done"][rank] for rank in range(8)]
        del job["rank_done"]
    return {
        "gets": {t: counts[t] for t in ("memory", "disk")},
        "h2d_bytes": {t: sizes[t] for t in ("memory", "disk")},
        "ssd_read_bytes": sizes["ssd_read"],
        "peak_concurrent_h2d_ranks": peak_ranks,
        "h2d_service_ns_sum": dict(h2d_ns),
        "h2d_queue_ns_sum": dict(queue_ns),
        "staging_capacity_bytes_per_client": capacity,
        "staging_peak_bytes_by_client": peaks,
        "staging_final_bytes_by_client": used,
        "staging_wait_objects": waits,
        "staging_wait_ns_sum": wait_ns,
        "disk_complete_to_get_ns_sum": disk_lag,
        "disk_complete_to_get_ns_max": max_disk_lag,
        "loads": list(loads.values()),
    }


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run", type=Path)
    print(json.dumps(audit(p.parse_args(argv).run), indent=2))
