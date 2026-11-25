"""Audit reference runs from their recorded settings, replica events and SSD arrival ledger."""

import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

from check_staged_restores import audit as audit_restores


def verify(folder):
    def load(name):
        return json.loads((folder / name).read_bytes())

    launch, summary = load("launch.json"), load("summary.json")
    ssd = load("ssd-summary.json")
    store = launch["store"]
    reference = launch["reference_settings"]
    profile = Path(__file__).resolve().parents[1] / reference["file"]
    if (
        summary["status"] != "complete"
        or hashlib.sha256(profile.read_bytes()).hexdigest() != reference["sha256"]
        or json.loads(profile.read_bytes()) != reference["baseline"]
    ):
        raise ValueError(f"Run incomplete or reference profile differs in {folder}")
    memory, disk = set(), set()
    reads, starts, completions, gets, restores = (Counter() for _ in range(5))
    published_bytes = copied_bytes = 0
    disk_reads = []
    first_write = None
    with (folder / "events.jsonl").open() as stream:
        for line in stream:
            row = json.loads(line)
            event, key, size = row["event"], row.get("key"), row.get("bytes", 0)
            if event == "put_complete":
                memory.add(key)
            elif event == "evict_memory":
                memory.remove(key)
            elif event == "disk_publish":
                disk.add(key)
                published_bytes += size
            elif event == "evict_disk":
                disk.remove(key)
            elif event == "disk_write_submit" and first_write is None:
                first_write = row["time_ns"]
            elif event == "save_queued":
                copied_bytes += row["copy_bytes"]
                copy_ns = math.ceil(row["copy_bytes"] * 1e9 / (store["put_mib_s"] * 2**20))
                if row["complete_ns"] - row["start_ns"] != round(store["save_ms"] * 1e6) + copy_ns:
                    raise ValueError(
                        f"Rank PUT service time disagrees with allocated bytes in {folder}"
                    )
            if event in (
                "get_memory",
                "get_disk",
                "disk_read_submit",
                "disk_read_complete",
                "get_complete",
            ):
                identity = (row["request"], row["rank"], row["job"], key, size)
                if event == "get_disk":
                    if key in memory or key not in disk:
                        raise ValueError(f"Disk GET without a disk-only replica for {key}")
                    reads[identity] += 1
                    gets[identity] += 1
                    disk_reads.append(
                        {k: row[k] for k in ("time_ns", "request", "rank", "key", "bytes")}
                    )
                elif event == "get_memory":
                    if key not in memory:
                        raise ValueError(f"Memory GET without a memory replica for {key}")
                    gets[identity] += 1
                elif event == "disk_read_submit":
                    starts[identity] += 1
                elif event == "disk_read_complete":
                    completions[identity] += 1
                else:
                    restores[identity] += 1
    counts, sizes = summary["counts"], summary["bytes"]
    full = ssd["scopes"]["full"]
    if (
        reads != starts
        or reads != completions
        or gets != restores
        or copied_bytes != sizes.get("put_allocated", 0)
        or full["write_bytes"] != published_bytes
        or full["write_bytes"] != sizes.get("disk_write_submit", 0)
        or full["read_bytes"] != sum(item[-1] * n for item, n in reads.items())
    ):
        raise ValueError(f"GET, PUT or SSD accounting failed in {folder}")
    if launch.get("restore_path") == "host-staged" and audit_restores(folder) != load(
        "restore-summary.json"
    ):
        raise ValueError(f"Restore stage audit differs from the recorded report in {folder}")
    return {
        "directory": str(folder),
        "profile_sha256": reference["sha256"],
        "max_model_len": launch["max_model_len"],
        "requests": summary["requests"],
        "unique_keys": summary["unique_keys"],
        "local_reuse_tokens": sum(row["local_reuse_tokens"] for row in load("requests.json")),
        "natural_disk_gets": counts.get("get_disk", 0),
        "natural_disk_keys": len({row["key"] for row in disk_reads}),
        "memory_gets": counts.get("get_memory", 0),
        "first_write_submit_ns": first_write,
        "last_request_finished_ns": ssd["last_request_finished_ns"],
        "write_submitted_after_foreground_bytes": ssd["submitted_after_last_finish_bytes"],
        "read_bytes": full["read_bytes"],
        "write_bytes": full["write_bytes"],
        "end_ns": full["end_ns"],
        "disk_read_events": disk_reads,
        "sha256": {
            name: digest(folder / name)
            for name in (
                "launch.json",
                "events.jsonl",
                "summary.json",
                "ssd-summary.json",
                "requests.json",
            )
        },
    }


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+", type=Path)
    options = parser.parse_args()
    print(json.dumps([verify(folder) for folder in options.runs], indent=2))
