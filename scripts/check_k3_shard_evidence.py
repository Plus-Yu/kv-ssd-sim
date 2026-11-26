"""Cross-check a saved Step 3 run against the pinned geometry and raw Store events."""

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from kv_ssd_sim.core.events import load_events
from kv_ssd_sim.core.paths import PROJECT_ROOT
from kv_ssd_sim.models.deployment import require


def check(directory):
    plan_path = PROJECT_ROOT / "configs/kimi-k3-b300-step2/resolved.json"
    plan = json.loads(plan_path.read_text())
    report = json.loads((directory / "summary.json").read_text())
    require(
        report["status"] == "passed"
        and report["step2_sha256"] == hashlib.sha256(plan_path.read_bytes()).hexdigest(),
        "Wrong plan or unsuccessful run",
    )
    require(
        report["lookup_tokens"]
        == {"rank0_only": 0, "missing_group2_rank7": 0, "all_disk_only": 1536},
        "Lookup result changed",
    )
    events = load_events(directory / "events.jsonl")
    objects = {obj["key"]: obj for obj in report["objects"]}
    require(len(objects) == len(report["objects"]) == 36, "Object inventory differs")
    actual = set()
    for obj in objects.values():
        group = plan["groups"][obj["group"]]
        require(
            obj["bytes"] == group["logical_bytes_per_object_per_shard"]
            and obj["rank"] == obj["shard"],
            "Size or rank mismatch",
        )
        actual.add((obj["group"], obj["shard"], obj["end_token"]))
    expected = {(group, rank, 1536) for group in range(3) for rank in range(8)} | {
        (3, 0, end) for end in range(128, 1537, 128)
    }
    require(actual == expected, "Missing logical shard or chunk")
    writes, begins, ends = [], [], []
    for event in events:
        if event["event"] not in {"store_put_end", "store_get_begin", "store_get_end"}:
            continue
        (key,) = event["keys"]
        obj = objects[key]
        if event["event"] == "store_put_end":
            require(
                event["result"] == [0]
                and event["object_bytes"] == [obj["bytes"]]
                and event["digests"] == [obj["sha256"]],
                "PUT evidence mismatch",
            )
            writes.append(key)
        elif event["event"] == "store_get_begin":
            require(
                event["tiers"] == {key: ["LOCAL_DISK"]} and event["bytes"] == obj["bytes"],
                "Not a disk-only read",
            )
            begins.append(key)
        else:
            require(
                event["result"] == [obj["bytes"]] and event["checksum_ok"] == [True],
                "GET byte count or checksum failed",
            )
            ends.append(key)
    require(
        len(writes) == len(set(writes)) and set(writes) == set(objects), "PUT coverage mismatch"
    )
    require(len(begins) == 135 and begins == ends, "GET coverage mismatch")
    restores = report["restores"]
    require([row["key"] for row in restores] == ends[15:], "Full restore disagrees with log")
    require(
        Counter(ends[:15]) == Counter(key for key, obj in objects.items() if obj["rank"] == 0),
        "Rank zero restore incomplete",
    )
    expected_reads = {
        (rank, key)
        for rank in range(8)
        for key, obj in objects.items()
        if obj["group"] == 3 or obj["shard"] == rank
    }
    require(
        len(restores) == len(expected_reads)
        and {(r["rank"], r["key"]) for r in restores} == expected_reads,
        "Rank coverage mismatch",
    )
    require(
        all(r["bytes"] == objects[r["key"]]["bytes"] for r in restores)
        and sum(obj["bytes"] for obj in objects.values()) == report["unique_object_bytes"],
        "Reported byte totals differ",
    )
    return {
        "status": "passed",
        "objects": len(objects),
        "disk_only_reads": len(ends),
        "all_rank_restores": len(restores),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    print(json.dumps(check(parser.parse_args().directory)))
