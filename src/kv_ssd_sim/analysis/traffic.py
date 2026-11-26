"""Object-level SSD submissions, completion times and bandwidth windows."""

import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory

from .reuse import describe_traffic


def export_demand(folder, events, end_ns, requests):
    """Join every submitted object to its own dispatch and completion, including the drain."""
    import csv
    import gzip

    pending = {}
    completed = []
    anchors = []
    with events.open() as source:
        for line in source:
            e = json.loads(line)
            name = e["event"]
            if name == "measurement_anchor":
                anchors.append(e)
            if name not in {
                "disk_write_submit",
                "disk_read_submit",
                "disk_read_start",
                "disk_write_start",
                "disk_publish",
                "disk_read_complete",
            }:
                continue
            identity = (e["io_id"], e["key"])
            if name.endswith("_submit"):
                if identity in pending:
                    raise ValueError(f"Duplicate SSD submission {identity}")
                pending[identity] = {
                    "op": "read" if name == "disk_read_submit" else "write",
                    "request": e["request"],
                    "rank": e["rank"],
                    "key": e["key"],
                    "bytes": e["bytes"],
                    "owner": e["owner"],
                    "staging_client": e["staging_client"],
                    **({"group": e["group"]} if "group" in e else {}),
                    "io_id": e["io_id"],
                    "submit_ns": e["time_ns"],
                }
            elif name.endswith("_start"):
                pending[identity]["start_ns"] = e["time_ns"]
            else:
                r = pending.pop(identity)
                r.update(complete_ns=e["time_ns"], bucket_id=e["bucket_id"])
                if not r["submit_ns"] <= r["start_ns"] <= r["complete_ns"] <= end_ns:
                    raise ValueError(f"SSD lifecycle out of order: {identity}")
                completed.append(r)
    if pending:
        raise ValueError("Unfinished SSD submissions at final boundary")
    completed.sort(key=lambda r: (r["submit_ns"], r["io_id"], r["key"]))
    with gzip.open(folder / "ssd-demand.jsonl.gz", "xt") as target:
        for r in completed:
            target.write(json.dumps(r) + "\n")
    width = 10_000_000_000
    begin = min(r["arrival_ns"] for r in requests)
    foreground_end = max(r["finished_ns"] for r in requests)

    def bins(start):
        result = [
            {
                "from_ns": start + i * width,
                "until_ns": min(end_ns, start + (i + 1) * width),
                "read_objects": 0,
                "read_bytes": 0,
                "write_objects": 0,
                "write_bytes": 0,
            }
            for i in range(1 + (end_ns - start) // width)
        ]
        for r in completed:
            if r["submit_ns"] >= start:
                b = result[(r["submit_ns"] - start) // width]
                b[r["op"] + "_objects"] += 1
                b[r["op"] + "_bytes"] += r["bytes"]
        return result

    boundaries = {"full": begin}
    if len(anchors) == 1:
        boundaries["measurement"] = anchors[0]["time_ns"]
    curves = {}
    with (folder / "ssd-10s.csv").open("x", newline="") as target:
        writer = None
        for name, start in boundaries.items():
            series = bins(start)
            if writer is None:
                writer = csv.DictWriter(target, ["scope", *series[0]])
                writer.writeheader()
            for b in series:
                writer.writerow(dict(scope=name, **b))
            curves[name] = dict(
                start_ns=start,
                end_ns=end_ns,
                bins=len(series),
                **{
                    op + "_bytes": sum(b[op + "_bytes"] for b in series) for op in ("read", "write")
                },
                **{
                    op + "_peak_10s_bytes": max(b[op + "_bytes"] for b in series)
                    for op in ("read", "write")
                },
            )
    return {
        "scopes": curves,
        "anchors": anchors,
        "last_request_finished_ns": foreground_end,
        "submitted_after_last_finish_bytes": sum(
            r["bytes"] for r in completed if r["submit_ns"] > foreground_end
        ),
        "completed_after_last_finish_bytes": sum(
            r["bytes"] for r in completed if r["complete_ns"] > foreground_end
        ),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error("Choose a new output directory")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    summary = json.loads((args.run / "summary.json").read_bytes())
    requests = json.loads((args.run / "requests.json").read_bytes())
    events = args.run / "events.jsonl"
    touches = sum(summary["counts"].get(k, 0) for k in ("put_complete", "get_memory", "get_disk"))
    with TemporaryDirectory(dir=args.output.parent) as temp:
        out = Path(temp) / "analysis"
        out.mkdir()
        _, reused = describe_traffic(events, out / "reuse.jsonl", touches)
        if reused != summary["cross_request_store_reuses"]:
            raise ValueError("Reuse ledger differs from the recorded summary")
        report = export_demand(out, events, summary["virtual_end_ns"], requests)
        (out / "ssd-summary.json").write_text(json.dumps(report, indent=2) + "\n")
        out.rename(args.output)
    print(json.dumps(report))
