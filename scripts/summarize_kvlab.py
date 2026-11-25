"""Summarize coverage, source/target clocks and the measured/modelled SSD windows."""

import argparse
import csv
import json
import sqlite3
from collections import Counter
from pathlib import Path

from run_kvlab_matrix import HISTORICAL_SCENARIOS, PUBLIC_SCENARIOS

from kv_ssd_sim.kvlab import LIMITS, NS, candidates, dump


def load(path):
    return json.loads(path.read_bytes())


def table(path, rows):
    with path.open("x", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def interval_curve(pairs):
    events = Counter()
    for left, right in pairs:
        if right < left:
            raise ValueError("Backwards request interval")
        events[left] += 1
        events[right] -= 1
    active = 0
    for when, d in sorted(events.items()):
        active += d
        yield when, active
    if active:
        raise ValueError("Unclosed interval")


def daily(db, limits):
    rows = []
    for cap in map(int, limits):
        bit = 1 << LIMITS.index(cap)
        query = """SELECT date(at/1000000000,'unixepoch'), count(*), sum(n_in),
               sum(n_in<=0 OR n_out<=0),sum(n_in+n_out>?),sum((direct & ?)>0),
               sum((closed & ?)>0),sum(CASE WHEN (closed & ?)>0 THEN n_in ELSE 0 END)
               FROM calls GROUP BY 1 ORDER BY 1"""
        for day, n, t, missing, over, fit, closed, closed_tokens in db.execute(
            query, (cap, bit, bit, bit)
        ):
            rows.append(
                {
                    "utc_date": day,
                    "cap_tokens": cap,
                    "requests": n,
                    "input_tokens": t,
                    "missing_usage_requests": missing,
                    "over_context_requests": over,
                    "fits_context": fit,
                    "dependency_closed": closed,
                    "closed_input_tokens": closed_tokens,
                }
            )
    return rows


def add_curve(curves, group, label, case, pairs):
    points = list(interval_curve(pairs))
    curves.extend(
        {
            "window_set": group,
            "window": label,
            "clock": case,
            "relative_ns": t,
            "in_flight": n,
        }
        for t, n in points
    )
    return max(n for _, n in points)


def collect(study, out):
    out.mkdir(parents=True, exist_ok=False)
    curves = []
    scenarios = []
    for group, windows, result in (
        ("reference", "reference", "demand-reference"),
        ("paired-reference", "paired", "demand-paired"),
    ):
        for label in ("calibration", "holdout"):
            source = load(study / windows / label / "window.json")
            t0 = source["start"]
            pairs = [
                (r["source_utc_ns"] - t0, r["source_end_utc_ns"] - t0)
                for r in source["source_requests"]
            ]

            source_peak = add_curve(curves, group, label, "recorded", pairs)
            base_launch = load(study / result / label / "base" / "launch.json")
            settings = (
                PUBLIC_SCENARIOS if "reference_settings" in base_launch else HISTORICAL_SCENARIOS
            )
            for setting in settings:
                folder = study / result / label / setting
                s = load(folder / "summary.json")
                d = load(folder / "ssd-summary.json")
                m = load(folder / "requests.json")
                if len(m) != len(pairs):
                    raise ValueError("Dropped source request")
                peak = add_curve(
                    curves, group, label, setting, [(r["arrival_ns"], r["finished_ns"]) for r in m]
                )
                full = d["scopes"]["full"]
                measured = d["scopes"]["measurement"]
                state = d["anchors"][0]
                writes = full["write_bytes"]
                reads = full["read_bytes"]
                if (
                    s["bytes"].get("disk_read_submit", 0),
                    s["bytes"].get("disk_write_submit", 0),
                ) != (reads, writes):
                    raise ValueError("Integrated arrival bins differ from raw Store counters")
                scenarios.append(
                    {
                        "window_set": group,
                        "window": label,
                        "setting": setting,
                        "unique_keys": s["unique_keys"],
                        "source_peak_inflight": source_peak,
                        "target_peak_inflight": peak,
                        "read_bytes": reads,
                        "write_bytes": writes,
                        "read_byte_fraction": reads / (reads + writes) if reads + writes else None,
                        "store_memory_get_bytes": s["bytes"].get("get_memory", 0),
                        "external_tokens": sum(r["external_load_tokens"] for r in m),
                        "local_reuse_tokens": sum(r["local_reuse_tokens"] for r in m),
                        "read_peak_10s_bytes": full["read_peak_10s_bytes"],
                        "write_peak_10s_bytes": full["write_peak_10s_bytes"],
                        "foreground_end_ns": d["last_request_finished_ns"],
                        "end_ns": full["end_ns"],
                        "tail_submitted_bytes": d["submitted_after_last_finish_bytes"],
                        "anchor_ns": state["time_ns"],
                        "anchor_store_memory_bytes": state["memory_payload_bytes"],
                        "anchor_local_cached_hashes": state["local_cached_hashes"],
                        "anchor_published_disk_bytes": state["published_disk_bytes"],
                        "after_anchor_read_bytes": measured["read_bytes"],
                        "after_anchor_write_bytes": measured["write_bytes"],
                    }
                )
    table(out / "sensitivity.csv", scenarios)
    table(out / "inflight.csv", curves)
    audit = load(study / "corpus/audit.json")
    with sqlite3.connect(
        (study / "corpus/index.sqlite").resolve().as_uri() + "?mode=ro", uri=True
    ) as db:
        table(out / "daily-coverage.csv", daily(db, audit["limits"]))
        unrestricted = candidates(db, 12288, 40 * 86400, audit["requests"])
    hold = []
    for n in (1, 2, 3):
        s = load(study / f"score-{n}/score.json")
        hold.append(
            {
                "trial": n,
                "frozen": s["freeze"],
                "all_object_checks": all(s["object_checks"].values()),
                "all_ssd_keys": s["object_sets_match"],
                "read_bytes": s["ssd_counts"]["read"]["native"]["bytes"],
                "write_bytes": s["ssd_counts"]["write"]["native"]["bytes"],
                "read_false_positive_bytes": s["ssd_counts"]["read"]["false_positive"]["bytes"],
                "read_missed_bytes": s["ssd_counts"]["read"]["missed"]["bytes"],
                "per_key_median_allocation_error_ms": s["boundary_model_minus_native_ms"][
                    "allocated"
                ]["median"],
                "per_key_median_memory_ready_error_ms": s["boundary_model_minus_native_ms"][
                    "memory_ready"
                ]["median"],
                "per_key_median_submit_error_ms": s["boundary_model_minus_native_ms"][
                    "write_submit"
                ]["median"],
                "per_key_median_disk_ack_error_ms": s["boundary_model_minus_native_ms"]["disk_ack"][
                    "median"
                ],
                "submit_after_last_request_bytes": s["submitted_after_foreground_bytes"],
            }
        )
    table(out / "native-holdout.csv", hold)
    regression = load(study / "regression.json")
    checks = regression["checks"]
    totals = regression["prediction"]
    slice_ok = all(
        checks[k]
        for k in (
            "native_completed",
            "requests_and_tokens",
            "inventory_key_size",
            "put_request_rank_key_bytes",
            "get_request_rank_key_bytes",
            "final_disk_keys",
        )
    )
    slice_ok &= (totals["unique_keys"], totals["unique_bytes"], totals["gets"]) == (
        208,
        2402574336,
        0,
    )
    slice_ok &= (
        regression["native_key_content_sha256"]
        == "173bfdbe0ff676b4301cee05be60c30795a68772776cbd8565773d966fda0999"
    )
    dump(
        out / "acceptance.json",
        {
            "full_corpus_verified": audit["source_sha256"],
            "n15_uncapped_search": {
                "cold_complete_prefixes_with_object": len(unrestricted),
                "maximum_calls": max((c["calls"] for c in unrestricted), default=0),
                "return_after_competitor_candidates": sum(c["quality"][0] for c in unrestricted),
            },
            "paired_frozen_write_identity_all_three": all(
                r["all_ssd_keys"] and r["all_object_checks"] and r["frozen"] for r in hold
            ),
            "natural_read_validation": "no native positive examples"
            if not any(r["read_bytes"] for r in hold)
            else "see per-trial confusion counts",
            "0504_required_object_regression": slice_ok,
            "0504_full_comparison": regression["status"],
            "0504_final_memory_keys": regression["final_memory_keys"],
            "reference_scenarios": len(scenarios),
            "scope": "object arrivals at the file backend; hardware and whole-corpus rates are uncalibrated",
        },
    )
    plot(study, out)


def plot(study, out):
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    styles = {
        "base": ("#162334", "-", "Reference baseline"),
        "compute-fast": ("#22947c", ":", "Compute time ×0.5"),
        "compute-slow": ("#a77925", "--", "Compute time ×2"),
        "storage-fast": ("#3078c4", "-", "Storage time ×0.5"),
        "storage-slow": ("#b84a46", "-", "Storage time ×2"),
        "ssd-half-bandwidth": ("#3078c4", "-", "SSD time ×2"),
        "ssd-quarter-bandwidth": ("#b84a46", "-", "SSD time ×4"),
    }
    fig, axes = plt.subplots(1, 2, figsize=(10.3, 4.4), sharex=True, sharey=True)
    for ax, label, title in zip(
        axes, ("calibration", "holdout"), ("Window A", "Window B"), strict=True
    ):
        launch = load(study / "demand-reference" / label / "base" / "launch.json")
        cases = PUBLIC_SCENARIOS if "reference_settings" in launch else HISTORICAL_SCENARIOS
        for case in cases:
            color, style, legend = styles[case]
            src = study / "demand-reference" / label / case / "ssd-10s.csv"
            with src.open() as f:
                rows = [r for r in csv.DictReader(f) if r["scope"] == "full"]
            edges = [int(r["from_ns"]) / NS for r in rows] + [int(rows[-1]["until_ns"]) / NS]
            ax.stairs(
                [int(r["write_bytes"]) / 1e9 for r in rows],
                edges,
                color=color,
                linestyle=style,
                linewidth=2 if case == "base" else 1.45,
                baseline=None,
                label=legend,
            )
        ax.set_title(title)
        ax.set_xlabel("Time since first arrival (s)")
        ax.grid(axis="y", alpha=0.22)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Write payload submitted per 10 s (GB)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=3, loc="lower center", frameon=False)
    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.27, top=0.89, wspace=0.15)
    fig.savefig(out / "reference-write-arrivals.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("study", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    collect(args.study, args.output)
