"""Plot the six controlled stage A sessions from the exported block measurements."""

import argparse
import csv
import gzip
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

COLORS = {"R": "#006c87", "W": "#b34c19"}


def records(folder, name):
    with gzip.open(folder / (name + ".csv.gz"), "rt", newline="") as fp:
        yield from csv.DictReader(fp)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    args = parser.parse_args()
    folder = args.folder
    data = json.loads((folder / "summary.json").read_text())
    trials = sorted(data["trials"], key=lambda t: (t["trial"][-2:], t["nand_read_ns"]))
    events = defaultdict(list)
    bins = defaultdict(list)
    for e in records(folder, "events"):
        if e["stage"] == "issue" and e["phase"] not in ("before_send", "after_confirmation"):
            events[e["trial"]].append(e)
    for row in records(folder, "bins"):
        if row["width_ms"] == "1000":
            bins[row["trial"]].append(row)
    titles = [
        f"Pair {int(t['trial'][-2:])}  ·  {t['nand_read_ns'] / 1000:g} µs NAND read" for t in trials
    ]
    for kind, filename in (("timeline", "timeline.png"), ("size", "sizes.png"), ("lba", "lba.png")):
        fig, axes = plt.subplots(3, 2, figsize=(10, 7.6), sharex=True, sharey=True)
        for ax, t, title in zip(axes.flat, trials, titles, strict=True):
            trial = t["trial"]
            m = t["markers_relative_s"]
            ax.set_title(title, fontsize=10)
            ax.spines[["top", "right"]].set_visible(False)
            ax.grid(axis="y", alpha=0.18)
            if kind == "size":
                for op, color in COLORS.items():
                    sizes = t["session"][op]["size_histogram"]
                    x = sorted(map(int, sizes))
                    n = sum(sizes.values())
                    running = 0
                    y = []
                    for size in x:
                        running += sizes[str(size)]
                        y.append(100 * running / n)
                    ax.step([2] + [v / 1024 for v in x], [0] + y, where="post", color=color)
                ax.set_xscale("log", base=2)
                ax.set_xlim(2, 2048)
                ax.set_ylim(-2, 104)
            else:
                ax.axvspan(m["between_turns"], m["cold_prepare"], color=".91")
                ax.axvspan(m["cold_prepare"], m["ready_to_send"], color="#fae3be")
                ax.axvline(m["turn2_before_get"], color=".4", lw=0.8, ls=":")
                if kind == "timeline":
                    for op, field in (("R", "read_MiB_s"), ("W", "write_MiB_s")):
                        rows = bins[trial]
                        edges = [float(b["start_s"]) for b in rows] + [m["after_confirmation"]]
                        ax.stairs([float(b[field]) for b in rows], edges, color=COLORS[op], lw=1.3)
                else:
                    for op, color in COLORS.items():
                        points = [e for e in events[trial] if e["op"] == op]
                        ax.scatter(
                            [float(e["relative_s"]) for e in points],
                            [int(e["sector"]) * 512 / 2**20 for e in points],
                            s=5,
                            alpha=0.65,
                            color=color,
                            linewidths=0,
                        )
        fig.supxlabel("I/O size (KiB)" if kind == "size" else "Seconds since first request send")
        fig.supylabel(
            {
                "timeline": "Issued MiB/s (1 s bins)",
                "size": "Cumulative share of operations (%)",
                "lba": "Starting address (MiB)",
            }[kind]
        )
        handles = [
            Line2D([], [], color=v, label=k) for k, v in zip(("Read", "Write"), COLORS.values())
        ]
        if kind != "size":
            handles.extend(
                [
                    Patch(color=".91", label="Between turns"),
                    Patch(color="#fae3be", label="Cold preparation"),
                    Line2D([], [], color=".4", ls=":", label="Second request send"),
                ]
            )
        fig.legend(handles=handles, loc="upper center", ncol=len(handles), frameon=False)
        fig.tight_layout(rect=(0.025, 0.025, 1, 0.94))
        fig.savefig(folder / filename, dpi=155)
        plt.close(fig)


if __name__ == "__main__":
    main()
