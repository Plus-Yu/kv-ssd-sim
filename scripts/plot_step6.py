#!/usr/bin/env python3
"""Plot the observed per-pair latency changes without aggregating away the three trials."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("summary", type=Path)
parser.add_argument("output", type=Path)
args = parser.parse_args()
stats = json.loads(args.summary.read_text())
fig, panels = plt.subplots(1, 3, figsize=(8.8, 3.25), layout="constrained")
colors = ["#345A86", "#B55931", "#507A58"]
for ax, metric, title in zip(
    panels,
    ("get_ms", "ttft_ms", "aiperf_ttft_ms"),
    ("Mooncake GET", "Native TTFT", "Client TTFT"),
    strict=True,
):
    for i, color in enumerate(colors, 1):
        pair = sorted(
            [r for r in stats["trials"] if r["trial"].endswith(f"{i:02d}")],
            key=lambda r: r["nand_read_ns"],
        )
        ax.plot(
            [0, 1],
            [r["restore"][metric] / 1000 for r in pair],
            "o-",
            color=color,
            label=f"Pair {i}",
            lw=1.4,
            alpha=0.85,
        )
    ax.set(
        title=title,
        xticks=[0, 1],
        xticklabels=["40 µs", "2 ms"],
        xlim=(-0.22, 1.22),
        ylim=(0, None),
    )
    ax.set_xlabel("NAND page read setting")
    ax.set_ylabel("Second request (s)")
    ax.grid(axis="y", alpha=0.2)
    ax.spines[["top", "right"]].set_visible(False)
panels[0].legend(frameon=False, fontsize=8)
fig.savefig(args.output, dpi=160)
