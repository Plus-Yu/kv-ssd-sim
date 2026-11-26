"""Verify two natural FEMU trials and export B2 device timelines."""

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from kv_ssd_sim.analysis.block_capture import check_receipt
from kv_ssd_sim.analysis.block_io import NS, check, measurements, parse_trace, stats
from kv_ssd_sim.analysis.femu import verify


def read(path):
    return json.loads(path.read_text())


def matched_settings(path):
    launch = read(path / "launch.json")
    argv = launch["command"]
    manifest = read(path / "joint/server/run_manifest.json")
    return {
        "inputs": launch["guest_inputs_sha256"],
        "runtime_hashes": {
            k.replace(f"outputs/{path.name}/", "trial/"): v
            for k, v in launch["inputs_sha256"].items()
        },
        "server": {
            k: manifest[k]
            for k in (
                "harness_source_sha256",
                "versions",
                "tokenizer_sha256",
                "prefill_ms",
                "decode_ms",
                "read_delay_ms",
                "max_model_len",
                "num_blocks",
                "max_inflight",
                "hybrid_config",
                "cold_between_requests",
            )
        },
        "femu_revision": launch["femu_revision"],
        "device": argv[argv.index("-device") + 1],
        **{k: argv[argv.index(k) + 1] for k in ("-machine", "-cpu", "-m", "-smp")},
    }


def inspect(path, mode):
    natural = verify(path)["natural"]
    capture = natural["capture"]
    receipt = read(path / "capture.json")
    entries = check_receipt(receipt, receipt["device"], mode)
    blocks, _ = parse_trace(
        (path / "block-trace.txt").read_text(), receipt["device"], stream_entries=entries
    )
    rows = natural["requests"]
    begin = rows[0]["http_start_ns"]
    end = capture["device_end_ns"]
    timeline = measurements(blocks, begin, end, NS)
    result = {
        **natural,
        "trial": path.name,
        "session_s": (rows[-1]["response_end_ns"] - begin) / NS,
        "request_s": [(row["response_end_ns"] - row["http_start_ns"]) / NS for row in rows],
        "observed_s": (end - begin) / NS,
        "io_first_send_to_device_end": stats(blocks, begin, end),
        "max_qd": max(b["qd_peak"] for b in timeline),
    }
    return result, timeline, [b for b in blocks if begin <= b.time < end and b.stage == "issue"]


def plot(path, runs, bins):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(
        2, 2, sharex="col", sharey="row", figsize=(11, 5.9), layout="constrained"
    )
    for col, (mode, rows) in enumerate(bins.items()):
        top, bottom = axes[:, col]
        run = runs[mode]
        marks = run["requests"]
        begin = marks[0]["http_start_ns"]
        xs = [b["start_s"] for b in rows] + [run["observed_s"]]
        for field, color, label in (
            ("write_MiB_s", "#205c91", "Write"),
            ("read_MiB_s", "#ba540c", "Read"),
        ):
            top.stairs([b[field] for b in rows], xs, color=color, label=label)
        bottom.stairs([b["qd_peak"] for b in rows], xs, color="#25594f")
        for ax in (top, bottom):
            for i, r in enumerate(marks):
                ax.axvspan(
                    (r["http_start_ns"] - begin) / NS,
                    (r["response_end_ns"] - begin) / NS,
                    color="slateblue",
                    alpha=0.10,
                    label="HTTP active" if i == 0 else None,
                )
            ax.axvline(
                (run["capture"]["sender_drained_ns"] - begin) / NS,
                ls=":",
                color="#333333",
                label="Sender drained",
            )
            ax.set_xlim(0, xs[-1])
            ax.set_ylim(bottom=0)
        top.set_title(f"{mode.capitalize()} capture")
        bottom.set_xlabel("Guest seconds from first HTTP send")
    axes[0, 0].set_ylabel("Issued MiB/s (1 s bins)")
    axes[1, 0].set_ylabel("Peak in-flight data I/O")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncols=4)
    fig.savefig(path / "timeline.png", dpi=160)
    plt.close(fig)


def plot_addresses(path, run, issues):
    import matplotlib.pyplot as plt

    fig, (address, sizes) = plt.subplots(
        1, 2, figsize=(11, 3.8), layout="constrained", gridspec_kw={"width_ratios": [2, 1]}
    )
    begin = run["requests"][0]["http_start_ns"]
    for op, name, color in (("R", "Read", "#ba540c"), ("W", "Write", "#205c91")):
        group = [b for b in issues if b.op == op]
        if group:
            address.scatter(
                [(b.time - begin) / NS for b in group],
                [b.sector * 512 / 2**20 for b in group],
                color=color,
                s=15,
                alpha=0.75,
                label=name,
            )
        hist = run["io_first_send_to_device_end"][op]["size_histogram"]
        if hist:
            sizes.scatter([n / 1024 for n in hist], list(hist.values()), color=color, label=name)
    response = (run["capture"]["last_response_ns"] - begin) / NS
    address.axvline(response, color="#444444", linestyle="--", label="Last HTTP response")
    address.set(
        xlim=(0, run["observed_s"]),
        xlabel="Guest seconds from first HTTP send",
        ylabel="Start address (MiB from LBA 0)",
        title="Data I/O issued to FEMU",
    )
    sizes.set(
        xscale="log", xlabel="I/O size (KiB, log scale)", ylabel="Count", title="Issued I/O sizes"
    )
    address.legend()
    fig.savefig(path / "addresses-sizes.png", dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stream", type=Path)
    parser.add_argument("buffer", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    settings = matched_settings(args.stream)
    check(settings == matched_settings(args.buffer), "Runtime, device or workload settings differ")
    runs, bins, issues = {}, {}, {}
    for mode in ("stream", "buffer"):
        runs[mode], bins[mode], issues[mode] = inspect(getattr(args, mode), mode)
    for a, b in zip(runs["stream"]["requests"], runs["buffer"]["requests"], strict=True):
        for field in (
            "prompt_tokens",
            "output_tokens",
            "wire_prompt_sha256",
            "local_reuse_tokens",
            "computed_prompt_tokens",
            "external_load_tokens",
        ):
            check(a[field] == b[field], f"Different replay treatment: {field}")
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "summary.json").write_text(
        json.dumps({"settings": settings, "runs": runs}, indent=2) + "\n"
    )
    with (args.output / "timeline.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["mode", *bins["stream"][0]])
        writer.writeheader()
        for mode, series in bins.items():
            writer.writerows({"mode": mode, **row} for row in series)
    plot(args.output, runs, bins)
    plot_addresses(args.output, runs["stream"], issues["stream"])


if __name__ == "__main__":
    main()
