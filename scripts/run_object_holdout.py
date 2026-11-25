"""Freeze a single-client object prediction and run its input against the native HTTP/Store path."""

import argparse
import asyncio
import hashlib
import json
import os
import signal
import sys
import time
from pathlib import Path

import aiohttp

from kv_ssd_sim.runtime import PROJECT_ROOT as ROOT
from kv_ssd_sim.runtime import free_port


def checksum(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


async def run(args):
    from transformers import AutoTokenizer

    prediction = args.prediction
    launch = json.loads((prediction / "launch.json").read_bytes())
    inputs = [json.loads(p.read_bytes()) for p in args.inputs]
    if [checksum(p) for p in args.inputs] != [r["sha256"] for r in launch["compiled_inputs"]]:
        raise ValueError("Holdout input differs from the prediction")
    if (
        launch["store"]["clients"] != 1
        or json.loads((prediction / "summary.json").read_bytes())["status"] != "complete"
        or os.environ.get("PYTHONHASHSEED") != launch["pythonhashseed"]
    ):
        raise ValueError("Require a completed one-client prediction with the same hash seed")
    tokenizer = AutoTokenizer.from_pretrained(ROOT / "configs/sim-tokenizer", local_files_only=True)
    args.output.mkdir(parents=True, exist_ok=False)
    frozen = {
        name: checksum(prediction / name)
        for name in ("launch.json", "requests.json", "events.jsonl", "objects.json", "summary.json")
    }
    for name in ("ssd-demand.jsonl.gz", "ssd-10s.csv", "ssd-summary.json"):
        if (prediction / name).exists():
            frozen[name] = checksum(prediction / name)
    (args.output / "frozen-prediction.json").write_text(
        json.dumps({"created_unix_ns": time.time_ns(), "sha256": frozen}, indent=2) + "\n"
    )
    config = {"profile": "k3-tp8"}
    for key in ("segment", "staging", "quota"):
        mib, remainder = divmod(launch["store"][key + "_bytes"], 1 << 20)
        if remainder:
            raise ValueError("Native server requires whole MiB budgets")
        config[key + "_mib"] = mib
    profile = args.output / "store.json"
    profile.write_text(json.dumps(config, indent=2) + "\n")
    lease = launch["store"]["lease_ms"]
    if int(lease) != lease:
        raise ValueError("Native master requires integer milliseconds")
    if (
        launch["store"].get("high_watermark", 0.90),
        launch["store"].get("eviction_ratio", 0.05),
    ) != (0.90, 0.05):
        raise ValueError(
            "The native holdout uses the pinned master's default 0.90/0.05 eviction settings"
        )
    port = free_port()
    url = f"http://127.0.0.1:{port}"
    command = [
        sys.executable,
        "-m",
        "kv_ssd_sim",
        "serve",
        "--output",
        str(args.output / "server"),
        "--port",
        str(port),
        "--master-lease-ms",
        str(int(lease)),
        "--num-blocks",
        str(launch["num_blocks"]),
        "--max-model-len",
        str(launch["max_model_len"]),
        "--prefill-ms",
        str(args.native_prefill_ms if args.native_prefill_ms is not None else launch["prefill_ms"]),
        "--decode-ms",
        str(args.native_decode_ms if args.native_decode_ms is not None else launch["decode_ms"]),
        "--hybrid-config",
        str(profile),
    ]
    rows = {r["id"]: r for trace in inputs for r in trace["requests"]}
    env = os.environ.copy()
    observed = None
    if args.observer:
        observer = args.observer.resolve()
        observed = json.loads(observer.with_suffix(".json").read_bytes())
        wheel = Path(launch["native_master"]["wheel_dso"])
        if (
            checksum(observer) != observed["binary_sha256"]
            or checksum(wheel) != observed["wheel_dso_sha256"]
            or checksum(ROOT / "native/object_replay_observer.cpp") != observed["source_sha256"]
        ):
            raise ValueError("Client observer differs from its pinned build")
        ledger = (args.output / "store-boundaries.tsv").resolve()
        ledger.touch(exist_ok=False)
        env.update(
            LD_PRELOAD=":".join(filter(None, (str(observer), env.get("LD_PRELOAD")))),
            KV_SSD_OBSERVED_DSO=str(wheel),
            KV_SSD_OBSERVER_LOG=str(ledger),
            KV_SSD_MEASUREMENT_IDS=json.dumps(
                [rid for rid, row in rows.items() if row.get("measurement_anchor")]
            ),
        )
    (args.output / "native-launch.json").write_text(
        json.dumps({"command": command, "observer": observed}, indent=2) + "\n"
    )
    markers = {(rid, k): asyncio.Event() for rid in rows for k in ("start", "finish")}
    stamps = {}
    timing = []
    with (args.output / "server.log").open("w") as log:
        process = await asyncio.create_subprocess_exec(
            *command, cwd=ROOT, stdout=log, stderr=asyncio.subprocess.STDOUT, env=env
        )
        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=600)) as client:
                deadline = time.monotonic() + 120
                while time.monotonic() < deadline:
                    if process.returncode is not None:
                        raise RuntimeError(
                            "Native server exited during startup; inspect server.log"
                        )
                    try:
                        async with client.get(url + "/health") as response:
                            if response.status == 200:
                                break
                    except aiohttp.ClientConnectionError:
                        pass
                    await asyncio.sleep(0.2)
                else:
                    raise TimeoutError("Native server startup timed out")
                zero = time.monotonic()

                async def issue(row, offset):
                    rid = row["id"]
                    release = zero + offset + row["root_arrival_ms"] / 1000
                    for parent, marker, ms in row["dependencies"]:
                        await markers[parent, marker].wait()
                        release = max(release, stamps[parent, marker] + ms / 1000)
                    await asyncio.sleep(max(0, release - time.monotonic()))
                    content = tokenizer.decode(row["tokens"], clean_up_tokenization_spaces=False)
                    messages = [{"role": "user", "content": content}]
                    if (
                        tokenizer.apply_chat_template(
                            messages, tokenize=True, add_generation_prompt=True, return_dict=False
                        )
                        != row["tokens"]
                    ):
                        raise ValueError(f"Wire token round trip failed: {rid}")
                    stamps[rid, "start"] = time.monotonic()
                    markers[rid, "start"].set()
                    async with client.post(
                        url + "/v1/chat/completions",
                        headers={"X-Request-ID": rid},
                        json={
                            "model": "kv-ssd-sim",
                            "messages": messages,
                            "max_tokens": row["output_tokens"],
                        },
                    ) as resp:
                        body = await resp.json()
                        if (
                            resp.status != 200
                            or body.get("usage", {}).get("completion_tokens")
                            != row["output_tokens"]
                        ):
                            raise RuntimeError(
                                f"Native HTTP failed: {rid}, {resp.status}, {str(body)[:400]}"
                            )
                    stamps[rid, "finish"] = time.monotonic()
                    markers[rid, "finish"].set()
                    timing.append(
                        {
                            "id": rid,
                            "send_s": stamps[rid, "start"] - zero,
                            "finish_s": stamps[rid, "finish"] - zero,
                        }
                    )

                async with asyncio.TaskGroup() as group:
                    for trace, offset in zip(inputs, launch["root_offsets_s"], strict=True):
                        for row in trace["requests"]:
                            group.create_task(issue(row, offset))
                (args.output / "client.json").write_text(json.dumps(timing, indent=2) + "\n")
        finally:
            if process.returncode is None:
                process.send_signal(signal.SIGINT)
                try:
                    await asyncio.wait_for(process.wait(), 100)
                except TimeoutError:
                    process.kill()
                    await process.wait()
    summary = json.loads((args.output / "server/summary.json").read_bytes())
    if (
        process.returncode
        or summary["status"] != "passed"
        or summary["requests"] != len(rows)
        or summary["checksum_errors"]
    ):
        raise RuntimeError("Native trial failed; inspect server.log and server/summary.json")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("inputs", type=Path, nargs="+")
    p.add_argument("--prediction", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--observer", type=Path, help="Pinned passive Mooncake client observer")
    p.add_argument("--native-prefill-ms", type=float)
    p.add_argument("--native-decode-ms", type=float)
    asyncio.run(run(p.parse_args()))
