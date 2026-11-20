#!/usr/bin/env python3
"""Offline audit of the installed AIPerf WEKA loader; never sends requests."""

import argparse
import hashlib
import json
import os
from importlib.metadata import version
from pathlib import Path


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    source = root / "datasets/kimi-k3-b300-step1/slice.json"
    tokenizer_dir = root / "configs/sim-tokenizer"
    # Installed AIPerf mishandles absolute local paths with HF_HUB_OFFLINE.
    # Both calls below use the existing absolute tokenizer directory.
    os.environ.pop("HF_HUB_OFFLINE", None)
    os.environ.pop("TRANSFORMERS_OFFLINE", None)
    from aiperf.common import random_generator as rng
    from aiperf.common.tokenizer import Tokenizer
    from aiperf.config.config import BenchmarkConfig
    from aiperf.config.resolution.plan import BenchmarkRun
    from aiperf.dataset.generator.prompt import PromptGenerator
    from aiperf.dataset.loader.weka_trace import WekaTraceLoader
    from aiperf.dataset.loader.weka_trace_models import WekaTrace
    from transformers import AutoTokenizer

    if version("aiperf") != "0.13.0":
        raise ValueError("This audit is pinned to AIPerf 0.13.0")
    raw = json.loads(source.read_text())
    WekaTrace.model_validate(raw)
    WekaTrace.model_validate(
        json.loads((root / "references/kimi-k3-b300/trace_0001.source.json").read_text())
    )
    rng.init(42)
    cfg = BenchmarkConfig(
        models=["kv-ssd-sim"],
        endpoint={"urls": ["http://127.0.0.1:8000"]},
        datasets=[
            {
                "name": "main",
                "type": "file",
                "path": str(source),
                "format": "weka_trace",
                "max_context_length": 131072,
                "use_think_time_only": True,
            }
        ],
        phases=[{"name": "profiling", "type": "concurrency", "requests": 2, "concurrency": 1}],
        tokenizer={"name": str(tokenizer_dir)},
    )
    run = BenchmarkRun(
        benchmark_id="offline-input-audit",
        cfg=cfg,
        artifact_dir=args.output.parent,
        random_seed=42,
    )
    hf = AutoTokenizer.from_pretrained(tokenizer_dir, local_files_only=True)
    tokenizer = Tokenizer.from_pretrained(str(tokenizer_dir))
    generator = PromptGenerator(prompts=None, prefix_prompts=None, tokenizer=tokenizer)
    loader = WekaTraceLoader(filename=str(source), run=run, prompt_generator=generator)
    conversations = loader.convert_to_conversations(loader.load_dataset())
    if len(conversations) != 1 or len(conversations[0].turns) != 2:
        raise ValueError("AIPerf changed the selected conversation/request coverage")
    rows, token_lists = [], []
    for index, turn in enumerate(conversations[0].turns):
        original = raw["requests"][index]
        if turn.source_outer_idx != index or turn.no_request:
            raise ValueError("Source request mapping changed")
        # Each selected turn has a full reconstructed prompt; turn 2 resets context.
        if index and not turn.reset_context:
            raise ValueError("Expected a self-contained reconstructed second prompt")
        encoded = hf.apply_chat_template(
            turn.raw_messages,
            tokenize=True,
            add_generation_prompt=True,
        )
        tokens = encoded["input_ids"] if hasattr(encoded, "keys") else encoded
        delay = turn.delay or 0
        if (len(tokens), turn.max_tokens, delay) != (
            original["in"],
            original["out"],
            original["think_time"] * 1000,
        ):
            raise ValueError("Input length, output budget or think time drifted")
        token_lists.append(tokens)
        rows.append(
            {
                "source_path": f"/requests/{index}",
                "input_tokens": len(tokens),
                "output_budget": turn.max_tokens,
                "delay_ms": delay,
                "reset_context": turn.reset_context,
                "token_ids_sha256": hashlib.sha256(
                    json.dumps(tokens, separators=(",", ":")).encode()
                ).hexdigest(),
            }
        )
    lcp = next(
        (i for i, (a, b) in enumerate(zip(*token_lists)) if a != b), min(map(len, token_lists))
    )
    expected = json.loads((source.parent / "slice-audit.json").read_text())["consecutive_prefixes"][
        0
    ]["shared_complete_tokens"]
    if lcp != expected:
        raise ValueError(f"Shared prefix drifted: {lcp} != {expected}")
    result = {
        "status": "passed",
        "scope": "offline loader + chat template; no HTTP/runtime/FEMU",
        "aiperf_version": version("aiperf"),
        "transformers_version": version("transformers"),
        "random_seed": 42,
        "source_sha256": digest(source),
        "tokenizer_files": {
            p.name: digest(p) for p in sorted(tokenizer_dir.iterdir()) if p.is_file()
        },
        "context_mode": str(conversations[0].context_mode),
        "conversation_count": 1,
        "request_count": len(rows),
        "requests": rows,
        "shared_prefix_tokens": lcp,
        "match_unit": 128,
        "aligned_shared_prefix_tokens": lcp // 128 * 128,
        "timing": "causal think time; timestamps retained as metadata, not fixed arrivals",
        "synthesis_caps": {"max_isl": None, "max_osl": None},
        "token_semantics": "synthetic local tokenizer, not K3 tokenizer or original Claude text",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"status": "passed", "requests": len(rows), "shared_prefix_tokens": lcp}))


if __name__ == "__main__":
    main()
