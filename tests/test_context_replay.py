import hashlib
import json
import os
from types import SimpleNamespace

import pytest

from kv_ssd_sim.core.paths import PROJECT_ROOT as ROOT
from kv_ssd_sim.experiments.runner import build_parser, run
from kv_ssd_sim.simulation.clock import Clock
from kv_ssd_sim.simulation.storage.config import StoreConfig
from kv_ssd_sim.simulation.storage.device import MIB
from kv_ssd_sim.simulation.storage.transfer import TransferPorts
from kv_ssd_sim.simulation.timing import ContextTiming


def read(path):
    return json.loads(path.read_bytes())


def clock_model(context_scale=1):
    profile = read(ROOT / "configs/compute/b300-tp8-context.json")
    plan = read(ROOT / "configs/kimi-k3-b300-step2/resolved.json")
    model = read(ROOT / "references/kimi-k3-b300/model-config.json")["text_config"]
    return ContextTiming.from_profile(profile, plan, model, context_scale=context_scale)


def test_paired_short_and_long_anchors_and_unmeasured_tail():
    model = clock_model()
    assert model.step_ms([]) == 0
    assert model.step_ms([(0, 1024)]) == pytest.approx(241)
    assert model.step_ms([(1536, 0)] * 32) == pytest.approx(24.12)
    assert model.step_ms([(8512, 0)] * 16) == pytest.approx(22.3)
    assert model.step_ms([(8512, 0)] * 12) == pytest.approx((18.1 + 22.3) / 2)
    assert model.step_ms([(8512, 0)]) == pytest.approx(9.45 + 6976 * 13824 / 7.75e9)
    assert model.step_ms([(1000000, 0)]) > model.step_ms([(8512, 0)])
    assert clock_model(2).step_ms([(8512, 0)] * 8) == pytest.approx(18.1)
    assert clock_model(2).step_ms([(1000000, 0)]) > model.step_ms([(1000000, 0)])
    with pytest.raises(ValueError, match="batch exceeds"):
        model.step_ms([(100, 0)] * 33)


def test_variable_length_mixed_batches_use_each_sequence_history():
    model = clock_model()
    short = model.step_ms([(0, 256), (20000, 0)])
    assert model.step_ms([(1000000, 256), (20000, 0)]) > short
    assert model.step_ms([(0, 1), (1000000, 0)]) > model.step_ms([(0, 1), (20000, 0)])
    assert model.step_ms([(0, 512)]) + model.step_ms([(512, 512)]) == pytest.approx(241)
    lo, hi = 1536, 8512
    assert model.step_ms([(lo, 0)] * 4 + [(hi, 0)] * 4) == pytest.approx((14.57 + 18.1) / 2)


def test_d2h_and_h2d_wait_for_the_same_host_lane():
    clock = Clock()
    cfg = StoreConfig(1, MIB, MIB, MIB, put_mib_s=1, memory_mib_s=1, h2d_mib_s=1, host_copy_lanes=1)
    events = []
    ports = TransferPorts(
        clock, cfg, lambda op, *a, **kw: events.append((op, clock.now)), lambda r: None
    )
    assert ports.reserve_copy(MIB, 1) == (0, 10**9)  # D2H from a timed PUT
    obj = SimpleNamespace(owner=0, size=MIB)
    ports.submit(SimpleNamespace(obj=obj, origin={"rank": 1}, tier="memory", done=None))
    assert ports.reserve_copy(0, 1) == (0, 0)
    assert ports.reserve_copy(MIB, 1) == (2 * 10**9, 3 * 10**9)
    clock.advance(3 * 10**9)
    assert events == [("h2d_ready", 0), ("h2d_start", 10**9), ("h2d_complete", 2 * 10**9)]
    assert not ports.busy()


def compiled_input(path, count, prompt, output):
    rows = []
    for i in range(count):
        tokens = [200 + i] * prompt
        digest = hashlib.sha256(b"".join(t.to_bytes(4, "big") for t in tokens)).hexdigest()
        rows.append(
            {
                "id": f"r{i}",
                "session": f"s{i}",
                "source_index": [0, 0],
                "tokens": tokens,
                "prompt_sha256": digest,
                "output_tokens": output,
                "dependencies": [],
                "root_arrival_ms": 0,
            }
        )
    path.write_text(json.dumps({"schema": 1, "trace_id": "context-probe", "requests": rows}))


@pytest.mark.integration
@pytest.mark.skipif(
    os.getenv("KV_SSD_RUN_NATIVE_TESTS") != "1", reason="requires native vLLM and Mooncake"
)
@pytest.mark.parametrize(
    "shape, flags",
    [
        ((1, 1520, 160), ["--no-save-decode-cache"]),
        ((1, 1520, 160), ["--save-decode-cache"]),
        ((32, 32, 8), ["--max-num-seqs", "32", "--max-num-batched-tokens", "2048"]),
    ],
)
def test_native_decode_boundary_and_batch_limit(tmp_path, shape, flags):
    path, out = tmp_path / "input.json", tmp_path / "result"
    compiled_input(path, *shape)
    run(
        build_parser().parse_args(
            [str(path), "--output", str(out), "--profile", "reference-context", *flags]
        )
    )
    scheduler = read(out / "launch.json")["scheduler"]
    enabled = "--save-decode-cache" in flags
    assert scheduler["save_decode_cache"] == enabled
    assert scheduler["prefix_cache_retention_input"] == 0
    assert scheduler["prefix_cache_retention_interval"] == (1536 if enabled else 0)
    events = [json.loads(l) for l in (out / "events.jsonl").read_text().splitlines()]
    steps = [e for e in events if e["event"] == "compute_step"]
    assert max(e["batch"] for e in steps) == shape[0]
    if shape[0] == 1:
        # 1,536 lies in generated output. The native hybrid save only covers the
        # complete checkpoint, including one shared MLA key and all 24 KDA shards.
        writes = [e for e in events if e["event"] == "put_complete"]
        boundary = [e for e in writes if e["boundary_tokens"] == 1536]
        assert [sum(e["group"] == g for e in boundary) for g in range(4)] == (
            [8, 8, 8, 1] if enabled else [0, 0, 0, 0]
        )
