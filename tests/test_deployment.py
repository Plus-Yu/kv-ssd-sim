import json

import pytest

from kv_ssd_sim.core.paths import PROJECT_ROOT
from kv_ssd_sim.models.deployment import capacity, convert

CONFIG = PROJECT_ROOT / "configs/kimi-k3-b300-step2"


def load(name):
    return json.loads((CONFIG / f"{name}.json").read_text())


@pytest.mark.parametrize(
    "file,key,value,error",
    [
        ("server", "checkpoint_tensor_bytes", 4 * 10**12, "GPU KV"),
        ("mooncake", "global_segment_size_bytes_per_client", 500 * 10**9, "host budget"),
        ("ssd", "offload_quota_bytes_per_client", 500 * 10**9, "selected SSD"),
        ("ssd", "offload_path_template", "/mnt/collides", "distinct"),
    ],
)
def test_reject_impossible_capacity(file, key, value, error):
    reference = json.loads((PROJECT_ROOT / "configs/kimi-k3-b300-reference.json").read_text())
    values = {name: load(name) for name in ("server", "mooncake", "ssd")}
    values[file][key] = value
    with pytest.raises(ValueError, match=error):
        capacity(reference, values["server"], values["mooncake"], values["ssd"])


def test_pinned_native_plan():
    plan = convert(CONFIG)
    assert json.loads(json.dumps(plan)) == load("resolved")
    geometry = plan["geometry"]
    assert geometry["manager_block_tokens"] == 1536
    assert geometry["hash_tokens"] == geometry["hit_alignment_tokens"] == 128
    assert geometry["kda_component_bytes"] == [27648, 786432]
    assert geometry["mla_bytes_per_token"] == 576
    assert len([entry for entry in plan["layers"] if entry["padding_bytes"] == 70656]) == 69
    assert sum(entry["kind"] == "mla" for entry in plan["layers"]) == 24
    for group in plan["groups"]:
        shards = {rank["store_shard_by_group"][group["group"]] for rank in plan["ranks"]}
        assert (
            len(shards)
            == group["store_shards"]
            == (1 if group["spec"] == "MLAAttentionSpec" else 8)
        )
    assert [rank["kda_heads_half_open"] for rank in plan["ranks"]] == [
        [start, start + 12] for start in range(0, 96, 12)
    ]
    assert plan["checkpoint_probes"][0]["internal_checkpoint_position"] is None
    assert plan["checkpoint_probes"][1]["internal_checkpoint_position"] == 1664


def test_storage_budget_independent_of_gpu_weights():
    ref = json.loads((PROJECT_ROOT / "configs/kimi-k3-b300-reference.json").read_text())
    server, store, disk = [load(name) for name in ("server", "mooncake", "ssd")]
    first = capacity(ref, server, store, disk)
    server["gpu_reserves_bytes_per_rank"]["activations_and_workspace"] += 1024**3
    second = capacity(ref, server, store, disk)
    assert (
        first["gpu_per_rank"]["kv_pool_budget"] - second["gpu_per_rank"]["kv_pool_budget"]
        == 1024**3
    )
    assert first["host_node"] == second["host_node"] and first["ssd_node"] == second["ssd_node"]
