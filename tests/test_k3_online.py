from importlib import import_module
from pathlib import Path

import pytest

from kv_ssd_sim.k3_online import RankConnectors


def test_rank_completions_accumulate_across_steps():
    from types import SimpleNamespace
    from unittest.mock import Mock

    connectors = [Mock(), Mock()]

    def result(ids):
        return SimpleNamespace(
            finished_sending=set(), finished_recving=set(ids), failed_recving=set()
        )

    connectors[0].get_transfer_results.side_effect = [result(["a"]), result([])]
    connectors[1].get_transfer_results.side_effect = [result([]), result(["a"])]
    fanout = RankConnectors(connectors)
    assert not fanout.get_transfer_results(set()).finished_recving
    assert fanout.get_transfer_results(set()).finished_recving == {"a"}


def test_shard_ledger_rejects_partial_checkpoint(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    analyze = import_module("analyze_k3_ssd")

    def batch(group, rank):
        mla = group == 3
        size = 32 if mla else 64
        fmt = "attention_head_major" if mla else "mamba_state"
        key = (
            f"p@store_tp:8@store_format:{fmt}@tp_rank:{rank}@pcp0@dcp0@pp_rank:0@group:{group}@abcd"
        )
        return [
            {"event": "store_put_begin", "keys": [key], "bytes": size},
            {
                "event": "store_put_end",
                "keys": [key],
                "logical_rank": rank,
                "object_bytes": [size],
                "result": [0],
                "digests": ["f" * 64],
            },
        ]

    plan = {
        "groups": [
            {
                "group": g,
                "spec": "MLAAttentionSpec" if g == 3 else "MambaSpec",
                "logical_bytes_per_object_per_shard": 32 if g == 3 else 64,
            }
            for g in range(4)
        ]
    }
    records = batch(3, 0)
    for g in range(3):
        for r in range(8):
            records.extend(batch(g, r))
    assert analyze.transfer_ledger(records, plan)["unique_keys"] == 25
    with pytest.raises(ValueError, match="Incomplete KDA snapshot"):
        analyze.transfer_ledger(records[:-2], plan)
