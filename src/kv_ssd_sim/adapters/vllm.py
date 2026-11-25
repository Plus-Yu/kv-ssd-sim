"""Native key and shard-layout planning shared by both replay paths."""


def metadata_rank_worker(groups, unit, rank, model_name, prefix):
    """Construct the pinned worker’s native key/layout planner without Store services."""
    from vllm.distributed.kv_transfer.kv_connector.v1.mooncake_store import worker as upstream
    from vllm.distributed.kv_transfer.kv_connector.v1.mooncake_store.layout import (
        HeadMajorStoreLayout,
    )

    worker = upstream.MooncakeStoreWorker.__new__(upstream.MooncakeStoreWorker)
    (
        worker.tp_rank,
        worker.tp_size,
        worker.pp_rank,
        worker.pp_size,
        worker.dcp_size,
        worker.pcp_size,
    ) = rank, 8, 0, 1, 1, 1
    worker._kv_cache_groups, worker.hash_block_size, worker.requested_store_chunk_size = (
        groups,
        unit,
        unit,
    )
    worker.store_tp_size = 8
    worker._group_tp_replication_factors = worker._compute_group_tp_replication_factors()
    plans = tuple(
        worker._make_store_group_plan(group, 8, HeadMajorStoreLayout, factor)
        for group, factor in zip(groups, worker._group_tp_replication_factors, strict=True)
    )
    worker.token_dbs = worker._build_token_databases(
        upstream.KeyMetadata(model_name, rank, 0, 0, 0, cache_prefix=prefix), plans
    )
    worker._init_lookup_key_prefixes()
    return worker
