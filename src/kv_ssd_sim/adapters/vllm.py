"""Native key and shard-layout planning shared by both replay paths."""

from ..core.paths import PROJECT_ROOT

VLLM_VERSION = "0.34.0rc1.dev302+gbe47189f3.cpu"


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


def scheduler_config(max_model_len, block_size, hybrid_config=None, *, k3=False):
    """Common scheduler, prefix and connector settings for payload and metadata workers."""
    from vllm.config import (
        CacheConfig,
        DeviceConfig,
        KVTransferConfig,
        ModelConfig,
        ParallelConfig,
        SchedulerConfig,
        VllmConfig,
    )

    hybrid = k3 or hybrid_config is not None
    model_path = PROJECT_ROOT / "configs/model"
    return VllmConfig(
        model_config=ModelConfig(
            model=str(model_path),
            skip_tokenizer_init=True,
            dtype="float16",
            max_model_len=max_model_len,
            hf_overrides={"max_position_embeddings": max_model_len},
            seed=0,
            enforce_eager=True,
        ),
        device_config=DeviceConfig(device="cpu"),
        scheduler_config=SchedulerConfig(
            is_encoder_decoder=False,
            max_num_seqs=8,
            max_num_batched_tokens=4096 if k3 else 128,
            max_model_len=max_model_len,
            enable_chunked_prefill=True,
            async_scheduling=False,
            watermark=0.0,
        ),
        cache_config=CacheConfig(
            block_size=block_size,
            enable_prefix_caching=True,
            cache_dtype="auto",
            gpu_memory_utilization=0.5,
            mamba_cache_mode="align" if hybrid else "none",
        ),
        parallel_config=ParallelConfig(tensor_parallel_size=1, pipeline_parallel_size=1),
        kv_transfer_config=KVTransferConfig(
            kv_connector="MooncakeStoreConnector",
            kv_role="kv_both",
            kv_connector_extra_config={
                "cache_prefix": (
                    "k3-online-tp8"
                    if k3
                    else "kv-ssd-sim-hybrid-" + hybrid_config.fingerprint
                    if hybrid_config
                    else "kv-ssd-sim-v1"
                ),
                **({"store_tp_size": 8 if k3 else 1} if hybrid else {}),
                "load_async": True,
                "lookup_async": False,
                "save_decode_cache": False,
            },
        ),
    )
