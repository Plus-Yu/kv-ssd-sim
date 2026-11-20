"""Convert the pinned K3 reference into metadata; never allocate a target pool."""

from __future__ import annotations

import hashlib
import json
import math
from decimal import Decimal
from importlib.metadata import version
from pathlib import Path
from types import SimpleNamespace

from .native import VLLM_VERSION
from .runtime import PROJECT_ROOT


def require(ok, message):
    if not ok:
        raise ValueError(message)


def byte_count(value, label, minimum=0):
    require(type(value) is int and value >= minimum, f"Invalid byte count: {label}")
    return value


def ceil_div(value, unit):
    return -(-value // unit)


def capacity(reference, server, store, disk):
    hardware = reference["reference_deployment"]
    ranks = reference["parallelism"]["tensor_parallel_size"]
    require(
        ranks == hardware["gpu_count"] == store["client_count"] == store["store_tp_size"],
        "One GPU and embedded Store client required per rank",
    )
    gpu = hardware["gpu_nominal_memory_GB_each"] * 10**9
    utilization = Decimal(str(reference["scheduler"]["gpu_memory_utilization"]))
    require(0 < utilization <= 1, "GPU utilization must be in (0, 1]")
    usable = int(gpu * utilization)
    weights = ceil_div(byte_count(server["checkpoint_tensor_bytes"], "checkpoint", 1), ranks)
    reserves = server["gpu_reserves_bytes_per_rank"]
    require(
        set(reserves)
        == {
            "weight_replication_imbalance_and_format",
            "activations_and_workspace",
            "communication_buffers",
            "cuda_graphs",
            "runtime_misc",
        },
        "Incomplete GPU reserve ledger",
    )
    budget = usable - weights - sum(byte_count(value, key) for key, value in reserves.items())
    require(budget > 0, "No remaining GPU KV budget")
    host = hardware["nominal_host_RAM_TB"] * 10**12
    clients = {
        key: byte_count(store[key], key, 1) * ranks
        for key in (
            "global_segment_size_bytes_per_client",
            "local_buffer_size_bytes_per_client",
            "disk_staging_size_bytes_per_client",
        )
    }
    host_reserve = byte_count(server["host_reserved_non_store_bytes"], "host reserve")
    require(sum(clients.values()) + host_reserve <= host, "Mooncake exceeds host budget")
    require(
        store["offload_enabled"] is True and disk["apply_target_capacity_to_guest"] is False,
        "Target budgets must not resize FEMU",
    )
    require(
        disk["selected_drive_count"] == 1 <= hardware["nominal_cache_nvme_count"],
        "Only a single target SSD is modeled",
    )
    drive = int(Decimal(str(hardware["nominal_cache_nvme_TB_each"])) * 10**12)
    disk_reserve = byte_count(disk["drive_reserved_bytes"], "drive reserve")
    quota = byte_count(disk["offload_quota_bytes_per_client"], "disk quota", 1) * ranks
    require(quota + disk_reserve <= drive, "Per-client quotas exceed selected SSD")
    paths = [disk["offload_path_template"].format(rank=rank) for rank in range(ranks)]
    require(
        len(set(paths)) == ranks and all(Path(path).is_absolute() for path in paths),
        "Each Store client needs a distinct absolute offload path",
    )
    return {
        "gpu_per_rank": {
            "nominal": gpu,
            "utilization_envelope": usable,
            "outside_envelope": gpu - usable,
            "checkpoint_average_rounded_up": weights,
            "reserves": reserves,
            "kv_pool_budget": budget,
        },
        "host_node": {
            "nominal": host,
            "non_store_reserve": host_reserve,
            "store_node_totals_by_per_client_setting": clients,
            "unassigned": host - host_reserve - sum(clients.values()),
        },
        "ssd_node": {
            "selected_capacity": drive,
            "reserved": disk_reserve,
            "quota_total": quota,
            "unassigned": drive - quota - disk_reserve,
            "client_paths": paths,
        },
    }


def convert(config_dir, root=PROJECT_ROOT):
    return native_deployment(config_dir, root)[0]


def native_deployment(config_dir, root=PROJECT_ROOT):
    import torch
    from vllm.config import AttentionConfig, CacheConfig, SchedulerConfig
    from vllm.distributed.kv_transfer.kv_connector.v1.mooncake_store.layout import (
        HeadMajorStoreLayout,
    )
    from vllm.distributed.kv_transfer.kv_connector.v1.mooncake_store.worker import (
        MooncakeStoreWorker,
    )
    from vllm.model_executor.layers.mamba.checkpoint import compute_mamba_prefill_checkpoints
    from vllm.model_executor.layers.mamba.mamba_utils import get_conv_state_layout
    from vllm.models.kimi_k3.nvidia.kda import KimiK3DeltaAttention
    from vllm.models.kimi_k3.nvidia.mla import MultiHeadLatentAttention
    from vllm.transformers_utils.configs.kimi_linear import KimiLinearConfig
    from vllm.v1.attention.backends.mla.tokenspeed_mla import TokenspeedMLABackend
    from vllm.v1.core.kv_cache_planning import get_kv_cache_config_from_groups, get_kv_cache_groups
    from vllm.v1.core.kv_cache_utils import (
        resolve_cache_hit_alignment_tokens,
        resolve_kv_cache_block_sizes,
    )
    from vllm.v1.core.single_type_kv_cache_manager import register_all_kvcache_specs

    inputs = {}

    def read(path, label):
        data = path.read_bytes()
        inputs[label] = hashlib.sha256(data).hexdigest()
        return json.loads(data)

    require(version("vllm") == VLLM_VERSION, "Unvalidated local vLLM version")
    reference = read(root / "configs/kimi-k3-b300-reference.json", "reference")
    model = read(root / reference["model"]["config"], "model")["text_config"]
    require(inputs["model"] == reference["model"]["config_sha256"], "Model digest mismatch")
    server, store, disk = (
        read(config_dir / f"{label}.json", label) for label in ("server", "mooncake", "ssd")
    )
    ledger = capacity(reference, server, store, disk)
    tp = reference["parallelism"]["tensor_parallel_size"]
    require(
        reference["parallelism"]
        == {
            "tensor_parallel_size": 8,
            "pipeline_parallel_size": 1,
            "data_parallel_size": 1,
            "decode_context_parallel_size": 1,
            "expert_parallel": False,
        },
        "Unsupported parallel layout",
    )
    cache = reference["cache"]
    sched = reference["scheduler"]
    match = cache["prefix_match_unit"]
    require(
        server["attention_backend"] == "TOKENSPEED_MLA"
        and server["layout"] == "LBHNC"
        and server["kda_prefill_backend"] == "flashkda"
        and get_conv_state_layout() == server["conv_state_layout"] == "SD",
        "Unsupported backend/cache layout",
    )
    require(
        cache["mla_dtype"] == "fp8"
        and not sched["speculative_decoding"]
        and cache["enable_prefix_caching"],
        "Expected plain FP8 prefix-cached K3 without speculation",
    )
    require(store["store_chunk_size_tokens"] == match, "Store chunk must match the prefix unit")
    k3 = KimiLinearConfig.from_dict(model)
    linear = k3.linear_attn_config
    require(
        k3.mla_use_nope and k3.dsa_config is None and not k3.num_nextn_predict_layers,
        "Unexpected K3 variant",
    )
    require(
        sorted(linear["full_attn_layers"] + linear["kda_layers"])
        == list(range(1, k3.num_hidden_layers + 1)),
        "Incomplete or overlapping layer map",
    )
    heads, dim, kernel = (
        linear[key] for key in ("num_heads", "head_dim", "short_conv_kernel_size")
    )
    require(heads % tp == 0, "KDA heads do not divide TP")
    width = k3.kv_lora_rank + k3.qk_rope_head_dim
    local_heads = heads // tp
    cache_config = CacheConfig(
        block_size=1,
        mamba_block_size=1,
        mamba_cache_mode="align",
        cache_dtype="fp8",
        mamba_cache_dtype=cache["kda_convolution_dtype"],
        mamba_ssm_cache_dtype=cache["kda_recurrent_dtype"],
        prefix_match_unit=match,
        enable_prefix_caching=True,
    )
    cache_config.kv_cache_layout = server["layout"]
    cfg = SimpleNamespace(
        cache_config=cache_config,
        model_config=SimpleNamespace(
            dtype=torch.bfloat16, hf_text_config=k3, max_model_len=sched["max_model_len"]
        ),
        parallel_config=SimpleNamespace(tensor_parallel_size=tp, decode_context_parallel_size=1),
        num_speculative_tokens=0,
        kv_transfer_config=None,
        attention_config=AttentionConfig(),
        scheduler_config=SchedulerConfig(
            is_encoder_decoder=False,
            max_model_len=sched["max_model_len"],
            max_num_seqs=sched["max_num_seqs"],
            max_num_batched_tokens=sched["max_num_batched_tokens"],
            disable_hybrid_kv_cache_manager=False,
        ),
    )

    def native_layer(kind):
        cls = KimiK3DeltaAttention if kind == "kda" else MultiHeadLatentAttention
        layer = cls.__new__(cls)
        torch.nn.Module.__init__(layer)
        if kind == "kda":
            layer.model_config, layer.cache_config = cfg.model_config, cache_config
            layer.tp_size, layer.num_heads, layer.head_dim = tp, heads, dim
            layer.conv_size, layer.num_spec = kernel, 0
            layer.kda_prefill_backend = server["kda_prefill_backend"]
        else:
            layer.kv_cache_dtype, layer.head_size = cache["mla_dtype"], width
            layer.non_causal_multi_token_decode = False
        return layer.get_kv_cache_spec(cfg)

    initial_mla, initial_kda = native_layer("mla"), native_layer("kda")
    row = initial_mla.page_size_bytes
    kernel_choices = TokenspeedMLABackend.get_supported_kernel_block_sizes()
    grid = math.lcm(128, min(kernel_choices), initial_kda.prefill_checkpoint_alignment)
    block = grid * ceil_div(initial_kda.real_page_size_bytes, row * grid)
    require(block % match == 0, "Manager block is incompatible with prefix unit")
    cache_config.block_size = cache_config.mamba_block_size = block
    cache_config.mamba_page_size_padded = block * row
    specs, layers = {}, []
    for index in range(k3.num_hidden_layers):
        kind = "kda" if k3.is_kda_layer(index) else "mla"
        name = f"language_model.model.layers.{index}.self_attn"
        spec = native_layer(kind)
        expected = (
            3 * (kernel - 1) * local_heads * dim * 2 + local_heads * dim * dim * 4
            if kind == "kda"
            else block * width
        )
        require(
            spec.real_page_size_bytes == expected and spec.page_size_bytes == block * row,
            f"Native byte or padding mismatch at {name}",
        )
        if kind == "kda":
            require(
                spec.shapes == ((kernel - 1, 3 * local_heads * dim), (local_heads, dim, dim))
                and spec.dtypes == (torch.bfloat16, torch.float32)
                and not spec.tp_replicated,
                f"Native KDA state mismatch at {name}",
            )
        else:
            require(
                spec.num_kv_heads == spec.max_tp_shards == 1
                and spec.head_size_v == 0
                and spec.state_content_size_bytes == width,
                f"Native MLA layout mismatch at {name}",
            )
        specs[name] = spec
        layers.append(
            {
                "index": index,
                "name": name,
                "kind": kind,
                "native_spec": type(spec).__name__,
                "content_bytes": expected,
                "page_bytes": spec.page_size_bytes,
                "padding_bytes": spec.page_size_bytes - expected,
            }
        )
    register_all_kvcache_specs(cfg)
    groups = get_kv_cache_groups(cfg, specs)
    budget = ledger["gpu_per_rank"]["kv_pool_budget"]
    allocation = get_kv_cache_config_from_groups(cfg, groups, budget)
    scheduler_block, hash_block = resolve_kv_cache_block_sizes(allocation, cfg)
    hit = resolve_cache_hit_alignment_tokens(allocation, cfg, scheduler_block, hash_block)
    require(hash_block == hit == match, "Native prefix matching fell back")
    positions = {
        name: group_id for group_id, group in enumerate(groups) for name in group.layer_names
    }
    require(
        set(positions) == set(specs)
        and sum(len(group.layer_names) for group in groups) == len(specs),
        "Planner layer coverage changed",
    )
    for entry in layers:
        entry["group"] = positions[entry["name"]]
    pool_sizes = {tensor.size for tensor in allocation.kv_cache_tensors}
    require(len(pool_sizes) == 1 and allocation.num_blocks > 1, "Invalid shared native pool")
    allocated = pool_sizes.pop()
    stride = allocated // allocation.num_blocks
    require(allocated <= budget < allocated + stride, "Native budget rounding changed")
    summaries = []
    rank_rows = []
    for rank in range(tp):
        worker = SimpleNamespace(
            tp_rank=rank,
            tp_size=tp,
            dcp_size=1,
            hash_block_size=hash_block,
            _kv_cache_groups=groups,
            requested_store_chunk_size=store["store_chunk_size_tokens"],
        )
        shard_ids = []
        for group_id, group in enumerate(groups):
            factor = MooncakeStoreWorker._spec_tp_replication_factor(worker, group.kv_cache_spec)
            plan = MooncakeStoreWorker._make_store_group_plan(
                worker, group, tp, HeadMajorStoreLayout, factor
            )
            require(plan is not None, "Store TP planner rejected group")
            shard_ids.append(plan.tp_rank)
            if rank == 0:
                payload = len(group.layer_names) * (
                    row * match if plan.store_chunk_size else initial_kda.state_content_size_bytes
                )
                require(
                    store["disk_staging_size_bytes_per_client"]
                    >= ceil_div(payload, 4096) * 4096 + 8192,
                    "Disk staging cannot hold one aligned Store object",
                )
                summaries.append(
                    {
                        "group": group_id,
                        "spec": type(group.kv_cache_spec).__name__,
                        "layers": len(group.layer_names),
                        "tp_replicas": factor,
                        "store_shards": plan.store_shard_count,
                        "store_chunk_tokens": plan.store_chunk_size,
                        "logical_bytes_per_object_per_shard": payload,
                        "schema_fingerprint": plan.schema_fingerprint,
                    }
                )
        rank_rows.append(
            {
                "rank": rank,
                "kda_heads_half_open": [rank * local_heads, (rank + 1) * local_heads],
                "store_shard_by_group": shard_ids,
                "budget_bytes": budget,
                "native_pool_bytes": allocated,
                "rounding_remainder_bytes": budget - allocated,
                "native_pool_blocks": allocation.num_blocks,
                "store_client_config": {
                    "global_segment_size": store["global_segment_size_bytes_per_client"],
                    "local_buffer_size": store["local_buffer_size_bytes_per_client"],
                },
                "offload_environment": {
                    "MOONCAKE_OFFLOAD_LOCAL_BUFFER_SIZE_BYTES": str(
                        store["disk_staging_size_bytes_per_client"]
                    ),
                    "MOONCAKE_OFFLOAD_TOTAL_SIZE_LIMIT_BYTES": str(
                        disk["offload_quota_bytes_per_client"]
                    ),
                    "MOONCAKE_OFFLOAD_FILE_STORAGE_PATH": ledger["ssd_node"]["client_paths"][rank],
                },
            }
        )
    slice_data = read(root / reference["input"]["slice"], "slice")
    require(inputs["slice"] == reference["input"]["slice_sha256"], "Trace slice digest mismatch")
    probes = []
    for end, query in [
        (block, block),
        (block + match + 1, block + match + 1),
        *(
            (req["in"], (req["in"] - 1) % sched["max_num_batched_tokens"] + 1)
            for req in slice_data["requests"]
        ),
    ]:
        offsets, columns = compute_mamba_prefill_checkpoints(
            [end], [query], hash_block, block, initial_kda.prefill_checkpoint_alignment, False, hit
        )
        probes.append(
            {
                "start": end - query,
                "end": end,
                "internal_checkpoint_position": end - query + offsets[0] if offsets[0] else None,
                "column": columns[0],
            }
        )
    return {
        "schema_version": 1,
        "scope": "CPU native metadata only; no device residency, forward, Store I/O or FEMU measurement",
        "source_sha256": inputs,
        "native_vllm": VLLM_VERSION,
        "reference_vllm_commit": reference["reference_deployment"]["vllm_source_commit"],
        "geometry": {
            "mla_components": [k3.kv_lora_rank, k3.qk_rope_head_dim],
            "mla_bytes_per_token": row,
            "mla_native_storage_dtype": str(initial_mla.dtype),
            "mla_quant_mode": initial_mla.kv_quant_mode.name,
            "mla_shape_per_page": [1, block, width],
            "kda_shapes": initial_kda.shapes,
            "kda_dtypes": [str(dtype) for dtype in initial_kda.dtypes],
            "kda_component_bytes": [
                math.prod(shape) * dtype.itemsize
                for shape, dtype in zip(initial_kda.shapes, initial_kda.dtypes, strict=True)
            ],
            "kda_state_bytes": initial_kda.state_content_size_bytes,
            "kda_internal_state_alignment_bytes": initial_kda.state_alignment,
            "kernel_block_options": kernel_choices,
            "manager_block_tokens": scheduler_block,
            "hash_tokens": hash_block,
            "hit_alignment_tokens": hit,
            "prefill_checkpoint_alignment_tokens": initial_kda.prefill_checkpoint_alignment,
            "num_prefill_checkpoint_blocks": initial_kda.num_prefill_checkpoint_blocks,
            "common_page_bytes": row * block,
            "shared_pool_bytes_per_block": stride,
        },
        "layers": layers,
        "groups": summaries,
        "ranks": rank_rows,
        "checkpoint_probes": probes,
        "capacity_bytes": ledger,
        "mooncake": store,
        "ssd": disk,
    }, allocation
