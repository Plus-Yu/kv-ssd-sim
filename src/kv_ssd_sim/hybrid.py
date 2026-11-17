"""Configurable MLA latent blocks and KDA convolution/recurrent snapshots.

Only payload mathematics is synthetic. Specs, allocator, checkpoint positions,
copy-on-write instructions and Store descriptors come from the pinned vLLM.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
from pathlib import Path

import numpy as np


@dataclasses.dataclass(frozen=True)
class HybridConfig:
    layers: tuple[str, ...] = ("mla", "kda")
    mla_latent_dim: int = 48
    mla_rope_dim: int = 16
    kda_heads: int = 2
    kda_head_dim: int = 16
    conv_kernel: int = 4
    conv_dtype: str = "float16"
    recurrent_dtype: str = "float32"

    @classmethod
    def load(cls, path: Path):
        values = json.loads(path.read_text())
        if "layers" in values:
            values["layers"] = tuple(values["layers"])
        result = cls(**values)
        if not 2 <= len(result.layers) <= 128 or set(result.layers) != {"mla", "kda"}:
            raise ValueError("Hybrid layers must contain both mla and kda, with 2..128 layers")
        for name in ("mla_latent_dim", "mla_rope_dim", "kda_heads", "kda_head_dim", "conv_kernel"):
            value = getattr(result, name)
            if type(value) is not int or not 1 <= value <= 4096:
                raise ValueError(f"Invalid positive geometry: {name}")
        if result.conv_kernel < 2:
            raise ValueError("conv_kernel must be >= 2")
        if result.conv_dtype not in {"float16", "bfloat16", "float32"}:
            raise ValueError("Unsupported convolution dtype")
        if result.recurrent_dtype not in {"float32", "bfloat16"}:
            raise ValueError("KDA recurrent state must be float32 or bfloat16")
        return result

    @property
    def fingerprint(self):
        return hashlib.sha256(
            json.dumps(dataclasses.asdict(self), sort_keys=True).encode()
        ).hexdigest()[:16]

    @property
    def page_bytes(self):
        sizes = {"float16": 2, "bfloat16": 2, "float32": 4}
        conv = (
            3 * self.kda_heads * self.kda_head_dim * (self.conv_kernel - 1) * sizes[self.conv_dtype]
        )
        recurrent = self.kda_heads * self.kda_head_dim**2 * sizes[self.recurrent_dtype]
        return (
            math.ceil(
                max(64 * (self.mla_latent_dim + self.mla_rope_dim) * 2, conv + recurrent) / 256
            )
            * 256
        )


class HybridBuffers:
    def __init__(self, config, num_blocks):
        import torch
        from vllm.model_executor.layers.mamba.mamba_utils import (
            MambaStateDtypeCalculator,
            MambaStateShapeCalculator,
        )
        from vllm.v1.attention.backends.registry import MambaAttentionBackendEnum
        from vllm.v1.kv_cache_interface import KVCacheGroupSpec, MambaSpec, MLAAttentionSpec

        self.config = config
        self.names = [f"model.layers.{i}.{kind}" for i, kind in enumerate(config.layers)]
        mla = MLAAttentionSpec(
            block_size=64,
            num_kv_heads=1,
            head_size=config.mla_latent_dim + config.mla_rope_dim,
            dtype=torch.float16,
            page_size_padded=config.page_bytes,
        )
        # Kimi's upstream KDA inherits GatedDeltaNetAttention.mamba_type=GDN_ATTN;
        # its state geometry/dtypes are calculated with the KDA-specific helpers.
        kda = MambaSpec(
            block_size=64,
            shapes=MambaStateShapeCalculator.kda_state_shape(
                1, config.kda_heads, config.kda_head_dim, conv_kernel_size=config.conv_kernel
            ),
            dtypes=MambaStateDtypeCalculator.kda_state_dtype(
                torch.float16, config.conv_dtype, config.recurrent_dtype
            ),
            mamba_type=MambaAttentionBackendEnum.GDN_ATTN,
            mamba_cache_mode="align",
            num_prefill_checkpoint_blocks=1,
            prefill_checkpoint_alignment=16,
            num_heads=config.kda_heads,
            page_size_padded=config.page_bytes,
        )
        self.specs = {"mla": mla, "kda": kda}
        self.groups = [
            KVCacheGroupSpec(
                [n for n, k in zip(self.names, config.layers) if k == kind], self.specs[kind]
            )
            for kind in ("mla", "kda")
        ]
        pool_count = max(config.layers.count(kind) for kind in ("mla", "kda"))
        if num_blocks * pool_count * config.page_bytes > 1024**3:
            raise ValueError("Hybrid CPU pool exceeds the prototype's 1 GiB guard")
        self.pools = [
            torch.zeros((num_blocks, config.page_bytes), dtype=torch.uint8)
            for _ in range(pool_count)
        ]
        # Native hybrid groups share physical pages: a global block ID belongs
        # to one group at a time. Padding smaller groups matches that allocator.
        self.raw = {}
        for group in self.groups:
            for index, name in enumerate(group.layer_names):
                self.raw[name] = self.pools[index]
        self.caches = {}
        self.views = {}
        for name, kind in zip(self.names, config.layers):
            raw = self.raw[name]
            if kind == "mla":
                content = (config.mla_latent_dim + config.mla_rope_dim) * 2
                tensor = raw[:, : 64 * content].view(num_blocks, 1, 64, content)
            else:
                tensor = raw[:, : kda.state_content_size_bytes].view(num_blocks, 1, 1, -1)
            self.caches[name] = tensor
            self.views[name] = tensor.numpy()
        self.raw_views = [tensor.numpy() for tensor in self.pools]

    def describe(self):
        return {
            "config": dataclasses.asdict(self.config),
            "page_bytes": self.config.page_bytes,
            "kda_shapes": self.specs["kda"].shapes,
            "kda_dtypes": [str(t) for t in self.specs["kda"].dtypes],
            "kda_snapshot_bytes_per_layer": self.specs["kda"].state_content_size_bytes,
            "mla_bytes_per_token_per_layer": (self.config.mla_latent_dim + self.config.mla_rope_dim)
            * 2,
            "state_math": "deterministic prefix digests; no attention arithmetic",
            "checkpoint_alignment": 16,
            "store_tp_size": 1,
        }

    def copy_blocks(self, copies, events):
        for copy in copies or []:
            for raw in self.raw_views:
                raw[copy.dst_block_id] = raw[copy.src_block_id]
            events.emit("kv_block_copied", source=copy.src_block_id, destination=copy.dst_block_id)

    def materialize(self, runtime, request, start, end):
        from vllm.model_executor.layers.mamba.checkpoint import compute_mamba_prefill_checkpoints

        tables = runtime.scheduler.kv_cache_manager.get_blocks(request.request_id).get_block_ids()
        patterns = runtime.payloads.patterns[request.request_id]
        # The native manager can suppress unretained internal checkpoints. Only
        # export a checkpoint when its actual physical column is non-null.
        offsets, cols = compute_mamba_prefill_checkpoints(
            [end], [end - start], 64, 64, 16, False, 64
        )
        for layer, (name, kind) in enumerate(zip(self.names, self.config.layers)):
            view = self.views[name]
            if kind == "mla":
                pos = start
                width = view.shape[-1]
                while pos < end:
                    col, offset = divmod(pos, 64)
                    stop = min(end, (col + 1) * 64)
                    # Compact patterns are tiled only across the scheduled range.
                    view[tables[0][col], 0, offset : offset + stop - pos] = np.tile(
                        patterns[layer, pos:stop], (1, math.ceil(width / 32))
                    )[:, :width]
                    pos = stop
            else:
                snapshots = [(math.ceil(end / 64) - 1, end, "running")]
                if offsets[0] and cols[0] >= 0 and tables[1][cols[0]] != 0:
                    snapshots.append((cols[0], start + offsets[0], "checkpoint"))
                for col, boundary, role in snapshots:
                    block = tables[1][col]
                    if block == 0:
                        raise RuntimeError("Native KDA running state cannot use the null block")
                    width = view.shape[-1]
                    view[block, 0, 0] = np.tile(
                        patterns[layer, boundary - 1], math.ceil(width / 32)
                    )[:width]
                    runtime.events.emit(
                        "kda_state_materialized",
                        request_id=request.request_id,
                        layer=name,
                        block=block,
                        boundary_tokens=boundary,
                        role=role,
                        state_bytes=width,
                    )
