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
        if values.get("profile") == "k3-tp8":
            from .kimi_k3 import K3Profile

            return K3Profile.load(values)
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
