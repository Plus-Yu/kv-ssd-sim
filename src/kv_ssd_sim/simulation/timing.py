"""Virtual scheduler-step time for fixed controls and the public TP8 reference."""

import math
from bisect import bisect_left
from dataclasses import dataclass
from itertools import pairwise


@dataclass(frozen=True)
class StepTiming:
    prefill_ms: float | None
    decode_ms: float
    decode_ms_batch_8: float
    prefill_tokens_s: float | None = None
    scale: float = 1

    def __post_init__(self):
        if self.prefill_ms is None and self.prefill_tokens_s is None:
            raise ValueError("A prefill duration or rate is required")
        for value in (
            self.prefill_ms,
            self.prefill_tokens_s,
            self.decode_ms,
            self.decode_ms_batch_8,
            self.scale,
        ):
            if value is not None and (not math.isfinite(value) or value <= 0):
                raise ValueError("Compute parameters must be finite and positive")

    def duration_ms(self, fresh_tokens, batch, decode_only_sequences):
        if not batch:
            return 0
        if not 1 <= batch <= 8:
            raise ValueError("The TP8 scheduler profile covers batches from 1 to 8")
        decode = self.decode_ms + (self.decode_ms_batch_8 - self.decode_ms) * (batch - 1) / 7
        if not fresh_tokens:
            return decode * self.scale
        if self.prefill_ms is None:
            # A short or mixed step still has to execute a forward pass for the batch.
            return max(decode, fresh_tokens * 1000 / self.prefill_tokens_s) * self.scale
        return max(self.prefill_ms, decode if decode_only_sequences else 0) * self.scale


@dataclass(frozen=True)
class ContextTiming:
    """Single-deployment latency anchors plus incremental MLA work per TP rank."""

    prefill_anchor_tokens: int
    prefill_anchor_ms: float
    decode_context_tokens: tuple[int, int]
    decode_batch: tuple[int, ...]
    decode_short_ms: tuple[float, ...]
    decode_long_ms: tuple[float | None, ...]
    ms_per_query_key_pair: float
    ms_per_history_token: float
    scale: float

    @classmethod
    def from_profile(cls, profile, plan, model, scale=1.0, context_scale=1.0):
        target = {
            "model": "Kimi-K3",
            "gpu": "B300",
            "tp": 8,
            "dcp": 1,
            "pp": 1,
            "speculative": False,
        }
        if profile["target"] != target or not profile["sources"].get("prefill_and_decode"):
            raise ValueError("Expected paired timing sources for a single K3 TP8/DCP1 B300 node")
        (mla,) = (g for g in plan["groups"] if g["spec"] == "MLAAttentionSpec")
        heads, remainder = divmod(model["num_attention_heads"], target["tp"])
        if remainder:
            raise ValueError("MLA heads must divide evenly across TP ranks")
        qk_dim = model["qk_nope_head_dim"] + model["qk_rope_head_dim"]
        pair_flops = 2 * heads * mla["layers"] * (qk_dim + model["v_head_dim"])
        history_bytes = mla["layers"] * plan["geometry"]["mla_bytes_per_token"]
        tflops, gb_s = profile["dense_tflops_per_gpu"], profile["hbm_gb_s_per_gpu"]
        if not all(math.isfinite(x) and x > 0 for x in (tflops, gb_s, scale)):
            raise ValueError(
                "Compute scale and effective hardware rates must be finite and positive"
            )
        if not math.isfinite(context_scale) or context_scale < 0:
            raise ValueError("Context cost scale must be finite and nonnegative")
        return cls(
            profile["prefill_anchor_tokens"],
            profile["prefill_anchor_ms"],
            tuple(profile["decode_context_tokens"]),
            tuple(profile["decode_batch"]),
            tuple(profile["decode_short_ms"]),
            tuple(profile["decode_long_ms"]),
            context_scale * pair_flops / (tflops * 1e9),
            context_scale * history_bytes / (gb_s * 1e6),
            scale,
        )

    def __post_init__(self):
        short, long = self.decode_context_tokens
        batches = self.decode_batch
        if not (isinstance(short, int) and isinstance(long, int) and 0 < short < long):
            raise ValueError("Decode contexts must be increasing positive token counts")
        if (
            not batches
            or batches[0] != 1
            or any(type(b) is not int for b in batches)
            or any(a >= b for a, b in pairwise(batches))
            or not len(batches) == len(self.decode_short_ms) == len(self.decode_long_ms)
        ):
            raise ValueError("Expected increasing batch anchors with paired timing rows")
        measured = [
            self.prefill_anchor_ms,
            *self.decode_short_ms,
            *(v for v in self.decode_long_ms if v is not None),
        ]
        if (
            type(self.prefill_anchor_tokens) is not int
            or self.prefill_anchor_tokens < 1
            or not all(math.isfinite(x) and x > 0 for x in measured)
        ):
            raise ValueError("Invalid timing anchor")
        if self.prefill_anchor_ms / self.prefill_anchor_tokens <= (
            self.ms_per_query_key_pair * self.prefill_anchor_tokens / 2
        ):
            raise ValueError("Inferred short-context prefill cost must stay positive")

    def _decode_ms(self, batch, histories):
        short, long = self.decode_context_tokens
        if batch > self.decode_batch[-1]:
            raise ValueError("Scheduled batch exceeds timing profile coverage")
        index = bisect_left(self.decode_batch, batch)
        previous = max(0, index - 1)
        width = self.decode_batch[index] - self.decode_batch[previous]
        weight = (batch - self.decode_batch[previous]) / width if width else 0
        within = sum(max(0, min(h, long) - short) for h in histories)
        tail = sum(max(0, h - long) for h in histories)

        def anchor(i):
            base = self.decode_short_ms[i]
            long_ms = self.decode_long_ms[i]
            # A missing cell has only the MLA traffic estimate as its slope.
            slope = (
                self.ms_per_history_token
                if long_ms is None
                else ((long_ms - base) / ((long - short) * self.decode_batch[i]))
            )
            return base, slope

        # Interpolate full, homogeneous-batch latency first, then distribute the
        # context increment over the actual decode sequences in this mixed step.
        base0, slope0 = anchor(previous)
        base1, slope1 = anchor(index)
        extra = (1 - weight) * slope0 * self.decode_batch[
            previous
        ] + weight * slope1 * self.decode_batch[index]
        return (
            base0
            + weight * (base1 - base0)
            + within / batch * extra
            + tail * self.ms_per_history_token
        )

    def step_ms(self, sequences):
        """Each pair is (already-computed context, fresh prompt tokens in this step)."""
        if not sequences:
            return 0
        n = self.prefill_anchor_tokens
        base = self.prefill_anchor_ms / n
        prefill = sum(
            q * (base + self.ms_per_query_key_pair * (history + (q - n) / 2))
            for history, q in sequences
        )
        decode = self._decode_ms(len(sequences), [h for h, q in sequences if not q])
        return max(prefill, decode) * self.scale
