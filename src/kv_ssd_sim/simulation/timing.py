"""Virtual scheduler-step time for fixed controls and the public TP8 reference."""

import math
from dataclasses import dataclass


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
