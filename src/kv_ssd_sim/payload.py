"""Prepare compact placeholder bytes before replay; copy only scheduled slots.

The digest algorithm and byte layout exactly match the original implementation.
Preparation never populates the native KV pool or makes a Store object visible.
"""

from __future__ import annotations

import hashlib
from itertools import chain, repeat

import numpy as np

SYNTHETIC_TOKEN = 100
MAX_PREPARED_BYTES = 512 * 1024 * 1024


class PreparedPayloads:
    def __init__(self, requests, *, num_layers=2, max_bytes=MAX_PREPARED_BYTES):
        self.num_layers = num_layers
        self.nbytes = sum(
            (len(r.prompt_token_ids) + r.output_tokens) * num_layers * 32 for r in requests
        )
        if self.nbytes > max_bytes:
            raise ValueError(
                f"Prepared payload needs {self.nbytes} bytes, limit is {max_bytes}; "
                "select a smaller trace slice"
            )
        self.patterns = {}
        for request in requests:
            length = len(request.prompt_token_ids) + request.output_tokens
            patterns = np.empty((num_layers, length, 32), dtype=np.uint8)
            state = hashlib.sha256(b"kv-ssd-sim-v1")
            for position, token in enumerate(
                chain(request.prompt_token_ids, repeat(SYNTHETIC_TOKEN, request.output_tokens))
            ):
                state.update(int(token).to_bytes(4, "big", signed=False))
                prefix = state.digest()
                for layer in range(num_layers):
                    digest = hashlib.sha256(prefix + bytes([layer])).digest()
                    patterns[layer, position] = np.frombuffer(digest, dtype=np.uint8)
            patterns.flags.writeable = False
            self.patterns[request.request_id] = patterns

    def copy_range(self, request_id, start, end, block_ids, caches, block_size):
        """Caches are uint8 views shaped [blocks, K/V, tokens, heads, 2, 32].

        Each 32-byte pattern repeats across the two halves of a head, both
        heads and K/V. NumPy broadcasts directly into registered CPU storage;
        there is no temporary expanded tensor and no full-prefix hash here.
        Ranges can arrive out of order after preemption or external restoration.
        """
        patterns = self.patterns[request_id]
        if not 0 <= start <= end <= patterns.shape[1]:
            raise ValueError("Scheduled range lies outside the prepared request")
        if len(caches) != self.num_layers:
            raise ValueError("Prepared layer count differs from native caches")
        position = start
        while position < end:
            logical_block, offset = divmod(position, block_size)
            stop = min(end, position + block_size - offset)
            block = block_ids[logical_block]
            for layer, cache in enumerate(caches):
                cache[block, :, offset : offset + stop - position, :, :, :] = patterns[
                    layer, position:stop
                ][None, :, None, None, :]
            position = stop
