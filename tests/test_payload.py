import hashlib
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

from kv_ssd_sim.payload import PreparedPayloads


def reference(tokens, layers=2):
    """The original per-token writer, expressed as raw bytes for comparison."""
    result = np.zeros((layers, 2, len(tokens), 2, 64), dtype=np.uint8)
    state = hashlib.sha256(b"kv-ssd-sim-v1")
    for pos, token in enumerate(tokens):
        state.update(token.to_bytes(4, "big"))
        for layer in range(layers):
            digest = hashlib.sha256(state.digest() + bytes([layer])).digest()
            result[layer, :, pos] = np.frombuffer(digest * 8, dtype=np.uint8).reshape(2, 2, 64)
    return result


def test_scheduled_copies_match_original_bytes_across_blocks_and_decode():
    prompt = tuple(range(4, 74))
    request = SimpleNamespace(request_id="r", prompt_token_ids=prompt, output_tokens=5)
    prepared = PreparedPayloads([request])
    caches = [np.full((4, 2, 64, 2, 2, 32), 0xA5, dtype=np.uint8) for _ in range(2)]
    blocks = [2, 0]
    # Nonsequential scheduling also covers recomputation after preemption.
    with patch("kv_ssd_sim.payload.hashlib.sha256", side_effect=AssertionError("runtime hash")):
        for start, end in [(64, 70), (0, 23), (23, 64), (70, 74), (74, 75), (0, 16)]:
            prepared.copy_range("r", start, end, blocks, caches, 64)
    expected = reference(list(prompt) + [100] * 5)
    for layer, cache in enumerate(caches):
        logical = np.concatenate([cache[b].reshape(2, 64, 2, 64) for b in blocks], axis=1)
        np.testing.assert_array_equal(logical[:, :75], expected[layer])
        assert np.all(logical[:, 75:] == 0xA5)
        assert np.all(cache[1] == 0xA5)  # Unallocated block remains untouched.


def test_restored_prefix_is_never_materialized_again():
    request = SimpleNamespace(request_id="r", prompt_token_ids=tuple(range(70)), output_tokens=1)
    prepared = PreparedPayloads([request])
    caches = [np.full((2, 2, 64, 2, 2, 32), 0xC3, dtype=np.uint8) for _ in range(2)]
    prepared.copy_range("r", 64, 70, [0, 1], caches, 64)
    for cache in caches:
        assert np.all(cache[0] == 0xC3)
        assert np.all(cache[1, :, 6:] == 0xC3)


def test_preparation_has_a_memory_limit():
    request = SimpleNamespace(request_id="r", prompt_token_ids=(1, 2), output_tokens=1)
    with pytest.raises(ValueError, match="select a smaller trace"):
        PreparedPayloads([request], max_bytes=1)
