"""Wire length/prefix and rejection contracts for the synthetic HTTP endpoint."""

from pathlib import Path

import pytest
from transformers import AutoTokenizer

from kv_ssd_sim.server import parse_chat


@pytest.fixture(scope="module")
def tokenizer():
    return AutoTokenizer.from_pretrained(
        str(Path(__file__).resolve().parents[1] / "configs/sim-tokenizer"),
        local_files_only=True,
    )


def test_chat_boundaries_preserve_token_identity(tokenizer):
    def parse(messages):
        return parse_chat(
            {"model": "sim", "messages": messages, "max_tokens": 4},
            tokenizer,
            model="sim",
            token_limit=1024,
        )[0]

    prefix = "An actual shared prefix. " * 12
    single = parse([{"role": "user", "content": prefix + "tail"}])
    multi = parse([{"role": "system", "content": prefix}, {"role": "user", "content": "tail"}])
    assert single == multi
    assert len(single) == len(prefix) + 4
    assert tokenizer.encode(tokenizer.decode(list(single)), add_special_tokens=False) == list(
        single
    )


@pytest.mark.parametrize(
    "patch",
    [
        {"max_tokens": 0},
        {"max_tokens": True},
        {"stream": "true"},
        {"stream_options": None},
        {"model": "other"},
        {"n": 2},
        {"messages": [{"role": "user", "content": "x" * 128}]},
        {"messages": [{"role": "user", "content": [{"type": "image_url"}]}]},
    ],
)
def test_invalid_requests_are_rejected_without_truncation(tokenizer, patch):
    body = {"model": "sim", "messages": [{"role": "user", "content": "hello"}], "max_tokens": 4}
    body.update(patch)
    with pytest.raises((ValueError, TypeError)):
        parse_chat(body, tokenizer, model="sim", token_limit=128)
