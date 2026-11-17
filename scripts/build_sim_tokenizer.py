#!/usr/bin/env python3
"""Build an offline reversible character tokenizer for synthetic trace experiments."""

import argparse
from pathlib import Path

from tokenizers import Tokenizer, decoders, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast


def build(output: Path):
    # No BPE merges, normalization or role delimiters: concatenation preserves
    # token identity, including at AIPerf multi-turn message boundaries.
    vocab = {"[PAD]": 0, "[BOS]": 1, "[EOS]": 2, "[UNK]": 3}
    vocab.update({chr(i): i + 4 for i in range(31996)})
    backend = Tokenizer(models.WordLevel(vocab, unk_token="[UNK]"))
    backend.pre_tokenizer = pre_tokenizers.Split("", behavior="isolated")
    backend.decoder = decoders.Fuse()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        pad_token="[PAD]",
        bos_token="[BOS]",
        eos_token="[EOS]",
        unk_token="[UNK]",
        model_max_length=1048576,
        clean_up_tokenization_spaces=False,
    )
    tokenizer.chat_template = "{% for message in messages %}{{ message['content'] }}{% endfor %}"
    output.mkdir(parents=True, exist_ok=False)
    tokenizer.save_pretrained(output)
    sample = "A shared prefix.\n中文"
    assert tokenizer.decode(tokenizer.encode(sample, add_special_tokens=False)) == sample
    assert len(tokenizer.encode(sample, add_special_tokens=False)) == len(sample)
    print(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("configs/sim-tokenizer"))
    build(parser.parse_args().output)
