"""Sequence construction and batching.

Sequence layout::

    <bos> t1 t2 ... tn <sep> c1 c2 ... c6 <eos>
    |---- prompt (masked) ---|---- supervised targets ----|
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch.utils.data import Dataset

from codeformer.data.io import Example
from codeformer.data.tokenizer import SimpleTokenizer
from codeformer.data.vocab import Vocab

IGNORE_INDEX = -100


def max_text_tokens(max_seq_len: int, n_levels: int) -> int:
    """Room for text = total - (bos + sep + n_levels codes + eos)."""
    room = max_seq_len - n_levels - 3
    if room < 1:
        raise ValueError("max_seq_len too small for the number of levels")
    return room


def encode_prompt(
    text: str, vocab: Vocab, tokenizer: SimpleTokenizer, max_seq_len: int, n_levels: int
) -> list[int]:
    toks = tokenizer.tokenize(text)[: max_text_tokens(max_seq_len, n_levels)]
    return [vocab.bos_id, *(vocab.id(t) for t in toks), vocab.sep_id]


class CodeDataset(Dataset):
    def __init__(
        self,
        examples: Sequence[Example],
        vocab: Vocab,
        tokenizer: SimpleTokenizer,
        max_seq_len: int,
        n_levels: int,
    ) -> None:
        self.items: list[tuple[list[int], list[int]]] = []
        for ex in examples:
            prompt = encode_prompt(ex.text, vocab, tokenizer, max_seq_len, n_levels)
            target = []
            for level, code in enumerate(ex.codes):
                cid = vocab.code_id(level, code)
                if cid is None:
                    raise KeyError(f"Code {code!r} at level {level + 1} missing from vocab")
                target.append(cid)
            target.append(vocab.eos_id)
            self.items.append((prompt + target, [IGNORE_INDEX] * len(prompt) + target))

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> tuple[list[int], list[int]]:
        return self.items[idx]


def make_collate_fn(pad_id: int):
    def collate(batch: list[tuple[list[int], list[int]]]) -> dict[str, torch.Tensor]:
        max_len = max(len(ids) for ids, _ in batch)
        input_ids = torch.full((len(batch), max_len), pad_id, dtype=torch.long)
        labels = torch.full((len(batch), max_len), IGNORE_INDEX, dtype=torch.long)
        mask = torch.zeros((len(batch), max_len), dtype=torch.bool)
        for i, (ids, lab) in enumerate(batch):
            n = len(ids)
            input_ids[i, :n] = torch.tensor(ids)
            labels[i, :n] = torch.tensor(lab)
            mask[i, :n] = True
        return {"input_ids": input_ids, "labels": labels, "attention_mask": mask}

    return collate
