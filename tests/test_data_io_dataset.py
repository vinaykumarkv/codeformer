import json

import pytest
import torch

from codeformer.data.codes import CodeSpec
from codeformer.data.dataset import IGNORE_INDEX, CodeDataset, encode_prompt, make_collate_fn
from codeformer.data.io import (
    Example,
    load_examples,
    write_examples_csv,
    write_examples_jsonl,
)
from codeformer.data.tokenizer import SimpleTokenizer
from codeformer.data.vocab import Vocab

SPEC = CodeSpec(3, "-")


def test_csv_roundtrip_and_aliases(tmp_path):
    p = tmp_path / "d.csv"
    p.write_text("predictor_text,output_text\nhello world,a-b-c\n")
    assert load_examples(p, SPEC) == [Example("hello world", ("a", "b", "c"))]


def test_jsonl_realtime_schema_and_list_codes(tmp_path):
    p = tmp_path / "d.jsonl"
    rows = [
        {"predictor_text": "x y", "output_text": "a-b-c"},
        {"text": "z", "codes": ["d", "e", "f"]},
    ]
    p.write_text("\n".join(json.dumps(r) for r in rows) + "\n\n")
    out = load_examples(p, SPEC)
    assert [e.codes for e in out] == [("a", "b", "c"), ("d", "e", "f")]


def test_malformed_rows_skipped_or_raised(tmp_path):
    p = tmp_path / "d.csv"
    p.write_text("text,codes\nok,a-b-c\nbad,a-b\n,a-b-c\n")
    assert len(load_examples(p, SPEC)) == 1
    with pytest.raises(ValueError):
        load_examples(p, SPEC, strict=True)


def test_unsupported_format(tmp_path):
    p = tmp_path / "d.txt"
    p.write_text("x")
    with pytest.raises(ValueError):
        load_examples(p, SPEC)


def test_writers(tmp_path):
    ex = [Example("t", ("a", "b", "c"))]
    write_examples_csv(tmp_path / "a.csv", ex, SPEC)
    write_examples_jsonl(tmp_path / "a.jsonl", ex, SPEC)
    assert load_examples(tmp_path / "a.csv", SPEC) == ex
    assert load_examples(tmp_path / "a.jsonl", SPEC) == ex


@pytest.fixture
def vocab_tok():
    tok = SimpleTokenizer()
    v = Vocab()
    v.extend([tok.tokenize("one two three")], [("a", "b", "c")])
    return v, tok


def test_dataset_layout_and_label_masking(vocab_tok):
    v, tok = vocab_tok
    ds = CodeDataset([Example("one two", ("a", "b", "c"))], v, tok, max_seq_len=16, n_levels=3)
    ids, labels = ds[0]
    assert ids[0] == v.bos_id and ids[3] == v.sep_id and ids[-1] == v.eos_id
    assert len(ids) == 1 + 2 + 1 + 3 + 1
    assert labels[:4] == [IGNORE_INDEX] * 4  # prompt is not supervised
    assert labels[4:] == ids[4:]  # codes + eos are


def test_prompt_truncation(vocab_tok):
    v, tok = vocab_tok
    p = encode_prompt("one " * 100, v, tok, max_seq_len=10, n_levels=3)
    assert len(p) + 3 + 1 == 10  # prompt + 3 codes + eos == max_seq_len


def test_unknown_code_raises(vocab_tok):
    v, tok = vocab_tok
    with pytest.raises(KeyError):
        CodeDataset([Example("one", ("zz", "b", "c"))], v, tok, 16, 3)


def test_collate_padding(vocab_tok):
    v, tok = vocab_tok
    ds = CodeDataset(
        [Example("one", ("a", "b", "c")), Example("one two three", ("a", "b", "c"))],
        v,
        tok,
        16,
        3,
    )
    batch = make_collate_fn(v.pad_id)([ds[0], ds[1]])
    assert batch["input_ids"].shape == batch["labels"].shape == batch["attention_mask"].shape
    assert batch["attention_mask"][0].sum() == len(ds[0][0])
    assert (batch["input_ids"][0][~batch["attention_mask"][0]] == v.pad_id).all()
    assert (batch["labels"][0][~batch["attention_mask"][0]] == IGNORE_INDEX).all()
    assert batch["input_ids"].dtype == torch.long
