import json

import pytest

from codeformer.cli import main
from codeformer.data.codes import CodeSpec
from codeformer.data.io import write_examples_csv, write_examples_jsonl
from codeformer.data.synthetic import make_synthetic


def test_end_to_end_cli(tmp_path, capsys):
    spec = CodeSpec()
    train, val, rt = tmp_path / "t.csv", tmp_path / "v.csv", tmp_path / "rt.jsonl"
    write_examples_csv(train, make_synthetic(300, spec, seed=1), spec)
    write_examples_csv(val, make_synthetic(50, spec, seed=2), spec)
    write_examples_jsonl(rt, make_synthetic(30, spec, seed=3), spec)
    cfg = tmp_path / "c.yaml"
    cfg.write_text(
        "model: {n_layers: 1, n_heads: 2, d_model: 32, d_ff: 64, max_seq_len: 32, n_future: 6}\n"
        "train: {epochs: 2, batch_size: 32}\n"
    )
    ck, ck2 = tmp_path / "m.pt", tmp_path / "m2.pt"

    assert (
        main(
            [
                "train",
                "--config",
                str(cfg),
                "--train",
                str(train),
                "--val",
                str(val),
                "--out",
                str(ck),
            ]
        )
        == 0
    )
    assert json.loads(capsys.readouterr().out)["checkpoint"] == str(ck)

    assert main(["predict", "--checkpoint", str(ck), "--text", "l1k01 parcel", "--top-k", "2"]) == 0
    lines = capsys.readouterr().out.strip().splitlines()
    assert len(lines) == 2 and lines[0].split("\t")[0].count("-") == 5

    assert main(["predict", "--checkpoint", str(ck), "--text", "x", "--mode", "parallel"]) == 0
    capsys.readouterr()

    assert (
        main(
            [
                "finetune",
                "--checkpoint",
                str(ck),
                "--data",
                str(rt),
                "--replay",
                str(train),
                "--epochs",
                "1",
                "--out",
                str(ck2),
            ]
        )
        == 0
    )
    capsys.readouterr()

    assert main(["evaluate", "--checkpoint", str(ck2), "--data", str(val)]) == 0
    assert "path_accuracy" in json.loads(capsys.readouterr().out)


def test_train_without_val_file_uses_split(tmp_path, capsys):
    spec = CodeSpec()
    train = tmp_path / "t.csv"
    write_examples_csv(train, make_synthetic(120, spec, seed=1), spec)
    cfg = tmp_path / "c.yaml"
    cfg.write_text(
        "model: {n_layers: 1, n_heads: 2, d_model: 32, d_ff: 64, n_future: 6}\ntrain: {epochs: 1}\n"
    )
    assert (
        main(
            ["train", "--config", str(cfg), "--train", str(train), "--out", str(tmp_path / "m.pt")]
        )
        == 0
    )


def test_missing_command_exits():
    with pytest.raises(SystemExit):
        main([])
