from dataclasses import replace

import pytest

from codeformer.data.codes import CodeSpec
from codeformer.data.io import Example, load_examples, write_examples_jsonl
from codeformer.evaluation import evaluate_predictor
from codeformer.inference.predictor import Predictor
from codeformer.training.checkpoint import load_checkpoint
from codeformer.training.pipeline import filter_known, incremental_update

NEW_PATH = ("99", "98", "97", "96", "95", "94")


def _new_examples(n=40):
    return [Example(f"brandnewplace{i % 3} zebra parcel", NEW_PATH) for i in range(n)]


def test_incremental_adds_new_codes_and_vocab(trained, tmp_path):
    out = tmp_path / "inc.pt"
    before = load_checkpoint(trained["path"])
    incremental_update(
        trained["path"],
        _new_examples(),
        out,
        replay_examples=trained["train"][:200],
        epochs=25,
        lr=2e-3,
    )
    after = load_checkpoint(out)
    assert len(after["vocab"]) > len(before["vocab"])
    assert after["model"].tok_emb.num_embeddings == len(after["vocab"])
    assert NEW_PATH in after["hierarchy"] and NEW_PATH not in before["hierarchy"]
    # old ids unchanged => old knowledge still addressable
    for tok in before["vocab"].to_list()[:50]:
        assert after["vocab"].id(tok) == before["vocab"].id(tok)

    p = Predictor.from_checkpoint(out)
    assert p.predict("brandnewplace1 zebra parcel", top_k=1)[0][0].codes == NEW_PATH


def test_incremental_with_replay_retains_old_skill(trained, tmp_path):
    out = tmp_path / "inc2.pt"
    incremental_update(
        trained["path"],
        _new_examples(30),
        out,
        replay_examples=trained["train"],
        replay_ratio=3.0,
        epochs=4,
        lr=5e-4,
    )
    m = evaluate_predictor(Predictor.from_checkpoint(out), trained["val"][:100])
    assert m["path_accuracy"] > 0.4


def test_incremental_reads_realtime_schema(trained, tmp_path):
    spec = CodeSpec()
    f = tmp_path / "rt.jsonl"
    write_examples_jsonl(f, _new_examples(5), spec)
    ex = load_examples(f, spec)
    assert len(ex) == 5 and ex[0].codes == NEW_PATH


def test_filter_known_drops_unseen(predictor):
    ex = [Example("a", NEW_PATH), Example("a", tuple(predictor.hierarchy.to_list()[0]))]
    assert len(filter_known(ex, predictor.vocab)) == 1


def test_empty_update_rejected(trained, tmp_path):
    with pytest.raises(ValueError):
        incremental_update(trained["path"], [], tmp_path / "x.pt")


def test_original_checkpoint_untouched(trained, tmp_path):
    h = trained["path"].stat().st_mtime_ns
    incremental_update(trained["path"], _new_examples(8), tmp_path / "o.pt", epochs=1)
    assert trained["path"].stat().st_mtime_ns == h


def test_train_config_replace_helper(tiny_cfg):
    assert replace(tiny_cfg.train, epochs=1).epochs == 1
