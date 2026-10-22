import pytest
import torch

from codeformer.data.synthetic import make_synthetic
from codeformer.evaluation import evaluate_predictor
from codeformer.training.checkpoint import load_checkpoint


def test_training_reduces_loss_and_saves_checkpoint(trained):
    r = trained["result"]
    assert r.train_losses[-1] < r.train_losses[0] * 0.5
    assert trained["path"].exists()
    assert r.best_val_loss < r.val_losses[0]


def test_checkpoint_roundtrip_is_exact(trained, predictor):
    ck = load_checkpoint(trained["path"])
    x = torch.randint(0, len(ck["vocab"]), (1, 10))
    ck["model"].eval()
    with torch.no_grad():
        a = ck["model"](x).logits[0]
        b = predictor.model(x).logits[0]
    assert torch.allclose(a, b)
    assert len(ck["hierarchy"]) == len(predictor.hierarchy) > 0


def test_checkpoint_rejects_wrong_version(tmp_path):
    p = tmp_path / "bad.pt"
    torch.save({"format_version": 999}, p)
    with pytest.raises(ValueError):
        load_checkpoint(p)


@pytest.mark.parametrize("mode", ["autoregressive", "parallel"])
def test_model_actually_learns_both_modes(predictor, trained, mode):
    metrics = evaluate_predictor(predictor, trained["val"], mode=mode)
    chance = 1 / 216
    assert metrics["path_accuracy"] > 0.5, metrics  # >> chance (~0.5%)
    assert metrics["path_accuracy"] > 50 * chance
    assert metrics["valid_path_rate"] == 1.0
    assert all(a > 0.7 for a in metrics["level_accuracy"])


def test_predictions_are_always_valid_paths(predictor):
    nonsense = ["completely unrelated gibberish zzz", "", "loc6x999999 loc1x7"]
    for mode in ("autoregressive", "parallel"):
        for out in predictor.predict(nonsense, top_k=1, mode=mode):
            assert out[0].codes in predictor.hierarchy


def test_beam_search_topk_sorted_unique_and_valid(predictor):
    res = predictor.predict("ship to l3k03 l2k02 please", top_k=5, beam_width=8)[0]
    assert len(res) == 5
    scores = [p.score for p in res]
    assert scores == sorted(scores, reverse=True)
    assert len({p.codes for p in res}) == 5
    assert all(p.codes in predictor.hierarchy for p in res)
    assert all(len(p.codes) == 6 and len(p.level_probs) == 6 for p in res)
    assert 0 < sum(scores) <= 1.0 + 1e-6


def test_greedy_equals_beam_width_one(predictor):
    a = predictor.predict("l2k02 l1k01 parcel", top_k=1)[0][0]
    b = predictor.predict("l2k02 l1k01 parcel", top_k=1, beam_width=1)[0][0]
    assert a.codes == b.codes


def test_batch_matches_single_in_parallel_mode(predictor):
    """Left-padding in a mixed-length batch must not change the answer."""
    texts = ["l6k01 l5k01", "please ship the parcel to l4k01 l3k03 near the office address"]
    batched = predictor.predict(texts, mode="parallel")
    for t, b in zip(texts, batched, strict=True):
        single = predictor.predict(t, mode="parallel")[0]
        assert single[0].codes == b[0].codes
        assert single[0].score == pytest.approx(b[0].score, rel=1e-3)


def test_hierarchy_constraint_matters(predictor):
    """Without the hierarchy the level-restricted output may be invalid; with it, never."""
    texts = [e.text for e in make_synthetic(40, seed=99)]
    with_h = predictor.predict(texts, mode="parallel", use_hierarchy=True)
    assert all(p[0].codes in predictor.hierarchy for p in with_h)
    without = predictor.predict(texts, mode="parallel", use_hierarchy=False)
    assert all(len(p[0].codes) == 6 for p in without)  # still level-correct


def test_invalid_arguments(predictor):
    with pytest.raises(ValueError):
        predictor.predict("x", mode="nope")
    with pytest.raises(ValueError):
        predictor.predict("x", top_k=0)


def test_parallel_requires_enough_heads(predictor):
    from dataclasses import replace

    from codeformer.inference.predictor import Predictor
    from codeformer.models.transformer import CodeTransformer

    cfg = replace(predictor.model.cfg, n_future=2)
    p = Predictor(CodeTransformer(cfg), predictor.vocab, predictor.hierarchy, predictor.spec)
    with pytest.raises(ValueError, match="n_future"):
        p.predict("x", mode="parallel")
    assert len(p.predict("x", mode="autoregressive")[0]) == 1  # AR still works


def test_overfit_tiny_dataset(tmp_path, tiny_cfg, spec):
    """Sanity: the whole stack can memorise 12 examples."""
    from dataclasses import replace

    from codeformer.inference.predictor import Predictor
    from codeformer.training.pipeline import train_model

    data = make_synthetic(12, spec, seed=5)
    cfg = replace(tiny_cfg, train=replace(tiny_cfg.train, epochs=200, batch_size=12))
    train_model(data, None, cfg, tmp_path / "m.pt", device="cpu")
    p = Predictor.from_checkpoint(tmp_path / "m.pt")
    m = evaluate_predictor(p, data)
    assert m["path_accuracy"] >= 0.9
