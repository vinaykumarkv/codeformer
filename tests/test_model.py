import pytest
import torch

from codeformer.config import ModelConfig, config_from_dict
from codeformer.models.transformer import CodeTransformer


@pytest.fixture
def model():
    torch.manual_seed(0)
    cfg = ModelConfig(
        vocab_size=50,
        n_layers=2,
        n_heads=2,
        d_model=32,
        d_ff=64,
        max_seq_len=16,
        dropout=0.0,
        n_future=3,
    )
    return CodeTransformer(cfg).eval()


def test_output_shapes(model):
    x = torch.randint(0, 50, (4, 10))
    out = model(x)
    assert len(out.logits) == 3
    assert all(lg.shape == (4, 10, 50) for lg in out.logits)
    assert out.loss is None


def test_causality(model):
    """Changing future tokens must not change earlier hidden states."""
    x = torch.randint(0, 50, (1, 10))
    y = x.clone()
    y[0, 7:] = (y[0, 7:] + 1) % 50
    with torch.no_grad():
        hx, hy = model.hidden(x), model.hidden(y)
    assert torch.allclose(hx[:, :7], hy[:, :7], atol=1e-5)
    assert not torch.allclose(hx[:, 7:], hy[:, 7:], atol=1e-5)


def test_padding_invariance_left_and_right(model):
    x = torch.randint(5, 50, (1, 6))
    with torch.no_grad():
        ref = model.hidden(x)[0]
        right = model.hidden(
            torch.cat([x, torch.zeros(1, 4, dtype=torch.long)], 1),
            torch.tensor([[1] * 6 + [0] * 4]),
        )[0, :6]
        left = model.hidden(
            torch.cat([torch.zeros(1, 4, dtype=torch.long), x], 1),
            torch.tensor([[0] * 4 + [1] * 6]),
        )[0, 4:]
    assert torch.allclose(ref, right, atol=1e-5)
    assert torch.allclose(ref, left, atol=1e-5)


def test_no_nan_with_fully_padded_rows(model):
    x = torch.randint(0, 50, (2, 8))
    mask = torch.zeros(2, 8, dtype=torch.long)
    mask[0, :] = 1
    assert torch.isfinite(model.hidden(x, mask)).all()


def test_multi_token_loss_alignment(model):
    """Head j at position t is scored against labels[t+1+j]."""
    x = torch.randint(5, 50, (2, 9))
    labels = x.clone()
    out = model(x, labels=labels)
    assert out.loss is not None and torch.isfinite(out.loss)
    assert len(out.head_losses) == 3
    # recompute head-1 loss by hand
    lg = out.logits[1][:, :-2].reshape(-1, 50)
    manual = torch.nn.functional.cross_entropy(lg, labels[:, 2:].reshape(-1))
    assert torch.allclose(manual, out.head_losses[1], atol=1e-5)


def test_all_ignored_labels_give_finite_loss(model):
    x = torch.randint(0, 50, (2, 6))
    out = model(x, labels=torch.full_like(x, -100))
    assert torch.isfinite(out.loss) and out.loss.item() == 0.0


def test_backward_updates_all_heads(model):
    model.train()
    x = torch.randint(5, 50, (2, 9))
    model(x, labels=x).loss.backward()
    for a in model.adapters:
        assert all(p.grad is not None and p.grad.abs().sum() > 0 for p in a.parameters())


def test_sequence_too_long(model):
    with pytest.raises(ValueError):
        model(torch.zeros(1, 17, dtype=torch.long))


def test_head_index_checked(model):
    with pytest.raises(IndexError):
        model.head_logits(torch.zeros(1, 2, 32), 3)


def test_resize_vocab_preserves_weights_and_outputs(model):
    x = torch.randint(0, 50, (1, 8))
    with torch.no_grad():
        before = model(x).logits[0][..., :50]
    old_w = model.tok_emb.weight.detach().clone()
    model.resize_vocab(60)
    assert model.tok_emb.num_embeddings == 60 and model.cfg.vocab_size == 60
    assert torch.equal(model.tok_emb.weight[:50], old_w)
    with torch.no_grad():
        after = model(x).logits[0]
    assert after.shape[-1] == 60
    assert torch.allclose(before, after[..., :50], atol=1e-5)
    with pytest.raises(ValueError):
        model.resize_vocab(10)


def test_config_validation():
    with pytest.raises(ValueError):
        ModelConfig(d_model=30, n_heads=4)
    with pytest.raises(ValueError):
        config_from_dict({"model": {"nonsense": 1}})
    with pytest.raises(ValueError):
        config_from_dict({"bogus": {}})
    assert config_from_dict({"model": {"d_model": 64, "n_heads": 4}}).model.d_model == 64


def test_vocab_size_required():
    with pytest.raises(ValueError):
        CodeTransformer(ModelConfig())
