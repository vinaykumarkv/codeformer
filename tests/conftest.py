from __future__ import annotations

import pytest

from codeformer.config import AppConfig, DataConfig, ModelConfig, TrainConfig
from codeformer.data.codes import CodeSpec
from codeformer.data.synthetic import make_synthetic
from codeformer.inference.predictor import Predictor
from codeformer.training.pipeline import train_model


@pytest.fixture(scope="session")
def spec() -> CodeSpec:
    return CodeSpec(n_levels=6, delimiter="-")


@pytest.fixture(scope="session")
def tiny_cfg() -> AppConfig:
    return AppConfig(
        data=DataConfig(),
        model=ModelConfig(
            n_layers=2, n_heads=2, d_model=64, d_ff=128, max_seq_len=32, dropout=0.0, n_future=6
        ),
        train=TrainConfig(
            epochs=20, batch_size=32, lr=3e-3, early_stopping_patience=0, log_every=10_000
        ),
    )


@pytest.fixture(scope="session")
def trained(tmp_path_factory, tiny_cfg, spec):
    """A small model trained once per test session (a few seconds on CPU)."""
    train = make_synthetic(1500, spec, seed=1)
    val = make_synthetic(200, spec, seed=2)
    path = tmp_path_factory.mktemp("ckpt") / "model.pt"
    result = train_model(train, val, tiny_cfg, path, device="cpu")
    return {"path": path, "result": result, "val": val, "train": train}


@pytest.fixture(scope="session")
def predictor(trained) -> Predictor:
    return Predictor.from_checkpoint(trained["path"])
