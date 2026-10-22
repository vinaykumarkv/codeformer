"""Typed configuration objects and YAML loading."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TypeVar

import yaml

T = TypeVar("T")


@dataclass
class DataConfig:
    n_levels: int = 6
    delimiter: str = "-"
    min_freq: int = 1
    lowercase: bool = True


@dataclass
class ModelConfig:
    vocab_size: int = 0  # filled in after the vocabulary is built
    n_layers: int = 4
    n_heads: int = 4
    d_model: int = 128
    d_ff: int = 512
    max_seq_len: int = 64
    dropout: float = 0.1
    n_future: int = 6
    aux_loss_weight: float = 0.5

    def __post_init__(self) -> None:
        if self.d_model % self.n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        if self.n_future < 1:
            raise ValueError("n_future must be >= 1")


@dataclass
class TrainConfig:
    epochs: int = 15
    batch_size: int = 64
    lr: float = 5e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.05
    grad_clip: float = 1.0
    seed: int = 42
    num_workers: int = 0
    mixed_precision: bool = False
    early_stopping_patience: int = 4
    log_every: int = 50


@dataclass
class AppConfig:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)


def _build(cls: type[T], values: dict[str, Any] | None) -> T:
    values = values or {}
    allowed = {f.name for f in dataclasses.fields(cls)}  # type: ignore[arg-type]
    unknown = set(values) - allowed
    if unknown:
        raise ValueError(f"Unknown config keys for {cls.__name__}: {sorted(unknown)}")
    return cls(**values)


def config_from_dict(raw: dict[str, Any] | None) -> AppConfig:
    raw = raw or {}
    unknown = set(raw) - {"data", "model", "train"}
    if unknown:
        raise ValueError(f"Unknown top-level config sections: {sorted(unknown)}")
    cfg = AppConfig(
        data=_build(DataConfig, raw.get("data")),
        model=_build(ModelConfig, raw.get("model")),
        train=_build(TrainConfig, raw.get("train")),
    )
    if cfg.model.n_future < 1:
        raise ValueError("model.n_future must be >= 1")
    return cfg


def load_config(path: str | Path | None) -> AppConfig:
    if path is None:
        return AppConfig()
    with open(path, encoding="utf-8") as fh:
        return config_from_dict(yaml.safe_load(fh))
