"""Single-file checkpoints: weights + configs + vocab + code hierarchy.

Only plain tensors/dicts/lists are stored, so ``torch.load(weights_only=True)`` is safe.
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import torch

from codeformer.config import DataConfig, ModelConfig
from codeformer.data.codes import CodeHierarchy, CodeSpec
from codeformer.data.vocab import Vocab
from codeformer.models.transformer import CodeTransformer

FORMAT_VERSION = 1


def save_checkpoint(
    path: str | Path,
    model: CodeTransformer,
    vocab: Vocab,
    hierarchy: CodeHierarchy,
    spec: CodeSpec,
    data_cfg: DataConfig,
    extra: dict | None = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format_version": FORMAT_VERSION,
        "model_state": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "model_config": asdict(model.cfg),
        "data_config": asdict(data_cfg),
        "spec": spec.to_dict(),
        "vocab": vocab.to_list(),
        "paths": hierarchy.to_list(),
        "extra": extra or {},
    }
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)  # atomic on POSIX: readers never see a half-written file


def load_checkpoint(path: str | Path, device: str | torch.device = "cpu") -> dict:
    payload = torch.load(Path(path), map_location=device, weights_only=True)
    if payload.get("format_version") != FORMAT_VERSION:
        raise ValueError(f"Unsupported checkpoint format: {payload.get('format_version')}")
    model_cfg = ModelConfig(**payload["model_config"])
    model = CodeTransformer(model_cfg)
    model.load_state_dict(payload["model_state"])
    model.to(device)
    spec = CodeSpec.from_dict(payload["spec"])
    return {
        "model": model,
        "vocab": Vocab.from_list(payload["vocab"]),
        "hierarchy": CodeHierarchy.from_list(spec.n_levels, payload["paths"]),
        "spec": spec,
        "data_cfg": DataConfig(**payload["data_config"]),
        "extra": payload["extra"],
    }
