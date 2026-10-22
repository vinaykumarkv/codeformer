"""Metrics for hierarchical code prediction."""

from __future__ import annotations

from collections.abc import Sequence

from codeformer.data.io import Example
from codeformer.inference.predictor import Predictor


def evaluate_predictor(
    predictor: Predictor,
    examples: Sequence[Example],
    mode: str = "autoregressive",
    top_k: int = 3,
    batch_size: int = 64,
) -> dict:
    """Per-level accuracy, exact full-path accuracy, top-k path accuracy, valid-path rate."""
    n = predictor.spec.n_levels
    level_hits = [0] * n
    exact = topk_hit = valid = 0
    k = 1 if mode == "parallel" else top_k
    for i in range(0, len(examples), batch_size):
        chunk = examples[i : i + batch_size]
        preds = predictor.predict([e.text for e in chunk], top_k=k, mode=mode)
        for ex, plist in zip(chunk, preds, strict=True):
            best = plist[0].codes
            for lv in range(n):
                level_hits[lv] += int(best[lv] == ex.codes[lv])
            exact += int(best == ex.codes)
            topk_hit += int(any(p.codes == ex.codes for p in plist))
            valid += int(best in predictor.hierarchy)
    total = max(len(examples), 1)
    return {
        "n": len(examples),
        "level_accuracy": [h / total for h in level_hits],
        "path_accuracy": exact / total,
        f"top{k}_path_accuracy": topk_hit / total,
        "valid_path_rate": valid / total,
    }
