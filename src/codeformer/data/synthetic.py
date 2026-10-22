"""Synthetic hierarchical dataset generator (used by tests, demos and CI)."""

from __future__ import annotations

import itertools
import random

from codeformer.data.codes import CodeSpec
from codeformer.data.io import Example

_FILLER = "please deliver parcel near the office package ship to urgent customer address".split()


def make_synthetic(
    n: int,
    spec: CodeSpec | None = None,
    branching: tuple[int, ...] = (3, 3, 3, 2, 2, 2),
    n_paths: int = 100,
    max_drop: int = 1,
    seed: int = 0,
) -> list[Example]:
    """Create ``n`` (text, codes) pairs.

    * A fixed, sparse set of ``n_paths`` valid paths is drawn from the full product of
      ``branching`` (independent of ``seed``, so train/val/realtime share one hierarchy).
    * Each text mentions the keyword ``l{level}k{code}`` for all but at most ``max_drop`` random
      levels, plus filler words. Missing levels must be inferred from the hierarchy.
    """
    spec = spec or CodeSpec()
    branching = tuple(list(branching)[: spec.n_levels]) + (2,) * max(
        0, spec.n_levels - len(branching)
    )
    code_values = [[f"{i + 1:02d}" for i in range(b)] for b in branching]
    all_paths = list(itertools.product(*code_values))
    paths = random.Random(12345).sample(all_paths, min(n_paths, len(all_paths)))

    rng = random.Random(seed)
    out: list[Example] = []
    for _ in range(n):
        path = rng.choice(paths)
        n_drop = rng.randint(0, min(max_drop, spec.n_levels - 1))
        dropped = set(rng.sample(range(spec.n_levels), k=n_drop))
        words = [f"l{lv + 1}k{path[lv]}" for lv in range(spec.n_levels) if lv not in dropped]
        words += rng.sample(_FILLER, k=rng.randint(2, 4))
        rng.shuffle(words)
        out.append(Example(text=" ".join(words), codes=tuple(path)))
    return out
