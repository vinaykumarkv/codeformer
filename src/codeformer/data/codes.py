"""Hierarchical code handling: parsing "L1-L2-...-Ln" strings and valid-path tracking."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class CodeSpec:
    """Describes the structure of an output code path."""

    n_levels: int = 6
    delimiter: str = "-"

    def to_dict(self) -> dict:
        return {"n_levels": self.n_levels, "delimiter": self.delimiter}

    @classmethod
    def from_dict(cls, d: dict) -> CodeSpec:
        return cls(n_levels=int(d["n_levels"]), delimiter=str(d["delimiter"]))


def parse_codes(value: str | Sequence[str], spec: CodeSpec) -> tuple[str, ...]:
    """Parse a delimited string or a list into a validated tuple of ``n_levels`` codes."""
    if isinstance(value, str):
        parts = [p.strip() for p in value.split(spec.delimiter)]
    else:
        parts = [str(p).strip() for p in value]
    if len(parts) != spec.n_levels:
        raise ValueError(f"Expected {spec.n_levels} code levels, got {len(parts)}: {value!r}")
    if any(not p for p in parts):
        raise ValueError(f"Empty code level in {value!r}")
    return tuple(parts)


def format_codes(codes: Iterable[str], spec: CodeSpec) -> str:
    return spec.delimiter.join(codes)


class CodeHierarchy:
    """Prefix tree over observed code paths.

    Used at inference time to guarantee that predicted paths exist, i.e. that the
    L3 code is a real child of the predicted (L1, L2).
    """

    def __init__(self, n_levels: int) -> None:
        self.n_levels = n_levels
        self._children: dict[tuple[str, ...], set[str]] = defaultdict(set)
        self._paths: set[tuple[str, ...]] = set()

    def __len__(self) -> int:
        return len(self._paths)

    def __contains__(self, path: object) -> bool:
        return path in self._paths

    def add(self, path: Sequence[str]) -> None:
        path = tuple(path)
        if len(path) != self.n_levels:
            raise ValueError(f"Path must have {self.n_levels} levels, got {len(path)}")
        self._paths.add(path)
        for i in range(self.n_levels):
            self._children[path[:i]].add(path[i])

    def add_many(self, paths: Iterable[Sequence[str]]) -> None:
        for p in paths:
            self.add(p)

    def children(self, prefix: Sequence[str]) -> set[str]:
        return self._children.get(tuple(prefix), set())

    def to_list(self) -> list[list[str]]:
        return [list(p) for p in sorted(self._paths)]

    @classmethod
    def from_list(cls, n_levels: int, paths: Iterable[Sequence[str]]) -> CodeHierarchy:
        h = cls(n_levels)
        h.add_many(paths)
        return h
