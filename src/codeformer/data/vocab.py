"""Unified vocabulary: special tokens + text tokens + level-tagged code tokens.

Code tokens are named ``L{level}:{code}`` (level is 1-based) so the same raw code
value used at two different levels maps to two different ids. Text tokens never
contain ``:`` (the tokenizer splits on it), so the namespaces cannot collide.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable

PAD, UNK, BOS, SEP, EOS = "<pad>", "<unk>", "<bos>", "<sep>", "<eos>"
SPECIALS = (PAD, UNK, BOS, SEP, EOS)
_CODE_RE = re.compile(r"^L(\d+):")


def code_token(level: int, code: str) -> str:
    """``level`` is 0-based here; token names are 1-based (L1..Ln)."""
    return f"L{level + 1}:{code}"


class Vocab:
    def __init__(self, tokens: Iterable[str] | None = None) -> None:
        self._id_to_token: list[str] = []
        self._token_to_id: dict[str, int] = {}
        for t in SPECIALS:
            self.add(t)
        for t in tokens or []:
            self.add(t)

    def __len__(self) -> int:
        return len(self._id_to_token)

    def __contains__(self, token: str) -> bool:
        return token in self._token_to_id

    def add(self, token: str) -> int:
        if token not in self._token_to_id:
            self._token_to_id[token] = len(self._id_to_token)
            self._id_to_token.append(token)
        return self._token_to_id[token]

    def id(self, token: str) -> int:
        return self._token_to_id.get(token, self._token_to_id[UNK])

    def token(self, idx: int) -> str:
        return self._id_to_token[idx]

    @property
    def pad_id(self) -> int:
        return self._token_to_id[PAD]

    @property
    def unk_id(self) -> int:
        return self._token_to_id[UNK]

    @property
    def bos_id(self) -> int:
        return self._token_to_id[BOS]

    @property
    def sep_id(self) -> int:
        return self._token_to_id[SEP]

    @property
    def eos_id(self) -> int:
        return self._token_to_id[EOS]

    # -- code helpers ------------------------------------------------------
    def code_id(self, level: int, code: str) -> int | None:
        """Return the id for a code at a 0-based level, or None if unseen."""
        return self._token_to_id.get(code_token(level, code))

    def code_ids_for_level(self, level: int) -> list[int]:
        prefix = f"L{level + 1}:"
        return [i for i, t in enumerate(self._id_to_token) if t.startswith(prefix)]

    def code_of(self, idx: int) -> tuple[int, str] | None:
        """Return (0-based level, raw code) for a code token id, else None."""
        tok = self._id_to_token[idx]
        m = _CODE_RE.match(tok)
        if not m:
            return None
        return int(m.group(1)) - 1, tok[m.end() :]

    # -- persistence -------------------------------------------------------
    def to_list(self) -> list[str]:
        return list(self._id_to_token)

    @classmethod
    def from_list(cls, tokens: list[str]) -> Vocab:
        if tuple(tokens[: len(SPECIALS)]) != SPECIALS:
            raise ValueError("Corrupt vocab: special tokens missing or reordered")
        v = cls()
        for t in tokens[len(SPECIALS) :]:
            v.add(t)
        return v

    # -- construction ------------------------------------------------------
    def extend(
        self,
        token_lists: Iterable[list[str]],
        code_paths: Iterable[tuple[str, ...]],
        min_freq: int = 1,
    ) -> int:
        """Add unseen text tokens (freq >= min_freq) and code tokens. Returns #added."""
        before = len(self)
        counts: Counter[str] = Counter()
        for toks in token_lists:
            counts.update(toks)
        for tok, c in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])):
            if c >= min_freq:
                self.add(tok)
        for path in code_paths:
            for level, code in enumerate(path):
                self.add(code_token(level, code))
        return len(self) - before
