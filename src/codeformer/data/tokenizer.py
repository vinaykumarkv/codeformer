"""Minimal dependency-free text tokenizer.

Swap this out for a subword tokenizer (e.g. HF ``tokenizers``) by implementing the
same single method: ``tokenize(text) -> list[str]``.
"""

from __future__ import annotations

import re

_TOKEN_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)


class SimpleTokenizer:
    def __init__(self, lowercase: bool = True) -> None:
        self.lowercase = lowercase

    def tokenize(self, text: str) -> list[str]:
        if self.lowercase:
            text = text.lower()
        return _TOKEN_RE.findall(text)
