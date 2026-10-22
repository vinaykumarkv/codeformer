"""Inference: constrained beam search and single-pass parallel decoding.

Two decoding modes
------------------
``autoregressive``  Head 0 only. Each level is generated conditioned on the previous
                    ones; beam search returns the top-k *complete paths* ranked by joint
                    probability.
``parallel``        All levels from ONE forward pass using the multi-token heads at the
                    ``<sep>`` position (needs ``n_future >= n_levels``). Fastest; levels
                    are predicted jointly-but-independently, then repaired greedily
                    against the hierarchy.

Both modes can restrict each step to (a) tokens of the right level, and (b) children
of the already-chosen prefix, so the output is always a valid path.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import torch

from codeformer.data.codes import CodeHierarchy, CodeSpec, format_codes
from codeformer.data.dataset import encode_prompt
from codeformer.data.tokenizer import SimpleTokenizer
from codeformer.data.vocab import Vocab
from codeformer.models.transformer import CodeTransformer
from codeformer.training.checkpoint import load_checkpoint

MODES = ("autoregressive", "parallel")


@dataclass(frozen=True)
class Prediction:
    codes: tuple[str, ...]
    score: float  # joint probability of the path under the model (restricted vocabulary)
    level_probs: tuple[float, ...]

    def as_string(self, spec: CodeSpec) -> str:
        return format_codes(self.codes, spec)


class Predictor:
    def __init__(
        self,
        model: CodeTransformer,
        vocab: Vocab,
        hierarchy: CodeHierarchy,
        spec: CodeSpec,
        tokenizer: SimpleTokenizer | None = None,
        device: str | torch.device = "cpu",
    ) -> None:
        self.model = model.to(device).eval()
        self.vocab = vocab
        self.hierarchy = hierarchy
        self.spec = spec
        self.tokenizer = tokenizer or SimpleTokenizer()
        self.device = torch.device(device)
        self._level_ids = [
            torch.tensor(vocab.code_ids_for_level(lv), dtype=torch.long)
            for lv in range(spec.n_levels)
        ]
        self._allowed_cache: dict[tuple[str, ...], torch.Tensor] = {}

    @classmethod
    def from_checkpoint(cls, path: str | Path, device: str = "cpu") -> Predictor:
        ck = load_checkpoint(path, device)
        return cls(
            ck["model"],
            ck["vocab"],
            ck["hierarchy"],
            ck["spec"],
            SimpleTokenizer(ck["data_cfg"].lowercase),
            device,
        )

    # ------------------------------------------------------------------
    def _prompt(self, text: str) -> list[int]:
        return encode_prompt(
            text, self.vocab, self.tokenizer, self.model.cfg.max_seq_len, self.spec.n_levels
        )

    def _allowed(self, prefix: tuple[str, ...], use_hierarchy: bool) -> torch.Tensor:
        """Token ids permitted at level ``len(prefix)`` after choosing ``prefix``."""
        level = len(prefix)
        if use_hierarchy:
            if prefix not in self._allowed_cache:
                ids = [
                    i
                    for c in sorted(self.hierarchy.children(prefix))
                    if (i := self.vocab.code_id(level, c)) is not None
                ]
                self._allowed_cache[prefix] = torch.tensor(ids, dtype=torch.long)
            ids_t = self._allowed_cache[prefix]
            if len(ids_t):
                return ids_t
        return self._level_ids[level]

    def _code(self, token_id: int) -> str:
        info = self.vocab.code_of(token_id)
        assert info is not None
        return info[1]

    # ------------------------------------------------------------------
    @torch.inference_mode()
    def predict(
        self,
        texts: Sequence[str] | str,
        top_k: int = 1,
        mode: str = "autoregressive",
        beam_width: int | None = None,
        use_hierarchy: bool = True,
    ) -> list[list[Prediction]]:
        """Return, for each text, up to ``top_k`` predictions sorted by score."""
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        if top_k < 1:
            raise ValueError("top_k must be >= 1")
        if isinstance(texts, str):
            texts = [texts]
        if mode == "parallel":
            return self._predict_parallel(texts, use_hierarchy)
        width = max(beam_width or top_k, top_k)
        return [self._beam_search(self._prompt(t), width, top_k, use_hierarchy) for t in texts]

    def _beam_search(
        self, prompt: list[int], width: int, top_k: int, use_hierarchy: bool
    ) -> list[Prediction]:
        # beam: (cumulative logprob, codes, per-level logprobs)
        beams: list[tuple[float, tuple[str, ...], tuple[float, ...]]] = [(0.0, (), ())]
        for _ in range(self.spec.n_levels):
            # every beam has the same number of codes, so rows share one length (no padding)
            ids = torch.tensor(
                [
                    prompt + [self.vocab.code_id(i, c) for i, c in enumerate(codes)]
                    for _, codes, _ in beams
                ],
                device=self.device,
            )
            logits = self.model(ids).logits[0][:, -1, :].float()
            logp = torch.log_softmax(logits, dim=-1).cpu()
            cands = []
            for bi, (score, codes, lps) in enumerate(beams):
                allowed = self._allowed(codes, use_hierarchy)
                # renormalise over the allowed set so scores are comparable probabilities
                sub = torch.log_softmax(logp[bi, allowed], dim=-1)
                k = min(width, len(allowed))
                vals, idx = sub.topk(k)
                for v, i in zip(vals.tolist(), idx.tolist(), strict=True):
                    code = self._code(int(allowed[i]))
                    cands.append((score + v, (*codes, code), (*lps, v)))
            cands.sort(key=lambda c: c[0], reverse=True)
            beams = cands[:width]
        return [
            Prediction(codes, math.exp(s), tuple(math.exp(x) for x in lps))
            for s, codes, lps in beams[:top_k]
        ]

    def _predict_parallel(
        self, texts: Sequence[str], use_hierarchy: bool
    ) -> list[list[Prediction]]:
        n = self.spec.n_levels
        if self.model.n_future < n:
            raise ValueError(
                f"parallel mode needs n_future >= n_levels ({self.model.n_future} < {n})"
            )
        prompts = [self._prompt(t) for t in texts]
        width = max(len(p) for p in prompts)
        pad = self.vocab.pad_id
        ids = torch.tensor([[pad] * (width - len(p)) + p for p in prompts], device=self.device)
        mask = torch.tensor(
            [[0] * (width - len(p)) + [1] * len(p) for p in prompts], device=self.device
        )
        h = self.model.hidden(ids, mask)[:, -1:, :]  # <sep> state, left-padded so index -1
        per_level = [
            torch.log_softmax(self.model.head_logits(h, j)[:, 0, :].float(), dim=-1).cpu()
            for j in range(n)
        ]
        out: list[list[Prediction]] = []
        for b in range(len(texts)):
            codes: list[str] = []
            lps: list[float] = []
            for level in range(n):
                allowed = self._allowed(tuple(codes), use_hierarchy)
                sub = torch.log_softmax(per_level[level][b, allowed], dim=-1)
                v, i = sub.max(dim=0)
                codes.append(self._code(int(allowed[int(i)])))
                lps.append(float(v))
            out.append(
                [Prediction(tuple(codes), math.exp(sum(lps)), tuple(math.exp(x) for x in lps))]
            )
        return out
