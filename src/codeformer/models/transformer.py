"""GPT-style decoder-only transformer with multi-token prediction heads.

Head ``j`` (0-based) reads the hidden state at position ``t`` and predicts the token
at position ``t + 1 + j``. Head 0 is the standard next-token head; heads 1..k-1 are
auxiliary heads. With ``n_future >= n_levels`` the hidden state at ``<sep>`` yields
all L1..Ln codes in a single forward pass ("parallel decoding"), while head 0 alone
supports classic constrained autoregressive decoding/beam search.

All heads share the (tied) input-embedding matrix as output projection.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from codeformer.config import ModelConfig

IGNORE_INDEX = -100


@dataclass
class ModelOutput:
    logits: list[torch.Tensor]  # n_future tensors of shape (B, T, V)
    loss: torch.Tensor | None = None
    head_losses: list[torch.Tensor] | None = None


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.n_heads = cfg.n_heads
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model)
        self.dropout = cfg.dropout

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        b, t, c = x.shape
        q, k, v = self.qkv(x).split(c, dim=-1)
        q, k, v = (z.view(b, t, self.n_heads, c // self.n_heads).transpose(1, 2) for z in (q, k, v))
        y = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, dropout_p=self.dropout if self.training else 0.0
        )
        return self.proj(y.transpose(1, 2).contiguous().view(b, t, c))


class MLP(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.fc1 = nn.Linear(cfg.d_model, cfg.d_ff)
        self.fc2 = nn.Linear(cfg.d_ff, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.fc2(F.gelu(self.fc1(x))))


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.mlp = MLP(cfg)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        x = x + self.drop(self.attn(self.ln1(x), attn_mask))
        return x + self.mlp(self.ln2(x))


class HeadAdapter(nn.Module):
    """Residual MLP that specialises the shared trunk for a look-ahead offset."""

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.ln = nn.LayerNorm(cfg.d_model)
        self.mlp = MLP(cfg)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return h + self.mlp(self.ln(h))


class CodeTransformer(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        if cfg.vocab_size <= 0:
            raise ValueError("ModelConfig.vocab_size must be set before building the model")
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.max_seq_len, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layers))
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.adapters = nn.ModuleList(HeadAdapter(cfg) for _ in range(cfg.n_future - 1))
        self.apply(self._init_weights)
        for name, p in self.named_parameters():
            if name.endswith("fc2.weight") or name.endswith("proj.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * cfg.n_layers))

    @staticmethod
    def _init_weights(m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)

    @property
    def n_future(self) -> int:
        return self.cfg.n_future

    # ------------------------------------------------------------------
    def hidden(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        b, t = input_ids.shape
        if t > self.cfg.max_seq_len:
            raise ValueError(f"Sequence length {t} exceeds max_seq_len {self.cfg.max_seq_len}")
        device = input_ids.device
        if attention_mask is None:
            key_mask = torch.ones((b, t), dtype=torch.bool, device=device)
            pos = torch.arange(t, device=device).unsqueeze(0).expand(b, t)
        else:
            key_mask = attention_mask.bool()
            # Positions count real tokens only, so left- and right-padding behave alike.
            pos = (key_mask.long().cumsum(-1) - 1).clamp(min=0)
        causal = torch.tril(torch.ones((t, t), dtype=torch.bool, device=device))
        allowed = causal[None] & key_mask[:, None, :]
        # Let every position attend to itself so padded rows never softmax over nothing.
        allowed = (allowed | torch.eye(t, dtype=torch.bool, device=device)[None]).unsqueeze(1)

        x = self.drop(self.tok_emb(input_ids) + self.pos_emb(pos))
        for block in self.blocks:
            x = block(x, allowed)
        return self.ln_f(x)

    def head_logits(self, h: torch.Tensor, head: int) -> torch.Tensor:
        if not 0 <= head < self.n_future:
            raise IndexError(f"head {head} out of range for n_future={self.n_future}")
        if head > 0:
            h = self.adapters[head - 1](h)
        return F.linear(h, self.tok_emb.weight)  # tied output projection

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
    ) -> ModelOutput:
        h = self.hidden(input_ids, attention_mask)
        logits = [self.head_logits(h, j) for j in range(self.n_future)]
        if labels is None:
            return ModelOutput(logits=logits)
        loss, head_losses = self.multi_token_loss(logits, labels)
        return ModelOutput(logits=logits, loss=loss, head_losses=head_losses)

    def multi_token_loss(
        self, logits: list[torch.Tensor], labels: torch.Tensor
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """Weighted mean of per-head cross entropies. ``labels`` are *unshifted*."""
        head_losses: list[torch.Tensor] = []
        weights: list[float] = []
        t = labels.shape[1]
        for j, lg in enumerate(logits):
            shift = 1 + j
            if t <= shift:
                continue
            tgt = labels[:, shift:].reshape(-1)
            pred = lg[:, :-shift].reshape(-1, lg.size(-1)).float()
            n = (tgt != IGNORE_INDEX).sum().clamp(min=1)
            head_losses.append(
                F.cross_entropy(pred, tgt, ignore_index=IGNORE_INDEX, reduction="sum") / n
            )
            weights.append(1.0 if j == 0 else self.cfg.aux_loss_weight)
        w = torch.tensor(weights, device=labels.device)
        total = sum(wi * li for wi, li in zip(w, head_losses, strict=True)) / w.sum()
        return total, head_losses

    # ------------------------------------------------------------------
    def resize_vocab(self, new_size: int) -> None:
        """Grow the embedding/output matrix, preserving existing rows."""
        old = self.tok_emb
        if new_size < old.num_embeddings:
            raise ValueError("Shrinking the vocabulary is not supported")
        if new_size == old.num_embeddings:
            return
        new = nn.Embedding(new_size, old.embedding_dim).to(old.weight.device, old.weight.dtype)
        nn.init.normal_(new.weight, std=0.02)
        with torch.no_grad():
            new.weight[: old.num_embeddings] = old.weight
        self.tok_emb = new
        self.cfg.vocab_size = new_size

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())
