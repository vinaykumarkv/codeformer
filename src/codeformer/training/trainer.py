"""Training loop: AdamW, warmup + cosine schedule, grad clipping, early stopping."""

from __future__ import annotations

import logging
import math
import random
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from codeformer.config import TrainConfig
from codeformer.data.dataset import make_collate_fn
from codeformer.models.transformer import CodeTransformer

logger = logging.getLogger(__name__)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_device(device: str | None = None) -> torch.device:
    if device:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass
class TrainResult:
    train_losses: list[float] = field(default_factory=list)
    val_losses: list[float] = field(default_factory=list)
    best_val_loss: float = math.inf
    best_epoch: int = -1
    steps: int = 0


class Trainer:
    def __init__(
        self,
        model: CodeTransformer,
        cfg: TrainConfig,
        pad_id: int,
        device: str | torch.device | None = None,
        on_best: Callable[[CodeTransformer], None] | None = None,
    ) -> None:
        self.model = model
        self.cfg = cfg
        self.pad_id = pad_id
        self.device = resolve_device(str(device) if device is not None else None)
        self.on_best = on_best
        self.model.to(self.device)

    def _loader(self, ds: Dataset, shuffle: bool) -> DataLoader:
        return DataLoader(
            ds,
            batch_size=self.cfg.batch_size,
            shuffle=shuffle,
            num_workers=self.cfg.num_workers,
            collate_fn=make_collate_fn(self.pad_id),
            drop_last=False,
        )

    def _to_device(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return {k: v.to(self.device) for k, v in batch.items()}

    @torch.no_grad()
    def evaluate(self, ds: Dataset) -> float:
        self.model.eval()
        total, n = 0.0, 0
        for batch in self._loader(ds, shuffle=False):
            batch = self._to_device(batch)
            out = self.model(**batch)
            bs = batch["input_ids"].size(0)
            total += out.loss.item() * bs
            n += bs
        return total / max(n, 1)

    def fit(self, train_ds: Dataset, val_ds: Dataset | None = None) -> TrainResult:
        set_seed(self.cfg.seed)
        loader = self._loader(train_ds, shuffle=True)
        total_steps = max(1, self.cfg.epochs * len(loader))
        warmup = max(1, int(self.cfg.warmup_ratio * total_steps))

        decay, no_decay = [], []
        for n, p in self.model.named_parameters():
            (no_decay if p.ndim < 2 or "ln" in n else decay).append(p)
        opt = torch.optim.AdamW(
            [
                {"params": decay, "weight_decay": self.cfg.weight_decay},
                {"params": no_decay, "weight_decay": 0.0},
            ],
            lr=self.cfg.lr,
            betas=(0.9, 0.95),
        )

        def lr_lambda(step: int) -> float:
            if step < warmup:
                return (step + 1) / warmup
            progress = (step - warmup) / max(1, total_steps - warmup)
            return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress))

        sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
        use_amp = self.cfg.mixed_precision and self.device.type == "cuda"
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

        result = TrainResult()
        bad_epochs = 0
        for epoch in range(self.cfg.epochs):
            self.model.train()
            running, count = 0.0, 0
            for batch in loader:
                batch = self._to_device(batch)
                with torch.autocast(self.device.type, enabled=use_amp):
                    out = self.model(**batch)
                opt.zero_grad(set_to_none=True)
                scaler.scale(out.loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)
                scaler.step(opt)
                scaler.update()
                sched.step()
                result.steps += 1
                running += out.loss.item()
                count += 1
                if result.steps % self.cfg.log_every == 0:
                    logger.info("step %d loss %.4f", result.steps, running / count)
            train_loss = running / max(count, 1)
            result.train_losses.append(train_loss)

            if val_ds is not None and len(val_ds) > 0:
                val_loss = self.evaluate(val_ds)
                result.val_losses.append(val_loss)
                logger.info("epoch %d train %.4f val %.4f", epoch + 1, train_loss, val_loss)
                if val_loss < result.best_val_loss - 1e-4:
                    result.best_val_loss, result.best_epoch, bad_epochs = val_loss, epoch, 0
                    if self.on_best:
                        self.on_best(self.model)
                else:
                    bad_epochs += 1
                    if 0 < self.cfg.early_stopping_patience <= bad_epochs:
                        logger.info("early stopping at epoch %d", epoch + 1)
                        break
            else:
                logger.info("epoch %d train %.4f", epoch + 1, train_loss)
                result.best_epoch = epoch
        return result
