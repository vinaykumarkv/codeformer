"""High-level training entry points: from-scratch training and incremental updates."""

from __future__ import annotations

import logging
import random
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path

from codeformer.config import AppConfig, TrainConfig
from codeformer.data.codes import CodeHierarchy, CodeSpec
from codeformer.data.dataset import CodeDataset
from codeformer.data.io import Example
from codeformer.data.tokenizer import SimpleTokenizer
from codeformer.data.vocab import Vocab
from codeformer.models.transformer import CodeTransformer
from codeformer.training.checkpoint import load_checkpoint, save_checkpoint
from codeformer.training.trainer import Trainer, TrainResult

logger = logging.getLogger(__name__)


def filter_known(examples: Sequence[Example], vocab: Vocab) -> list[Example]:
    """Drop examples containing codes the vocabulary has never seen (cannot be scored)."""
    kept = [
        ex
        for ex in examples
        if all(vocab.code_id(lv, c) is not None for lv, c in enumerate(ex.codes))
    ]
    if len(kept) < len(examples):
        logger.warning("Dropped %d examples with unseen codes", len(examples) - len(kept))
    return kept


def train_model(
    train_examples: Sequence[Example],
    val_examples: Sequence[Example] | None,
    cfg: AppConfig,
    out_path: str | Path,
    device: str | None = None,
) -> TrainResult:
    """Train from scratch; ``out_path`` always holds the best (or final) checkpoint."""
    if not train_examples:
        raise ValueError("No training examples")
    spec = CodeSpec(cfg.data.n_levels, cfg.data.delimiter)
    tokenizer = SimpleTokenizer(cfg.data.lowercase)

    vocab = Vocab()
    vocab.extend(
        (tokenizer.tokenize(e.text) for e in train_examples),
        (e.codes for e in train_examples),
        min_freq=cfg.data.min_freq,
    )
    hierarchy = CodeHierarchy(spec.n_levels)
    hierarchy.add_many(e.codes for e in train_examples)

    model_cfg = replace(cfg.model, vocab_size=len(vocab))
    model = CodeTransformer(model_cfg)
    logger.info(
        "Model: %.2fM params, vocab=%d, paths=%d",
        model.num_parameters() / 1e6,
        len(vocab),
        len(hierarchy),
    )

    train_ds = CodeDataset(train_examples, vocab, tokenizer, model_cfg.max_seq_len, spec.n_levels)
    val_ds = None
    if val_examples:
        val_ds = CodeDataset(
            filter_known(val_examples, vocab),
            vocab,
            tokenizer,
            model_cfg.max_seq_len,
            spec.n_levels,
        )

    def save(m: CodeTransformer) -> None:
        save_checkpoint(
            out_path,
            m,
            vocab,
            hierarchy,
            spec,
            cfg.data,
            extra={"train_config": cfg.train.__dict__},
        )

    trainer = Trainer(model, cfg.train, vocab.pad_id, device, on_best=save)
    result = trainer.fit(train_ds, val_ds)
    if val_ds is None or len(val_ds) == 0:
        save(model)
    return result


def incremental_update(
    checkpoint_path: str | Path,
    new_examples: Sequence[Example],
    out_path: str | Path,
    replay_examples: Sequence[Example] | None = None,
    replay_ratio: float = 1.0,
    epochs: int = 3,
    lr: float = 1e-4,
    batch_size: int = 32,
    seed: int = 0,
    device: str | None = None,
) -> TrainResult:
    """Continue training on fresh (predictor_text, output_text) pairs.

    * New text tokens / codes / paths are added; the embedding matrix grows in place.
    * ``replay_examples`` (a sample of old data, ``replay_ratio`` x len(new)) is mixed in
      to limit catastrophic forgetting.
    """
    if not new_examples:
        raise ValueError("No new examples")
    ck = load_checkpoint(checkpoint_path)
    model: CodeTransformer = ck["model"]
    vocab: Vocab = ck["vocab"]
    hierarchy: CodeHierarchy = ck["hierarchy"]
    spec: CodeSpec = ck["spec"]
    data_cfg = ck["data_cfg"]
    tokenizer = SimpleTokenizer(data_cfg.lowercase)

    added = vocab.extend(
        (tokenizer.tokenize(e.text) for e in new_examples),
        (e.codes for e in new_examples),
        min_freq=1,
    )
    model.resize_vocab(len(vocab))
    hierarchy.add_many(e.codes for e in new_examples)
    logger.info("Added %d vocab entries; vocab now %d", added, len(vocab))

    mixed = list(new_examples)
    if replay_examples and replay_ratio > 0:
        rng = random.Random(seed)
        k = min(len(replay_examples), int(len(new_examples) * replay_ratio))
        mixed += rng.sample(list(replay_examples), k)

    ds = CodeDataset(
        filter_known(mixed, vocab), vocab, tokenizer, model.cfg.max_seq_len, spec.n_levels
    )
    tcfg = TrainConfig(
        epochs=epochs,
        lr=lr,
        batch_size=batch_size,
        seed=seed,
        warmup_ratio=0.1,
        early_stopping_patience=0,
    )
    result = Trainer(model, tcfg, vocab.pad_id, device).fit(ds, None)
    save_checkpoint(
        out_path,
        model,
        vocab,
        hierarchy,
        spec,
        data_cfg,
        extra={"incremental_from": str(checkpoint_path)},
    )
    return result
