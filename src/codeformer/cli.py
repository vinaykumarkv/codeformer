"""Command line interface: train | finetune | predict | evaluate | serve."""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys

from codeformer.config import load_config
from codeformer.data.codes import CodeSpec
from codeformer.data.io import load_examples


def _cmd_train(a: argparse.Namespace) -> int:
    from codeformer.training.pipeline import train_model

    cfg = load_config(a.config)
    spec = CodeSpec(cfg.data.n_levels, cfg.data.delimiter)
    train = load_examples(a.train, spec)
    if a.val:
        val = load_examples(a.val, spec)
    else:
        rng = random.Random(cfg.train.seed)
        rng.shuffle(train)
        cut = int(len(train) * a.val_split)
        val, train = train[:cut], train[cut:]
    res = train_model(train, val, cfg, a.out, device=a.device)
    print(
        json.dumps(
            {
                "best_val_loss": res.best_val_loss,
                "best_epoch": res.best_epoch,
                "steps": res.steps,
                "checkpoint": str(a.out),
            }
        )
    )
    return 0


def _cmd_finetune(a: argparse.Namespace) -> int:
    from codeformer.training.checkpoint import load_checkpoint
    from codeformer.training.pipeline import incremental_update

    spec = load_checkpoint(a.checkpoint)["spec"]
    new = load_examples(a.data, spec)
    replay = load_examples(a.replay, spec) if a.replay else None
    res = incremental_update(
        a.checkpoint,
        new,
        a.out,
        replay,
        a.replay_ratio,
        a.epochs,
        a.lr,
        a.batch_size,
        device=a.device,
    )
    print(
        json.dumps(
            {"steps": res.steps, "final_loss": res.train_losses[-1], "checkpoint": str(a.out)}
        )
    )
    return 0


def _cmd_predict(a: argparse.Namespace) -> int:
    from codeformer.inference.predictor import Predictor

    p = Predictor.from_checkpoint(a.checkpoint, a.device or "cpu")
    out = p.predict(a.text, top_k=a.top_k, mode=a.mode)
    for pr in out[0]:
        print(f"{pr.as_string(p.spec)}\t{pr.score:.4f}")
    return 0


def _cmd_evaluate(a: argparse.Namespace) -> int:
    from codeformer.evaluation import evaluate_predictor
    from codeformer.inference.predictor import Predictor

    p = Predictor.from_checkpoint(a.checkpoint, a.device or "cpu")
    ex = load_examples(a.data, p.spec)
    print(json.dumps(evaluate_predictor(p, ex, mode=a.mode, top_k=a.top_k), indent=2))
    return 0


def _cmd_serve(a: argparse.Namespace) -> int:
    import uvicorn

    from codeformer.api.app import create_app

    uvicorn.run(create_app(a.checkpoint), host=a.host, port=a.port)
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="codeformer", description=__doc__)
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="command", required=True)

    t = sub.add_parser("train", help="train from scratch")
    t.add_argument("--config", default=None)
    t.add_argument("--train", required=True)
    t.add_argument("--val", default=None)
    t.add_argument("--val-split", type=float, default=0.1)
    t.add_argument("--out", required=True)
    t.add_argument("--device", default=None)
    t.set_defaults(fn=_cmd_train)

    f = sub.add_parser("finetune", help="incremental update on realtime data")
    f.add_argument("--checkpoint", required=True)
    f.add_argument("--data", required=True, help="CSV/JSONL with predictor_text/output_text")
    f.add_argument("--replay", default=None, help="old training data to mix in")
    f.add_argument("--replay-ratio", type=float, default=1.0)
    f.add_argument("--epochs", type=int, default=3)
    f.add_argument("--lr", type=float, default=1e-4)
    f.add_argument("--batch-size", type=int, default=32)
    f.add_argument("--out", required=True)
    f.add_argument("--device", default=None)
    f.set_defaults(fn=_cmd_finetune)

    r = sub.add_parser("predict", help="predict codes for one text")
    r.add_argument("--checkpoint", required=True)
    r.add_argument("--text", required=True)
    r.add_argument("--top-k", type=int, default=1)
    r.add_argument("--mode", choices=["autoregressive", "parallel"], default="autoregressive")
    r.add_argument("--device", default=None)
    r.set_defaults(fn=_cmd_predict)

    e = sub.add_parser("evaluate", help="accuracy on a labelled file")
    e.add_argument("--checkpoint", required=True)
    e.add_argument("--data", required=True)
    e.add_argument("--top-k", type=int, default=3)
    e.add_argument("--mode", choices=["autoregressive", "parallel"], default="autoregressive")
    e.add_argument("--device", default=None)
    e.set_defaults(fn=_cmd_evaluate)

    s = sub.add_parser("serve", help="run the REST API")
    s.add_argument("--checkpoint", default=None)
    s.add_argument("--host", default="0.0.0.0")  # noqa: S104
    s.add_argument("--port", type=int, default=8000)
    s.set_defaults(fn=_cmd_serve)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
