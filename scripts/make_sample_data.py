"""Generate synthetic train/val CSVs and a realtime JSONL for demos and smoke tests."""

from __future__ import annotations

import argparse
from pathlib import Path

from codeformer.data.codes import CodeSpec
from codeformer.data.io import write_examples_csv, write_examples_jsonl
from codeformer.data.synthetic import make_synthetic


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/sample")
    ap.add_argument("--n-train", type=int, default=4000)
    ap.add_argument("--n-val", type=int, default=400)
    ap.add_argument("--n-realtime", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    spec = CodeSpec()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    write_examples_csv(out / "train.csv", make_synthetic(a.n_train, spec, seed=a.seed), spec)
    write_examples_csv(out / "val.csv", make_synthetic(a.n_val, spec, seed=a.seed + 1), spec)
    write_examples_jsonl(
        out / "realtime.jsonl", make_synthetic(a.n_realtime, spec, seed=a.seed + 2), spec
    )
    print(f"wrote train.csv, val.csv, realtime.jsonl to {out}")


if __name__ == "__main__":
    main()
