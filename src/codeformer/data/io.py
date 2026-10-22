"""Reading training / realtime data from CSV or JSONL.

Accepted column names (first match wins):
  * text  : ``text``, ``predictor_text``, ``input``
  * codes : ``codes``, ``output_text``, ``target``, ``code``

The codes value can be a delimited string (``"IN-MH-MUM-400-01-A"``) or, in JSONL,
a list of strings.
"""

from __future__ import annotations

import csv
import json
import logging
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

from codeformer.data.codes import CodeSpec, parse_codes

logger = logging.getLogger(__name__)

TEXT_COLUMNS = ("text", "predictor_text", "input")
CODE_COLUMNS = ("codes", "output_text", "target", "code")


@dataclass(frozen=True)
class Example:
    text: str
    codes: tuple[str, ...]


def _pick(row: dict, candidates: tuple[str, ...]):
    for c in candidates:
        if c in row and row[c] not in (None, ""):
            return row[c]
    raise KeyError(f"None of the columns {candidates} found/non-empty in row keys {list(row)}")


def _iter_rows(path: Path) -> Iterator[dict]:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        with open(path, newline="", encoding="utf-8") as fh:
            yield from csv.DictReader(fh)
    elif suffix in (".jsonl", ".ndjson"):
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    yield json.loads(line)
    else:
        raise ValueError(f"Unsupported data format {suffix!r}; use .csv or .jsonl")


def load_examples(path: str | Path, spec: CodeSpec, strict: bool = False) -> list[Example]:
    """Load examples; malformed rows are skipped (and counted) unless ``strict``."""
    path = Path(path)
    out: list[Example] = []
    skipped = 0
    for i, row in enumerate(_iter_rows(path)):
        try:
            text = str(_pick(row, TEXT_COLUMNS)).strip()
            codes = parse_codes(_pick(row, CODE_COLUMNS), spec)
        except (KeyError, ValueError) as exc:
            if strict:
                raise ValueError(f"{path}: row {i}: {exc}") from exc
            skipped += 1
            continue
        out.append(Example(text=text, codes=codes))
    if skipped:
        logger.warning("Skipped %d malformed rows in %s", skipped, path)
    return out


def write_examples_csv(path: str | Path, examples: list[Example], spec: CodeSpec) -> None:
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["text", "codes"])
        for ex in examples:
            w.writerow([ex.text, spec.delimiter.join(ex.codes)])


def write_examples_jsonl(path: str | Path, examples: list[Example], spec: CodeSpec) -> None:
    """Realtime schema: ``predictor_text`` / ``output_text``."""
    with open(path, "w", encoding="utf-8") as fh:
        for ex in examples:
            row = {"predictor_text": ex.text, "output_text": spec.delimiter.join(ex.codes)}
            fh.write(json.dumps(row) + "\n")
