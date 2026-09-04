"""Manifest schema for the evaluation pipeline (TASKS P1.6).

The manifest is the single source of truth that both the labelling pipeline
(workstream B) and the activation caching pipeline (workstream A) key off.
Every row records one (figure, question) pair, its gold answer, the model's
prediction, and whether the prediction is correct under relaxed accuracy.

Schema:
    figure_id   : str   — image filename without directory (e.g. "two_col_1234.png")
    question    : str   — question text
    gold        : str   — ground-truth answer from ChartQA
    prediction  : str   — model's generated answer (empty until inference runs)
    correct     : bool  — is_correct(gold, prediction) under relaxed accuracy

File format: JSON Lines (.jsonl), one JSON object per line.

Usage:
    >>> from src.eval.manifest import ManifestRow, write_manifest, read_manifest
    >>> rows = [ManifestRow("chart.png", "What is X?", "42", "42", True)]
    >>> write_manifest(rows, "results/qwen_chartqa_test.jsonl")
    >>> loaded = read_manifest("results/qwen_chartqa_test.jsonl")
"""

from __future__ import annotations

import json
import pathlib
from dataclasses import asdict, dataclass
from typing import Sequence


@dataclass(frozen=True)
class ManifestRow:
    """One row of the evaluation manifest.

    Frozen so that rows are hashable and can be deduplicated in a set.
    """
    figure_id: str
    question: str
    gold: str
    prediction: str
    correct: bool

    def __post_init__(self):
        if not self.figure_id:
            raise ValueError("figure_id must not be empty")
        if not self.question:
            raise ValueError("question must not be empty")
        if not self.gold:
            raise ValueError("gold must not be empty")


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def write_manifest(
    rows: Sequence[ManifestRow],
    path: str | pathlib.Path,
) -> None:
    """Write manifest rows to a JSON Lines file.

    Each line is a JSON object.  The file is overwritten if it exists.
    Parent directories are created as needed.
    """
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            json.dump(asdict(row), f, ensure_ascii=False)
            f.write("\n")


def read_manifest(path: str | pathlib.Path) -> list[ManifestRow]:
    """Read a JSON Lines manifest back into a list of ManifestRow.

    Raises
    ------
    FileNotFoundError
        If *path* does not exist.
    KeyError
        If a required field is missing from a line.
    """
    path = pathlib.Path(path)
    rows: list[ManifestRow] = []
    with open(path, "r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            try:
                rows.append(ManifestRow(
                    figure_id=obj["figure_id"],
                    question=obj["question"],
                    gold=obj["gold"],
                    prediction=obj["prediction"],
                    correct=obj["correct"],
                ))
            except KeyError as exc:
                raise KeyError(
                    f"Line {lineno}: missing required field {exc}"
                ) from exc
    return rows


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_manifest(rows: Sequence[ManifestRow]) -> list[str]:
    """Run sanity checks on a manifest.  Returns a list of warning strings.

    Checks:
    - No exact duplicate rows (same figure_id + question).
    - No rows with empty prediction (unless manifest is pre-inference).
    """
    warnings: list[str] = []

    seen: set[tuple[str, str]] = set()
    for i, row in enumerate(rows):
        key = (row.figure_id, row.question)
        if key in seen:
            warnings.append(
                f"Row {i}: duplicate (figure_id={row.figure_id!r}, "
                f"question={row.question!r})"
            )
        seen.add(key)

    empty_pred = sum(1 for r in rows if not r.prediction)
    if empty_pred:
        warnings.append(
            f"{empty_pred} row(s) have empty predictions (pre-inference manifest?)"
        )

    return warnings
