"""Surface features for the E2 control (TASKS P5.4).

E2 is the experiment the project's credibility rests on. Our failure categories
correlate with cheap surface properties of the item: "value absent from the
figure" is nearly the same statement as "answer string absent from the figure's
text". A probe encoding only that would score well while learning nothing about
mechanism.

So we train a classifier on these features ALONE and report the margin by which
the activation probe beats it. The margin, not the raw AUROC, is the evidence.

Features, per SPEC Section 6 E2:
    answer_in_figure   is the answer string present in the figure's text
    answer_type        numeric / boolean / categorical / other
    answer_length      characters in the answer
    question_template  which template generated the question
    figure_type        bar_chart / node_link / ...

Nothing here may touch model activations. That is the entire point.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np

_NUM = re.compile(r"^-?\d+(?:\.\d+)?$")
_BOOLS = {"yes", "no", "true", "false"}


def _normalise(text: str) -> str:
    t = str(text).strip().lower()
    t = t.rstrip("%").strip()
    if _NUM.match(t) and t.endswith(".0"):
        t = t[:-2]
    return t


def parse_number(text: str) -> float | None:
    t = _normalise(text)
    return float(t) if _NUM.match(t) else None


def answer_in_figure(answer: str, figure_text: Iterable[str],
                     rel_tol: float = 0.01) -> bool:
    """Is the answer present in the figure's own text?

    Exact match on normalised strings, plus a numeric comparison so that "45"
    matches a tick label of "45.0". The tolerance is deliberately much tighter
    than ChartQA's 5 percent scoring tolerance: this asks whether the value
    appears on the figure, not whether it is close enough to be graded correct.
    """
    target = _normalise(answer)
    if not target:
        return False

    numbers = []
    for item in figure_text:
        if _normalise(item) == target:
            return True
        n = parse_number(item)
        if n is not None:
            numbers.append(n)

    a = parse_number(answer)
    if a is None:
        return False
    return any(abs(a - n) <= rel_tol * max(abs(n), 1.0) for n in numbers)


def answer_type(answer: str) -> str:
    t = _normalise(answer)
    if t in _BOOLS:
        return "boolean"
    if parse_number(t) is not None:
        return "numeric"
    if not t:
        return "other"
    return "categorical"


@dataclass(frozen=True)
class SurfaceItem:
    """The minimum an item must expose to be featurised. No activations."""
    answer: str
    figure_text: tuple[str, ...]
    question_template: str
    figure_type: str


def featurise(items: Sequence[SurfaceItem]) -> tuple[np.ndarray, list[str]]:
    """Featurise items into a dense matrix plus column names.

    Categorical columns are one-hot encoded against the vocabulary observed in
    `items`, so fit and transform must see the same set. That is fine here: the
    baseline is refit per experiment rather than shipped.
    """
    templates = sorted({i.question_template for i in items})
    fig_types = sorted({i.figure_type for i in items})
    ans_types = ["numeric", "boolean", "categorical", "other"]

    names = (["answer_in_figure", "answer_length"]
             + [f"answer_type={t}" for t in ans_types]
             + [f"template={t}" for t in templates]
             + [f"figure_type={t}" for t in fig_types])

    rows = []
    for it in items:
        at = answer_type(it.answer)
        row = [float(answer_in_figure(it.answer, it.figure_text)),
               float(len(str(it.answer)))]
        row += [1.0 if at == t else 0.0 for t in ans_types]
        row += [1.0 if it.question_template == t else 0.0 for t in templates]
        row += [1.0 if it.figure_type == t else 0.0 for t in fig_types]
        rows.append(row)

    X = np.asarray(rows, dtype=np.float64)
    # standardise answer_length so the L2 penalty is not dominated by its scale
    col = names.index("answer_length")
    sd = X[:, col].std()
    if sd > 0:
        X[:, col] = (X[:, col] - X[:, col].mean()) / sd
    return X, names
