"""ChartQA relaxed accuracy metric (SPEC Section 4.2, TASKS P1.1).

ChartQA uses "relaxed accuracy" rather than exact match:
  - Numeric answers: correct when |pred - gold| <= tolerance * |gold|.
  - String answers:  correct when lowercased, stripped strings match exactly.

Getting this wrong marks correct answers as errors, and those mislabelled items
land in the structural class and poison every probe downstream.  The tolerance
boundary is unit-tested in both directions (test_relaxed_accuracy.py).

Usage:
    >>> from src.eval.relaxed_accuracy import is_correct, relaxed_accuracy
    >>> is_correct("100", "104.9")
    True
    >>> is_correct("100", "105.1")
    False
    >>> relaxed_accuracy(["100", "yes"], ["105", "yes"])
    1.0
"""

from __future__ import annotations

import re
from typing import Sequence


# ---------------------------------------------------------------------------
# Numeric parsing
# ---------------------------------------------------------------------------

# Strips leading/trailing whitespace, optional leading currency symbol ($, €),
# commas used as thousands separators, and trailing percent sign, then attempts
# float conversion.
_STRIP_RE = re.compile(
    r"^\s*"              # leading whitespace
    r"[$€£]?\s*"         # optional currency symbol
    r"([-+]?"            # optional sign
    r"[\d,]*\.?\d+)"     # digits with optional commas and decimal
    r"\s*%?\s*$"         # optional trailing percent and whitespace
)


def _try_parse_float(s: str) -> float | None:
    """Attempt to parse *s* as a float, tolerating common formatting.

    Returns the float value on success, or ``None`` if *s* is not numeric.
    """
    m = _STRIP_RE.match(s)
    if m is None:
        return None
    try:
        return float(m.group(1).replace(",", ""))
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# Core metric
# ---------------------------------------------------------------------------

def is_correct(
    gold: str,
    prediction: str,
    tolerance: float = 0.05,
) -> bool:
    """Judge a single (gold, prediction) pair under relaxed accuracy.

    Parameters
    ----------
    gold : str
        Ground-truth answer string from ChartQA.
    prediction : str
        Model's predicted answer string.
    tolerance : float, optional
        Relative tolerance for numeric comparisons.  Default is 0.05 (5 %),
        matching ChartQA's official evaluation.

    Returns
    -------
    bool
        ``True`` if the prediction is correct under relaxed accuracy.

    Rules
    -----
    1. If *both* ``gold`` and ``prediction`` parse as numbers:
       - If ``gold == 0``:  correct iff ``prediction == 0`` exactly.
       - Otherwise:  correct iff ``|pred - gold| <= tolerance * |gold|``.
    2. Otherwise: case-insensitive, whitespace-stripped string equality.
    """
    g_num = _try_parse_float(gold)
    p_num = _try_parse_float(prediction)

    if g_num is not None and p_num is not None:
        # Both are numeric.
        if g_num == 0.0:
            return p_num == 0.0
        # Relative tolerance comparison with 1e-9 machine epsilon for floating-point rounding
        return abs(p_num - g_num) <= (tolerance * abs(g_num)) + 1e-9

    # Fall back to string comparison.
    return gold.strip().lower() == prediction.strip().lower()


def relaxed_accuracy(
    golds: Sequence[str],
    predictions: Sequence[str],
    tolerance: float = 0.05,
) -> float:
    """Compute relaxed accuracy over parallel sequences.

    Parameters
    ----------
    golds : sequence of str
        Ground-truth answers.
    predictions : sequence of str
        Model predictions, aligned with *golds*.
    tolerance : float, optional
        Passed through to :func:`is_correct`.

    Returns
    -------
    float
        Fraction of items judged correct (0.0 to 1.0).

    Raises
    ------
    ValueError
        If the two sequences differ in length.
    """
    if len(golds) != len(predictions):
        raise ValueError(
            f"Length mismatch: {len(golds)} golds vs {len(predictions)} predictions"
        )
    if len(golds) == 0:
        return 0.0
    correct = sum(
        is_correct(g, p, tolerance) for g, p in zip(golds, predictions)
    )
    return correct / len(golds)
