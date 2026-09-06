"""E4 recovery matrix and its pre-registered test (TASKS P6.6, P6.7).

The matrix is failure type x intervention, each cell holding
`P(correct | type, intervention)`. SPEC asks whether it has "diagonal
structure". That phrase needs sharpening before there is data to look at,
because the matrix is 2x4 rather than square and `I_abstain` is scored as risk
avoided rather than accuracy gained.

What we actually care about is an **interaction**: does the right repair depend
on which failure occurred? Two claims, of different strength:

  A. The effect of an intervention differs by failure type.
  B. The *best* intervention differs by failure type.

B is the one the project needs. A type-aware controller earns nothing over a
single fixed policy unless the argmax actually moves. A is a precondition for B,
and is what we can test with reasonable power.

**Pre-registered primary test.** A difference in differences, directional:

    delta_crop = [P(ok | structural, I_crop)   - P(ok | structural, I_0)]
               - [P(ok | fabrication, I_crop)  - P(ok | fabrication, I_0)]

Prediction: `delta_crop > 0`. Cropping should rescue structural misreadings,
where the answer was on the figure all along, more than it rescues fabrications,
where it was never there. Inference is a cluster bootstrap over figures.

`I_verify` gets the same contrast as a secondary test. `I_abstain` is excluded
from all of these on purpose: abstention does not produce a correct answer, so
its value is not a recovery rate, and it belongs to E5's risk-coverage analysis
instead. Folding it in here would compare quantities with different units.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from src.common.linear import cluster_bootstrap_ci

STRUCTURAL, FABRICATION = "structural", "fabrication"
BASELINE = "I_0"
ABSTAIN = "I_abstain"


@dataclass(frozen=True)
class Outcome:
    """One (failure type, intervention) trial on one erroneous answer."""
    figure_id: str
    failure_type: str
    intervention: str
    recovered: bool


def recovery_matrix(outcomes: Sequence[Outcome]) -> dict[tuple[str, str], float]:
    """P(recovered | failure type, intervention) for every observed cell."""
    sums: dict[tuple[str, str], list[int]] = {}
    for o in outcomes:
        key = (o.failure_type, o.intervention)
        cell = sums.setdefault(key, [0, 0])
        cell[0] += int(o.recovered)
        cell[1] += 1
    return {k: v[0] / v[1] for k, v in sums.items()}


def cell_counts(outcomes: Sequence[Outcome]) -> dict[tuple[str, str], int]:
    counts: dict[tuple[str, str], int] = {}
    for o in outcomes:
        key = (o.failure_type, o.intervention)
        counts[key] = counts.get(key, 0) + 1
    return counts


def _rate(rec: np.ndarray, mask: np.ndarray) -> float:
    return float(rec[mask].mean()) if mask.any() else float("nan")


def difference_in_differences(outcomes: Sequence[Outcome],
                              intervention: str,
                              baseline: str = BASELINE) -> float:
    """How much more the intervention helps structural than fabrication.

    Positive means the repair is type-specific in the predicted direction.
    Zero means the taxonomy buys nothing for this repair, however well the
    probes perform.
    """
    if intervention == ABSTAIN:
        raise ValueError(
            "I_abstain has no recovery rate; it is scored as risk avoided and "
            "belongs to E5's risk-coverage analysis, not this contrast")

    rec = np.array([o.recovered for o in outcomes], dtype=float)
    ftype = np.array([o.failure_type for o in outcomes])
    iv = np.array([o.intervention for o in outcomes])
    return _did_from_arrays(rec, ftype, iv, intervention, baseline,
                            np.arange(len(outcomes)))


def _did_from_arrays(rec, ftype, iv, intervention, baseline, idx):
    r, f, i = rec[idx], ftype[idx], iv[idx]
    s = (_rate(r, (f == STRUCTURAL) & (i == intervention))
         - _rate(r, (f == STRUCTURAL) & (i == baseline)))
    b = (_rate(r, (f == FABRICATION) & (i == intervention))
         - _rate(r, (f == FABRICATION) & (i == baseline)))
    return s - b


def did_with_ci(outcomes: Sequence[Outcome],
                intervention: str,
                baseline: str = BASELINE,
                n_boot: int = 2000,
                seed: int = 42) -> dict:
    """The pre-registered statistic with a cluster-bootstrapped 95% interval.

    Grouped by figure, because a figure contributes several erroneous answers
    and their outcomes are correlated (TASKS Section 5.5).
    """
    rec = np.array([o.recovered for o in outcomes], dtype=float)
    ftype = np.array([o.failure_type for o in outcomes])
    iv = np.array([o.intervention for o in outcomes])
    figs = np.array([o.figure_id for o in outcomes])

    point = _did_from_arrays(rec, ftype, iv, intervention, baseline,
                             np.arange(len(outcomes)))
    lo, hi = cluster_bootstrap_ci(
        lambda idx: _did_from_arrays(rec, ftype, iv, intervention, baseline, idx),
        figs, n_boot=n_boot, seed=seed)
    return {"intervention": intervention, "delta": point, "ci95": (lo, hi),
            "excludes_zero": not (lo <= 0.0 <= hi)}


def best_intervention_by_type(outcomes: Sequence[Outcome]) -> dict[str, str]:
    """Which repair wins for each failure type. The argmax behind claim B."""
    matrix = recovery_matrix(outcomes)
    out = {}
    for ftype in (STRUCTURAL, FABRICATION):
        cells = {iv: v for (t, iv), v in matrix.items()
                 if t == ftype and iv != ABSTAIN}
        if cells:
            out[ftype] = max(cells, key=lambda k: (cells[k], k))
    return out


def argmax_flip_stability(outcomes: Sequence[Outcome],
                          n_boot: int = 2000, seed: int = 42) -> float:
    """Fraction of bootstrap resamples in which the two types pick different repairs.

    Reported alongside the difference in differences, never instead of it. An
    argmax is a fragile statistic: two cells within noise of each other flip on
    resampling, and a single point estimate hides that. Below roughly 0.9, treat
    the flip as unsupported however clean the point matrix looks.
    """
    figs = np.array([o.figure_id for o in outcomes])
    uniq = np.unique(figs)
    index_of = {g: np.flatnonzero(figs == g) for g in uniq}
    rng = np.random.default_rng(seed)
    arr = list(outcomes)

    flips = 0
    for _ in range(n_boot):
        drawn = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([index_of[g] for g in drawn])
        best = best_intervention_by_type([arr[i] for i in idx])
        if len(best) == 2 and best[STRUCTURAL] != best[FABRICATION]:
            flips += 1
    return flips / n_boot
