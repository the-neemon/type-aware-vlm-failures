"""Shared linear-model primitives.

Both the activation probes (P5.1 to P5.3) and the E2 surface baseline (P5.4)
are L2 logistic regressions scored by AUROC, so the fitting, the metric, the
residualisation and the bootstrap live here and are imported by both.

Implemented on numpy and scipy rather than scikit-learn: SPEC specifies plain
L2 logistic regression, it is short, and it is one less dependency to pin on
the cluster.
"""

from __future__ import annotations

from typing import Callable, Sequence

import numpy as np
from scipy.optimize import minimize


# ---------------------------------------------------------------------------
# Metric
# ---------------------------------------------------------------------------

def auroc(y_true: np.ndarray, scores: np.ndarray) -> float:
    """Rank-based AUROC, ties averaged. Returns nan if either class is empty."""
    y = np.asarray(y_true).astype(bool)
    s = np.asarray(scores, dtype=np.float64)
    n_pos, n_neg = int(y.sum()), int((~y).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s), dtype=np.float64)
    ranks[order] = np.arange(1, len(s) + 1, dtype=np.float64)

    s_sorted = s[order]
    i = 0
    while i < len(s_sorted):
        j = i
        while j + 1 < len(s_sorted) and s_sorted[j + 1] == s_sorted[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = ranks[order[i:j + 1]].mean()
        i = j + 1

    return (ranks[y].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def fit_logistic(X: np.ndarray, y: np.ndarray, l2: float = 1.0) -> np.ndarray:
    """Fit L2-regularised logistic regression. Returns [intercept, *weights].

    The intercept is not penalised. That matters on imbalanced classes: the
    fabrication class is expected to be scarce, and penalising the intercept
    biases the model toward the majority class.
    """
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    Xb = np.hstack([np.ones((len(X), 1)), X])

    def objective(w):
        z = Xb @ w
        ll = np.sum(np.logaddexp(0.0, z) - y * z)
        p = 1.0 / (1.0 + np.exp(-z))
        grad = Xb.T @ (p - y)
        grad[1:] += l2 * w[1:]
        return ll + 0.5 * l2 * np.sum(w[1:] ** 2), grad

    return minimize(objective, np.zeros(Xb.shape[1]), jac=True,
                    method="L-BFGS-B").x


def predict_scores(X: np.ndarray, w: np.ndarray) -> np.ndarray:
    Xb = np.hstack([np.ones((len(X), 1)), np.asarray(X, dtype=np.float64)])
    return Xb @ w


# ---------------------------------------------------------------------------
# Residualisation (E2, second half)
# ---------------------------------------------------------------------------

def residualise(A: np.ndarray, S: np.ndarray,
                fit_on: np.ndarray | None = None) -> np.ndarray:
    """Remove everything linearly predictable from surface features S out of A.

    `fit_on` is a boolean mask selecting the rows the projection is estimated
    from; pass the training mask. Estimating it on the pooled set leaks test
    information through the projection, which is a quieter version of the same
    mistake as selecting a layer on test data.
    """
    A = np.asarray(A, dtype=np.float64)
    Sb = np.hstack([np.ones((len(S), 1)), np.asarray(S, dtype=np.float64)])
    rows = slice(None) if fit_on is None else np.asarray(fit_on, dtype=bool)
    coef, *_ = np.linalg.lstsq(Sb[rows], A[rows], rcond=None)
    return A - Sb @ coef


# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------

def cluster_bootstrap_ci(statistic: Callable[[np.ndarray], float],
                         groups: Sequence,
                         n_boot: int = 2000,
                         alpha: float = 0.05,
                         seed: int = 42) -> tuple[float, float]:
    """Percentile CI, resampling GROUPS with replacement rather than items.

    Items sharing a figure are correlated, because one figure contributes
    several questions. Bootstrapping over items treats them as independent and
    returns intervals that are too narrow.

    Measured, because the effect is narrower than the usual advice suggests:
    clustering widens the interval only when the label AND the score are both
    correlated within figure. If only one is, the two bootstraps agree to within
    noise. Our case has both, which is why this is the default here: a hard
    figure produces several errors at once, and the activations behind its
    questions share the same figure encoding. See
    `TestClusterBootstrap.test_wider_when_label_and_score_both_cluster`.

    `statistic` takes an index array into the original rows and returns a
    scalar. Draws that produce nan (a resample with only one class present) are
    dropped rather than propagated.
    """
    groups = np.asarray(groups)
    uniq = np.unique(groups)
    index_of = {g: np.flatnonzero(groups == g) for g in uniq}
    rng = np.random.default_rng(seed)

    vals = []
    for _ in range(n_boot):
        drawn = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([index_of[g] for g in drawn])
        v = statistic(idx)
        if not np.isnan(v):
            vals.append(v)

    if not vals:
        return float("nan"), float("nan")
    return (float(np.percentile(vals, 100 * alpha / 2)),
            float(np.percentile(vals, 100 * (1 - alpha / 2))))
