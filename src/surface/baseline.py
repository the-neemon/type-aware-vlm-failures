"""L2 logistic regression, AUROC, and residualisation for E2 (TASKS P5.4, P5.5).

Implemented on numpy and scipy rather than scikit-learn. SPEC specifies plain
L2 logistic regression, which is forty lines, and the cluster environment is
not yet pinned. One less dependency to argue about.

Two things live here:

    fit_logistic / auroc     the surface baseline itself (E2, first half)
    residualise              removing surface features from activations before
                             probing (E2, second half, following Sahoo et al.)
"""

from __future__ import annotations

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
    # average ranks within ties
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

    The intercept is not penalised, which matters on imbalanced classes: the
    fabrication class is expected to be scarce, and penalising the intercept
    would bias the baseline toward predicting the majority class and understate
    how well surface features actually do.
    """
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    Xb = np.hstack([np.ones((len(X), 1)), X])

    def objective(w):
        z = Xb @ w
        # log(1 + exp(z)) computed stably
        ll = np.sum(np.logaddexp(0.0, z) - y * z)
        penalty = 0.5 * l2 * np.sum(w[1:] ** 2)
        p = 1.0 / (1.0 + np.exp(-z))
        grad = Xb.T @ (p - y)
        grad[1:] += l2 * w[1:]
        return ll + penalty, grad

    res = minimize(objective, np.zeros(Xb.shape[1]), jac=True, method="L-BFGS-B")
    return res.x


def predict_scores(X: np.ndarray, w: np.ndarray) -> np.ndarray:
    Xb = np.hstack([np.ones((len(X), 1)), np.asarray(X, dtype=np.float64)])
    return Xb @ w


def one_vs_rest_auroc(X_train, y_train, X_test, y_test, classes, l2=1.0):
    """Per-class one-vs-rest AUROC, mirroring how the activation probes are fit."""
    out = {}
    for c in classes:
        w = fit_logistic(X_train, (np.asarray(y_train) == c).astype(float), l2)
        out[c] = auroc((np.asarray(y_test) == c), predict_scores(X_test, w))
    return out


# ---------------------------------------------------------------------------
# Residualisation (E2, second half)
# ---------------------------------------------------------------------------

def residualise(A: np.ndarray, S: np.ndarray) -> np.ndarray:
    """Remove everything linearly predictable from surface features S out of A.

    Least-squares projection with an intercept. What remains is the part of the
    activation that the cheap features cannot explain, and a probe trained on it
    is the honest test of whether the signal is mechanistic. Sahoo et al. report
    a probe at 100 percent accuracy collapsing to chance under exactly this
    operation, which is why it is mandatory rather than optional.

    Fit the projection on training data only and apply it to test data; fitting
    on the pooled set leaks test information through the projection.
    """
    A = np.asarray(A, dtype=np.float64)
    Sb = np.hstack([np.ones((len(S), 1)), np.asarray(S, dtype=np.float64)])
    coef, *_ = np.linalg.lstsq(Sb, A, rcond=None)
    return A - Sb @ coef
