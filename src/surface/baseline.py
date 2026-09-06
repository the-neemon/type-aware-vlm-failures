"""Surface baseline for E2 (TASKS P5.4, P5.5).

The linear-model primitives live in `src.common.linear` because the activation
probes use exactly the same ones; this module is the E2-specific layer on top.
"""

from __future__ import annotations

import numpy as np

from src.common.linear import (  # re-exported: callers import from either
    auroc, cluster_bootstrap_ci, fit_logistic, predict_scores, residualise,
)

__all__ = ["auroc", "cluster_bootstrap_ci", "fit_logistic", "predict_scores",
           "residualise", "one_vs_rest_auroc"]


def one_vs_rest_auroc(X_train, y_train, X_test, y_test, classes, l2=1.0):
    """Per-class one-vs-rest AUROC, mirroring how the activation probes are fit."""
    out = {}
    for c in classes:
        w = fit_logistic(X_train, (np.asarray(y_train) == c).astype(float), l2)
        out[c] = auroc((np.asarray(y_test) == c), predict_scores(X_test, w))
    return out
