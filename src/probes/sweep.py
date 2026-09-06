"""Per-layer probe sweep, layer selection, and cross-transfer (P5.1 to P5.3, P5.6).

Three probes, all L2 logistic regressions on cached activations: `binary`
(correct vs incorrect), and one-vs-rest `structural` and `fabrication`.

**The layer-selection guard is the point of this module.** Picking the best
layer by test AUROC is the classic way to invent a result, and TASKS Section
5.4 asks for code that cannot make that mistake rather than a comment asking
people not to. So the API is shaped to forbid it:

    sweep_layers(train, val)      -> per-layer validation AUROC. No test data.
    select_layer(sweep_result)    -> one layer. Takes only the sweep result.
    evaluate_at(layer, test)      -> the reported number. Layer already fixed.

`sweep_layers` has no parameter that test data could be passed as, so selecting
on test requires deliberately misusing the API rather than forgetting a rule.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

from src.common.linear import auroc, cluster_bootstrap_ci, fit_logistic, predict_scores

# activations for one probe: {layer index: (n_items, n_features)}
Activations = Mapping[int, np.ndarray]


@dataclass(frozen=True)
class SweepResult:
    """Per-layer validation AUROC. Carries no test data, by construction."""
    auroc_by_layer: dict[int, float]
    weights_by_layer: dict[int, np.ndarray]

    def best_layer(self) -> int:
        return max(self.auroc_by_layer,
                   key=lambda k: (self.auroc_by_layer[k], -k))


def sweep_layers(train: Activations, y_train: np.ndarray,
                 val: Activations, y_val: np.ndarray,
                 l2: float = 1.0) -> SweepResult:
    """Fit one probe per layer on train, score each on validation.

    Deliberately takes no test set. See the module docstring.
    """
    if set(train) != set(val):
        raise ValueError("train and val must cover the same layers")

    aurocs, weights = {}, {}
    for layer in sorted(train):
        w = fit_logistic(train[layer], y_train, l2)
        weights[layer] = w
        aurocs[layer] = auroc(y_val, predict_scores(val[layer], w))
    return SweepResult(aurocs, weights)


def select_layer(result: SweepResult) -> int:
    """Choose the reporting layer. Takes only validation evidence."""
    return result.best_layer()


def evaluate_at(layer: int, result: SweepResult,
                test: Activations, y_test: np.ndarray,
                figure_ids: Sequence | None = None) -> dict:
    """Score the already-chosen layer on test, with a cluster bootstrap CI.

    `figure_ids` groups the bootstrap by figure rather than by item, because a
    figure contributes several questions and item-level resampling produces
    intervals that are too narrow (TASKS Section 5.5).
    """
    w = result.weights_by_layer[layer]
    scores = predict_scores(test[layer], w)
    point = auroc(y_test, scores)

    out = {"layer": layer, "auroc": point}
    if figure_ids is not None:
        y = np.asarray(y_test)
        lo, hi = cluster_bootstrap_ci(
            lambda idx: auroc(y[idx], scores[idx]), figure_ids)
        out["ci95"] = (lo, hi)
    return out


# ---------------------------------------------------------------------------
# E3: cross-transfer
# ---------------------------------------------------------------------------

def cross_transfer(train: Activations, y_train: np.ndarray,
                   test: Activations, y_test: np.ndarray,
                   layer: int, classes: Sequence[str],
                   l2: float = 1.0) -> dict[tuple[str, str], float]:
    """AUROC of each class's probe evaluated against every class.

    Off-diagonal cells are the finding. Strong cross-transfer means one signal
    wearing two labels: the probes are not separating failure types, they are
    both reading the same thing. That is a reportable result, not a failure.
    """
    y_tr, y_te = np.asarray(y_train), np.asarray(y_test)
    out = {}
    for trained_on in classes:
        w = fit_logistic(train[layer], (y_tr == trained_on).astype(float), l2)
        scores = predict_scores(test[layer], w)
        for evaluated_on in classes:
            out[(trained_on, evaluated_on)] = auroc(y_te == evaluated_on, scores)
    return out


# ---------------------------------------------------------------------------
# P4.5: pipeline sanity probe
# ---------------------------------------------------------------------------

def sanity_probe(train: Activations, val: Activations,
                 y_train: np.ndarray, y_val: np.ndarray,
                 threshold: float = 0.9) -> tuple[bool, float]:
    """Can the activations predict something trivially decodable?

    Pass a target like figure type or question template. If the best layer
    cannot clear `threshold`, the caching is broken and every downstream null
    result is about the code rather than about the hypothesis. Run this before
    writing any sentence containing the phrase "does not separate".
    """
    result = sweep_layers(train, y_train, val, y_val)
    best = max(v for v in result.auroc_by_layer.values() if not np.isnan(v))
    return best >= threshold, best
