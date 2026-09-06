"""Tests for the probe sweep (P5.1 to P5.3, P5.6) and shared linear primitives.

pytest-style to match src/eval, runnable standalone with
`python3 src/probes/test_sweep.py` before pytest is pinned.
"""

from __future__ import annotations

import inspect

import numpy as np

from src.common.linear import auroc, cluster_bootstrap_ci, residualise
from src.probes.sweep import (
    cross_transfer, evaluate_at, sanity_probe, select_layer, sweep_layers,
)

LAYERS = (0, 3, 7, 11)


def _data(n, signal_layer=7, strength=4.0, seed=0, n_feat=6):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, n).astype(float)
    acts = {L: rng.normal(size=(n, n_feat)) for L in LAYERS}
    if signal_layer is not None:
        acts[signal_layer][:, 0] += strength * y
    return acts, y


class TestLayerSelection:
    def test_finds_the_signal_layer(self):
        tr, ytr = _data(400, seed=1)
        va, yva = _data(200, seed=2)
        assert select_layer(sweep_layers(tr, ytr, va, yva)) == 7

    def test_mismatched_layers_raise(self):
        tr, ytr = _data(100, seed=1)
        va, yva = _data(100, seed=2)
        del va[11]
        try:
            sweep_layers(tr, ytr, va, yva)
        except ValueError:
            return
        raise AssertionError("mismatched layer sets should raise")

    def test_sweep_cannot_be_handed_test_data(self):
        # The guard is structural, not a comment. If someone adds a test
        # parameter to sweep_layers, this fails and they have to justify it.
        params = set(inspect.signature(sweep_layers).parameters)
        assert not any("test" in p for p in params), params

    def test_result_carries_no_test_data(self):
        tr, ytr = _data(200, seed=1)
        va, yva = _data(100, seed=2)
        fields = set(vars(sweep_layers(tr, ytr, va, yva)))
        assert fields == {"auroc_by_layer", "weights_by_layer"}


class TestEvaluateAt:
    def test_reports_selected_layer_only(self):
        tr, ytr = _data(400, seed=1)
        va, yva = _data(200, seed=2)
        te, yte = _data(200, seed=3)
        r = sweep_layers(tr, ytr, va, yva)
        out = evaluate_at(select_layer(r), r, te, yte)
        assert out["layer"] == 7 and out["auroc"] > 0.9
        assert "ci95" not in out          # absent unless figure_ids given

    def test_ci_brackets_the_point_estimate(self):
        tr, ytr = _data(400, seed=1)
        va, yva = _data(200, seed=2)
        te, yte = _data(200, seed=3)
        r = sweep_layers(tr, ytr, va, yva)
        figs = np.repeat(np.arange(50), 4)
        out = evaluate_at(7, r, te, yte, figs)
        lo, hi = out["ci95"]
        assert lo <= out["auroc"] <= hi


class TestClusterBootstrap:
    def test_wider_when_label_and_score_both_cluster(self):
        # The claim in TASKS Section 5.5, tested rather than asserted. Writing
        # it revealed the condition is narrower than the usual advice: the
        # bootstraps agree unless BOTH the label and the score are correlated
        # within figure. Our case has both, so this is the realistic scenario:
        # hard figures produce several errors at once, and the activations
        # behind one figure's questions share its encoding.
        rng = np.random.default_rng(42)
        n_fig, per_fig = 60, 4
        figs = np.repeat(np.arange(n_fig), per_fig)
        hard = rng.random(n_fig) < 0.5
        y = (rng.random(n_fig * per_fig)
             < np.where(hard, 0.85, 0.15)[figs]).astype(float)
        scores = (rng.normal(size=n_fig)[figs] + 0.8 * y
                  + rng.normal(scale=0.6, size=len(y)))

        def stat(idx):
            return auroc(y[idx], scores[idx])

        clo, chi = cluster_bootstrap_ci(stat, figs, n_boot=600)
        ilo, ihi = cluster_bootstrap_ci(stat, np.arange(len(y)), n_boot=600)
        assert (chi - clo) > 1.2 * (ihi - ilo)

    def test_agrees_with_item_bootstrap_when_only_the_label_clusters(self):
        # The negative half of the same finding: clustering is not a free
        # widening, so a wide E4 interval cannot be explained away as "we
        # clustered". Guards against someone dropping the grouping because it
        # "made no difference" on a case where it genuinely does not.
        rng = np.random.default_rng(42)
        n_fig, per_fig = 60, 4
        figs = np.repeat(np.arange(n_fig), per_fig)
        hard = rng.random(n_fig) < 0.5
        y = (rng.random(n_fig * per_fig)
             < np.where(hard, 0.85, 0.15)[figs]).astype(float)
        scores = 0.8 * y + rng.normal(scale=1.0, size=len(y))

        def stat(idx):
            return auroc(y[idx], scores[idx])

        clo, chi = cluster_bootstrap_ci(stat, figs, n_boot=600)
        ilo, ihi = cluster_bootstrap_ci(stat, np.arange(len(y)), n_boot=600)
        assert abs((chi - clo) - (ihi - ilo)) < 0.05

    def test_all_one_class_gives_nan(self):
        lo, hi = cluster_bootstrap_ci(lambda idx: float("nan"),
                                      np.arange(10), n_boot=20)
        assert np.isnan(lo) and np.isnan(hi)


class TestCrossTransfer:
    def test_independent_signals_do_not_transfer(self):
        rng = np.random.default_rng(42)
        n = 600
        lab = rng.integers(0, 2, n)
        y = np.where(lab == 0, "structural", "fabrication")
        acts = {7: rng.normal(size=(n, 8))}
        acts[7][:, 0] += 4.0 * (lab == 0)      # only structural is encoded
        te = {7: acts[7][300:]}
        out = cross_transfer({7: acts[7][:300]}, y[:300], te, y[300:], 7,
                             ["structural", "fabrication"])
        assert out[("structural", "structural")] > 0.9
        # the fabrication probe has nothing of its own to learn here
        assert out[("structural", "fabrication")] < 0.15

    def test_one_signal_two_labels_transfers(self):
        # exactly the failure E3 exists to detect: both labels riding one axis
        rng = np.random.default_rng(42)
        n = 600
        lab = rng.integers(0, 2, n)
        y = np.where(lab == 0, "structural", "fabrication")
        acts = {7: rng.normal(size=(n, 8))}
        acts[7][:, 0] += 4.0 * lab
        out = cross_transfer({7: acts[7][:300]}, y[:300],
                             {7: acts[7][300:]}, y[300:], 7,
                             ["structural", "fabrication"])
        assert out[("fabrication", "fabrication")] > 0.9
        assert out[("structural", "structural")] > 0.9


class TestSanityProbe:
    def test_passes_on_decodable_target(self):
        tr, ytr = _data(400, seed=1)
        va, yva = _data(200, seed=2)
        ok, best = sanity_probe(tr, va, ytr, yva)
        assert ok and best > 0.9

    def test_fails_on_pure_noise(self):
        tr, ytr = _data(400, signal_layer=None, seed=1)
        va, yva = _data(200, signal_layer=None, seed=2)
        ok, best = sanity_probe(tr, va, ytr, yva)
        assert not ok and best < 0.9


class TestResidualiseFitOn:
    def test_projection_estimated_on_train_only(self):
        rng = np.random.default_rng(42)
        S = rng.normal(size=(200, 3))
        A = S @ rng.normal(size=(3, 4))
        mask = np.zeros(200, dtype=bool)
        mask[:100] = True
        R = residualise(A, S, fit_on=mask)
        # a projection estimated on half must still clean the other half,
        # since the relationship is genuinely linear
        assert np.abs(R).max() < 1e-8

    def test_pooled_fit_is_the_default(self):
        rng = np.random.default_rng(42)
        S = rng.normal(size=(50, 2))
        A = rng.normal(size=(50, 2))
        assert residualise(A, S).shape == A.shape


if __name__ == "__main__":
    import sys, traceback
    passed = failed = 0
    for cls in [v for v in dict(globals()).values()
                if isinstance(v, type) and v.__name__.startswith("Test")]:
        inst = cls()
        for name in dir(inst):
            if not name.startswith("test_"):
                continue
            try:
                getattr(inst, name)()
                passed += 1
            except Exception:
                failed += 1
                print(f"FAIL {cls.__name__}.{name}")
                traceback.print_exc()
    print(f"\n{passed} passed, {failed} failed")
    sys.exit(1 if failed else 0)
