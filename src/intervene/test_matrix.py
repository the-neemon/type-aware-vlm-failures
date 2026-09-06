"""Tests for the E4 recovery matrix and its pre-registered test (P6.6, P6.7)."""

from __future__ import annotations

import numpy as np

from src.intervene.matrix import (
    Outcome, argmax_flip_stability, best_intervention_by_type, cell_counts,
    difference_in_differences, did_with_ci, recovery_matrix,
)


def _build(spec, per_fig=2, seed=0):
    """spec: {(failure_type, intervention): (n, recovery probability)}."""
    rng = np.random.default_rng(seed)
    out, fig = [], 0
    for (ftype, iv), (n, p) in spec.items():
        for i in range(n):
            if i % per_fig == 0:
                fig += 1
            out.append(Outcome(f"f{fig}", ftype, iv, bool(rng.random() < p)))
    return out


class TestRecoveryMatrix:
    def test_rates(self):
        out = [Outcome("f1", "structural", "I_crop", True),
               Outcome("f1", "structural", "I_crop", False),
               Outcome("f2", "fabrication", "I_crop", False)]
        m = recovery_matrix(out)
        assert m[("structural", "I_crop")] == 0.5
        assert m[("fabrication", "I_crop")] == 0.0

    def test_counts(self):
        out = [Outcome("f1", "structural", "I_0", True),
               Outcome("f1", "structural", "I_0", False)]
        assert cell_counts(out)[("structural", "I_0")] == 2


class TestDifferenceInDifferences:
    def test_known_value(self):
        # structural gains 0.4, fabrication gains 0.1, so delta is 0.3
        out = ([Outcome("f1", "structural", "I_0", i < 2) for i in range(10)]
               + [Outcome("f2", "structural", "I_crop", i < 6) for i in range(10)]
               + [Outcome("f3", "fabrication", "I_0", i < 1) for i in range(10)]
               + [Outcome("f4", "fabrication", "I_crop", i < 2) for i in range(10)])
        assert abs(difference_in_differences(out, "I_crop") - 0.3) < 1e-9

    def test_zero_when_both_types_respond_alike(self):
        out = ([Outcome("f1", "structural", "I_0", i < 3) for i in range(10)]
               + [Outcome("f2", "structural", "I_crop", i < 6) for i in range(10)]
               + [Outcome("f3", "fabrication", "I_0", i < 3) for i in range(10)]
               + [Outcome("f4", "fabrication", "I_crop", i < 6) for i in range(10)])
        assert abs(difference_in_differences(out, "I_crop")) < 1e-9

    def test_abstain_is_refused(self):
        # abstention produces no correct answer, so it has no recovery rate;
        # silently returning one would compare different units
        out = [Outcome("f1", "structural", "I_abstain", False)]
        try:
            difference_in_differences(out, "I_abstain")
        except ValueError:
            return
        raise AssertionError("I_abstain must be refused in this contrast")


class TestDidWithCI:
    def test_detects_a_large_real_effect(self):
        out = _build({("structural", "I_0"): (300, 0.25),
                      ("structural", "I_crop"): (300, 0.65),
                      ("fabrication", "I_0"): (300, 0.10),
                      ("fabrication", "I_crop"): (300, 0.15)}, seed=1)
        r = did_with_ci(out, "I_crop", n_boot=400)
        assert r["delta"] > 0.2 and r["excludes_zero"]

    def test_null_effect_interval_covers_zero(self):
        out = _build({("structural", "I_0"): (300, 0.25),
                      ("structural", "I_crop"): (300, 0.40),
                      ("fabrication", "I_0"): (300, 0.25),
                      ("fabrication", "I_crop"): (300, 0.40)}, seed=2)
        r = did_with_ci(out, "I_crop", n_boot=400)
        assert not r["excludes_zero"]

    def test_ci_brackets_the_point(self):
        out = _build({("structural", "I_0"): (200, 0.25),
                      ("structural", "I_crop"): (200, 0.55),
                      ("fabrication", "I_0"): (200, 0.10),
                      ("fabrication", "I_crop"): (200, 0.20)}, seed=3)
        r = did_with_ci(out, "I_crop", n_boot=400)
        lo, hi = r["ci95"]
        assert lo <= r["delta"] <= hi


class TestArgmax:
    def test_best_excludes_abstain(self):
        out = ([Outcome("f1", "structural", "I_crop", True)] * 10
               + [Outcome("f2", "structural", "I_abstain", True)] * 10
               + [Outcome("f3", "structural", "I_verify", False)] * 10)
        assert best_intervention_by_type(out)["structural"] == "I_crop"

    def test_flip_is_stable_when_the_gap_is_wide(self):
        out = _build({("structural", "I_crop"): (300, 0.70),
                      ("structural", "I_verify"): (300, 0.20),
                      ("fabrication", "I_crop"): (300, 0.15),
                      ("fabrication", "I_verify"): (300, 0.60)}, seed=4)
        assert argmax_flip_stability(out, n_boot=300) > 0.95

    def test_flip_is_unstable_when_cells_are_tied(self):
        # the case the statistic exists to expose: a clean-looking point matrix
        # whose argmax is noise
        out = _build({("structural", "I_crop"): (300, 0.40),
                      ("structural", "I_verify"): (300, 0.40),
                      ("fabrication", "I_crop"): (300, 0.40),
                      ("fabrication", "I_verify"): (300, 0.40)}, seed=5)
        assert argmax_flip_stability(out, n_boot=300) < 0.9


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
