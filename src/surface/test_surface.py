"""Tests for the E2 surface baseline.

pytest-style to match src/eval, but runnable standalone with
`python3 src/surface/test_surface.py` so it works before pytest is pinned.
"""

from __future__ import annotations

import numpy as np

from src.surface.baseline import auroc, fit_logistic, predict_scores, residualise
from src.surface.features import SurfaceItem, answer_in_figure, answer_type, featurise
from src.surface.run_e2_baseline import figure_text


class TestAnswerInFigure:
    def test_exact_string(self):
        assert answer_in_figure("B", ["A", "B", "C"])

    def test_absent(self):
        assert not answer_in_figure("D", ["A", "B", "C"])

    def test_numeric_formatting(self):
        assert answer_in_figure("45", ["45.0"])
        assert answer_in_figure("45.0", ["45"])

    def test_percent_suffix_stripped(self):
        assert answer_in_figure("45%", ["45"])

    def test_tolerance_is_tight(self):
        # 1 percent, deliberately much tighter than ChartQA's 5 percent scoring
        assert answer_in_figure("100.5", ["100"])
        assert not answer_in_figure("104", ["100"])

    def test_empty_answer(self):
        assert not answer_in_figure("", ["A"])


class TestAnswerType:
    def test_numeric(self):
        assert answer_type("45") == "numeric"
        assert answer_type("-3.5") == "numeric"

    def test_boolean(self):
        assert answer_type("Yes") == "boolean"

    def test_categorical(self):
        assert answer_type("Category A") == "categorical"

    def test_empty(self):
        assert answer_type("") == "other"


class TestFeaturise:
    def test_shape_and_names(self):
        items = [SurfaceItem("45", ("45",), "read_value", "bar_chart"),
                 SurfaceItem("D", ("A",), "absent_category", "bar_chart")]
        X, names = featurise(items)
        assert X.shape == (2, len(names))
        assert "answer_in_figure" in names
        assert "template=absent_category" in names

    def test_no_activation_features_leak_in(self):
        _, names = featurise([SurfaceItem("1", ("1",), "t", "f")])
        assert all("layer" not in n and "activation" not in n for n in names)


class TestFigureText:
    def test_bar_chart(self):
        fig = {"figure_type": "bar_chart", "categories": ["A", "B"], "values": [10, 20]}
        assert set(figure_text(fig)) == {"A", "B", "10", "20"}

    def test_node_link(self):
        fig = {"figure_type": "node_link", "path_nodes": ["A", "B"],
               "edges": [["A", "B"], ["B", "C"]]}
        assert set(figure_text(fig)) == {"A", "B", "C"}

    def test_deduplicates(self):
        fig = {"figure_type": "node_link", "path_nodes": ["A"], "edges": [["A", "A"]]}
        assert figure_text(fig) == ("A",)

    def test_unknown_type_raises(self):
        # must not silently return empty text; that would make every answer
        # look absent from the figure and quietly invert the feature
        try:
            figure_text({"figure_type": "scatter"})
        except ValueError:
            return
        raise AssertionError("unknown figure_type should raise")


class TestAUROC:
    def test_perfect_separation(self):
        assert auroc(np.array([0, 0, 1, 1]), np.array([0.1, 0.2, 0.8, 0.9])) == 1.0

    def test_inverted(self):
        assert auroc(np.array([0, 0, 1, 1]), np.array([0.9, 0.8, 0.2, 0.1])) == 0.0

    def test_all_ties_is_half(self):
        assert auroc(np.array([0, 0, 1, 1]), np.array([1.0, 1.0, 1.0, 1.0])) == 0.5

    def test_single_class_is_nan(self):
        assert np.isnan(auroc(np.array([1, 1, 1]), np.array([0.1, 0.2, 0.3])))


class TestLogistic:
    def test_recovers_separable_signal(self):
        rng = np.random.default_rng(42)
        y = np.r_[np.zeros(60), np.ones(60)]
        X = np.c_[np.r_[rng.normal(0, 1, 60), rng.normal(5, 1, 60)]]
        w = fit_logistic(X, y)
        assert auroc(y, predict_scores(X, w)) > 0.99

    def test_noise_is_near_chance(self):
        rng = np.random.default_rng(42)
        y = np.r_[np.zeros(60), np.ones(60)]
        X = rng.normal(size=(120, 1))
        w = fit_logistic(X, y)
        assert 0.35 < auroc(y, predict_scores(X, w)) < 0.65

    def test_intercept_unpenalised_on_imbalance(self):
        # 10:1 imbalance, one informative feature; heavy L2 must not flatten it
        rng = np.random.default_rng(42)
        y = np.r_[np.zeros(100), np.ones(10)]
        X = np.c_[np.r_[rng.normal(0, 1, 100), rng.normal(4, 1, 10)]]
        w = fit_logistic(X, y, l2=10.0)
        assert auroc(y, predict_scores(X, w)) > 0.9


class TestResidualise:
    def test_removes_linear_dependence(self):
        rng = np.random.default_rng(42)
        S = rng.normal(size=(200, 3))
        A = S @ rng.normal(size=(3, 4)) + rng.normal(scale=0.1, size=(200, 4))
        R = residualise(A, S)
        assert abs(np.corrcoef(R[:, 0], S[:, 0])[0, 1]) < 1e-8

    def test_leaves_independent_signal(self):
        rng = np.random.default_rng(42)
        S = rng.normal(size=(200, 2))
        A = rng.normal(size=(200, 2))
        assert np.linalg.norm(residualise(A, S)) > 0.8 * np.linalg.norm(A)


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
