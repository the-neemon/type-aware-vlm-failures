"""Tests for pool building, assignment, and multi-rater agreement."""

from __future__ import annotations

import collections
import json
import math

from src.eval.manifest import ManifestRow, write_manifest
from src.label.multi_rater import (
    disagreements, fleiss_kappa, load_raters, pairwise, shared_items,
)
from src.label.pool import assign, build_pool, item_id, sanitize


def _manifests(tmp_path):
    rows = [ManifestRow(f"fig{i}.png", f"q{i}", "10", "12" if i % 2 else "10", i % 2 == 0)
            for i in range(40)]
    a, b = tmp_path / "qwen.jsonl", tmp_path / "llava.jsonl"
    write_manifest(rows, a)
    write_manifest(rows, b)          # same questions wrong for both models
    return [("qwen", a), ("llava", b)]


class TestBuildPool:
    def test_keeps_only_incorrect(self, tmp_path):
        pool, key = build_pool(_manifests(tmp_path), "test")
        assert len(pool) == len(key) == 40      # 20 wrong per model

    def test_same_question_two_models_does_not_collide(self, tmp_path):
        # the bug an ID of figure::question would have: both models erring on
        # one question would merge into a single item
        pool, _ = build_pool(_manifests(tmp_path), "test")
        assert len({p["item_id"] for p in pool}) == 40

    def test_pool_is_blind(self, tmp_path):
        pool, _ = build_pool(_manifests(tmp_path), "test")
        for p in pool:
            assert set(p) == {"item_id", "image_path", "question",
                              "gold_answer", "model_answer"}
            assert "qwen" not in json.dumps(p) and "llava" not in json.dumps(p)

    def test_order_does_not_reveal_model(self, tmp_path):
        pool, key = build_pool(_manifests(tmp_path), "test")
        model = {k["item_id"]: k["model"] for k in key}
        first_half = [model[p["item_id"]] for p in pool[:20]]
        assert 0 < first_half.count("qwen") < 20

    def test_image_path_uses_split(self, tmp_path):
        pool, _ = build_pool(_manifests(tmp_path), "test")
        assert all(p["image_path"].startswith("test/png/") for p in pool)

    def test_deterministic(self, tmp_path):
        a, _ = build_pool(_manifests(tmp_path), "test")
        b, _ = build_pool(_manifests(tmp_path), "test")
        assert a == b

    def test_id_is_stable_and_opaque(self):
        assert item_id("qwen", "f.png", "q") == item_id("qwen", "f.png", "q")
        assert item_id("qwen", "f.png", "q") != item_id("llava", "f.png", "q")


def _pool(n):
    return [{"item_id": f"i{k}", "image_path": "", "question": f"q{k}",
             "gold_answer": "1", "model_answer": "2"} for k in range(n)]


class TestAssign:
    def test_balanced_load(self):
        tasks = assign(_pool(1000), ["naman", "yash", "shrish", "sanjith"], 200)
        assert sorted(len(v) for v in tasks.values()) == [300, 300, 300, 300]

    def test_every_item_covered_overlap_exactly_twice(self):
        tasks = assign(_pool(1000), ["a", "b", "c", "d"], 200)
        counts = collections.Counter(it["item_id"] for v in tasks.values() for it in v)
        assert len(counts) == 1000
        assert collections.Counter(counts.values()) == {1: 800, 2: 200}

    def test_every_pair_shares_items(self):
        # all four people checked against each other, not one pair
        tasks = assign(_pool(1000), ["a", "b", "c", "d"], 200)
        ids = {n: {it["item_id"] for it in v} for n, v in tasks.items()}
        for x in ids:
            for y in ids:
                if x < y:
                    assert len(ids[x] & ids[y]) >= 33

    def test_shared_items_spread_through_queue(self):
        tasks = assign(_pool(1000), ["a", "b", "c", "d"], 200)
        shared = {i for i, c in collections.Counter(
            it["item_id"] for v in tasks.values() for it in v).items() if c == 2}
        first_third = [it["item_id"] in shared for it in tasks["a"][:100]]
        assert 10 < sum(first_third) < 60      # ~33 expected, not all 100

    def test_rejects_duplicate_names(self):
        try:
            assign(_pool(10), ["Naman", "naman"], 2)
        except ValueError:
            return
        raise AssertionError("duplicate annotators should raise")

    def test_sanitize(self):
        assert sanitize("  Naman Singhal ") == "naman_singhal"


class TestFleiss:
    def test_hand_computed(self):
        # P_bar = 0.75, p_s = 5/8, p_f = 3/8, P_e = 34/64 -> 7/15
        k = fleiss_kappa([["structural", "structural"], ["structural", "structural"],
                          ["fabrication", "fabrication"], ["structural", "fabrication"]])
        assert abs(k - 7 / 15) < 1e-9

    def test_perfect_agreement(self):
        assert fleiss_kappa([["structural"] * 2, ["fabrication"] * 2]) == 1.0

    def test_single_category_is_nan_not_perfect(self):
        # would otherwise pass the gate on a degenerate set
        assert math.isnan(fleiss_kappa([["structural"] * 2] * 5))

    def test_three_raters_on_some_items(self):
        k = fleiss_kappa([["structural"] * 3, ["fabrication"] * 2, ["structural", "ambiguous"]])
        assert -1 <= k <= 1

    def test_single_rated_items_ignored(self):
        assert math.isnan(fleiss_kappa([["structural"], ["fabrication"]]))


def _write_rater(d, name, recs):
    with open(d / f"{name}.jsonl", "w") as f:
        for iid, label in recs:
            f.write(json.dumps({"item_id": iid, "label": label, "question": iid,
                                "gold_answer": "1", "model_answer": "2"}) + "\n")


class TestRaters:
    def test_last_record_wins(self, tmp_path):
        _write_rater(tmp_path, "a", [("x", "structural"), ("x", "fabrication")])
        ratings, bad = load_raters(tmp_path)
        assert ratings["a"]["x"]["label"] == "fabrication" and bad == 0

    def test_bad_lines_skipped_and_counted(self, tmp_path):
        (tmp_path / "a.jsonl").write_text(
            '{"item_id": "x", "label": "structural"}\nnot json\n'
            '{"item_id": "y", "label": "bogus"}\n{"label": "structural"}\n')
        ratings, bad = load_raters(tmp_path)
        assert list(ratings["a"]) == ["x"] and bad == 3

    def test_tasks_subdir_ignored(self, tmp_path):
        (tmp_path / "tasks").mkdir()
        _write_rater(tmp_path / "tasks", "pool", [("x", "structural")])
        _write_rater(tmp_path, "a", [("x", "structural")])
        assert list(load_raters(tmp_path)[0]) == ["a"]

    def test_pairwise_and_disagreements(self, tmp_path):
        _write_rater(tmp_path, "a", [("x", "structural"), ("y", "fabrication"), ("z", "structural")])
        _write_rater(tmp_path, "b", [("x", "structural"), ("y", "structural")])
        ratings, _ = load_raters(tmp_path)
        (row,) = pairwise(ratings)
        assert row["n"] == 2 and row["agreement"] == 0.5 and not row["enough_items"]
        assert set(shared_items(ratings, ["a", "b"])) == {"x", "y"}
        (d,) = disagreements(ratings, ["a", "b"])
        assert d["item_id"] == "y" and d["votes"]["b"][0] == "structural"
