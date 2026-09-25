"""Tests for pool building, assignment, and multi-rater agreement."""

from __future__ import annotations

import collections
import json
import math

from src.label.multi_rater import (
    consensus, disagreements, fleiss_kappa, load_raters, pairwise, shared_items,
)
from src.label.pool import assign, build_pool, item_id as _iid, sanitize


def _predictions(tmp_path):
    paths = []
    for model in ("qwen2_5_vl_7b", "llava_next_mistral_7b"):
        p = tmp_path / f"{model}_test.jsonl"
        with open(p, "w") as f:
            for i in range(40):     # same questions for both models, half wrong
                f.write(json.dumps({
                    "item_id": _iid(model, f"fig{i}.png", f"q{i}"), "model": model,
                    "figure_id": f"fig{i}.png", "question": f"q{i}", "gold": "10",
                    "prediction": "12" if i % 2 else "10", "correct": i % 2 == 0,
                    "split": "test", "source": "human"}) + "\n")
        paths.append(p)
    return paths


class TestBuildPool:
    def test_keeps_only_incorrect(self, tmp_path):
        pool, key = build_pool(_predictions(tmp_path))
        assert len(pool) == len(key) == 40      # 20 wrong per model

    def test_ids_carried_through_unchanged(self, tmp_path):
        # the join key for predictions, Claude labels, activations and E4
        pool, key = build_pool(_predictions(tmp_path))
        expected = {_iid(m, f"fig{i}.png", f"q{i}")
                    for m in ("qwen2_5_vl_7b", "llava_next_mistral_7b") for i in range(1, 40, 2)}
        assert {p["item_id"] for p in pool} == expected

    def test_same_question_two_models_stays_two_items(self, tmp_path):
        pool, _ = build_pool(_predictions(tmp_path))
        assert len({p["item_id"] for p in pool}) == 40

    def test_pool_is_blind(self, tmp_path):
        pool, _ = build_pool(_predictions(tmp_path))
        for p in pool:
            assert set(p) == {"item_id", "image_path", "question",
                              "gold_answer", "model_answer"}
            assert "qwen" not in json.dumps(p) and "llava" not in json.dumps(p)

    def test_order_does_not_reveal_model(self, tmp_path):
        pool, key = build_pool(_predictions(tmp_path))
        model = {k["item_id"]: k["model"] for k in key}
        first_half = [model[p["item_id"]] for p in pool[:20]]
        assert 0 < first_half.count("qwen2_5_vl_7b") < 20

    def test_image_path_uses_split(self, tmp_path):
        pool, _ = build_pool(_predictions(tmp_path))
        assert all(p["image_path"].startswith("test/png/") for p in pool)

    def test_deterministic(self, tmp_path):
        paths = _predictions(tmp_path)
        assert build_pool(paths) == build_pool(paths)

    def test_duplicate_ids_raise(self, tmp_path):
        p = _predictions(tmp_path)[0]
        try:
            build_pool([p, p])
        except ValueError:
            return
        raise AssertionError("the same predictions file twice should raise")


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

    def test_consensus(self, tmp_path):
        _write_rater(tmp_path, "a", [("x", "structural"), ("y", "fabrication"), ("z", "structural")])
        _write_rater(tmp_path, "b", [("x", "structural"), ("y", "structural"), ("w", "ambiguous")])
        ratings, _ = load_raters(tmp_path)
        labels, unresolved = consensus(ratings, ["a", "b"])
        assert {l["item_id"]: (l["label"], l["how"]) for l in labels} == {
            "x": ("structural", "agreed"), "z": ("structural", "single"),
            "w": ("ambiguous", "single")}
        assert unresolved == ["y"]              # a tie is not broken by vote

    def test_adjudication_overrides(self, tmp_path):
        _write_rater(tmp_path, "a", [("y", "fabrication")])
        _write_rater(tmp_path, "b", [("y", "structural")])
        _write_rater(tmp_path, "adjudicated", [("y", "structural")])
        ratings, _ = load_raters(tmp_path)
        labels, unresolved = consensus(ratings, ["a", "b"])
        assert labels == [{"item_id": "y", "label": "structural", "how": "adjudicated"}]
        assert unresolved == []


def test_item_id_is_the_inf_md_scheme():
    # pinned: inference mints ids with this function, so any change to it
    # orphans every label and activation already written
    import hashlib
    raw = "qwen2_5_vl_7b\x1ffig.png\x1fWhat is A?".encode()
    assert _iid("qwen2_5_vl_7b", "fig.png", "What is A?") == hashlib.sha1(raw).hexdigest()[:12]
