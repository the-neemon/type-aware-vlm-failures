"""Tests for shards, final assembly, and the acceptance checks."""

import numpy as np
import pytest

from src.extract import cache_io
from src.label.pool import item_id
from src.probes.dataset import load_activations, validate_cache

MODEL = "qwen2_5_vl_7b"
LAYERS, POSITIONS, HIDDEN = [0, 1], ["vision_mean", "vision_max", "query_last", "query_mean"], 8


def make_items(n):
    items = []
    for i in range(n):
        fig, q = f"fig_{i // 2}.png", f"question {i}"
        items.append({"item_id": item_id(MODEL, fig, q), "figure_id": fig, "question": q})
    return items


def acts_for(items, seed=0):
    rng = np.random.default_rng(seed)
    return rng.standard_normal((len(items), len(LAYERS), len(POSITIONS), HIDDEN)).astype(np.float16)


def write(shard_dir, items, acts):
    ids = [it["item_id"] for it in items]
    cache_io.write_shard(shard_dir / cache_io.shard_name(ids), ids,
                         [it["figure_id"] for it in items], acts, LAYERS, POSITIONS)


def rows_for(items):
    return {it["item_id"]: {"item_id": it["item_id"], "model": MODEL, "figure_id": it["figure_id"],
                            "question": it["question"], "gold": "1", "prediction": "1",
                            "correct": True, "split": "test", "source": "human"}
            for it in items}


class TestShards:
    def test_roundtrip_and_no_temporary_left(self, tmp_path):
        items = make_items(3)
        acts = acts_for(items)
        write(tmp_path, items, acts)
        [path] = cache_io.shard_paths(tmp_path)
        s = cache_io.read_shard(path)
        assert s["item_ids"] == [it["item_id"] for it in items]
        assert s["layers"] == LAYERS and s["positions"] == POSITIONS
        np.testing.assert_array_equal(s["acts"], acts)
        assert not list(tmp_path.glob("*.tmp"))

    def test_float32_is_refused(self, tmp_path):
        items = make_items(1)
        with pytest.raises(ValueError, match="float16"):
            write(tmp_path, items, acts_for(items).astype(np.float32))

    def test_cached_ids_span_every_shard(self, tmp_path):
        items = make_items(5)
        write(tmp_path, items[:2], acts_for(items[:2]))
        write(tmp_path, items[2:], acts_for(items[2:]))
        assert cache_io.cached_item_ids(tmp_path) == {it["item_id"] for it in items}
        assert cache_io.cached_item_ids(tmp_path / "absent") == set()


class TestAssemble:
    def setup_method(self):
        self.items = make_items(5)
        self.acts = acts_for(self.items)

    def build(self, tmp_path, order=None):
        # shards written in an order unrelated to the dataset order
        write(tmp_path / "shards", self.items[3:], self.acts[3:])
        write(tmp_path / "shards", self.items[:3], self.acts[:3])
        order = order or [it["item_id"] for it in self.items]
        out = tmp_path / "test.npz"
        cache_io.assemble(tmp_path / "shards", order, LAYERS, POSITIONS, out)
        return out

    def test_rows_follow_the_given_order(self, tmp_path):
        order = [it["item_id"] for it in reversed(self.items)]
        out = self.build(tmp_path, order)
        with np.load(out, allow_pickle=True) as z:
            assert list(z["item_ids"]) == order
            assert list(z["figure_ids"]) == [it["figure_id"] for it in reversed(self.items)]
            for j, layer in enumerate(LAYERS):
                for k, pos in enumerate(POSITIONS):
                    np.testing.assert_array_equal(z[f"L{layer}_{pos}"], self.acts[::-1, j, k])

    def test_the_probe_loader_reads_it(self, tmp_path):
        out = self.build(tmp_path)
        report = validate_cache(out, expect_layers=len(LAYERS), expect_hidden=HIDDEN)
        assert report["n_items"] == 5 and not report["warnings"]
        by_layer, ids, figs = load_activations(out, "query_last")
        assert ids == [it["item_id"] for it in self.items] and len(figs) == 5
        np.testing.assert_array_equal(by_layer[1], self.acts[:, 1, 2])

    def test_duplicate_item_across_shards_raises(self, tmp_path):
        write(tmp_path, self.items[:2], self.acts[:2])
        write(tmp_path, self.items[1:3], self.acts[1:3])
        with pytest.raises(ValueError, match="more than one shard"):
            cache_io.assemble(tmp_path, [it["item_id"] for it in self.items[:3]],
                              LAYERS, POSITIONS, tmp_path / "out.npz")

    def test_missing_item_raises(self, tmp_path):
        write(tmp_path, self.items[:4], self.acts[:4])
        with pytest.raises(ValueError, match="1 missing"):
            cache_io.assemble(tmp_path, [it["item_id"] for it in self.items],
                              LAYERS, POSITIONS, tmp_path / "out.npz")

    def test_shard_from_another_layer_set_raises(self, tmp_path):
        write(tmp_path, self.items, self.acts)
        with pytest.raises(ValueError, match="holds layers"):
            cache_io.assemble(tmp_path, [it["item_id"] for it in self.items],
                              [0, 2], POSITIONS, tmp_path / "out.npz")


class TestChecks:
    def test_predictions_pass(self):
        items = make_items(3)
        assert cache_io.check_predictions(rows_for(items), MODEL, 3) == []

    def test_predictions_catch_count_field_and_hash_problems(self):
        items = make_items(3)
        rows = rows_for(items)
        first, second = list(rows)[:2]
        del rows[first]["source"]
        rows[second]["question"] = "edited question"
        problems = cache_io.check_predictions(rows, MODEL, 4)
        assert any("3 prediction rows" in p for p in problems)
        assert any("lacks ['source']" in p for p in problems)
        assert any("does not hash" in p for p in problems)

    def test_cache_passes(self, tmp_path):
        items = make_items(4)
        write(tmp_path / "shards", items, acts_for(items))
        cache_io.assemble(tmp_path / "shards", [it["item_id"] for it in items],
                          LAYERS, POSITIONS, tmp_path / "c.npz")
        assert cache_io.check_cache(tmp_path / "c.npz", rows_for(items),
                                    LAYERS, POSITIONS, HIDDEN) == []

    def test_cache_catches_mismatched_items_and_layers(self, tmp_path):
        items = make_items(4)
        write(tmp_path / "shards", items[:3], acts_for(items[:3]))
        cache_io.assemble(tmp_path / "shards", [it["item_id"] for it in items[:3]],
                          LAYERS, POSITIONS, tmp_path / "c.npz")
        problems = cache_io.check_cache(tmp_path / "c.npz", rows_for(items),
                                        [0, 1, 2], POSITIONS, HIDDEN)
        assert any("1 only in predictions" in p for p in problems)
        assert any("4 missing" in p for p in problems)


def test_append_prediction_is_one_line_per_row(tmp_path):
    path = tmp_path / "p" / "preds.jsonl"
    rows = rows_for(make_items(2))
    for row in rows.values():
        cache_io.append_prediction(row, path)
    assert cache_io.read_predictions(path) == rows
    assert cache_io.read_predictions(tmp_path / "absent.jsonl") == {}
