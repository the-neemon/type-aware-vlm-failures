"""Tests for probe dataset assembly (P5.1 to P5.4).

Everything here runs on synthetic fixtures built to the contract in inf.md, so
the assembly layer is verified before any real inference exists. When the real
cache lands it should drop straight in.

pytest-style to match src/eval and src/probes. Run with

    python3 -m src.probes.test_dataset

from the repository root. Note it is `-m`, not a direct path: these modules
import each other as `src.*`, which needs the repo root on sys.path.
"""

from __future__ import annotations

import json
import pathlib
import tempfile

import numpy as np

from src.label.pool import item_id
from src.probes.dataset import (
    AMBIGUOUS, ProbeConfig, assert_no_figure_leak, assign_figure_splits,
    build_dataset, load_activations, load_predictions, resolve_labels,
)

MODEL = "qwen2_5_vl_7b"
LAYERS = (0, 3, 7)
HIDDEN = 8


# ---------------------------------------------------------------------------
# Fixtures: a tiny world built to the inf.md contract
# ---------------------------------------------------------------------------

def _world(n_figures=12, questions_per_figure=3, seed=0):
    """Build matching predictions, activations and labels.

    Deliberately several questions per figure, which is what makes the
    figure-level split matter.
    """
    rng = np.random.default_rng(seed)
    rows, ids, figs = [], [], []
    for f in range(n_figures):
        fid = f"fig_{f:03d}.png"
        for q in range(questions_per_figure):
            question = f"what is value {q} in figure {f}?"
            iid = item_id(MODEL, fid, question)
            correct = bool((f + q) % 3)          # ~1/3 incorrect
            rows.append({
                "item_id": iid, "model": MODEL, "figure_id": fid,
                "question": question, "gold": "7",
                "prediction": "7" if correct else "9",
                "correct": correct, "split": "test", "source": "human",
            })
            ids.append(iid)
            figs.append(fid)

    acts = {f"L{L}_query_last": rng.normal(size=(len(ids), HIDDEN)).astype(np.float16)
            for L in LAYERS}
    acts |= {f"L{L}_vision_mean": rng.normal(size=(len(ids), HIDDEN)).astype(np.float16)
             for L in LAYERS}
    return rows, ids, figs, acts


def _write_world(tmp: pathlib.Path, rows, ids, figs, acts, shuffle_preds=False):
    preds = tmp / "preds.jsonl"
    ordered = list(rows)
    if shuffle_preds:
        # Reverse the manifest relative to the cache. If anything joins by
        # position instead of by item_id, this is what exposes it.
        ordered = ordered[::-1]
    with open(preds, "w", encoding="utf-8") as f:
        for r in ordered:
            f.write(json.dumps(r) + "\n")

    npz = tmp / "acts.npz"
    np.savez_compressed(npz, item_ids=np.array(ids),
                        figure_ids=np.array(figs, dtype=object), **acts)
    return preds, npz


def _write_labels(tmp: pathlib.Path, rows, rater="claude", pattern=None):
    d = tmp / "annotations"
    d.mkdir(exist_ok=True)
    with open(d / f"{rater}.jsonl", "w", encoding="utf-8") as f:
        for i, r in enumerate(x for x in rows if not x["correct"]):
            label = pattern(i) if pattern else ("structural", "fabrication")[i % 2]
            f.write(json.dumps({"item_id": r["item_id"], "label": label}) + "\n")
    return d


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def test_load_activations_selects_one_position():
    with tempfile.TemporaryDirectory() as t:
        tmp = pathlib.Path(t)
        rows, ids, figs, acts = _world()
        _, npz = _write_world(tmp, rows, ids, figs, acts)

        by_layer, item_ids, figure_ids = load_activations(npz, "query_last")
        assert sorted(by_layer) == list(LAYERS)
        assert item_ids == ids
        assert figure_ids == figs
        assert by_layer[3].shape == (len(ids), HIDDEN)

        # a different position gives different numbers, same shape
        other, _, _ = load_activations(npz, "vision_mean")
        assert other[3].shape == by_layer[3].shape
        assert not np.allclose(other[3].astype(float), by_layer[3].astype(float))


def test_load_activations_rejects_cache_without_item_ids():
    with tempfile.TemporaryDirectory() as t:
        npz = pathlib.Path(t) / "bad.npz"
        np.savez_compressed(npz, L0_query_last=np.zeros((4, HIDDEN)))
        try:
            load_activations(npz, "query_last")
        except ValueError as e:
            assert "item_ids" in str(e)
        else:
            raise AssertionError("a cache with no item_ids must be rejected")


def test_load_activations_rejects_row_count_mismatch():
    with tempfile.TemporaryDirectory() as t:
        npz = pathlib.Path(t) / "bad.npz"
        np.savez_compressed(npz, item_ids=np.array(["a", "b", "c"]),
                            L0_query_last=np.zeros((2, HIDDEN)))
        try:
            load_activations(npz, "query_last")
        except ValueError as e:
            assert "rows" in str(e)
        else:
            raise AssertionError("row-count mismatch must be rejected")


def test_load_predictions_requires_item_id_and_rejects_duplicates():
    with tempfile.TemporaryDirectory() as t:
        p = pathlib.Path(t) / "p.jsonl"
        p.write_text(json.dumps({"figure_id": "a.png", "correct": True}) + "\n")
        try:
            load_predictions(p)
        except ValueError as e:
            assert "item_id" in str(e)
        else:
            raise AssertionError("a row without item_id must be rejected")

        row = {"item_id": "dup", "figure_id": "a.png", "correct": True}
        p.write_text(json.dumps(row) + "\n" + json.dumps(row) + "\n")
        try:
            load_predictions(p)
        except ValueError as e:
            assert "duplicate" in str(e)
        else:
            raise AssertionError("duplicate item_ids must be rejected")


# ---------------------------------------------------------------------------
# The join. This is the test that matters.
# ---------------------------------------------------------------------------

def test_join_is_by_item_id_not_by_position():
    """Reversing the manifest must not change a single label.

    If anything joined by row order, every y would be wrong while every shape
    stayed right, and the probe would report a plausible AUROC for nonsense.
    """
    with tempfile.TemporaryDirectory() as t:
        tmp = pathlib.Path(t)
        rows, ids, figs, acts = _world()

        p1, npz = _write_world(tmp, rows, ids, figs, acts, shuffle_preds=False)
        d1 = build_dataset("binary", npz, p1)

        tmp2 = tmp / "reversed"; tmp2.mkdir()
        p2, npz2 = _write_world(tmp2, rows, ids, figs, acts, shuffle_preds=True)
        d2 = build_dataset("binary", npz2, p2)

        for split in ("train", "val", "test"):
            assert list(d1.item_ids[split]) == list(d2.item_ids[split])
            assert np.array_equal(d1.y[split], d2.y[split])
            for L in LAYERS:
                assert np.array_equal(d1.X[split][L], d2.X[split][L])


def test_activation_rows_follow_their_own_item_id():
    """Each row of X must be the row the cache stored for that item_id."""
    with tempfile.TemporaryDirectory() as t:
        tmp = pathlib.Path(t)
        rows, ids, figs, acts = _world()
        preds, npz = _write_world(tmp, rows, ids, figs, acts)
        # alignment, not scaling, is under test here, so compare raw rows
        ds = build_dataset("binary", npz, preds,
                           config=ProbeConfig(standardize=False))

        raw = acts["L3_query_last"]
        pos = {iid: i for i, iid in enumerate(ids)}
        for split in ("train", "val", "test"):
            for k, iid in enumerate(ds.item_ids[split]):
                assert np.array_equal(ds.X[split][3][k], raw[pos[iid]])


# ---------------------------------------------------------------------------
# Splits
# ---------------------------------------------------------------------------

def test_split_assignment_is_deterministic_and_stable():
    figs = [f"f{i}.png" for i in range(200)]
    a = assign_figure_splits(figs, seed=42)
    b = assign_figure_splits(figs, seed=42)
    assert a == b

    # adding figures must not reshuffle the ones already assigned
    more = assign_figure_splits(figs + [f"g{i}.png" for i in range(50)], seed=42)
    assert all(more[f] == a[f] for f in figs)

    # a different seed genuinely re-partitions
    assert assign_figure_splits(figs, seed=7) != a


def test_assert_no_figure_leak_catches_a_leak():
    assert_no_figure_leak({"train": ["a.png"], "test": ["b.png"]})
    try:
        assert_no_figure_leak({"train": ["a.png"], "test": ["a.png"]})
    except AssertionError as e:
        assert "a.png" in str(e)
    else:
        raise AssertionError("a figure in two splits must be caught")


def test_built_dataset_never_leaks_a_figure():
    with tempfile.TemporaryDirectory() as t:
        tmp = pathlib.Path(t)
        rows, ids, figs, acts = _world(n_figures=30)
        preds, npz = _write_world(tmp, rows, ids, figs, acts)
        ds = build_dataset("binary", npz, preds)
        seen = {}
        for split in ("train", "val", "test"):
            for f in ds.figure_ids[split]:
                assert seen.setdefault(f, split) == split


def test_all_three_tasks_share_one_split():
    """A figure must land in the same split for every probe.

    Otherwise an item is train for the binary probe and test for a type probe,
    and comparing the two numbers means nothing.
    """
    with tempfile.TemporaryDirectory() as t:
        tmp = pathlib.Path(t)
        rows, ids, figs, acts = _world(n_figures=30)
        preds, npz = _write_world(tmp, rows, ids, figs, acts)
        ann = _write_labels(tmp, rows)

        where = {}
        for task in ("binary", "structural", "fabrication"):
            ds = build_dataset(task, npz, preds, ann)
            for split in ("train", "val", "test"):
                for f in ds.figure_ids[split]:
                    assert where.setdefault(f, split) == split


# ---------------------------------------------------------------------------
# Populations and labels
# ---------------------------------------------------------------------------

def test_binary_task_keeps_both_classes():
    with tempfile.TemporaryDirectory() as t:
        tmp = pathlib.Path(t)
        rows, ids, figs, acts = _world()
        preds, npz = _write_world(tmp, rows, ids, figs, acts)
        ds = build_dataset("binary", npz, preds)
        total = sum(ds.n(s) for s in ("train", "val", "test"))
        assert total == len(rows), "binary probe must see every item"
        ys = np.concatenate([ds.y[s] for s in ("train", "val", "test")])
        assert set(np.unique(ys)) == {0.0, 1.0}, "needs both correct and incorrect"


def test_type_tasks_see_only_errors_and_drop_ambiguous():
    with tempfile.TemporaryDirectory() as t:
        tmp = pathlib.Path(t)
        rows, ids, figs, acts = _world()
        preds, npz = _write_world(tmp, rows, ids, figs, acts)
        # every third labelled item is ambiguous
        ann = _write_labels(tmp, rows,
                            pattern=lambda i: AMBIGUOUS if i % 3 == 2
                            else ("structural", "fabrication")[i % 2])
        ds = build_dataset("structural", npz, preds, ann)

        n_err = sum(1 for r in rows if not r["correct"])
        kept = sum(ds.n(s) for s in ("train", "val", "test"))
        assert ds.dropped_ambiguous > 0
        assert kept + ds.dropped_ambiguous == n_err
        # and no correct item slipped in
        by_id = {r["item_id"]: r for r in rows}
        for split in ("train", "val", "test"):
            assert all(not by_id[i]["correct"] for i in ds.item_ids[split])


def test_structural_and_fabrication_are_complementary_labels():
    with tempfile.TemporaryDirectory() as t:
        tmp = pathlib.Path(t)
        rows, ids, figs, acts = _world()
        preds, npz = _write_world(tmp, rows, ids, figs, acts)
        ann = _write_labels(tmp, rows)
        s = build_dataset("structural", npz, preds, ann)
        f = build_dataset("fabrication", npz, preds, ann)
        for split in ("train", "val", "test"):
            assert list(s.item_ids[split]) == list(f.item_ids[split])
            assert np.array_equal(s.y[split], 1.0 - f.y[split])


def test_errors_only_makes_the_two_type_probes_algebraically_linked():
    """Documents a real limitation rather than asserting a feature.

    With rest = "the other failure type", the fabrication label is the inverted
    structural label on the same items. So any probe scores the two exactly
    oppositely and E3's off-diagonal is forced, not measured. This test exists
    so nobody later reports that off-diagonal as evidence of separability.
    """
    with tempfile.TemporaryDirectory() as t:
        tmp = pathlib.Path(t)
        rows, ids, figs, acts = _world(n_figures=30)
        preds, npz = _write_world(tmp, rows, ids, figs, acts)
        ann = _write_labels(tmp, rows)
        cfg = ProbeConfig(type_probe_rest="errors_only")
        s_ds = build_dataset("structural", npz, preds, ann, cfg)
        f_ds = build_dataset("fabrication", npz, preds, ann, cfg)
        for split in ("train", "val", "test"):
            assert np.array_equal(s_ds.y[split], 1.0 - f_ds.y[split])


def test_include_correct_makes_the_type_probes_genuinely_different():
    """With correct items in the negative class the labels stop being flips."""
    with tempfile.TemporaryDirectory() as t:
        tmp = pathlib.Path(t)
        rows, ids, figs, acts = _world(n_figures=30)
        preds, npz = _write_world(tmp, rows, ids, figs, acts)
        ann = _write_labels(tmp, rows)
        cfg = ProbeConfig(type_probe_rest="include_correct")
        s_ds = build_dataset("structural", npz, preds, ann, cfg)
        f_ds = build_dataset("fabrication", npz, preds, ann, cfg)

        n_all = sum(s_ds.n(x) for x in ("train", "val", "test"))
        n_err = sum(1 for r in rows if not r["correct"])
        assert n_all > n_err, "correct items must now be in the population"

        flipped = all(np.array_equal(s_ds.y[x], 1.0 - f_ds.y[x])
                      for x in ("train", "val", "test"))
        assert not flipped, "the two probes must no longer be label flips"

        # a correct item is a negative for BOTH probes
        by_id = {r["item_id"]: r for r in rows}
        for split in ("train", "val", "test"):
            for k, iid in enumerate(s_ds.item_ids[split]):
                if by_id[iid]["correct"]:
                    assert s_ds.y[split][k] == 0.0 and f_ds.y[split][k] == 0.0


def test_classes_field_distinguishes_all_three_groups():
    """E3 needs to tell correct items from the two failure types.

    Deriving the class from the binary `y` would relabel every correct item as
    the opposite failure type and collapse three groups into two, which forces
    the cross-transfer off-diagonal. This is the regression guard for that.
    """
    with tempfile.TemporaryDirectory() as t:
        tmp = pathlib.Path(t)
        rows, ids, figs, acts = _world(n_figures=30)
        preds, npz = _write_world(tmp, rows, ids, figs, acts)
        ann = _write_labels(tmp, rows)

        ds = build_dataset("structural", npz, preds, ann,
                           ProbeConfig(type_probe_rest="include_correct"))
        seen = set()
        for split in ("train", "val", "test"):
            seen |= set(ds.classes[split].tolist())
            assert len(ds.classes[split]) == ds.n(split)
        assert seen == {"structural", "fabrication", "correct"}, seen

        errors_only = build_dataset("structural", npz, preds, ann,
                                    ProbeConfig(type_probe_rest="errors_only"))
        seen2 = set()
        for split in ("train", "val", "test"):
            seen2 |= set(errors_only.classes[split].tolist())
        assert "correct" not in seen2


def test_standardisation_uses_train_statistics_only():
    """Val and test must be scaled with TRAIN mean and std, never their own.

    If each split were standardised with its own statistics, the test set's
    distribution would leak into the features the probe is scored on.
    """
    from src.probes.dataset import standardize_by_train
    rng = np.random.default_rng(0)
    X = {"train": {0: rng.normal(5.0, 2.0, size=(200, 4))},
         "val":   {0: rng.normal(50.0, 9.0, size=(80, 4))},     # deliberately shifted
         "test":  {0: rng.normal(-30.0, 0.5, size=(80, 4))}}
    Z = standardize_by_train(X)
    assert np.allclose(Z["train"][0].mean(axis=0), 0.0, atol=1e-5)
    assert np.allclose(Z["train"][0].std(axis=0), 1.0, atol=1e-4)
    # val/test keep their shift relative to train, proving train stats were used
    assert Z["val"][0].mean() > 10
    assert Z["test"][0].mean() < -10


def test_standardisation_leaves_constant_features_finite():
    from src.probes.dataset import standardize_by_train
    X = {"train": {0: np.ones((50, 3))}, "val": {0: np.ones((10, 3))},
         "test": {0: np.ones((10, 3))}}
    Z = standardize_by_train(X)
    assert all(np.isfinite(Z[s][0]).all() for s in Z)


def test_build_dataset_standardises_by_default_and_can_be_disabled():
    with tempfile.TemporaryDirectory() as t:
        tmp = pathlib.Path(t)
        rows, ids, figs, acts = _world(n_figures=30)
        preds, npz = _write_world(tmp, rows, ids, figs, acts)
        on = build_dataset("binary", npz, preds)
        off = build_dataset("binary", npz, preds, config=ProbeConfig(standardize=False))
        assert abs(float(on.X["train"][3].mean())) < 1e-4
        assert on.X["train"][3].dtype == np.float32
        assert off.X["train"][3].dtype == np.float16    # raw, untouched


def test_resolve_labels_reads_a_single_llm_label_file():
    """Claude's labels are one file outside annotations/, per docs/annotation.md."""
    with tempfile.TemporaryDirectory() as t:
        f = pathlib.Path(t) / "qwen2_5_vl_7b_test.claude.jsonl"
        f.write_text("\n".join([
            json.dumps({"item_id": "a", "label": "structural", "rationale": "misread"}),
            json.dumps({"item_id": "b", "label": "fabrication", "rationale": "absent"}),
            json.dumps({"item_id": "c", "label": "nonsense"}),        # rejected
            "not json",                                              # rejected
            json.dumps({"item_id": "a", "label": "ambiguous"}),      # later line wins
        ]) + "\n")
        labels, stats = resolve_labels(f)
        assert labels == {"a": "ambiguous", "b": "fabrication"}
        assert stats["n_raters"] == 1 and stats["n_bad_lines"] == 2


def test_type_task_accepts_a_label_file_path():
    with tempfile.TemporaryDirectory() as t:
        tmp = pathlib.Path(t)
        rows, ids, figs, acts = _world(n_figures=30)
        preds, npz = _write_world(tmp, rows, ids, figs, acts)
        d = _write_labels(tmp, rows)                   # directory with claude.jsonl
        via_dir = build_dataset("structural", npz, preds, d)
        via_file = build_dataset("structural", npz, preds, d / "claude.jsonl")
        for split in ("train", "val", "test"):
            assert list(via_dir.item_ids[split]) == list(via_file.item_ids[split])
            assert np.array_equal(via_dir.y[split], via_file.y[split])


def test_config_rejects_unknown_rest_definition():
    try:
        ProbeConfig(type_probe_rest="everything_else")
    except ValueError as e:
        assert "type_probe_rest" in str(e)
    else:
        raise AssertionError("an unknown rest definition must be rejected")


def test_resolve_labels_majority_and_ties():
    with tempfile.TemporaryDirectory() as t:
        d = pathlib.Path(t) / "annotations"; d.mkdir()
        def put(rater, recs):
            with open(d / f"{rater}.jsonl", "w", encoding="utf-8") as fh:
                for r in recs:
                    fh.write(json.dumps(r) + "\n")

        put("claude", [{"item_id": "x", "label": "structural"},
                       {"item_id": "y", "label": "structural"}])
        put("alice",  [{"item_id": "x", "label": "structural"},
                       {"item_id": "y", "label": "fabrication"}])
        put("bob",    [{"item_id": "x", "label": "fabrication"}])

        labels, stats = resolve_labels(d)
        assert labels["x"] == "structural"          # 2 of 3
        assert labels["y"] == AMBIGUOUS             # 1 v 1 tie
        assert stats["n_raters"] == 3
        assert stats["n_ties_to_ambiguous"] == 1


# ---------------------------------------------------------------------------
# End to end into the probe primitives
# ---------------------------------------------------------------------------

def test_dataset_feeds_sweep_layers():
    from src.probes.sweep import evaluate_at, select_layer, sweep_layers

    with tempfile.TemporaryDirectory() as t:
        tmp = pathlib.Path(t)
        rows, ids, figs, acts = _world(n_figures=40)
        preds, npz = _write_world(tmp, rows, ids, figs, acts)
        ds = build_dataset("binary", npz, preds)

        res = sweep_layers(ds.X["train"], ds.y["train"], ds.X["val"], ds.y["val"])
        layer = select_layer(res)
        assert layer in LAYERS
        out = evaluate_at(layer, res, ds.X["test"], ds.y["test"],
                          figure_ids=ds.figure_ids["test"])
        assert "auroc" in out and "ci95" in out
        # random features: AUROC should be unremarkable, but must be a number
        assert 0.0 <= out["auroc"] <= 1.0


def test_config_rejects_unknown_position():
    try:
        ProbeConfig(position="last_token")
    except ValueError as e:
        assert "position" in str(e)
    else:
        raise AssertionError("an unknown position must be rejected")


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"  ok  {fn.__name__}")
    print(f"\n{len(fns)} tests passed")


def test_the_agreed_label_set_is_read_and_only_two_labels_train_type_probes():
    """computation and not_an_error (27 Sep label set) are excluded, never negatives."""
    with tempfile.TemporaryDirectory() as t:
        tmp = pathlib.Path(t)
        rows, ids, figs, acts = _world()
        preds, npz = _write_world(tmp, rows, ids, figs, acts)
        five = ("structural", "fabrication", "computation", "not_an_error", "ambiguous")
        f = tmp / "labels.claude.jsonl"
        errors = [r for r in rows if not r["correct"]]
        f.write_text("".join(json.dumps({"item_id": r["item_id"], "label": five[i % 5]}) + "\n"
                             for i, r in enumerate(errors)))
        labels, stats = resolve_labels(f)
        assert set(labels.values()) == set(five) and stats["n_bad_lines"] == 0

        ds = build_dataset("structural", npz, preds, f)
        kept = [i for s in ("train", "val", "test") for i in ds.item_ids[s]]
        assert {labels[i] for i in kept} == {"structural", "fabrication"}
        assert ds.dropped_ambiguous == sum(labels[r["item_id"]] not in five[:2] for r in errors)


# ---------------------------------------------------------------------------
# Layer count from the cache (Qwen 28, LLaVA-NeXT 32)
# ---------------------------------------------------------------------------

def _cache_with_layers(path, n_layers, positions=("vision_mean", "query_last")):
    arrays = {f"L{L}_{p}": np.zeros((2, 4), np.float16) for L in range(n_layers) for p in positions}
    np.savez(path, item_ids=np.array(["a", "b"]), figure_ids=np.array(["f", "f"]), **arrays)
    return path


def test_cached_layers_reads_every_layer_including_llavas_deepest(tmp_path):
    from src.probes.dataset import cached_layers
    path = _cache_with_layers(tmp_path / "llava.npz", 32)
    assert cached_layers(path) == list(range(32))
    assert cached_layers(_cache_with_layers(tmp_path / "qwen.npz", 28)) == list(range(28))


def test_cached_layers_without_the_position_raises(tmp_path):
    import pytest
    from src.probes.dataset import cached_layers
    path = _cache_with_layers(tmp_path / "c.npz", 3, positions=("vision_mean",))
    with pytest.raises(ValueError, match="query_last"):
        cached_layers(path)


def test_the_probe_scripts_no_longer_hard_code_28_layers():
    root = pathlib.Path(__file__).parent
    for name in ("run_sweep.py", "run_synth.py", "run_synth_types.py"):
        text = (root / name).read_text()
        assert "N_LAYERS" not in text and "range(28)" not in text, name
