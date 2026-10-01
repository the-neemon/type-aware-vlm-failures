"""run_followups and the 1 Oct additions to run_chartqa_types, on planted signals."""

import argparse
import json

import numpy as np

from src.probes import run_chartqa_types as rct
from src.probes import run_followups as rf
from src.probes.question_type import question_type
from src.probes.run_synth_types import select_and_assemble

HIDDEN, LAYERS = 10, (0, 1)


def test_question_type_rule():
    assert question_type("What is the difference between A and B?") == "arithmetic"
    assert question_type("What is the average of the blue bars?") == "arithmetic"
    assert question_type("By how much did sales rise?") == "arithmetic"
    assert question_type("What is the ratio of men to women?") == "arithmetic"
    assert question_type("What is the value of the gray segment?") == "retrieval"
    assert question_type("Which year has the third largest value?") == "retrieval"
    assert question_type("How many bars are above 40?") == "retrieval"


def _cache(tmp, name, n_figs, signal, seed, fig_prefix="fig"):
    """Items over n_figs charts. `signal`: 'acts' plants the error in L1 only;
    'surface' makes errors exactly the long questions, and L1 a copy of length."""
    rng = np.random.default_rng(seed)
    preds, ids, figs = [], [], []
    for f in range(n_figs):
        for q in range(3):
            wrong = bool(rng.random() < 0.3)
            long_q = wrong if signal == "surface" else bool(rng.random() < 0.5)
            iid = f"{name}_{f}_{q}"
            preds.append({"item_id": iid, "figure_id": f"{fig_prefix}{f}",
                          "question": "q" * (40 if long_q else 10), "gold": "5",
                          "source": "human" if q % 2 else "augmented",
                          "n_vision_tokens": 100, "correct": not wrong})
            ids.append(iid)
            figs.append(f"{fig_prefix}{f}")
    acts = {}
    for L in LAYERS:
        A = rng.normal(size=(len(preds), HIDDEN))
        if L == 1:
            for i, r in enumerate(preds):
                if signal == "acts":
                    A[i, 0] += 3.0 * (not r["correct"])
                else:
                    A[i, 0] += 3.0 * (len(r["question"]) > 20)
        acts[f"L{L}_query_last"] = A.astype(np.float16)
    np.savez_compressed(tmp / f"{name}.npz", item_ids=np.array(ids),
                        figure_ids=np.array(figs, dtype=object), **acts)
    (tmp / f"{name}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in preds))
    return tmp / f"{name}.npz", tmp / f"{name}.jsonl"


def _args(tmp, npz, preds, **kw):
    a = dict(activations=npz, predictions=preds, drop_not_an_error=None, val_human=None,
             seeds=[42, 1], layers=None, l2=[1.0, 100.0], jobs=2, out=tmp / "r.json")
    a.update(kw)
    return argparse.Namespace(**a)


def test_binary_finds_the_layer_and_survives_residualising(tmp_path):
    npz, preds = _cache(tmp_path, "main", 150, "acts", 0)
    vnpz, vpreds = _cache(tmp_path, "vh", 60, "acts", 1, fig_prefix="val")
    rf.run_binary(_args(tmp_path, npz, preds, val_human=[vnpz, vpreds]))
    r = json.loads((tmp_path / "r.json").read_text())
    for seed in ("42", "1"):
        assert r["seeds"][seed]["plain"]["layer"] == 1
        assert r["seeds"][seed]["plain"]["test_auroc"] > 0.85
        assert r["seeds"][seed]["residualised"]["test_auroc"] > 0.85
    assert r["val_human_as_test"]["auroc"] > 0.85
    assert r["val_human_as_training"]["test_auroc"] > 0.85
    assert 0 <= r["calibration"]["platt_on_val"]["ece"] < 0.2


def test_residualising_removes_a_surface_only_signal(tmp_path):
    npz, preds = _cache(tmp_path, "main", 150, "surface", 0)
    rf.run_binary(_args(tmp_path, npz, preds, seeds=[42]))
    r = json.loads((tmp_path / "r.json").read_text())["seeds"]["42"]
    assert r["plain"]["test_auroc"] > 0.95
    assert r["residualised"]["test_auroc"] < 0.7


def test_val_human_sharing_a_chart_is_refused(tmp_path):
    npz, preds = _cache(tmp_path, "main", 30, "acts", 0)
    vnpz, vpreds = _cache(tmp_path, "vh", 10, "acts", 1)          # same figure names
    try:
        rf.run_binary(_args(tmp_path, npz, preds, val_human=[vnpz, vpreds]))
    except AssertionError as e:
        assert "shares" in str(e)
    else:
        raise AssertionError("expected the shared charts to be refused")


def test_grid_features():
    manifest = {"figure_id": "f", "categories": ["A", "B", "C"], "values": [34, 50, 134],
                "axis_min": 0, "axis_max": 200, "tick_density": "medium"}   # step 20
    items = [{"figure_id": "f", "asks_about": a} for a in ("A", "B", "C", "Missing")]

    import pathlib, tempfile
    with tempfile.TemporaryDirectory() as d:
        path = pathlib.Path(d) / "m.jsonl"
        path.write_text(json.dumps(manifest) + "\n")
        G = rf.grid_features(items, [path])
    # 34: nearest gridline 40 (6 away = 0.3 steps), nearest ten 30 (4 away)
    np.testing.assert_allclose(G[0], [0.3, 0.4, 6 / 34, 4 / 34], rtol=1e-5)
    np.testing.assert_allclose(G[1], [0.5, 0.0, 10 / 50, 0.0], rtol=1e-5)
    np.testing.assert_allclose(G[2], [0.3, 0.4, 6 / 134, 4 / 134], rtol=1e-5)
    assert (G[3] == 0).all()


def _types_world(tmp, n_figs=160):
    kinds = ["correct"] * 5 + ["structural", "computation"]
    rng = np.random.default_rng(0)
    preds, labels, ids, figs = [], [], [], []
    for f in range(n_figs):
        for q in range(3):
            kind = kinds[(f + q) % len(kinds)]
            arith = kind == "computation" or (kind == "correct" and q == 0)
            iid = f"i{f}_{q}"
            preds.append({"item_id": iid, "figure_id": f"fig{f}",
                          "question": ("What is the difference?" if arith
                                       else "What is the value?"),
                          "gold": "5", "source": "human", "n_vision_tokens": 100,
                          "correct": kind == "correct", "kind": kind})
            if kind != "correct":
                labels.append({"item_id": iid, "label": kind})
            ids.append(iid)
            figs.append(f"fig{f}")
    acts = {}
    for L in LAYERS:
        A = rng.normal(size=(len(preds), HIDDEN))
        if L == 1:
            for i, r in enumerate(preds):
                A[i, 0] += 3.0 * (r["kind"] == "structural")
                A[i, 1] += 3.0 * (r["kind"] == "computation")
        acts[f"L{L}_query_last"] = A.astype(np.float16)
    np.savez_compressed(tmp / "a.npz", item_ids=np.array(ids),
                        figure_ids=np.array(figs, dtype=object), **acts)
    (tmp / "p.jsonl").write_text("".join(json.dumps(r) + "\n" for r in preds))
    (tmp / "l.jsonl").write_text("".join(json.dumps(r) + "\n" for r in labels))
    return tmp / "a.npz", tmp / "p.jsonl", tmp / "l.jsonl"


def test_question_type_filter_keeps_only_that_type(tmp_path):
    npz, preds, labels = _types_world(tmp_path)
    items, _ = rct.load_items(npz, preds, labels, qtype="arithmetic")
    assert {it["cls"] for it in items} == {"C", "K"}
    items, _ = rct.load_items(npz, preds, labels, qtype="retrieval")
    assert {it["cls"] for it in items} == {"C", "S"}


def test_residualised_cell_keeps_an_activation_signal(tmp_path):
    npz, preds, labels = _types_world(tmp_path)
    items, _ = rct.load_items(npz, preds, labels)
    cls = np.array([it["cls"] for it in items])
    folds = np.array([it["fold"] for it in items])
    figs = np.array([it["figure_id"] for it in items])
    res = {1: rct._cell(("structural", ("resid", 1), items, (1.0, 100.0)))[2]}
    assert set(res[1]) == set(range(rct.N_FOLDS))
    scores, _ = select_and_assemble(res, len(items), folds)
    assert rct.matrix(scores, cls, figs, {"S_vs_C": ("S", "C")})["S_vs_C"]["auroc"] > 0.85
