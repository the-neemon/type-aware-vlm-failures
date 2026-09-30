"""run_chartqa_types on a planted signal: misreads along one direction, arithmetic
errors along another, not_an_error items indistinguishable from correct ones."""

import json

import numpy as np

from src.probes import run_chartqa_types as rct
from src.probes.run_synth_types import select_and_assemble

HIDDEN, LAYERS = 12, (0, 1)
KINDS = ["correct"] * 6 + ["structural", "computation", "not_an_error", "ambiguous"]


def _world(tmp, n_figs=160, n_fabrication=8, seed=0):
    rng = np.random.default_rng(seed)
    preds, labels, ids, figs = [], [], [], []
    for f in range(n_figs):
        kinds = [KINDS[(f + q) % len(KINDS)] for q in range(3)]
        if f < n_fabrication:
            kinds[0] = "fabrication"
        if f == n_figs - 1:
            kinds[0] = "unlabelled"
        for q, kind in enumerate(kinds):
            iid = f"i{f}_{q}"
            preds.append({"item_id": iid, "figure_id": f"fig{f}", "question": "q" * (q + 3),
                          "gold": "5", "source": "human" if q % 2 else "augmented",
                          "n_vision_tokens": 100, "correct": kind == "correct",
                          "kind": kind})
            if kind not in ("correct", "unlabelled"):
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


def test_classes_probes_and_a_diagonal_matrix(tmp_path):
    npz, preds, labels = _world(tmp_path)
    items, unlabelled = rct.load_items(npz, preds, labels)
    assert unlabelled == 1
    cls = np.array([it["cls"] for it in items])
    assert "A" not in set(cls) and (cls == "F").sum() == 8
    probes = rct.trainable(cls)
    assert set(probes) == {"structural", "computation", "struct_vs_comp"}   # no F probe
    assert "F_vs_C" in rct.reportable(cls)

    folds = np.array([it["fold"] for it in items])
    figs = np.array([it["figure_id"] for it in items])
    m = {}
    for p in probes:
        res = {L: rct._cell((p, L, items, (1.0, 100.0)))[2] for L in LAYERS}
        scores, chosen = select_and_assemble(res, len(items), folds)
        assert all(c["layer"] == 1 for c in chosen.values())
        m[p] = rct.matrix(scores, cls, figs, rct.reportable(cls))
    assert m["structural"]["S_vs_C"]["auroc"] > 0.85
    assert m["computation"]["K_vs_C"]["auroc"] > 0.85
    assert m["struct_vs_comp"]["S_vs_K"]["auroc"] > 0.85
    assert abs(m["structural"]["K_vs_C"]["auroc"] - 0.5) < 0.15       # blind to the other type
    assert abs(m["structural"]["N_vs_C"]["auroc"] - 0.5) < 0.15       # not_an_error looks correct


def test_surface_baseline_runs_on_the_same_items(tmp_path):
    npz, preds, labels = _world(tmp_path)
    items, _ = rct.load_items(npz, preds, labels)
    res = rct._cell(("structural", "surface", items, (1.0,)))[2]
    assert set(res) == set(range(rct.N_FOLDS))
