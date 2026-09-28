"""run_synth_types on a planted signal: misreads along one direction, made-up
numbers along another. The matrix must come out diagonal."""

import json
import pathlib

import numpy as np
import pytest

from src.probes import run_synth_types as rst

HIDDEN, LAYERS = 12, (0, 1)


def _pilot(tmp: pathlib.Path, name: str, n_figs: int, signal: bool, rng):
    rows, ids, figs, manifest = [], [], [], []
    for f in range(n_figs):
        fid = f"{name}_{f:04d}"
        manifest.append({"figure_id": fid, "axis_max": 100, "tick_density": "sparse",
                         "categories": ["Apple", "Mango"]})
        misread, made_up = f % 4 == 0, f % 3 == 0
        for absent, pred in ((False, "20" if misread else "40"), (True, "45" if made_up else "0")):
            iid = f"{fid}-{int(absent)}"
            rows.append({"item_id": iid, "figure_id": fid, "question": "q", "gold":
                         "not present" if absent else "40", "prediction": pred,
                         "template": "absent_value" if absent else "read_value",
                         "absent": absent, "phrasing": "value_of",
                         "asks_about": "Guava" if absent else "Apple", "lookalike_of": None,
                         "correct": not absent and not misread})
            ids.append(iid)
            figs.append(fid)
    acts = {}
    for L in LAYERS:
        A = rng.normal(size=(len(rows), HIDDEN))
        if signal and L == 1:
            for i, r in enumerate(rows):
                A[i, 0] += 3.0 * (not r["absent"] and not r["correct"])      # structural
                A[i, 1] += 3.0 * (r["absent"] and r["prediction"] != "0")    # fabrication
                A[i, 2] += 3.0 * r["absent"]                                  # absence
        acts[f"L{L}_query_last"] = A.astype(np.float16)
    np.savez_compressed(tmp / f"{name}.npz", item_ids=np.array(ids),
                        figure_ids=np.array(figs, dtype=object), **acts)
    (tmp / f"{name}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    (tmp / f"{name}_manifest.jsonl").write_text("".join(json.dumps(m) + "\n" for m in manifest))
    return (name, str(tmp / f"{name}.npz"), str(tmp / f"{name}.jsonl"),
            str(tmp / f"{name}_manifest.jsonl"))


def _run(tmp, signal):
    rng = np.random.default_rng(0)
    pilots = [_pilot(tmp, "pa", 120, signal, rng), _pilot(tmp, "pb", 120, signal, rng)]
    items = rst.load_items(pilots)
    cls = np.array([it["cls"] for it in items])
    folds = np.array([it["fold"] for it in items])
    figs = np.array([it["figure_id"] for it in items])
    out = {}
    for probe in rst.PROBES:
        results = {L: rst._cell((probe, L, items, (1.0, 100.0)))[2] for L in LAYERS}
        scores, chosen = rst.select_and_assemble(results, len(items), folds)
        assert not np.isnan(scores).any()
        out[probe] = (rst.matrix(scores, cls, figs), chosen)
    return cls, folds, out


def test_classes_and_folds(tmp_path):
    cls, folds, _ = _run(tmp_path, signal=False)
    assert set(cls) == {"C", "S", "F", "Z"}
    assert set(folds) == set(range(rst.N_FOLDS))


def test_planted_directions_give_a_diagonal_matrix(tmp_path):
    _, _, out = _run(tmp_path, signal=True)
    m = {p: v[0] for p, v in out.items()}
    assert m["structural"]["S_vs_C"]["auroc"] > 0.85
    assert m["fab_vs_zero"]["F_vs_Z"]["auroc"] > 0.85
    assert abs(m["structural"]["F_vs_Z"]["auroc"] - 0.5) < 0.2       # blind to the other type
    assert abs(m["fab_vs_zero"]["S_vs_C"]["auroc"] - 0.5) < 0.2
    assert all(v[1][k]["layer"] == 1 for v in out.values() for k in v[1])   # found the layer


def test_no_signal_stays_near_chance(tmp_path):
    _, _, out = _run(tmp_path, signal=False)
    assert abs(out["structural"][0]["S_vs_C"]["auroc"] - 0.5) < 0.2


@pytest.mark.parametrize("fid", ["absent_000001", "look_000123"])
def test_fold_is_stable(fid):
    assert rst.fold_of(fid) == rst.fold_of(fid) and 0 <= rst.fold_of(fid) < rst.N_FOLDS
