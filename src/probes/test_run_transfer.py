"""run_transfer on planted signals.

Synthetic misreads sit along dimension 0 at layer 1 (test_run_synth_types._pilot).
Fake ChartQA items put their structural errors on dimension 0 too (shared) or on
dimension 5 (not shared), and computation errors on dimension 3.
"""

import argparse
import json
import pathlib

import numpy as np

from src.probes import run_transfer as rt
from src.probes.test_run_synth_types import HIDDEN, LAYERS, _pilot


def _chartqa(tmp: pathlib.Path, shared: bool, rng, shuffle_labels=False):
    rows, labels, ids, figs = [], [], [], []
    for f in range(300):
        fid = f"cq_{f:04d}.png"
        for q in range(2):
            iid = f"{fid}-{q}"
            kind = ("S" if f % 5 == 0 else "P" if f % 5 == 1 else "C") if q == 0 else "C"
            # question length drawn independently of the class, so the surface
            # baseline has nothing real to find
            rows.append({"item_id": iid, "figure_id": fid,
                         "question": "w " * int(rng.integers(3, 15)),
                         "gold": "40", "prediction": "40" if kind == "C" else "41",
                         "correct": kind == "C", "source": "human", "n_vision_tokens": 580})
            if kind != "C":
                labels.append({"item_id": iid,
                               "label": "structural" if kind == "S" else "computation"})
            ids.append(iid)
            figs.append(fid)
    if shuffle_labels:
        perm = rng.permutation(len(labels))
        labels = [{**labels[i], "item_id": labels[j]["item_id"]} for i, j in enumerate(perm)]
    kind_of = {l["item_id"]: l["label"] for l in labels}
    acts = {}
    for L in LAYERS:
        A = rng.normal(size=(len(rows), HIDDEN))
        if L == 1:
            for i, r in enumerate(rows):
                k = kind_of.get(r["item_id"])
                A[i, 0 if shared else 5] += 3.0 * (k == "structural")
                A[i, 3] += 3.0 * (k == "computation")
        acts[f"L{L}_query_last"] = A.astype(np.float16)
    np.savez_compressed(tmp / "cq.npz", item_ids=np.array(ids),
                        figure_ids=np.array(figs, dtype=object), **acts)
    (tmp / "cq.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    (tmp / "labels.jsonl").write_text("".join(json.dumps(l) + "\n" for l in labels))
    return [str(tmp / "cq.npz"), str(tmp / "cq.jsonl"), str(tmp / "labels.jsonl")]


def _args(tmp, chartqa, pilots=None):
    return argparse.Namespace(chartqa=chartqa, pilot=pilots, layers=list(LAYERS),
                              l2=[1.0, 100.0], jobs=1, out=tmp / "out")


def _pilots(tmp, rng):
    return [list(_pilot(tmp, "pa", 120, True, rng)), list(_pilot(tmp, "pb", 120, True, rng))]


def _report(tmp):
    return json.loads((tmp / "out.json").read_text())


def test_chartqa_classes(tmp_path):
    items = rt.load_chartqa_items(*_chartqa(tmp_path, True, np.random.default_rng(0)))
    counts = {c: sum(it["cls"] == c for it in items) for c in "CSP"}
    assert counts == {"C": 480, "S": 60, "P": 60}


def test_shared_direction_transfers(tmp_path):
    rng = np.random.default_rng(0)
    rt.run_transfer(_args(tmp_path, _chartqa(tmp_path, True, rng), _pilots(tmp_path, rng)))
    p = _report(tmp_path)["probes"]["structural"]
    assert p["selected_on_synthetic"]["layer"] == 1
    assert p["chartqa"]["S_vs_C"]["auroc"] > 0.85
    assert p["chartqa"]["P_vs_C"]["auroc"] < 0.65      # does not fire on computation


def test_unshared_direction_transfers_much_worse(tmp_path):
    # Relative, not absolute: a probe fit on synthetic data carries small random
    # weights on every dimension, so a large unshared shift can still leak a
    # little. What must hold is that sharing the direction is what transfers.
    auc = {}
    for shared in (True, False):
        d = tmp_path / str(shared)
        d.mkdir()
        rng = np.random.default_rng(0)
        rt.run_transfer(_args(d, _chartqa(d, shared, rng), _pilots(d, rng)))
        auc[shared] = _report(d)["probes"]["structural"]["chartqa"]["S_vs_C"]["auroc"]
    assert auc[True] > auc[False] + 0.2


def test_selection_never_sees_chartqa_labels(tmp_path):
    chosen = []
    for shuffle in (False, True):
        d = tmp_path / str(shuffle)
        d.mkdir()
        rng = np.random.default_rng(0)
        pilots = _pilots(d, rng)
        rt.run_transfer(_args(d, _chartqa(d, True, np.random.default_rng(1),
                                          shuffle_labels=shuffle), pilots))
        chosen.append({k: v["selected_on_synthetic"]
                       for k, v in _report(d)["probes"].items()})
    assert chosen[0] == chosen[1]


def test_computation_matrix_is_diagonal(tmp_path):
    rt.run_computation(_args(tmp_path, _chartqa(tmp_path, True, np.random.default_rng(0))))
    probes = _report(tmp_path)["probes"]
    assert probes["structural"]["matrix"]["S_vs_C"]["auroc"] > 0.85
    assert probes["computation"]["matrix"]["P_vs_C"]["auroc"] > 0.85
    assert probes["comp_vs_struct"]["matrix"]["P_vs_S"]["auroc"] > 0.85
    assert probes["structural"]["matrix"]["P_vs_C"]["auroc"] < 0.65
    assert probes["computation"]["matrix"]["S_vs_C"]["auroc"] < 0.65
    # the surface features carry no class information here
    assert probes["computation"]["surface_baseline"]["matrix"]["P_vs_C"]["auroc"] < 0.65
