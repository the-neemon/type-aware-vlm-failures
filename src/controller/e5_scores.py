"""E5 step 1: out-of-fold probe scores for every synthetic value question.

The controller (src/controller/e5.py) needs, for each value question of pilots 1
and 2, three probe scores computed without that question's chart in training:

    binary   wrong (S, F, Z) vs right (C, R): will the answer be wrong?
    missing  missing bar (F, Z, R) vs shown bar (C, S): is the asked bar absent?
    misread  S vs C: will a shown bar be misread? (scored on every item)

Classes as in src/probes/run_synth_types.py, plus R, a correct refusal of a
missing bar ("None"; 15 Qwen items, 0 LLaVA), which E3 leaves out but the
controller must still handle. Method as in run_synth_types: query_last, 5 folds
by figure (the same `fold_of`), layer and L2 chosen per outer fold by inner
cross-validation, so every score is out of fold.

Output: one JSON line per item (item_id, figure_id, pilot, split, fold, cls, and
the three scores) plus a report of the layers chosen. CPU only; run on Ada,
where the caches are:

    python -m src.controller.e5_scores \\
        --pilot pilot1 <synth_pilot.npz> <..._synth_pilot.jsonl> <absent_pilot_v1/manifest.jsonl> \\
        --pilot pilot2 <synth_pilot2.npz> <..._synth_pilot2.jsonl> <absent_pilot_v2/manifest.jsonl> \\
        --out results/e5/<model>_scores
"""

import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import json
import pathlib
import time
from multiprocessing import Pool

import numpy as np

from src.common.linear import auroc
from src.probes.dataset import load_predictions
from src.probes.run_synth import rescore
from src.probes.run_synth_types import (
    DEFAULT_L2, N_FOLDS, VALUE_TEMPLATES, activations, cross_validate, fold_of, pilot_layers,
    select_and_assemble,
)

PROBES = {"binary": ("SFZ", "CR"), "missing": ("FZR", "CS"), "misread": ("S", "C")}


def item_class(r) -> str | None:
    """C, S, F, Z as in run_synth_types, plus R for a correct refusal of a missing bar."""
    if r["template"] not in VALUE_TEMPLATES:
        return None
    if not r["absent"]:
        return "C" if r["correct"] else "S"
    return {"fabricated": "F", "zero": "Z", "rejected": "R"}.get(r["outcome"])


def load_items(pilots) -> list[dict]:
    items = []
    for name, npz, preds_path, manifest in pilots:
        preds = rescore(load_predictions(preds_path), pathlib.Path(manifest))
        for r in preds.values():
            cls = item_class(r)
            if cls is None:
                continue
            items.append({"item_id": r["item_id"], "figure_id": r["figure_id"], "pilot": name,
                          "split": r.get("split"), "npz": str(npz), "cls": cls,
                          "fold": fold_of(r["figure_id"])})
    return items


def probe_labels(cls: np.ndarray, probe: str) -> np.ndarray:
    """"P" / "N" / "-" per item, for cross_validate's single-class contrast."""
    pos, neg = PROBES[probe]
    return np.where(np.isin(cls, list(pos)), "P", np.where(np.isin(cls, list(neg)), "N", "-"))


def _cell(args):
    probe, layer, items, l2_grid = args
    cls = probe_labels(np.array([it["cls"] for it in items]), probe)
    folds = np.array([it["fold"] for it in items])
    X = activations(items, layer)
    return probe, layer, cross_validate(X, cls, folds, "p", l2_grid, probes={"p": ("P", "N")})


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--pilot", nargs=4, action="append", required=True,
                    metavar=("NAME", "NPZ", "PREDICTIONS", "MANIFEST"))
    ap.add_argument("--layers", nargs="*", type=int, default=None)
    ap.add_argument("--l2", nargs="*", type=float, default=list(DEFAULT_L2))
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    ap.add_argument("--out", required=True, type=pathlib.Path)
    args = ap.parse_args()
    layers = args.layers if args.layers is not None else pilot_layers(args.pilot)

    items = load_items(args.pilot)
    cls = np.array([it["cls"] for it in items])
    folds = np.array([it["fold"] for it in items])
    print("items:", {c: int((cls == c).sum()) for c in "CSFZR"}, flush=True)

    cells = [(p, L, items, tuple(args.l2)) for p in PROBES for L in layers]
    results = {p: {} for p in PROBES}
    t0 = time.time()
    with Pool(args.jobs) as pool:
        for n, (p, L, res) in enumerate(pool.imap_unordered(_cell, cells), 1):
            results[p][L] = res
            if n % max(1, len(cells) // 10) == 0 or n == len(cells):
                print(f"  {n}/{len(cells)} cells [{time.time() - t0:.0f}s]", flush=True)

    scores, report = {}, {"position": "query_last", "n_folds": N_FOLDS,
                          "pilots": [p[0] for p in args.pilot],
                          "classes": {c: int((cls == c).sum()) for c in "CSFZR"}, "probes": {}}
    for p in PROBES:
        s, chosen = select_and_assemble(results[p], len(items), folds)
        lab = probe_labels(cls, p)
        own = lab != "-"
        scores[p] = s
        report["probes"][p] = {"positive": PROBES[p][0], "negative": PROBES[p][1],
                               "auroc": auroc(lab[own] == "P", s[own]),
                               "chosen_per_fold": chosen}
        print(f"{p:8s} AUROC {report['probes'][p]['auroc']:.3f}", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out.with_suffix(".jsonl"), "w", encoding="utf-8") as f:
        for i, it in enumerate(items):
            row = {k: it[k] for k in ("item_id", "figure_id", "pilot", "split", "fold", "cls")}
            row.update({p: float(scores[p][i]) for p in PROBES})
            f.write(json.dumps(row) + "\n")
    args.out.with_suffix(".json").write_text(json.dumps(report, indent=2, default=float) + "\n")
    print(f"wrote {args.out.with_suffix('.jsonl')} and .json")


if __name__ == "__main__":
    main()
