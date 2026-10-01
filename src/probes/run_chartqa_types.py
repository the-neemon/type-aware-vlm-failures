"""Type-aware probes on ChartQA with the labelled errors, and the cross-type matrix.

The ChartQA counterpart of src/probes/run_synth_types.py, using the five-label
annotations (results/labels/<model>_test.claude.jsonl) instead of synthetic
outcomes. Classes:

    C  correct answer (relaxed accuracy)
    S  structural: the answer is explained by misreading the figure
    K  computation: inputs read correctly, arithmetic or logic wrong
    F  fabrication: content the figure does not support
    N  not_an_error: scored wrong, but the answer is actually right

`ambiguous` items are left out. Probes, each a linear probe on query_last:

    structural     S vs C
    computation    K vs C
    struct_vs_comp S vs K   (the hypothesis on the error types ChartQA produces)

A probe is trained only when its positive class has at least MIN_POSITIVES
items (so there is no fabrication probe: 17 for LLaVA, 1 for Qwen). Every probe
scores every item out of fold, and each is reported on every contrast in
TARGETS, including F vs C (do the fabrications look like misreads?) and N vs C
(answers that were right after all should look like correct ones).

Method as in run_synth_types: 5 folds by figure; within each outer fold, layer
and L2 chosen by inner cross-validation on the other folds; query_last fixed in
advance (best for every ChartQA probe so far). Baseline: the same probes on the
surface features src/probes/run_sweep.py uses (question source, question length,
number of image tokens, numeric gold), with no activations.

Two options added on 1 Oct 2026 (todos.md):

  --question-type arithmetic|retrieval
        keep only items whose question is of that type (src/probes/question_type.py),
        correct and wrong alike. Computation vs correct within arithmetic
        questions cannot be won by detecting that the question asks for arithmetic.
  --residualise
        also run every probe on activations with the four surface features
        regressed out (E2, Sahoo et al.). The projection is fitted per outer fold
        on that fold's training rows only, as src/common/linear.residualise asks.

    python -m src.probes.run_chartqa_types --activations <test.npz> \\
        --predictions <model>_test.jsonl --labels results/labels/<model>_test.claude.jsonl \\
        --out results/probes/chartqa_types_<model>
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

from src.common.linear import auroc, residualise
from src.probes.dataset import cached_layers, load_predictions, resolve_labels
from src.probes.question_type import question_type
from src.probes.run_synth_types import (DEFAULT_L2, N_FOLDS, activations, cross_validate,
                                        fold_of, matrix, select_and_assemble)

PROBES = {"structural": ("S", "C"), "computation": ("K", "C"), "struct_vs_comp": ("S", "K")}
TARGETS = {"S_vs_C": ("S", "C"), "K_vs_C": ("K", "C"), "S_vs_K": ("S", "K"),
           "F_vs_C": ("F", "C"), "N_vs_C": ("N", "C")}
LABEL_CLASS = {"structural": "S", "computation": "K", "fabrication": "F", "not_an_error": "N"}
MIN_POSITIVES = 20      # to train a probe
MIN_TARGET = 5          # to report a contrast


def load_items(npz, predictions, labels_path, qtype=None):
    """Every scored item with a class; `ambiguous` and unlabelled errors left out.

    With `qtype`, only items whose question_type() is `qtype`.
    """
    preds = load_predictions(predictions)
    labels, _ = resolve_labels(labels_path)
    items, unlabelled = [], 0
    for r in preds.values():
        if r["correct"]:
            cls = "C"
        elif r["item_id"] in labels:
            cls = LABEL_CLASS.get(labels[r["item_id"]])
        else:
            unlabelled += 1
            cls = None
        if cls is None or (qtype and question_type(r["question"]) != qtype):
            continue
        items.append({"item_id": r["item_id"], "figure_id": r["figure_id"], "npz": str(npz),
                      "cls": cls, "fold": fold_of(r["figure_id"]),
                      "surface": [r["source"] == "human", len(r["question"]),
                                  r.get("n_vision_tokens", 0),
                                  r["gold"].replace(".", "", 1).replace("-", "", 1).isdigit()]})
    return items, unlabelled


def surface(items):
    return np.array([it["surface"] for it in items], dtype=np.float32)


def _cell(args):
    """One probe at one layer. `layer` is an int, "surface", or ("resid", int)."""
    probe, layer, items, l2_grid = args
    cls = np.array([it["cls"] for it in items])
    folds = np.array([it["fold"] for it in items])
    if layer == "surface":
        X = surface(items)
    elif isinstance(layer, tuple):
        return probe, layer, cross_validate_residualised(
            activations(items, layer[1]), surface(items), cls, folds, probe, l2_grid)
    else:
        X = activations(items, layer)
    return probe, layer, cross_validate(X, cls, folds, probe, l2_grid, probes=PROBES)


def cross_validate_residualised(A, S, cls, folds, probe, l2_grid):
    """cross_validate on A with S regressed out, the projection fitted per outer
    fold on the rows that fold trains on. Uses no labels."""
    out = {}
    for k in range(N_FOLDS):
        Xk = residualise(A, S, fit_on=folds != k).astype(np.float32)
        out.update(cross_validate(Xk, cls, folds, probe, l2_grid, probes=PROBES, outer=[k]))
    return out


def trainable(cls):
    return {p: pn for p, pn in PROBES.items()
            if min((cls == pn[0]).sum(), (cls == pn[1]).sum()) >= MIN_POSITIVES}


def reportable(cls):
    return {t: pn for t, pn in TARGETS.items()
            if min((cls == pn[0]).sum(), (cls == pn[1]).sum()) >= MIN_TARGET}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--activations", required=True, type=pathlib.Path)
    ap.add_argument("--predictions", required=True, type=pathlib.Path)
    ap.add_argument("--labels", required=True, type=pathlib.Path)
    ap.add_argument("--layers", nargs="*", type=int, default=None,
                    help="default: every layer in the cache")
    ap.add_argument("--l2", nargs="*", type=float, default=list(DEFAULT_L2))
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    ap.add_argument("--question-type", choices=("arithmetic", "retrieval"), default=None)
    ap.add_argument("--residualise", action="store_true",
                    help="also run every probe with the surface features regressed out")
    ap.add_argument("--out", required=True, type=pathlib.Path)
    args = ap.parse_args()
    layers = args.layers if args.layers is not None else cached_layers(args.activations)

    items, unlabelled = load_items(args.activations, args.predictions, args.labels,
                                   args.question_type)
    cls = np.array([it["cls"] for it in items])
    folds = np.array([it["fold"] for it in items])
    figs = np.array([it["figure_id"] for it in items])
    probes, targets = trainable(cls), reportable(cls)
    counts = {c: int((cls == c).sum()) for c in "CSKFN"}
    print(f"items: {counts}; errors without a label (left out): {unlabelled}")
    print(f"probes: {list(probes)}; contrasts: {list(targets)}", flush=True)

    variants = [*layers, "surface"] + ([("resid", L) for L in layers] if args.residualise else [])
    cells = [(p, L, items, tuple(args.l2)) for p in probes for L in variants]
    results = {p: {} for p in probes}
    resid = {p: {} for p in probes}
    base = {}
    t0 = time.time()
    with Pool(args.jobs) as pool:
        for n, (p, L, res) in enumerate(pool.imap_unordered(_cell, cells), 1):
            if L == "surface":
                base[p] = res
            elif isinstance(L, tuple):
                resid[p][L[1]] = res
            else:
                results[p][L] = res
            if n % max(1, len(cells) // 10) == 0 or n == len(cells):
                print(f"  {n}/{len(cells)} cells [{time.time() - t0:.0f}s]", flush=True)

    report = {"position": "query_last", "n_folds": N_FOLDS, "classes": counts,
              "question_type": args.question_type, "residualised": args.residualise,
              "unlabelled_errors": unlabelled, "probes": {}}
    for p, (pos, neg) in probes.items():
        scores, chosen = select_and_assemble(results[p], len(items), folds)
        b_scores, b_chosen = select_and_assemble({"surface": base[p]}, len(items), folds)
        own = np.isin(cls, (pos, neg))
        by_layer = {}
        for L in layers:
            s_L, _ = select_and_assemble({L: results[p][L]}, len(items), folds)
            by_layer[L] = auroc((cls[own] == pos).astype(float), s_L[own])
        report["probes"][p] = {
            "trained_on": f"{pos} vs {neg}", "chosen_per_fold": chosen,
            "matrix": matrix(scores, cls, figs, targets),
            "surface_baseline": {"chosen_per_fold": b_chosen,
                                 "matrix": matrix(b_scores, cls, figs, targets)},
            "own_task_auroc_by_layer": by_layer}
        if args.residualise:
            r_scores, r_chosen = select_and_assemble(resid[p], len(items), folds)
            report["probes"][p]["residualised"] = {
                "chosen_per_fold": r_chosen, "matrix": matrix(r_scores, cls, figs, targets)}

    print(f"\n{'probe':15s}" + "".join(f"{t:>22s}" for t in targets))
    for p, e in report["probes"].items():
        rows = [("acts", e["matrix"]), ("surface", e["surface_baseline"]["matrix"])]
        if "residualised" in e:
            rows.append(("resid", e["residualised"]["matrix"]))
        for tag, m in rows:
            print(f"{p if tag == 'acts' else '':15s}" + "".join(
                f"{m[t]['auroc']:>8.3f} [{m[t]['ci95'][0]:.2f},{m[t]['ci95'][1]:.2f}]"
                for t in targets) + f"  {tag}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.with_suffix(".json").write_text(json.dumps(report, indent=2, default=float) + "\n")
    print(f"wrote {args.out.with_suffix('.json')}")


if __name__ == "__main__":
    main()
