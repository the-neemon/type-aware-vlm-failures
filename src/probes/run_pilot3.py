"""Pilot 3: do the missing-bar probes of pilots 1 and 2 fire on an out-of-range year?

Pilot 3 (src/synth/out_of_range.py, AGENTS.md decision 4) asks each chart of
consecutive years about a shown year and, in identical words, about a year one
to three years outside the shown range. Pilots 1 and 2 asked about a category
that is not a bar, so there "fabrication" and "the category is missing" were the
same thing. This script asks whether the internal signal found there is about
missing categories specifically, or about questions the chart cannot answer.

Two probes, both linear on query_last, both trained on the value questions of
pilots 1 and 2 only and applied to pilot 3 unchanged:

    missing_vs_shown   absent name vs shown name, whatever the model answered
                       (mid report Table 4, there per family on pilot 2 alone)
    fabrication        made-up number for a missing name (F) vs correct read of a
                       shown one (C): the E3 "fabrication" probe

Layer and L2 are chosen by out-of-fold AUROC on pilots 1 and 2 (5 folds by
chart), so no pilot-3 answer touches either probe; per-layer pilot-3 AUROCs are
reported as description only. Every pilot-3 contrast gets a 95% bootstrap
interval over charts.

Pilot-3 classes, from the model's own answers (src/synth/items.py):

    C  shown year, read correctly        S  shown year, misread
    R  out-of-range year, rejected       Z  out-of-range year, "0"
    F  out-of-range year, fabricated     U  out-of-range year, unclear

The outcome table also checks what the fabricated numbers are: a copy of a shown
bar's value, or close to the least-squares extrapolation of the series
(`extrapolated` in the manifest), both within the 5% relative tolerance.

    python -m src.probes.run_pilot3 \\
        --pilot pilot1 $ACT/<model>/synth_pilot.npz results/predictions/<model>_synth_pilot.jsonl $SYNTH/absent_pilot_v1/manifest.jsonl \\
        --pilot pilot2 $ACT/<model>/synth_pilot2.npz results/predictions/<model>_synth_pilot2.jsonl $SYNTH/absent_pilot_v2/manifest.jsonl \\
        --target $ACT/<model>/synth_pilot3.npz results/predictions/<model>_synth_pilot3.jsonl $SYNTH/absent_pilot_v3/manifest.jsonl \\
        --out results/probes/pilot3_<model>.json
"""

import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import json
import pathlib
import sys
import time
from collections import Counter
from multiprocessing import Pool

import numpy as np

from src.common.linear import auroc, cluster_bootstrap_ci
from src.eval.relaxed_accuracy import is_correct
from src.probes.dataset import cached_layers, clear_activation_cache, load_predictions
from src.probes.run_synth import rescore
from src.probes.run_synth_types import DEFAULT_L2, VALUE_TEMPLATES, activations, fold_of
from src.probes.run_transfer import contrast, fit, oof_auroc, score

PROBES = {"missing_vs_shown": ("A", "P"), "fabrication": ("F", "C")}
TARGET_CLASSES = "CSRZFU"
OUTCOME_CLASS = {"rejected": "R", "zero": "Z", "fabricated": "F", "unclear": "U"}
# what each probe is scored on in pilot 3; "A" = any out-of-range year, "P" = any shown year
TARGETS = {"absent_vs_shown": (("R", "Z", "F", "U"), ("C", "S")),
           "R_vs_C": (("R",), ("C",)), "Z_vs_C": (("Z",), ("C",)),
           "F_vs_C": (("F",), ("C",)), "F_vs_Z": (("F",), ("Z",)),
           "F_vs_R": (("F",), ("R",)), "S_vs_C": (("S",), ("C",))}
MIN_TARGET = 5


def target_class(r) -> str:
    if not r["absent"]:
        return "C" if r["correct"] else "S"
    return OUTCOME_CLASS[r["outcome"]]


def read_manifest(path):
    return {f["figure_id"]: f for f in map(json.loads, pathlib.Path(path)
                                           .read_text(encoding="utf-8").splitlines())}


def load_value_items(name, npz, preds_path, manifest):
    """Value-question items of one pilot, rescored with the current scorer."""
    preds = rescore(load_predictions(preds_path), pathlib.Path(manifest))
    items = []
    for r in preds.values():
        if r["template"] not in VALUE_TEMPLATES:
            continue
        items.append({"item_id": r["item_id"], "figure_id": f"{name}/{r['figure_id']}",
                      "npz": str(npz), "fold": fold_of(r["figure_id"]),
                      "absent": bool(r["absent"]), "correct": bool(r["correct"]),
                      "outcome": r.get("outcome"), "prediction": r["prediction"],
                      "raw_figure_id": r["figure_id"], "asks_about": r["asks_about"]})
    return items


def training_classes(items):
    """Per probe, each training item's class: A/P for missing_vs_shown, F/C/other."""
    ms = np.array(["A" if it["absent"] else "P" for it in items])
    fab = np.array(["F" if it["absent"] and it["outcome"] == "fabricated" else
                    "C" if not it["absent"] and it["correct"] else "-" for it in items])
    return {"missing_vs_shown": ms, "fabrication": fab}


def _cell(args):
    """One layer: per probe and L2, out-of-fold AUROC on pilots 1-2 and pilot-3 scores."""
    layer, train, target, l2_grid = args
    Xs, Xt = activations(train, layer), activations(target, layer)
    clear_activation_cache()
    folds = np.array([it["fold"] for it in train])
    out = {}
    for name, cls in training_classes(train).items():
        pos, neg = PROBES[name]
        by_l2 = {l2: oof_auroc(Xs, cls, folds, pos, neg, l2) for l2 in l2_grid}
        l2 = max(by_l2, key=lambda v: np.nan_to_num(by_l2[v], nan=-1.0))
        member, y = contrast(cls, pos, neg)
        out[name] = {"train_oof_auroc": by_l2[l2], "l2": l2,
                     "scores": score(Xt, fit(Xs[member], y[member], l2))}
    return layer, out


def evaluate(scores, cls, figs, targets):
    out = {}
    for t, (pos, neg) in targets.items():
        m = np.isin(cls, pos + neg)
        y, s, g = np.isin(cls[m], pos).astype(float), scores[m], figs[m]
        out[t] = {"auroc": auroc(y, s),
                  "ci95": list(cluster_bootstrap_ci(lambda i: auroc(y[i], s[i]), g)),
                  "n_pos": int(y.sum()), "n_neg": int((1 - y).sum())}
    return out


def _number(text):
    try:
        return float(str(text).strip().rstrip(".").replace(",", "").rstrip("%").lstrip("$"))
    except ValueError:
        return None


def outcome_table(target, manifest):
    """Counts per class, and what the fabricated numbers are."""
    figures = read_manifest(manifest)
    counts = Counter(it["cls"] for it in target)
    fab = [it for it in target if it["cls"] == "F"]
    copy_bar = near_extrap = 0
    for it in fab:
        f = figures[it["raw_figure_id"]]
        absent = next(q for q in f["questions"] if q["absent"])
        v = _number(it["prediction"])
        if v is None:
            continue
        copy_bar += any(is_correct(str(b), str(v)) for b in f["values"])
        near_extrap += is_correct(str(round(absent["extrapolated"], 4)), str(v))
    def direction(fid):
        return next(q for q in figures[fid]["questions"] if q["absent"])["direction"]
    by_direction = Counter((direction(it["raw_figure_id"]), it["cls"])
                           for it in target if it["absent"])
    return {"counts": {c: counts.get(c, 0) for c in TARGET_CLASSES},
            "fabricated_copies_a_shown_bar": copy_bar,
            "fabricated_near_extrapolation": near_extrap,
            "absent_outcome_by_direction": {f"{d}:{c}": n for (d, c), n in
                                            sorted(by_direction.items())}}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--pilot", nargs=4, action="append", required=True,
                    metavar=("NAME", "NPZ", "PREDICTIONS", "MANIFEST"),
                    help="a training pilot (pilots 1 and 2)")
    ap.add_argument("--target", nargs=3, required=True,
                    metavar=("NPZ", "PREDICTIONS", "MANIFEST"), help="pilot 3")
    ap.add_argument("--layers", nargs="*", type=int, default=None)
    ap.add_argument("--l2", nargs="*", type=float, default=list(DEFAULT_L2))
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    ap.add_argument("--command", default=None,
                    help="the command line to record in the report (paths as variables)")
    ap.add_argument("--out", required=True, type=pathlib.Path)
    args = ap.parse_args()

    train = [it for p in args.pilot for it in load_value_items(*p)]
    target = load_value_items("pilot3", *args.target)
    for it in target:
        it["cls"] = target_class(it)
    layers = args.layers or cached_layers(args.target[0])
    if set(layers) - set(cached_layers(args.pilot[0][1])):
        raise SystemExit("training and pilot-3 caches hold different layers")
    tcls = np.array([it["cls"] for it in target])
    tfigs = np.array([it["figure_id"] for it in target])
    targets = {t: pn for t, pn in TARGETS.items()
               if min(np.isin(tcls, pn[0]).sum(), np.isin(tcls, pn[1]).sum()) >= MIN_TARGET}
    tc = training_classes(train)
    print("train:", {p: dict(Counter(c)) for p, c in tc.items()})
    print("pilot 3:", dict(Counter(tcls)), flush=True)

    per_layer = {}
    t0 = time.time()
    with Pool(args.jobs) as pool:
        for n, (L, out) in enumerate(pool.imap_unordered(
                _cell, [(L, train, target, tuple(args.l2)) for L in layers]), 1):
            per_layer[L] = out
            print(f"  layer {L} done ({n}/{len(layers)}) [{time.time() - t0:.0f}s]", flush=True)

    report = {"command": args.command or " ".join(sys.argv), "position": "query_last",
              "training_pilots": [p[0] for p in args.pilot],
              "training_classes": {p: dict(Counter(c)) for p, c in tc.items()},
              "outcomes": outcome_table(target, args.target[2]), "probes": {}}
    for name in PROBES:
        best = max(layers, key=lambda L: np.nan_to_num(per_layer[L][name]["train_oof_auroc"],
                                                       nan=-1.0))
        sel = per_layer[best][name]
        report["probes"][name] = {
            "trained_on": "{} vs {}".format(*PROBES[name]),
            "selected_on_pilots_1_2": {"layer": best, "l2": sel["l2"],
                                       "train_oof_auroc": sel["train_oof_auroc"]},
            "pilot3": evaluate(sel["scores"], tcls, tfigs, targets),
            "pilot3_absent_vs_shown_by_layer_descriptive_only": {
                L: evaluate(per_layer[L][name]["scores"], tcls, tfigs,
                            {"a": TARGETS["absent_vs_shown"]})["a"]["auroc"] for L in layers},
            "train_oof_auroc_by_layer": {L: per_layer[L][name]["train_oof_auroc"]
                                         for L in layers}}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, default=float) + "\n")
    print(f"wrote {args.out}")
    print("outcomes:", json.dumps(report["outcomes"]))
    for name, e in report["probes"].items():
        sel = e["selected_on_pilots_1_2"]
        print(f"\n{name}: layer {sel['layer']}, L2 {sel['l2']:g}, "
              f"pilots 1-2 OOF AUROC {sel['train_oof_auroc']:.3f}")
        for t, m in e["pilot3"].items():
            print(f"  pilot 3 {t:16s} {m['auroc']:.3f} [{m['ci95'][0]:.2f}, {m['ci95'][1]:.2f}]"
                  f"  (n={m['n_pos']} vs {m['n_neg']})")


if __name__ == "__main__":
    main()
