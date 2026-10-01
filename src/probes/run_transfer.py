"""Two cheap CPU experiments on Qwen's labelled ChartQA errors (todos.md).

Both reuse run_synth_types's folds, activations and fitting, so their numbers are
comparable with results/synth_type_probes.md.

transfer
    Does the synthetic structural probe flag real misreads? Train the structural
    (S vs C) and missing-bar (F vs C) probes on every synthetic value question
    (pilots 1 and 2), then score ChartQA test items with them, unchanged. Layer and
    L2 are chosen on the synthetic data alone, by out-of-fold AUROC over the
    synthetic folds, so no ChartQA label touches the probe. Per-layer ChartQA
    AUROCs are also reported, as description only, never for selection.

computation
    Computation errors are Qwen's largest class of real errors (93 of 317). On
    ChartQA alone, three probes with the E3 cross matrix:
        structural     S vs C
        computation    P vs C
        comp_vs_struct P vs S
    5 figure-grouped folds; layer and L2 chosen per outer fold by inner CV on the
    other folds. Baseline: the same probes on the four surface features of the E1
    baseline (source, question length, vision tokens, numeric gold).

Classes on ChartQA: C = correct, S = structural, P = computation, F =
fabrication, from results/labels/<model>_test.claude.jsonl. not_an_error and
ambiguous are left out. Transfer also reports F vs C when there are at least
MIN_TARGET fabrications (LLaVA has 17, Qwen 1): does the missing-bar probe flag
real fabrications? Any model works; pass that model's caches and labels.

    python -m src.probes.run_transfer transfer \\
        --chartqa <test.npz> <qwen2_5_vl_7b_test.jsonl> <labels.jsonl> \\
        --pilot pilot1 <synth_pilot.npz> <preds.jsonl> <manifest.jsonl> \\
        --pilot pilot2 <synth_pilot2.npz> <preds.jsonl> <manifest.jsonl> \\
        --out results/probes/transfer_chartqa_qwen2_5_vl_7b
    python -m src.probes.run_transfer computation \\
        --chartqa <test.npz> <qwen2_5_vl_7b_test.jsonl> <labels.jsonl> \\
        --out results/probes/computation_chartqa_qwen2_5_vl_7b
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

from src.common.linear import auroc, cluster_bootstrap_ci, fit_logistic, predict_scores
from src.probes.dataset import (
    cached_layers, clear_activation_cache, load_predictions, resolve_labels,
)
from src.probes.run_synth_types import (
    DEFAULT_L2, N_FOLDS, activations, fold_of, load_items as load_synth_items,
)

LABEL_CLASS = {"structural": "S", "computation": "P", "fabrication": "F"}
MIN_TARGET = 5
TRANSFER_PROBES = {"structural": ("S", "C"), "missing_bar": ("F", "C")}
CHARTQA_PROBES = {"structural": ("S", "C"), "computation": ("P", "C"),
                  "comp_vs_struct": ("P", "S")}
CHARTQA_TARGETS = {"S_vs_C": ("S", "C"), "P_vs_C": ("P", "C"), "P_vs_S": ("P", "S")}
TRANSFER_TARGETS = {**CHARTQA_TARGETS, "F_vs_C": ("F", "C")}


# ---------------------------------------------------------------------------
# Items
# ---------------------------------------------------------------------------

def load_chartqa_items(npz, preds_path, labels_path):
    """ChartQA test items with class C, S or P, joined on item_id."""
    labels, _ = resolve_labels(labels_path)
    items = []
    for r in load_predictions(preds_path).values():
        cls = "C" if r["correct"] else LABEL_CLASS.get(labels.get(r["item_id"]))
        if cls is None:
            continue
        items.append({"item_id": r["item_id"], "figure_id": r["figure_id"], "npz": str(npz),
                      "cls": cls, "fold": fold_of(r["figure_id"]),
                      "surface": [r.get("source") == "human", len(r["question"]),
                                  r.get("n_vision_tokens", 0), _numeric(r["gold"])]})
    return items


def _numeric(s) -> bool:
    try:
        float(str(s).replace(",", "").rstrip("%"))
        return True
    except ValueError:
        return False


def surface(items):
    return np.array([it["surface"] for it in items], dtype=np.float32)


# ---------------------------------------------------------------------------
# Fitting
# ---------------------------------------------------------------------------

def fit(X, y, l2):
    """Standardise on the fitting rows, then L2 logistic. Returns scoring params."""
    mu, sd = X.mean(0), X.std(0)
    sd = np.where(sd < 1e-6, 1.0, sd)
    return mu, sd, fit_logistic((X - mu) / sd, y, l2)


def score(X, params):
    mu, sd, w = params
    return predict_scores((X - mu) / sd, w)


def contrast(cls, pos, neg):
    member = np.isin(cls, (pos, neg))
    return member, (cls == pos).astype(float)


def oof_auroc(X, cls, folds, pos, neg, l2):
    """Pooled out-of-fold AUROC of one probe at one L2 over all folds."""
    member, y = contrast(cls, pos, neg)
    s = np.full(len(cls), np.nan)
    for k in range(N_FOLDS):
        train, test = member & (folds != k), member & (folds == k)
        if test.any() and len(set(y[train])) == 2:
            s[test] = score(X[test], fit(X[train], y[train], l2))
    ok = member & ~np.isnan(s)
    return auroc(y[ok], s[ok])


def evaluate(scores, cls, figs, targets):
    out = {}
    for t, (pos, neg) in targets.items():
        m = np.isin(cls, (pos, neg))
        y, s, g = (cls[m] == pos).astype(float), scores[m], figs[m]
        out[t] = {"auroc": auroc(y, s),
                  "ci95": list(cluster_bootstrap_ci(lambda i: auroc(y[i], s[i]), g)),
                  "n_pos": int(y.sum()), "n_neg": int((1 - y).sum())}
    return out


# ---------------------------------------------------------------------------
# transfer
# ---------------------------------------------------------------------------

def _transfer_cell(args):
    layer, synth, chartqa, l2_grid = args
    Xs, Xc = activations(synth, layer), activations(chartqa, layer)
    clear_activation_cache()
    cls = np.array([it["cls"] for it in synth])
    folds = np.array([it["fold"] for it in synth])
    out = {}
    for name, (pos, neg) in TRANSFER_PROBES.items():
        by_l2 = {l2: oof_auroc(Xs, cls, folds, pos, neg, l2) for l2 in l2_grid}
        l2 = max(by_l2, key=lambda v: np.nan_to_num(by_l2[v], nan=-1.0))
        member, y = contrast(cls, pos, neg)
        out[name] = {"synth_oof_auroc": by_l2[l2], "l2": l2,
                     "chartqa_scores": score(Xc, fit(Xs[member], y[member], l2))}
    return layer, out


def run_transfer(args):
    synth = load_synth_items(args.pilot)
    chartqa = load_chartqa_items(*args.chartqa)
    layers = args.layers or cached_layers(args.chartqa[0])
    ccls = np.array([it["cls"] for it in chartqa])
    cfigs = np.array([it["figure_id"] for it in chartqa])
    print("synthetic:", {c: sum(it["cls"] == c for it in synth) for c in "CSFZ"})
    print("chartqa:  ", {c: int((ccls == c).sum()) for c in "CSPF"}, flush=True)
    targets = {t: pn for t, pn in TRANSFER_TARGETS.items()
               if min((ccls == pn[0]).sum(), (ccls == pn[1]).sum()) >= MIN_TARGET}

    per_layer = {}
    t0 = time.time()
    with Pool(args.jobs) as pool:
        cells = [(L, synth, chartqa, tuple(args.l2)) for L in layers]
        for n, (L, out) in enumerate(pool.imap_unordered(_transfer_cell, cells), 1):
            per_layer[L] = out
            print(f"  layer {L} done ({n}/{len(layers)}) [{time.time() - t0:.0f}s]", flush=True)

    report = {"probes": {}, "chartqa_classes": {c: int((ccls == c).sum()) for c in "CSPF"}}
    for name in TRANSFER_PROBES:
        # selection on synthetic evidence only
        best = max(layers, key=lambda L: np.nan_to_num(per_layer[L][name]["synth_oof_auroc"],
                                                       nan=-1.0))
        chosen = per_layer[best][name]
        report["probes"][name] = {
            "selected_on_synthetic": {"layer": best, "l2": chosen["l2"],
                                      "synth_oof_auroc": chosen["synth_oof_auroc"]},
            "chartqa": evaluate(chosen["chartqa_scores"], ccls, cfigs, targets),
            "chartqa_S_vs_C_by_layer_descriptive_only": {
                L: auroc(*_pair(ccls, per_layer[L][name]["chartqa_scores"], "S", "C"))
                for L in layers},
        }
    _write(report, args.out)
    for name, e in report["probes"].items():
        sel = e["selected_on_synthetic"]
        print(f"\n{name}: layer {sel['layer']}, L2 {sel['l2']}, "
              f"synthetic OOF AUROC {sel['synth_oof_auroc']:.3f}")
        for t, m in e["chartqa"].items():
            print(f"  ChartQA {t}: {m['auroc']:.3f} [{m['ci95'][0]:.2f}, {m['ci95'][1]:.2f}]"
                  f"  (n={m['n_pos']} vs {m['n_neg']})")


def _pair(cls, s, pos, neg):
    m = np.isin(cls, (pos, neg))
    return (cls[m] == pos).astype(float), s[m]


# ---------------------------------------------------------------------------
# computation
# ---------------------------------------------------------------------------

def _cv_cell(args):
    """Per outer fold and L2: mean inner-CV AUROC and scores for that fold."""
    probe, layer, items, l2_grid = args
    cls = np.array([it["cls"] for it in items])
    folds = np.array([it["fold"] for it in items])
    X = surface(items) if layer == "surface" else activations(items, layer)
    clear_activation_cache()
    pos, neg = CHARTQA_PROBES[probe]
    member, y = contrast(cls, pos, neg)
    out = {}
    for k in range(N_FOLDS):
        outer = folds == k
        res = {}
        for l2 in l2_grid:
            inner = []
            for j in range(N_FOLDS):
                if j == k:
                    continue
                train, val = member & ~outer & (folds != j), member & (folds == j)
                if len(set(y[val])) == 2 and len(set(y[train])) == 2:
                    inner.append(auroc(y[val], score(X[val], fit(X[train], y[train], l2))))
            res[l2] = (float(np.mean(inner)) if inner else float("nan"),
                       score(X[outer], fit(X[member & ~outer], y[member & ~outer], l2)))
        out[k] = res
    return probe, layer, out


def _assemble(results, n, folds):
    scores, chosen = np.full(n, np.nan), {}
    for k in range(N_FOLDS):
        (layer, l2), (_, s) = max(
            (((L, l2), v) for L, by_fold in results.items() for l2, v in by_fold[k].items()),
            key=lambda kv: np.nan_to_num(kv[1][0], nan=-1.0))
        scores[folds == k] = s
        chosen[k] = {"layer": layer, "l2": l2}
    return scores, chosen


def run_computation(args):
    items = load_chartqa_items(*args.chartqa)
    layers = args.layers or cached_layers(args.chartqa[0])
    cls = np.array([it["cls"] for it in items])
    folds = np.array([it["fold"] for it in items])
    figs = np.array([it["figure_id"] for it in items])
    counts = {c: {"total": int((cls == c).sum()),
                  "per_fold": [int(((cls == c) & (folds == k)).sum()) for k in range(N_FOLDS)]}
              for c in "CSP"}
    print("chartqa:", {c: v["total"] for c, v in counts.items()}, flush=True)

    cells = [(p, L, items, tuple(args.l2)) for p in CHARTQA_PROBES for L in [*layers, "surface"]]
    acts, surf = {p: {} for p in CHARTQA_PROBES}, {}
    t0 = time.time()
    with Pool(args.jobs) as pool:
        for n, (p, L, res) in enumerate(pool.imap_unordered(_cv_cell, cells), 1):
            (surf.setdefault(p, {}) if L == "surface" else acts[p])[L] = res
            if n % max(1, len(cells) // 10) == 0 or n == len(cells):
                print(f"  {n}/{len(cells)} cells [{time.time() - t0:.0f}s]", flush=True)

    report = {"n_folds": N_FOLDS, "classes": counts, "probes": {}}
    for p in CHARTQA_PROBES:
        s, chosen = _assemble(acts[p], len(items), folds)
        bs, bchosen = _assemble({"surface": surf[p]["surface"]}, len(items), folds)
        report["probes"][p] = {
            "trained_on": "{} vs {}".format(*CHARTQA_PROBES[p]), "chosen_per_fold": chosen,
            "matrix": evaluate(s, cls, figs, CHARTQA_TARGETS),
            "surface_baseline": {"chosen_per_fold": bchosen,
                                 "matrix": evaluate(bs, cls, figs, CHARTQA_TARGETS)}}
    _write(report, args.out)
    print(f"\n{'probe':15s}" + "".join(f"{t:>22s}" for t in CHARTQA_TARGETS))
    for p, e in report["probes"].items():
        for tag, m in (("acts", e["matrix"]), ("surface", e["surface_baseline"]["matrix"])):
            print(f"{p if tag == 'acts' else '':15s}" + "".join(
                f"{m[t]['auroc']:>8.3f} [{m[t]['ci95'][0]:.2f},{m[t]['ci95'][1]:.2f}]"
                for t in CHARTQA_TARGETS) + f"  {tag}")


# ---------------------------------------------------------------------------

def _write(report, out):
    out.parent.mkdir(parents=True, exist_ok=True)
    out.with_suffix(".json").write_text(json.dumps(report, indent=2, default=float) + "\n")
    print(f"wrote {out.with_suffix('.json')}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("transfer", "computation"):
        s = sub.add_parser(name)
        s.add_argument("--chartqa", nargs=3, required=True,
                       metavar=("NPZ", "PREDICTIONS", "LABELS"))
        s.add_argument("--layers", nargs="*", type=int, default=None)
        s.add_argument("--l2", nargs="*", type=float, default=list(DEFAULT_L2))
        s.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) - 2))
        s.add_argument("--out", required=True, type=pathlib.Path)
        if name == "transfer":
            s.add_argument("--pilot", nargs=4, action="append", required=True,
                           metavar=("NAME", "NPZ", "PREDICTIONS", "MANIFEST"))
    args = ap.parse_args()
    (run_transfer if args.cmd == "transfer" else run_computation)(args)


if __name__ == "__main__":
    main()
