"""Type-aware probes on the synthetic value questions, with the cross-type matrix (E3).

"What is the value of X?" is the one template where both failure types occur
under identical wording. Pooling synthetic pilots 1 and 2 gives four classes:

    C  correct read of a bar that is on the chart
    S  structural: misread of a bar that is on the chart
    F  fabrication: a made-up number for a name that is not on the chart
    Z  "0" for a name that is not on the chart (a refusal expressed as a number,
       docs/taxonomy.md; an error, but not fabrication)

Rejections of an absent name ("None", 15 items) are too few to use and are left out.

Probes, each a linear probe on the query_last activation:

    structural     S vs C
    fabrication    F vs C
    fab_vs_zero    F vs Z   (making up a number, beyond noticing the bar is missing)

Every probe scores every item out of fold, and the matrix reports each probe's
AUROC on each contrast (S vs C, F vs C, Z vs C, F vs Z). The diagonal is each
probe on its own task; the off-diagonal cells ask whether a probe trained for one
failure type also fires on the other. Type-aware signals look like a diagonal.

Why cross-validation: 46 structural errors in all. One held-out test split would
hold about 9, so every item is scored out of fold instead. Folds are by figure
(5, hashed), so no chart is on both sides of a fit. Within each outer fold, layer
and L2 are chosen by inner cross-validation over the remaining folds, never on
the fold being scored. query_last is fixed in advance: it was best for every
contrast in results/probes/synth_pilot2_qwen2_5_vl_7b.json.

Baseline: the same probes and folds on chart metadata instead of activations
(pilot, axis range, tick density, bar count, lookalike present, phrasing). Misreads
concentrate on 0-200 sparse axes and made-up numbers on lookalike names, so the
activations have to beat what the chart settings alone predict.

    python -m src.probes.run_synth_types \\
        --pilot pilot1 <synth_pilot.npz> <..._synth_pilot.jsonl> <absent_pilot_v1/manifest.jsonl> \\
        --pilot pilot2 <synth_pilot2.npz> <..._synth_pilot2.jsonl> <absent_pilot_v2/manifest.jsonl> \\
        --out results/probes/synth_types_qwen2_5_vl_7b
"""

import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import hashlib
import json
import pathlib
import time
from multiprocessing import Pool

import numpy as np

from src.common.linear import auroc, cluster_bootstrap_ci, fit_logistic, predict_scores
from src.probes.dataset import cached_layers, load_activations, load_predictions
from src.probes.run_synth import rescore

POSITION = "query_last"
N_FOLDS = 5
DEFAULT_L2 = (1.0, 10.0, 100.0, 1000.0, 10000.0)
PROBES = {"structural": ("S", "C"), "fabrication": ("F", "C"), "fab_vs_zero": ("F", "Z")}
TARGETS = {"S_vs_C": ("S", "C"), "F_vs_C": ("F", "C"), "Z_vs_C": ("Z", "C"),
           "F_vs_Z": ("F", "Z")}
VALUE_TEMPLATES = ("read_value", "absent_category", "absent_value")


def item_class(r) -> str | None:
    if r["template"] not in VALUE_TEMPLATES:
        return None
    if not r["absent"]:
        return "C" if r["correct"] else "S"
    return {"fabricated": "F", "zero": "Z"}.get(r["outcome"])


def fold_of(figure_id: str, seed: int = 42) -> int:
    return int(hashlib.sha1(f"{seed}\x1f{figure_id}".encode()).hexdigest()[:8], 16) % N_FOLDS


def load_items(pilots):
    """Value-question items of every pilot, with class, fold and metadata."""
    items = []
    for name, npz, preds_path, manifest in pilots:
        preds = rescore(load_predictions(preds_path), pathlib.Path(manifest))
        figures = {f["figure_id"]: f for f in map(json.loads, pathlib.Path(manifest)
                                                  .read_text(encoding="utf-8").splitlines()) }
        for r in preds.values():
            cls = item_class(r)
            if cls is None:
                continue
            f = figures[r["figure_id"]]
            items.append({"item_id": r["item_id"], "figure_id": r["figure_id"], "pilot": name,
                          "npz": str(npz), "cls": cls, "fold": fold_of(r["figure_id"]),
                          "meta": {"pilot": name, "axis_max": f["axis_max"],
                                   "tick_density": f["tick_density"],
                                   "n_bars": len(f["categories"]),
                                   "lookalike": r.get("lookalike_of") is not None,
                                   "phrasing": r["phrasing"]}})
    return items


def activations(items, layer):
    """(n_items, hidden) at `layer`, rows in `items` order, joined on item_id."""
    rows = []
    for npz in dict.fromkeys(it["npz"] for it in items):
        by_layer, ids, _ = load_activations(npz, POSITION, (layer,))
        row_of = {iid: i for i, iid in enumerate(ids)}
        rows.append((npz, by_layer[layer], row_of))
    out = None
    for npz, A, row_of in rows:
        sel = [k for k, it in enumerate(items) if it["npz"] == npz]
        if out is None:
            out = np.zeros((len(items), A.shape[1]), dtype=np.float32)
        out[sel] = A[[row_of[items[k]["item_id"]] for k in sel]].astype(np.float32)
    return out


def metadata(items):
    keys = sorted({f"{k}={v}" for it in items for k, v in it["meta"].items()})
    col = {k: j for j, k in enumerate(keys)}
    X = np.zeros((len(items), len(keys)), dtype=np.float32)
    for i, it in enumerate(items):
        for k, v in it["meta"].items():
            X[i, col[f"{k}={v}"]] = 1
    return X


def _fit_score(X, y, train, score, l2):
    mu, sd = X[train].mean(0), X[train].std(0)
    sd = np.where(sd < 1e-6, 1.0, sd)
    w = fit_logistic((X[train] - mu) / sd, y[train], l2)
    return predict_scores((X[score] - mu) / sd, w)


def cross_validate(X, cls, folds, probe, l2_grid, probes=None, outer=None):
    """Per outer fold: inner-CV AUROC per L2, and scores for every item of that fold.

    `probes` maps a probe name to its (positive, negative) classes; default PROBES.
    `outer` limits the outer folds computed (default all), for callers whose
    features depend on the outer fold.
    """
    pos, neg = (probes or PROBES)[probe]
    member = np.isin(cls, (pos, neg))
    y = (cls == pos).astype(float)
    out = {}
    for k in (range(N_FOLDS) if outer is None else outer):
        fold_k = folds == k
        res = {}
        for l2 in l2_grid:
            inner = []
            for j in range(N_FOLDS):
                if j == k:
                    continue
                train = member & ~fold_k & (folds != j)
                val = member & (folds == j)
                if len(set(y[val])) == 2:
                    inner.append(auroc(y[val], _fit_score(X, y, train, val, l2)))
            res[l2] = (float(np.mean(inner)) if inner else float("nan"),
                       _fit_score(X, y, member & ~fold_k, fold_k, l2))
        out[k] = res
    return out


def _cell(args):
    probe, layer, items, l2_grid = args
    cls = np.array([it["cls"] for it in items])
    folds = np.array([it["fold"] for it in items])
    X = metadata(items) if layer == "meta" else activations(items, layer)
    return probe, layer, cross_validate(X, cls, folds, probe, l2_grid)


def select_and_assemble(results, n_items, folds):
    """Per outer fold, the (layer, L2) with the best inner AUROC; out-of-fold scores."""
    scores = np.full(n_items, np.nan)
    chosen = {}
    for k in range(N_FOLDS):
        (layer, l2), (_, s) = max(
            (((layer, l2), v) for layer, by_fold in results.items()
             for l2, v in by_fold[k].items()),
            key=lambda kv: np.nan_to_num(kv[1][0], nan=-1.0))
        scores[folds == k] = s
        chosen[k] = {"layer": layer, "l2": l2}
    return scores, chosen


def matrix(scores, cls, figs, targets=None):
    out = {}
    for t, (pos, neg) in (targets or TARGETS).items():
        m = np.isin(cls, (pos, neg))
        y, s, g = (cls[m] == pos).astype(float), scores[m], figs[m]
        out[t] = {"auroc": auroc(y, s),
                  "ci95": list(cluster_bootstrap_ci(lambda i: auroc(y[i], s[i]), g)),
                  "n_pos": int(y.sum()), "n_neg": int((1 - y).sum())}
    return out


def pilot_layers(pilots) -> list[int]:
    """The layers every pilot's cache holds. Caches from different models disagree."""
    found = {npz: cached_layers(npz) for _, npz, _, _ in pilots}
    if len({tuple(v) for v in found.values()}) != 1:
        raise SystemExit(f"the pilots' caches hold different layers: {found}")
    return next(iter(found.values()))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--pilot", nargs=4, action="append", required=True,
                    metavar=("NAME", "NPZ", "PREDICTIONS", "MANIFEST"))
    ap.add_argument("--layers", nargs="*", type=int, default=None,
                    help="default: every layer in the cache, read from its keys (Qwen 28, LLaVA-NeXT 32)")
    ap.add_argument("--l2", nargs="*", type=float, default=list(DEFAULT_L2))
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    ap.add_argument("--out", required=True, type=pathlib.Path)
    args = ap.parse_args()
    if args.layers is None:
        args.layers = pilot_layers(args.pilot)

    items = load_items(args.pilot)
    cls = np.array([it["cls"] for it in items])
    folds = np.array([it["fold"] for it in items])
    figs = np.array([it["figure_id"] for it in items])
    counts = {c: {"total": int((cls == c).sum()),
                  "per_fold": [int(((cls == c) & (folds == k)).sum()) for k in range(N_FOLDS)]}
              for c in "CSFZ"}
    print("items:", {c: v["total"] for c, v in counts.items()}, flush=True)

    cells = [(p, L, items, tuple(args.l2)) for p in PROBES for L in [*args.layers, "meta"]]
    results = {p: {} for p in PROBES}
    meta_results = {}
    t0 = time.time()
    with Pool(args.jobs) as pool:
        for n, (p, L, res) in enumerate(pool.imap_unordered(_cell, cells), 1):
            (meta_results.setdefault(p, {}) if L == "meta" else results[p])[L] = res
            if n % max(1, len(cells) // 10) == 0 or n == len(cells):
                print(f"  {n}/{len(cells)} cells [{time.time() - t0:.0f}s]", flush=True)

    report = {"position": POSITION, "n_folds": N_FOLDS, "pilots": [p[0] for p in args.pilot],
              "classes": counts, "probes": {}}
    for p in PROBES:
        scores, chosen = select_and_assemble(results[p], len(items), folds)
        meta_scores, meta_chosen = select_and_assemble({"meta": meta_results[p]["meta"]},
                                                       len(items), folds)
        # own-task AUROC by layer, L2 chosen per fold by inner CV at that layer
        by_layer = {}
        pos, neg = PROBES[p]
        own = np.isin(cls, (pos, neg))
        for L in args.layers:
            s_L, _ = select_and_assemble({L: results[p][L]}, len(items), folds)
            by_layer[L] = auroc((cls[own] == pos).astype(float), s_L[own])
        report["probes"][p] = {
            "trained_on": f"{pos} vs {neg}", "chosen_per_fold": chosen,
            "matrix": matrix(scores, cls, figs),
            "metadata_baseline": {"chosen_per_fold": meta_chosen,
                                  "matrix": matrix(meta_scores, cls, figs)},
            "own_task_auroc_by_layer": by_layer}

    print(f"\n{'probe':13s}" + "".join(f"{t:>22s}" for t in TARGETS))
    for p, e in report["probes"].items():
        for tag, m in (("acts", e["matrix"]), ("metadata", e["metadata_baseline"]["matrix"])):
            print(f"{p if tag == 'acts' else '':13s}" + "".join(
                f"{m[t]['auroc']:>8.3f} [{m[t]['ci95'][0]:.2f},{m[t]['ci95'][1]:.2f}]"
                for t in TARGETS) + f"  {tag}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.with_suffix(".json").write_text(json.dumps(report, indent=2, default=float) + "\n")
    print(f"wrote {args.out.with_suffix('.json')}")


if __name__ == "__main__":
    main()
