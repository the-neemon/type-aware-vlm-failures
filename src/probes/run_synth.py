"""Probes on the synthetic fabrication pilot (results/synth_fabrication_pilots.md).

Pilot 2 showed Qwen almost never rejects a question about a bar that is not
there (6 of 900), so the planned "fabricated vs rejected" probe has no negative
class. These contrasts ask what the activations do carry instead:

  absent_*_vs_present   Does the model represent that the asked-about name is not
                        on the chart, although it answers anyway? Present and
                        absent questions of one family use identical words and the
                        same charts, and every name is a bar on some charts and
                        absent on others, so the text alone cannot tell them apart.
                        Built-in control: in Qwen's prompt the image tokens come
                        before the question and attention is causal, so the vision
                        positions cannot see the question. Both questions of a
                        chart get identical vision vectors; a vision probe must sit
                        at chance, and if it does not, something leaks.
  value_fab_vs_zero     Among absent value questions, a made-up number (mostly a
                        lookalike bar's value) vs "0". Also within lookalike items
                        only, since having a lookalike on the chart predicts
                        fabrication by itself.
  compare_fab_vs_present  Among absent compare questions, naming the absent bar or
                        its lookalike vs naming the present bar.
  read_misread          Present value questions misread vs read right (structural).

Selection of (position, L2, layer) is on validation; test is scored once. Splits
are by figure (src.probes.dataset.assign_figure_splits), shared by all contrasts.
Baselines use question-text features only (asked name, other name, phrasing).

    python -m src.probes.run_synth --activations <synth_pilot2.npz> \\
        --predictions <qwen2_5_vl_7b_synth_pilot2.jsonl> --manifest <manifest.jsonl> \\
        --out results/probes/synth_pilot2_qwen2_5_vl_7b
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
from src.probes.dataset import (POSITIONS, assert_no_figure_leak, assign_figure_splits,
                                cached_layers, load_activations, load_predictions)
from src.synth import items as synth_items

DEFAULT_L2 = (1.0, 10.0, 100.0, 1000.0, 10000.0)
VISION = ("vision_mean", "vision_max")
MIN_PER_CLASS = 5          # in each of val and test, or the contrast is skipped


def _family(r):
    return {"read_value": "value", "absent_value": "value", "absent_category": "value",
            "compare_present": "compare", "absent_compare": "compare",
            "neighbor_present": "neighbor", "absent_neighbor": "neighbor"}[r["template"]]


# name -> (row filter, label, description)
CONTRASTS = {
    "absent_value_vs_present": (
        lambda r: _family(r) == "value", lambda r: r["absent"],
        "value question: asked name absent (1) vs a shown bar (0)"),
    "absent_compare_vs_present": (
        lambda r: _family(r) == "compare", lambda r: r["absent"],
        "compare question: one name absent (1) vs both shown (0)"),
    "absent_neighbor_vs_present": (
        lambda r: _family(r) == "neighbor", lambda r: r["absent"],
        "neighbor question: asked name absent (1) vs a shown bar (0)"),
    "value_fab_vs_zero": (
        lambda r: r["template"] == "absent_value" and r["outcome"] in ("fabricated", "zero"),
        lambda r: r["outcome"] == "fabricated",
        "absent value question: made-up number (1) vs \"0\" (0)"),
    "value_fab_vs_zero_lookalike": (
        lambda r: (r["template"] == "absent_value" and r["lookalike_of"] is not None
                   and r["outcome"] in ("fabricated", "zero")),
        lambda r: r["outcome"] == "fabricated",
        "as above, lookalike names only"),
    "compare_fab_vs_present": (
        lambda r: (r["template"] == "absent_compare"
                   and r["outcome"] in ("fabricated", "picked_present")),
        lambda r: r["outcome"] == "fabricated",
        "absent compare question: names the absent bar or its lookalike (1) vs the shown one (0)"),
    "read_misread": (
        lambda r: r["template"] == "read_value", lambda r: not r["correct"],
        "value question about a shown bar: misread (1) vs read right (0)"),
}


def rescore(preds: dict, manifest: pathlib.Path) -> dict:
    """Re-apply the current scorer, which may be newer than the run's."""
    figures = {}
    for line in manifest.read_text(encoding="utf-8").splitlines():
        if line.strip():
            f = json.loads(line)
            figures[f["figure_id"]] = f
    for r in preds.values():
        item = dict(r, gold=r["gold"], categories=figures[r["figure_id"]]["categories"])
        r.update(synth_items.score(item, r["prediction"]))
    return preds


def build_index(preds, item_ids, figure_ids, seed=42):
    """Per contrast: row indices into the cache, labels, figures and splits."""
    split_of = assign_figure_splits(figure_ids, seed=seed)
    row_of = {iid: i for i, iid in enumerate(item_ids)}
    out = {}
    for name, (keep, label, _) in CONTRASTS.items():
        rows = [r for r in preds.values() if keep(r)]
        idx = np.array([row_of[r["item_id"]] for r in rows])
        y = np.array([bool(label(r)) for r in rows], dtype=float)
        figs = np.array([r["figure_id"] for r in rows])
        split = np.array([split_of[f] for f in figs])
        assert_no_figure_leak({s: figs[split == s] for s in ("train", "val", "test")})
        out[name] = {"idx": idx, "y": y, "figs": figs, "split": split, "rows": rows}
    return out


def usable(c):
    return all(min(c["y"][c["split"] == s].sum(), (1 - c["y"][c["split"] == s]).sum())
               >= MIN_PER_CLASS for s in ("val", "test"))


def _standardise(X, train):
    mu, sd = X[train].mean(0), X[train].std(0)
    return (X - mu) / np.where(sd < 1e-6, 1.0, sd)


def _fit_cell(args):
    npz, pos, layer, index, l2_grid = args
    by_layer, _, _ = load_activations(npz, pos, (layer,))
    A = by_layer[layer]
    out = {}
    for name, c in index.items():
        X = _standardise(A[c["idx"]].astype(np.float32), c["split"] == "train")
        tr, va = c["split"] == "train", c["split"] == "val"
        out[name] = {l2: auroc(c["y"][va], predict_scores(X[va], fit_logistic(X[tr], c["y"][tr], l2)))
                     for l2 in l2_grid}
    return pos, layer, out


def test_scores(npz, c, pos, layer, l2):
    """Fit on train at the chosen setting; scores for the test rows only."""
    by_layer, _, _ = load_activations(npz, pos, (layer,))
    X = _standardise(by_layer[layer][c["idx"]].astype(np.float32), c["split"] == "train")
    tr, te = c["split"] == "train", c["split"] == "test"
    return predict_scores(X[te], fit_logistic(X[tr], c["y"][tr], l2))


def test_at(npz, c, pos, layer, l2):
    s = test_scores(npz, c, pos, layer, l2)
    te = c["split"] == "test"
    y, figs = c["y"][te], c["figs"][te]
    return auroc(y, s), cluster_bootstrap_ci(lambda i: auroc(y[i], s[i]), figs)


def subset_auroc(c, scores, keep):
    """Test AUROC over the test rows `keep(row)` selects."""
    rows = [r for r, sp in zip(c["rows"], c["split"]) if sp == "test"]
    m = np.array([keep(r) for r in rows])
    return auroc(c["y"][c["split"] == "test"][m], scores[m])


def text_baseline(c, l2_grid):
    """Logistic on one-hot question-text features; L2 picked on val, scored on test."""
    keys = sorted({f"{k}={r.get(k)}" for r in c["rows"]
                   for k in ("asks_about", "compared_with", "phrasing")})
    col = {k: j for j, k in enumerate(keys)}
    X = np.zeros((len(c["rows"]), len(keys)))
    for i, r in enumerate(c["rows"]):
        for k in ("asks_about", "compared_with", "phrasing"):
            X[i, col[f"{k}={r.get(k)}"]] = 1
    tr, va, te = (c["split"] == s for s in ("train", "val", "test"))
    fits = {l2: fit_logistic(X[tr], c["y"][tr], l2) for l2 in l2_grid}
    best = max(l2_grid, key=lambda l2: np.nan_to_num(auroc(c["y"][va], predict_scores(X[va], fits[l2]))))
    return auroc(c["y"][te], predict_scores(X[te], fits[best]))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--activations", required=True, type=pathlib.Path)
    ap.add_argument("--predictions", required=True, type=pathlib.Path)
    ap.add_argument("--manifest", required=True, type=pathlib.Path)
    ap.add_argument("--positions", nargs="*", default=list(POSITIONS))
    ap.add_argument("--layers", nargs="*", type=int, default=None,
                    help="default: every layer in the cache, read from its keys (Qwen 28, LLaVA-NeXT 32)")
    ap.add_argument("--l2", nargs="*", type=float, default=list(DEFAULT_L2))
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    ap.add_argument("--out", required=True, type=pathlib.Path)
    args = ap.parse_args()
    if args.layers is None:
        args.layers = cached_layers(args.activations, args.positions[0])

    preds = rescore(load_predictions(args.predictions), args.manifest)
    _, item_ids, figure_ids = load_activations(args.activations, args.positions[0],
                                               (args.layers[0],))
    if set(item_ids) != set(preds):
        raise SystemExit("cache and predictions hold different items")
    index = build_index(preds, item_ids, figure_ids)
    skipped = [n for n, c in index.items() if not usable(c)]
    index = {n: c for n, c in index.items() if n not in skipped}
    print(f"contrasts: {list(index)}; skipped (fewer than {MIN_PER_CLASS} per class "
          f"in val or test): {skipped}", flush=True)

    cells = [(str(args.activations), p, L, index, tuple(args.l2))
             for p in args.positions for L in args.layers]
    val = {n: {} for n in index}                       # name -> {(pos, l2, layer): auroc}
    t0 = time.time()
    with Pool(args.jobs) as pool:
        for k, (p, L, res) in enumerate(pool.imap_unordered(_fit_cell, cells), 1):
            for n, by_l2 in res.items():
                for l2, a in by_l2.items():
                    val[n][(p, l2, L)] = a
            if k % max(1, len(cells) // 10) == 0 or k == len(cells):
                print(f"  {k}/{len(cells)} cells [{time.time() - t0:.0f}s]", flush=True)

    report = {"activations": str(args.activations), "predictions": str(args.predictions),
              "skipped": skipped, "contrasts": {}}
    for n, c in index.items():
        def best(keys):
            keys = [k for k in keys if not np.isnan(val[n][k])]
            return max(keys, key=val[n].get) if keys else None
        entry = {"description": CONTRASTS[n][2],
                 "n": {s: [int((c["y"][c["split"] == s] == v).sum()) for v in (1, 0)]
                       for s in ("train", "val", "test")},
                 "text_baseline_test_auroc": text_baseline(c, args.l2)}
        for tag, keys in (("probe", list(val[n])),
                          ("vision_only", [k for k in val[n] if k[0] in VISION])):
            k = best(keys)
            if k is None:
                continue
            test, ci = test_at(str(args.activations), c, *k[:1], k[2], k[1])
            entry[tag] = {"position": k[0], "l2": k[1], "layer": k[2],
                          "val_auroc": val[n][k], "test_auroc": test, "test_ci95": list(ci)}
        if n.startswith("absent_") and "probe" in entry:
            pr = entry["probe"]
            s = test_scores(str(args.activations), c, pr["position"], pr["layer"], pr["l2"])
            # the hardest absent names: a lookalike of one bar is on the chart
            entry["test_auroc_lookalike_absent_only"] = subset_auroc(
                c, s, lambda r: not r["absent"] or r["lookalike_of"] is not None)
            entry["test_auroc_unrelated_absent_only"] = subset_auroc(
                c, s, lambda r: not r["absent"] or r["lookalike_of"] is None)
            # E3-style cross test: among present questions, does the absence
            # probe's score separate misread answers from right ones?
            rows = [r for r, sp in zip(c["rows"], c["split"]) if sp == "test"]
            present = np.array([not r["absent"] for r in rows])
            wrong = np.array([not r["correct"] for r in rows], dtype=float)
            entry["cross_present_errors"] = {
                "n_wrong": int(wrong[present].sum()), "n_right": int((1 - wrong[present]).sum()),
                "auroc_absence_score_predicts_error": auroc(wrong[present], s[present])}
        entry["val_by_layer_best_position"] = {
            L: max(val[n][(p, l2, L)] for p in args.positions for l2 in args.l2)
            for L in args.layers}
        if "lookalike" not in n and n.startswith(("value_fab", "compare_fab")):
            te = c["split"] == "test"
            flag = np.array([r["lookalike_of"] is not None for r in c["rows"]], dtype=float)
            entry["lookalike_flag_test_auroc"] = auroc(c["y"][te], flag[te])
        report["contrasts"][n] = entry
        pr = entry.get("probe", {})
        print(f"{n:30s} n(test pos/neg)={entry['n']['test']}  probe test "
              f"{pr.get('test_auroc', float('nan')):.3f} {pr.get('position')} L{pr.get('layer')}  "
              f"text {entry['text_baseline_test_auroc']:.3f}  vision "
              f"{entry.get('vision_only', {}).get('test_auroc', float('nan')):.3f}", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.with_suffix(".json").write_text(json.dumps(report, indent=2, default=float) + "\n")
    print(f"wrote {args.out.with_suffix('.json')}")


if __name__ == "__main__":
    main()
