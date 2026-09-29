"""E1 probes and the P4.5 sanity probe, swept in parallel.

    # sanity first: can the cache decode something it obviously contains?
    python -m src.probes.run_sweep --target source --out runs/sanity
    # the binary probe, with its controls
    python -m src.probes.run_sweep --target binary --out runs/binary
    # the failure-type probes, once labels exist
    python -m src.probes.run_sweep --target structural \
        --labels results/labels/qwen2_5_vl_7b_test.claude.jsonl --out runs/structural

Selection of (position, l2, layer) is on VALIDATION only; test is scored once,
at the configuration already fixed.

**Parallelism.** The unit of work is one (position, layer) pair, fitted at every
L2 in the grid. A worker loads a single (2500, 3584) array, so memory per worker
is tiny, and every worker runs single-threaded BLAS. One process with ~35 BLAS
threads on matrices this size spends most of its time synchronising threads;
many single-threaded processes scale close to linearly. The environment
variables below must be set before numpy is first imported, which is why they
sit at the very top of this module.

**Targets.**
  source   P4.5 sanity: human-written vs augmented question. The style is in the
           prompt tokens, so a working cache must decode it at AUROC >= 0.9.
           Below that, the caching is broken and no later number means anything.
  binary   E1: is the model's answer wrong. Reported next to a surface-only
           baseline and scored WITHIN each question source, because ~80% of
           errors are human-written questions and a probe that only detected
           question style would still score well overall.
  structural / fabrication
           E1 type probes, one-vs-rest. Need --labels. --rest chooses what the
           negative class is (ProbeConfig.type_probe_rest): errors_only makes
           the two probes one probe with the label flipped, include_correct
           makes them genuinely different. Run both.
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
from src.probes.dataset import (POSITIONS, ProbeConfig, build_dataset, cached_layers,
                                load_predictions)

DEFAULT_L2 = (1.0, 10.0, 100.0, 1000.0, 10000.0)


TYPE_TARGETS = ("structural", "fabrication")


def _y(target, ds, split, preds):
    if target in ("binary",) + TYPE_TARGETS:
        return ds.y[split]
    if target == "source":
        return np.array([preds[i]["source"] == "human" for i in ds.item_ids[split]],
                        dtype=float)
    raise ValueError(target)


def _dataset(target, npz, pred, labels, rest, pos, layers):
    task = target if target in TYPE_TARGETS else "binary"
    return build_dataset(task, npz, pred, labels if task in TYPE_TARGETS else None,
                         config=ProbeConfig(position=pos, layers=layers,
                                            type_probe_rest=rest))


def _fit_cell(args):
    """One (position, layer): fit every L2 on train, score on val."""
    target, npz, pred, labels, rest, pos, layer, l2_grid = args
    preds = load_predictions(pred)
    ds = _dataset(target, npz, pred, labels, rest, pos, (layer,))
    Xtr, Xva = ds.X["train"][layer], ds.X["val"][layer]
    ytr, yva = _y(target, ds, "train", preds), _y(target, ds, "val", preds)
    out = {}
    for l2 in l2_grid:
        w = fit_logistic(Xtr, ytr, l2)
        out[l2] = auroc(yva, predict_scores(Xva, w))
    return pos, layer, out


def sweep(target, npz, pred, positions, layers, l2_grid, jobs, labels=None,
          rest="errors_only"):
    cells = [(target, str(npz), str(pred), str(labels) if labels else None, rest,
              p, L, tuple(l2_grid))
             for p in positions for L in layers]
    curves = {(p, l2): {} for p in positions for l2 in l2_grid}
    t0 = time.time()
    with Pool(jobs) as pool:
        for k, (p, L, res) in enumerate(pool.imap_unordered(_fit_cell, cells), 1):
            for l2, a in res.items():
                curves[(p, l2)][L] = a
            if k % max(1, len(cells) // 10) == 0 or k == len(cells):
                print(f"  {k}/{len(cells)} cells  [{time.time() - t0:.0f}s]", flush=True)
    return curves


def select(curves):
    """Joint argmax over (position, l2, layer), on validation only."""
    best, best_val = None, -1.0
    for (p, l2), by_layer in curves.items():
        for L, a in by_layer.items():
            if not np.isnan(a) and a > best_val:
                best, best_val = (p, l2, L), a
    if best is None:
        raise SystemExit(
            "every validation AUROC is undefined: the validation split holds only one "
            "class for this target. With few labelled items of one type this happens; "
            "it means there is not enough data to select a layer, not a bug.")
    return best, best_val


def _ci(y, s, figs):
    return cluster_bootstrap_ci(lambda idx: auroc(y[idx], s[idx]), figs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--activations", required=True, type=pathlib.Path)
    ap.add_argument("--predictions", required=True, type=pathlib.Path)
    ap.add_argument("--target", choices=("source", "binary") + TYPE_TARGETS, required=True)
    ap.add_argument("--labels", type=pathlib.Path,
                    help="label file or annotations directory; required for type targets")
    ap.add_argument("--rest", choices=("errors_only", "include_correct"),
                    default="errors_only")
    ap.add_argument("--positions", nargs="*", default=list(POSITIONS))
    ap.add_argument("--layers", nargs="*", type=int, default=None,
                    help="default: every layer in the cache, read from its keys (Qwen 28, LLaVA-NeXT 32)")
    ap.add_argument("--l2", nargs="*", type=float, default=list(DEFAULT_L2))
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    ap.add_argument("--out", type=pathlib.Path, required=True)
    args = ap.parse_args()
    if args.layers is None:
        args.layers = cached_layers(args.activations, args.positions[0])
    if args.target in TYPE_TARGETS and not args.labels:
        ap.error(f"--target {args.target} needs --labels")
    args.out.mkdir(parents=True, exist_ok=True)

    print(f"target={args.target} positions={args.positions} layers={len(args.layers)} "
          f"l2={args.l2} jobs={args.jobs}", flush=True)
    t0 = time.time()
    curves = sweep(args.target, args.activations, args.predictions,
                   args.positions, args.layers, args.l2, args.jobs,
                   labels=args.labels, rest=args.rest)
    (pos, l2, L), best_val = select(curves)

    report = {"target": args.target, "labels": str(args.labels) if args.labels else None,
              "rest": args.rest, "l2_grid": args.l2, "positions": args.positions,
              "layers": args.layers,
              "val_curves": {f"{p}|{l}": {str(k): v for k, v in sorted(c.items())}
                             for (p, l), c in curves.items()},
              "selected": {"position": pos, "l2": l2, "layer": L, "val_auroc": best_val}}

    print(f"\nper-position best on VALIDATION:")
    for p in args.positions:
        a, LL, ll = max((a, LL, ll) for (pp, ll), c in curves.items() if pp == p
                        for LL, a in c.items() if not np.isnan(a))
        print(f"  {p:<12} {a:.3f}  at L{LL} l2={ll:g}")
    print(f"\nselected on VALIDATION: {pos} L{L} l2={l2:g}  (val {best_val:.3f})")

    if args.target == "source":
        passed = best_val >= 0.9
        report["sanity_pass"] = passed
        print(f"\nP4.5 SANITY: {'PASS' if passed else 'FAIL'} (threshold 0.9)")
        if not passed:
            print("  The cache cannot decode question style, which is written into the "
                  "prompt tokens.\n  Treat every downstream number as a pipeline bug "
                  "until this is explained.")
    elif args.target in TYPE_TARGETS:
        ds = _dataset(args.target, args.activations, args.predictions, args.labels,
                      args.rest, pos, (L,))
        w = fit_logistic(ds.X["train"][L], ds.y["train"], l2)
        s = predict_scores(ds.X["test"][L], w)
        y, f = ds.y["test"], ds.figure_ids["test"]
        counts = {sp: {"n": ds.n(sp), "positive": int(ds.y[sp].sum())}
                  for sp in ("train", "val", "test")}
        print(f"\n=== {args.target.upper()} PROBE (rest = {args.rest}) ===")
        for sp, c in counts.items():
            print(f"  {sp:<5} n={c['n']:<5} {args.target}={c['positive']}")
        print(f"  dropped as ambiguous: {ds.dropped_ambiguous}")
        a = auroc(y, s)
        if np.isnan(a):
            print("  TEST AUROC undefined: the test split holds only one class.")
            report["test"] = {"auroc": None, "counts": counts}
        else:
            lo, hi = _ci(y, s, f)
            print(f"  TEST AUROC {a:.3f}  95% CI [{lo:.3f}, {hi:.3f}]")
            if counts["test"]["positive"] < 20:
                print(f"  WARNING: only {counts['test']['positive']} positives in test; "
                      "treat this interval as the result, not the point estimate.")
            report["test"] = {"auroc": a, "ci95": [lo, hi], "counts": counts}
        if args.rest == "errors_only":
            print("  NOTE: with rest = errors_only the structural and fabrication probes "
                  "are one probe\n  with the label flipped; report one of them, not both "
                  "as separate evidence.")
        report["dropped_ambiguous"] = ds.dropped_ambiguous
    else:
        preds = load_predictions(args.predictions)
        ds = build_dataset("binary", args.activations, args.predictions,
                           config=ProbeConfig(position=pos, layers=(L,)))
        w = fit_logistic(ds.X["train"][L], ds.y["train"], l2)   # deterministic refit
        s = predict_scores(ds.X["test"][L], w)
        y, f = ds.y["test"], ds.figure_ids["test"]
        a = auroc(y, s); lo, hi = _ci(y, s, f)
        print(f"\n=== BINARY PROBE, TEST (n={len(y)}, errors={int(y.sum())}) ===")
        print(f"  AUROC {a:.3f}  95% CI [{lo:.3f}, {hi:.3f}]")
        report["test"] = {"n": int(len(y)), "errors": int(y.sum()), "auroc": a, "ci95": [lo, hi]}

        def feats(split):
            return np.array([[preds[i]["source"] == "human", len(preds[i]["question"]),
                              preds[i].get("n_vision_tokens", 0),
                              preds[i]["gold"].replace(".", "", 1).replace("-", "", 1).isdigit()]
                             for i in ds.item_ids[split]], dtype=float)
        S = {k: feats(k) for k in ("train", "val", "test")}
        mu, sd = S["train"].mean(0), S["train"].std(0); sd[sd == 0] = 1
        S = {k: (v - mu) / sd for k, v in S.items()}
        sl2 = max(args.l2, key=lambda l: auroc(ds.y["val"], predict_scores(
            S["val"], fit_logistic(S["train"], ds.y["train"], l))))
        ss = predict_scores(S["test"], fit_logistic(S["train"], ds.y["train"], sl2))
        sa = auroc(y, ss); slo, shi = _ci(y, ss, f)
        print(f"\n=== SURFACE-ONLY BASELINE (source, question length, vision tokens, numeric gold) ===")
        print(f"  AUROC {sa:.3f}  95% CI [{slo:.3f}, {shi:.3f}]")
        print(f"  probe minus surface: {a - sa:+.3f}")
        report["surface_baseline"] = {"auroc": sa, "ci95": [slo, shi], "l2": sl2}

        print(f"\n=== WITHIN-SOURCE AUROC (failure, or just question style?) ===")
        src = np.array([preds[i]["source"] for i in ds.item_ids["test"]])
        report["within_source"] = {}
        for g in ("human", "augmented"):
            m = src == g
            ne = int(y[m].sum())
            if ne in (0, int(m.sum())):
                print(f"  {g:<10} n={int(m.sum())} errors={ne}: one class only"); continue
            ga = auroc(y[m], s[m]); glo, ghi = _ci(y[m], s[m], f[m])
            print(f"  {g:<10} n={int(m.sum()):<4} errors={ne:<3} AUROC {ga:.3f}  95% CI [{glo:.3f}, {ghi:.3f}]")
            report["within_source"][g] = {"n": int(m.sum()), "errors": ne,
                                          "auroc": ga, "ci95": [glo, ghi]}

    report["seconds"] = time.time() - t0
    (args.out / "report.json").write_text(json.dumps(report, indent=1, default=float))
    print(f"\nwrote {args.out / 'report.json'}  ({report['seconds']:.0f}s)")


if __name__ == "__main__":
    main()
