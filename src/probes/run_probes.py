"""Run E1 and E3 over a cached activation set (P5.1, P5.2, P5.3, P5.6).

    python3 -m src.probes.run_probes \
        --activations ~/activations/qwen2_5_vl_7b/test.npz \
        --predictions results/predictions/qwen2_5_vl_7b_test.jsonl \
        --annotations annotations/ \
        --position query_last \
        --out results/probes_qwen_query_last.json

What it does, and the order matters:

  P4.5  sanity probe first. If the activations cannot predict something
        trivially decodable, the caching is broken and every number below is
        about the code rather than the hypothesis. Run before believing a null.
  E1    per-layer validation AUROC for `binary`, `structural`, `fabrication`;
        the layer is chosen on validation and scored once on test.
  E3    cross-transfer between the two type probes at the selected layer.

**E2 is not here.** The surface baseline lives in `src/surface/` and needs the
figure's text, which the activation cache does not carry. See the note at the
bottom of this file: ChartQA ships the underlying data tables, which is the
cheapest source for it. No probe number should be reported without it
(TASKS 5.10).
"""

from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np

from src.probes.dataset import (
    POSITIONS, ProbeConfig, TYPE_TASKS, build_dataset, validate_cache,
)
from src.probes.sweep import (
    cross_transfer, evaluate_at, select_layer, sweep_layers,
)


def _run_one(task, paths, cfg, positions, l2_grid) -> tuple[dict, object, object]:
    """Sweep every (position, layer) on validation, then score test once.

    The extractor caches four pooled positions per layer, so there are two
    things to choose rather than one. **Both are chosen on validation.**

    That is the whole point. Picking the best layer by test AUROC is the classic
    way to invent a result, and `sweep_layers` is shaped so it cannot happen by
    accident. Sweeping positions re-opens the same hole one level up: four
    positions times 28 layers is 112 chances to find something, and picking the
    winner by test score would inflate the reported number just as effectively.
    So the joint argmax is taken over validation only, and test is touched once,
    at the pair already fixed.

    One consequence to state in the paper rather than hide: the *validation*
    number is optimistically biased, because it is the maximum over 4 positions
    x 28 layers x |l2_grid| combinations, which is several hundred draws on a
    validation set of a few hundred items. The *test* number is not, because
    test is scored exactly once at a configuration chosen without it. Report the
    test AUROC with its interval; use the validation curve to show shape, never
    as the headline.
    """
    from dataclasses import replace

    results, datasets = {}, {}
    for pos in positions:
        ds = build_dataset(task, paths["activations"], paths["predictions"],
                           paths["annotations"], config=replace(cfg, position=pos))
        datasets[pos] = ds
        for l2 in l2_grid:
            results[(pos, l2)] = sweep_layers(
                ds.X["train"], ds.y["train"], ds.X["val"], ds.y["val"], l2=l2)

    # Joint argmax on VALIDATION over (position, L2, layer). Deterministic:
    # ties break toward the earlier position, the smaller L2 and the shallower
    # layer, because the iteration order is fixed and the test is strict.
    best, best_val = None, -1.0
    for (pos, l2), res in results.items():
        for layer, a in res.auroc_by_layer.items():
            if not np.isnan(a) and a > best_val:
                best, best_val = (pos, l2, layer), a
    best_pos, best_l2, best_layer = best

    ds = datasets[best_pos]
    scored = evaluate_at(best_layer, results[(best_pos, best_l2)],
                         ds.X["test"], ds.y["test"],
                         figure_ids=ds.figure_ids["test"])
    return {
        "task": task,
        "n": {k: ds.n(k) for k in ("train", "val", "test")},
        "positive_rate": {k: ds.positive_rate(k) for k in ("train", "val", "test")},
        "dropped_ambiguous": ds.dropped_ambiguous,
        "auroc_by_layer_val": {
            pos: {str(k): v
                  for k, v in results[(pos, best_l2)].auroc_by_layer.items()}
            for pos in positions},
        "selected_position": best_pos,
        "selected_l2": best_l2,
        "selected_layer": best_layer,
        "selection_val_auroc": best_val,
        "l2_grid": list(l2_grid),
        "test": scored,
    }, results[(best_pos, best_l2)], ds


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--activations", required=True, type=pathlib.Path)
    ap.add_argument("--predictions", required=True, type=pathlib.Path)
    ap.add_argument("--annotations", type=pathlib.Path,
                    help="directory of per-rater .jsonl files; omit to run the "
                         "binary probe alone, which needs no labels")
    ap.add_argument("--position", default="all",
                    help="'all' sweeps every cached pooled position and picks "
                         "the (position, layer) pair on VALIDATION; or name one "
                         "of vision_mean/vision_max/query_last/query_mean")
    ap.add_argument("--no-validate", action="store_true",
                    help="skip the activation-cache structure check")
    ap.add_argument("--layers", type=int, nargs="*", default=None,
                    help="subset of cached layers; default is every layer present")
    ap.add_argument("--rest", default="errors_only",
                    choices=("errors_only", "include_correct"),
                    help="what the type probes treat as the negative class. "
                         "errors_only matches the hypothesis but makes E3's "
                         "off-diagonal algebraically forced; include_correct "
                         "makes cross-transfer informative. Run both.")
    ap.add_argument("--l2", type=float, nargs="*",
                    default=[0.1, 1.0, 10.0, 100.0, 1000.0],
                    help="L2 grid, selected on VALIDATION. The default spans "
                         "four orders of magnitude because the probe runs at "
                         "p >> n: 3584 features against ~1500 training rows "
                         "for the binary probe and ~240 for the type probes. "
                         "A single weak value overfits and reports chance.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=pathlib.Path)
    args = ap.parse_args()

    positions = list(POSITIONS) if args.position == "all" else [args.position]
    cfg = ProbeConfig(
        layers=tuple(args.layers) if args.layers else None,
        position=positions[0], l2=args.l2[0], seed=args.seed,
        type_probe_rest=args.rest,
    )

    if not args.no_validate:
        rep = validate_cache(args.activations)
        print(f"cache OK: {rep['n_items']} items, "
              f"positions {sorted(k for k, v in rep['layers_by_position'].items() if v)}, "
              f"{len(rep['layers_by_position'][positions[0]])} layers")
        for w in rep["warnings"]:
            print(f"  warning: {w}")

    paths = {"activations": args.activations, "predictions": args.predictions,
             "annotations": args.annotations}

    report: dict = {
        "config": {
            "positions_swept": positions, "l2_grid": args.l2,
            "seed": cfg.seed,
            "type_probe_rest": cfg.type_probe_rest,
            "layers": list(cfg.layers) if cfg.layers else "all",
            "activations": str(args.activations),
            "predictions": str(args.predictions),
            "annotations": str(args.annotations) if args.annotations else None,
        },
        "probes": {},
    }

    # --- E1: binary ------------------------------------------------------
    report["probes"]["binary"], _, _ = _run_one("binary", paths, cfg, positions, args.l2)
    _print(report["probes"]["binary"])

    # --- E1: the two type probes, and E3 --------------------------------
    if args.annotations:
        for task in TYPE_TASKS:
            report["probes"][task], _, _ = _run_one(task, paths, cfg, positions, args.l2)
            _print(report["probes"][task])

        # E3 needs one (position, layer) and one population. Use whatever the
        # structural probe selected on validation; both type probes share items.
        from dataclasses import replace as _replace
        sel_pos = report["probes"]["structural"]["selected_position"]
        layer = report["probes"]["structural"]["selected_layer"]
        ds = build_dataset("structural", args.activations, args.predictions,
                           args.annotations, config=_replace(cfg, position=sel_pos))
        # Use the true class per item. Deriving it from the binary y would
        # relabel every correct item as "fabrication" and collapse the three
        # groups back into two, which forces the off-diagonal even in
        # include_correct mode.
        transfer = cross_transfer(
            ds.X["train"], ds.classes["train"],
            ds.X["test"], ds.classes["test"],
            layer=layer, classes=list(TYPE_TASKS), l2=cfg.l2)
        report["cross_transfer"] = {
            "position": sel_pos, "layer": layer,
            "cells": {f"{a}->{b}": v for (a, b), v in transfer.items()},
        }
        print(f"\nE3 cross-transfer at {sel_pos} layer {layer}")
        for (a, b), v in sorted(transfer.items()):
            mark = "" if a == b else "   <- off-diagonal"
            print(f"  trained {a:<12} tested {b:<12} AUROC {v:.3f}{mark}")
        if cfg.type_probe_rest == "errors_only":
            report["cross_transfer"]["degenerate"] = True
            print("\n  WARNING: rest = errors_only, so the two type probes are "
                  "one probe\n  with the label flipped. Every off-diagonal cell "
                  "above is exactly\n  1 - its diagonal by construction, not by "
                  "measurement. Do not report\n  it as evidence. Re-run with "
                  "--rest include_correct for an E3 that\n  can actually come "
                  "out either way.")
        else:
            print("\n  Strong off-diagonal transfer means one signal wearing two "
                  "labels.\n  That is a reportable finding, not a failure.")
    else:
        print("\nno --annotations given; type probes and E3 skipped")

    print("\nREMINDER: no probe number is a result without its E2 surface "
          "baseline\n(TASKS 5.10). See the note in this module's docstring.")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2, default=float))
        print(f"\nwrote {args.out}")
    return 0


def _print(r: dict) -> None:
    print(f"\n=== {r['task']} ===")
    print(f"  n train/val/test : {r['n']['train']}/{r['n']['val']}/{r['n']['test']}"
          f"   positive rate (train) {r['positive_rate']['train']:.3f}")
    if r["dropped_ambiguous"]:
        print(f"  dropped ambiguous: {r['dropped_ambiguous']}")
    for pos, by_layer in r["auroc_by_layer_val"].items():
        top = sorted(((float(v), k) for k, v in by_layer.items()), reverse=True)[:3]
        star = " *" if pos == r["selected_position"] else "  "
        print(f" {star} {pos:<12} best val: " +
              ", ".join(f"L{k}={v:.3f}" for v, k in top))
    print(f"  selected on VALIDATION: {r['selected_position']} "
          f"L{r['selected_layer']} l2={r['selected_l2']:g} "
          f"(val {r['selection_val_auroc']:.3f})")
    t = r["test"]
    ci = t.get("ci95")
    ci_s = f"  95% CI [{ci[0]:.3f}, {ci[1]:.3f}]" if ci else ""
    print(f"  TEST: AUROC {t['auroc']:.3f}{ci_s}")


if __name__ == "__main__":
    raise SystemExit(main())

# ---------------------------------------------------------------------------
# Note on wiring E2, for whoever picks it up
# ---------------------------------------------------------------------------
# src/surface/features.py::answer_in_figure needs the text that appears in the
# figure. Do not OCR the PNGs for this. ChartQA ships the underlying data tables
# alongside the images:
#
#     ~/data/ChartQA/test/tables/<figure_id>.csv
#
# Those are the exact values and category names the chart was rendered from, so
# they are a cleaner and cheaper source of "is the answer present in the figure"
# than anything recovered from pixels.
#
# Once featurised, E2 has two halves and TASKS 5.10 wants both:
#   1. the surface-only baseline AUROC, reported next to every probe number;
#   2. the probe re-run on residualise(A, S, fit_on=train_mask), which is the
#      number that decides whether we have a mechanism or a confound.
# Fit the projection on train rows only. Estimating it on the pooled set leaks
# test information, which is a quieter version of selecting a layer on test.
