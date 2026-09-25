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

from src.probes.dataset import ProbeConfig, TYPE_TASKS, build_dataset
from src.probes.sweep import (
    cross_transfer, evaluate_at, select_layer, sweep_layers,
)


def _run_one(ds, l2: float) -> dict:
    """Sweep on validation, pick a layer, score test once."""
    res = sweep_layers(ds.X["train"], ds.y["train"],
                       ds.X["val"], ds.y["val"], l2=l2)
    layer = select_layer(res)
    scored = evaluate_at(layer, res, ds.X["test"], ds.y["test"],
                         figure_ids=ds.figure_ids["test"])
    return {
        "task": ds.task,
        "n": {s: ds.n(s) for s in ("train", "val", "test")},
        "positive_rate": {s: ds.positive_rate(s) for s in ("train", "val", "test")},
        "dropped_ambiguous": ds.dropped_ambiguous,
        "auroc_by_layer_val": {str(k): v for k, v in res.auroc_by_layer.items()},
        "selected_layer": layer,
        "test": scored,
    }, res


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--activations", required=True, type=pathlib.Path)
    ap.add_argument("--predictions", required=True, type=pathlib.Path)
    ap.add_argument("--annotations", type=pathlib.Path,
                    help="directory of per-rater .jsonl files; omit to run the "
                         "binary probe alone, which needs no labels")
    ap.add_argument("--position", default="query_last",
                    help="which pooled vector to probe; sweep all four and "
                         "report them all rather than picking one")
    ap.add_argument("--layers", type=int, nargs="*", default=None,
                    help="subset of cached layers; default is every layer present")
    ap.add_argument("--rest", default="errors_only",
                    choices=("errors_only", "include_correct"),
                    help="what the type probes treat as the negative class. "
                         "errors_only matches the hypothesis but makes E3's "
                         "off-diagonal algebraically forced; include_correct "
                         "makes cross-transfer informative. Run both.")
    ap.add_argument("--l2", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=pathlib.Path)
    args = ap.parse_args()

    cfg = ProbeConfig(
        layers=tuple(args.layers) if args.layers else None,
        position=args.position, l2=args.l2, seed=args.seed,
        type_probe_rest=args.rest,
    )

    report: dict = {
        "config": {
            "position": cfg.position, "l2": cfg.l2, "seed": cfg.seed,
            "type_probe_rest": cfg.type_probe_rest,
            "layers": list(cfg.layers) if cfg.layers else "all",
            "activations": str(args.activations),
            "predictions": str(args.predictions),
            "annotations": str(args.annotations) if args.annotations else None,
        },
        "probes": {},
    }

    # --- E1: binary ------------------------------------------------------
    binary = build_dataset("binary", args.activations, args.predictions, config=cfg)
    report["probes"]["binary"], _ = _run_one(binary, cfg.l2)
    _print(report["probes"]["binary"])

    # --- E1: the two type probes, and E3 --------------------------------
    if args.annotations:
        sweeps = {}
        for task in TYPE_TASKS:
            ds = build_dataset(task, args.activations, args.predictions,
                               args.annotations, config=cfg)
            report["probes"][task], sweeps[task] = _run_one(ds, cfg.l2)
            _print(report["probes"][task])

        # E3 needs one layer and one population; use the structural probe's
        # selected layer, since both type probes share the same items.
        ds = build_dataset("structural", args.activations, args.predictions,
                           args.annotations, config=cfg)
        layer = report["probes"]["structural"]["selected_layer"]
        # Use the true class per item. Deriving it from the binary y would
        # relabel every correct item as "fabrication" and collapse the three
        # groups back into two, which forces the off-diagonal even in
        # include_correct mode.
        transfer = cross_transfer(
            ds.X["train"], ds.classes["train"],
            ds.X["test"], ds.classes["test"],
            layer=layer, classes=list(TYPE_TASKS), l2=cfg.l2)
        report["cross_transfer"] = {
            "layer": layer,
            "cells": {f"{a}->{b}": v for (a, b), v in transfer.items()},
        }
        print("\nE3 cross-transfer at layer", layer)
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
    best = sorted(((float(v), k) for k, v in r["auroc_by_layer_val"].items()),
                  reverse=True)[:5]
    print("  top validation layers: " +
          ", ".join(f"L{k}={v:.3f}" for v, k in best))
    t = r["test"]
    ci = t.get("ci95")
    ci_s = f"  95% CI [{ci[0]:.3f}, {ci[1]:.3f}]" if ci else ""
    print(f"  TEST at layer {t['layer']}: AUROC {t['auroc']:.3f}{ci_s}")


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
