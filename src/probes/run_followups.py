"""Follow-up checks on the main probes (todos.md, 1 Oct 2026). Two subcommands.

binary
    The E1 binary probe (is the answer wrong?) on query_last, re-run four ways in
    one pass over the cache:

    seeds        the whole procedure (figure split, layer and L2 chosen on
                 validation, test scored once) repeated over several split
                 seeds. Seed 42 is the reported split; the others say how much
                 the number moves with the split alone.
    residualised the four surface features (source, question length, image
                 tokens, numeric gold) regressed out of the activations before
                 fitting (E2, Sahoo et al.), the projection fitted on train rows
                 only. Per seed, like the plain probe.
    calibration  for the seed-42 probe: expected calibration error and Brier
                 score of its probabilities on test, raw and after Platt scaling
                 fitted on validation. The planned controller (E5) thresholds
                 probe scores, so it needs to know whether they mean what they say.
    val_human    (optional, --val-human) a second test set from charts the main
                 cache never saw: 960 human-written validation questions, Qwen
                 only. (a) The seed-42 probe, unchanged, scored on all of
                 val_human. (b) val_human added to the training rows, layer and L2
                 again chosen on validation, scored on the usual test split.

    python -m src.probes.run_followups binary --activations <test.npz> \\
        --predictions <model>_test.jsonl [--drop-not-an-error <labels.jsonl>] \\
        [--val-human <val_human.npz> <val_human.jsonl>] --out <report.json>

grid
    The synthetic misread baseline with the value's position relative to the
    gridlines, declared before running (todos.md): an earlier diagnostic found
    that distance to the nearest multiple of 10 predicts LLaVA's misreads at
    0.852, above its probe, but it was chosen after seeing the data. Two numeric
    features are added to the chart-metadata baseline of run_synth_types:

        grid_offset   distance from the asked bar's value to the nearest gridline,
                      in units of the gridline step (0 on a line, 0.5 midway)
        ten_offset    distance to the nearest multiple of 10, divided by 10

    Added after the first run, so post hoc and reported as such: the same two
    distances divided by the value itself (rel_grid_offset, rel_ten_offset). The
    earlier 0.852 used this relative form, and it is the one that matches the
    scorer, whose 5% tolerance is relative: rounding 34 to 30 is wrong, 134 to
    130 is not.

    Only the structural probe (S vs C, misread vs correct read) is run: missing-bar
    items have no value, so the features are undefined for them. Reported next to
    the activation probe's S vs C from results/probes/synth_types_<model>.json,
    plus each feature's AUROC on its own with no fitting.

    python -m src.probes.run_followups grid --pilot pilot1 <npz> <preds> <manifest> \\
        --pilot pilot2 <npz> <preds> <manifest> --probe-report <synth_types.json> \\
        --out <report.json>
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

from src.common.linear import (auroc, cluster_bootstrap_ci, fit_logistic, predict_scores,
                               residualise)
from src.probes.dataset import (assign_figure_splits, cached_layers, load_activations,
                                load_predictions, resolve_labels)
from src.probes.run_synth_types import (DEFAULT_L2, cross_validate, load_items, metadata,
                                        select_and_assemble)
from src.synth.bar_charts import TICK_COUNTS

POSITION = "query_last"
SEEDS = (42, 1, 2, 3, 4)
N_BINS = 10


# ---------------------------------------------------------------------------
# binary
# ---------------------------------------------------------------------------

def surface_row(r):
    gold = str(r["gold"])
    return [r["source"] == "human", len(r["question"]), r.get("n_vision_tokens", 0),
            gold.replace(".", "", 1).replace("-", "", 1).isdigit()]


def load_rows(npz, predictions, drop=()):
    """Items of one cache, joined on item_id: ids, figures, y (wrong), surface, source."""
    _, ids, figs = load_activations(npz, POSITION, (cached_layers(npz)[0],))
    preds = load_predictions(predictions)
    keep = [i for i, iid in enumerate(ids) if iid in preds and iid not in drop]
    rows = [preds[ids[i]] for i in keep]
    return {"npz": str(npz), "rows": np.array(keep),
            "figs": np.array([figs[i] if figs else preds[ids[i]]["figure_id"] for i in keep]),
            "y": np.array([not r["correct"] for r in rows], dtype=float),
            "S": np.array([surface_row(r) for r in rows], dtype=np.float64),
            "source": np.array([r["source"] for r in rows])}


def layer_matrix(data, layer):
    by_layer, _, _ = load_activations(data["npz"], POSITION, (layer,))
    return by_layer[layer][data["rows"]].astype(np.float64)


def _standardise(X, fit_rows):
    mu, sd = X[fit_rows].mean(0), X[fit_rows].std(0)
    return (X - mu) / np.where(sd < 1e-6, 1.0, sd)


def _binary_cell(args):
    """One layer: every seed x variant x L2, fitted on train; val and test scores."""
    layer, main, extra, seeds, l2_grid = args
    A = layer_matrix(main, layer)
    out = {}
    for seed in seeds:
        split = main["split"][seed]
        tr, va, te = split == "train", split == "val", split == "test"
        variants = {"plain": _standardise(A, tr)}
        variants["residualised"] = residualise(variants["plain"], _standardise(main["S"], tr),
                                               fit_on=tr)
        for name, X in variants.items():
            for l2 in l2_grid:
                w = fit_logistic(X[tr], main["y"][tr], l2)
                s = predict_scores(X, w)
                out[(seed, name, l2)] = {"val": auroc(main["y"][va], s[va]),
                                         "s_val": s[va], "s_test": s[te]}
    if extra is not None:
        # val_human, seed 42 only: (a) the plain probe scored on it; (b) added to train
        split = main["split"][42]
        tr, va, te = split == "train", split == "val", split == "test"
        B = layer_matrix(extra, layer)
        mu, sd = A[tr].mean(0), A[tr].std(0)
        sd = np.where(sd < 1e-6, 1.0, sd)
        for l2 in l2_grid:
            w = fit_logistic((A[tr] - mu) / sd, main["y"][tr], l2)
            out[(42, "plain", l2)]["s_extra"] = predict_scores((B - mu) / sd, w)
        both = np.vstack([A[tr], B])
        mu2, sd2 = both.mean(0), both.std(0)
        sd2 = np.where(sd2 < 1e-6, 1.0, sd2)
        y_both = np.concatenate([main["y"][tr], extra["y"]])
        for l2 in l2_grid:
            w = fit_logistic((both - mu2) / sd2, y_both, l2)
            s = predict_scores((A - mu2) / sd2, w)
            out[(42, "plus_val_human", l2)] = {"val": auroc(main["y"][va], s[va]),
                                               "s_val": s[va], "s_test": s[te]}
    return layer, out


def _ci(y, s, figs):
    return list(cluster_bootstrap_ci(lambda i: auroc(y[i], s[i]), figs))


def calibration(y, p):
    """Expected calibration error (10 equal-width bins) and Brier score."""
    bins = np.minimum((p * N_BINS).astype(int), N_BINS - 1)
    ece = sum(abs(p[bins == b].mean() - y[bins == b].mean()) * (bins == b).mean()
              for b in range(N_BINS) if (bins == b).any())
    return {"ece": float(ece), "brier": float(np.mean((p - y) ** 2)),
            "mean_p": float(p.mean()), "base_rate": float(y.mean())}


def platt(s_val, y_val, s_test):
    """Fit sigmoid(a*s + b) on validation, apply to test."""
    w = fit_logistic(s_val[:, None], y_val, 1e-6)
    return 1 / (1 + np.exp(-predict_scores(s_test[:, None], w)))


def run_binary(args):
    drop = set()
    if args.drop_not_an_error:
        labels, _ = resolve_labels(args.drop_not_an_error)
        drop = {k for k, v in labels.items() if v == "not_an_error"}
    main = load_rows(args.activations, args.predictions, drop)
    main["split"] = {seed: np.array([assign_figure_splits(main["figs"], seed=seed)[f]
                                     for f in main["figs"]]) for seed in args.seeds}
    extra = load_rows(*args.val_human) if args.val_human else None
    if extra is not None:
        shared = set(extra["figs"]) & set(main["figs"])
        assert not shared, f"val_human shares {len(shared)} charts with the main cache"
    layers = args.layers or cached_layers(args.activations)
    print(f"items {len(main['y'])} (errors {int(main['y'].sum())}, dropped {len(drop)}); "
          f"layers {len(layers)}; seeds {args.seeds}"
          + (f"; val_human {len(extra['y'])} (errors {int(extra['y'].sum())})" if extra else ""),
          flush=True)

    cells = {}
    t0 = time.time()
    with Pool(args.jobs) as pool:
        jobs = [(L, main, extra, tuple(args.seeds), tuple(args.l2)) for L in layers]
        for n, (L, out) in enumerate(pool.imap_unordered(_binary_cell, jobs), 1):
            for key, v in out.items():
                cells[(L,) + key] = v
            print(f"  layer {L} done ({n}/{len(layers)}) [{time.time() - t0:.0f}s]", flush=True)

    def select(seed, variant):
        """(layer, l2) with the best validation AUROC. Test is never consulted."""
        keys = [k for k in cells if k[1] == seed and k[2] == variant]
        return max(keys, key=lambda k: np.nan_to_num(cells[k]["val"], nan=-1.0))

    report = {"activations": str(args.activations), "position": POSITION,
              "n_items": int(len(main["y"])), "n_errors": int(main["y"].sum()),
              "dropped_not_an_error": len(drop), "seeds": {}}
    for seed in args.seeds:
        split = main["split"][seed]
        te, va = split == "test", split == "val"
        y, figs = main["y"][te], main["figs"][te]
        report["seeds"][seed] = {}
        for variant in ("plain", "residualised"):
            k = select(seed, variant)
            s = cells[k]["s_test"]
            entry = {"layer": k[0], "l2": k[3], "val_auroc": cells[k]["val"],
                     "test_auroc": auroc(y, s), "n_test": int(te.sum()),
                     "test_errors": int(y.sum())}
            if seed == 42:
                entry["ci95"] = _ci(y, s, figs)
                src = main["source"][te]
                entry["within_source"] = {g: auroc(y[src == g], s[src == g])
                                          for g in ("human", "augmented")}
            report["seeds"][seed][variant] = entry
            if seed == 42 and variant == "plain":
                p_raw = 1 / (1 + np.exp(-s))
                p_cal = platt(cells[k]["s_val"], main["y"][va], s)
                report["calibration"] = {"raw": calibration(y, p_raw),
                                         "platt_on_val": calibration(y, p_cal)}
                if extra is not None:
                    se = cells[k]["s_extra"]
                    report["val_human_as_test"] = {
                        "layer": k[0], "l2": k[3], "n": int(len(extra["y"])),
                        "errors": int(extra["y"].sum()), "auroc": auroc(extra["y"], se),
                        "ci95": _ci(extra["y"], se, extra["figs"])}
    if extra is not None:
        k = select(42, "plus_val_human")
        te = main["split"][42] == "test"
        y, s = main["y"][te], cells[k]["s_test"]
        report["val_human_as_training"] = {"layer": k[0], "l2": k[3],
                                           "val_auroc": cells[k]["val"],
                                           "test_auroc": auroc(y, s),
                                           "ci95": _ci(y, s, main["figs"][te])}
    plain = [report["seeds"][s]["plain"]["test_auroc"] for s in args.seeds]
    resid = [report["seeds"][s]["residualised"]["test_auroc"] for s in args.seeds]
    report["over_seeds"] = {v: {"mean": float(np.mean(a)), "sd": float(np.std(a, ddof=1)) if len(a) > 1 else 0.0,
                                "min": float(np.min(a)), "max": float(np.max(a))}
                            for v, a in (("plain", plain), ("residualised", resid))}
    _write(report, args.out)

    print(f"\n{'seed':>5} {'plain':>18} {'residualised':>18}")
    for seed in args.seeds:
        e = report["seeds"][seed]
        print(f"{seed:>5} " + " ".join(
            f"{e[v]['test_auroc']:>8.3f} (L{e[v]['layer']:>2} {e[v]['l2']:g})".rjust(18)
            for v in ("plain", "residualised")))
    for v, m in report["over_seeds"].items():
        print(f"  {v}: mean {m['mean']:.3f} sd {m['sd']:.3f} range [{m['min']:.3f}, {m['max']:.3f}]")
    print("calibration (seed 42, test):", json.dumps(report["calibration"]))
    for k in ("val_human_as_test", "val_human_as_training"):
        if k in report:
            print(f"{k}: {report[k]}")


# ---------------------------------------------------------------------------
# grid
# ---------------------------------------------------------------------------

GRID_FEATURES = ("grid_offset", "ten_offset", "rel_grid_offset", "rel_ten_offset")
DECLARED = 2        # the first two were declared before running; the rest are post hoc


def grid_features(items, manifests):
    """GRID_FEATURES of the asked bar's value; 0 for missing-bar items."""
    figures = {}
    for path in manifests:
        for line in pathlib.Path(path).read_text(encoding="utf-8").splitlines():
            if line.strip():
                f = json.loads(line)
                figures[f["figure_id"]] = f
    out = np.zeros((len(items), len(GRID_FEATURES)), dtype=np.float32)
    for i, it in enumerate(items):
        if it["asks_about"] is None:
            continue
        f = figures[it["figure_id"]]
        if it["asks_about"] not in f["categories"]:
            continue                                      # a missing bar
        v = f["values"][f["categories"].index(it["asks_about"])]
        step = (f["axis_max"] - f["axis_min"]) / (TICK_COUNTS[f["tick_density"]] - 1)
        r = (v - f["axis_min"]) / step
        to_grid = abs(r - round(r)) * step
        to_ten = abs(v - 10 * round(v / 10))
        out[i] = [to_grid / step, to_ten / 10, to_grid / max(v, 1), to_ten / max(v, 1)]
    return out


def run_grid(args):
    items = load_items(args.pilot)
    preds = {}
    for _, _, path, _ in args.pilot:
        preds.update(load_predictions(path))
    for it in items:
        it["asks_about"] = preds[it["item_id"]].get("asks_about")
    cls = np.array([it["cls"] for it in items])
    folds = np.array([it["fold"] for it in items])
    figs = np.array([it["figure_id"] for it in items])
    G = grid_features(items, [p[3] for p in args.pilot])
    sc = np.isin(cls, ("S", "C"))
    y = (cls[sc] == "S").astype(float)
    print("items:", {c: int((cls == c).sum()) for c in "CSFZ"}, flush=True)

    report = {"declared": "1 Oct 2026, before running; see module docstring",
              "n_misread": int(y.sum()), "n_correct": int((1 - y).sum()),
              "post_hoc_features": list(GRID_FEATURES[DECLARED:]),
              "single_feature_auroc_no_fitting": {
                  f: auroc(y, G[sc, j]) for j, f in enumerate(GRID_FEATURES)},
              "cross_validated": {}}
    M = metadata(items)
    for name, X in (("metadata", M),
                    ("metadata_plus_grid", np.hstack([M, G[:, :DECLARED]])),
                    ("grid_only", G[:, :DECLARED]),
                    ("post_hoc_metadata_plus_all_grid", np.hstack([M, G])),
                    ("post_hoc_all_grid_only", G)):
        res = cross_validate(X, cls, folds, "structural", tuple(args.l2))
        scores, chosen = select_and_assemble({name: res}, len(items), folds)
        report["cross_validated"][name] = {
            "S_vs_C": auroc(y, scores[sc]),
            "ci95": list(cluster_bootstrap_ci(lambda i: auroc(y[i], scores[sc][i]), figs[sc])),
            "chosen_per_fold": chosen}
    if args.probe_report:
        probe = json.loads(pathlib.Path(args.probe_report).read_text())
        report["activation_probe_S_vs_C"] = probe["probes"]["structural"]["matrix"]["S_vs_C"]
    _write(report, args.out)
    print(json.dumps({k: v for k, v in report.items() if k != "cross_validated"}, indent=1))
    for name, e in report["cross_validated"].items():
        print(f"  {name:20s} S vs C {e['S_vs_C']:.3f} [{e['ci95'][0]:.2f}, {e['ci95'][1]:.2f}]")


# ---------------------------------------------------------------------------

def _write(report, out):
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, default=float) + "\n")
    print(f"wrote {out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("binary")
    b.add_argument("--activations", required=True, type=pathlib.Path)
    b.add_argument("--predictions", required=True, type=pathlib.Path)
    b.add_argument("--drop-not-an-error", type=pathlib.Path, default=None,
                   help="label file; its not_an_error items are left out")
    b.add_argument("--val-human", nargs=2, type=pathlib.Path, default=None,
                   metavar=("NPZ", "PREDICTIONS"))
    b.add_argument("--seeds", nargs="*", type=int, default=list(SEEDS))
    g = sub.add_parser("grid")
    g.add_argument("--pilot", nargs=4, action="append", required=True,
                   metavar=("NAME", "NPZ", "PREDICTIONS", "MANIFEST"))
    g.add_argument("--probe-report", type=pathlib.Path, default=None)
    for s in (b, g):
        s.add_argument("--layers", nargs="*", type=int, default=None)
        s.add_argument("--l2", nargs="*", type=float, default=list(DEFAULT_L2))
        s.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) - 2))
        s.add_argument("--out", required=True, type=pathlib.Path)
    args = ap.parse_args()
    if args.cmd == "binary" and 42 not in args.seeds:
        ap.error("--seeds must include 42, the reported split")
    (run_binary if args.cmd == "binary" else run_grid)(args)


if __name__ == "__main__":
    main()
