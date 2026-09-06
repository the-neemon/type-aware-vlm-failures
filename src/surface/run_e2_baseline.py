"""Run the E2 surface baseline over a synthetic manifest (TASKS P5.4).

Reports per-class AUROC for the surface features alone, plus ablations that
isolate which feature is carrying the separation. Run:

    PYTHONPATH=. python3 -m src.surface.run_e2_baseline <synth_dir>

Until inference lands (P1.3, P1.4) there are no real model answers, so this
script stands answers in to exercise the pipeline end to end. Those numbers are
NOT results and the script says so loudly. Swap `_placeholder_answers` for the
real manifest predictions the moment they exist.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np

from src.surface.baseline import one_vs_rest_auroc
from src.surface.features import SurfaceItem, featurise

CLASSES = ["structural", "fabrication"]


def figure_text(fig: dict) -> tuple[str, ...]:
    """Everything a reader could see written on the figure.

    This is what `answer_in_figure` is checked against, so it must include every
    label and value the figure actually renders, and nothing it does not.
    """
    kind = fig["figure_type"]
    if kind == "bar_chart":
        seen = [str(c) for c in fig["categories"]] + [str(v) for v in fig["values"]]
    elif kind == "node_link":
        nodes = list(fig["path_nodes"]) + [n for e in fig["edges"] for n in e]
        seen = [str(n) for n in nodes]
    else:
        raise ValueError(
            f"unknown figure_type {kind!r}; add its text extraction here rather "
            f"than letting answer_in_figure silently see nothing")
    return tuple(dict.fromkeys(seen))


def _placeholder_answer(q: dict, fig: dict, rng) -> str:
    """Stand in for a model's wrong answer, of the type the item targets.

    NOT a model, and not a result. A crude stand-in so the pipeline can run
    before inference exists.
    """
    on_figure = [t for t in figure_text(fig)]
    gold = str(q["gold_answer"])

    if q["target_failure_type"] == "fabrication":
        if fig["figure_type"] == "bar_chart":
            return str(round(float(max(fig["values"])) + rng.integers(20, 60)))
        return "MISSING"

    others = [t for t in on_figure if t != gold]
    return others[rng.integers(len(others))] if others else gold


def load(synth_dir: pathlib.Path, rng):
    items, labels, splits = [], [], []
    for line in (synth_dir / "manifest.jsonl").read_text().splitlines():
        fig = json.loads(line)
        text = figure_text(fig)
        for q in fig["questions"]:
            items.append(SurfaceItem(
                answer=_placeholder_answer(q, fig, rng),
                figure_text=text,
                question_template=q["template"],
                figure_type=fig["figure_type"],
            ))
            labels.append(q["target_failure_type"])
            splits.append(fig["split"])
    return items, np.array(labels), np.array(splits)


def report(name, X, y, tr, te, names, drop=()):
    keep = [i for i, n in enumerate(names)
            if not any(n == d or n.startswith(d + "=") for d in drop)]
    Xk = X[:, keep]
    if Xk.shape[1] == 0:
        return
    scores = one_vs_rest_auroc(Xk[tr], y[tr], Xk[te], y[te], CLASSES)
    cells = "  ".join(f"{c}={scores[c]:.3f}" for c in CLASSES)
    print(f"  {name:<42} {cells}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("synth_dir", type=pathlib.Path, nargs="+",
                    help="one or more directories containing manifest.jsonl")
    args = ap.parse_args()

    rng = np.random.default_rng(42)
    items, ys, sp = [], [], []
    for d in args.synth_dir:
        i, l, s = load(d, rng)
        items += i; ys.append(l); sp.append(s)
    y, splits = np.concatenate(ys), np.concatenate(sp)
    X, names = featurise(items)
    tr, te = splits == "train", splits == "test"

    print("=" * 72)
    print("PLACEHOLDER ANSWERS. Not results. Replace with real predictions")
    print("from P1.3 / P1.4 before reporting any of this.")
    print("=" * 72)
    print(f"\n{len(items)} items, {tr.sum()} train / {te.sum()} test, "
          f"figure-level split from the manifest.")
    print(f"class balance: " + ", ".join(
        f"{c}={int((y == c).sum())}" for c in CLASSES) + "\n")

    print("Per-class AUROC, surface features only:")
    report("all surface features", X, y, tr, te, names)
    report("question_template alone", X, y, tr, te, names,
           drop=("answer_in_figure", "answer_length", "answer_type", "figure_type"))
    report("answer_in_figure alone", X, y, tr, te, names,
           drop=("answer_length", "answer_type", "template", "figure_type"))
    report("without template or answer_in_figure", X, y, tr, te, names,
           drop=("template", "answer_in_figure"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
