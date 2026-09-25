"""Run the E4 cross-intervention experiment: every repair on every labelled error.

Inputs follow inf.md's contract: the predictions file for one model and split,
and a labels file keyed by the same `item_id`. Only incorrect answers labelled
`structural` or `fabrication` are used; `ambiguous` items are dropped, as the
rubric says. Run on Ada inside the pinned environment (inf.md Section 2):

    python -m src.intervene.run_matrix \\
        --predictions results/predictions/qwen2_5_vl_7b_test.jsonl \\
        --labels annotations \\
        --image-root ~/data/ChartQA \\
        --out results/e4/qwen2_5_vl_7b_test.outcomes.jsonl

`--labels` takes either the `annotations/` folder (cross.md 3.2), whose
per-person files are merged first, or a single labels file keyed by `item_id`.
In the folder case, items the annotators disagree on and nobody has adjudicated
are left out and counted, rather than settled by picking one person's label.

For synthetic charts, add `--figures <synth dir>/manifest.jsonl` with
`--image-root <synth dir>`: that supplies each chart's bar geometry, which is
what makes `I_crop` possible.

Output is one row per (item, repair), appended and fsynced as it goes, so a
killed job resumes where it stopped. Field names follow cross.md 7. On Ada, run
it through `scripts/intervene.sbatch`. `--summarize` prints the matrix and the
pre-registered test (results/e4-preregistration.md) from that file without
touching the GPU.

Scoring uses the project's relaxed accuracy, and answers are only
whitespace-stripped, exactly like the original run (inf.md 4.1). Anything else
would make a repaired answer and an original answer incomparable.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
from typing import Callable, Iterable

from PIL import Image

from src.eval.relaxed_accuracy import is_correct
from src.extract import qwen
from src.intervene.matrix import (
    Outcome, argmax_flip_stability, cell_counts, did_with_ci, recovery_matrix,
)
from src.intervene.interventions import (
    BASELINE, CROP, REASK, REQUERIED, UPSAMPLE, VERIFY, Figure, Query,
    build_query, parse_verified,
)
from src.label.multi_rater import consensus, load_raters

TYPES = ("structural", "fabrication")
Responder = Callable[[Query], str]


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------

def read_jsonl(path: str | pathlib.Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_labels(path: str | pathlib.Path) -> tuple[list[dict], int]:
    """Labels keyed by item_id, from a file or from the annotations folder.

    Returns (labels, unresolved). For the folder, every human rater file is
    merged through the consensus step. LLM rater files are excluded, since E4's
    labels are the human ground truth, and `adjudicated.jsonl` settles
    disagreements the way it does everywhere else.
    """
    path = pathlib.Path(path)
    if not path.is_dir():
        return read_jsonl(path), 0
    ratings, _ = load_raters(path)
    humans = [r for r in ratings
              if r != "adjudicated" and "claude" not in r and "llm" not in r]
    labels, unresolved = consensus(ratings, humans)
    return labels, len(unresolved)


def select_items(predictions: Iterable[dict], labels: Iterable[dict]) -> tuple[list[dict], dict]:
    """Incorrect answers with a structural or fabrication label, plus the counts."""
    label_of = {r["item_id"]: r["label"] for r in labels}   # last record wins
    wrong = [r for r in predictions if not r["correct"]]
    items = [{**r, "failure_type": label_of[r["item_id"]]}
             for r in wrong if label_of.get(r["item_id"]) in TYPES]
    counts = {
        "incorrect": len(wrong),
        "used": len(items),
        "ambiguous": sum(label_of.get(r["item_id"]) == "ambiguous" for r in wrong),
        "unlabelled": sum(r["item_id"] not in label_of for r in wrong),
        **{t: sum(i["failure_type"] == t for i in items) for t in TYPES},
    }
    return items, counts


def figure_loader(image_root: str | pathlib.Path,
                  figures: str | pathlib.Path | None) -> Callable[[dict], Figure]:
    """Map a prediction row to its image, plus bar geometry when a manifest has it."""
    root = pathlib.Path(image_root).expanduser()
    geometry = {f["figure_id"]: f for f in read_jsonl(figures)} if figures else {}

    def load(row: dict) -> Figure:
        g = geometry.get(row["figure_id"])
        if g is None:                                   # ChartQA layout
            return Figure(Image.open(root / row["split"] / "png" / row["figure_id"]))
        return Figure(Image.open(root / g["image_path"]),
                      categories=tuple(g.get("categories", ())),
                      bar_boxes=tuple(tuple(b) for b in g.get("bar_boxes", ())),
                      plot_box=tuple(g["plot_box"]) if g.get("plot_box") else None)
    return load


# ---------------------------------------------------------------------------
# The model
# ---------------------------------------------------------------------------

class QwenResponder:
    """The model, loaded and queried through `src.extract.qwen` (cross.md 3.3)."""

    def __init__(self):
        self.model, self.processor = qwen.load()

    def __call__(self, q: Query) -> str:
        return qwen.generate(self.model, self.processor, q.image, q.prompt,
                             sample=q.sample, max_new_tokens=q.max_new_tokens)


# ---------------------------------------------------------------------------
# Running
# ---------------------------------------------------------------------------

def _append(row: dict, out: pathlib.Path) -> None:
    with open(out, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def run(items: list[dict], load_figure: Callable[[dict], Figure], responder: Responder,
        out: str | pathlib.Path, interventions: Iterable[str] = REQUERIED,
        log: Callable[[str], None] = print) -> None:
    """Apply every repair to every item, skipping (item, repair) pairs already on disk."""
    out = pathlib.Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = {(r["item_id"], r["intervention"]) for r in read_jsonl(out)} if out.is_file() else set()
    interventions = list(interventions)

    for n, it in enumerate(items, 1):
        base = {"item_id": it["item_id"], "figure_id": it["figure_id"],
                "failure_type": it["failure_type"], "gold": it["gold"]}
        if (it["item_id"], BASELINE) not in done:
            # I_0 is the original answer, wrong by selection; recorded so the
            # matrix has its baseline cell without a model call
            _append({**base, "intervention": BASELINE, "applicable": True,
                     "prediction": it["prediction"], "recovered": False}, out)

        todo = [iv for iv in interventions if (it["item_id"], iv) not in done]
        if not todo:
            continue
        fig = load_figure(it)
        for iv in todo:
            q = build_query(iv, it["question"], fig)
            if q is None:
                _append({**base, "intervention": iv, "applicable": False}, out)
                continue
            raw = responder(q)
            pred, parsed = parse_verified(raw) if iv == VERIFY else (raw.strip(), True)
            _append({**base, "intervention": iv, "applicable": True, "raw": raw,
                     "prediction": pred, "parsed": parsed, "note": q.note,
                     "recovered": bool(parsed and is_correct(it["gold"], pred))}, out)
        log(f"[{n}/{len(items)}] {it['item_id']} {it['failure_type']}")


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def summarize(out: str | pathlib.Path, n_boot: int = 2000) -> str:
    rows = read_jsonl(out)
    outcomes = [Outcome(r["figure_id"], r["failure_type"], r["intervention"], r["recovered"])
                for r in rows if r.get("applicable")]
    skipped = sum(not r.get("applicable") for r in rows)
    unparsed = sum(r.get("parsed") is False for r in rows)
    matrix, counts = recovery_matrix(outcomes), cell_counts(outcomes)
    ivs = [BASELINE] + [iv for iv in REQUERIED if any(k[1] == iv for k in counts)]

    lines = ["| Failure type | " + " | ".join(ivs) + " |",
             "| --- |" + " --- |" * len(ivs)]
    for t in TYPES:
        cells = [f"{matrix[(t, iv)]:.3f} (n={counts[(t, iv)]})" if (t, iv) in counts else "n/a"
                 for iv in ivs]
        lines.append(f"| {t} | " + " | ".join(cells) + " |")

    def did_line(iv, baseline, label):
        if not all((t, x) in counts for t in TYPES for x in (iv, baseline)):
            return f"- {label}: not computable, a cell is empty"
        r = did_with_ci(outcomes, iv, baseline=baseline, n_boot=n_boot)
        lo, hi = r["ci95"]
        return (f"- {label}: delta = {r['delta']:+.3f}, 95% CI [{lo:+.3f}, {hi:+.3f}]"
                f"{', excludes 0' if r['excludes_zero'] else ''}")

    lines += ["", "**Pre-registered primary test**",
              did_line(CROP, BASELINE, "I_crop vs I_0"),
              "", "**Secondary**",
              did_line(VERIFY, BASELINE, "I_verify vs I_0"),
              did_line(CROP, UPSAMPLE, "I_crop vs I_upsample (the crop's gain beyond pixels)"),
              did_line(CROP, REASK, "I_crop vs I_reask (beyond simply asking twice)"),
              "", f"Argmax flip stability: {argmax_flip_stability(outcomes, n_boot=n_boot):.2f} "
              "(below about 0.9 the flip is unsupported)",
              f"Not applicable (no geometry for I_crop): {skipped} rows. "
              f"Unparseable I_verify responses, scored as not recovered: {unparsed}."]
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description="E4 cross-intervention run (see module docstring)")
    ap.add_argument("--out", type=pathlib.Path, required=True)
    ap.add_argument("--summarize", action="store_true", help="print results from --out and exit")
    ap.add_argument("--predictions", type=pathlib.Path)
    ap.add_argument("--labels", type=pathlib.Path,
                    help="the annotations/ folder, or one labels file")
    ap.add_argument("--image-root", type=pathlib.Path)
    ap.add_argument("--figures", type=pathlib.Path, help="synthetic manifest with bar geometry")
    ap.add_argument("--interventions", nargs="+", default=list(REQUERIED), choices=REQUERIED)
    ap.add_argument("--limit", type=int, help="first N items only, for a smoke run")
    args = ap.parse_args()

    if args.summarize:
        print(summarize(args.out))
        return
    if not (args.predictions and args.labels and args.image_root):
        ap.error("--predictions, --labels and --image-root are required to run")

    labels, unresolved = load_labels(args.labels)
    items, counts = select_items(read_jsonl(args.predictions), labels)
    print("items:", json.dumps({**counts, "unresolved_disagreements": unresolved}))
    if counts["unlabelled"]:
        print(f"warning: {counts['unlabelled']} incorrect answers have no label and are skipped")
    run(items[:args.limit] if args.limit else items,
        figure_loader(args.image_root, args.figures), QwenResponder(), args.out,
        args.interventions)
    print(summarize(args.out))


if __name__ == "__main__":
    main()
