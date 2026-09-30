"""Batches for the LLM annotator, and a checked append of its labels (docs/taxonomy.md Part A).

    batches  predictions -> batch_NNN.jsonl of blind items, in the annotation
             pool's seed-42 order (src/label/pool.py), N items each.
    append   one labelled batch -> the model's label file, after checking every
             label and reason against the rubric and every item_id against the
             batch it came from.

    python -m src.label.llm_label batches \\
        --predictions results/predictions/llava_next_mistral_7b_test.jsonl \\
        --image-root ~/data/ChartQA --out <dir> --size 25
    python -m src.label.llm_label append --batch <dir>/batch_001.jsonl \\
        --labels <dir>/batch_001.labels.jsonl \\
        --out results/labels/llava_next_mistral_7b_test.claude.jsonl \\
        --annotator "claude-opus-5-5 (...)"

Blind means what the Qwen labelling saw: chart, question, gold, model answer. No
model field, no data tables, no other labels.
"""

from __future__ import annotations

import argparse
import datetime
import json
import pathlib

from src.label.pool import build_pool

LABELS = ("structural", "fabrication", "computation", "not_an_error", "ambiguous")
REASONS = {
    "not_an_error": ("format_equivalent", "valid_reading", "gold_error"),
    "ambiguous": ("question_ambiguous", "gold_error", "unreadable", "other"),
}


def read_jsonl(path: pathlib.Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def make_batches(predictions: pathlib.Path, image_root: pathlib.Path, size: int) -> list[list[dict]]:
    """The model's errors in pool order, cut into batches of blind items."""
    pool, _ = build_pool([predictions])
    items = []
    for p in pool:
        image = image_root / p["image_path"]
        if not image.is_file():
            raise SystemExit(f"missing chart image {image}")
        items.append({"item_id": p["item_id"], "image": str(image), "question": p["question"],
                      "gold_answer": p["gold_answer"], "model_answer": p["model_answer"]})
    return [items[i:i + size] for i in range(0, len(items), size)]


def check(labels: list[dict], batch: list[dict]) -> list[str]:
    """Problems with one labelled batch; empty means it can be appended."""
    problems = []
    expected = [b["item_id"] for b in batch]
    got = [l.get("item_id") for l in labels]
    if sorted(got) != sorted(expected):
        problems.append(f"item_ids differ from the batch: {len(set(expected) - set(got))} missing, "
                        f"{len(set(got) - set(expected))} unexpected, {len(got) - len(set(got))} repeated")
    for l in labels:
        label, reason = l.get("label"), l.get("reason")
        if label not in LABELS:
            problems.append(f"{l.get('item_id')}: unknown label {label!r}")
        elif label in REASONS and reason not in REASONS[label]:
            problems.append(f"{l.get('item_id')}: {label} needs a reason in {REASONS[label]}, got {reason!r}")
        elif label not in REASONS and reason is not None:
            problems.append(f"{l.get('item_id')}: {label} takes no reason, got {reason!r}")
        if not str(l.get("rationale", "")).strip():
            problems.append(f"{l.get('item_id')}: empty rationale")
    return problems


def append(labels: list[dict], batch: list[dict], out: pathlib.Path, annotator: str) -> int:
    problems = check(labels, batch)
    if problems:
        raise SystemExit("batch not appended:\n  " + "\n  ".join(problems))
    done = {r["item_id"] for r in read_jsonl(out)} if out.is_file() else set()
    already = [l["item_id"] for l in labels if l["item_id"] in done]
    if already:
        raise SystemExit(f"{len(already)} items are already labelled in {out}, e.g. {already[0]}")
    now = datetime.datetime.now(datetime.timezone.utc).isoformat()
    order = {b["item_id"]: i for i, b in enumerate(batch)}
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "a", encoding="utf-8") as f:
        for l in sorted(labels, key=lambda l: order[l["item_id"]]):
            f.write(json.dumps({"item_id": l["item_id"], "label": l["label"], "reason": l.get("reason"),
                                "rationale": l["rationale"].strip(), "annotator": annotator,
                                "timestamp": now}, ensure_ascii=False) + "\n")
    return len(labels)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("batches")
    b.add_argument("--predictions", type=pathlib.Path, required=True)
    b.add_argument("--image-root", type=pathlib.Path, required=True)
    b.add_argument("--out", type=pathlib.Path, required=True)
    b.add_argument("--size", type=int, default=25)
    a = sub.add_parser("append")
    a.add_argument("--batch", type=pathlib.Path, required=True)
    a.add_argument("--labels", type=pathlib.Path, required=True)
    a.add_argument("--out", type=pathlib.Path, required=True)
    a.add_argument("--annotator", required=True)
    args = ap.parse_args()

    if args.cmd == "batches":
        batches = make_batches(args.predictions, args.image_root.expanduser(), args.size)
        args.out.mkdir(parents=True, exist_ok=True)
        for i, batch in enumerate(batches, 1):
            with open(args.out / f"batch_{i:03d}.jsonl", "w", encoding="utf-8") as f:
                f.writelines(json.dumps(it, ensure_ascii=False) + "\n" for it in batch)
        print(f"{sum(map(len, batches))} errors in {len(batches)} batches -> {args.out}")
    else:
        n = append(read_jsonl(args.labels), read_jsonl(args.batch), args.out, args.annotator)
        print(f"appended {n} labels -> {args.out}")


if __name__ == "__main__":
    main()
