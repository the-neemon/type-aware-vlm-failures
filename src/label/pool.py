"""Build the annotation pool and split it between annotators (TASKS P2.4, P2.6).

Two steps, both deterministic (seed 42, the project convention):

    build  : inference manifests -> pool.jsonl (blind) + pool_key.jsonl (private)
    assign : pool.jsonl -> tasks/<annotator>.jsonl, with a designed overlap

Blinding is enforced here rather than trusted to the UI. Two leaks it closes:

1. Item IDs are opaque hashes of (model, figure, question). An ID built from
   `figure::question` alone would collide whenever both models get the same
   question wrong, silently merging two different answers into one item.
2. The pool is shuffled, so an annotator cannot infer the model from position
   (every Qwen error first, then every LLaVA error).

The model for each item lives only in pool_key.jsonl, which annotators never
need to open. Joining labels back to models happens after annotation.

The overlap is what makes kappa computable at all. Each overlap item goes to
exactly two annotators, cycling through every pair, so all four people are
checked against each other rather than one pair carrying the whole agreement
estimate. Usage:

    python -m src.label.pool build --split test \\
        --source qwen=results/qwen_chartqa_test.jsonl \\
        --source llava=results/llava_chartqa_test.jsonl \\
        --out annotations/tasks
    python -m src.label.pool assign --pool annotations/tasks/pool.jsonl \\
        --annotators naman yash shrish sanjith --overlap 200 --out annotations/tasks
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import pathlib
import random
import re
from typing import Sequence

from src.eval.manifest import read_manifest

SEED = 42


def sanitize(name: str) -> str:
    """Annotator name -> file stem. Shared by assign and the app so they agree."""
    stem = re.sub(r"[^a-z0-9_-]+", "_", name.strip().lower()).strip("_")
    if not stem:
        raise ValueError(f"annotator name {name!r} has no usable characters")
    return stem


def item_id(model: str, figure_id: str, question: str) -> str:
    raw = f"{model}\x1f{figure_id}\x1f{question}".encode("utf-8")
    return hashlib.sha1(raw).hexdigest()[:12]


def build_pool(sources: Sequence[tuple[str, str | pathlib.Path]],
               split: str) -> tuple[list[dict], list[dict]]:
    """Keep only incorrect answers, mint opaque IDs, shuffle. Returns (pool, key)."""
    pool, key = [], []
    for model, path in sources:
        for row in read_manifest(path):
            if row.correct:
                continue
            iid = item_id(model, row.figure_id, row.question)
            pool.append({
                "item_id": iid,
                "image_path": f"{split}/png/{row.figure_id}",
                "question": row.question,
                "gold_answer": row.gold,
                "model_answer": row.prediction,
            })
            key.append({"item_id": iid, "model": model,
                        "figure_id": row.figure_id, "split": split})

    ids = [p["item_id"] for p in pool]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate item IDs: a manifest repeats a (figure, question)")

    random.Random(SEED).shuffle(pool)
    return pool, key


def assign(pool: Sequence[dict], annotators: Sequence[str],
           overlap: int) -> dict[str, list[dict]]:
    """Split the pool: `overlap` items to two people each, the rest to one person.

    With 1000 items, four annotators and an overlap of 200, everyone gets 300:
    200 of their own plus 100 shared with someone else.
    """
    names = [sanitize(a) for a in annotators]
    if len(names) < 2 or len(set(names)) != len(names):
        raise ValueError("need at least two distinct annotators")
    if not 0 <= overlap <= len(pool):
        raise ValueError(f"overlap {overlap} outside 0..{len(pool)}")

    rng = random.Random(SEED)
    items = list(pool)
    rng.shuffle(items)

    tasks: dict[str, list[dict]] = {n: [] for n in names}
    pairs = list(itertools.combinations(names, 2))
    for i, it in enumerate(items[:overlap]):
        for n in pairs[i % len(pairs)]:
            tasks[n].append(it)
    # Solo items go to whoever has fewest so far, which absorbs the uneven split
    # of the overlap across pairs (200 does not divide by 6).
    for it in items[overlap:]:
        tasks[min(names, key=lambda n: len(tasks[n]))].append(it)

    # Interleave shared and solo items so the kappa set is spread across every
    # sitting instead of sitting at the front of the queue, where it would only
    # measure how people labelled while fresh.
    for n in names:
        rng.shuffle(tasks[n])
    return tasks


def _write(rows: Sequence[dict], path: pathlib.Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _read(path: pathlib.Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build", help="inference manifests -> blind pool + private key")
    b.add_argument("--source", action="append", required=True,
                   metavar="MODEL=MANIFEST", help="repeat once per model")
    b.add_argument("--split", required=True, choices=["train", "val", "test"])
    b.add_argument("--out", type=pathlib.Path, default=pathlib.Path("annotations/tasks"))

    a = sub.add_parser("assign", help="pool -> one task file per annotator")
    a.add_argument("--pool", type=pathlib.Path, required=True)
    a.add_argument("--annotators", nargs="+", required=True)
    a.add_argument("--overlap", type=int, default=200)
    a.add_argument("--out", type=pathlib.Path, default=pathlib.Path("annotations/tasks"))

    args = ap.parse_args()
    if args.cmd == "build":
        sources = [tuple(s.split("=", 1)) for s in args.source]
        pool, key = build_pool(sources, args.split)
        _write(pool, args.out / "pool.jsonl")
        _write(key, args.out / "pool_key.jsonl")
        print(f"pool: {len(pool)} incorrect answers -> {args.out / 'pool.jsonl'}")
    else:
        tasks = assign(_read(args.pool), args.annotators, args.overlap)
        for name, rows in tasks.items():
            _write(rows, args.out / f"{name}.jsonl")
            print(f"{name}: {len(rows)} items")


if __name__ == "__main__":
    main()
