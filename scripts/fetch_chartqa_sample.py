"""Pull a handful of real ChartQA test items to local files.

Two consumers, both of which need real figures rather than synthetic stand-ins:
  - src/extract/smoke_qwen.py (P0.5), which needs exactly one.
  - scripts/verify_vision_tokens.py (P0.10), which needs at least five of
    differing pixel sizes, because the whole point is to catch a rounding bug
    in smart_resize that only shows up at particular aspect ratios.

This is NOT the ChartQA ingest. That is Shrish's P1.1 and it owns the manifest
schema, the splits and relaxed accuracy. This script exists so workstream A is
not blocked waiting on it, and it writes to node-local scratch, not the repo.

    python scripts/fetch_chartqa_sample.py --out /scratch/vlm-failures/chartqa_sample --n 5
"""

import argparse
import json
import pathlib

# ChartQA test split. First entry that loads wins; the mirrors carry the same
# figures under different column names, so the column probing below is not
# optional politeness, it is what makes the fallback work.
CANDIDATES = [
    ("HuggingFaceM4/ChartQA", "test"),
    ("ahmed-masry/ChartQA", "test"),
]

IMAGE_KEYS = ("image", "img", "figure")
QUESTION_KEYS = ("query", "question", "input")
ANSWER_KEYS = ("label", "answer", "answers", "output")


def first_key(row, keys):
    for k in keys:
        if k in row:
            return k
    raise KeyError(f"none of {keys} in columns {list(row)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=pathlib.Path, required=True)
    ap.add_argument("--n", type=int, default=5)
    args = ap.parse_args()

    manifest_path = args.out / "manifest.json"
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text())
        if len(existing) >= args.n:
            print(f"sample already present at {args.out} "
                  f"({len(existing)} items); nothing to do")
            return
    args.out.mkdir(parents=True, exist_ok=True)

    from datasets import load_dataset

    last_err = None
    ds = None
    for repo, split in CANDIDATES:
        try:
            print(f"trying {repo} [{split}] ...")
            ds = load_dataset(repo, split=split)
            print(f"loaded {repo}: {len(ds)} rows, columns {ds.column_names}")
            break
        except Exception as e:  # noqa: BLE001 - we genuinely want the next mirror
            print(f"  failed: {type(e).__name__}: {e}")
            last_err = e
    if ds is None:
        raise SystemExit(f"could not load ChartQA from any mirror: {last_err}")

    row0 = ds[0]
    ik = first_key(row0, IMAGE_KEYS)
    qk = first_key(row0, QUESTION_KEYS)
    ak = first_key(row0, ANSWER_KEYS)
    print(f"columns in use: image={ik!r} question={qk!r} answer={ak!r}")

    # Spread the picks across the split so the figures differ in pixel size.
    # Five consecutive rows are often five crops of one source and would not
    # exercise smart_resize at all, which is exactly what P0.10 is testing.
    stride = max(1, len(ds) // args.n)
    picks = [i * stride for i in range(args.n)]

    manifest = []
    for rank, idx in enumerate(picks):
        row = ds[idx]
        img = row[ik].convert("RGB")
        name = f"chartqa_test_{idx:06d}.png"
        img.save(args.out / name)

        gold = row[ak]
        if isinstance(gold, list):
            gold = gold[0] if gold else ""

        manifest.append({
            "rank": rank,
            "source_index": idx,
            "figure_id": name,
            "width": img.width,
            "height": img.height,
            "question": row[qk],
            "gold": gold,
        })
        print(f"  [{rank}] {name}  {img.width}x{img.height}  q={row[qk][:60]!r}")

    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"wrote {len(manifest)} figures + manifest to {args.out}")


if __name__ == "__main__":
    main()
