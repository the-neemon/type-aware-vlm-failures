"""Bar charts asked about one bar that is there and one that is not (fabrication pilot).

Every chart gets two questions with identical wording:

    present   "What is the value of Mango?"   Mango is a bar; gold is its value
    absent    "What is the value of Guava?"   Guava is not; gold is "not present"

The absent name comes from the same theme as the bars (another fruit on a
fruit chart), so it is a plausible category rather than an odd one out. The
point is the contrast the fabrication probe trains on: absent questions the
model answered with a number (fabricated) against absent questions it rejected,
all under one template, so nothing in the question text separates the two
classes. The present questions give the same contrast for misreads (read
correctly against misread), and a pipeline check: Qwen should read most bars.

This differs from src/synth/bar_charts.py on purpose. There every chart has
categories A to E, the absent one is always F, and every question asks about A
and B, so all prompts are the same string. Here names, bar count, axis range,
tick density, phrasing and target bar all vary per chart. Rendering is shared
(`save_bar_chart`), so the bar geometry E4's crop needs is recorded the same way.

Deterministic from --seed. Each image's `pixel_sha256` is a hash of its decoded
RGBA pixels, not the file bytes, so --verify checks a folder's images against
its own manifest whatever the PNG encoder did. Across machines, names, values,
questions and splits always match, but pixels and bar geometry match only under
the same matplotlib build (requirements.txt explains): generate on Ada, in the
GPU environment, and treat that copy as the dataset.

    python -m src.synth.absent_pairs data/synth/absent_pilot_v1 --num-charts 300
    python -m src.synth.absent_pairs data/synth/absent_pilot_v1 --verify
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import random
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

from src.synth.bar_charts import create_split_assignments, save_bar_chart, write_manifest

logger = logging.getLogger(__name__)

# Fourteen names per theme: at most seven bars leaves at least seven to draw the
# absent name from. Short single words, so no tick label is truncated or wrapped.
THEMES = {
    "fruit": ["Apple", "Banana", "Cherry", "Mango", "Grape", "Lemon", "Peach",
              "Pear", "Plum", "Kiwi", "Orange", "Melon", "Papaya", "Guava"],
    "country": ["France", "Brazil", "Japan", "Kenya", "Canada", "Peru", "Norway",
                "India", "Egypt", "Mexico", "Spain", "Chile", "Ghana", "Italy"],
    "animal": ["Tiger", "Zebra", "Panda", "Koala", "Otter", "Eagle", "Horse",
               "Camel", "Llama", "Bison", "Moose", "Rabbit", "Falcon", "Lynx"],
    "sport": ["Tennis", "Rugby", "Hockey", "Golf", "Soccer", "Cricket", "Boxing",
              "Rowing", "Cycling", "Skiing", "Judo", "Fencing", "Karate", "Squash"],
    "city": ["Paris", "Tokyo", "Lagos", "Lima", "Oslo", "Delhi", "Cairo",
             "Rome", "Seoul", "Dubai", "Quito", "Hanoi", "Vienna", "Dublin"],
    "instrument": ["Piano", "Violin", "Guitar", "Flute", "Drums", "Cello", "Harp",
                   "Trumpet", "Banjo", "Oboe", "Sitar", "Tuba", "Organ", "Lute"],
}

# One phrasing per chart, shared by its present and absent question.
PHRASINGS = {
    "value_of": "What is the value of {name}?",
    "value_for": "What is the value for {name}?",
    "value_of_category": "What is the value of category {name}?",
    "bar_height": "How high is the bar for {name}?",
}

BAR_COUNTS = (4, 5, 6, 7)
AXIS_MAXES = (50, 100)
TICK_DENSITIES = ("sparse", "medium")
ABSENT_GOLD = "not present"


def pixel_sha256(path: Path) -> str:
    """Hash of the decoded pixels, independent of how the PNG was compressed."""
    with Image.open(path) as image:
        pixels = np.asarray(image.convert("RGBA"))
    return hashlib.sha256(pixels.tobytes() + str(pixels.shape).encode()).hexdigest()


def draw_chart(rng: random.Random) -> dict:
    """Theme, bars, axis and the two questions for one chart."""
    theme = rng.choice(sorted(THEMES))
    bar_count = rng.choice(BAR_COUNTS)
    names = rng.sample(THEMES[theme], bar_count + 1)
    categories, absent_name = names[:bar_count], names[bar_count]
    axis_max = rng.choice(AXIS_MAXES)
    separation = axis_max // 25              # 2 on a 0-50 axis, 4 on 0-100
    # No zero-height bars: "0" is also the most likely fabricated value.
    values = rng.sample(range(separation, axis_max + 1, separation), bar_count)
    target = rng.randrange(bar_count)
    phrasing = rng.choice(sorted(PHRASINGS))
    wording = PHRASINGS[phrasing]
    questions = [
        {"template": "read_value", "target_failure_type": "structural", "absent": False,
         "phrasing": phrasing, "asks_about": categories[target],
         "question": wording.format(name=categories[target]), "gold_answer": values[target]},
        {"template": "absent_category", "target_failure_type": "fabrication", "absent": True,
         "phrasing": phrasing, "asks_about": absent_name,
         "question": wording.format(name=absent_name), "gold_answer": ABSENT_GOLD},
    ]
    return {"theme": theme, "categories": categories, "values": values,
            "axis_min": 0, "axis_max": axis_max, "min_height_separation": separation,
            "tick_density": rng.choice(TICK_DENSITIES), "questions": questions}


def validate_entries(entries: list[dict], output_dir: Path) -> None:
    """The properties the probe contrast relies on. Raises on the first violation."""
    ids = [e["figure_id"] for e in entries]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate figure ids")
    for e in entries:
        fid, cats, vals = e["figure_id"], e["categories"], e["values"]
        if not (output_dir / e["image_path"]).is_file():
            raise FileNotFoundError(f"missing image for {fid}")
        if len(set(cats)) != len(cats) or len(cats) != len(vals):
            raise ValueError(f"{fid}: categories must be distinct, one per value")
        if any(not e["axis_min"] < v <= e["axis_max"] for v in vals):
            raise ValueError(f"{fid}: a value is zero or outside the axis")
        gaps = [abs(a - b) for i, a in enumerate(vals) for b in vals[i + 1:]]
        if min(gaps) < e["min_height_separation"]:
            raise ValueError(f"{fid}: two bars are closer than the minimum separation")
        present, absent = e["questions"]
        if present["asks_about"] not in cats or absent["asks_about"] in cats:
            raise ValueError(f"{fid}: present/absent question targets are wrong")
        if present["gold_answer"] != vals[cats.index(present["asks_about"])]:
            raise ValueError(f"{fid}: present gold is not the bar's value")
        if absent["asks_about"] not in THEMES[e["theme"]]:
            raise ValueError(f"{fid}: absent name is not from the chart's theme")
        if (present["question"].replace(present["asks_about"], "{name}")
                != absent["question"].replace(absent["asks_about"], "{name}")):
            raise ValueError(f"{fid}: present and absent questions are worded differently")


def generate(output_dir, num_charts: int = 300, seed: int = 42,
             split_ratios=(0.7, 0.15, 0.15), prefix: str = "absent") -> list[dict]:
    """Render the charts and write manifest.jsonl. Refuses to overwrite."""
    output_dir = Path(output_dir)
    manifest_path = output_dir / "manifest.jsonl"
    if manifest_path.exists():
        raise FileExistsError(f"refusing to overwrite existing {manifest_path}")
    image_dir = output_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(seed)
    splits = create_split_assignments(num_charts, split_ratios, rng)
    entries = []
    for index, split in enumerate(splits):
        figure_id = f"{prefix}_{index:06d}"
        image_path = image_dir / f"{figure_id}.png"
        if image_path.exists():
            raise FileExistsError(f"refusing to overwrite existing {image_path}")
        chart = draw_chart(random.Random(rng.randrange(2**32)))
        geometry = save_bar_chart(image_path, chart["categories"], chart["values"],
                                  chart["axis_min"], chart["axis_max"], chart["tick_density"])
        entries.append({"figure_id": figure_id, "figure_type": "bar_chart", "split": split,
                        "image_path": str(image_path.relative_to(output_dir)),
                        "pixel_sha256": pixel_sha256(image_path), **chart, **geometry})
        if (index + 1) % 50 == 0:
            logger.info("rendered %d/%d charts", index + 1, num_charts)

    validate_entries(entries, output_dir)
    write_manifest(entries, manifest_path)
    (output_dir / "generation.json").write_text(json.dumps(
        {"num_charts": num_charts, "seed": seed, "split_ratios": list(split_ratios),
         "prefix": prefix}, indent=2) + "\n")
    logger.info("wrote %s: %d charts, %d questions, splits %s", manifest_path, len(entries),
                sum(len(e["questions"]) for e in entries), dict(Counter(splits)))
    return entries


def verify(output_dir) -> list[str]:
    """Problems with a generated folder: failed checks, or images whose pixels differ."""
    output_dir = Path(output_dir)
    entries = [json.loads(line) for line in
               (output_dir / "manifest.jsonl").read_text(encoding="utf-8").splitlines() if line]
    try:
        validate_entries(entries, output_dir)
    except (ValueError, FileNotFoundError) as exc:
        return [str(exc)]
    return [f"{e['figure_id']}: pixels differ from the manifest" for e in entries
            if pixel_sha256(output_dir / e["image_path"]) != e["pixel_sha256"]]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("output_dir")
    ap.add_argument("--num-charts", type=int, default=300)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--verify", action="store_true",
                    help="check an existing folder against its manifest instead of generating")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s",
                        datefmt="%H:%M:%S")
    if args.verify:
        problems = verify(args.output_dir)
        print("\n".join(problems[:20]) if problems else "OK: every image matches the manifest")
        return 1 if problems else 0
    generate(args.output_dir, args.num_charts, args.seed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
