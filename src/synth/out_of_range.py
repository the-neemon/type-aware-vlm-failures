"""Third fabrication pilot: a year outside the chart's range (AGENTS.md decision 4).

In pilots 1 and 2 every absent question named a category that is not a bar
(Guava on a fruit chart), so a "fabrication" there is, by construction, an
answer about a missing category. This pilot asks about something the chart's
own axis implies but does not show. Each chart plots consecutive years, and
gets two questions with identical wording:

    present   "What was the value in 2017?"   2017 is a bar; gold is its value
    absent    "What was the value in 2021?"   the bars run 2014 to 2019

The absent year lies before the first bar or after the last, one to three
years out (`direction`, `years_beyond`). Values rise, fall or vary with no
trend (`trend`), so a model that extrapolates a rising series has something to
extrapolate from, and the scorer's "fabricated" outcome can be checked against
`extrapolated`, the least-squares line evaluated at the absent year.

Templates are pilot 2's value templates (`read_value`, `absent_value`, family
`value`), so src/synth/items.py scores the absent question as rejected, "0" or
fabricated, and the probe and E4 code read pilot 3 like the earlier pilots.
Values are integers off every multiple of 5 on 0-50, 0-100 or 0-200 axes, as in
pilot 2, so present reads keep producing misreads.

Deterministic from --seed. Generate on Ada, as with the earlier pilots
(requirements.txt says why):

    python -m src.synth.out_of_range ~/data/synth/absent_pilot_v3 --num-charts 300
    python -m src.synth.out_of_range ~/data/synth/absent_pilot_v3 --verify
"""

from __future__ import annotations

import argparse
import json
import logging
import random
from collections import Counter
from pathlib import Path

from src.synth.absent_lookalikes import AXIS_MAXES, MIN_GAP, TICK_DENSITIES, draw_values
from src.synth.absent_pairs import pixel_sha256
from src.synth.bar_charts import create_split_assignments, save_bar_chart, write_manifest

logger = logging.getLogger(__name__)

# One phrasing per chart, shared by its present and absent question.
PHRASINGS = {
    "value_in": "What was the value in {year}?",
    "value_for": "What is the value for {year}?",
    "bar_height": "How high is the bar for {year}?",
    "value_of_year": "What value does the chart show for {year}?",
}

YEAR_COUNTS = (4, 5, 6, 7)
FIRST_YEARS = range(2005, 2017)          # last bar at most 2022
TRENDS = ("rising", "falling", "none")
DIRECTIONS = ("before", "after")
YEARS_BEYOND = (1, 2, 3)
ABSENT_GOLD = "not present"


def extrapolate(years: list[int], values: list[int], year: int) -> float:
    """The least-squares line through the bars, evaluated at `year`."""
    n = len(years)
    mx, my = sum(years) / n, sum(values) / n
    slope = (sum((x - mx) * (y - my) for x, y in zip(years, values))
             / sum((x - mx) ** 2 for x in years))
    return round(my + slope * (year - mx), 1)


def draw_chart(rng: random.Random) -> dict:
    """Years, bars, axis and the two questions for one chart."""
    count = rng.choice(YEAR_COUNTS)
    first = rng.choice(FIRST_YEARS)
    years = list(range(first, first + count))
    axis_max = rng.choice(AXIS_MAXES)
    values = draw_values(rng, count, axis_max)
    trend = rng.choice(TRENDS)
    if trend != "none":
        values.sort(reverse=trend == "falling")
    target = rng.randrange(count)
    direction = rng.choice(DIRECTIONS)
    beyond = rng.choice(YEARS_BEYOND)
    absent_year = years[-1] + beyond if direction == "after" else years[0] - beyond
    phrasing = rng.choice(sorted(PHRASINGS))
    wording = PHRASINGS[phrasing]
    categories = [str(y) for y in years]

    def question(template, absent, year, gold, **extra):
        return {"template": template, "family": "value", "absent": absent,
                "target_failure_type": "fabrication" if absent else "structural",
                "phrasing": phrasing, "asks_about": str(year),
                "question": wording.format(year=year), "gold_answer": gold,
                "lookalike_of": None, **extra}

    questions = [
        question("read_value", False, years[target], values[target]),
        question("absent_value", True, absent_year, ABSENT_GOLD, direction=direction,
                 years_beyond=beyond, extrapolated=extrapolate(years, values, absent_year)),
    ]
    return {"theme": "year", "categories": categories, "values": values, "trend": trend,
            "axis_min": 0, "axis_max": axis_max, "min_height_separation": MIN_GAP[axis_max],
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
        years = [int(c) for c in cats]
        if years != list(range(years[0], years[0] + len(years))) or len(vals) != len(years):
            raise ValueError(f"{fid}: bars must be consecutive years, one value each")
        if any(v % 5 == 0 or not 0 < v <= e["axis_max"] for v in vals):
            raise ValueError(f"{fid}: a value is on a multiple of 5 or off the axis")
        ordered = sorted(vals)
        if min(b - a for a, b in zip(ordered, ordered[1:])) < e["min_height_separation"]:
            raise ValueError(f"{fid}: two bars are closer than the minimum separation")
        if e["trend"] == "rising" and vals != ordered or \
                e["trend"] == "falling" and vals != ordered[::-1]:
            raise ValueError(f"{fid}: values do not follow the {e['trend']} trend")
        present, absent = e["questions"]
        if [present["template"], absent["template"]] != ["read_value", "absent_value"]:
            raise ValueError(f"{fid}: unexpected question set")
        if present["asks_about"] not in cats or \
                present["gold_answer"] != vals[cats.index(present["asks_about"])]:
            raise ValueError(f"{fid}: present question is not about a bar, or gold is wrong")
        year = int(absent["asks_about"])
        expected = years[-1] + absent["years_beyond"] if absent["direction"] == "after" \
            else years[0] - absent["years_beyond"]
        if year != expected or absent["asks_about"] in cats:
            raise ValueError(f"{fid}: absent year is not {absent['years_beyond']} "
                             f"{absent['direction']} the shown range")
        if (present["question"].replace(present["asks_about"], "{year}")
                != absent["question"].replace(absent["asks_about"], "{year}")):
            raise ValueError(f"{fid}: present and absent questions are worded differently")


def generate(output_dir, num_charts: int = 300, seed: int = 42,
             split_ratios=(0.7, 0.15, 0.15), prefix: str = "range") -> list[dict]:
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
                                  chart["axis_min"], chart["axis_max"], chart["tick_density"],
                                  xlabel="Year")
        entries.append({"figure_id": figure_id, "figure_type": "bar_chart", "split": split,
                        "image_path": str(image_path.relative_to(output_dir)),
                        "pixel_sha256": pixel_sha256(image_path), **chart, **geometry})
        if (index + 1) % 50 == 0:
            logger.info("rendered %d/%d charts", index + 1, num_charts)

    validate_entries(entries, output_dir)
    write_manifest(entries, manifest_path)
    (output_dir / "generation.json").write_text(json.dumps(
        {"generator": "src.synth.out_of_range", "num_charts": num_charts, "seed": seed,
         "split_ratios": list(split_ratios), "prefix": prefix}, indent=2) + "\n")
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
