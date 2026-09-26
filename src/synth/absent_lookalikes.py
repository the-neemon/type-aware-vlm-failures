"""Second fabrication pilot: lookalike absent names, three question families, harder charts.

The first pilot (src/synth/absent_pairs.py) asked "What is the value of Guava?"
about a bar that is not there. Qwen answered "0" 276 times in 300, "None" 9
times, and another number 15 times, mostly the value of a bar with a similar
name (Dubai -> Dublin's value, Eagle -> Falcon's). Its present-bar reads were
95 percent right. This pilot changes three things:

1. Lookalike absent names. Category names come in pairs that look or mean
   alike (Dubai/Dublin, Eagle/Falcon, Lute/Flute). A chart shows at most one
   member of a pair. Each absent question asks about the partner of a shown bar
   (lookalike) or about a name from an unused pair (unrelated), half and half,
   and records which in `lookalike_of`.

2. Question families where "0" is not a natural answer. Each family has a
   present question and an absent one in the same words:

       value      "What is the value of Mango?"            "... of Guava?"
       compare    "Which is larger, Mango or Apple?"        "Which is larger, Guava or Apple?"
       neighbor   "Which bar is immediately to the right    "... to the right of Guava?"
                   of Mango?"

   Naming any bar in answer to the absent neighbor question locates a bar that
   does not exist, so it is a fabrication with no "0"-style escape. For the
   absent compare question, answering the absent name is a fabrication, but
   answering the present one ("Apple") is arguably the same escape as "0" and
   is scored separately (src/synth/items.py).

3. Harder charts, for more present-bar misreads (the structural side). Values
   are integers that are never multiples of 5, so they fall between gridlines,
   on 0-50, 0-100 or 0-200 axes. Half the present compare questions ask about
   the two closest bars.

Six questions per chart. Deterministic from --seed. Generate on Ada, as with
the first pilot (requirements.txt says why):

    python -m src.synth.absent_lookalikes ~/data/synth/absent_pilot_v2 --num-charts 300
    python -m src.synth.absent_lookalikes ~/data/synth/absent_pilot_v2 --verify
"""

from __future__ import annotations

import argparse
import json
import logging
import random
from collections import Counter
from pathlib import Path

from src.synth.absent_pairs import pixel_sha256
from src.synth.bar_charts import create_split_assignments, save_bar_chart, write_manifest

logger = logging.getLogger(__name__)

# Ten or more pairs per theme: seven bars use seven pairs, and up to three
# unrelated absent names need three more. Pairs look alike (Dubai/Dublin) or
# mean alike (Eagle/Falcon); both kinds produced fabrications in the first pilot.
PAIRS = {
    "city": [("Dubai", "Dublin"), ("Vienna", "Venice"), ("Oslo", "Osaka"),
             ("Delhi", "Dhaka"), ("Hanoi", "Hanover"), ("Quito", "Kyoto"),
             ("Lagos", "Lahore"), ("Paris", "Parma"), ("Seoul", "Seville"),
             ("Lima", "Lille"), ("Cairo", "Cairns")],
    "fruit": [("Lemon", "Lime"), ("Peach", "Pear"), ("Apple", "Apricot"),
              ("Grape", "Grapefruit"), ("Cherry", "Cranberry"), ("Melon", "Mango"),
              ("Orange", "Tangerine"), ("Papaya", "Guava"), ("Plum", "Prune"),
              ("Banana", "Plantain")],
    "animal": [("Eagle", "Falcon"), ("Camel", "Llama"), ("Tiger", "Lion"),
               ("Horse", "Zebra"), ("Rabbit", "Hare"), ("Moose", "Elk"),
               ("Bison", "Buffalo"), ("Otter", "Beaver"), ("Wolf", "Coyote"),
               ("Panda", "Koala")],
    "instrument": [("Lute", "Flute"), ("Violin", "Viola"), ("Guitar", "Banjo"),
                   ("Oboe", "Clarinet"), ("Trumpet", "Trombone"), ("Piano", "Organ"),
                   ("Harp", "Lyre"), ("Drums", "Bongos"), ("Cello", "Bass"),
                   ("Accordion", "Harmonica")],
    "sport": [("Tennis", "Squash"), ("Hockey", "Lacrosse"), ("Judo", "Karate"),
              ("Skiing", "Skating"), ("Rowing", "Sailing"), ("Boxing", "Wrestling"),
              ("Cricket", "Baseball"), ("Golf", "Polo"), ("Rugby", "Football"),
              ("Fencing", "Archery")],
    "country": [("Austria", "Australia"), ("Niger", "Nigeria"), ("Iran", "Iraq"),
                ("Chile", "China"), ("Mali", "Malta"), ("Guinea", "Ghana"),
                ("India", "Indonesia"), ("Spain", "Portugal"), ("Sweden", "Norway"),
                ("Slovakia", "Slovenia")],
}

# One phrasing per family per chart, shared by its present and absent question.
PHRASINGS = {
    "value": {"value_of": "What is the value of {a}?",
              "value_for": "What is the value for {a}?",
              "bar_height": "How high is the bar for {a}?"},
    "compare": {"larger": "Which is larger, {a} or {b}?",
                "higher_value": "Which has a higher value, {a} or {b}?",
                "bigger_bar": "Which bar is bigger: {a} or {b}?"},
    "neighbor": {"right_of": "Which bar is immediately to the right of {a}?",
                 "after": "Which category comes right after {a}?",
                 "next_bar": "What is the next bar after {a}?"},
}

BAR_COUNTS = (4, 5, 6, 7)
AXIS_MAXES = (50, 100, 200)
TICK_DENSITIES = ("sparse", "medium")
ABSENT_GOLD = "not present"
MIN_GAP = {50: 2, 100: 2, 200: 4}


def draw_values(rng: random.Random, bar_count: int, axis_max: int) -> list[int]:
    """Distinct integers off every multiple of 5, at least MIN_GAP apart, none near zero."""
    candidates = [v for v in range(axis_max // 10, axis_max + 1) if v % 5]
    while True:
        values = rng.sample(candidates, bar_count)
        ordered = sorted(values)
        if min(b - a for a, b in zip(ordered, ordered[1:])) >= MIN_GAP[axis_max]:
            return values


def draw_chart(rng: random.Random) -> dict:
    """Theme, bars, axis and the six questions for one chart."""
    theme = rng.choice(sorted(PAIRS))
    bar_count = rng.choice(BAR_COUNTS)
    pairs = rng.sample(PAIRS[theme], bar_count + 3)
    shown = [pair if rng.random() < 0.5 else pair[::-1] for pair in pairs[:bar_count]]
    categories = [pair[0] for pair in shown]
    partner = {pair[0]: pair[1] for pair in shown}
    unrelated = [rng.choice(pair) for pair in pairs[bar_count:]]
    axis_max = rng.choice(AXIS_MAXES)
    values = draw_values(rng, bar_count, axis_max)
    value_of = dict(zip(categories, values))

    # Three absent names, one per family: each is a lookalike of a distinct shown
    # bar or an unrelated name, independently, half and half.
    decoys = rng.sample(categories, 3)
    absent = []
    for family, decoy, other in zip(PHRASINGS, decoys, unrelated):
        lookalike = rng.random() < 0.5
        absent.append((family, partner[decoy] if lookalike else other,
                       decoy if lookalike else None))
    absent = {family: (name, of) for family, name, of in absent}

    phrasing = {family: rng.choice(sorted(options)) for family, options in PHRASINGS.items()}

    def ask(family, a, b=None):
        return PHRASINGS[family][phrasing[family]].format(a=a, b=b)

    def question(template, family, absent_q, asks_about, gold, **extra):
        return {"template": template, "family": family, "absent": absent_q,
                "target_failure_type": "fabrication" if absent_q else "structural",
                "phrasing": phrasing[family], "asks_about": asks_about,
                "gold_answer": gold, **extra}

    questions = []
    target = rng.choice(categories)
    questions.append(question("read_value", "value", False, target, value_of[target],
                              question=ask("value", target)))
    name, of = absent["value"]
    questions.append(question("absent_value", "value", True, name, ABSENT_GOLD,
                              lookalike_of=of, question=ask("value", name)))

    if rng.random() < 0.5:
        ordered = sorted(categories, key=value_of.get)
        a, b = min(zip(ordered, ordered[1:]), key=lambda p: value_of[p[1]] - value_of[p[0]])
        closest = True
    else:
        a, b = rng.sample(categories, 2)
        closest = False
    a, b = (a, b) if rng.random() < 0.5 else (b, a)
    questions.append(question("compare_present", "compare", False, a,
                              max((a, b), key=value_of.get), compared_with=b,
                              closest_pair=closest, gap=abs(value_of[a] - value_of[b]),
                              question=ask("compare", a, b)))
    name, of = absent["compare"]
    other = rng.choice([c for c in categories if c != of])
    first, second = (name, other) if rng.random() < 0.5 else (other, name)
    questions.append(question("absent_compare", "compare", True, name, ABSENT_GOLD,
                              lookalike_of=of, compared_with=other,
                              question=ask("compare", first, second)))

    position = rng.randrange(bar_count - 1)
    questions.append(question("neighbor_present", "neighbor", False, categories[position],
                              categories[position + 1],
                              question=ask("neighbor", categories[position])))
    name, of = absent["neighbor"]
    questions.append(question("absent_neighbor", "neighbor", True, name, ABSENT_GOLD,
                              lookalike_of=of, question=ask("neighbor", name)))

    return {"theme": theme, "categories": categories, "values": values,
            "absent_partners": [partner[c] for c in categories],
            "axis_min": 0, "axis_max": axis_max, "min_height_separation": MIN_GAP[axis_max],
            "tick_density": rng.choice(TICK_DENSITIES), "questions": questions}


def validate_entries(entries: list[dict], output_dir: Path) -> None:
    """The properties the probe contrasts rely on. Raises on the first violation."""
    ids = [e["figure_id"] for e in entries]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate figure ids")
    for e in entries:
        fid, cats, vals = e["figure_id"], e["categories"], e["values"]
        value_of = dict(zip(cats, vals))
        if not (output_dir / e["image_path"]).is_file():
            raise FileNotFoundError(f"missing image for {fid}")
        if len(set(cats)) != len(cats) or len(cats) != len(vals):
            raise ValueError(f"{fid}: categories must be distinct, one per value")
        if set(cats) & set(e["absent_partners"]):
            raise ValueError(f"{fid}: a chart shows both members of a lookalike pair")
        if any(v % 5 == 0 or not 0 < v <= e["axis_max"] for v in vals):
            raise ValueError(f"{fid}: a value is on a multiple of 5 or off the axis")
        ordered = sorted(vals)
        if min(b - a for a, b in zip(ordered, ordered[1:])) < e["min_height_separation"]:
            raise ValueError(f"{fid}: two bars are closer than the minimum separation")
        qs = e["questions"]
        if [q["template"] for q in qs] != ["read_value", "absent_value", "compare_present",
                                           "absent_compare", "neighbor_present",
                                           "absent_neighbor"]:
            raise ValueError(f"{fid}: unexpected question set")
        absent_names = [q["asks_about"] for q in qs if q["absent"]]
        if len(set(absent_names)) != 3 or set(absent_names) & set(cats):
            raise ValueError(f"{fid}: absent names must be three distinct non-bars")
        for q in qs:
            if q["absent"]:
                of = q["lookalike_of"]
                if of is not None and e["absent_partners"][cats.index(of)] != q["asks_about"]:
                    raise ValueError(f"{fid}: {q['asks_about']} is not {of}'s lookalike")
            elif q["asks_about"] not in cats:
                raise ValueError(f"{fid}: present question asks about a non-bar")
        read, _, comp, _, neigh, _ = qs
        if read["gold_answer"] != value_of[read["asks_about"]]:
            raise ValueError(f"{fid}: read_value gold is not the bar's value")
        pair = (comp["asks_about"], comp["compared_with"])
        if comp["gold_answer"] != max(pair, key=value_of.get):
            raise ValueError(f"{fid}: compare gold is not the larger bar")
        if cats.index(neigh["gold_answer"]) != cats.index(neigh["asks_about"]) + 1:
            raise ValueError(f"{fid}: neighbor gold is not the next bar")
        for family in PHRASINGS:
            if len({q["phrasing"] for q in qs if q["family"] == family}) != 1:
                raise ValueError(f"{fid}: {family} questions are worded differently")


def generate(output_dir, num_charts: int = 300, seed: int = 43,
             split_ratios=(0.7, 0.15, 0.15), prefix: str = "look") -> list[dict]:
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
        {"generator": "src.synth.absent_lookalikes", "num_charts": num_charts, "seed": seed,
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
    ap.add_argument("--seed", type=int, default=43)
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
