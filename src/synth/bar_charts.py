import json
import logging
import math
import random
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt

from src.synth.questions import build_bar_questions


logger = logging.getLogger(__name__)

TICK_COUNTS = {"sparse": 6, "medium": 11, "dense": 21}
SPLIT_NAMES = ("train", "validation", "test")


def validate_generation_settings(
    num_charts,
    bar_count,
    axis_min,
    axis_max,
    min_height_separation,
    tick_density,
    split_ratios,
):
    """Validate chart-generation settings before writing any output."""
    if num_charts < 1:
        raise ValueError("num_charts must be at least 1")
    if not 2 <= bar_count <= 26:
        raise ValueError("bar_count must be between 2 and 26")
    if axis_min >= axis_max:
        raise ValueError("axis_min must be smaller than axis_max")
    if min_height_separation <= 0:
        raise ValueError("min_height_separation must be positive")
    if tick_density not in TICK_COUNTS:
        raise ValueError(f"tick_density must be one of {tuple(TICK_COUNTS)}")
    if len(split_ratios) != len(SPLIT_NAMES) or any(ratio < 0 for ratio in split_ratios):
        raise ValueError("split_ratios must contain three non-negative values")
    if not math.isclose(sum(split_ratios), 1.0, abs_tol=1e-9):
        raise ValueError("split_ratios must sum to 1")

    available_values = math.floor(
        (axis_max - axis_min) / min_height_separation
    ) + 1
    if available_values < bar_count:
        raise ValueError(
            "axis range cannot fit the requested bars at the minimum separation"
        )


def create_split_assignments(num_charts, split_ratios, random_generator):
    """Create deterministic split assignments with ratios rounded by remainder."""
    exact_counts = [num_charts * ratio for ratio in split_ratios]
    split_counts = [math.floor(count) for count in exact_counts]
    remaining = num_charts - sum(split_counts)
    remainder_order = sorted(
        range(len(SPLIT_NAMES)),
        key=lambda index: exact_counts[index] - split_counts[index],
        reverse=True,
    )
    for index in remainder_order[:remaining]:
        split_counts[index] += 1

    assignments = []
    for split_name, split_count in zip(SPLIT_NAMES, split_counts):
        assignments.extend([split_name] * split_count)
    random_generator.shuffle(assignments)
    return assignments


def generate_bar_values(
    bar_count,
    axis_min,
    axis_max,
    min_height_separation,
    random_generator,
):
    """Generate bar values with a guaranteed minimum pairwise separation."""
    value_count = math.floor((axis_max - axis_min) / min_height_separation) + 1
    value_slots = random_generator.sample(range(value_count), bar_count)
    return [axis_min + slot * min_height_separation for slot in value_slots]


def save_bar_chart(
    image_path,
    categories,
    values,
    axis_min,
    axis_max,
    tick_density,
):
    """Render one bar chart to a PNG file."""
    figure, axis = plt.subplots(figsize=(8, 6), dpi=120)
    axis.bar(categories, values, color="#4C78A8", width=0.65)
    axis.set_ylim(axis_min, axis_max)
    tick_count = TICK_COUNTS[tick_density]
    tick_step = (axis_max - axis_min) / (tick_count - 1)
    axis.set_yticks([axis_min + index * tick_step for index in range(tick_count)])
    axis.set_xlabel("Category")
    axis.set_ylabel("Value")
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(image_path)
    plt.close(figure)


def validate_manifest_entries(entries, output_dir, min_height_separation):
    """Validate IDs, image paths, splits, ranges, and value separation."""
    figure_ids = [entry["figure_id"] for entry in entries]
    if len(figure_ids) != len(set(figure_ids)):
        raise ValueError("manifest contains duplicate figure IDs")

    for entry in entries:
        if entry["split"] not in SPLIT_NAMES:
            raise ValueError(f"invalid split for {entry['figure_id']}")
        if not (output_dir / entry["image_path"]).is_file():
            raise FileNotFoundError(f"missing image for {entry['figure_id']}")
        values = entry["values"]
        if any(value < entry["axis_min"] or value > entry["axis_max"] for value in values):
            raise ValueError(f"out-of-range value for {entry['figure_id']}")
        smallest_gap = min(
            abs(left - right)
            for index, left in enumerate(values)
            for right in values[index + 1 :]
        )
        if smallest_gap < min_height_separation:
            raise ValueError(f"insufficient height separation for {entry['figure_id']}")


def write_manifest(entries, manifest_path):
    """Write manifest entries as deterministic JSON Lines."""
    with manifest_path.open("w", encoding="utf-8") as manifest_file:
        for entry in entries:
            manifest_file.write(json.dumps(entry, sort_keys=True) + "\n")


def generate_bar_chart_dataset(
    output_dir,
    num_charts=20,
    bar_count=5,
    axis_min=0,
    axis_max=100,
    min_height_separation=5,
    tick_density="medium",
    split_ratios=(0.7, 0.15, 0.15),
    seed=42,
):
    """Generate reproducible bar charts and return their validated manifest."""
    output_dir = Path(output_dir)
    validate_generation_settings(
        num_charts,
        bar_count,
        axis_min,
        axis_max,
        min_height_separation,
        tick_density,
        split_ratios,
    )
    logger.info("Validated generation settings for %d charts", num_charts)

    image_dir = output_dir / "images"
    manifest_path = output_dir / "manifest.jsonl"
    if manifest_path.exists():
        raise FileExistsError(f"refusing to overwrite existing {manifest_path}")
    image_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Prepared output directory: %s", output_dir)

    random_generator = random.Random(seed)
    assignments = create_split_assignments(num_charts, split_ratios, random_generator)
    categories = [chr(ord("A") + index) for index in range(bar_count)]
    entries = []

    for index, split_name in enumerate(assignments):
        figure_id = f"bar_{index:06d}"
        image_path = image_dir / f"{figure_id}.png"
        if image_path.exists():
            raise FileExistsError(f"refusing to overwrite existing {image_path}")
        chart_seed = random_generator.randrange(2**32)
        values = generate_bar_values(
            bar_count,
            axis_min,
            axis_max,
            min_height_separation,
            random.Random(chart_seed),
        )
        save_bar_chart(
            image_path,
            categories,
            values,
            axis_min,
            axis_max,
            tick_density,
        )
        entries.append(
            {
                "axis_max": axis_max,
                "axis_min": axis_min,
                "categories": categories,
                "figure_id": figure_id,
                "figure_type": "bar_chart",
                "image_path": str(image_path.relative_to(output_dir)),
                "min_height_separation": min_height_separation,
                "seed": chart_seed,
                "split": split_name,
                "tick_density": tick_density,
                "values": values,
                "questions": build_bar_questions(categories, values),
            }
        )

    logger.info("Rendered %d bar-chart images", len(entries))
    validate_manifest_entries(entries, output_dir, min_height_separation)
    logger.info("Validated manifest entries and figure-level splits")
    write_manifest(entries, manifest_path)
    logger.info("Wrote manifest: %s", manifest_path)
    return entries
