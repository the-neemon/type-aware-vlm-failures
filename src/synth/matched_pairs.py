import logging
from collections import defaultdict
from pathlib import Path

from src.synth.bar_charts import create_split_assignments, save_bar_chart, write_manifest
from src.synth.node_links import create_node_link_graph, save_node_link_diagram
from src.synth.questions import build_bar_questions, build_node_questions


logger = logging.getLogger(__name__)


def validate_matched_pair_entries(entries, output_dir):
    """Validate pair membership, variants, splits, and generated images."""
    grouped_entries = defaultdict(list)
    for entry in entries:
        grouped_entries[entry["pair_id"]].append(entry)
        if not (output_dir / entry["image_path"]).is_file():
            raise FileNotFoundError(f"missing image for {entry['figure_id']}")

    for pair_id, pair_entries in grouped_entries.items():
        if len(pair_entries) != 2:
            raise ValueError(f"{pair_id} must contain exactly two figures")
        if {entry["pair_variant"] for entry in pair_entries} != {"easy", "hard"}:
            raise ValueError(f"{pair_id} must contain easy and hard variants")
        if len({entry["split"] for entry in pair_entries}) != 1:
            raise ValueError(f"{pair_id} variants must share one split")


def generate_matched_pair_dataset(output_dir, pair_count=10, seed=16):
    """Generate pairs varying either visual precision or relational depth."""
    import random

    if pair_count < 2:
        raise ValueError("pair_count must be at least 2")
    output_dir = Path(output_dir)
    image_dir = output_dir / "images"
    manifest_path = output_dir / "manifest.jsonl"
    if manifest_path.exists():
        raise FileExistsError(f"refusing to overwrite existing {manifest_path}")
    image_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Prepared matched-pair output directory: %s", output_dir)

    assignments = create_split_assignments(
        pair_count, (0.7, 0.15, 0.15), random.Random(seed)
    )
    entries = []
    for pair_index, split_name in enumerate(assignments):
        pair_id = f"pair_{pair_index:06d}"
        if pair_index % 2 == 0:
            entries.extend(
                generate_visual_precision_pair(image_dir, pair_id, split_name)
            )
        else:
            entries.extend(
                generate_relational_depth_pair(image_dir, pair_id, split_name, pair_index)
            )

    validate_matched_pair_entries(entries, output_dir)
    logger.info("Validated matched-pair membership and split integrity")
    write_manifest(entries, manifest_path)
    logger.info("Generated %d matched pairs across %d figures", pair_count, len(entries))
    logger.info("Wrote matched-pair manifest: %s", manifest_path)
    return entries


def generate_visual_precision_pair(image_dir, pair_id, split_name):
    """Create two comparison charts with equal reasoning and different precision."""
    categories = ["A", "B", "C", "D"]
    variants = {"easy": [25, 75, 40, 60], "hard": [48, 52, 30, 70]}
    entries = []
    for variant, values in variants.items():
        figure_id = f"{pair_id}_{variant}"
        image_path = image_dir / f"{figure_id}.png"
        save_bar_chart(image_path, categories, values, 0, 100, "medium")
        entries.append(
            {
                "figure_id": figure_id,
                "figure_type": "bar_chart",
                "image_path": f"images/{image_path.name}",
                "pair_id": pair_id,
                "pair_variant": variant,
                "split": split_name,
                "varied_factor": "visual_precision",
                "fixed_factor": "relational_depth",
                "values": values,
                "questions": [build_bar_questions(categories, values)[1]],
            }
        )
    return entries


def generate_relational_depth_pair(image_dir, pair_id, split_name, pair_index):
    """Create two path diagrams with equal crossings and different path lengths."""
    entries = []
    for variant, path_length in (("easy", 1), ("hard", 4)):
        figure_id = f"{pair_id}_{variant}"
        image_path = image_dir / f"{figure_id}.png"
        graph, positions, path_nodes = create_node_link_graph(
            path_length, 1, pair_index
        )
        save_node_link_diagram(image_path, graph, positions)
        entries.append(
            {
                "figure_id": figure_id,
                "figure_type": "node_link",
                "image_path": f"images/{image_path.name}",
                "pair_id": pair_id,
                "pair_variant": variant,
                "split": split_name,
                "varied_factor": "relational_depth",
                "fixed_factor": "edge_crossings",
                "path_length": path_length,
                "questions": [build_node_questions(path_nodes)[0]],
            }
        )
    return entries
