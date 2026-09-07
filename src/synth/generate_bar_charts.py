import argparse
import logging
from collections import Counter

from src.synth.bar_charts import generate_bar_chart_dataset


def parse_arguments():
    """Parse command-line settings for synthetic bar-chart generation."""
    parser = argparse.ArgumentParser(
        description="Generate reproducible synthetic bar charts and a JSONL manifest."
    )
    parser.add_argument("output_dir")
    parser.add_argument("--num-charts", type=int, default=20)
    parser.add_argument("--bar-count", type=int, default=5)
    parser.add_argument("--axis-min", type=int, default=0)
    parser.add_argument("--axis-max", type=int, default=100)
    parser.add_argument("--min-height-separation", type=int, default=5)
    parser.add_argument(
        "--tick-density", choices=("sparse", "medium", "dense"), default="medium"
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def configure_logging():
    """Configure concise timestamped progress logs."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def main():
    """Generate a dataset and log its final split counts."""
    configure_logging()
    args = parse_arguments()
    entries = generate_bar_chart_dataset(
        output_dir=args.output_dir,
        num_charts=args.num_charts,
        bar_count=args.bar_count,
        axis_min=args.axis_min,
        axis_max=args.axis_max,
        min_height_separation=args.min_height_separation,
        tick_density=args.tick_density,
        seed=args.seed,
    )
    split_counts = Counter(entry["split"] for entry in entries)
    logging.info(
        "Generation complete | total=%d | train=%d | validation=%d | test=%d",
        len(entries),
        split_counts["train"],
        split_counts["validation"],
        split_counts["test"],
    )


if __name__ == "__main__":
    main()
