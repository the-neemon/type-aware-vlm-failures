import argparse
import logging

from src.synth.matched_pairs import generate_matched_pair_dataset
from src.synth.node_links import generate_node_link_dataset


def parse_arguments():
    """Parse the node-link or matched-pair generation command."""
    parser = argparse.ArgumentParser(description="Generate follow-up synthetic data.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    node_parser = subparsers.add_parser("nodes")
    node_parser.add_argument("output_dir")
    node_parser.add_argument("--num-figures", type=int, default=20)
    node_parser.add_argument("--path-length", type=int, default=3)
    node_parser.add_argument("--edge-crossings", type=int, default=1)
    node_parser.add_argument("--seed", type=int, default=42)

    pair_parser = subparsers.add_parser("pairs")
    pair_parser.add_argument("output_dir")
    pair_parser.add_argument("--pair-count", type=int, default=10)
    pair_parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    """Run the selected generator with timestamped stage logs."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    args = parse_arguments()
    if args.command == "nodes":
        generate_node_link_dataset(
            args.output_dir,
            num_figures=args.num_figures,
            path_length=args.path_length,
            edge_crossings=args.edge_crossings,
            seed=args.seed,
        )
    else:
        generate_matched_pair_dataset(
            args.output_dir, pair_count=args.pair_count, seed=args.seed
        )
    logging.info("Follow-up generation complete | type=%s", args.command)


if __name__ == "__main__":
    main()
