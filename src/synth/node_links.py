import logging
from pathlib import Path

import matplotlib
import networkx as nx

matplotlib.use("Agg")
from matplotlib import pyplot as plt

from src.synth.bar_charts import create_split_assignments, write_manifest
from src.synth.questions import build_node_questions


logger = logging.getLogger(__name__)


def validate_node_settings(num_figures, path_length, edge_crossings):
    """Validate node-link controls before creating output."""
    if num_figures < 1:
        raise ValueError("num_figures must be at least 1")
    if not 1 <= path_length <= 10:
        raise ValueError("path_length must be between 1 and 10")
    if not 0 <= edge_crossings <= 8:
        raise ValueError("edge_crossings must be between 0 and 8")


def create_node_link_graph(path_length, edge_crossings, figure_index=0):
    """Create a connected path graph with an exact number of edge crossings."""
    graph = nx.Graph()
    path_nodes = [f"P{figure_index}_{index}" for index in range(path_length + 1)]
    positions = {node: (index, 2.0) for index, node in enumerate(path_nodes)}
    nx.add_path(graph, path_nodes)
    attachment_node = path_nodes[-1]

    for crossing_index in range(edge_crossings):
        x_start = path_length + 2 + crossing_index * 3
        nodes = [f"X{figure_index}_{crossing_index}_{suffix}" for suffix in "ABCD"]
        positions.update(
            {
                nodes[0]: (x_start, 0.0),
                nodes[1]: (x_start + 2, -2.0),
                nodes[2]: (x_start + 2, 0.0),
                nodes[3]: (x_start, -2.0),
            }
        )
        graph.add_edge(nodes[0], nodes[1])
        graph.add_edge(nodes[2], nodes[3])
        graph.add_edge(nodes[0], nodes[2])
        graph.add_edge(attachment_node, nodes[0])
        attachment_node = nodes[2]
    return graph, positions, path_nodes


def segments_cross(first_start, first_end, second_start, second_end):
    """Return whether two straight segments cross strictly inside both."""
    def orientation(point_a, point_b, point_c):
        return (point_b[0] - point_a[0]) * (point_c[1] - point_a[1]) - (
            point_b[1] - point_a[1]
        ) * (point_c[0] - point_a[0])

    first_side = orientation(first_start, first_end, second_start)
    second_side = orientation(first_start, first_end, second_end)
    third_side = orientation(second_start, second_end, first_start)
    fourth_side = orientation(second_start, second_end, first_end)
    return first_side * second_side < 0 and third_side * fourth_side < 0


def count_edge_crossings(graph, positions):
    """Count crossings among edges that do not share a node."""
    edges = list(graph.edges())
    return sum(
        segments_cross(positions[a], positions[b], positions[c], positions[d])
        for index, (a, b) in enumerate(edges)
        for c, d in edges[index + 1 :]
        if not {a, b}.intersection((c, d))
    )


def save_node_link_diagram(image_path, graph, positions):
    """Render one node-link diagram to a PNG file."""
    figure, axis = plt.subplots(figsize=(9, 5), dpi=120)
    nx.draw_networkx(
        graph,
        positions,
        ax=axis,
        node_color="#72B7B2",
        edge_color="#4C4C4C",
        node_size=850,
        font_size=7,
        width=1.8,
    )
    axis.set_axis_off()
    figure.tight_layout()
    figure.savefig(image_path)
    plt.close(figure)


def generate_node_link_dataset(
    output_dir,
    num_figures=20,
    path_length=3,
    edge_crossings=1,
    split_ratios=(0.7, 0.15, 0.15),
    seed=42,
):
    """Generate controlled node-link figures and a JSONL manifest."""
    import random

    validate_node_settings(num_figures, path_length, edge_crossings)
    output_dir = Path(output_dir)
    image_dir = output_dir / "images"
    manifest_path = output_dir / "manifest.jsonl"
    if manifest_path.exists():
        raise FileExistsError(f"refusing to overwrite existing {manifest_path}")
    image_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Prepared node-link output directory: %s", output_dir)

    random_generator = random.Random(seed)
    assignments = create_split_assignments(num_figures, split_ratios, random_generator)
    entries = []
    for index, split_name in enumerate(assignments):
        figure_id = f"node_{index:06d}"
        image_path = image_dir / f"{figure_id}.png"
        if image_path.exists():
            raise FileExistsError(f"refusing to overwrite existing {image_path}")
        graph, positions, path_nodes = create_node_link_graph(
            path_length, edge_crossings, index
        )
        observed_crossings = count_edge_crossings(graph, positions)
        if observed_crossings != edge_crossings:
            raise ValueError(
                f"expected {edge_crossings} crossings, observed {observed_crossings}"
            )
        save_node_link_diagram(image_path, graph, positions)
        entries.append(
            {
                "edge_crossings": observed_crossings,
                "edges": [list(edge) for edge in graph.edges()],
                "figure_id": figure_id,
                "figure_type": "node_link",
                "image_path": str(image_path.relative_to(output_dir)),
                "path_length": path_length,
                "path_nodes": path_nodes,
                "questions": build_node_questions(path_nodes),
                "seed": seed,
                "split": split_name,
            }
        )

    logger.info("Rendered and validated %d node-link diagrams", len(entries))
    write_manifest(entries, manifest_path)
    logger.info("Wrote node-link manifest: %s", manifest_path)
    return entries
