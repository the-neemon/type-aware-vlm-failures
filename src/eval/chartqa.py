"""ChartQA dataset loader, split statistics, and figure-level split assertion.

Covers three Day 1 deliverables:
  - ChartQA ingest and figure-vs-question counts per split.
  - P5.4 assertion: no figure_id appears in more than one split.
  - Pre-inference manifest generation from raw ChartQA data.

ChartQA layout (as downloaded from HuggingFace ahmed-masry/ChartQA):
    data_root/
      train/
        train_human.json       [{"imgname": "...", "query": "...", "label": "..."}, ...]
        train_augmented.json
      val/
        val_human.json
        val_augmented.json
      test/
        test_human.json
        test_augmented.json
      png/                     all chart images live here (flat)

Usage:
    # Print split summary:
    python -m src.eval.chartqa path/to/ChartQA

    # As library:
    >>> from src.eval.chartqa import load_chartqa, print_split_summary
    >>> data = load_chartqa("path/to/ChartQA")
    >>> print_split_summary(data)
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
from collections import defaultdict
from typing import TypedDict


class ChartQAItem(TypedDict):
    """One question-answer pair from ChartQA."""
    figure_id: str
    question: str
    gold: str
    split: str        # "train", "val", or "test"
    source: str       # "human" or "augmented"


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def _load_json_split(
    data_root: pathlib.Path,
    split: str,
    source: str,
) -> list[ChartQAItem]:
    """Load a single ChartQA JSON file and normalise its fields.

    Parameters
    ----------
    data_root : Path
        Root directory of the ChartQA dataset.
    split : str
        One of "train", "val", "test".
    source : str
        One of "human", "augmented".

    Returns
    -------
    list[ChartQAItem]
    """
    path = data_root / split / f"{split}_{source}.json"
    if not path.exists():
        return []

    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    items: list[ChartQAItem] = []
    for entry in raw:
        items.append(ChartQAItem(
            figure_id=entry["imgname"],
            question=entry["query"],
            gold=str(entry["label"]),
            split=split,
            source=source,
        ))
    return items


def load_chartqa(
    data_root: str | pathlib.Path,
    splits: tuple[str, ...] = ("train", "val", "test"),
    sources: tuple[str, ...] = ("human", "augmented"),
) -> dict[str, list[ChartQAItem]]:
    """Load all requested ChartQA splits.

    Returns
    -------
    dict mapping ``"{split}_{source}"`` (e.g. ``"test_human"``) to a list of
    :class:`ChartQAItem` dicts.
    """
    data_root = pathlib.Path(data_root)
    result: dict[str, list[ChartQAItem]] = {}
    for split in splits:
        for source in sources:
            key = f"{split}_{source}"
            items = _load_json_split(data_root, split, source)
            if items:
                result[key] = items
    return result


# ---------------------------------------------------------------------------
# Split statistics
# ---------------------------------------------------------------------------

def split_stats(items: list[ChartQAItem]) -> dict[str, int]:
    """Count figures and questions in a flat list of items.

    Returns
    -------
    dict with keys "questions" and "figures".
    """
    figures = {item["figure_id"] for item in items}
    return {"questions": len(items), "figures": len(figures)}


def print_split_summary(data: dict[str, list[ChartQAItem]]) -> None:
    """Print a table of figure-vs-question counts per split.

    This is the deliverable: "figure-versus-question counts for every split"
    (TASKS.md Section 1.1, Shrish's row).
    """
    print()
    print(f"{'Split':<22s}  {'Questions':>10s}  {'Figures':>8s}  {'Q/Fig':>6s}")
    print("-" * 52)

    total_q = 0
    total_f_set: set[str] = set()

    for key in sorted(data.keys()):
        items = data[key]
        stats = split_stats(items)
        q = stats["questions"]
        f = stats["figures"]
        ratio = q / f if f > 0 else 0
        print(f"{key:<22s}  {q:>10d}  {f:>8d}  {ratio:>6.2f}")
        total_q += q
        total_f_set.update(item["figure_id"] for item in items)

    print("-" * 52)
    print(f"{'TOTAL':<22s}  {total_q:>10d}  {len(total_f_set):>8d}  "
          f"{total_q / len(total_f_set) if total_f_set else 0:>6.2f}")
    print()


# ---------------------------------------------------------------------------
# P5.4 assertion: figure-level split integrity
# ---------------------------------------------------------------------------

def assert_figure_level_splits(
    data: dict[str, list[ChartQAItem]],
    split_groups: tuple[tuple[str, ...], ...] | None = None,
) -> None:
    """Assert that no figure_id appears in more than one split group.

    By default, groups are (train*, val*, test*) — i.e. all keys starting with
    "train" form one group, "val" another, "test" the third.  Within a group
    (e.g. train_human + train_augmented), overlap is expected and allowed.

    Parameters
    ----------
    data : dict
        Output of :func:`load_chartqa`.
    split_groups : tuple of tuples of str, optional
        Custom grouping.  Each inner tuple lists the split keys that belong
        together.  Default: group by the prefix before the first underscore.

    Raises
    ------
    AssertionError
        If any figure appears in more than one group, with a diagnostic message
        listing the offending figures.
    """
    # Build groups.
    if split_groups is None:
        # Group by prefix: "train_human" → "train", "test_augmented" → "test".
        groups: dict[str, set[str]] = defaultdict(set)
        for key, items in data.items():
            prefix = key.split("_")[0]
            groups[prefix].update(item["figure_id"] for item in items)
    else:
        groups = {}
        for group_keys in split_groups:
            group_name = "+".join(group_keys)
            figures: set[str] = set()
            for key in group_keys:
                if key in data:
                    figures.update(item["figure_id"] for item in data[key])
            groups[group_name] = figures

    # Check pairwise disjointness.
    group_names = list(groups.keys())
    violations: list[str] = []
    for i, name_a in enumerate(group_names):
        for name_b in group_names[i + 1:]:
            overlap = groups[name_a] & groups[name_b]
            if overlap:
                sample = sorted(overlap)[:5]
                violations.append(
                    f"  {name_a} ∩ {name_b}: {len(overlap)} shared figures "
                    f"(e.g. {', '.join(sample)})"
                )

    if violations:
        msg = (
            "P5.4 VIOLATED: figure_id(s) appear in multiple splits.\n"
            "ChartQA has multiple questions per figure.  Splitting by question\n"
            "leaks the figure across train and test and inflates every probe.\n"
            + "\n".join(violations)
        )
        raise AssertionError(msg)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="ChartQA split statistics and figure-level split check."
    )
    parser.add_argument(
        "data_root",
        type=str,
        help="Path to the ChartQA dataset root directory.",
    )
    parser.add_argument(
        "--assert-splits",
        action="store_true",
        default=True,
        help="Run the P5.4 figure-level split assertion (default: True).",
    )
    args = parser.parse_args()

    data = load_chartqa(args.data_root)

    if not data:
        print(f"ERROR: no ChartQA JSON files found under {args.data_root}",
              file=sys.stderr)
        sys.exit(1)

    print_split_summary(data)

    if args.assert_splits:
        try:
            assert_figure_level_splits(data)
            print("✓ P5.4 check passed: no figure_id appears in multiple splits.")
        except AssertionError as exc:
            print(f"✗ {exc}", file=sys.stderr)
            sys.exit(1)


if __name__ == "__main__":
    main()
