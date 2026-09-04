"""Tests for ChartQA loader and the P5.4 split assertion.

Uses synthetic fixtures (tiny JSON files) rather than requiring the full
ChartQA download.

Run:
    python -m pytest src/eval/test_chartqa.py -v
"""

import json
import pathlib
import tempfile

import pytest

from src.eval.chartqa import (
    assert_figure_level_splits,
    load_chartqa,
    split_stats,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _write_chartqa_json(
    root: pathlib.Path,
    split: str,
    source: str,
    entries: list[dict],
) -> None:
    """Write a fake ChartQA JSON file at the expected path."""
    split_dir = root / split
    split_dir.mkdir(parents=True, exist_ok=True)
    path = split_dir / f"{split}_{source}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(entries, f)


@pytest.fixture
def clean_chartqa(tmp_path):
    """A clean ChartQA dataset with NO figure overlap across splits."""
    _write_chartqa_json(tmp_path, "train", "human", [
        {"imgname": "fig_A.png", "query": "Q1 about A", "label": "42"},
        {"imgname": "fig_A.png", "query": "Q2 about A", "label": "yes"},
        {"imgname": "fig_B.png", "query": "Q1 about B", "label": "100"},
    ])
    _write_chartqa_json(tmp_path, "train", "augmented", [
        {"imgname": "fig_C.png", "query": "Q1 about C", "label": "7"},
    ])
    _write_chartqa_json(tmp_path, "test", "human", [
        {"imgname": "fig_D.png", "query": "Q1 about D", "label": "no"},
        {"imgname": "fig_E.png", "query": "Q1 about E", "label": "50"},
    ])
    _write_chartqa_json(tmp_path, "test", "augmented", [
        {"imgname": "fig_F.png", "query": "Q1 about F", "label": "200"},
    ])
    return tmp_path


@pytest.fixture
def leaky_chartqa(tmp_path):
    """A ChartQA dataset where fig_A leaks across train and test."""
    _write_chartqa_json(tmp_path, "train", "human", [
        {"imgname": "fig_A.png", "query": "Q1 about A", "label": "42"},
    ])
    _write_chartqa_json(tmp_path, "test", "human", [
        {"imgname": "fig_A.png", "query": "Q2 about A", "label": "43"},
    ])
    return tmp_path


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

class TestLoadChartQA:
    def test_loads_all_splits(self, clean_chartqa):
        data = load_chartqa(clean_chartqa)
        assert "train_human" in data
        assert "train_augmented" in data
        assert "test_human" in data
        assert "test_augmented" in data

    def test_correct_item_count(self, clean_chartqa):
        data = load_chartqa(clean_chartqa)
        assert len(data["train_human"]) == 3
        assert len(data["train_augmented"]) == 1
        assert len(data["test_human"]) == 2
        assert len(data["test_augmented"]) == 1

    def test_fields_populated(self, clean_chartqa):
        data = load_chartqa(clean_chartqa)
        item = data["train_human"][0]
        assert item["figure_id"] == "fig_A.png"
        assert item["question"] == "Q1 about A"
        assert item["gold"] == "42"
        assert item["split"] == "train"
        assert item["source"] == "human"

    def test_missing_split_skipped(self, clean_chartqa):
        data = load_chartqa(clean_chartqa, splits=("val",))
        assert len(data) == 0

    def test_gold_is_string(self, clean_chartqa):
        """Gold answers must be strings even if the JSON has a number."""
        data = load_chartqa(clean_chartqa)
        for key in data:
            for item in data[key]:
                assert isinstance(item["gold"], str)


# ---------------------------------------------------------------------------
# Split statistics
# ---------------------------------------------------------------------------

class TestSplitStats:
    def test_figure_count_deduplicates(self, clean_chartqa):
        data = load_chartqa(clean_chartqa)
        stats = split_stats(data["train_human"])
        # fig_A appears twice, fig_B once → 2 unique figures
        assert stats["figures"] == 2
        assert stats["questions"] == 3

    def test_single_item(self, clean_chartqa):
        data = load_chartqa(clean_chartqa)
        stats = split_stats(data["train_augmented"])
        assert stats["figures"] == 1
        assert stats["questions"] == 1


# ---------------------------------------------------------------------------
# P5.4: figure-level split assertion
# ---------------------------------------------------------------------------

class TestAssertFigureLevelSplits:
    def test_clean_data_passes(self, clean_chartqa):
        data = load_chartqa(clean_chartqa)
        # Should not raise
        assert_figure_level_splits(data)

    def test_leaky_data_fails(self, leaky_chartqa):
        data = load_chartqa(leaky_chartqa)
        with pytest.raises(AssertionError, match="P5.4 VIOLATED"):
            assert_figure_level_splits(data)

    def test_leaky_data_mentions_figure(self, leaky_chartqa):
        data = load_chartqa(leaky_chartqa)
        with pytest.raises(AssertionError, match="fig_A.png"):
            assert_figure_level_splits(data)

    def test_overlap_within_same_split_is_ok(self, clean_chartqa):
        """fig_A in train_human appears twice — that's fine, same split."""
        data = load_chartqa(clean_chartqa)
        # Just confirm no assertion error
        assert_figure_level_splits(data)
