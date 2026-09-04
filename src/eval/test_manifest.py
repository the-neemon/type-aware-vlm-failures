"""Tests for the manifest schema (TASKS P1.6).

Run:
    python -m pytest src/eval/test_manifest.py -v
"""

import pathlib

import pytest

from src.eval.manifest import ManifestRow, read_manifest, validate_manifest, write_manifest


class TestManifestRow:
    def test_create(self):
        row = ManifestRow("fig.png", "What?", "42", "42", True)
        assert row.figure_id == "fig.png"
        assert row.correct is True

    def test_frozen(self):
        row = ManifestRow("fig.png", "What?", "42", "42", True)
        with pytest.raises(AttributeError):
            row.correct = False  # type: ignore

    def test_empty_figure_id_raises(self):
        with pytest.raises(ValueError, match="figure_id"):
            ManifestRow("", "What?", "42", "42", True)

    def test_empty_question_raises(self):
        with pytest.raises(ValueError, match="question"):
            ManifestRow("fig.png", "", "42", "42", True)

    def test_empty_gold_raises(self):
        with pytest.raises(ValueError, match="gold"):
            ManifestRow("fig.png", "What?", "", "42", True)

    def test_hashable(self):
        row = ManifestRow("fig.png", "What?", "42", "42", True)
        assert hash(row) is not None
        assert row in {row}


class TestWriteAndRead:
    def test_roundtrip(self, tmp_path):
        path = tmp_path / "manifest.jsonl"
        rows = [
            ManifestRow("fig1.png", "Q1", "42", "42", True),
            ManifestRow("fig2.png", "Q2", "yes", "no", False),
        ]
        write_manifest(rows, path)
        loaded = read_manifest(path)
        assert loaded == rows

    def test_creates_parent_dirs(self, tmp_path):
        path = tmp_path / "a" / "b" / "c" / "manifest.jsonl"
        write_manifest([ManifestRow("f.png", "Q", "1", "1", True)], path)
        assert path.exists()

    def test_unicode(self, tmp_path):
        path = tmp_path / "manifest.jsonl"
        row = ManifestRow("图.png", "Qué?", "42", "42", True)
        write_manifest([row], path)
        loaded = read_manifest(path)
        assert loaded[0].figure_id == "图.png"

    def test_empty_prediction_allowed(self, tmp_path):
        """Pre-inference manifests have empty predictions."""
        path = tmp_path / "manifest.jsonl"
        row = ManifestRow("f.png", "Q", "42", "", False)
        write_manifest([row], path)
        loaded = read_manifest(path)
        assert loaded[0].prediction == ""


class TestValidateManifest:
    def test_clean_manifest(self):
        rows = [
            ManifestRow("fig1.png", "Q1", "42", "42", True),
            ManifestRow("fig2.png", "Q2", "yes", "no", False),
        ]
        warnings = validate_manifest(rows)
        assert len(warnings) == 0

    def test_duplicate_detected(self):
        rows = [
            ManifestRow("fig1.png", "Q1", "42", "42", True),
            ManifestRow("fig1.png", "Q1", "42", "43", False),
        ]
        warnings = validate_manifest(rows)
        assert any("duplicate" in w for w in warnings)

    def test_empty_prediction_warning(self):
        rows = [
            ManifestRow("fig1.png", "Q1", "42", "", False),
        ]
        warnings = validate_manifest(rows)
        assert any("empty predictions" in w for w in warnings)
