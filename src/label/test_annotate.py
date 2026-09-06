"""Unit tests for the annotation tool and agreement calculations (TASKS P2.2, P2.4)."""

import io
import json
import pathlib
import pytest

from src.label.agreement import (
    compute_cohen_kappa,
    evaluate_against_ground_truth,
    format_evaluation_report,
)
from src.label.annotate import run_cli
from src.label.annotation import (
    ALLOWED_LABELS,
    KEY_MAP,
    AnnotationItem,
    AnnotationRecord,
    append_annotation,
    load_completed_ids,
    load_items,
    make_record,
)


def test_allowed_labels():
    assert set(ALLOWED_LABELS) == {"structural", "fabrication", "ambiguous"}
    assert KEY_MAP["s"] == "structural"
    assert KEY_MAP["f"] == "fabrication"
    assert KEY_MAP["a"] == "ambiguous"


def test_annotation_item_blinding_strips_model_and_labels(tmp_path):
    """Ensure that model names and ground truth labels are stripped on load."""
    input_file = tmp_path / "raw_errors.jsonl"
    raw_data = [
        {
            "item_id": "test_001",
            "model_name": "qwen2.5-vl-7b",
            "model": "qwen",
            "target_failure_type": "structural",
            "ground_truth": "structural",
            "previous_annotator_label": "fabrication",
            "figure_id": "chart_1.png",
            "question": "What is the value of bar A?",
            "gold_answer": "42",
            "model_answer": "55",
        }
    ]
    with open(input_file, "w", encoding="utf-8") as f:
        for item in raw_data:
            f.write(json.dumps(item) + "\n")

    items = load_items(input_file)
    assert len(items) == 1
    item = items[0]

    # Verify presented item attributes do NOT contain model or prior label
    assert not hasattr(item, "model_name")
    assert not hasattr(item, "model")
    assert not hasattr(item, "target_failure_type")
    assert not hasattr(item, "ground_truth")
    assert not hasattr(item, "previous_annotator_label")
    assert item.item_id == "test_001"
    assert item.question == "What is the value of bar A?"
    assert item.gold_answer == "42"
    assert item.model_answer == "55"


def test_immediate_disk_append(tmp_path):
    """Verify that every record is appended immediately to disk."""
    out_file = tmp_path / "annotations.jsonl"
    assert not out_file.exists()

    item1 = AnnotationItem("id_1", "img1.png", "Q1", "10", "20")
    record1 = make_record(item1, "structural", "Misread height", "Shrish")
    append_annotation(record1, out_file)

    assert out_file.exists()
    lines = out_file.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    rec1_disk = json.loads(lines[0])
    assert rec1_disk["item_id"] == "id_1"
    assert rec1_disk["label"] == "structural"
    assert rec1_disk["rationale"] == "Misread height"
    assert rec1_disk["annotator"] == "Shrish"

    # Append second record and verify it exists immediately
    item2 = AnnotationItem("id_2", "img2.png", "Q2", "not present", "100")
    record2 = make_record(item2, "fabrication", "Category does not exist", "Shrish")
    append_annotation(record2, out_file)

    lines = out_file.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    rec2_disk = json.loads(lines[1])
    assert rec2_disk["item_id"] == "id_2"
    assert rec2_disk["label"] == "fabrication"


def test_resume_skips_completed_items(tmp_path):
    """Verify completed items are identified and skipped."""
    out_file = tmp_path / "annotations.jsonl"
    item1 = AnnotationItem("id_1", "img1.png", "Q1", "10", "20")
    record1 = make_record(item1, "structural", "reason", "Shrish")
    append_annotation(record1, out_file)

    completed = load_completed_ids(out_file)
    assert completed == {"id_1"}

    # Simulate running with 2 items where id_1 is already done
    item2 = AnnotationItem("id_2", "img2.png", "Q2", "A", "B")
    items = [item1, item2]

    # In CLI runner, only item2 should be pending
    pending = [it for it in items if it.item_id not in completed]
    assert len(pending) == 1
    assert pending[0].item_id == "id_2"


def test_invalid_label_raises_error():
    item = AnnotationItem("id_1", "img.png", "Q", "1", "2")
    with pytest.raises(ValueError, match="Invalid label"):
        AnnotationRecord(
            item_id=item.item_id,
            question=item.question,
            gold_answer=item.gold_answer,
            model_answer=item.model_answer,
            label="hallucination",  # Invalid label
            rationale="test",
            annotator="Shrish",
            timestamp="2026-09-02T00:00:00Z",
        )


def test_compute_cohen_kappa():
    # Perfect agreement
    labels_a = ["structural", "fabrication", "ambiguous", "structural"]
    labels_b = ["structural", "fabrication", "ambiguous", "structural"]
    kappa = compute_cohen_kappa(labels_a, labels_b)
    assert kappa == 1.0

    # High agreement
    labels_a = ["structural"] * 90 + ["fabrication"] * 10
    labels_b = ["structural"] * 88 + ["fabrication"] * 12
    kappa_high = compute_cohen_kappa(labels_a, labels_b)
    assert kappa_high > 0.75

    # Empty inputs
    assert compute_cohen_kappa([], []) == 1.0


def test_evaluate_against_ground_truth():
    annotations = [
        {"item_id": "item1", "label": "structural", "question": "Q1", "gold_answer": "1", "model_answer": "2", "rationale": "r1"},
        {"item_id": "item2", "label": "fabrication", "question": "Q2", "gold_answer": "X", "model_answer": "Y", "rationale": "r2"},
        {"item_id": "item3", "label": "structural", "question": "Q3", "gold_answer": "5", "model_answer": "9", "rationale": "r3"},
    ]
    ground_truth = {
        "item1": "structural",
        "item2": "fabrication",
        "item3": "fabrication",  # Disagreement
    }

    report = evaluate_against_ground_truth(annotations, ground_truth)
    assert report["matched_items"] == 3
    assert report["agreement_rate"] == round(2 / 3, 4)
    assert len(report["disagreements"]) == 1
    assert report["disagreements"][0]["item_id"] == "item3"
    assert report["disagreements"][0]["annotator_label"] == "structural"
    assert report["disagreements"][0]["ground_truth"] == "fabrication"

    formatted = format_evaluation_report(report)
    assert "**Agreement rate**: 66.7%" in formatted
    assert "Disagreements (1 items)" in formatted


def test_run_cli_simulation(tmp_path, monkeypatch):
    """Simulate CLI input via stdin monkeypatching."""
    out_file = tmp_path / "cli_annotations.jsonl"
    items = [
        AnnotationItem("item_1", "fake.png", "What is A?", "10", "15"),
        AnnotationItem("item_2", "fake.png", "What is Z?", "not present", "99"),
    ]

    # Provide input for item 1 ('s', then rationale) and item 2 ('f', then rationale)
    simulated_input = "s\nHeight read error\nf\nAbsent category claimed\n"
    monkeypatch.setattr("sys.stdin", io.StringIO(simulated_input))

    count = run_cli(items, out_file, annotator="Shrish", open_image=False)
    assert count == 2
    assert out_file.exists()

    records = [json.loads(line) for line in out_file.read_text(encoding="utf-8").splitlines()]
    assert len(records) == 2
    assert records[0]["item_id"] == "item_1"
    assert records[0]["label"] == "structural"
    assert records[0]["rationale"] == "Height read error"
    assert records[1]["item_id"] == "item_2"
    assert records[1]["label"] == "fabrication"
    assert records[1]["rationale"] == "Absent category claimed"
