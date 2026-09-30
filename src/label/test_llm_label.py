"""Tests for the LLM annotator's batches and the checked append."""

import json

import pytest

from src.label import llm_label

BATCH = [{"item_id": "a"}, {"item_id": "b"}]


def lab(iid, label, reason=None, rationale="Chart shows 5; model said 6."):
    return {"item_id": iid, "label": label, "reason": reason, "rationale": rationale}


def test_a_well_formed_batch_passes():
    assert llm_label.check([lab("a", "structural"), lab("b", "not_an_error", "gold_error")], BATCH) == []


@pytest.mark.parametrize("bad, message", [
    (lab("b", "misread"), "unknown label"),
    (lab("b", "ambiguous"), "needs a reason"),
    (lab("b", "not_an_error", "question_ambiguous"), "needs a reason"),
    (lab("b", "computation", "other"), "takes no reason"),
    (lab("b", "structural", rationale=" "), "empty rationale"),
])
def test_rubric_violations_are_caught(bad, message):
    assert any(message in p for p in llm_label.check([lab("a", "structural"), bad], BATCH))


def test_item_ids_must_match_the_batch():
    problems = llm_label.check([lab("a", "structural"), lab("z", "structural")], BATCH)
    assert any("1 missing, 1 unexpected" in p for p in problems)


def test_append_keeps_batch_order_stamps_and_refuses_repeats(tmp_path):
    out = tmp_path / "labels.jsonl"
    llm_label.append([lab("b", "computation"), lab("a", "structural")], BATCH, out, "annotator-x")
    rows = [json.loads(l) for l in out.read_text().splitlines()]
    assert [r["item_id"] for r in rows] == ["a", "b"]
    assert all(r["annotator"] == "annotator-x" and r["timestamp"] for r in rows)
    with pytest.raises(SystemExit, match="already labelled"):
        llm_label.append([lab("a", "structural"), lab("b", "computation")], BATCH, out, "annotator-x")


def test_a_bad_batch_is_not_written(tmp_path):
    out = tmp_path / "labels.jsonl"
    with pytest.raises(SystemExit, match="not appended"):
        llm_label.append([lab("a", "structural")], BATCH, out, "x")
    assert not out.exists()


def test_corrections_append_after_the_original_and_need_a_labelled_item(tmp_path):
    out = tmp_path / "labels.jsonl"
    llm_label.append([lab("a", "structural"), lab("b", "computation")], BATCH, out, "x")
    llm_label.correct([lab("a", "not_an_error", "format_equivalent")], out, "audit")
    rows = [json.loads(l) for l in out.read_text().splitlines()]
    assert [r["item_id"] for r in rows] == ["a", "b", "a"]
    assert rows[-1]["label"] == "not_an_error" and rows[-1]["annotator"] == "audit"
    with pytest.raises(SystemExit, match="not labelled yet"):
        llm_label.correct([lab("z", "structural")], out, "audit")
    with pytest.raises(SystemExit, match="needs a reason"):
        llm_label.correct([lab("a", "ambiguous")], out, "audit")
