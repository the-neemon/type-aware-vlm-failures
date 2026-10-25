"""Tests for the human agreement report."""

from __future__ import annotations

from src.label.human_agreement import compare, consensus_labels, report


def _ratings(a, b, adjudicated=None):
    rec = lambda lab: {"label": lab, "question": "q", "gold_answer": "1", "model_answer": "2"}
    r = {"shrish": {i: rec(l) for i, l in a.items()}, "sanjith": {i: rec(l) for i, l in b.items()}}
    if adjudicated:
        r["adjudicated"] = {i: {"label": l} for i, l in adjudicated.items()}
    return r


def test_compare_perfect_and_partial():
    x = {"i1": "structural", "i2": "computation", "i3": "ambiguous"}
    assert compare(x, x, list(x))["kappa"] == 1.0
    assert compare(x, {"i1": "structural"}, list(x))["n"] == 1


def test_consensus_keeps_agreed_and_adjudicated():
    r = _ratings({"i1": "structural", "i2": "computation"},
                 {"i1": "structural", "i2": "fabrication"}, adjudicated={"i2": "fabrication"})
    assert consensus_labels(r, ["shrish", "sanjith"]) == {"i1": "structural", "i2": "fabrication"}


def test_report_splits_by_model_and_lists_disagreements():
    a = {"q1": "structural", "q2": "computation", "l1": "structural", "l2": "not_an_error"}
    b = {"q1": "structural", "q2": "computation", "l1": "fabrication", "l2": "not_an_error"}
    key = [{"item_id": i, "model": "qwen2_5_vl_7b" if i[0] == "q" else "llava_next_mistral_7b"}
           for i in a]
    text = report(_ratings(a, b), ["shrish", "sanjith"], key, dict(a), "cmd")
    assert "| qwen2_5_vl_7b | 2 | 1.000 |" in text
    assert "## 3. Disagreements (1, 1 unresolved)" in text
    assert "Incomplete" not in text
