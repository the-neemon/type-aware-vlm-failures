"""e5_scores on the planted signal of test_run_synth_types: misreads, made-up
numbers and absence each on their own direction at layer 1."""

import json
import sys

import numpy as np

from src.controller import e5_scores
from src.probes.test_run_synth_types import _pilot


def test_classes_include_correct_refusals():
    base = {"template": "absent_value", "absent": True}
    assert e5_scores.item_class({**base, "outcome": "rejected"}) == "R"
    assert e5_scores.item_class({**base, "outcome": "zero"}) == "Z"
    assert e5_scores.item_class({"template": "read_value", "absent": False,
                                 "correct": False}) == "S"
    assert e5_scores.item_class({"template": "absent_compare", "absent": True,
                                 "outcome": "fabricated"}) is None


def test_probe_labels_cover_the_right_classes():
    cls = np.array(list("CSFZR"))
    assert list(e5_scores.probe_labels(cls, "binary")) == list("NPPPN")
    assert list(e5_scores.probe_labels(cls, "missing")) == list("NNPPP")
    assert list(e5_scores.probe_labels(cls, "misread")) == ["N", "P", "-", "-", "-"]


def test_end_to_end_scores_every_item_out_of_fold(tmp_path, monkeypatch):
    rng = np.random.default_rng(0)
    pilots = [_pilot(tmp_path, "pa", 120, True, rng), _pilot(tmp_path, "pb", 120, True, rng)]
    argv = ["e5_scores", "--layers", "0", "1", "--l2", "1", "100", "--jobs", "1",
            "--out", str(tmp_path / "out" / "m_scores")]
    for p in pilots:
        argv += ["--pilot", *p]
    monkeypatch.setattr(sys, "argv", argv)
    e5_scores.main()

    rows = [json.loads(l) for l in (tmp_path / "out" / "m_scores.jsonl").read_text().splitlines()]
    assert len(rows) == 480 and all(np.isfinite(r[p]) for r in rows for p in e5_scores.PROBES)
    report = json.loads((tmp_path / "out" / "m_scores.json").read_text())
    for p in ("missing", "misread"):
        assert report["probes"][p]["auroc"] > 0.85, p
        assert all(c["layer"] == 1 for c in report["probes"][p]["chosen_per_fold"].values())
