"""Tests for the synthetic-pilot probe contrasts."""

import numpy as np

from src.probes import run_synth


def _rows(n_figs=60):
    rows = {}
    for f in range(n_figs):
        for t, absent, outcome, correct in (("read_value", False, None, f % 7 != 0),
                                            ("absent_value", True,
                                             "fabricated" if f % 3 == 0 else "zero", False)):
            iid = f"{t}-{f}"
            rows[iid] = {"item_id": iid, "figure_id": f"fig{f}", "template": t,
                         "absent": absent, "outcome": outcome, "correct": correct,
                         "lookalike_of": "X" if f % 2 else None, "asks_about": f"n{f % 5}",
                         "phrasing": "value_of"}
    return rows


def test_contrasts_select_the_right_rows_and_share_figure_splits():
    preds = _rows()
    item_ids = list(preds)
    figure_ids = [preds[i]["figure_id"] for i in item_ids]
    index = run_synth.build_index(preds, item_ids, figure_ids)

    absent = index["absent_value_vs_present"]
    assert len(absent["idx"]) == 120 and absent["y"].sum() == 60
    fab = index["value_fab_vs_zero"]
    assert fab["y"].sum() == 20 and len(fab["idx"]) == 60
    look = index["value_fab_vs_zero_lookalike"]
    assert all(r["lookalike_of"] is not None for r in look["rows"])
    assert index["compare_fab_vs_present"]["idx"].size == 0

    # one figure, one split, in every contrast
    split_of = {}
    for c in index.values():
        for f, s in zip(c["figs"], c["split"]):
            assert split_of.setdefault(f, s) == s


def test_row_indices_point_at_the_right_cache_rows():
    preds = _rows()
    item_ids = list(reversed(preds))                  # cache order differs from file order
    figure_ids = [preds[i]["figure_id"] for i in item_ids]
    c = run_synth.build_index(preds, item_ids, figure_ids)["read_misread"]
    assert [item_ids[i] for i in c["idx"]] == [r["item_id"] for r in c["rows"]]


def test_text_baseline_is_near_chance_when_names_carry_no_signal():
    preds = _rows(200)
    item_ids = list(preds)
    c = run_synth.build_index(preds, item_ids, [preds[i]["figure_id"] for i in item_ids])
    auc = run_synth.text_baseline(c["absent_value_vs_present"], (1.0, 100.0))
    assert 0.35 < auc < 0.65
    assert np.isnan(run_synth.auroc(np.ones(3), np.arange(3.0)))
