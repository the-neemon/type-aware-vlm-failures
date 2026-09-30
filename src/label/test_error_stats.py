import pytest

from src.label.error_stats import rescaled_match, stats


@pytest.mark.parametrize("gold,pred,expected", [
    ("43", "0.43", True), ("0.6", "60", True), ("47.6", "0.48", True),
    ("43", "0.5", False), ("43", "43", False), ("yes", "0.43", False), ("43", "no", False),
])
def test_rescaled_match(gold, pred, expected):
    assert rescaled_match(gold, pred) is expected


def _row(iid, source, correct, gold="1", pred="1"):
    return {"item_id": iid, "source": source, "correct": correct, "gold": gold, "prediction": pred}


def test_counts_by_source():
    preds = [_row("a", "human", False, "43", "0.43"), _row("b", "human", False),
             _row("c", "augmented", False), _row("d", "augmented", True)]
    labels = [{"item_id": "a", "label": "not_an_error", "reason": "format_equivalent"},
              {"item_id": "b", "label": "ambiguous", "reason": "gold_error"},
              {"item_id": "c", "label": "structural"}]
    s = stats(preds, labels)
    assert s["by_source"]["human"]["errors"] == 2 and s["by_source"]["augmented"]["real_errors"] == 1
    assert s["gold_wrong"] == 1 and s["rescaled"]["count"] == 1
    assert s["accuracy"] == {"relaxed": 0.25, "rescaled_counted_correct": 0.5,
                             "not_an_error_counted_correct": 0.5}


def test_labels_must_cover_the_errors():
    with pytest.raises(SystemExit, match="exactly"):
        stats([_row("a", "human", False), _row("b", "human", False)],
              [{"item_id": "a", "label": "structural"}])


def test_a_later_label_line_replaces_an_earlier_one():
    preds = [_row("a", "human", False), _row("b", "human", False)]
    labels = [{"item_id": "a", "label": "structural"}, {"item_id": "b", "label": "computation"},
              {"item_id": "a", "label": "not_an_error", "reason": "gold_error"}]
    s = stats(preds, labels)
    assert s["by_source"]["human"]["errors"] == 2 and s["gold_wrong"] == 1
    assert s["by_source"]["human"]["real_errors"] == 1
