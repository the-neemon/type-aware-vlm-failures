"""Tests for the second fabrication pilot and the scoring of its absent questions."""

import pytest

from src.synth import absent_lookalikes as gen
from src.synth import items


@pytest.fixture(scope="module")
def generated(tmp_path_factory):
    out = tmp_path_factory.mktemp("look")
    return out, gen.generate(out, num_charts=40, seed=5)


def test_pairs_are_distinct_names_and_leave_room_for_three_unrelated_absent_names():
    for theme, pairs in gen.PAIRS.items():
        names = [n for pair in pairs for n in pair]
        assert len(names) == len(set(names)), theme
        assert len(pairs) >= max(gen.BAR_COUNTS) + 3, theme


def test_six_questions_and_the_lookalike_bookkeeping(generated):
    _, entries = generated
    for e in entries:
        qs = e["questions"]
        assert len(qs) == 6 and sum(q["absent"] for q in qs) == 3
        for q in qs:
            if q["absent"] and q["lookalike_of"]:
                partner = e["absent_partners"][e["categories"].index(q["lookalike_of"])]
                assert q["asks_about"] == partner
            if q["absent"]:
                assert q["asks_about"] not in e["categories"]
                assert q["asks_about"] in q["question"]


def test_charts_are_harder_than_pilot_one(generated):
    _, entries = generated
    assert all(v % 5 for e in entries for v in e["values"])
    assert {e["axis_max"] for e in entries} == set(gen.AXIS_MAXES)


def test_lookalike_and_unrelated_absent_names_are_both_common(generated):
    _, entries = generated
    flags = [q["lookalike_of"] is not None for e in entries for q in e["questions"] if q["absent"]]
    assert 0.3 < sum(flags) / len(flags) < 0.7


def test_verify_and_determinism(generated, tmp_path):
    out, entries = generated
    assert gen.verify(out) == []
    again = gen.generate(tmp_path / "again", num_charts=40, seed=5)
    assert [e["pixel_sha256"] for e in again] == [e["pixel_sha256"] for e in entries]
    assert [e["questions"] for e in again] == [e["questions"] for e in entries]


def test_loader_carries_the_fields_the_scorer_needs(generated):
    out, entries = generated
    loaded = items.load_manifest_items(out / "manifest.jsonl", ("train", "validation", "test"))
    assert len(loaded) == 6 * len(entries)
    compare = next(it for it in loaded if it["template"] == "absent_compare")
    assert compare["compared_with"] in compare["categories"]
    assert "lookalike_of" in compare


@pytest.mark.parametrize("prediction, outcome", [
    ("Guava", "fabricated"), ("guava.", "fabricated"), ("Apple", "picked_present"),
    ("Guava is not shown", "rejected"), ("None", "rejected"), ("Both", "unclear"),
])
def test_classify_compare(prediction, outcome):
    assert items.classify_compare(prediction, "Guava", "Apple") == outcome


@pytest.mark.parametrize("prediction, outcome", [
    ("Mango", "fabricated"), ("mango", "fabricated"), ("None", "rejected"),
    ("There is no Guava", "rejected"), ("Kiwi", "unclear"),
])
def test_classify_neighbor(prediction, outcome):
    assert items.classify_neighbor(prediction, ["Apple", "Mango"]) == outcome


def test_score_routes_each_absent_template_to_its_classifier():
    base = {"absent": True, "phrasing": "p", "gold": "not present",
            "categories": ["Apple", "Mango"]}
    compare = dict(base, template="absent_compare", asks_about="Guava", compared_with="Apple")
    neighbor = dict(base, template="absent_neighbor", asks_about="Guava")
    value = dict(base, template="absent_value", asks_about="Guava", lookalike_of=None)
    assert items.score(compare, "Apple")["outcome"] == "picked_present"
    assert items.score(neighbor, "Mango")["outcome"] == "fabricated"
    assert items.score(value, "0") | {} == {
        "template": "absent_value", "absent": True, "phrasing": "p", "asks_about": "Guava",
        "lookalike_of": None, "outcome": "zero", "correct": False}
