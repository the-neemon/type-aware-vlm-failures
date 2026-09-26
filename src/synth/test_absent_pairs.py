"""Tests for the fabrication-pilot generator and the absent-answer scorer."""

import json

import pytest

from src.synth import absent_pairs, items


@pytest.fixture(scope="module")
def generated(tmp_path_factory):
    out = tmp_path_factory.mktemp("absent")
    return out, absent_pairs.generate(out, num_charts=12, seed=7)


def test_each_chart_asks_one_present_and_one_absent_question_in_the_same_words(generated):
    _, entries = generated
    for e in entries:
        present, absent = e["questions"]
        assert present["asks_about"] in e["categories"]
        assert absent["asks_about"] not in e["categories"]
        assert absent["asks_about"] in absent_pairs.THEMES[e["theme"]]
        assert present["phrasing"] == absent["phrasing"]
        assert present["gold_answer"] == e["values"][e["categories"].index(present["asks_about"])]
        assert absent["gold_answer"] == "not present"


def test_charts_vary(generated):
    _, entries = generated
    assert len({e["theme"] for e in entries}) > 1
    assert len({len(e["categories"]) for e in entries}) > 1
    assert len({e["questions"][0]["question"] for e in entries}) == len(entries)


def test_generation_is_deterministic_and_verifiable(generated, tmp_path):
    out, entries = generated
    again = absent_pairs.generate(tmp_path / "again", num_charts=12, seed=7)
    assert [e["pixel_sha256"] for e in again] == [e["pixel_sha256"] for e in entries]
    assert absent_pairs.verify(out) == []


def test_verify_catches_a_changed_image(generated, tmp_path):
    from PIL import Image

    _, entries = generated
    copy = tmp_path / "copy"
    absent_pairs.generate(copy, num_charts=12, seed=7)
    target = copy / entries[0]["image_path"]
    with Image.open(target) as image:
        image = image.convert("RGB")
    image.putpixel((0, 0), (255, 0, 0))
    image.save(target)
    assert absent_pairs.verify(copy) == [f"{entries[0]['figure_id']}: pixels differ from the manifest"]


def test_refuses_to_overwrite(generated):
    out, _ = generated
    with pytest.raises(FileExistsError):
        absent_pairs.generate(out, num_charts=12, seed=7)


def test_loader_gives_two_items_per_chart_with_image_paths(generated):
    out, entries = generated
    loaded = items.load_manifest_items(out / "manifest.jsonl", ("train", "validation", "test"))
    assert len(loaded) == 2 * len(entries)
    assert all(it["image"].is_file() and it["source"] == "synthetic" for it in loaded)
    assert [it["absent"] for it in loaded[:2]] == [False, True]
    test_only = items.load_manifest_items(out / "manifest.jsonl", ("test",))
    assert len(test_only) == 2 * sum(e["split"] == "test" for e in entries)


@pytest.mark.parametrize("prediction, outcome", [
    ("35", "fabricated"), ("0", "zero"), ("0.0", "zero"), ("42.5%", "fabricated"),
    ("1,200.", "fabricated"),
    ("Not shown", "rejected"), ("None", "rejected"), ("N/A", "rejected"),
    ("There is no Guava bar", "rejected"), ("No data", "rejected"), ("Unknown.", "rejected"),
    ("Mango", "unclear"), ("45 (not shown)", "unclear"), ("", "unclear"),
])
def test_classify_absent(prediction, outcome):
    assert items.classify_absent(prediction) == outcome


def test_score_uses_relaxed_accuracy_for_present_questions():
    present = {"template": "read_value", "absent": False, "phrasing": "value_of",
               "asks_about": "Mango", "gold": "40"}
    assert items.score(present, "41")["correct"]          # within relaxed 5%
    assert not items.score(present, "30")["correct"]
    assert "outcome" not in items.score(present, "41")
    absent = dict(present, absent=True, gold="not present")
    assert items.score(absent, "Not present") == dict(
        template="read_value", absent=True, phrasing="value_of", asks_about="Mango",
        outcome="rejected", correct=True)
