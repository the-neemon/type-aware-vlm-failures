"""Tests for the third fabrication pilot: a year outside the chart's range."""

import json

import pytest

from src.synth import items
from src.synth import out_of_range as gen


@pytest.fixture(scope="module")
def generated(tmp_path_factory):
    out = tmp_path_factory.mktemp("range")
    return out, gen.generate(out, num_charts=40, seed=5)


def test_two_questions_per_chart_with_identical_wording(generated):
    _, entries = generated
    for e in entries:
        present, absent = e["questions"]
        assert (present["absent"], absent["absent"]) == (False, True)
        assert present["phrasing"] == absent["phrasing"]
        assert present["question"].replace(present["asks_about"], "Y") == \
            absent["question"].replace(absent["asks_about"], "Y")


def test_the_absent_year_lies_outside_the_shown_range(generated):
    _, entries = generated
    for e in entries:
        years = [int(c) for c in e["categories"]]
        absent = e["questions"][1]
        year = int(absent["asks_about"])
        assert year < years[0] or year > years[-1]
        gap = years[0] - year if absent["direction"] == "before" else year - years[-1]
        assert gap == absent["years_beyond"] in gen.YEARS_BEYOND


def test_trends_and_extrapolation(generated):
    _, entries = generated
    assert {e["trend"] for e in entries} == set(gen.TRENDS)
    rising = next(e for e in entries if e["trend"] == "rising")
    assert rising["values"] == sorted(rising["values"])
    assert gen.extrapolate([2010, 2011, 2012], [10, 20, 30], 2014) == 50.0
    assert gen.extrapolate([2010, 2011, 2012], [30, 20, 10], 2008) == 50.0


def test_charts_match_pilot_two_difficulty(generated):
    _, entries = generated
    assert all(v % 5 for e in entries for v in e["values"])


def test_verify_passes_then_catches_a_tampered_manifest(generated):
    out, _ = generated
    assert gen.verify(out) == []
    lines = (out / "manifest.jsonl").read_text().splitlines()
    entry = json.loads(lines[0])
    entry["questions"][1]["asks_about"] = entry["categories"][0]
    tampered = out.parent / "tampered"
    tampered.mkdir()
    (tampered / "images").symlink_to(out / "images")
    (tampered / "manifest.jsonl").write_text("\n".join([json.dumps(entry)] + lines[1:]) + "\n")
    assert gen.verify(tampered)


def test_generation_is_deterministic_and_refuses_to_overwrite(tmp_path, generated):
    out, entries = generated
    again = gen.generate(tmp_path / "again", num_charts=40, seed=5)
    strip = lambda es: [{k: v for k, v in e.items() if k != "image_path"} for e in es]
    assert strip(again) == strip(entries)
    with pytest.raises(FileExistsError):
        gen.generate(out, num_charts=40, seed=5)


def test_items_load_and_score_like_the_earlier_value_questions(generated):
    out, _ = generated
    loaded = items.load_manifest_items(out / "manifest.jsonl", ("train", "validation", "test"))
    assert len(loaded) == 80
    absent = next(it for it in loaded if it["absent"])
    assert absent["template"] == "absent_value" and absent["family"] == "value"
    assert {"direction", "years_beyond", "extrapolated"} <= set(absent)
    assert items.score(absent, "None")["outcome"] == "rejected"
    assert items.score(absent, "0")["outcome"] == "zero"
    row = items.score(absent, "37")
    assert row["outcome"] == "fabricated" and row["direction"] == absent["direction"]
    present = next(it for it in loaded if not it["absent"])
    assert items.score(present, present["gold"])["correct"]
