"""Tests for the E4 repairs and runner. No GPU: the model is a scripted stand-in."""

from __future__ import annotations

import json
import pathlib

import numpy as np
import pytest
from PIL import Image

from src.intervene.repairs import (
    ANSWER_SUFFIX, CROP, REASK, UPSAMPLE, VERIFY, VERIFY_SUFFIX, Figure, build_query,
    parse_verified, question_crop, question_targets, upscale_to_budget,
)
from src.intervene.run_e4 import figure_loader, read_jsonl, run, select_items, summarize
from src.synth.bar_charts import generate_bar_chart_dataset

BAR_RGB = np.array([76, 120, 168])


@pytest.fixture(scope="module")
def synth(tmp_path_factory):
    d = tmp_path_factory.mktemp("synth")
    generate_bar_chart_dataset(d, num_charts=6)
    return d, read_jsonl(d / "manifest.jsonl")


def _figure(d, entry):
    return Figure(Image.open(d / entry["image_path"]), tuple(entry["categories"]),
                  tuple(map(tuple, entry["bar_boxes"])), tuple(entry["plot_box"]))


def _bar_pixels(img):
    a = np.asarray(img.convert("RGB")).astype(int)
    return (np.abs(a - BAR_RGB).sum(-1) < 40)


class TestUpscale:
    def test_fills_budget_without_exceeding_it(self):
        out = upscale_to_budget(Image.new("RGB", (800, 557)))
        assert 0.95e6 < out.width * out.height <= 1_000_000

    def test_never_shrinks(self):
        assert upscale_to_budget(Image.new("RGB", (2000, 1000))).size == (2000, 1000)

    def test_scale_is_capped(self):
        assert upscale_to_budget(Image.new("RGB", (50, 50))).size == (200, 200)


class TestTargets:
    CATS = ("A", "B", "C", "D", "E")

    def test_single(self):
        assert question_targets("What is the value of category A?", self.CATS) == [0]

    def test_pair(self):
        assert question_targets("Which category is larger, B or D?", self.CATS) == [1, 3]

    def test_absent_category(self):
        assert question_targets("What is the value of category F?", self.CATS) == []

    def test_letter_inside_a_word_does_not_match(self):
        assert question_targets("Are Bars Equal?", self.CATS) == []


class TestCrop:
    def test_keeps_axis_strip_and_only_target_bar(self, synth):
        d, man = synth
        entry = next(e for e in man if min(e["values"]) > 0)
        fig = _figure(d, entry)
        crop = question_crop(fig, [0])
        assert crop.height == fig.image.height and crop.width < fig.image.width / 2
        # the y-axis strip is unchanged, pixel for pixel
        strip_w = fig.plot_box[0] + 2
        assert (np.asarray(crop)[:, :strip_w] == np.asarray(fig.image)[:, :strip_w]).all()
        # exactly one bar's worth of bar-coloured columns survives
        cols = _bar_pixels(crop).any(0)
        l, t, r, b = fig.bar_boxes[0]
        assert abs(cols.sum() - (r - l)) <= 3

    def test_zero_height_bar_is_still_croppable(self, synth):
        # the reason geometry is recorded at render time: a zero bar leaves no
        # pixels for detection to find, but its box is still known
        d, man = synth
        for e in man:
            if 0 in e["values"]:
                i = e["values"].index(0)
                crop = question_crop(_figure(d, e), [i])
                assert crop.width > e["plot_box"][0]
                return
        pytest.skip("no zero-valued bar in this sample")


class TestParseVerified:
    def test_answer_line(self):
        assert parse_verified("The bar for A reaches 35.\nAnswer: 35") == ("35", True)

    def test_markdown_bold(self):
        assert parse_verified("Bar A.\n**Answer:** 35") == ("35", True)

    def test_last_answer_line_wins(self):
        assert parse_verified("Answer: 30\nActually\nAnswer: 35") == ("35", True)

    def test_unparseable_is_flagged(self):
        assert parse_verified("It is 35.") == ("It is 35.", False)


class TestBuildQuery:
    def test_reask_is_sampled_with_the_original_prompt(self, synth):
        d, man = synth
        q = build_query(REASK, "Q?", _figure(d, man[0]))
        assert q.sample and q.prompt == "Q?" + ANSWER_SUFFIX

    def test_verify_prompt_and_budget(self, synth):
        d, man = synth
        q = build_query(VERIFY, "Q?", _figure(d, man[0]))
        assert q.prompt == "Q?" + VERIFY_SUFFIX and q.max_new_tokens > 32 and not q.sample

    def test_crop_needs_geometry(self):
        assert build_query(CROP, "category A?", Figure(Image.new("RGB", (10, 10)))) is None

    def test_crop_on_absent_category_shows_whole_figure(self, synth):
        d, man = synth
        fig = _figure(d, man[0])
        q = build_query(CROP, "What is the value of category Z?", fig)
        assert "absent" in q.note
        assert q.image.size == upscale_to_budget(fig.image).size

    def test_upsample_is_bigger(self, synth):
        d, man = synth
        fig = _figure(d, man[0])
        q = build_query(UPSAMPLE, "Q?", fig)
        assert q.image.width > fig.image.width

    def test_baseline_is_not_requeried(self, synth):
        d, man = synth
        with pytest.raises(ValueError):
            build_query("I_0", "Q?", _figure(d, man[0]))


def _rows(man):
    """Two wrong answers per chart: one structural, one fabrication, per the templates."""
    preds, labels = [], []
    for e in man:
        for q in e["questions"]:
            if q["template"] not in ("read_value", "absent_category"):
                continue
            iid = f"{e['figure_id']}:{q['template']}"
            preds.append({"item_id": iid, "model": "fake", "figure_id": e["figure_id"],
                          # prefixed because every chart asks the same template
                          # text, and the scripted model below looks answers up by
                          # question; the runner itself keys by item_id
                          "question": f"[{e['figure_id']}] {q['question']}",
                          "gold": str(q["gold_answer"]),
                          "prediction": "12345", "correct": False, "split": "test",
                          "source": "synthetic"})
            labels.append({"item_id": iid, "label": q["target_failure_type"]})
    return preds, labels


class Scripted:
    """Recovers structural errors under every repair and fabrications under none."""

    def __init__(self, preds, labels):
        kind = {l["item_id"]: l["label"] for l in labels}
        self.by_q = {p["question"]: (p["gold"], kind[p["item_id"]]) for p in preds}
        self.calls = 0

    def __call__(self, q):
        self.calls += 1
        base = q.prompt.split("\n")[0]
        gold, kind = self.by_q[base]
        ans = gold if kind == "structural" else "999"
        return f"The relevant bar.\nAnswer: {ans}" if VERIFY_SUFFIX in q.prompt else ans


class TestRunner:
    def test_select_items(self):
        preds = [{"item_id": i, "correct": c} for i, c in
                 [("a", False), ("b", False), ("c", False), ("d", True)]]
        labels = [{"item_id": "a", "label": "structural"},
                  {"item_id": "b", "label": "ambiguous"}]
        items, counts = select_items(preds, labels)
        assert [i["item_id"] for i in items] == ["a"]
        assert counts == {"incorrect": 3, "used": 1, "ambiguous": 1, "unlabelled": 1,
                          "structural": 1, "fabrication": 0}

    def test_end_to_end_resume_and_summary(self, synth, tmp_path):
        d, man = synth
        preds, labels = _rows(man)
        items, _ = select_items(preds, labels)
        out = tmp_path / "e4.jsonl"
        fake = Scripted(preds, labels)
        run(items, figure_loader(d, d / "manifest.jsonl"), fake, out, log=lambda s: None)

        rows = read_jsonl(out)
        assert len(rows) == len(items) * 5                      # I_0 + four repairs
        assert fake.calls == len(items) * 4
        for r in rows:
            expect = r["intervention"] != "I_0" and r["failure_type"] == "structural"
            assert r["recovered"] is expect

        # resume: nothing left to do, so the model is never called
        def boom(q):
            raise AssertionError("resumed run re-queried the model")
        run(items, figure_loader(d, d / "manifest.jsonl"), boom, out, log=lambda s: None)
        assert len(read_jsonl(out)) == len(rows)

        text = summarize(out, n_boot=200)
        assert "I_crop vs I_0: delta = +1.000" in text and "excludes 0" in text

    def test_chartqa_item_without_geometry_skips_crop(self, tmp_path):
        root = tmp_path / "chartqa"
        (root / "test" / "png").mkdir(parents=True)
        Image.new("RGB", (80, 60), "white").save(root / "test" / "png" / "x.png")
        item = {"item_id": "i1", "figure_id": "x.png", "split": "test", "question": "Q?",
                "gold": "5", "prediction": "7", "failure_type": "structural"}
        out = tmp_path / "e4.jsonl"
        run([item], figure_loader(root, None), lambda q: "5", out, log=lambda s: None)
        crop = [r for r in read_jsonl(out) if r["intervention"] == CROP]
        assert crop == [{"item_id": "i1", "figure_id": "x.png", "failure_type": "structural",
                         "gold": "5", "intervention": CROP, "applicable": False}]
