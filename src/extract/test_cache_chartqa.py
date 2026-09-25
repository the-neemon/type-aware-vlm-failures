"""Tests for the caching entry point: config checks, and `run` end to end on a fake model.

The fake model is called the way `generate()` calls the real one (a prefill,
then decode steps), so these tests exercise the hooks, the shards, resuming,
assembly and validation together without a GPU.
"""

import json
from dataclasses import replace
from types import SimpleNamespace

import pytest

from src.extract import cache_chartqa, cache_io, qwen
from src.extract.cache_chartqa import REPO, RunConfig, load_config, resolve_layers

torch = pytest.importorskip("torch")
from torch import nn  # noqa: E402

IMG, SPECIAL, HIDDEN = 9, 0, 4
QUESTIONS = {"Q0": "1", "Q1": "2", "Q2": "3"}       # question -> gold


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

class TestConfig:
    def test_committed_configs_agree_with_the_code(self):
        cfg = load_config(REPO / "configs/inference.yaml", REPO / "configs/activations.yaml")
        assert cfg.model_key == qwen.MODEL_KEY
        assert cfg.layers == tuple(range(28)) and cfg.hidden_size == 3584
        assert cfg.splits == ("test_human", "test_augmented") and cfg.expected_items == 2500

    def test_a_conflict_refuses_to_run(self, tmp_path):
        act = (REPO / "configs/activations.yaml").read_text().replace(
            "max_pixels: 1000000", "max_pixels: 2000000")
        (tmp_path / "activations.yaml").write_text(act)
        with pytest.raises(SystemExit, match="activations max_pixels"):
            load_config(REPO / "configs/inference.yaml", tmp_path / "activations.yaml")

    def test_resolve_layers(self):
        assert resolve_layers("all", 3) == (0, 1, 2)
        assert resolve_layers([2, 0], 3) == (0, 2)
        for bad in ([0, 0], [3], [], "some"):
            with pytest.raises(SystemExit):
                resolve_layers(bad, 3)


# ---------------------------------------------------------------------------
# A fake model and a fake ChartQA
# ---------------------------------------------------------------------------

class FakeBlock(nn.Module):
    def forward(self, h):
        return (h + 1,)


class FakeQwen:
    """Stands in for the three functions of src.extract.qwen that `run` calls."""

    def __init__(self, answers, n_layers=3):
        self.answers = dict(answers)       # prompt -> answer
        self.fail_on = None
        self.generated = []
        self.layers = nn.ModuleList(FakeBlock() for _ in range(n_layers))

    def load(self):
        model = SimpleNamespace(
            model=SimpleNamespace(language_model=SimpleNamespace(layers=self.layers)),
            config=SimpleNamespace(image_token_id=IMG, _attn_implementation="sdpa"))
        processor = SimpleNamespace(image_token_id=IMG,
                                    tokenizer=SimpleNamespace(all_special_ids=[SPECIAL]))
        return model, processor

    def build_inputs(self, model, processor, image, prompt):
        ids = torch.tensor([[SPECIAL, 5, IMG, IMG, 6]])
        return {"input_ids": ids, "attention_mask": torch.ones_like(ids), "prompt": prompt}

    def generate_from_inputs(self, model, processor, inputs):
        question = inputs["prompt"].removesuffix(qwen.ANSWER_SUFFIX)
        if question == self.fail_on:
            raise RuntimeError("simulated CUDA error")
        h = torch.zeros(1, inputs["input_ids"].shape[1], HIDDEN)
        for block in self.layers:                     # prefill
            h = block(h)[0]
        x = h[:, -1:]
        for _ in range(2):                            # decode steps
            for block in self.layers:
                x = block(x)[0]
        self.generated.append(question)
        return self.answers[question]


@pytest.fixture
def setup(tmp_path, monkeypatch):
    from PIL import Image

    root = tmp_path / "ChartQA"
    (root / "test" / "png").mkdir(parents=True)
    entries = [{"imgname": f"fig{i}.png", "query": q, "label": g}
               for i, (q, g) in enumerate(QUESTIONS.items())]
    (root / "test" / "test_human.json").write_text(json.dumps(entries))
    for e in entries:
        Image.new("RGB", (8, 8), "white").save(root / "test" / "png" / e["imgname"])

    # two right, one wrong: accuracy 0.667 clears the 0.60 floor
    fake = FakeQwen({"Q0": "1", "Q1": "2", "Q2": "7"})
    for name in ("load", "build_inputs", "generate_from_inputs"):
        monkeypatch.setattr(qwen, name, getattr(fake, name))
    monkeypatch.setattr(cache_chartqa, "record_segment", lambda *a, **k: None)
    monkeypatch.setattr(cache_chartqa, "frozen_settings", lambda *a, **k: {})

    cfg = RunConfig(model_key=qwen.MODEL_KEY, data_root=root, splits=("test_human",),
                    expected_items=3, layers_setting="all", layers=(0, 1, 2), n_layers=3,
                    hidden_size=HIDDEN, positions=cache_chartqa.POSITIONS,
                    vision_token_range=(1, 10))
    paths = cache_chartqa.paths_for(cfg.model_key, "test", tmp_path / "preds", tmp_path / "acts")
    return cfg, paths, fake


def lines(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def assemble_and_validate(cfg, paths):
    order = [it["item_id"] for it in cache_chartqa.load_items(cfg)]
    cache_io.assemble(paths.shards, order, cfg.layers, cfg.positions, paths.cache)
    return cache_chartqa.validate(cfg, paths, paths.cache, None)


# ---------------------------------------------------------------------------
# run
# ---------------------------------------------------------------------------

class TestRun:
    def test_fresh_run_writes_rows_shards_and_a_passing_cache(self, setup):
        cfg, paths, fake = setup
        cache_chartqa.run(cfg, paths, limit=None, shard_size=2)

        rows = lines(paths.predictions)
        assert [r["question"] for r in rows] == list(QUESTIONS)
        assert [r["correct"] for r in rows] == [True, True, False]
        assert all(r["n_vision_tokens"] == 2 and r["source"] == "human" for r in rows)
        assert len(cache_io.shard_paths(paths.shards)) == 2          # 2 + 1

        assert assemble_and_validate(cfg, paths) == 0
        summary = json.loads(paths.summary.read_text())
        assert summary["passed"] and summary["n_incorrect"] == 1
        assert summary["relaxed_accuracy"]["human"] == 0.6667

    def test_second_run_is_a_no_op(self, setup):
        cfg, paths, fake = setup
        cache_chartqa.run(cfg, paths, limit=None, shard_size=2)
        fake.generated.clear()
        cache_chartqa.run(cfg, paths, limit=None, shard_size=2)
        assert fake.generated == []

    def test_a_crash_keeps_finished_items_and_resumes_after_them(self, setup):
        cfg, paths, fake = setup
        fake.fail_on = "Q2"
        with pytest.raises(RuntimeError, match="simulated"):
            cache_chartqa.run(cfg, paths, limit=None, shard_size=250)
        assert len(cache_io.cached_item_ids(paths.shards)) == 2      # flushed on the way out

        fake.fail_on, fake.generated = None, []
        cache_chartqa.run(cfg, paths, limit=None, shard_size=250)
        assert fake.generated == ["Q2"]
        assert len(lines(paths.predictions)) == 3
        assert assemble_and_validate(cfg, paths) == 0

    def test_a_row_without_activations_is_regenerated_not_duplicated(self, setup):
        cfg, paths, fake = setup
        cache_chartqa.run(cfg, paths, limit=None, shard_size=2)
        # a hard kill after the third row was written, before its shard was
        [lone] = [p for p in cache_io.shard_paths(paths.shards)
                  if len(cache_io.read_shard(p)["item_ids"]) == 1]
        lone.unlink()

        fake.generated = []
        cache_chartqa.run(cfg, paths, limit=None, shard_size=2)
        assert fake.generated == ["Q2"]
        assert len(lines(paths.predictions)) == 3
        assert assemble_and_validate(cfg, paths) == 0

    def test_a_regenerated_answer_that_differs_stops_the_run(self, setup):
        cfg, paths, fake = setup
        cache_chartqa.run(cfg, paths, limit=None, shard_size=2)
        for p in cache_io.shard_paths(paths.shards):
            p.unlink()
        fake.answers["Q0"] = "something else"
        with pytest.raises(SystemExit, match="not reproducing"):
            cache_chartqa.run(cfg, paths, limit=None, shard_size=2)

    def test_limit_runs_a_prefix(self, setup):
        cfg, paths, fake = setup
        cache_chartqa.run(cfg, paths, limit=2, shard_size=250)
        assert fake.generated == ["Q0", "Q1"]
        order = [it["item_id"] for it in cache_chartqa.load_items(cfg, 2)]
        cache_io.assemble(paths.shards, order, cfg.layers, cfg.positions, paths.cache)
        assert cache_chartqa.validate(cfg, paths, paths.cache, 2) == 0


def test_low_accuracy_fails_validation(setup):
    cfg, paths, fake = setup
    fake.answers = {q: "wrong" for q in QUESTIONS}
    cache_chartqa.run(cfg, paths, limit=None, shard_size=250)
    assert assemble_and_validate(cfg, paths) == 1
    assert "below 0.6" in " ".join(json.loads(paths.summary.read_text())["problems"])


def test_wrong_item_count_is_caught_before_any_gpu_work(setup):
    cfg, paths, fake = setup
    with pytest.raises(SystemExit, match="expects 4"):
        cache_chartqa.run(replace(cfg, expected_items=4), paths, limit=None, shard_size=2)
    assert fake.generated == []
