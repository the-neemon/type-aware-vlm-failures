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


# ---------------------------------------------------------------------------
# A synthetic manifest instead of ChartQA
# ---------------------------------------------------------------------------

def test_a_synthetic_manifest_runs_and_scores_absent_questions(setup, tmp_path):
    from PIL import Image

    cfg, paths, fake = setup
    root = tmp_path / "synth"
    (root / "images").mkdir(parents=True)
    figures = []
    for i, (bar, absent, split) in enumerate([("Mango", "Guava", "train"),
                                              ("Paris", "Oslo", "test")]):
        Image.new("RGB", (8, 8), "white").save(root / "images" / f"f{i}.png")
        figures.append({"figure_id": f"f{i}", "split": split, "image_path": f"images/f{i}.png",
                        "categories": [bar, "Other"],
                        "questions": [
                            {"template": "read_value", "absent": False, "phrasing": "value_of",
                             "asks_about": bar, "question": f"What is the value of {bar}?",
                             "gold_answer": 40},
                            {"template": "absent_category", "absent": True,
                             "phrasing": "value_of", "asks_about": absent,
                             "question": f"What is the value of {absent}?",
                             "gold_answer": "not present"}]})
    (root / "manifest.jsonl").write_text("".join(json.dumps(f) + "\n" for f in figures))
    fake.answers = {"What is the value of Mango?": "40", "What is the value of Guava?": "35",
                    "What is the value of Paris?": "40", "What is the value of Oslo?": "Not shown"}
    cfg = replace(cfg, data_root=root, manifest=root / "manifest.jsonl",
                  splits=("train", "validation", "test"), expected_items=4)

    cache_chartqa.run(cfg, paths, limit=None, shard_size=250)
    rows = lines(paths.predictions)
    assert [r["correct"] for r in rows] == [True, False, True, True]
    assert [r.get("outcome") for r in rows] == [None, "fabricated", None, "rejected"]
    assert all(r["source"] == "synthetic" for r in rows)

    assert assemble_and_validate(cfg, paths) == 0
    summary = json.loads(paths.summary.read_text())
    assert summary["relaxed_accuracy"]["answerable"] == 1.0
    assert summary["absent_outcomes"] == {"absent_category": {"fabricated": 1, "rejected": 1}}

    # the manifest's splits filter figures, so a test-only config sees one chart
    assert len(cache_chartqa.load_items(replace(cfg, splits=("test",), expected_items=2))) == 2


@pytest.mark.parametrize("name", ["inference_synth_pilot.yaml", "inference_synth_pilot2.yaml"])
def test_the_synthetic_pilot_configs_agree_with_the_code(name):
    cfg = load_config(REPO / "configs" / name, REPO / "configs/activations.yaml")
    assert cfg.manifest is not None and cfg.manifest.name == "manifest.jsonl"
    chartqa = load_config(REPO / "configs/inference.yaml", REPO / "configs/activations.yaml")
    assert chartqa.manifest is None


# ---------------------------------------------------------------------------
# LLaVA-NeXT: the same job with the second model
# ---------------------------------------------------------------------------

from src.extract import llava  # noqa: E402

LLAVA_CONFIGS = ["inference.yaml", "inference_val_human.yaml",
                 "inference_synth_pilot.yaml", "inference_synth_pilot2.yaml"]


@pytest.mark.parametrize("name", LLAVA_CONFIGS)
def test_each_llava_config_is_its_qwen_config_with_only_the_model_changed(name):
    import yaml

    qwen_cfg = yaml.safe_load((REPO / "configs" / name).read_text())
    llava_cfg = yaml.safe_load((REPO / "configs/llava" / name).read_text())
    assert llava_cfg["model_id"] == llava.MODEL_ID and llava_cfg["model_key"] == llava.MODEL_KEY
    strip = lambda c: {k: v for k, v in c.items() if k not in ("model_id", "model_key")}
    assert strip(llava_cfg) == strip(qwen_cfg)

    cfg = load_config(REPO / "configs/llava" / name, REPO / "configs/activations.yaml")
    assert cfg.model_key == llava.MODEL_KEY
    assert cfg.layers == tuple(range(32)) and cfg.hidden_size == 4096


def test_both_models_get_the_same_prompt_and_decoding():
    assert llava.ANSWER_SUFFIX == qwen.ANSWER_SUFFIX
    assert (llava.SEED, llava.MAX_NEW_TOKENS) == (qwen.SEED, qwen.MAX_NEW_TOKENS) == (42, 32)
    assert llava.generate_from_inputs is qwen.generate_from_inputs
    assert llava.build_inputs is qwen.build_inputs


def test_an_unknown_model_key_is_refused(tmp_path):
    text = (REPO / "configs/inference.yaml").read_text().replace(
        "model_key: qwen2_5_vl_7b", "model_key: some_other_vlm")
    (tmp_path / "inference.yaml").write_text(text)
    with pytest.raises(SystemExit, match="unknown model_key 'some_other_vlm'"):
        load_config(tmp_path / "inference.yaml", REPO / "configs/activations.yaml")


class TensorBlock(nn.Module):
    """Returns hidden_states itself, like MistralDecoderLayer on transformers 4.57.6."""
    def forward(self, h):
        return h + 1


class FakeLlava(FakeQwen):
    def __init__(self, answers, n_layers=4, block=TensorBlock):
        super().__init__(answers, n_layers)
        self.layers = nn.ModuleList(block() for _ in range(n_layers))

    def generate_from_inputs(self, model, processor, inputs):
        question = inputs["prompt"].removesuffix(llava.ANSWER_SUFFIX)
        h = torch.zeros(1, inputs["input_ids"].shape[1], HIDDEN)
        for block in self.layers:                     # prefill
            h = block(h)
        for block in self.layers:                     # one decode step
            block(h[:, -1:])
        self.generated.append(question)
        return self.answers[question]


@pytest.fixture
def llava_setup(setup, monkeypatch):
    cfg, _, _ = setup                                  # reuse its fake ChartQA
    # one right of three: 0.33 clears LLaVA's floor, would fail Qwen's 0.60
    fake = FakeLlava({"Q0": "1", "Q1": "9", "Q2": "7"})
    for name in ("load", "build_inputs", "generate_from_inputs"):
        monkeypatch.setattr(llava, name, getattr(fake, name))
    cfg = replace(cfg, model_key=llava.MODEL_KEY, layers=(0, 1, 2, 3), n_layers=4)
    root = cfg.data_root.parent
    paths = cache_chartqa.paths_for(cfg.model_key, "test", root / "preds", root / "acts")
    return cfg, paths, fake


def test_llava_runs_end_to_end_under_its_own_names_and_floor(llava_setup, setup):
    cfg, paths, fake = llava_setup
    cache_chartqa.run(cfg, paths, limit=None, shard_size=2)

    assert paths.predictions.name == "llava_next_mistral_7b_test.jsonl"
    assert paths.cache.parent.name == "llava_next_mistral_7b"
    rows = lines(paths.predictions)
    assert [r["correct"] for r in rows] == [True, False, False]
    assert all(r["model"] == llava.MODEL_KEY for r in rows)
    # the model key is inside the hash, so the same questions get new item_ids
    qwen_ids = {it["item_id"] for it in cache_chartqa.load_items(setup[0])}
    assert not qwen_ids & {r["item_id"] for r in rows}

    assert assemble_and_validate(cfg, paths) == 0
    with np_load(paths.cache) as z:
        assert sorted(k for k in z.files if k.endswith("_query_last")) == \
            [f"L{i}_query_last" for i in range(4)]


def test_llava_below_its_floor_fails_validation(llava_setup):
    cfg, paths, fake = llava_setup
    fake.answers = {q: "wrong" for q in QUESTIONS}
    cache_chartqa.run(cfg, paths, limit=None, shard_size=250)
    assert assemble_and_validate(cfg, paths) == 1
    assert "below 0.27" in " ".join(json.loads(paths.summary.read_text())["problems"])


def test_a_llava_block_that_returns_a_tuple_stops_the_run(setup, monkeypatch):
    cfg, _, _ = setup
    fake = FakeLlava({"Q0": "1"}, n_layers=4, block=FakeBlock)     # Qwen-style 1-tuples
    monkeypatch.setattr(llava, "load", fake.load)
    monkeypatch.setattr(llava, "build_inputs", fake.build_inputs)
    monkeypatch.setattr(llava, "generate_from_inputs",
                        lambda m, p, i: FakeQwen.generate_from_inputs(fake, m, p, i))
    cfg = replace(cfg, model_key=llava.MODEL_KEY, layers=(0, 1, 2, 3), n_layers=4)
    root = cfg.data_root.parent
    paths = cache_chartqa.paths_for(cfg.model_key, "test", root / "preds", root / "acts")
    with pytest.raises(TypeError, match="expected a tensor"):
        cache_chartqa.run(cfg, paths, limit=None, shard_size=2)


def np_load(path):
    import numpy as np
    return np.load(path, allow_pickle=True)


# ---------------------------------------------------------------------------
# The run record: Qwen's must not change, or its runs could not resume
# ---------------------------------------------------------------------------

@pytest.fixture
def fake_env(monkeypatch):
    import importlib.metadata
    monkeypatch.setattr(importlib.metadata, "version", lambda name: f"v-{name}")
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda i=0: "NVIDIA GeForce RTX 2080 Ti")
    model = SimpleNamespace(config=SimpleNamespace(_commit_hash="abc123", _attn_implementation="sdpa"),
                            parameters=lambda: iter([torch.zeros(1, dtype=torch.float16)]))
    return model


def test_qwen_frozen_settings_are_unchanged(fake_env):
    import platform

    class Qwen2VLImageProcessorFast:
        max_pixels = 1_000_000

    cfg = load_config(REPO / "configs/inference.yaml", REPO / "configs/activations.yaml")
    got = cache_chartqa.frozen_settings(
        cfg, fake_env, SimpleNamespace(image_processor=Qwen2VLImageProcessorFast()))
    # exactly the keys and values the 26 Sep Qwen runs recorded
    assert got == {
        "model_id": "Qwen/Qwen2.5-VL-7B-Instruct", "model_key": "qwen2_5_vl_7b",
        "model_revision": "abc123", "python": platform.python_version(),
        "torch": torch.__version__, "transformers": "v-transformers",
        "qwen_vl_utils": "v-qwen-vl-utils", "accelerate": "v-accelerate",
        "dtype": "torch.float16", "attn_implementation": "sdpa",
        "image_processor": "Qwen2VLImageProcessorFast", "processor_use_fast": True,
        "max_pixels": 1_000_000, "image_processor_max_pixels": 1_000_000,
        "prompt_suffix": qwen.ANSWER_SUFFIX, "do_sample": False, "max_new_tokens": 32,
        "seed": 42, "gpu_name": "NVIDIA GeForce RTX 2080 Ti", "layers_config": "all",
        "layers": list(range(28)), "hidden_size": 3584, "positions": list(cache_chartqa.POSITIONS),
        "dataset_root": str(cfg.data_root), "dataset_splits": ["test_human", "test_augmented"],
        "expected_items": 2500,
    }


def test_llava_frozen_settings_record_its_own_preprocessing(fake_env):
    class LlavaNextImageProcessorFast:
        image_grid_pinpoints = [[336, 672], [672, 336], [672, 672], [1008, 336], [336, 1008]]
        size = {"shortest_edge": 336}
        crop_size = {"height": 336, "width": 336}
        do_pad = True

    processor = SimpleNamespace(image_processor=LlavaNextImageProcessorFast(), patch_size=14,
                                vision_feature_select_strategy="default")
    cfg = load_config(REPO / "configs/llava/inference.yaml", REPO / "configs/activations.yaml")
    got = cache_chartqa.frozen_settings(cfg, fake_env, processor)
    assert got["model_id"] == llava.MODEL_ID and got["model_key"] == llava.MODEL_KEY
    assert got["image_grid_pinpoints"][2] == [672, 672] and got["patch_size"] == 14
    assert got["vision_feature_select_strategy"] == "default"
    assert got["layers"] == list(range(32)) and got["hidden_size"] == 4096
    assert "max_pixels" not in got and "qwen_vl_utils" not in got
    assert got["prompt_suffix"] == qwen.ANSWER_SUFFIX
