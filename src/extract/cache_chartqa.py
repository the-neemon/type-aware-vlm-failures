"""A VLM over ChartQA or a synthetic set: answers plus prefill activations (inf.md, TASKS P1.3, P4.1-P4.4).

Spec: qwen_chartqa_activation_caching_spec.md. Three steps, run in this order
by scripts/cache_chartqa.sbatch:

    run       answer every question once, greedily, and cache the four pooled
              vectors at every configured decoder layer from the prefill of
              that same generate() call. Resumable.
    assemble  merge the shards into one compressed .npz, rows in dataset order.
    validate  the acceptance checks, plus accuracy, error yield and vision-token
              counts, written next to the predictions as <tag>.summary.json.

    python -m src.extract.cache_chartqa run
    python -m src.extract.cache_chartqa assemble --out /scratch/.../test.npz
    python -m src.extract.cache_chartqa validate

Where things go, for the default --tag test:

    results/predictions/qwen2_5_vl_7b_test.jsonl        predictions (inf.md 3.2)
    results/predictions/qwen2_5_vl_7b_test.run.json     versions and settings used
    results/predictions/qwen2_5_vl_7b_test.summary.json acceptance report
    ~/activations/qwen2_5_vl_7b/shards_test/            resumable shards
    ~/activations/qwen2_5_vl_7b/test.npz                the cache (inf.md 3.3)

Shards are written straight to $HOME rather than node-local /scratch: a job
resubmitted after a timeout may land on another node, and resuming has to see
every shard already written. The final .npz is compressed on /scratch and
staged back with `stage_out`, like every other job here.

The model comes from the inference config's `model_key`: qwen2_5_vl_7b (the
default config) or llava_next_mistral_7b (configs/llava/). Its outputs are named
by that key, so the two models never share a file.

Settings are not chosen here. The model is loaded and queried through its module
in `MODELS` (src/extract/qwen.py, which E4's repairs also use, or
src/extract/llava.py), and this script refuses to start if the inference config
or configs/activations.yaml disagree with it.
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import pathlib
import platform
import statistics
import time
from collections import Counter
from dataclasses import dataclass

import yaml

from src.eval.chartqa import load_chartqa
from src.eval.relaxed_accuracy import is_correct
from src.extract import cache_io, llava, qwen
from src.extract.prefill import POSITIONS
from src.label.pool import item_id
from src.synth import items as synth_items

REPO = pathlib.Path(__file__).resolve().parents[2]

# The model modules, by the model_key an inference config names. Each supplies
# MODEL_ID, MODEL_KEY, ANSWER_SUFFIX, SEED, MAX_NEW_TOKENS, DECODER_OUTPUT,
# MIN_ACCURACY, config_checks, processor_settings, load, build_inputs and
# generate_from_inputs.
MODELS = {m.MODEL_KEY: m for m in (qwen, llava)}


def model_module(model_key: str):
    if model_key not in MODELS:
        raise SystemExit(f"unknown model_key {model_key!r}; known: {sorted(MODELS)}")
    return MODELS[model_key]


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RunConfig:
    model_key: str
    data_root: pathlib.Path
    splits: tuple[str, ...]            # e.g. ("test_human", "test_augmented")
    expected_items: int
    layers_setting: object             # "all" or a list, as written in the config
    layers: tuple[int, ...]            # resolved, zero-based decoder block indices
    n_layers: int
    hidden_size: int
    positions: tuple[str, ...]
    vision_token_range: tuple[int, int]
    # A synthetic manifest (src/synth/items.py) instead of ChartQA: `splits` then
    # name the manifest's figure splits. None for every ChartQA run.
    manifest: pathlib.Path | None = None


def resolve_layers(setting, n_layers: int) -> tuple[int, ...]:
    """`all` -> every block; otherwise an explicit list of distinct in-range indices."""
    if setting == "all":
        return tuple(range(n_layers))
    if (not isinstance(setting, list) or not setting
            or not all(isinstance(i, int) and 0 <= i < n_layers for i in setting)
            or len(set(setting)) != len(setting)):
        raise SystemExit(f"configs/activations.yaml layers must be 'all' or distinct "
                         f"integers in 0..{n_layers - 1}, got {setting!r}")
    return tuple(sorted(setting))


def load_config(inference_yaml: pathlib.Path, activations_yaml: pathlib.Path) -> RunConfig:
    """Read both configs and fail loudly wherever they disagree with the model's module."""
    inf = yaml.safe_load(inference_yaml.read_text())
    act = yaml.safe_load(activations_yaml.read_text())
    m = model_module(inf["model_key"])
    model = act["models"][inf["model_key"]]

    checks = [
        ("inference model_id", inf["model_id"], m.MODEL_ID),
        ("inference model_key", inf["model_key"], m.MODEL_KEY),
        ("inference prompt_suffix", inf["prompt_suffix"], m.ANSWER_SUFFIX),
        ("inference decoding.do_sample", inf["decoding"]["do_sample"], False),
        ("inference decoding.max_new_tokens", inf["decoding"]["max_new_tokens"], m.MAX_NEW_TOKENS),
        ("inference decoding.seed", inf["decoding"]["seed"], m.SEED),
        ("inference batch_size", inf["batch_size"], 1),
        ("activations dtype", act["dtype"], "float16"),
        ("activations gpu_dtype", act["gpu_dtype"], "float16"),
        ("activations gpu_attn_implementation", act["gpu_attn_implementation"], "sdpa"),
        ("activations processor_use_fast", act["processor_use_fast"], True),
        ("activations hf_id", model["hf_id"], m.MODEL_ID),
        ("activations positions", list(act["positions"]), list(POSITIONS)),
        *m.config_checks(model),
    ]
    conflicts = [f"{name}: config says {got!r}, the code uses {want!r}"
                 for name, got, want in checks if got != want]
    if conflicts:
        raise SystemExit("config conflict, refusing to run:\n  " + "\n  ".join(conflicts))

    n_layers = model["num_hidden_layers"]
    data_root = pathlib.Path(inf["dataset"]["root"]).expanduser()
    manifest = inf["dataset"].get("manifest")
    return RunConfig(
        model_key=inf["model_key"],
        data_root=data_root,
        splits=tuple(inf["dataset"]["splits"]),
        expected_items=inf["dataset"]["expected_items"],
        layers_setting=act["layers"],
        layers=resolve_layers(act["layers"], n_layers),
        n_layers=n_layers,
        hidden_size=model["hidden_size"],
        positions=tuple(act["positions"]),
        vision_token_range=tuple(model["measured_vision_tokens_range"]),
        manifest=data_root / manifest if manifest else None,
    )


@dataclass(frozen=True)
class Paths:
    predictions: pathlib.Path
    run_meta: pathlib.Path
    summary: pathlib.Path
    shards: pathlib.Path          # durable, in $HOME
    cache: pathlib.Path           # durable final .npz, in $HOME


def paths_for(model_key: str, tag: str, predictions_dir: pathlib.Path,
              activations_dir: pathlib.Path) -> Paths:
    stem = f"{model_key}_{tag}"
    act = activations_dir / model_key
    return Paths(predictions=predictions_dir / f"{stem}.jsonl",
                 run_meta=predictions_dir / f"{stem}.run.json",
                 summary=predictions_dir / f"{stem}.summary.json",
                 shards=act / f"shards_{tag}",
                 cache=act / f"{tag}.npz")


def load_items(cfg: RunConfig, limit: int | None = None) -> list[dict]:
    """Items in dataset order, each carrying its item_id and image path.

    ChartQA, or a synthetic manifest when the config names one. ChartQA contains
    repeated figure/question pairs, including repeats within one source. A
    deterministic occurrence number keeps every question distinct.
    """
    if cfg.manifest is not None:
        items = synth_items.load_manifest_items(cfg.manifest, cfg.splits)
    else:
        data = load_chartqa(cfg.data_root, splits=tuple({s.split("_")[0] for s in cfg.splits}))
        missing = [s for s in cfg.splits if s not in data]
        if missing:
            raise SystemExit(f"ChartQA splits {missing} not found under {cfg.data_root}")
        items = [dict(it, image=cfg.data_root / it["split"] / "png" / it["figure_id"])
                 for split in cfg.splits for it in data[split]]
    if len(items) != cfg.expected_items:
        raise SystemExit(f"loaded {len(items)} questions, configs/inference.yaml expects "
                         f"{cfg.expected_items}")
    occurrences: dict[tuple[str, str, str], int] = {}
    for it in items:
        key = (it["source"], it["figure_id"], it["question"])
        occurrence = occurrences.get(key, 0)
        occurrences[key] = occurrence + 1
        it["occurrence"] = occurrence
        it["item_id"] = item_id(
            cfg.model_key, it["figure_id"], it["question"], it["source"], occurrence)
    if len({it["item_id"] for it in items}) != len(items):
        raise SystemExit("ChartQA items still have duplicate item_ids after source disambiguation")
    return items[:limit] if limit else items


# ---------------------------------------------------------------------------
# Run metadata
# ---------------------------------------------------------------------------

def frozen_settings(cfg: RunConfig, model, processor) -> dict:
    """Everything that changes an activation. A resumed run must match it exactly."""
    import importlib.metadata

    import torch

    m = model_module(cfg.model_key)
    version = importlib.metadata.version
    settings = {
        "model_id": m.MODEL_ID,
        "model_key": cfg.model_key,
        "model_revision": getattr(model.config, "_commit_hash", None),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": version("transformers"),
        "accelerate": version("accelerate"),
        "dtype": str(next(model.parameters()).dtype),
        "attn_implementation": model.config._attn_implementation,
        "image_processor": type(processor.image_processor).__name__,
        "processor_use_fast": True,
        **m.processor_settings(processor),
        "prompt_suffix": m.ANSWER_SUFFIX,
        "do_sample": False,
        "max_new_tokens": m.MAX_NEW_TOKENS,
        "seed": m.SEED,
        "gpu_name": torch.cuda.get_device_name(0),
        "layers_config": cfg.layers_setting,
        "layers": list(cfg.layers),
        "hidden_size": cfg.hidden_size,
        "positions": list(cfg.positions),
        "dataset_root": str(cfg.data_root),
        "dataset_splits": list(cfg.splits),
        "expected_items": cfg.expected_items,
    }
    if cfg.manifest is not None:        # absent for ChartQA, so its runs still resume
        settings["dataset_manifest"] = str(cfg.manifest)
    return json.loads(json.dumps(settings))       # the form it takes on disk


def record_segment(path: pathlib.Path, frozen: dict, model, n_todo: int) -> None:
    """Append this job to the run record, refusing to resume under different settings."""
    import torch

    if path.is_file():
        meta = json.loads(path.read_text())
        changed = sorted(k for k in frozen.keys() | meta["frozen"].keys()
                         if frozen.get(k) != meta["frozen"].get(k))
        if changed:
            raise SystemExit(f"cannot resume: {changed} differ from the run recorded in {path}. "
                             "Mixing them would put two configurations in one cache.")
    else:
        meta = {"frozen": frozen, "segments": []}
    meta["segments"].append({
        "started_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "node": platform.node(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "gpus": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
        "device_map": {k: str(v) for k, v in model.hf_device_map.items()},
        "items_to_run": n_todo,
    })
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(meta, indent=2) + "\n")


# ---------------------------------------------------------------------------
# run
# ---------------------------------------------------------------------------

def run(cfg: RunConfig, paths: Paths, limit: int | None, shard_size: int) -> None:
    import numpy as np
    from PIL import Image

    from src.extract.prefill import PrefillCapture, build_pool_masks, decoder_layers

    items = load_items(cfg, limit)
    stored = cache_io.read_predictions(paths.predictions)
    cached = cache_io.cached_item_ids(paths.shards)
    if cached - stored.keys():
        raise SystemExit(f"{len(cached - stored.keys())} items have activations but no "
                         f"prediction row in {paths.predictions}; the two files are out of step")
    todo = [it for it in items if it["item_id"] not in cached]
    print(f"{len(items)} items: {len(cached)} already cached, {len(todo)} to run", flush=True)
    if not todo:
        return

    m = model_module(cfg.model_key)
    model, processor = m.load()
    layers = decoder_layers(model, cfg.n_layers)
    if model.config._attn_implementation != "sdpa":
        raise SystemExit(f"model loaded with {model.config._attn_implementation} attention, not sdpa")
    # LLaVA's config calls it image_token_index; image_token_id is its alias
    image_token_id = model.config.image_token_id
    if image_token_id != processor.image_token_id:
        raise SystemExit(f"model and processor disagree on the image token id: "
                         f"{image_token_id} vs {processor.image_token_id}")
    special_ids = processor.tokenizer.all_special_ids
    record_segment(paths.run_meta, frozen_settings(cfg, model, processor), model, len(todo))

    buffer: list[tuple[dict, np.ndarray]] = []

    def flush() -> None:
        ids = [row["item_id"] for row, _ in buffer]
        cache_io.write_shard(paths.shards / cache_io.shard_name(ids), ids,
                             [row["figure_id"] for row, _ in buffer],
                             np.stack([acts for _, acts in buffer]), cfg.layers, cfg.positions)
        print(f"wrote shard of {len(buffer)} items", flush=True)
        buffer.clear()

    with PrefillCapture(layers, cfg.layers, cfg.hidden_size, m.DECODER_OUTPUT) as capture:
        try:
            for n, it in enumerate(todo, 1):
                t0 = time.time()
                image = Image.open(it["image"]).convert("RGB")
                inputs = m.build_inputs(model, processor, image, it["question"] + m.ANSWER_SUFFIX)
                masks = build_pool_masks(inputs["input_ids"], inputs["attention_mask"],
                                         image_token_id, special_ids)
                capture.arm(masks)
                prediction = m.generate_from_inputs(model, processor, inputs)
                acts = capture.collect().numpy()

                row = {"item_id": it["item_id"], "model": cfg.model_key,
                       "figure_id": it["figure_id"], "question": it["question"],
                       "gold": it["gold"], "prediction": prediction,
                       "correct": is_correct(it["gold"], prediction),
                       "split": it["split"], "source": it["source"],
                       "occurrence": it["occurrence"],
                       "n_vision_tokens": masks.n_vision}
                if cfg.manifest is not None:
                    row.update(synth_items.score(it, prediction))
                previous = stored.get(it["item_id"])
                if previous is None:
                    cache_io.append_prediction(row, paths.predictions)
                elif previous["prediction"] != prediction:
                    # The row was written before the job died, its shard was not.
                    # Greedy decoding on the frozen setup must reproduce it.
                    raise SystemExit(
                        f"item {it['item_id']} regenerated as {prediction!r}, but its stored "
                        f"answer is {previous['prediction']!r}. The decoding is not reproducing, "
                        "so these activations would not belong to the labelled answer. Stop here.")
                buffer.append((row, acts))

                print(f"[{n}/{len(todo)}] {it['item_id']} {'ok   ' if row['correct'] else 'WRONG'} "
                      f"vision={masks.n_vision} {time.time() - t0:.1f}s {prediction!r}", flush=True)
                if len(buffer) == shard_size:
                    flush()
        finally:
            # every buffered item is complete, so keep it even when the run dies
            if buffer:
                flush()


# ---------------------------------------------------------------------------
# assemble and validate
# ---------------------------------------------------------------------------

def summarize(rows: list[dict], vision_range: tuple[int, int]) -> dict:
    """Accuracy per sub-split (inf.md 8.3), error yield (8.4) and vision tokens (8.5).

    Synthetic rows also get accuracy per template and, per absent template, the
    counts of each outcome (src/synth/items.py).
    Their `answerable` accuracy leaves out absent questions, whose correct answer
    is a refusal: that is the number the pipeline floor applies to.
    """
    def accuracy(subset):
        return round(sum(r["correct"] for r in subset) / len(subset), 4) if subset else None

    n_correct = sum(r["correct"] for r in rows)
    tokens = [r["n_vision_tokens"] for r in rows]
    lo, hi = vision_range
    synthetic = any("template" in r for r in rows)
    by_source = ({"answerable": accuracy([r for r in rows if not r["absent"]]),
                  **{t: accuracy([r for r in rows if r["template"] == t])
                     for t in sorted({r["template"] for r in rows})}} if synthetic else
                 {src: accuracy([r for r in rows if r["source"] == src])
                  for src in ("human", "augmented")})
    extra = ({"absent_outcomes": {
        t: dict(sorted(Counter(r["outcome"] for r in rows if r["template"] == t).items()))
        for t in sorted({r["template"] for r in rows if r["absent"]})}}
        if synthetic else {})
    return {
        "relaxed_accuracy": {
            "overall": accuracy(rows),
            **by_source,
        },
        **extra,
        "n_correct": n_correct,
        "n_incorrect": len(rows) - n_correct,
        "error_rate": round(1 - n_correct / len(rows), 4) if rows else None,
        "vision_tokens": {
            "min": min(tokens), "max": max(tokens),
            "mean": round(statistics.mean(tokens), 1), "median": statistics.median(tokens),
            "outside_measured_range": sum(not lo <= t <= hi for t in tokens),
            "measured_range": [lo, hi],
        } if tokens else None,
    }


def validate(cfg: RunConfig, paths: Paths, cache: pathlib.Path, limit: int | None) -> int:
    items = load_items(cfg, limit)
    rows = cache_io.read_predictions(paths.predictions)
    problems = cache_io.check_predictions(rows, cfg.model_key, len(items))
    if set(rows) != {it["item_id"] for it in items}:
        problems.append("prediction item_ids do not match the dataset's")
    problems += cache_io.check_cache(cache, rows, cfg.layers, cfg.positions, cfg.hidden_size)

    summary = summarize(list(rows.values()), cfg.vision_token_range)
    accuracy = summary["relaxed_accuracy"]
    overall = accuracy.get("answerable", accuracy["overall"])
    floor = model_module(cfg.model_key).MIN_ACCURACY
    if overall is not None and overall < floor:
        problems.append(f"relaxed accuracy {overall:.3f} is below {floor}: check the "
                        "prompt, decoding and scoring before labelling any of these errors")

    report = {"passed": not problems, "problems": problems, "cache": str(cache),
              "predictions": str(paths.predictions), **summary,
              "validated_at": datetime.datetime.now(datetime.timezone.utc).isoformat()}
    paths.summary.parent.mkdir(parents=True, exist_ok=True)
    paths.summary.write_text(json.dumps(report, indent=2) + "\n")

    print(json.dumps({k: report[k] for k in ("relaxed_accuracy", "absent_outcomes", "n_correct",
                                             "n_incorrect", "error_rate", "vision_tokens")
                      if k in report}, indent=2))
    if summary["vision_tokens"] and summary["vision_tokens"]["outside_measured_range"]:
        print(f"note: {summary['vision_tokens']['outside_measured_range']} items fall outside "
              f"the measured {list(cfg.vision_token_range)} vision-token range")
    print(("PASSED" if not problems else "FAILED:\n  " + "\n  ".join(problems))
          + f"\nreport: {paths.summary}")
    return 0 if not problems else 1


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description="VLM answers and prefill activations (Qwen2.5-VL-7B or LLaVA-NeXT)")
    ap.add_argument("step", choices=("run", "assemble", "validate"))
    ap.add_argument("--tag", default="test",
                    help="names every output; use something else for a smoke run")
    ap.add_argument("--limit", type=int,
                    help="first N items only, for a smoke run; pass the same N to every step")
    ap.add_argument("--shard-size", type=int, default=250)
    ap.add_argument("--predictions-dir", type=pathlib.Path, default=REPO / "results/predictions")
    ap.add_argument("--activations-dir", type=pathlib.Path,
                    default=pathlib.Path("~/activations").expanduser(),
                    help="durable root for shards and the final cache")
    ap.add_argument("--out", type=pathlib.Path,
                    help="assemble: where to write the .npz (node-local scratch on Ada)")
    ap.add_argument("--cache", type=pathlib.Path,
                    help="validate: the .npz to check; defaults to the durable copy")
    ap.add_argument("--inference-config", type=pathlib.Path, default=REPO / "configs/inference.yaml")
    ap.add_argument("--activations-config", type=pathlib.Path, default=REPO / "configs/activations.yaml")
    args = ap.parse_args()

    cfg = load_config(args.inference_config, args.activations_config)
    paths = paths_for(cfg.model_key, args.tag, args.predictions_dir, args.activations_dir)

    if args.step == "run":
        run(cfg, paths, args.limit, args.shard_size)
        return 0
    if args.step == "assemble":
        out = args.out or paths.cache
        order = [it["item_id"] for it in load_items(cfg, args.limit)]
        cache_io.assemble(paths.shards, order, cfg.layers, cfg.positions, out)
        print(f"assembled {len(order)} items x {len(cfg.layers)} layers x "
              f"{len(cfg.positions)} positions -> {out}")
        return 0
    return validate(cfg, paths, args.cache or paths.cache, args.limit)


if __name__ == "__main__":
    raise SystemExit(main())
