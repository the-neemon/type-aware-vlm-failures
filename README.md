# Type-Aware Failure Prediction for Chart Reasoning in Vision-Language Models

Advanced NLP, Monsoon 2026, IIIT Hyderabad.
Yash More, Naman Singhal, Shrish Kadam, Sanjith Ganapathi.

Do vision-language models fail on charts in two mechanically different ways
(misreading the figure versus fabricating a value), is that difference visible
in the model's internal state before it generates, and is knowing it worth
anything? Two frozen models (Qwen2.5-VL-7B, LLaVA-NeXT-7B), ChartQA plus three
synthetic bar-chart pilots, linear probes on cached activations, a repair
experiment (E4) and a type-aware controller (E5).

- [SPEC.md](SPEC.md): the design and what counts as done.
- [docs/taxonomy.md](docs/taxonomy.md): the five error labels.
- [paper/](paper/): the mid and final reports.

## Results and how to reproduce them

Every number in the reports comes from a file under `results/`. Each script's
docstring gives its full arguments; the table names the script and the files it
writes. GPU steps run as SLURM jobs on RTX 2080 Ti nodes; probe and analysis
steps need only a CPU and the cached activations.

| Step | What it produces | Command | Output |
| --- | --- | --- | --- |
| Synthetic data | Pilot 1, 2, 3 charts and questions | `python -m src.synth.absent_pairs`, `src.synth.absent_lookalikes`, `src.synth.out_of_range` | `$HOME/data/synth/absent_pilot_v{1,2,3}/` (hash in each `configs/inference_synth_pilot*.yaml`) |
| Inference and caching (GPU) | Answers and pooled activations, every layer | `TAG=<tag> INFERENCE_CONFIG=configs/[llava/]inference*.yaml sbatch scripts/cache_chartqa.sbatch` | `results/predictions/*.jsonl` (+ `.run.json`, `.summary.json`); activations `.npz` kept off the repo |
| Error labels | Five-label annotation of every ChartQA error | LLM annotator following `docs/taxonomy.md` (`src/label/llm_label.py`); human check in `src/label/app.py` | `results/labels/*.claude.jsonl` |
| Error statistics | Labels by source, gold-answer errors | `python -m src.label.error_stats` | `results/labels/*.error_stats.json` |
| E1 binary probes | Will the answer be wrong? | `python -m src.probes.run_sweep --target binary` (and `--target source` for the sanity probe) | `results/probes/binary_*_test.{json,md}`, `sanity_*` |
| E1/E2 follow-ups | Five splits, residualised, calibration, val_human | `python -m src.probes.run_followups binary` | `results/probes/binary_followups*.json` |
| E3 synthetic | Type probes and cross-transfer matrix | `python -m src.probes.run_synth_types`; gridline baseline `run_followups grid` | `results/probes/synth_types_*.json`, `synth_grid_baseline_*.json`, `results/synth_type_probes.md` |
| E3 ChartQA | Misread vs computation probes, per question type, residualised | `python -m src.probes.run_chartqa_types [--question-type ...] [--residualise]` | `results/probes/chartqa_types_*.json`, `results/chartqa_type_probes.md` |
| Transfer | Synthetic probes applied to ChartQA | `python -m src.probes.run_transfer transfer` | `results/probes/transfer_chartqa_*.json` |
| E4 (GPU) | Every repair on every synthetic value-question error | `PREDICTIONS=... SYNTH_MANIFEST=... sbatch scripts/intervene.sbatch` | `results/e4/*.outcomes.jsonl`, `results/e4.md` |
| E5 re-ask (GPU) | The same repairs on the correct answers | as E4 with `CORRECT=1` | `results/e4/*.correct.outcomes.jsonl` |
| E5 | Controller vs selective prediction | `python -m src.controller.e5_scores` (CPU job), then `python -m src.controller.e5 --models ...` | `results/e5/`, `results/e5.md` |

E4 is pre-registered in [results/e4-preregistration.md](results/e4-preregistration.md).

## Setup

Everything runs on Ada, partition `u22`, on RTX 2080 Ti nodes. The GPU type is
frozen for the project (`configs/activations.yaml`), because a model's answers
and the activations recorded alongside them have to come from the same kernels.

Some nodes have broken GPU drivers or mixed card types. Keep their names in a
`BAD_NODES` shell variable (comma-separated, not committed) and pass
`--exclude="$BAD_NODES"` to every GPU job; each job also checks the card at
start-up and stops rather than fall back to the CPU.

Build the environment from inside a job, never on the head node, which runs an
older glibc than the compute nodes:

```bash
srun -p u22 --constraint=2080ti --exclude="$BAD_NODES" --gres=gpu:1 \
     -c 8 --mem=32G -t 01:00:00 bash scripts/setup_env.sh
```

Then:

```bash
sbatch scripts/smoke.sbatch           # one figure, one question, one answer
sbatch scripts/verify_tokens.sbatch   # P0.10 vision-token check
```

`requirements.txt` holds exact pins, not floors. Activations move silently
between library versions and this project's claim rests on activations, so a
teammate who resolves a different `transformers` build has a correctness
problem, not a convenience one.

### Getting ChartQA onto Ada

Run this once, on any compute node. It takes a few minutes and is idempotent,
so it is safe to call unconditionally from a job script:

```bash
bash scripts/download_chartqa.sh          # ~875 MB download, 1.1 GiB on disk
bash scripts/download_chartqa.sh --json-only   # laptop: split stats, no images
```

You get 32,719 questions over 20,882 figures, laid out as
`$HOME/data/ChartQA/{train,val,test}/{split}_{human,augmented}.json` plus the
PNGs. Check it with:

```bash
python -m src.eval.chartqa ~/data/ChartQA
```

which prints the per-split summary and runs the P5.4 no-shared-figure assertion.

**Each of us needs our own copy.** Home directories on Ada are `drwx------`, so
one person's download is unreadable by the rest of the team, and there is no
shared writable path on the cluster to put a single copy in (see
[results/ada_filesystem.md](results/ada_filesystem.md)). At 1.1 GiB against a
30 GiB quota that is an acceptable duplication; it is not worth loosening
permissions on a home directory to avoid.

It goes in `$HOME` rather than `/scratch` on purpose: `/scratch` is node-local
and purged at 7 days, so images cached there would be re-downloaded on whichever
node the scheduler happened to pick.

### Three Ada facts that shape every job

`scripts/ada_env.sh` encodes all three; the measurements behind them are in
[results/ada_filesystem.md](results/ada_filesystem.md).

- **`/scratch` is node-local, and it is really `/tmp`.** Same device, same
  inode, and `tmpreaper` purges it daily at 7 days of no access. It holds
  `HF_HOME` because two 7B checkpoints are ~33 GB against a 30 GB home quota,
  and `stage_in_model` repairs the cache on whichever node a job lands on.
  Anything durable is staged back to `$HOME` before the job exits, and the job
  fails loudly if that copy does not happen.
- **A 7B model does not fit on one card.** 2080 Tis hold 11 GiB; fp16 weights
  are ~16.6 GiB. Jobs ask for `--gres=gpu:2` and shard with `device_map="auto"`.
  Quantising is not available to us: it would perturb the activations we are
  measuring.
- **Turing has no bf16 tensor cores and no FlashAttention-2.** The project runs
  `float16` with `attn_implementation="sdpa"`. bf16 does not error on this
  hardware, it is merely emulated, which is why this is written down rather
  than left to be discovered.

## Synthetic data generation

The three pilots the results use, from the repository root (each output
directory must be new; generate on the cluster, since bar geometry depends on
the pinned matplotlib build):

```bash
python -m src.synth.absent_pairs      ~/data/synth/absent_pilot_v1 --num-charts 300
python -m src.synth.absent_lookalikes ~/data/synth/absent_pilot_v2 --num-charts 300
python -m src.synth.out_of_range      ~/data/synth/absent_pilot_v3 --num-charts 300
```

Each takes `--verify` to check a folder against its manifest. The earlier
generators (`src.synth.generate_bar_charts`, `generate_followups`, including the
unused node-link diagrams) are kept for reference.

## Models

| Model | Backbone | Hidden | Layers |
| --- | --- | --- | --- |
| Qwen2.5-VL-7B-Instruct | Qwen2.5-7B | 3584 | 28 |
| LLaVA-NeXT (llava-v1.6-mistral-7b-hf) | Mistral-7B-Instruct-v0.2 | 4096 | 32 |

Config snapshots are vendored under `configs/model_configs/` so the storage
budget is reproducible without network access. Note that the LLaVA-NeXT
`text_config` omits `hidden_size` and `num_hidden_layers`; both fall back to
Mistral-7B defaults and must be read from the backbone config, not the VLM one.

## Layout

```
configs/      run configs; vendored model configs under model_configs/
src/extract/  VLM inference and activation caching
src/label/    annotation tooling and inter-annotator agreement
src/synth/    synthetic bar-chart and node-link generators
src/probes/   binary, structural, fabrication probes
src/surface/  E2 surface-feature baseline and residualisation
src/intervene/ interventions and the E4 matrix
src/controller/ type-aware controller, oracle, binary policy
scripts/      Ada and SLURM job submission
results/      metrics and figures (small artefacts only)
paper/        ACL-style LaTeX for the mid and final write-ups
```

## Model and experiment tracking

HuggingFace and Weights & Biases links go here. Required by the course
guidelines for both the mid and final submissions.

- HuggingFace: none yet
- Weights & Biases: none yet
