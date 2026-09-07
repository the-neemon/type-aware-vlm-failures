# Type-Aware Failure Prediction for Chart and Diagram Reasoning in VLMs

Advanced NLP, Monsoon 2026, IIIT Hyderabad.
Yash More, Naman Singhal, Shrish Kadam, Sanjith Ganapathi.

Do vision-language models fail on charts in two mechanically different ways
(misreading the figure versus fabricating a value), is that difference visible
in the model's internal state before it generates, and is knowing it worth
anything?

- [SPEC.md](SPEC.md): what we are building and what counts as done.
- [TASKS.md](TASKS.md): who does what, when, and what will bite us.

## Status

Phase P0 (infrastructure). The environment is pinned and `scripts/smoke.sbatch`
runs one ChartQA figure through Qwen2.5-VL-7B end to end on Ada. No evaluation,
no activation caching and no probes yet.

## Setup

Everything runs on Ada, partition `u22`, on RTX 2080 Ti nodes. The GPU type is
frozen for the project (`configs/activations.yaml`), because a model's answers
and the activations recorded alongside them have to come from the same kernels.

Build the environment from inside a job, never on the head node, which runs an
older glibc than the compute nodes:

```bash
srun -p u22 --constraint=2080ti --exclude=gnode077 --gres=gpu:1 \
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

Run these commands from the repository root. Each output directory must be new;
the generators write PNG figures and a JSONL manifest.

```bash
python3 -m src.synth.generate_bar_charts data/synthetic_bars \
  --num-charts 20 --seed 42

python3 -m src.synth.generate_followups nodes data/synthetic_nodes \
  --num-figures 20 --path-length 4 --edge-crossings 3 --seed 42

python3 -m src.synth.generate_followups pairs data/synthetic_pairs \
  --pair-count 10 --seed 42
```

See [TAXONOMY.md](src/synth/TAXONOMY.md) for the synthetic failure labels.

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
