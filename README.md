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

Phase P0 (infrastructure). Nothing to reproduce yet.

## Setup

Not yet pinned. `requirements.txt` lands with P0.2.

Ada's `/scratch` is node-local, not shared across nodes. Decide where `HF_HOME`
points before downloading any checkpoint, and stage activation caches back to
shared storage at the end of every job. See TASKS.md P0.3 and P0.9.

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
