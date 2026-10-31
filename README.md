# Type-Aware Failure Prediction for Chart Reasoning in Vision-Language Models

Advanced NLP, Monsoon 2026, IIIT Hyderabad.
Yash More, Naman Singhal, Shrish Kadam, Sanjith Ganapathi.

## Setup

The final report is [paper/final/main.pdf](paper/final/main.pdf) and the final
slides are [paper/final/slides.pdf](paper/final/slides.pdf).

GPU steps run as SLURM jobs on RTX 2080 Ti nodes; probe and analysis steps need
only a CPU and the cached activations. Keep nodes with broken GPU drivers or
mixed cards in a `BAD_NODES` shell variable and pass `--exclude="$BAD_NODES"` to
every GPU job.

Build the environment inside a job, not on the login node:

```bash
srun -p u22 --constraint=2080ti --exclude="$BAD_NODES" --gres=gpu:1 \
     -c 8 --mem=32G -t 01:00:00 bash scripts/setup_env.sh
```

## Data

ChartQA (about 875 MB, into `$HOME/data/ChartQA`), then a check of the splits:

```bash
bash scripts/download_chartqa.sh
python -m src.eval.chartqa ~/data/ChartQA
```

The three synthetic pilots (each output directory must be new; add `--verify`
to check a folder against its manifest):

```bash
python -m src.synth.absent_pairs      ~/data/synth/absent_pilot_v1 --num-charts 300
python -m src.synth.absent_lookalikes ~/data/synth/absent_pilot_v2 --num-charts 300
python -m src.synth.out_of_range      ~/data/synth/absent_pilot_v3 --num-charts 300
```

## Running the experiments

Run from the repository root, in this order. GPU steps are SLURM jobs; the rest
are CPU steps, run inside a CPU job. Set these once; run each model-specific
block for both models.

```bash
export PROJ=$PWD            # the GPU jobs look for the repository here
export DURABLE=$HOME/vlm-failures-durable   # GPU job outputs that must outlive the node
M=qwen2_5_vl_7b             # then again with M=llava_next_mistral_7b
ACT=$HOME/activations       # where the caching job writes the .npz caches
SYNTH=$HOME/data/synth
P=results/predictions
L=results/labels
```

### 1. Inference and activation caching (GPU)

```bash
C=configs                   # configs/llava for LLaVA
for run in test:inference synth_pilot:inference_synth_pilot \
           synth_pilot2:inference_synth_pilot2 synth_pilot3:inference_synth_pilot3; do
  TAG=${run%%:*} INFERENCE_CONFIG=$C/${run#*:}.yaml \
    sbatch --exclude="$BAD_NODES" scripts/cache_chartqa.sbatch
done
# Qwen only: the held-out human-written validation questions
TAG=val_human INFERENCE_CONFIG=configs/inference_val_human.yaml \
  sbatch --exclude="$BAD_NODES" scripts/cache_chartqa.sbatch
```

Writes `$P/${M}_<tag>.jsonl` (with `.run.json` and `.summary.json`) and
`$ACT/$M/<tag>.npz`.

### 2. Error labels

```bash
# LLM labels: write batches, label them, append the labels
B=labelling_batches
python -m src.label.llm_label batches --predictions $P/${M}_test.jsonl \
    --image-root ~/data/ChartQA --out $B --size 25
python -m src.label.llm_label append --batch $B/batch_001.jsonl \
    --labels $B/batch_001.labels.jsonl --out $L/${M}_test.claude.jsonl \
    --annotator "<annotator name>"
python -m src.label.error_stats --predictions $P/${M}_test.jsonl \
    --labels $L/${M}_test.claude.jsonl --out $L/${M}_test.error_stats.json

# Human agreement check (both models at once)
python -m src.label.pool sample \
    --predictions $P/qwen2_5_vl_7b_test.jsonl --predictions $P/llava_next_mistral_7b_test.jsonl \
    --labels $L/qwen2_5_vl_7b_test.claude.jsonl --labels $L/llava_next_mistral_7b_test.claude.jsonl \
    --per-model 100 --out annotations/tasks
python -m src.label.pool assign --pool annotations/tasks/pool.jsonl \
    --annotators shrish sanjith --overlap 200 --out annotations/tasks
python -m streamlit run src/label/app.py      # each annotator labels here
python -m src.label.human_agreement --raters shrish sanjith \
    --labels $L/qwen2_5_vl_7b_test.claude.jsonl \
    --labels $L/llava_next_mistral_7b_test.claude.jsonl --out results/human_agreement.md
```

### 3. Probes on ChartQA (E1, E2, E3)

```bash
# E1: will the answer be wrong? (and the source sanity probe); each writes <out>/report.json
python -m src.probes.run_sweep --target binary --activations $ACT/$M/test.npz \
    --predictions $P/${M}_test.jsonl --out runs/binary_$M
python -m src.probes.run_sweep --target source --activations $ACT/$M/test.npz \
    --predictions $P/${M}_test.jsonl --out runs/sanity_$M

# E1/E2 follow-ups: five splits, residualised, calibration; add
# --val-human $ACT/$M/val_human.npz $P/${M}_val_human.jsonl for Qwen
python -m src.probes.run_followups binary --activations $ACT/$M/test.npz \
    --predictions $P/${M}_test.jsonl --out results/probes/binary_followups_$M.json
python -m src.probes.run_followups binary --activations $ACT/$M/test.npz \
    --predictions $P/${M}_test.jsonl --drop-not-an-error $L/${M}_test.claude.jsonl \
    --out results/probes/binary_followups_clean_$M.json

# E3: misread vs. computation error
for opt in "" "--question-type arithmetic" "--question-type retrieval" "--residualise"; do
  name=$(echo "$opt" | sed 's/--question-type //; s/--residualise/residualised/')
  python -m src.probes.run_chartqa_types --activations $ACT/$M/test.npz \
      --predictions $P/${M}_test.jsonl --labels $L/${M}_test.claude.jsonl $opt \
      --out results/probes/chartqa_types_${name:+${name}_}$M
done
```

### 4. Probes on synthetic charts (E3, pilot 3)

```bash
PILOTS="--pilot pilot1 $ACT/$M/synth_pilot.npz $P/${M}_synth_pilot.jsonl $SYNTH/absent_pilot_v1/manifest.jsonl
        --pilot pilot2 $ACT/$M/synth_pilot2.npz $P/${M}_synth_pilot2.jsonl $SYNTH/absent_pilot_v2/manifest.jsonl"

python -m src.probes.run_synth_types $PILOTS --out results/probes/synth_types_$M
python -m src.probes.run_followups grid $PILOTS \
    --probe-report results/probes/synth_types_$M.json \
    --out results/probes/synth_grid_baseline_$M.json
python -m src.probes.run_pilot3 $PILOTS \
    --target $ACT/$M/synth_pilot3.npz $P/${M}_synth_pilot3.jsonl $SYNTH/absent_pilot_v3/manifest.jsonl \
    --out results/probes/pilot3_$M.json
```

### 5. Repairs and the controller (E4, E5)

```bash
# E4 (GPU): every repair on every value-question error; CORRECT=1 re-asks the correct answers for E5
for pilot in synth_pilot:absent_pilot_v1 synth_pilot2:absent_pilot_v2; do
  for correct in "" 1; do
    CORRECT=$correct PREDICTIONS=$P/${M}_${pilot%%:*}.jsonl \
      SYNTH_MANIFEST=$SYNTH/${pilot#*:}/manifest.jsonl \
      sbatch --exclude="$BAD_NODES" scripts/intervene.sbatch
  done
done
cp $DURABLE/e4/*.outcomes.jsonl results/e4/   # once the jobs finish

# E5: probe scores per model, then the controller for both models
python -m src.controller.e5_scores $PILOTS --out results/e5/${M}_scores
python -m src.controller.e5 --models qwen2_5_vl_7b llava_next_mistral_7b
```

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
