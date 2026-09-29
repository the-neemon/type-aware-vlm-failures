#!/bin/bash
# Qwen2.5-VL-7B over ChartQA test: predictions plus prefill activations
# (inf.md, qwen_chartqa_activation_caching_spec.md, TASKS P1.3 and P4.1-P4.4).
#
#   LIMIT=6 SHARD_SIZE=4 TAG=smoke sbatch scripts/cache_chartqa.sbatch   # first: a smoke run
#   sbatch scripts/cache_chartqa.sbatch                                  # then: all 2,500
#
# Another ChartQA split: point INFERENCE_CONFIG at a config whose dataset block
# names it, and give it its own TAG so its shards and outputs never mix with
# the test run's:
#
#   TAG=val_human INFERENCE_CONFIG=configs/inference_val_human.yaml \
#       sbatch scripts/cache_chartqa.sbatch
#
# Synthetic charts work the same way, once generated (the config says how):
#
#   TAG=synth_pilot INFERENCE_CONFIG=configs/inference_synth_pilot.yaml \
#       sbatch scripts/cache_chartqa.sbatch
#
# LLaVA-NeXT: the same commands with its configs, which live in configs/llava/.
# The model and its output names come from the config's model_key:
#
#   LIMIT=6 SHARD_SIZE=4 TAG=smoke INFERENCE_CONFIG=configs/llava/inference.yaml \
#       sbatch scripts/cache_chartqa.sbatch
#   TAG=test INFERENCE_CONFIG=configs/llava/inference.yaml sbatch scripts/cache_chartqa.sbatch
#
# LLaVA's prompt is ~4x longer than Qwen's (2,242 image tokens for an 800x557
# figure against 580), so a 2,500-item run may need more than one 12 h slot.
# Resubmit, or pass a longer limit on the command line: sbatch --time=24:00:00 ...
#
# One GPU (a QOS that caps GPUs per user): LLaVA-NeXT runs on one 2080 Ti by
# holding the weights that do not fit in CPU RAM (src/extract/llava.py
# `placement`); same activations, slower. Override the request below:
#
#   TAG=test INFERENCE_CONFIG=configs/llava/inference.yaml sbatch --gres=gpu:1 scripts/cache_chartqa.sbatch
#
# Qwen has no single-GPU path; it needs both cards.
#
# Resumable: after a timeout or eviction, resubmit the same command. Items whose
# activations are already in a shard are skipped, on whichever node the job
# lands. Resubmitting the smoke run once it has finished is also a cheap check
# that resuming works: it should report every item as already cached.
#
# A smoke run (LIMIT set) keeps its predictions out of the repo, in
# $DURABLE/smoke, so a half-run file is never committed by mistake.
#
#SBATCH --job-name=vlmfail-cache
#SBATCH --partition=u22
#SBATCH --constraint=2080ti          # frozen card (decision 9)
#SBATCH --exclude=gnode066,gnode068,gnode077  # dead-GPU node; mixed-GPU node; dead-driver node
#SBATCH --gres=gpu:2                 # 7B fp16 is ~16.6 GB; a 2080 Ti holds 11 GB
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G                    # final assembly holds the ~1.9 GiB cache twice
#SBATCH --time=12:00:00              # ~7 h expected; the cluster default is 1 hour
#SBATCH --output=logs/cache-%j.out
#SBATCH --error=logs/cache-%j.err

set -euo pipefail
PROJ="${PROJ:-$HOME/anlp/project/type-aware-vlm-failures}"
source "$PROJ/scripts/ada_env.sh"

TAG="${TAG:-test}"
SHARD_SIZE="${SHARD_SIZE:-250}"
ACT_DIR="${ACT_DIR:-$HOME/activations}"
if [ -n "${LIMIT:-}" ]; then
    PRED_DIR="${PRED_DIR:-$DURABLE/smoke}"
else
    PRED_DIR="${PRED_DIR:-$PROJ/results/predictions}"
fi

INFERENCE_CONFIG="${INFERENCE_CONFIG:-$PROJ/configs/inference.yaml}"
case "$INFERENCE_CONFIG" in /*) ;; *) INFERENCE_CONFIG="$PROJ/$INFERENCE_CONFIG" ;; esac

# The model is whatever the config names, so the checkpoint staged in and the
# output directory can never disagree with what cache_chartqa loads.
read -r MODEL_KEY MODEL_ID < <(python - "$INFERENCE_CONFIG" <<'PY'
import sys, yaml
cfg = yaml.safe_load(open(sys.argv[1]))
print(cfg["model_key"], cfg["model_id"])
PY
)
: "${MODEL_ID:?could not read model_key and model_id from $INFERENCE_CONFIG}"
echo "model       : $MODEL_KEY ($MODEL_ID) from $INFERENCE_CONFIG"

ARGS=(--tag "$TAG" --predictions-dir "$PRED_DIR" --activations-dir "$ACT_DIR"
      --inference-config "$INFERENCE_CONFIG")
if [ -n "${LIMIT:-}" ]; then ARGS+=(--limit "$LIMIT"); fi

require_gpu_type "2080 Ti"
stage_in_model "$MODEL_ID"
bash "$PROJ/scripts/download_chartqa.sh"      # idempotent; a no-op once present

echo "quota before the run:"; quota -s | tail -3 || true

cd "$PROJ"
python -u -m src.extract.cache_chartqa run "${ARGS[@]}" --shard-size "$SHARD_SIZE"

# Compress on node-local disk, then stage back (P0.9): stage_out fails the job
# loudly if the copy does not reach $HOME.
WORK="$SCRATCH_ROOT/activations/$MODEL_KEY"
mkdir -p "$WORK"
python -u -m src.extract.cache_chartqa assemble "${ARGS[@]}" --out "$WORK/$TAG.npz"
stage_out "$WORK/$TAG.npz" "$ACT_DIR/$MODEL_KEY"

# Checks the staged copy in $HOME, not the one on /scratch.
python -u -m src.extract.cache_chartqa validate "${ARGS[@]}"

echo "CACHE COMPLETE: $ACT_DIR/$MODEL_KEY/$TAG.npz"
