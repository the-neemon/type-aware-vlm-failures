#!/bin/bash
# E4: every repair on every labelled error (TASKS P6.5, cross.md).
#
# Resumable: after a timeout or eviction, resubmit the same command and it
# continues from the last saved answer.
#
#   PREDICTIONS=results/predictions/qwen2_5_vl_7b_test.jsonl \
#   LABELS=annotations \
#   sbatch scripts/intervene.sbatch
#
# Synthetic arm, where I_crop has bar geometry to work with: also set
#   IMAGE_ROOT=<synth dir> FIGURES=<synth dir>/manifest.jsonl
#
#SBATCH --job-name=vlmfail-e4
#SBATCH --partition=u22
#SBATCH --constraint=2080ti          # frozen card (decision 9); same as the original run
#SBATCH --exclude=gnode068,gnode077  # mixed-GPU node; dead-driver node
#SBATCH --gres=gpu:2                 # 7B fp16 is ~16.6 GB; a 2080 Ti holds 11 GB
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=12:00:00              # explicit: the cluster default is 1 hour
#SBATCH --output=logs/e4-%j.out
#SBATCH --error=logs/e4-%j.err

set -euo pipefail
PROJ="${PROJ:-$HOME/anlp/project/type-aware-vlm-failures}"
source "$PROJ/scripts/ada_env.sh"

: "${PREDICTIONS:?set PREDICTIONS to an inf.md predictions file}"
: "${LABELS:?set LABELS to the annotations/ folder or a labels file}"
IMAGE_ROOT="${IMAGE_ROOT:-$HOME/data/ChartQA}"

require_gpu_type "2080 Ti"
stage_in_model "Qwen/Qwen2.5-VL-7B-Instruct"

# Written straight to $HOME, not node-local /scratch: a resubmitted job may land
# on a different node, and resuming needs to see every answer already saved.
OUT="$DURABLE/e4/$(basename "$PREDICTIONS" .jsonl).outcomes.jsonl"
mkdir -p "$(dirname "$OUT")"

cd "$PROJ"
python -m src.intervene.run_matrix \
    --predictions "$PREDICTIONS" --labels "$LABELS" \
    --image-root "$IMAGE_ROOT" ${FIGURES:+--figures "$FIGURES"} \
    --out "$OUT"

echo "E4 COMPLETE: $OUT"
