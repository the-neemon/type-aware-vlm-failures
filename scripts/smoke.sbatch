#!/bin/bash
# P0.5 deliverable: one Qwen2.5-VL-7B answer to one ChartQA figure.
#
# Deliberately minimal. No batching, no scoring, no prompt tuning. The only
# question it answers is whether the environment, the weights, the GPUs and
# the processor path all work together on Ada.
#
#   sbatch scripts/smoke.sbatch
#
#SBATCH --job-name=vlmfail-smoke
#SBATCH --partition=u22
#SBATCH --constraint=2080ti          # nodes are mixed; freeze the card (decision 9)
#SBATCH --exclude=gnode077           # known dead GPU
#SBATCH --gres=gpu:2                 # 7B fp16 is ~16.6 GB; a 2080 Ti holds 11 GB
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=02:00:00              # explicit: the cluster default is 1 hour
#SBATCH --output=logs/smoke-%j.out
#SBATCH --error=logs/smoke-%j.err

set -euo pipefail
PROJ="${PROJ:-$HOME/anlp/project/type-aware-vlm-failures}"
source "$PROJ/scripts/ada_env.sh"

require_gpu_type "2080 Ti"
stage_in_model "Qwen/Qwen2.5-VL-7B-Instruct"

# One figure, one question. Fetched once into node-local scratch; the fetch is
# a no-op if the sample is already there.
python "$PROJ/scripts/fetch_chartqa_sample.py" \
    --out "$SCRATCH_ROOT/chartqa_sample" --n 5

python "$PROJ/src/extract/smoke_qwen.py" \
    --sample-dir "$SCRATCH_ROOT/chartqa_sample" \
    --out "$SCRATCH_ROOT/smoke_result.json"

# P0.9: node-local /scratch is invisible from anywhere else. Stage back or the
# result is stranded on this node.
stage_out "$SCRATCH_ROOT/smoke_result.json" "$DURABLE/smoke"

echo "SMOKE TEST COMPLETE"
