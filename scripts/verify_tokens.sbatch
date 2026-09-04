#!/bin/bash
# P0.10: check the analytic vision-token counts in src/extract/storage_budget.py
# against what Qwen2VLImageProcessor actually produces. Exits non-zero on any
# mismatch, which would mean configs/activations.yaml needs recomputing before
# any caching starts.
#
#   sbatch scripts/verify_tokens.sbatch
#
#SBATCH --job-name=vlmfail-tokens
#SBATCH --partition=u22
#SBATCH --constraint=2080ti
#SBATCH --exclude=gnode077
#SBATCH --gres=gpu:1                 # processor only, but keep the node class identical
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:30:00
#SBATCH --output=logs/tokens-%j.out
#SBATCH --error=logs/tokens-%j.err

set -euo pipefail
PROJ="${PROJ:-$HOME/anlp/project/type-aware-vlm-failures}"
source "$PROJ/scripts/ada_env.sh"

stage_in_model "Qwen/Qwen2.5-VL-7B-Instruct"

python "$PROJ/scripts/fetch_chartqa_sample.py" \
    --out "$SCRATCH_ROOT/chartqa_sample" --n 5

# --max-pixels must match models.qwen2_5_vl_7b.max_pixels in configs/activations.yaml
python "$PROJ/scripts/verify_vision_tokens.py" \
    --max-pixels 1000000 \
    "$SCRATCH_ROOT"/chartqa_sample/*.png
