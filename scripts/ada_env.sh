# Shared environment preamble for every Ada job in this project.
# Source it, do not execute it:  source "$(dirname "$0")/ada_env.sh"
#
# Why this file exists: the SPEC assumed /scratch was a shared filesystem on
# Ada. It is not. /scratch is node-local disk, so the model cache has to be
# repaired per node, and anything durable has to be staged back to $HOME.
# Centralising that here keeps every job honest about it.

set -euo pipefail

# --- Paths -----------------------------------------------------------------
# PROJ is the repo checkout. DURABLE is the only storage every node can see.
# SCRATCH_ROOT is node-local: fast and huge, but invisible from anywhere else.
export PROJ="${PROJ:-$HOME/anlp/project/type-aware-vlm-failures}"
export DURABLE="${DURABLE:-$HOME/anlp/project/vlm-failures-durable}"
export SCRATCH_ROOT="${SCRATCH_ROOT:-/scratch/vlm-failures}"
export VENV="${VENV:-$HOME/envs/vlmfail}"

# HF_HOME must NOT live in $HOME: home quota is 30 GB and two 7B checkpoints
# are roughly 33 GB combined. It goes on node-local /scratch and is repaired
# on demand by stage_in_model below.
export HF_HOME="$SCRATCH_ROOT/hf"
export HF_HUB_ENABLE_HF_TRANSFER=0
export PIP_CACHE_DIR="$SCRATCH_ROOT/pipcache"
export TOKENIZERS_PARALLELISM=false

# --- Determinism -----------------------------------------------------------
# Every downstream label refers to one specific answer string (TASKS.md P1.2),
# so the answer has to be reproducible.
export PYTHONHASHSEED=0
export CUBLAS_WORKSPACE_CONFIG=:4096:8

mkdir -p "$SCRATCH_ROOT/hf" "$SCRATCH_ROOT/pipcache" "$DURABLE"
chmod 0777 "$SCRATCH_ROOT" "$SCRATCH_ROOT/hf" "$SCRATCH_ROOT/pipcache" 2>/dev/null || true

# shellcheck disable=SC1091
source "$VENV/bin/activate"

echo "--- ada_env ---"
echo "node        : $(hostname)"
echo "job         : ${SLURM_JOB_ID:-<none>}"
echo "gpus        : ${CUDA_VISIBLE_DEVICES:-<none>}"
echo "HF_HOME     : $HF_HOME"
echo "DURABLE     : $DURABLE"
echo "python      : $(python -V 2>&1)"
echo "---------------"

# Fail loudly if we landed on a card we did not ask for. Ada's nodes are mixed
# and configs/activations.yaml freezes one GPU type for the whole project,
# because answers and activations must come from the same kernels (P4.4).
require_gpu_type() {
    local want="$1"
    local got
    got="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
    if ! grep -qi "$want" <<<"$got"; then
        echo "FATAL: expected GPU matching '$want', got '$got'." >&2
        echo "Activations from a different card are not comparable. Aborting." >&2
        exit 1
    fi
    echo "gpu check   : OK ($got)"
}

# /scratch is node-local, so the cache may be cold on whichever node we landed
# on. This is idempotent: warm cache exits in seconds, cold cache downloads.
stage_in_model() {
    local repo="$1"
    echo "staging in $repo into $HF_HOME (no-op if already cached) ..."
    python - "$repo" <<'PY'
import sys
from huggingface_hub import snapshot_download
repo = sys.argv[1]
path = snapshot_download(repo_id=repo, allow_patterns=None)
print("cached at:", path)
PY
}

# P0.9: anything durable must leave node-local /scratch before the job exits,
# and the job must fail loudly if it does not.
stage_out() {
    local src="$1" dst="$2"
    mkdir -p "$dst"
    if ! cp -a "$src" "$dst"/; then
        echo "FATAL: stage-out of $src -> $dst failed. Output is stranded on $(hostname):/scratch." >&2
        exit 1
    fi
    echo "staged out: $src -> $dst"
}
