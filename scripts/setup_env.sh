#!/usr/bin/env bash
# P0.2: build the pinned project environment. Run this INSIDE an Ada job:
#
#   srun -p u22 --constraint=2080ti --exclude=gnode077 --gres=gpu:1 \
#        -c 8 --mem=32G -t 01:00:00 bash scripts/setup_env.sh
#
# Not on the head node: it runs an older glibc than the u22 compute nodes, so
# a venv built there will not import once a job picks it up.
#
# Where things go, and why:
#   venv  -> $HOME  It is small (6 GB), must be visible from every node, and
#                   must survive. $HOME is the only path on Ada that is
#                   shared, writable and durable at once.
#   pip cache, HF cache -> /scratch  Large and regenerable. /scratch is
#                   node-local and purged at 7 days, which is fine for things
#                   we can re-download and fatal for things we cannot.
# See results/ada_filesystem.md for the measurements behind that split.
set -euo pipefail

VENV="${VENV:-$HOME/envs/vlmfail}"
SCRATCH_ROOT="${SCRATCH_ROOT:-/scratch/vlm-failures}"
PROJ="${PROJ:-$HOME/anlp/project/type-aware-vlm-failures}"

echo "=== node: $(hostname) ==="
mkdir -p "$SCRATCH_ROOT/hf" "$SCRATCH_ROOT/pipcache"
chmod 0777 "$SCRATCH_ROOT" "$SCRATCH_ROOT/hf" "$SCRATCH_ROOT/pipcache" 2>/dev/null || true
export PIP_CACHE_DIR="$SCRATCH_ROOT/pipcache"

/bin/python3 -m venv "$VENV"
# shellcheck disable=SC1091
source "$VENV/bin/activate"
python -V
pip install -q -U pip wheel setuptools
pip install -q -r "$PROJ/requirements.txt"

echo "=== verifying the pins actually loaded ==="
python - <<'PY'
import torch, transformers, accelerate, datasets, numpy, PIL
print("torch       ", torch.__version__)
print("transformers", transformers.__version__)
print("accelerate  ", accelerate.__version__)
print("datasets    ", datasets.__version__)
print("numpy       ", numpy.__version__)
print("pillow      ", PIL.__version__)
assert torch.cuda.is_available(), "no CUDA visible; ask for --gres=gpu:1"
cap = torch.cuda.get_device_capability(0)
print("gpu         ", torch.cuda.get_device_name(0), cap)
assert cap == (7, 5), f"expected Turing sm_75 (2080 Ti), got {cap}"
a = torch.randn(512, 512, device="cuda", dtype=torch.float16)
assert (a @ a).float().sum().isfinite(), "fp16 matmul produced non-finite output"
print("fp16 matmul  OK")
from transformers import Qwen2_5_VLForConditionalGeneration  # noqa: F401
print("Qwen2_5_VL class importable")
PY
echo "=== done. venv at $VENV ==="
du -sh "$VENV"
