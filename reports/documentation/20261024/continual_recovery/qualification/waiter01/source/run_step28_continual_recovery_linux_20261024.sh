#!/usr/bin/env bash
set -eu
root="$(cd "$(dirname "$0")" && pwd)"
cd "$root"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false PYTHONDONTWRITEBYTECODE=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8 CUDA_VISIBLE_DEVICES=0
exec /home/yongpeng/miniconda3/envs/py310/bin/python -B \
    "$root/scripts/step28_continual_five_recover.py" wait \
    --gate "$root/authorization.json" --self-delete "$0"
