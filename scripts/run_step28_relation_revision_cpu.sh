#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export CUDA_VISIBLE_DEVICES=''
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false PYTHONDONTWRITEBYTECODE=1
# CPU identity is supplied only after shared-host resource inspection.
exec timeout --signal=TERM --kill-after=5s 595s taskset -c "${1:?CPU index required}" \
    /home/yongpeng/miniconda3/envs/py310/bin/python -B scripts/step28_relation_revision_verify.py
