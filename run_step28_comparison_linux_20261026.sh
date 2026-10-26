#!/usr/bin/env bash
set -euo pipefail
cd /home/yongpeng/cross-lingual/reports/documentation/20261026/continual_comparison/workspace
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=0
export CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONDONTWRITEBYTECODE=1
exec /home/yongpeng/miniconda3/envs/py310/bin/python -B scripts/step28_comparison_control.py \
  --gate /home/yongpeng/cross-lingual/reports/documentation/20261026/continual_comparison/formal_gate.json
