#!/usr/bin/env bash
set -euo pipefail
cd /home/yongpeng/cross-lingual/reports/documentation/20261030/comparison_resume/workspace
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false PYTHONDONTWRITEBYTECODE=1
exec /home/yongpeng/miniconda3/envs/py310/bin/python -B scripts/step28_comparison_control.py \
  --gate /home/yongpeng/cross-lingual/reports/documentation/20261030/comparison_resume/formal_gate.json
