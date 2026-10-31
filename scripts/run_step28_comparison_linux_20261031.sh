#!/usr/bin/env bash
set -euo pipefail
root=/home/yongpeng/cross-lingual/reports/documentation/20261031/comparison_finish
cd "$root/workspace"
export PYTHONPATH="$root/workspace/scripts"
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
exec /home/yongpeng/miniconda3/envs/py310/bin/python -B scripts/step28_comparison_finish.py --gate "$root/formal_gate.json"
