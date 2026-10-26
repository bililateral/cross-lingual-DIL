#!/usr/bin/env bash
set -euo pipefail
CONTEXT=/home/yongpeng/cross-lingual/reports/documentation/20261026/continual_defaults
WORKSPACE="$CONTEXT/waiting_workspace"
ACTIVE="$CONTEXT/active_defaults_wait.py"
test ! -e "$CONTEXT/formal"
test ! -e "$ACTIVE"
cd "$WORKSPACE"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$WORKSPACE/scripts" CUDA_VISIBLE_DEVICES=0 CUBLAS_WORKSPACE_CONFIG=:4096:8
(set -o noclobber; cat scripts/step28_defaults_control.py > "$ACTIVE")
exec /home/yongpeng/miniconda3/envs/py310/bin/python -B "$ACTIVE" \
    --workspace "$WORKSPACE" --gate "$CONTEXT/waiting/formal_gate.json"
