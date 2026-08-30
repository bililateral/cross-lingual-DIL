#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
if [[ "$#" -gt 0 ]]; then
  PYTHONDONTWRITEBYTECODE=1 \
    python -B scripts/step28_v13_v1_13_v9_4_1_replay_compatibility_fixture_linux_v1.py "$@"
  exit $?
fi
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
env -u LD_LIBRARY_PATH -u LD_PRELOAD \
  CUBLAS_WORKSPACE_CONFIG=:4096:8 \
  TOKENIZERS_PARALLELISM=false \
  HF_HUB_OFFLINE=1 \
  TRANSFORMERS_OFFLINE=1 \
  PYTHONDONTWRITEBYTECODE=1 \
  python -B scripts/step28_v13_v1_13_v9_4_1_replay_compatibility_fixture_linux_v1.py
