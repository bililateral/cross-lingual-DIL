#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
command_name="${1:-run}"
if [[ "$command_name" == "run" ]]; then
  nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
fi

env -u LD_LIBRARY_PATH -u LD_PRELOAD \
  CUBLAS_WORKSPACE_CONFIG=:4096:8 \
  TOKENIZERS_PARALLELISM=false \
  HF_HUB_OFFLINE=1 \
  TRANSFORMERS_OFFLINE=1 \
  PYTHONDONTWRITEBYTECODE=1 \
  python -B scripts/step28_v13_v1_13_v9_4_1_replay_full_english_compatibility_linux_v2.py "$@"
