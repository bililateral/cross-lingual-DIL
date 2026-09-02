#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-validate-contract}"
if [[ "$MODE" != "validate-contract" && "$MODE" != "smoke" && "$MODE" != "run" ]]; then
  echo "用法: bash $0 [validate-contract|smoke|run]" >&2
  exit 2
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
env -u LD_LIBRARY_PATH -u LD_PRELOAD \
  CUBLAS_WORKSPACE_CONFIG=:4096:8 \
  TOKENIZERS_PARALLELISM=false \
  HF_HUB_OFFLINE=1 \
  TRANSFORMERS_OFFLINE=1 \
  PYTHONDONTWRITEBYTECODE=1 \
  python -B scripts/step28_v13_v1_13_v9_4_1_english_initialized_labse_finetune_linux_v1.py "$MODE"
