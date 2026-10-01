#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-smoke}"
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
  python -B scripts/step28_labse_direct_finetune_linux.py "$MODE"
