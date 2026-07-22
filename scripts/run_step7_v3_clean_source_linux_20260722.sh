#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
POLICY="schema/step7_v3_clean_source_selection_policy.json"

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "Step7-v3 requires the Linux CUDA host: nvidia-smi is unavailable." >&2
  exit 2
fi
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

"${PYTHON_BIN}" scripts/step7_v3_encode_clean_models.py \
  --policy "${POLICY}" \
  --validate-config-only

"${PYTHON_BIN}" scripts/step7_v3_encode_clean_models.py \
  --policy "${POLICY}"

echo "Label-free Step7-v3 GPU scoring is complete."
echo "Sync only the files listed in step7_v3_gpu_output_manifest.json back to Windows."
echo "Raw-encoder ranking and all 25 candidate/control fits are evaluated on Windows."
