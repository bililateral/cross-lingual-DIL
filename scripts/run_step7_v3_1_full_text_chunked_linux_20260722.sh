#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
REQUESTED_PYTHON_BIN="${PYTHON_BIN}"
POLICY="schema/step7_v3_1_full_text_chunked_selection_policy.json"
RUNNER="scripts/run_step7_v3_1_full_text_chunked_linux_20260722.sh"
STAGER="scripts/step7_v3_1_materialize_gpu_workspace.py"
SOURCE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${SOURCE_ROOT}"

if ! PYTHON_BIN="$(command -v "${PYTHON_BIN}")"; then
  echo "Step7-v3.1 Python executable is unavailable: ${REQUESTED_PYTHON_BIN}" >&2
  exit 2
fi

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "Step7-v3.1 requires the Linux CUDA host: nvidia-smi is unavailable." >&2
  exit 2
fi
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

if [[ "${STEP7_V3_1_ISOLATED_WORKSPACE:-0}" != "1" ]]; then
  if [[ -n "${STEP7_V3_1_GPU_WORKSPACE:-}" ]]; then
    GPU_WORKSPACE="${STEP7_V3_1_GPU_WORKSPACE}"
    if [[ -e "${GPU_WORKSPACE}" ]]; then
      echo "STEP7_V3_1_GPU_WORKSPACE must name a new directory: ${GPU_WORKSPACE}" >&2
      exit 2
    fi
    mkdir -p "${GPU_WORKSPACE}"
  else
    GPU_WORKSPACE="$(mktemp -d "$(dirname "${SOURCE_ROOT}")/step7-v3-1-gpu-20260722.XXXXXX")"
  fi
  GPU_WORKSPACE="$(cd "${GPU_WORKSPACE}" && pwd)"
  "${PYTHON_BIN}" "${STAGER}" stage --destination "${GPU_WORKSPACE}"
  (
    cd "${GPU_WORKSPACE}"
    STEP7_V3_1_ISOLATED_WORKSPACE=1 \
      PYTHON_BIN="${PYTHON_BIN}" \
      bash "${RUNNER}"
  )
  "${PYTHON_BIN}" "${STAGER}" collect --workspace "${GPU_WORKSPACE}"
  echo "Label-free Step7-v3.1 full-text GPU encoding is complete."
  echo "Verified outputs were copied back to: ${SOURCE_ROOT}"
  echo "The isolated audit workspace was retained at: ${GPU_WORKSPACE}"
  exit 0
fi

"${PYTHON_BIN}" scripts/step7_v3_1_build_sync_manifest.py \
  --policy "${POLICY}" \
  --validate-only

"${PYTHON_BIN}" scripts/step7_v3_1_encode_chunked_models.py \
  --policy "${POLICY}" \
  --validate-config-only

"${PYTHON_BIN}" scripts/step7_v3_1_encode_chunked_models.py \
  --policy "${POLICY}"

echo "Isolated Step7-v3.1 GPU encoding completed; returning to source workspace."
