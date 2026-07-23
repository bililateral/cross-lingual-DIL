#!/usr/bin/env bash
set -euo pipefail

SOURCE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
WORKSPACE="$(mktemp -d "${HOME}/step7-v4-gpu-20260723.XXXXXX")"

cleanup() {
  status=$?
  if [[ $status -eq 0 && "$WORKSPACE" == "${HOME}/step7-v4-gpu-20260723."* ]]; then
    rm -rf -- "$WORKSPACE"
  else
    echo "Step7-v4 failed; isolated workspace retained for audit: $WORKSPACE" >&2
  fi
  exit "$status"
}
trap cleanup EXIT

cd "$SOURCE_ROOT"
"$PYTHON_BIN" scripts/step7_v4_prepare_source_data.py \
  --stage validate-existing
"$PYTHON_BIN" scripts/step7_v4_build_sync_manifest.py --validate-only
"$PYTHON_BIN" scripts/step7_v4_select_source_model.py --validate-config-only
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

"$PYTHON_BIN" scripts/step7_v4_materialize_gpu_workspace.py \
  stage --destination "$WORKSPACE"

cd "$WORKSPACE"
"$PYTHON_BIN" scripts/step7_v4_encode_item_models.py --validate-inputs-only
"$PYTHON_BIN" scripts/step7_v4_encode_item_models.py

cd "$SOURCE_ROOT"
"$PYTHON_BIN" scripts/step7_v4_materialize_gpu_workspace.py \
  collect --workspace "$WORKSPACE"

# The selector runs only after compact GPU outputs have been verified and
# collected.  It reads train labels first, locks every choice, and opens valid
# labels only for the final diagnostic. Historical test labels remain absent.
"$PYTHON_BIN" scripts/step7_v4_select_source_model.py
