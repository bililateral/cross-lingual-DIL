#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

output="reports/step28_model_experiment/v9_4_1_v6_style_transfer_source_v2_20260904"
building="$output.building"

if [[ -e "$output" ]]; then
  echo "正式输出已存在，拒绝覆盖：$output" >&2
  exit 1
fi
if [[ -e "$building" ]]; then
  rm -rf -- "$building"
fi

cleanup_failed_run() {
  status=$?
  if [[ $status -ne 0 && -e "$building" ]]; then
    rm -rf -- "$building"
    echo "失败的临时产物已删除：$building" >&2
  fi
  exit "$status"
}
trap cleanup_failed_run EXIT

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

run_python() {
  env -u LD_LIBRARY_PATH -u LD_PRELOAD \
    CUBLAS_WORKSPACE_CONFIG=:4096:8 \
    TOKENIZERS_PARALLELISM=false \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    python -B scripts/step28_v13_v1_13_v9_4_1_v6_style_transfer_source_linux_v2.py "$1"
}

run_python validate
run_python smoke
run_python run

trap - EXIT
echo "Step28 V6 英文风格来源初始化与中文开发零样本阶段完成：$output"
