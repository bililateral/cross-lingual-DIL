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
  echo "存在未处理运行：$building；先记录失败边界和必要证据，再清理，不自动覆盖重跑。" >&2
  exit 1
fi

stage="environment"
cleanup_failed_run() {
  status=$?
  if [[ $status -ne 0 && -e "$building" ]]; then
    cat > "$building/runtime_failure.txt" <<EOF
run=$output
command=$stage
exit_status=$status
status=RUNTIME_FAILURE_REQUIRES_DIAGNOSIS_NO_SCIENTIFIC_CONCLUSION
Record the actual cause, exposed evaluation boundary and critical hashes in docs/.
Then promptly remove failed payloads and temporary checkpoints. Do not retain them for possible later inspection.
EOF
    echo "运行失败，已保留小型进度和失败回执：$building。执行者须立即诊断、记录后清理大型产物。" >&2
  fi
  exit "$status"
}
trap cleanup_failed_run EXIT

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

run_python() {
  stage="$1"
  env -u LD_LIBRARY_PATH -u LD_PRELOAD \
    CUBLAS_WORKSPACE_CONFIG=:4096:8 \
    TOKENIZERS_PARALLELISM=false \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    python -B scripts/step28_style_transfer_source_linux.py "$1"
}

run_python validate
run_python smoke
run_python run

trap - EXIT
echo "Step28 V6 英文风格来源初始化与中文开发零样本阶段完成：$output"
