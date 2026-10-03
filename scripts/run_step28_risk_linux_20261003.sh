#!/usr/bin/env bash
set -euo pipefail

workspace="$(realpath -- "${1:?Pass the isolated reviewed workspace}")"
cpu="${2:?Pass the inspected CPU number}"
self="$(realpath -- "$0")"
project=/home/yongpeng/cross-lingual
case "$workspace" in
  "$project/reports/seller_alias_continual/20261003/risk_execution/"*/workspace) ;;
  *) exit 2 ;;
esac
[[ "$cpu" =~ ^[0-9]+$ && "$self" == "$workspace/scripts/run_step28_risk_linux_20261003.sh" ]]
cd "$workspace"
python=/home/yongpeng/miniconda3/envs/py310/bin/python
audit="$workspace/reports/cpu/audit.json"
authorization="$workspace/reports/authorization.json"
out="$workspace/reports/job"
queue="$workspace/reports/listener"
mkdir "$queue"
printf '%s\n' "$$" > "$queue/pid.txt"
date --iso-8601=seconds > "$queue/started.txt"
sha256sum -- "$self" > "$queue/script.sha256"
exec >> "$queue/events.log" 2>&1
trap 's=$?; printf "%s\n" "$s" > "$queue/exit_status.txt"; date --iso-8601=seconds > "$queue/finished.txt"' EXIT
unset LD_LIBRARY_PATH LD_PRELOAD
export CUDA_VISIBLE_DEVICES=0 CUBLAS_WORKSPACE_CONFIG=:4096:8
export TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

# Read saved CPU evidence and policy only. Waiting never enters a label parser.
"$python" -B - "$audit" "$authorization" <<'PY'
import json, sys
from pathlib import Path
audit, auth = [json.loads(Path(p).read_text()) for p in sys.argv[1:]]
assert audit['status'] == 'PASS_RISK_REPLAY_HANDMADE_CPU'
assert audit['contracts']['failed'] == 0 and audit['contracts']['skipped'] == 0
assert audit['native_risk_gradient_increment_verified']
assert audit['source_files'] == auth['source_files']
assert auth['review_and_primary_passed']
PY

while true; do
  # Include 64 MiB for a CUDA context above the frozen 24 GiB free-memory gate.
  gpu_free=$(nvidia-smi -i 0 --query-gpu=memory.free --format=csv,noheader,nounits)
  host_free=$(awk '/^MemAvailable:/ {print $2}' /proc/meminfo)
  disk_free=$(df -B1 --output=avail "$workspace" | tail -n 1 | tr -d ' ')
  read -r _ a b c d e f g h _ < <(grep -E "^cpu${cpu} " /proc/stat)
  total_before=$((a+b+c+d+e+f+g+h)); idle_before=$((d+e))
  sleep 2
  read -r _ a b c d e f g h _ < <(grep -E "^cpu${cpu} " /proc/stat)
  total_delta=$((a+b+c+d+e+f+g+h-total_before))
  idle_delta=$((d+e-idle_before))
  printf '%s gpu_free_mib=%s host_free_kib=%s disk_free_bytes=%s cpu=%s idle=%s/%s\n' \
    "$(date --iso-8601=seconds)" "$gpu_free" "$host_free" "$disk_free" "$cpu" "$idle_delta" "$total_delta"
  if ((gpu_free >= 24640 && host_free >= 16777216 && disk_free >= 17179869184 \
       && total_delta > 0 && idle_delta * 100 >= total_delta * 90)); then
    break
  fi
  sleep 58
done

mkdir "$out"
date --iso-8601=seconds > "$out/started.txt"
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv \
  > "$queue/gpu_processes_at_launch.csv"
set +e
/usr/bin/time -v -o "$out/resource_usage.log" \
  timeout --signal=TERM --kill-after=30s 12h \
  taskset -c "$cpu" "$python" -u -B scripts/step28_er_weight_run.py execute \
  --study risk --out "$out" --audit "$audit" --authorization "$authorization" \
  > "$out/train.log" 2>&1 &
child=$!
printf '%s\n' "$child" > "$queue/training_wrapper_pid.txt"
while kill -0 "$child" 2>/dev/null; do
  if grep -q '"event":"updates"' "$out/train.log"; then
    # Delete only this listener after actual new updates, preserving launch records.
    rm -- "$self"
    date --iso-8601=seconds > "$queue/script_deleted_after_updates.txt"
    break
  fi
  sleep 15
done
wait "$child"
status=$?
printf '%s\n' "$status" > "$out/exit_status.txt"
date --iso-8601=seconds > "$out/finished.txt"
printf '%s exit=%s; no automatic retry\n' "$(date --iso-8601=seconds)" "$status"
exit "$status"
