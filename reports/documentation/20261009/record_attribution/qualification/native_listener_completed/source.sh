#!/usr/bin/env bash
set -euo pipefail
project=/home/yongpeng/cross-lingual
base="$project/reports/seller_alias_continual/20261009/record_attribution_execution/20261009_182606"
workspace="$base/workspace"
cd "$workspace"
self="$(realpath -- "$0")"
test "$self" = "$workspace/scripts/run_step28_record_attribution_native_linux_20261009.sh"
qualification="$workspace/reports/qualification"
queue="$qualification/native_listener"
attempt="$qualification/gpu/attempt01"
out="$attempt/result"
gate="$qualification/native_gate.json"
python=/home/yongpeng/miniconda3/envs/py310/bin/python
test -f "$gate"
test ! -e "$attempt"
mkdir "$queue"
cp -- "$self" "$queue/source.sh"
printf '%s\n' "$$" > "$queue/pid.txt"
date --iso-8601=seconds > "$queue/started.txt"
sha256sum -- "$self" > "$queue/script.sha256"
exec >> "$queue/events.log" 2>&1
trap 's=$?; printf "%s\n" "$s" > "$queue/exit_status.txt"; date --iso-8601=seconds > "$queue/finished.txt"' EXIT
trap 'exit 143' TERM
trap 'exit 130' INT
unset LD_LIBRARY_PATH LD_PRELOAD
export CUDA_VISIBLE_DEVICES=0 CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTHONDONTWRITEBYTECODE=1
while true; do
    gpu_free=$(nvidia-smi -i 0 --query-gpu=memory.free --format=csv,noheader,nounits)
    gpu_jobs=$(nvidia-smi -i 0 --query-compute-apps=pid --format=csv,noheader,nounits)
    project_jobs=$(taskset -c 0 "$python" - <<'PY'
import psutil
process = psutil.Process()
ancestors = {process.pid, *(p.pid for p in process.parents())}
for p in psutil.process_iter(["pid", "username", "cmdline"]):
    if p.pid not in ancestors and any("step28" in arg for arg in (p.info["cmdline"] or [])):
        print(p.pid, p.info["username"], " ".join(p.info["cmdline"]))
PY
    )
    host_free=$(awk '/^MemAvailable:/ {print $2}' /proc/meminfo)
    disk_free=$(df -B1 --output=avail "$workspace" | tail -n 1 | tr -d ' ')
    read -r _ a b c d e f g h _ < <(grep -E '^cpu0 ' /proc/stat)
    total_before=$((a+b+c+d+e+f+g+h)); idle_before=$((d+e))
    sleep 2
    read -r _ a b c d e f g h _ < <(grep -E '^cpu0 ' /proc/stat)
    total_delta=$((a+b+c+d+e+f+g+h-total_before)); idle_delta=$((d+e-idle_before))
    printf '%s gpu_free_mib=%s gpu_pids=%s host_free_kib=%s disk_free_bytes=%s cpu0_idle=%s/%s project_jobs=%s\n' \
        "$(date --iso-8601=seconds)" "$gpu_free" "${gpu_jobs//$'\n'/,}" "$host_free" "$disk_free" \
        "$idle_delta" "$total_delta" "${project_jobs//$'\n'/,}"
    if [[ -z "$gpu_jobs" && -z "$project_jobs" ]] && ((gpu_free >= 30720 && host_free >= 75497472 \
            && disk_free >= 51539607552 && total_delta > 0 && idle_delta * 100 >= total_delta * 90)); then
        break
    fi
    sleep 58
done
mkdir -p "$attempt"
date --iso-8601=seconds > "$attempt/started.txt"
date --iso-8601=seconds > "$queue/launched.txt"
set +e
/usr/bin/time -v -o "$attempt/resources.txt" timeout --signal=TERM --kill-after=5 590 \
    taskset -c 0 "$python" -u -B scripts/step28_record_attribution_verify.py gpu \
    --output "$out" --gate "$gate" --prior-seconds 0 > "$attempt/console.txt" 2>&1 &
child=$!
printf '%s\n' "$child" > "$queue/verification_wrapper_pid.txt"
actual_update_recorded() {
    # These progress states occur only after a real first update and changed-weight checks.
    [[ -f "$out/result.json" ]] && grep -Eq \
        '"stage":[[:space:]]*"(native_paired_baseline_|native_diagnostic_|native_load_R0|native_first_allocation_R0|native_load_S|native_first_allocation_S)' \
        "$out/result.json"
}
while kill -0 "$child" 2>/dev/null; do
    if actual_update_recorded; then
        rm -- "$self"
        date --iso-8601=seconds > "$queue/script_deleted_after_update.txt"
        break
    fi
    sleep 1
done
wait "$child"
status=$?
if actual_update_recorded && [[ -f "$self" ]]; then
    rm -- "$self"
    date --iso-8601=seconds > "$queue/script_deleted_after_update.txt"
fi
printf '%s\n' "$status" > "$attempt/exit_status.txt"
date --iso-8601=seconds > "$attempt/finished.txt"
printf '%s verification_exit=%s; no automatic retry; formal training requires qualification disposition\n' "$(date --iso-8601=seconds)" "$status"
exit "$status"
