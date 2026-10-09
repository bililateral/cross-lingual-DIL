#!/usr/bin/env bash
set -euo pipefail
project=/home/yongpeng/cross-lingual
cd "$project"
self="$(realpath -- "$0")"
test "$self" = "$project/scripts/run_step28_numerics_linux_20261009.sh"
root="$project/reports/maintenance/20261009/runtime_numerics_gpu"
queue="$root/listener"
out="$root/attempt01"
python=/home/yongpeng/miniconda3/envs/py310/bin/python
test ! -e "$out"
# Atomic creation prevents duplicate listeners for the same authorized attempt.
mkdir "$queue"
printf '%s\n' "$$" > "$queue/pid.txt"
date --iso-8601=seconds > "$queue/started.txt"
sha256sum -- "$self" > "$queue/script.sha256"
exec >> "$queue/events.log" 2>&1
trap 's=$?; printf "%s\n" "$s" > "$queue/exit_status.txt"; date --iso-8601=seconds > "$queue/finished.txt"' EXIT
unset LD_LIBRARY_PATH LD_PRELOAD
export CUDA_VISIBLE_DEVICES=0 CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTHONDONTWRITEBYTECODE=1
while true; do
    gpu_free=$(nvidia-smi -i 0 --query-gpu=memory.free --format=csv,noheader,nounits)
    gpu_jobs=$(nvidia-smi -i 0 --query-compute-apps=pid --format=csv,noheader,nounits)
    host_free=$(awk '/^MemAvailable:/ {print $2}' /proc/meminfo)
    disk_free=$(df -B1 --output=avail "$root" | tail -n 1 | tr -d ' ')
    read -r _ a b c d e f g h _ < <(grep -E '^cpu0 ' /proc/stat)
    total_before=$((a+b+c+d+e+f+g+h)); idle_before=$((d+e))
    sleep 2
    read -r _ a b c d e f g h _ < <(grep -E '^cpu0 ' /proc/stat)
    total_delta=$((a+b+c+d+e+f+g+h-total_before)); idle_delta=$((d+e-idle_before))
    printf '%s gpu_free_mib=%s gpu_pids=%s host_free_kib=%s disk_free_bytes=%s cpu0_idle=%s/%s\n' \
        "$(date --iso-8601=seconds)" "$gpu_free" "${gpu_jobs//$'\n'/,}" "$host_free" "$disk_free" "$idle_delta" "$total_delta"
    if [[ -z "$gpu_jobs" ]] && ((gpu_free >= 30720 && host_free >= 75497472 && disk_free >= 4294967296 \
            && total_delta > 0 && idle_delta * 100 >= total_delta * 90)); then
        break
    fi
    sleep 58
done
# Python rechecks resources and source identities immediately before model loading.
date --iso-8601=seconds > "$queue/launched.txt"
set +e
/usr/bin/time -v -o "$root/resources.txt" timeout --signal=TERM --kill-after=5 285 \
    taskset -c 0 "$python" -u -B scripts/step28_record_replay_numerics.py \
    --output "$out" --sources "$root/sources.json" > "$root/console.txt" 2>&1 &
child=$!
printf '%s\n' "$child" > "$queue/verification_wrapper_pid.txt"
while kill -0 "$child" 2>/dev/null; do
    if [[ -f "$out/first_update.json" ]]; then
        rm -- "$self"
        date --iso-8601=seconds > "$queue/script_deleted_after_update.txt"
        break
    fi
    sleep 1
done
wait "$child"
status=$?
if [[ -f "$out/first_update.json" && -f "$self" ]]; then
    rm -- "$self"
    date --iso-8601=seconds > "$queue/script_deleted_after_update.txt"
fi
printf '%s\n' "$status" > "$root/exit_status.txt"
date --iso-8601=seconds > "$root/finished.txt"
printf '%s verification_exit=%s; no automatic retry\n' "$(date --iso-8601=seconds)" "$status"
exit "$status"
