#!/usr/bin/env bash
# Mechanical scheduling only; the reviewed test entry retains all scientific gates.
set -euo pipefail
cd /home/yongpeng/cross-lingual
watch_dir="$(realpath -m -- "${1:?Pass new listener directory}")"
formal_dir="$(realpath -m -- "${2:?Pass new formal output directory}")"
authorization="$(realpath -- "${3:?Pass reviewed authorization receipt}")"
for location in "$watch_dir" "$formal_dir" "$authorization"; do
  case "$location" in
    "$PWD/reports/seller_alias_continual/20260928/test_execution/"*) ;;
    *) echo 'Execution paths must remain in this test stage' >&2; exit 2 ;;
  esac
done
test ! -e "$formal_dir"
mkdir "$watch_dir"
printf '%s\n' "$$" > "$watch_dir/pid.txt"
date --iso-8601=seconds > "$watch_dir/started.txt"

write_status() {
  local state="$1"
  printf '{"time":"%s","state":"%s","pid":%s,"resource_observation_time":"%s","gpu":0,"gpu_pids":"%s","free_gpu_mib":%s,"gpu_utilization":%s,"available_host_kib":%s,"free_disk_kib":%s}\n' \
    "$(date --iso-8601=seconds)" "$state" "$$" "$probe_time" "$gpu_pids" \
    "$free_gpu" "$gpu_util" "$free_host" "$free_disk" \
    > "$watch_dir/status.json"
  cat "$watch_dir/status.json" >> "$watch_dir/events.jsonl"
}

while true; do
  gpu_row="$(nvidia-smi -i 0 \
    --query-gpu=memory.free,utilization.gpu \
    --format=csv,noheader,nounits)"
  read -r free_gpu gpu_util <<< "${gpu_row/,/ }"
  gpu_pids="$(nvidia-smi -i 0 \
    --query-compute-apps=pid --format=csv,noheader,nounits)"
  gpu_pids="${gpu_pids//$'\n'/,}"
  gpu_pids="${gpu_pids//[[:space:]]/}"
  free_host="$(awk '/^MemAvailable:/ {print $2}' /proc/meminfo)"
  free_disk="$(df -Pk . | awk 'NR==2 {print $4}')"
  probe_time="$(date --iso-8601=seconds)"
  for value in "$free_gpu" "$gpu_util" "$free_host" "$free_disk"; do
    [[ "$value" =~ ^[0-9]+$ ]] || { echo 'Resource probe failed' >&2; exit 3; }
  done
  [[ "$gpu_pids" =~ ^[0-9,]*$ ]] || { echo 'GPU PID probe failed' >&2; exit 3; }
  if [[ -z "$gpu_pids" ]] && (( gpu_util == 0 && free_gpu >= 24576 \
      && free_host >= 16777216 && free_disk >= 4194304 )); then
    break
  fi
  write_status WAITING_FOR_FREE_GPU_AND_RESOURCES
  sleep 60
done

# A single invocation: any failure stops here, with no retry or second label parse.
write_status FORMAL_ENTRY_START_REQUESTED
set +e
TEST_GPU=0 TEST_CPU=0 \
  bash scripts/run_step28_alias_test_linux_20260928.sh execute \
  "$formal_dir" "$authorization"
status=$?
set -e
printf '%s\n' "$status" > "$watch_dir/formal_exit_status.txt"
date --iso-8601=seconds > "$watch_dir/finished.txt"
write_status FORMAL_ENTRY_EXITED
exit "$status"
