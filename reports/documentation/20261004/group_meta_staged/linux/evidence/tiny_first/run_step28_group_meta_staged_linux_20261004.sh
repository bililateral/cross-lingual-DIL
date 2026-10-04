#!/usr/bin/env bash
# One bounded, explicitly selected handwritten attempt; no queue or retry loop.
set -euo pipefail
mode=${1:?tiny or staged}
cpu=${2:?inspected idle CPU}
run=${3:?fresh attempt name}
[[ "$mode" = tiny || "$mode" = staged ]]
[[ "$cpu" =~ ^[0-9]+$ && "$run" =~ ^[a-z0-9_]+$ ]]
[[ "$(realpath .)" = /home/yongpeng/cross-lingual/reports/documentation/20261004/group_meta_staged/workspace ]]
test -f reports/authorization.json
test ! -e "evidence/$run"
mkdir -p evidence
# Prior attempts remain frozen and count against the same total hour.
prior=../../group_meta_gpu/workspace/evidence
test -d "$prior"
spent=0
for file in "$prior"/*/seconds.txt; do
    value=$(cat "$file")
    [[ "$value" =~ ^[0-9]+$ ]]
    spent=$((spent + value))
done
(( spent == 101 ))
for file in evidence/*/seconds.txt; do
    [[ -f "$file" ]] || continue
    value=$(cat "$file")
    [[ "$value" =~ ^[0-9]+$ ]]
    spent=$((spent + value))
done
# An unfinished earlier attempt must be accounted for before another starts.
for directory in evidence/*; do
    [[ -d "$directory" ]] || continue
    test -f "$directory/seconds.txt"
done
remaining=$((3600 - spent))
(( remaining > 0 ))
python=/home/yongpeng/miniconda3/envs/py310/bin/python
"$python" -B - "$mode" <<'PY'
import json, pathlib, sys
auth = json.loads(pathlib.Path('reports/authorization.json').read_text())
assert auth['scope'] == 'group_meta_staged_handwritten_only'
assert auth['approved'] and auth['before_incremental_review']
assert auth['maximum_seconds'] == 3600 and auth['maximum_rss_gib'] == 128
assert auth['maximum_allocator_gib'] == 28 and auth['maximum_output_mib'] == 128
results = [json.loads(p.read_text()) for p in pathlib.Path('evidence').glob('*/result.json')]
attempts = list(pathlib.Path('evidence').glob('*/started.txt'))
# One incremental tiny check and one native staged attempt; no automatic retry.
assert len(attempts) < 2
mode = sys.argv[1]
if mode != 'tiny':
    assert any(r['mode'] == 'tiny' and r['status'] == 'PASS' for r in results)
assert not any(p.read_text().strip() == mode for p in pathlib.Path('evidence').glob('*/mode.txt'))
PY
gpu_free=$(nvidia-smi -i 0 --query-gpu=memory.free --format=csv,noheader,nounits)
gpu_pids=$(nvidia-smi -i 0 --query-compute-apps=pid --format=csv,noheader,nounits)
host_free=$(awk '/^MemAvailable:/ {print $2}' /proc/meminfo)
(( gpu_free >= 30720 && host_free >= 150994944 ))
[[ -z "$gpu_pids" ]]
mkdir "evidence/$run"
printf '%s\n' "$mode" > "evidence/$run/mode.txt"
cp scripts/step28_group_meta*.py tests/test_step28_group_meta*.py \
    scripts/run_step28_group_meta_staged_linux_20261004.sh "evidence/$run/"
date -Is > "evidence/$run/started.txt"
printf '%s\n' "gpu_free_mib=$gpu_free host_free_kib=$host_free remaining_seconds=$remaining cpu=$cpu" \
    > "evidence/$run/resources.txt"
started=$(date +%s)
child=''
finish() {
    status=$?
    trap - EXIT
    if [[ -n "$child" ]] && kill -0 "$child" 2>/dev/null; then
        kill -KILL "$child"
        wait "$child" || true
    fi
    date -Is > "evidence/$run/finished.txt"
    printf '%s\n' "$(( $(date +%s) - started ))" > "evidence/$run/seconds.txt"
    printf '%s\n' "$status" > "evidence/$run/exit_status.txt"
    exit "$status"
}
trap finish EXIT
trap 'exit 130' INT TERM
unset LD_LIBRARY_PATH LD_PRELOAD
export CUDA_VISIBLE_DEVICES=0 CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=scripts:tests
ulimit -f 131072
taskset -c "$cpu" "$python" -B -u scripts/step28_group_meta_staged_check.py "$mode" \
    --output "evidence/$run/result.json" > "evidence/$run/stdout.txt" 2> "evidence/$run/stderr.txt" &
child=$!
printf '%s\n' "$child" > "evidence/$run/pid.txt"
while kill -0 "$child" 2>/dev/null; do
    rss=$(ps -o rss= -p "$child" | tr -d ' ' || true)
    available=$(awk '/^MemAvailable:/ {print $2}' /proc/meminfo)
    bytes=$(du -sb evidence | cut -f1)
    elapsed=$(( $(date +%s) - started ))
    if (( elapsed >= remaining || ${rss:-0} > 134217728 || available < 16777216 || bytes > 134217728 )); then
        printf '%s\n' "elapsed=$elapsed rss_kib=${rss:-0} available_kib=$available evidence_bytes=$bytes" \
            > "evidence/$run/budget_stop.txt"
        exit 124
    fi
    sleep 1
done
wait "$child"
child=''
