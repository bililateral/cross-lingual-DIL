#!/usr/bin/env bash
# One bounded handwritten invocation, never formal training or automatic retry.
set -u
cd /home/yongpeng/cross-lingual || exit 2
mode=${1:?cpu or gpu}
core=${2:?idle CPU index}
[[ "$core" =~ ^[0-9]+$ ]] || exit 2
root=reports/documentation/20261004/relation_memory/implementation
case "$mode" in
  cpu) output="$root/cpu_repair"; seconds=1790; visible= ;;
  gpu) output="$root/gpu"; seconds=3595; visible=0 ;;
  *) exit 2 ;;
esac
[[ ! -e "$output" && ! -e "$output.console.txt" ]] || exit 2
export CUDA_VISIBLE_DEVICES="$visible" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false PYTHONDONTWRITEBYTECODE=1
started=$(date +%s)
printf 'started_at=%s\nlimit_seconds=%s\n' "$(date -Is)" "$seconds" > "$output.wrapper.txt"
timeout --signal=TERM --kill-after=5 "$seconds" taskset -c "$core" /home/yongpeng/miniconda3/envs/py310/bin/python scripts/step28_relation_memory_verify.py --mode "$mode" --output "$output" > "$output.console.txt" 2>&1
status=$?
printf 'exit_code=%s\nended_at=%s\ntotal_wall_seconds=%s\n' "$status" "$(date -Is)" "$(( $(date +%s) - started ))" >> "$output.wrapper.txt"
if [[ "$mode" == gpu ]]; then
  nvidia-smi --query-gpu=index,memory.used,memory.free --format=csv >> "$output.wrapper.txt"
fi
exit "$status"
