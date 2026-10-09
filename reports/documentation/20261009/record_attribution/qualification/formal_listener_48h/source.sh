#!/usr/bin/env bash
# Start with: nohup bash < this_file > bootstrap.log 2>&1 &
# Reading stdin keeps this waiting shell out of the existing "step28" training
# process filter. Its PID, purpose and complete source remain in formal_listener.
set -euo pipefail
case "$0" in bash|/bin/bash|/usr/bin/bash) ;; *) echo 'Launch bash with this file on stdin' >&2; exit 2 ;; esac
project=/home/yongpeng/cross-lingual
workspace="$project/reports/seller_alias_continual/20261009/record_attribution_execution/20261009_182606/workspace"
cd "$workspace"
self="$workspace/scripts/run_step28_record_attribution_auto_linux_20261009.sh"
qualification="$workspace/reports/qualification"
native_queue="$qualification/native_listener"
attempt="$qualification/gpu/attempt01"
queue="$qualification/formal_listener"
gate="$qualification/formal_gate.json"
job="$workspace/reports/job"
python=/home/yongpeng/miniconda3/envs/py310/bin/python
test -f "$self"
test -f "$native_queue/pid.txt"
test ! -e "$gate"
for path in "$job" "$job.wrapper.txt" "$job.console.txt"; do test ! -e "$path"; done
mkdir "$queue"
cp -- "$self" "$queue/source.sh"
sha256sum -- "$self" > "$queue/script.sha256"
printf '%s\n' "$$" > "$queue/pid.txt"
date --iso-8601=seconds > "$queue/started.txt"
exec >> "$queue/events.log" 2>&1
trap 's=$?; printf "%s\n" "$s" > "$queue/exit_status.txt"; date --iso-8601=seconds > "$queue/finished.txt"' EXIT
trap 'exit 143' TERM
trap 'exit 130' INT
unset LD_LIBRARY_PATH LD_PRELOAD
export CUDA_VISIBLE_DEVICES=0 CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTHONDONTWRITEBYTECODE=1

while [[ ! -f "$native_queue/finished.txt" ]]; do
    if ! kill -0 "$(cat "$native_queue/pid.txt")" 2>/dev/null; then
        echo 'Native listener exited without a completion receipt; stop, no retry'
        exit 1
    fi
    printf '%s waiting_for_native_completion\n' "$(date --iso-8601=seconds)"
    sleep 30
done
test "$(cat "$native_queue/exit_status.txt")" = 0
test -f "$attempt/finished.txt"
test "$(cat "$attempt/exit_status.txt")" = 0

# GATE_BEGIN: only completed, source-bound native evidence can create the gate.
taskset -c 0 "$python" - "$workspace" "$project" <<'PY'
from datetime import datetime
import math
from pathlib import Path
import sys

workspace, project = map(Path, sys.argv[1:])
sys.path.insert(0, str(workspace / "scripts"))
import step28_record_attribution_run as run
data = run.data
qualification = workspace / "reports/qualification"
queue = qualification / "formal_listener"
attempt = qualification / "gpu/attempt01"
native_gate_path = qualification / "native_gate.json"
native_gate = data.read_json(native_gate_path)
if (native_gate["status"] != "APPROVED_RECORD_ATTRIBUTION_NATIVE"
        or native_gate["review_disposition"] != "NO_OPEN_BLOCKERS"
        or native_gate["source_files"] != run.sources()):
    raise ValueError("Reviewed native source identity differs")
for key in ("cpu", "primary_review"):
    rec = native_gate[key]
    data.verify(workspace / rec["path"], rec)
p = run.policy()
gpu_path = attempt / "result/result.json"
gpu = data.read_json(gpu_path)
for field, maximum in (("cumulative_seconds", p["native_verification"]["maximum_seconds"]),
                       ("peak_rss_bytes", p["runtime"]["maximum_rss_bytes"]),
                       ("max_reserved_bytes", p["runtime"]["maximum_cuda_reserved_bytes"])):
    value = gpu[field]
    if not math.isfinite(value) or not 0 < value <= maximum:
        raise ValueError("Native budget differs: " + field)
if gpu["formal_inputs"] is not False or gpu["formal_labels"] is not False:
    raise ValueError("Native qualification accessed formal inputs")
outer_seconds = (datetime.fromisoformat((attempt / "finished.txt").read_text().strip())
                 - datetime.fromisoformat((attempt / "started.txt").read_text().strip())).total_seconds()
if not 0 < outer_seconds <= 600:
    raise ValueError("Native outer elapsed time exceeds the approved budget")
if sum(path.stat().st_size for path in attempt.rglob("*") if path.is_file()) > p["native_verification"]["maximum_evidence_bytes"]:
    raise ValueError("Native evidence exceeds the approved budget")
for arm in run.ARMS:
    row = gpu["native"]["architectures"][arm]
    if (row["native_state_unchanged"] is not True or row["next_model_adam_grad_bitwise_equal"] is not True
            or row["maximum_records"] != 224 or row["channel_tokens"] != 256):
        raise ValueError("Native update/diagnostic evidence incomplete: " + arm)
gate = dict(status="APPROVED_RECORD_ATTRIBUTION_FORMAL", job="reports/job",
            created_at=datetime.now().astimezone().isoformat(),
            source_files=native_gate["source_files"], review_disposition="NO_OPEN_BLOCKERS",
            cpu=native_gate["cpu"], gpu=data.record(gpu_path, workspace),
            primary_review=native_gate["primary_review"], runtime=p["runtime"], supervision=p["supervision"],
            authorization="User 2026-10-09: automatically start approved formal GPU training after handwritten GPU qualification passes; delete the active detector after an actual update")
candidate = queue / "candidate_gate.json"
data.write_json(candidate, gate)
run.validate_gate(workspace / "reports/job", candidate)
# Only make the seven already-authorized input files addressable; never read
# their contents here. The reviewed runner owns train/valid access accounting.
data_root = Path(run.config("R0")["data_root"])
links = []
for name in ("manifest.json", "validation.json", "groups.csv", "train/items.jsonl",
             "development/items.jsonl", "train/supervision/pairs.csv", "development/supervision/pairs.csv"):
    source, target = project / data_root / name, workspace / data_root / name
    if not source.is_file():
        raise FileNotFoundError(source)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_symlink():
        if target.resolve() != source.resolve():
            raise ValueError("Existing input link differs")
    elif target.exists():
        raise ValueError("Unexpected existing formal input")
    else:
        target.symlink_to(source)
    links.append(dict(path=target.relative_to(workspace).as_posix(), target=str(source)))
data.write_json(queue / "qualification.json", dict(status="PASS_FORMAL_START_PREREQUISITES",
    native_gate=data.record(native_gate_path, workspace), native_outer_seconds=outer_seconds,
    projected_total_seconds=gpu["native"]["projected_total_seconds"],
    projected_peak_output_bytes=gpu["native"]["projected_peak_output_bytes"], input_links=links,
    formal_data_contents_read=False))
candidate.rename(qualification / "formal_gate.json")
print("Native qualification and source-bound formal gate passed", flush=True)
PY
# GATE_END

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
for path in "$job" "$job.wrapper.txt" "$job.console.txt"; do test ! -e "$path"; done
date --iso-8601=seconds > "$queue/launched.txt"
bash scripts/run_step28_record_attribution_linux_20261009.sh 0 "$gate" "$job" &
child=$!
printf '%s\n' "$child" > "$queue/training_wrapper_pid.txt"
actual_update_recorded() {
    # The unchanged reviewed entry prints this only AFTER run.update returns.
    [[ -f "$job.console.txt" ]] && grep -m 1 -E \
        '"completed":[[:space:]]*[1-9][0-9]*,.*"event":[[:space:]]*"updates"' \
        "$job.console.txt" > "$queue/first_update.json"
}
while kill -0 "$child" 2>/dev/null; do
    if actual_update_recorded; then
        rm -- "$self"
        date --iso-8601=seconds > "$queue/script_deleted_after_update.txt"
        break
    fi
    sleep 2
done
status=0
wait "$child" || status=$?
if [[ -f "$self" ]] && actual_update_recorded; then
    rm -- "$self"
    date --iso-8601=seconds > "$queue/script_deleted_after_update.txt"
fi
printf '%s formal_exit=%s; no automatic retry\n' "$(date --iso-8601=seconds)" "$status"
exit "$status"
