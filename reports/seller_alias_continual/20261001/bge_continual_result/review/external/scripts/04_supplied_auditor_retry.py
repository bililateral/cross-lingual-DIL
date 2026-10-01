"""Correct only the reviewer-selected CPU, preserve the failed invocation and source."""
import hashlib, json, os, subprocess, sys
from pathlib import Path
p=Path('/mnt/data/bge_input/scripts/step28_bge_continual_audit.py')
cpus=sorted(os.sched_getaffinity(0))
print(json.dumps({'reason':'The reviewer first supplied --cpu 24 although this sandbox only permits CPUs 0..4. Failure occurred before output creation and before substantive audit. This is an invocation error, not a project numerical defect.','allowed_cpus':cpus,'chosen_cpu':cpus[0],'source_sha256':hashlib.sha256(p.read_bytes()).hexdigest()},ensure_ascii=False),flush=True)
command=[sys.executable,'-B',str(p),'--project','/mnt/data/bge_input','--job','/mnt/data/bge_input/reports/seller_alias_continual/20260930/bge_continual_execution/20260930_143700/job','--inventory','/mnt/data/bge_input/reports/seller_alias_continual/20261001/bge_continual_result/return_inventory.json','--output','/mnt/data/bge_review_evidence/outputs/supplied_auditor_reproduction','--cpu',str(cpus[0])]
print('COMMAND '+json.dumps(command),flush=True)
raise SystemExit(subprocess.run(command,check=False).returncode)
