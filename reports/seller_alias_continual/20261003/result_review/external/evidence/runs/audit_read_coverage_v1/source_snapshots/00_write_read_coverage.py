#!/usr/bin/env python3
"""Record the actual bounded read coverage of this review subtask."""
import hashlib
import json
from pathlib import Path

BASE = Path('/workspace/scratch/ca13b63db3f0/review_work')
PROJECT = BASE/'input/project'
OUT = Path(__file__).resolve().parent

paths = [
    BASE/'input/REQUEST.zh.md',
    PROJECT/'scripts/step28_er_weight_audit.py',
    PROJECT/'scripts/step28_bge_continual_audit.py',
    BASE/'evidence/src/run_record.py',
    PROJECT/'docs/SELLER_ALIAS_ER_LOW_RESULT.zh.md',
    PROJECT/'docs/SELLER_ALIAS_LOGIT_WEIGHT_RESULT.zh.md',
    PROJECT/'docs/SELLER_ALIAS_ER_WEIGHT_RESULT.zh.md',
    PROJECT/'docs/SELLER_ALIAS_ER_LOW.zh.md',
    PROJECT/'docs/SELLER_ALIAS_LOGIT_WEIGHT.zh.md',
    PROJECT/'docs/SELLER_ALIAS_BGE_CONTINUAL.zh.md',
    PROJECT/'reports/seller_alias_continual/20261001/bge_continual_result/review/report.zh.md',
    PROJECT/'reports/seller_alias_continual/20261002/er_low_execution/20261002_124018/job/execution.json',
    PROJECT/'reports/seller_alias_continual/20261002/logit_weight_execution/20261002_153200/workspace/reports/job/execution.json',
]
for study in ('er_low','logit_weight'):
    for filename in ('audit_execution.json','audit_stdout.log','audit_stderr.log'):
        paths.append(PROJECT/f'reports/seller_alias_continual/20261003/{study}_result/{filename}')

records = []
for path in paths:
    raw = path.read_bytes()
    records.append({'path':str(path.relative_to(BASE)), 'bytes':len(raw),
                    'sha256':hashlib.sha256(raw).hexdigest(),
                    'coverage':'full text read and substantively inspected; empty stderr verified empty when applicable'})

runtime_inputs = json.loads((OUT/'report_consistency.json').read_text())['inputs']
payload = {
    'review_part':'audit_replay_and_report_interpretation',
    'manual_full_read_files':records,
    'report_crosscheck_runtime_inputs':runtime_inputs,
    'project_audit_read_coverage':{
        'description':'Both unchanged audit scripts were fully read before execution. Request low/logit audit commands then accessed only supplied saved-result records and arrays.',
        'low':{'run':'runs/audit_replay_low_v2_cpu0','source_files_bound':31,'returned_files_bound':270,'matrix_count_sets_loaded':99,'stage_csv_rows_checked':8910},
        'logit':{'run':'runs/audit_replay_logit_v1_cpu0','source_files_bound':34,'returned_files_bound':198,'matrix_count_sets_loaded':63,'stage_csv_rows_checked':5346},
        'warning':'Hash-bound source counts are not claimed as this subagent manually reading all training sources. Full source semantics and independent numerical recomputation belong to separate main-review evidence.'
    },
    'not_independently_recomputed':['Formal labels and truth-dependent AP/MAP/Brier/log_loss','Native model/Adam reload or CUDA training','Calibration refit/optimum','Full memory text/labels/reference target payload','Server state','Unprovided historical evidence ZIPs'],
    'failed_and_revised_runs':[{'failed':'runs/audit_replay_low_v1','revision':'runs/audit_replay_low_v2_cpu0','changed':'--cpu 24 to --cpu 0','cause':'Review command mistake: selected CPU24 without evidence before inspecting affinity0–8 output. Failure at sched_setaffinity prior to output creation or saved-result reading.','project_failure':False,'injected_test':False}],
    'claim':'This record distinguishes complete manual reads, runtime array/record checks, hash binding and unverified scope.'
}
(OUT/'read_coverage.json').write_text(json.dumps(payload,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'manual_full_read_files':len(records),'runtime_crosscheck_inputs':len(runtime_inputs),'output':str(OUT/'read_coverage.json')},ensure_ascii=False))
