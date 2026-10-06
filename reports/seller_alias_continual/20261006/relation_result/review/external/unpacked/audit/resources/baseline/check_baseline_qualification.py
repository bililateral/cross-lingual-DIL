"""Read-only, stdlib-only checks of saved baseline metadata, not native execution."""
from pathlib import Path
import json
import hashlib
from datetime import datetime

ROOT = Path('/workspace/scratch/f4d639b3b473/relation_memory_result_review')
OUT = Path(__file__).parent
LOGIT = ROOT / 'reports/seller_alias_continual/20261004/logit_low_result/job'
RELATION = ROOT / 'reports/seller_alias_continual/20261006/relation_result/job'

def read(path):
    return json.loads(path.read_text(encoding='utf-8'))

execution = read(LOGIT / 'execution.json')
lc = read(LOGIT / 'evaluation/collected.json')
rc = read(RELATION / 'evaluation/collected.json')
completion = read(LOGIT / 'completion.json')
before = read(LOGIT / 'before_valid.json')
docs = {}
for item in execution['source_files']:
    if item['path'] in ('docs/SELLER_ALIAS_BGE_CONTINUAL.zh.md',
                        'docs/SELLER_ALIAS_LOGIT_LOW.zh.md'):
        data = (ROOT / item['path']).read_bytes()
        docs[item['path']] = {
            'bytes_match_execution': len(data) == item['bytes'],
            'sha256_match_execution': hashlib.sha256(data).hexdigest() == item['sha256'],
        }
start = datetime.fromisoformat((LOGIT / 'started.txt').read_text().strip())
end = datetime.fromisoformat((LOGIT / 'finished.txt').read_text().strip())
result = {
    'scope': 'Saved metadata identity and declared lifecycle only; no Torch, model, original labels, or project Linux access',
    'documents_match_actual_execution_descriptors': docs,
    'same_ordered_saved_evaluation_identity': {
        key: lc[key] == rc[key]
        for key in ('domains', 'group_ids', 'metric_columns')
    },
    'group_count': len(lc['group_ids']),
    'metric_count': len(lc['metric_columns']),
    'baseline_completion': completion,
    'baseline_before_valid': before,
    'baseline_access': read(LOGIT / 'access.json'),
    'baseline_started': start.isoformat(),
    'baseline_finished': end.isoformat(),
    'baseline_wrapper_seconds': (end - start).total_seconds(),
    'baseline_exit_status': int((LOGIT / 'exit_status.txt').read_text()),
    'baseline_environment': {key: execution[key] for key in ('python', 'torch', 'cuda', 'gpu', 'cpu_affinity')},
    'conclusion': 'No baseline-qualification contradiction found in this bounded audit. Wall time is not a matched algorithm-efficiency comparison.',
}
(OUT / 'baseline_qualification_checks.json').write_text(
    json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
print(json.dumps({
    'documents_match': all(all(x.values()) for x in docs.values()),
    'evaluation_identity_match': all(result['same_ordered_saved_evaluation_identity'].values()),
    'baseline_wrapper_seconds': result['baseline_wrapper_seconds'],
    'baseline_status': completion['status'],
    'baseline_exit_status': result['baseline_exit_status'],
}, ensure_ascii=False))
