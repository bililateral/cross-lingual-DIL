"""Record actual source-reading scope alongside numeric input-read evidence."""
from pathlib import Path
import hashlib
import json

P=Path('/workspace/scratch/ca13b63db3f0/review_work/input/project')
O=Path(__file__).resolve().parent
items=[
    ('../REQUEST.zh.md','full'),
    ('AGENTS.md','full'),
    ('docs/RESEARCH_DISCIPLINE.zh.md','full'),
    ('docs/SELLER_ALIAS_ER_LOW.zh.md','full'),
    ('docs/SELLER_ALIAS_LOGIT_WEIGHT.zh.md','full'),
    ('scripts/step28_alias_calibration.py','full'),
    ('scripts/step28_alias_ranking.py','full'),
    ('scripts/step28_continual_population_evaluate.py','full'),
    ('scripts/step28_bge_continual_evaluate.py','full'),
    ('scripts/step28_er_weight_evaluate.py','full'),
    ('scripts/step28_er_weight_run.py',[[1,177],[325,404]]),
    ('scripts/step28_chinese_base.py',[[680,700]]),
    ('reports/seller_alias_continual/20261002/logit_weight_execution/20261002_153200/workspace/scripts/step28_er_weight_evaluate.py','full'),
    ('reports/seller_alias_continual/20261002/logit_weight_execution/20261002_153200/workspace/scripts/step28_er_weight_run.py',[[106,183]]),
]
records=[]
for name,spans in items:
    path=P/name
    raw=path.read_bytes()
    records.append({'path':name,'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest(),
                    'line_count':len(raw.decode().splitlines()),'human_review_spans':spans})
coverage=json.loads((O/'read_coverage.json').read_text())
coverage['source_and_contract_human_review']=records
coverage['scope_note']='Only this delegated review slice is listed; root combines other source/execution/history audits. Computational record reads are not claimed as per-line human source analysis.'
(O/'read_coverage.json').write_text(json.dumps(coverage,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'numeric_input_files':len(coverage['files']),'source_contract_read_records':len(records)},ensure_ascii=False))
