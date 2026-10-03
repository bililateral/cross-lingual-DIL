#!/usr/bin/env python3
"""Preserve the scope of manual source reading and recorded numerical reads."""
import hashlib
import json
from pathlib import Path

here = Path(__file__).resolve().parent
root = Path('/workspace/scratch/ca13b63db3f0/review_work/input')
project = root/'project'
manual = [
    ('REQUEST.zh.md', 'full'),
    ('project/docs/SELLER_ALIAS_ER_LOW.zh.md', 'full'),
    ('project/docs/SELLER_ALIAS_LOGIT_WEIGHT.zh.md', 'full'),
    ('project/schema/step28_er_low_policy.json', 'full'),
    ('project/docs/SELLER_ALIAS_BGE_CONTINUAL.zh.md', 'lines 90-170, including calibration, seven endpoint definitions, bootstrap, original guards'),
    ('project/scripts/step28_bge_continual_evaluate.py', 'full'),
    ('project/scripts/step28_er_weight_evaluate.py', 'full, ER0.1 frozen evaluation implementation'),
    ('project/reports/seller_alias_continual/20261002/logit_weight_execution/20261002_153200/workspace/scripts/step28_er_weight_evaluate.py', 'full, LOGIT frozen evaluation implementation'),
    ('project/scripts/step28_continual_population_evaluate.py', 'full, underlying per-group metrics and confusion definitions'),
    ('project/scripts/step28_chinese_base.py', 'lines 680-715, rates/fixed_classification and automatic-report beginning'),
]
manual_records = []
for name, scope in manual:
    blob = (root/name).read_bytes()
    manual_records.append({'path':name,'bytes':len(blob),'sha256':hashlib.sha256(blob).hexdigest(),
                           'semantic_read_extent':scope,'note':'This metadata pass hashes the file; semantic reading happened before independent implementation.'})
numeric=json.loads((here/'numerical_v1/read_coverage.json').read_text())
extra=json.loads((here/'numerical_v1/interpretation_values.json').read_text())['read_records']
result = {'manual_source_reading':manual_records, 'independent_numeric_input_coverage':numeric,
          'historical_crosscheck_reads':extra,
          'project_metric_or_aggregation_functions_executed':False,
          'formal_labels_text_weights_cache_bodies_read':False,
          'notes':'Counts are saved confusion totals and authorized. No formal label inference or reconstruction was performed.'}
(here/'read_coverage.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'manual_files':len(manual_records),'independent_numeric_files':numeric['total_unique_input_files'],'historical_check_records':len(extra)},ensure_ascii=False))
