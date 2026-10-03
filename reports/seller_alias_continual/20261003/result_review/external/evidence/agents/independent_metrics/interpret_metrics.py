#!/usr/bin/env python3
"""Cross-check frozen historical comparison and extract review tables.

Only reads existing saved results and our independent numerical output.
"""
import csv
import hashlib
import json
from pathlib import Path
import sys

here = Path(__file__).resolve().parent
out = here / 'numerical_v1'
project = Path('/workspace/scratch/ca13b63db3f0/review_work/input/project')
read_records = []
def read_json(path):
    blob = path.read_bytes()
    read_records.append({'path':str(path),'bytes':len(blob),'sha256':hashlib.sha256(blob).hexdigest()})
    return json.loads(blob)

c = read_json(out/'independent_comparisons.json')
e = read_json(out/'independent_endpoints.json')
historical = project/'reports/seller_alias_continual/20261001/er_weight_execution/20261001_144452/job/evaluation/evaluation.json'
old = read_json(historical)['comparisons']['quarter_minus_seq']
ours = c['quarter_minus_seq']
old_checks = old['interpretation']['checks']
fresh_checks = {r['check']:r['pass'] for r in ours['gates']}
assert old_checks == fresh_checks
maximum_error = 0.
compared_scalars = 0
for role in ('primary', 'against_raw_reference'):
    for endpoint, metrics in old[role].items():
        for metric, value in metrics.items():
            for k in ('mean','conditional_95pct_interval','per_order'):
                x, y = ours[role][endpoint][metric][k], value[k]
                if isinstance(x,dict):
                    xs, ys = [x[q] for q in ('ABC','BCA','CAB')], [y[q] for q in ('ABC','BCA','CAB')]
                elif isinstance(x,list):
                    xs, ys = x, y
                else:
                    xs, ys = [x],[y]
                maximum_error = max(maximum_error, max(abs(a-b) for a,b in zip(xs,ys)))
                compared_scalars += len(xs)
assert maximum_error < 2e-12

failures = {name:[{k:v for k,v in row.items() if k in ('check','observed_mean','ci_low','ci_high','relation','rhs')}
                   for row in value['gates'] if not row['pass']] for name,value in c.items()}
table = []
for name, value in c.items():
    row = {'comparison':name,'passes':value['pass_count']}
    for ep in ('O','N','Z','final_all'):
        for metric in ('map','recall_at_5'):
            row[f'{ep}_{metric}'] = value['primary'][ep][metric]
    table.append(row)

rows = list(csv.DictReader((out/'independent_primary_all_domain_stage_deltas.csv').open()))
localized_map = {}
for name in c:
    subset = [r for r in rows if r['comparison']==name and r['stage']=='3' and r['metric']=='map']
    localized_map[name] = {'final_stage_9_order_domain_cells':len(subset),
                           'observed_regressions':[{'order':r['order'],'actual_domain':r['actual_domain'],'arrival':int(r['arrival']),'delta':float(r['delta'])} for r in subset if float(r['delta'])<0]}

absolute_table = {arm:{endpoint:e[arm]['primary'][endpoint]['map'] for endpoint in ('O','N','Z','F_first','F','G','final_all')}
                  for arm in ('seq','quarter','tenth','logit_quarter')}
payload = {'historical_quarter_minus_seq': {'pass_count':sum(old_checks.values()),'checks_equal':True,
          'numeric_scalars_compared':compared_scalars,'max_abs_error':maximum_error},
          'comparison_table':table, 'all_failed_gates':failures,
          'absolute_MAP':absolute_table,'local_final_MAP_changes':localized_map,
          'read_records':read_records}
(out/'interpretation_values.json').write_text(json.dumps(payload,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(payload,ensure_ascii=False,indent=2))
