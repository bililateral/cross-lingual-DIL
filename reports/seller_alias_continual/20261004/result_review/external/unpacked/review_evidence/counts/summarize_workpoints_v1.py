#!/usr/bin/env python3
"""Descriptive O/N/Z confusion totals from independent per-domain reconstruction.

This is not a new acceptance rule, CI, or independent replication. O and N include
120 group appearances over the three fixed orders; Z includes 60 appearances.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

p = argparse.ArgumentParser()
p.add_argument('--input', required=True)
p.add_argument('--output-dir', required=True)
a = p.parse_args()
inp = Path(a.input)
out = Path(a.output_dir)
out.mkdir(parents=True, exist_ok=True)
data = json.loads(inp.read_text())
records = []
for method in ['seq', 'tenth', 'logit_quarter', 'logit_tenth', 'risk']:
    for endpoint in ['O', 'N', 'Z']:
        rows = []
        for order in ['ABC', 'BCA', 'CAB']:
            selected = ([(3, d) for d in order[:2]] if endpoint == 'O' else
                        [(2, order[1]), (3, order[2])] if endpoint == 'N' else
                        [(3, order[2])])
            rows += [data[f'{order}_{method}_stage{stage}']['stage-cal']['by_domain'][d]
                     for stage, d in selected]
        c = {k: sum(r[k] for r in rows) for k in ['tp', 'fp', 'fn', 'tn']}
        tp, fp, fn, tn = [c[k] for k in ['tp', 'fp', 'fn', 'tn']]
        records.append(dict(method=method, endpoint=endpoint, role='stage-cal',
                            group_appearances=20 * len(rows), **c,
                            precision=tp / (tp + fp) if tp + fp else 0.,
                            recall=tp / (tp + fn), f1=2 * tp / (2 * tp + fp + fn),
                            specificity=tn / (tn + fp), fpr=fp / (fp + tn),
                            accuracy=(tp + tn) / (tp + fp + fn + tn)))
(out / 'workpoint_summary.json').write_text(json.dumps(records, indent=2) + '\n')
with (out / 'workpoint_summary.csv').open('w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(records[0]))
    w.writeheader()
    w.writerows(records)
print(json.dumps({'argv': [sys.executable, *sys.argv], 'input': str(inp),
                  'input_sha256': hashlib.sha256(inp.read_bytes()).hexdigest(),
                  'rows': len(records), 'scope': 'Descriptive pooled working points; no new acceptance rule.'}, indent=2))
