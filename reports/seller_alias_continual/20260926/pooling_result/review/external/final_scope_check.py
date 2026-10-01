#!/usr/bin/env python3
"""Final source immutability + blind threshold summaries. NumPy only; no labels/models."""
import argparse, hashlib, json, os, sys
from pathlib import Path
import numpy as np

def main(root, evidence):
    os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})
    def read(p): return json.loads(p.read_text(encoding='utf-8'))
    def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
    q=root/'reports/seller_alias_continual/20260926/pooling_result'
    j=root/'reports/seller_alias_continual/20260925/pooling_execution/20260925_125203/job'
    inv=read(q/'review/source_inventory.json')
    checks=[]
    for r in inv['files']:
        p=root/r['path']; actual={'path':r['path'],'bytes':p.stat().st_size,'sha256':sha(p)}
        if any(actual[k]!=r[k] for k in ('bytes','sha256')):raise ValueError('Submitted source changed: '+r['path'])
        checks.append(actual)
    ev=read(j/'evaluation/evaluation.json'); po=read(root/'schema/step28_alias_pooling_policy.json')
    old=root/po['historical_d']['run_root']/'split_rank'
    thresholds=[]
    for s in ('s0','s1','s2'):
        for kind in ('d','weighted'):
            run=f'{s}_{kind}'; ar=old if run=='s0_d' else j/'run'/run
            a=read(ar/'manifest.json'); sr=a['points']['6']['scores']['development']; p=ar/sr['path']
            if sha(p)!=sr['sha256'] or p.stat().st_size!=sr['bytes']:raise ValueError('score identity')
            x=np.load(p,allow_pickle=False).astype(np.float64)
            au=ev['runs'][run]['automatic_classification']; t=au['threshold']
            thresholds.append({'run':run,'threshold':t,'maximum_saved_valid_logit':float(x.max()),
                               'predicted_positive_pairs':int((x>=t).sum()),
                               'pooled':au['pooled']})
    csv_equal={name:sha(q/'analysis'/name)==sha(q/'reviewer_analysis'/name) for name in ('metrics.csv','trajectory.csv')}
    if not all(csv_equal.values()):raise ValueError('submitted output CSV differs')
    out={'status':'PASS_FINAL_SOURCE_IMMUTABILITY_AND_BLIND_THRESHOLD_CHECK',
         'submitted_source_files_unchanged':len(checks),'submitted_source_bytes':sum(r['bytes'] for r in checks),
         'csv_sha_equal_original_linux_and_webpage':csv_equal,
         'formal_label_reads':0,'formal_text_reads':0,'model_loads':0,'training_updates':0,
         'cpu_affinity':list(os.sched_getaffinity(0)),
         'threads_status':[s for s in Path('/proc/self/status').read_text().splitlines() if s.startswith('Threads:')],
         'automatic_thresholds':thresholds,'unchanged_sources':checks}
    target=evidence/'final_scope_check.json'
    if target.exists():raise FileExistsError(target)
    target.write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in out.items() if k!='unchanged_sources'},ensure_ascii=False,indent=2))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--evidence',type=Path,required=True)
    a=p.parse_args();main(a.root.resolve(),a.evidence.resolve())
