#!/usr/bin/env python3
"""Read already-saved qualifications/receipts; no models, labels or qualification execution."""
import argparse, hashlib, json, zipfile
from pathlib import Path
P='context/reports/documentation/20261015/balanced_pooling_repair/'
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--archive',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
    blob=a.archive.read_bytes()
    assert len(blob)==6123032 and hashlib.sha256(blob).hexdigest()=='07f941b2d689c4f9e2cd024364b57e3607bd9bb762868378a854b5be97ef83f6'
    z=zipfile.ZipFile(a.archive);used=[]
    def read(n):used.append(n);return json.loads(z.read(n))
    m=read('result/job/run/manifest.json');p=read('result/job/run/progress.json')
    keys=[k for k in m if m[k]!=p[k]];assert keys==['status']
    n=read(P+'qualification/gpu/attempt01/result/result.json');v=n['native']
    assert n['actual_updates']==3 and v['cumulative_updates']==11 and len(n['completed_updates'])==3
    assert n['formal_inputs'] is False and n['formal_labels'] is False
    for phase in ['A','B']:
        for field in ['pre_clip_gradient_norm','parameter_max_delta']:
            assert set(v['effects'][phase][field])=={'encoder','head'}
            assert all(x>0 for x in v['effects'][phase][field].values())
    pair=n['completed_updates'][1:]
    assert all(pair[0][k]==pair[1][k] for k in pair[0] if k!='solver_summary')
    assert v['next_update_bitwise_equal'] is True and v['F_pooling_restores_verified'] is True
    old=read('context/reports/seller_alias_continual/20261015/balanced_pooling_failure/evidence/reports/job/failure.json')['solver_failure']
    assert old['step']==248 and old['stage']==2 and old['arm']=='F' and len(old['failed_blocks'])==3
    e=read('result/job/evaluation/evaluation.json')
    x=e['forgetting_comparisons']['G-S']['raw']['log_loss']
    res={'input_sha256':hashlib.sha256(blob).hexdigest(),'programmatic_read_files':used,
         'progress_vs_manifest_differences':keys,'progress_status':p['status'],'manifest_status':m['status'],
         'native_record_only':{k:v[k] for k in ['fixture','effects','equality_counts','next_update_bitwise_equal','F_pooling_restores_verified','terminal_gpu_vjp_verified','solver_summary','terminal_stress']},
         'native_internal_vs_wrapper_accounting':{'internal_elapsed':n['elapsed_seconds'],'internal_cumulative_charged':n['cumulative_charged_seconds'],'approved_wrapper_elapsed_seconds':196.21727442741394,'approved_cumulative_charged_seconds':561,'wrapper_source':P+'qualification/native_disposition.zh.txt:6'},
         'old_failure_location_and_summary':old,'loss_forgetting_example_GminusS_raw_log_loss':x,
         'limits':'Qualification receipts read, not rerun. No independent recomputation of qualification model states/gradients/K. Progress status is a historical snapshot; all other progress fields equal final manifest.'}
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(res,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    print(json.dumps(res,ensure_ascii=False,indent=2,allow_nan=False))
if __name__=='__main__':main()
