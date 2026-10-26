#!/usr/bin/env python3
"""Independent recheck of supplied, already-opened group metrics.

Usage: python3 recompute_results.py EXTRACTED_INPUT_ROOT OUTPUT_DIRECTORY
Uses Python + NumPy only. No project imports, NPY reads, labels, model loading,
training, inference, data generation, or connection to a project host.
The only resampling reproduces the already-prespecified 5000 group bootstrap.
"""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
import argparse
from pathlib import Path
from datetime import datetime,timezone,timedelta
from collections import Counter
import hashlib
import json
import math
import platform
import time
import traceback
import numpy as np

F='reports/documentation/20261024/continual_retry/formal/'
R='reports/seller_alias_continual/20261026/continual_result/'
METHODS=('SEQ_ref','ER0.1_ref','LOGIT0.1_ref','S_tuned','H_pos_tuned')
ORDERS=('ABC','BCA','CAB')
SEEDS=(4,5,6)
ROLES=('raw','stage-cal','first-cal')
FIELDS=('O','N','Z','A_acq','F_first','F')
CORE=('1_1','2_1','2_2','3_1','3_2','3_3')
PAIRS=(('H_pos_tuned','S_tuned'),('ER0.1_ref','SEQ_ref'),('LOGIT0.1_ref','ER0.1_ref'),
       ('S_tuned','LOGIT0.1_ref'),('H_pos_tuned','LOGIT0.1_ref'),('H_pos_tuned','ER0.1_ref'),('H_pos_tuned','SEQ_ref'))
COLUMNS=('average_precision','trapezoidal_pr_auc','roc_auc','recall_at_fpr_1pct','brier','log_loss',
         'precision','recall','f1','specificity','balanced_accuracy','mcc','map','mrr',
         'recall_at_1','recall_at_3','recall_at_5','recall_at_10','ndcg_at_1','ndcg_at_3','ndcg_at_5','ndcg_at_10')


def json_read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(b):return hashlib.sha256(b).hexdigest()
def clean(x):
    if isinstance(x,dict):return {str(k):clean(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)):return [clean(v) for v in x]
    if isinstance(x,np.ndarray):return clean(x.tolist())
    if isinstance(x,(float,np.floating)):return float(x) if math.isfinite(x) else None
    if isinstance(x,np.integer):return int(x)
    if isinstance(x,np.bool_):return bool(x)
    return x

def write(p,x):p.write_text(json.dumps(clean(x),ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')


class Comparisons:
    def __init__(self):self.rows={}
    def check(self,kind,a,b,where):
        x,y=np.asarray(a,dtype=float),np.asarray(b,dtype=float)
        if x.shape!=y.shape:raise AssertionError(('shape',kind,where,x.shape,y.shape))
        if not np.array_equal(np.isnan(x),np.isnan(y)):raise AssertionError(('N/A',kind,where))
        if np.isinf(x).any() or np.isinf(y).any():raise AssertionError(('infinity',kind,where))
        good=np.isfinite(x)
        err=float(np.max(np.abs(x[good]-y[good]))) if good.any() else 0.
        row=self.rows.setdefault(kind,dict(comparisons=0,finite_values=0,NA_values=0,max_abs_error=0.,max_error_location=None))
        row['comparisons']+=1; row['finite_values']+=int(good.sum()); row['NA_values']+=int((~good).sum())
        if err>row['max_abs_error']:row.update(max_abs_error=err,max_error_location=where)
        if err>1e-11:raise AssertionError(('numeric',kind,where,err))


def derive(cells):
    # Explicit algebra. Expanded cells carry zeros in the other actual domains.
    old=(cells['3_1']+cells['3_2'])/2
    acq=(cells['1_1']+cells['2_2'])/2
    out=dict(O=old,A_acq=acq,F=acq-old,N=(cells['2_2']+cells['3_3'])/2,
             Z=cells['3_3'],F_first=cells['1_1']-cells['3_1'])
    if '1_2' in cells:
        out.update(D2=cells['1_1']-cells['2_1'],G2=cells['2_2']-cells['1_2'],
                   D3=(cells['2_1']+cells['2_2']-cells['3_1']-cells['3_2'])/2,
                   G3=cells['3_3']-cells['2_3'])
    return out


def from_counts(c):
    tp,fp,tn,fn=(c[k] for k in ('tp','fp','tn','fn'))
    denom=(tp+fp)*(tp+fn)*(tn+fp)*(tn+fn)
    return [tp/(tp+fp) if tp+fp else np.nan,tp/(tp+fn),2*tp/(2*tp+fp+fn),
            tn/(tn+fp),(tp/(tp+fn)+tn/(tn+fp))/2,
            (tp*tn-fp*fn)/math.sqrt(denom) if denom else np.nan]


def audit(root,out):
    start=time.monotonic(); cmp=Comparisons()
    saved=json_read(root/(F+'analysis/result.json'))
    confirmation=json_read(root/(F+'confirmation/result.json'))
    local=json_read(root/(R+'audit02/audit.json'))
    assert saved['columns']==list(COLUMNS)
    assert saved['bootstrap']==dict(actual_domains='ABC',clusters_per_domain=40,replicates=5000,seed=2026102302)
    assert saved['status']=='FROZEN_ANALYSIS_COMPLETE'
    expected=[(o,s,m,f'{o}_s{s}_{m}') for o in ORDERS for s in SEEDS for m in METHODS]
    assert set(confirmation['runs'])=={v[3] for v in expected}
    assert len(confirmation['groups'])==len(set(confirmation['groups']))==120
    assert len(confirmation['domains'])==120 and Counter(confirmation['domains'])==dict(A=40,B=40,C=40)
    canonical={d:[g for g,x in zip(confirmation['groups'],confirmation['domains']) if x==d] for d in 'ABC'}
    assert len(set(sum(canonical.values(),[])))==120
    assert confirmation['forwards']==11520
    assert set(saved['comparisons'])=={a+'-'+b for a,b in PAIRS}
    arrays={}; counts={}; contributions={}; point_values={}; file_records=[]; NA_counts={}
    scalar_cells=0
    for order,seed,method,rid in expected:
        cells=confirmation['runs'][rid]
        required=set(CORE)|({'1_2','2_3'} if method=='SEQ_ref' else set())
        assert set(cells)==required
        scalar_cells+=len(cells)
        for role in ROLES:
            expanded={}; simple={}
            for cell,p in cells.items():
                stage,pos=map(int,cell.split('_'))
                domain=order[pos-1]
                assert p['domain']==domain and p['ids']==canonical[domain]
                assert set(p['metrics'])==set(ROLES)
                rec=p['metrics'][role]['matrix']
                relative=F+'confirmation/'+rec['path']
                b=(root/relative).read_bytes()
                assert len(b)==rec['bytes'] and sha(b)==rec['sha256']
                j=json.loads(b)
                assert j['columns']==list(COLUMNS) and len(j['counts'])==40
                a=np.asarray(j['values'],dtype=float)
                assert a.shape==(40,22) and not np.isinf(a).any()
                assert not np.isnan(a[:,[k for k in range(22) if k not in (6,11)]]).any()
                count_array=np.asarray([[row[k] for k in ('tp','fp','tn','fn')] for row in j['counts']],dtype=np.int64)
                assert np.all(count_array>=0)
                assert np.all(count_array[:,0]+count_array[:,3]==20) and np.all(count_array[:,1]+count_array[:,2]==358)
                cmp.check('classification_from_saved_counts',[from_counts(row) for row in j['counts']],a[:,6:12],relative)
                counts[rid,role,cell]=count_array
                arrays[rid,role,cell]=a
                e=np.zeros((3,40,22)); e['ABC'.index(domain)]=a
                expanded[cell]=e; simple[cell]=a.mean(0)
                cell_saved=saved['per_run_actual_domain_cell_points'][f'{rid}_{role}_{cell}']
                assert cell_saved['domain']==domain and cell_saved['counts']==j['counts']
                cmp.check('all_saved_cell_means',simple[cell],cell_saved['values'],f'{rid}/{role}/{cell}')
                NA_counts[rid+'/'+role+'/'+cell]=dict(precision=int(np.isnan(a[:,6]).sum()),mcc=int(np.isnan(a[:,11]).sum()))
                file_records.append(dict(path=relative,bytes=len(b),sha256=sha(b)))
            contributions[rid,role]=derive(expanded)
            point_values[rid,role]=derive(simple)
            for field,value in point_values[rid,role].items():
                cmp.check('per_run_endpoints',value,saved['per_run_endpoint_points'][rid+'_'+role][field],f'{rid}/{role}/{field}')
        # Check saved rank/curve values across roles; this does not verify omitted raw score arrays.
        invariant=[0,1,2,3]+list(range(12,22))
        for cell in cells:
            for role in ROLES[1:]:
                cmp.check('saved_order_metric_invariance',arrays[rid,'raw',cell][:,invariant],arrays[rid,role,cell][:,invariant],rid+'/'+cell+'/'+role)
    assert scalar_cells==288 and len(file_records)==864
    assert len({r['path'] for r in file_records})==864
    actual_files={p.relative_to(root).as_posix() for p in (root/(F+'confirmation')).glob('metrics_*.json')}
    assert actual_files=={r['path'] for r in file_records}
    averages={}
    absolute={}
    for method in METHODS:
        absolute[method]={}
        for role in ROLES:
            absolute[method][role]={}
            for field in FIELDS:
                v=np.mean([contributions[rid,role][field] for _,_,m,rid in expected if m==method],axis=0)
                averages[method,role,field]=v
                point=v.mean(1).sum(0)
                absolute[method][role][field]=point
                cmp.check('absolute_endpoints',point,saved['absolute_endpoint_points'][method][role][field],f'{method}/{role}/{field}')
    # One frozen draw table, used by every role/field/method. Domain grouping stays fixed.
    draws=np.random.Generator(np.random.PCG64(2026102302)).integers(0,40,size=(5000,3,40))
    multiplicities=np.zeros((5000,3,40),dtype=np.int64)
    for d in range(3):
        np.add.at(multiplicities[:,d],(np.arange(5000)[:,None],draws[:,d]),1)
    assert np.all(multiplicities.sum(2)==40)
    weights=multiplicities.reshape(5000,120)/40.
    results={}; flat_rows=[]; interval_count=0; undefined_intervals=0
    for left,right in PAIRS:
        label=left+'-'+right; results[label]={}
        for role in ROLES:
            results[label][role]={}
            for field in FIELDS:
                delta=averages[left,role,field]-averages[right,role,field]
                missing=np.isnan(delta).any(axis=(0,1))
                values=delta.reshape(120,22).copy(); values[:,missing]=0.
                samples=weights@values
                ci=np.quantile(samples,[.025,.975],axis=0)
                mean=delta.mean(1).sum(0)
                by_metric={}
                for col,name in enumerate(COLUMNS):
                    row=dict(mean=None if missing[col] else float(mean[col]),
                             conditional_95pct_interval=None if missing[col] else ci[:,col].tolist())
                    old=saved['comparisons'][label][role][field][name]
                    if missing[col]:
                        assert old['mean'] is None and old['conditional_95pct_interval'] is None
                        undefined_intervals+=1
                        err=0.
                    else:
                        cmp.check('all_predefined_paired_intervals',[row['mean'],*row['conditional_95pct_interval']],
                                  [old['mean'],*old['conditional_95pct_interval']],f'{label}/{role}/{field}/{name}')
                        err=max(abs(x-y) for x,y in zip([row['mean'],*row['conditional_95pct_interval']],
                                                      [old['mean'],*old['conditional_95pct_interval']]))
                    interval_count+=1; by_metric[name]=row
                    flat_rows.append((label,role,field,name,row['mean'],None if missing[col] else ci[0,col],None if missing[col] else ci[1,col],err))
                results[label][role][field]=by_metric
    assert interval_count==2772
    hs=results['H_pos_tuned-S_tuned']['raw']
    gate=dict(O_lower_positive=hs['O']['map']['conditional_95pct_interval'][0]>0,
              F_upper_negative=hs['F']['map']['conditional_95pct_interval'][1]<0,
              N_point_nonnegative=hs['N']['map']['mean']>=0,
              Z_point_nonnegative=hs['Z']['map']['mean']>=0)
    assert all(gate.values())==saved['primary_pass']==False
    hsp={}; seq={}; domain_detail={}
    for o in ORDERS:
        for s in SEEDS:
            key=f'{o}_s{s}'
            hsp[key]={f:point_values[key+'_H_pos_tuned','raw'][f][12]-point_values[key+'_S_tuned','raw'][f][12] for f in FIELDS}
            seq[key]={f:v[12] for f,v in point_values[key+'_SEQ_ref','raw'].items()}
            for f in FIELDS:cmp.check('local_audit_HS_paths',hsp[key][f],local['h_minus_s_per_order_seed'][key][f],key+'/'+f)
            for f,v in seq[key].items():cmp.check('local_audit_SEQ_transitions',v,local['seq_same_transition'][key][f],key+'/'+f)
    for d in 'ABC':
        domain_detail[d]={}
        for m in METHODS:
            old=[]; acquired=[]
            for o,s,name,rid in expected:
                pos=o.index(d)+1
                if name==m and pos<3:
                    old.append(arrays[rid,'raw',f'3_{pos}'][:,12].mean())
                    acquired.append(arrays[rid,'raw',f'{pos}_{pos}'][:,12].mean())
            assert len(old)==6
            v=dict(old_final=np.mean(old),old_acquired=np.mean(acquired),old_forgetting=np.mean(acquired)-np.mean(old))
            domain_detail[d][m]=v
            for f,x in v.items():cmp.check('local_audit_actual_domains',x,local['actual_domain_map'][d][m][f],d+'/'+m+'/'+f)
    pooled={}
    for m in METHODS:
        pooled[m]={}
        for role in ROLES:
            v=sum((counts[rid,role,cell].sum(0) for _,_,name,rid in expected if name==m for cell in ('3_1','3_2','3_3')),np.zeros(4,dtype=np.int64))
            row={k:int(x) for k,x in zip(('tp','fp','tn','fn'),v)}
            assert row==local['final_all_domains_pooled_counts'][m][role]['counts']
            pooled[m][role]=dict(counts=row,classification=from_counts(row))
            cmp.check('local_audit_pooled_classification',pooled[m][role]['classification'],local['final_all_domains_pooled_counts'][m][role]['classification'],m+'/'+role)
    rounded_tables=root/(R+'audit02/tables.zh.txt')
    mode=None; table_absolute=table_delta=0; table_max_error=0.
    for line in rounded_tables.read_text().splitlines():
        if line.startswith('绝对端点：'):mode='absolute';continue
        if line.startswith('差值：'):mode='delta';continue
        if not line.strip() or mode is None:continue
        row=line.split('\t')
        if mode=='absolute':
            m,role,field=row[:3]; values=[None if x=='N/A' else float(x) for x in row[3:]]
            wanted=absolute[m][role][field];table_absolute+=1
        else:
            comparison,role,field,metric=row[:4];values=[None if x=='N/A' else float(x) for x in row[4:]]
            r=results[comparison][role][field][metric]
            wanted=[None]*3 if r['mean'] is None else [r['mean'],*r['conditional_95pct_interval']];table_delta+=1
        x,y=np.asarray(values,float),np.asarray(wanted,float)
        assert np.array_equal(np.isnan(x),np.isnan(y))
        valid=np.isfinite(x)
        err=float(np.max(np.abs(x[valid]-y[valid]))) if valid.any() else 0.
        table_max_error=max(table_max_error,err)
        assert np.allclose(x,y,rtol=1e-10,atol=1e-11,equal_nan=True)
    assert table_absolute==90 and table_delta==2772
    all_signs=[point_values[f'{o}_s{s}_H_pos_tuned','raw'][f][12]-point_values[f'{o}_s{s}_S_tuned','raw'][f][12] for o in ORDERS for s in SEEDS for f in ('O',)]
    hs_cells=[float(arrays[f'{o}_s{s}_H_pos_tuned','raw',cell][:,12].mean()-arrays[f'{o}_s{s}_S_tuned','raw',cell][:,12].mean()) for o in ORDERS for s in SEEDS for cell in CORE]
    output=dict(status='INDEPENDENT_PROVIDED_GROUP_METRICS_RECALCULATION_COMPLETE',
        scope='Supplied JSON group metrics/counts only; no NPY, labels, model/Memory loading, training, inference or new data.',
        started_date='2026-10-26',finished_at=datetime.now(timezone(timedelta(hours=8))).isoformat(),
        environment=dict(python=platform.python_version(),numpy=np.__version__,project_Linux=False),
        elapsed_review_environment_seconds=time.monotonic()-start,
        counts=dict(runs=45,independent_groups=120,groups_per_actual_domain=40,score_cells=scalar_cells,
                    metric_files=len(file_records),group_metric_rows=864*40,columns=22,
                    paired_statistic_entries=interval_count,undefined_statistic_entries=undefined_intervals,
                    full_table_absolute_rows=table_absolute,full_table_comparison_rows=table_delta),
        validation_summary=cmp.rows,all_saved_metric_paths_verified=True,actual_domain_and_group_identity_verified=True,
        full_text_table_max_rounding_error=table_max_error,
        bootstrap=dict(generator='NumPy PCG64',seed=2026102302,replicates=5000,units='actual-domain groups, same draws across all scores',
                       training_seed_resampled=False,training_order_resampled=False),
        primary_conditions=gate,primary_pass=all(gate.values()),
        raw_MAP_absolute={m:{f:absolute[m]['raw'][f][12] for f in FIELDS} for m in METHODS},
        raw_MAP_comparisons={c:{f:results[c]['raw'][f]['map'] for f in FIELDS} for c in results},
        H_minus_S_all_roles_endpoints={role:{field:results['H_pos_tuned-S_tuned'][role][field] for field in ('O','N','Z')} for role in ROLES},
        H_minus_S_per_path=hsp,actual_old_domain_MAP=domain_detail,SEQ_transitions=seq,
        descriptive_decomposition=dict(A_acq_fraction=hs['A_acq']['map']['mean']/hs['O']['map']['mean'],
                                       smaller_decline_fraction=-hs['F']['map']['mean']/hs['O']['map']['mean'],causal_interpretation=False),
        path_signs=dict(O_positive=sum(v['O']>0 for v in hsp.values()),N_positive=sum(v['N']>0 for v in hsp.values()),
                        Z_positive=sum(v['Z']>0 for v in hsp.values()),F_negative=sum(v['F']<0 for v in hsp.values()),
                        H_S_cell_positive=sum(x>0 for x in hs_cells),H_S_cell_negative=sum(x<0 for x in hs_cells),
                        SEQ_same_transition_learning_and_decline=sum(v['D2']>0 and v['G2']>0 for v in seq.values())+sum(v['D3']>0 and v['G3']>0 for v in seq.values())),
        final_all_domain_pooled_counts=pooled,NA_group_counts=NA_counts,
        limits=['Underlying label/score/query metrics not independently recomputed without omitted NPY/labels.',
                'No training or restore behavior re-executed; original receipts are not new runtime verification.',
                'Reported intervals condition on fixed systems, training stream, three orders and three seeds.',
                'No new comparisons, inference family, threshold or experimental data.'],
        execution_failures=[])
    write(out/'independent_recalculation.json',output)
    with (out/'recomputed_comparisons.tsv').open('w',encoding='utf-8') as stream:
        stream.write('comparison\trole\tfield\tmetric\tmean\tlower\tupper\tabs_error_vs_saved\n')
        for row in flat_rows:stream.write('\t'.join('N/A' if v is None else (f'{v:.17g}' if isinstance(v,(float,np.floating)) else str(v)) for v in row)+'\n')
    write(out/'metric_files_checked.json',dict(files=file_records,meaning='All files read as JSON arrays/counts and verified against confirmation matrix records, not against unprovided labels.'))
    print(json.dumps(clean(dict(status=output['status'],counts=output['counts'],primary_conditions=gate,
                               primary_pass=False,max_abs_error=max(v['max_abs_error'] for v in cmp.rows.values()),elapsed_seconds=output['elapsed_review_environment_seconds'])),ensure_ascii=False))


def main():
    p=argparse.ArgumentParser();p.add_argument('input_root',type=Path);p.add_argument('output',type=Path);a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=True)
    try:audit(a.input_root,a.output)
    except BaseException as e:
        write(a.output/'recalculation_failure.json',dict(status='FAILED_REVIEW_ENVIRONMENT_CHECK',error=repr(e),traceback=traceback.format_exc(),not_project_runtime=True))
        raise
if __name__=='__main__':main()
