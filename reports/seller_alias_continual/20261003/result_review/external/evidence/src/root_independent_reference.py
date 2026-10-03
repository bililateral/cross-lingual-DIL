#!/usr/bin/env python3
"""Independent scalar retrieval formulas and direct-index paired bootstrap.

Only the supplied saved small results and in-memory handwritten examples are
read. No project imports, model imports, formal labels, or training are used.
"""
import ast
import hashlib
import json
import math
from pathlib import Path
from typing import Any
import numpy as np

WORK = Path('/workspace/scratch/ca13b63db3f0/review_work')
PROJECT = WORK/'input'/'project'
OUT = WORK/'evidence'/'root_reference'
OUT.mkdir(parents=True, exist_ok=True)
ORDERS = ('ABC','BCA','CAB')
ENDPOINTS = ('O','N','Z','F_first','F','G','final_all')
ROOTS = {
    'low':PROJECT/'reports/seller_alias_continual/20261002/er_low_execution/20261002_124018/job/evaluation',
    'logit':PROJECT/'reports/seller_alias_continual/20261002/logit_weight_execution/20261002_153200/workspace/reports/job/evaluation',
}

def read_json(p):
    return json.loads(p.read_text())

def mean(vals):
    values=list(vals)
    return math.fsum(float(x) for x in values)/len(values)

def linear_percentile(values, p):
    ordered=sorted(float(x) for x in values)
    pos=(len(ordered)-1)*p
    lo=math.floor(pos)
    hi=math.ceil(pos)
    return ordered[lo]+(pos-lo)*(ordered[hi]-ordered[lo])

def endpoint(R, name):
    if name=='O': return (R[3,1]+R[3,2])/2
    if name=='N': return (R[2,2]+R[3,3])/2
    if name=='Z': return R[3,3]
    if name=='F_first': return R[1,1]-R[3,1]
    if name=='F': return ((R[1,1]-R[3,1])+(R[2,2]-R[3,2]))/2
    if name=='G': return ((R[2,2]-R[1,2])+(R[3,3]-R[2,3]))/2
    if name=='final_all': return (R[3,1]+R[3,2]+R[3,3])/3
    raise ValueError(name)

def direct_index_bootstrap():
    draws=np.random.Generator(np.random.PCG64(20260930)).integers(0,20,(5000,3,20))
    records=[]
    sets=[]
    for study,root in ROOTS.items():
        manifest=read_json(root/'collected.json')
        reused=read_json(root/'reference'/'collected.json')
        original=read_json(root/'evaluation.json')
        assert np.array_equal(draws,np.load(root/'bootstrap_draws.npy',allow_pickle=False))
        assert manifest['group_ids']==reused['group_ids']
        assert manifest['domains']==reused['domains']
        rows={d:np.array([i for i,x in enumerate(manifest['domains']) if x==d]) for d in 'ABC'}
        assert all(len(x)==20 for x in rows.values())
        matrices={}
        for container,source in [(manifest,root),(reused,root/'reference')]:
            for point,info in container['points'].items():
                matrices[point]=np.load(source/info['raw']['matrix']['path'],allow_pickle=False)
        comparisons= [('tenth','quarter'),('tenth','seq')] if study=='low' else [('logit_quarter','quarter'),('logit_quarter','seq'),('quarter','seq')]
        needed_arms=sorted({arm for pair in comparisons for arm in pair})
        cached={}
        for arm in needed_arms:
            for metric in ('map','recall_at_5'):
                k=manifest['metric_columns'].index(metric)
                per_order={}
                per_boot={}
                for order in ORDERS:
                    R={}
                    B={}
                    for t in (1,2,3):
                        point=f'{order}_shared' if t==1 else f'{order}_{arm}_stage{t}'
                        for j,d in enumerate(order,1):
                            scores=matrices[point][rows[d],k]
                            R[t,j]=mean(scores)
                            # Directly select group rows, with the actual domain's
                            # draw reused across all stages, methods, and orders.
                            B[t,j]=scores[draws[:, 'ABC'.index(d), :]].mean(axis=1)
                    per_order[order]={e:endpoint(R,e) for e in ENDPOINTS}
                    per_boot[order]={e:endpoint(B,e) for e in ENDPOINTS}
                cached[arm,metric]=(per_order,per_boot)
        for left,right in comparisons:
            comp=f'{left}_minus_{right}'
            for e in ENDPOINTS:
                for metric in ('map','recall_at_5'):
                    a,ab=cached[left,metric]
                    b,bb=cached[right,metric]
                    per={o:a[o][e]-b[o][e] for o in ORDERS}
                    value=mean(per.values())
                    boot=sum((ab[o][e]-bb[o][e])/3 for o in ORDERS)
                    ci=[linear_percentile(boot,.025),linear_percentile(boot,.975)]
                    expected=original['comparisons'][comp]['primary'][e][metric]
                    diffs=[abs(value-expected['mean'])]+[abs(x-y) for x,y in zip(ci,expected['conditional_95pct_interval'])]+[abs(per[o]-expected['per_order'][o]) for o in ORDERS]
                    rec={'study':study,'comparison':comp,'endpoint':e,'metric':metric,'mean':value,'conditional_95pct_interval':ci,'per_order':per,'original':expected,'max_absolute_difference':max(diffs)}
                    records.append(rec)
                    assert max(diffs)<2e-13,rec
        sets.append({'study':study,'draws_shape':list(draws.shape),'saved_draws_byte_values_equal':True,'actual_domain_group_counts':{d:len(rows[d]) for d in rows}})
    result={'method':'Explicit R(t,arrival) means and direct row indexing; Python sorted linear interpolation. No project aggregation or quantile functions imported.', 'scope':'Five comparisons; all seven endpoints for MAP and Recall@5; all 5000 common domain draws. AP/MAP inputs remain saved truth-derived group metrics.', 'comparisons':records,'draw_checks':sets,'records_checked':len(records),'max_absolute_difference':max(r['max_absolute_difference'] for r in records)}
    (OUT/'direct_index_bootstrap.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'direct_index_records':len(records),'max_absolute_difference':result['max_absolute_difference'],'draw_sets_match':len(sets)}))
    for r in records:
        if r['endpoint'] in ('O','N','Z','F_first','final_all') and r['metric']=='map':
            print(json.dumps({k:r[k] for k in ['comparison','endpoint','metric','mean','conditional_95pct_interval']},ensure_ascii=False))

def handwritten_reference():
    source=PROJECT/'scripts/step28_continual_population_evaluate.py'
    tree=ast.parse(source.read_text())
    names={'curve_metrics','confusion_metrics','classification','retrieval'}
    selected=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names]
    ns={'np':np,'math':math,'Any':Any,'KS':(1,3,5,10)}
    # Executing only the exact pure function definitions avoids module-level
    # project imports and has no model or data entrypoint side effects.
    exec(compile(ast.Module(body=selected,type_ignores=[]),str(source),'exec'),ns)
    n=28
    block=[]
    for g in range(4): block.extend([g]*3)
    for g in range(4,12): block.extend([g]*2)
    edges=[(i,j) for i in range(n) for j in range(i+1,n)]
    labels=np.array([int(block[i]==block[j]) for i,j in edges],dtype=np.uint8)
    assert len(edges)==378 and int(labels.sum())==20
    cases={
        'all_tied':np.zeros(378),
        'bounded_ties':np.array([((i*17+j*11)%19-9)/3 for i,j in edges],dtype=float),
        'positive_edges_above':np.where(labels==1,2.,-1.),
        'positive_edges_below':np.where(labels==1,-2.,1.),
    }
    rows=[]
    for name,scores in cases.items():
        edgescores={edge:float(s) for edge,s in zip(edges,scores)}
        expected=[]
        for q in range(n):
            candidates=[j for j in range(n) if j!=q]
            candidates.sort(key=lambda j:(-edgescores[tuple(sorted((q,j)))],j))
            hits=[i+1 for i,j in enumerate(candidates) if block[j]==block[q]]
            ap=mean((k+1)/r for k,r in enumerate(hits))
            rr=1/hits[0]
            recalls=[sum(r<=k for r in hits)/len(hits) for k in (1,3,5,10)]
            ndcg=[]
            for k in (1,3,5,10):
                dcg=math.fsum(1/math.log2(r+1) for r in hits if r<=k)
                idcg=math.fsum(1/math.log2(r+1) for r in range(1,min(k,len(hits))+1))
                ndcg.append(dcg/idcg)
            expected.append([ap,rr,*recalls,*ndcg])
        expected=np.array([mean(np.array(expected)[:,k]) for k in range(10)])
        actual=ns['retrieval'](labels[None,:],scores[None,:],n)[0]
        maxerr=float(np.max(np.abs(expected-actual)))
        assert maxerr<1e-14,(name,maxerr)
        thresholds=sorted(set(float(s) for s in scores),reverse=True)
        rec_prev=0.; prec_prev=1.; expected_ap=0.; expected_trap=0.; recall_fpr=0.
        for threshold in thresholds:
            tp=sum(int(y) for y,s in zip(labels,scores) if s>=threshold)
            fp=sum(1-int(y) for y,s in zip(labels,scores) if s>=threshold)
            recall=tp/20; precision=tp/(tp+fp)
            expected_ap+=(recall-rec_prev)*precision
            expected_trap+=(recall-rec_prev)*(precision+prec_prev)/2
            if fp/358<=.01: recall_fpr=max(recall_fpr,recall)
            rec_prev,prec_prev=recall,precision
        pos=[float(s) for s,y in zip(scores,labels) if y==1]
        neg=[float(s) for s,y in zip(scores,labels) if y==0]
        auc=mean((1. if a>b else .5 if a==b else 0.) for a in pos for b in neg)
        expected_curves={'average_precision':expected_ap,'trapezoidal_pr_auc':expected_trap,'roc_auc':auc,'recall_at_fpr_1pct':recall_fpr}
        actual_curves=ns['curve_metrics'](labels,scores)
        curve_error=max(abs(actual_curves[k]-v) for k,v in expected_curves.items())
        assert curve_error<1e-14,(name,expected_curves,actual_curves)
        rows.append({'case':name,'handwritten_pairs':378,'handwritten_positives':20,'retrieval_expected':expected.tolist(),'retrieval_actual':actual.tolist(),'retrieval_max_absolute_difference':maxerr,'curves_expected':expected_curves,'curves_actual':actual_curves,'curves_max_absolute_difference':curve_error})
    result={'scope':'Four handwritten 28-account examples, not formal-label recomputation or native training tests. Curves computed via independent threshold sets and pairwise AUROC; retrieval via explicit query/candidate loops.', 'source':{'path':str(source.relative_to(PROJECT)),'bytes':source.stat().st_size,'sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'executed_exact_function_definitions':sorted(names)},'cases':rows}
    (OUT/'handwritten_metric_reference.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'handwritten_cases':len(rows),'max_retrieval_error':max(r['retrieval_max_absolute_difference'] for r in rows),'max_curve_error':max(r['curves_max_absolute_difference'] for r in rows)}))

if __name__=='__main__':
    handwritten_reference()
    direct_index_bootstrap()
