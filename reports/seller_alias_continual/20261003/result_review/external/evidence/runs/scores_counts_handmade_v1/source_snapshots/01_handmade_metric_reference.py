#!/usr/bin/env python3
"""Three handmade complete-28-account cases and one analytic calibration case.

Only selected pure production functions are extracted via AST. No project module
is imported, and no production loader, model, label source or GPU is invoked.
"""
from pathlib import Path
import ast
import hashlib
import json
import math
import numpy as np
from independent_scores_counts import COLS, class_rates, softplus, sigmoid

P=Path('/workspace/scratch/ca13b63db3f0/review_work/input/project')
OUT=Path(__file__).resolve().parent
K=(1,3,5,10)
SOURCES=[]

def extract(path,functions,env):
    raw=path.read_bytes()
    source=raw.decode()
    parsed=ast.parse(source)
    selected=[node for node in parsed.body if isinstance(node,ast.FunctionDef) and node.name in functions]
    assert {node.name for node in selected}==set(functions)
    module=ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0),*selected],type_ignores=[])
    ast.fix_missing_locations(module)
    exec(compile(module,str(path),'exec'),env)
    SOURCES.append({'path':str(path.relative_to(P)),'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest(),
                    'extracted_functions':{node.name:[node.lineno,node.end_lineno] for node in selected}})
    return env

def scalar_curve(labels,scores):
    pairs=list(zip(scores.tolist(),labels.tolist()))
    npos=sum(labels)
    nneg=len(labels)-npos
    last_recall=last_fpr=0.0
    last_precision=1.0
    ap=area_pr=0.0
    recall_cap=0.0
    for threshold in sorted(set(scores),reverse=True):
        tp=sum(int(y) for s,y in pairs if s>=threshold)
        fp=sum(1-int(y) for s,y in pairs if s>=threshold)
        recall=tp/npos
        fpr=fp/nneg
        precision=tp/(tp+fp)
        ap+=(recall-last_recall)*precision
        area_pr+=(recall-last_recall)*(precision+last_precision)/2
        if fpr<=.01:
            recall_cap=max(recall_cap,recall)
        last_recall,last_fpr,last_precision=recall,fpr,precision
    positive=[s for s,y in pairs if y==1]
    negative=[s for s,y in pairs if y==0]
    # Independent Mann-Whitney pairwise definition (half credit for score ties).
    roc=math.fsum(1.0 if a>b else .5 if a==b else 0.0 for a in positive for b in negative)/(npos*nneg)
    return {'average_precision':ap,'trapezoidal_pr_auc':area_pr,'roc_auc':roc,'recall_at_fpr_1pct':recall_cap}

def scalar_retrieval(labels,scores):
    edge={}
    n=0
    for i in range(28):
        for j in range(i+1,28):
            edge[i,j]=(float(scores[n]),int(labels[n]))
            n+=1
    values=[]
    for i in range(28):
        candidates=[]
        for j in range(28):
            if i==j:
                continue
            score,rel=edge[min(i,j),max(i,j)]
            candidates.append((score,j,rel))
        candidates.sort(key=lambda t:(-t[0],t[1]))
        positive_ranks=[rank for rank,(_,_,rel) in enumerate(candidates,1) if rel]
        total=len(positive_ranks)
        assert total in (1,2)
        ap=math.fsum(k/rank for k,rank in enumerate(positive_ranks,1))/total
        rr=1/min(positive_ranks)
        recalls=[sum(r<=k for r in positive_ranks)/total for k in K]
        ndcgs=[math.fsum(1/math.log2(r+1) for r in positive_ranks if r<=k)/
                math.fsum(1/math.log2(r+1) for r in range(1,min(k,total)+1)) for k in K]
        values.append([ap,rr,*recalls,*ndcgs])
    return np.asarray([math.fsum(row[k] for row in values)/28 for k in range(10)])

def reference(labels,scores):
    result=scalar_curve(labels,scores)
    probs=[1/(1+math.exp(-float(s))) for s in scores]
    result['brier']=math.fsum((p-y)**2 for p,y in zip(probs,labels))/len(labels)
    result['log_loss']=-math.fsum(math.log(p) if y else math.log1p(-p) for p,y in zip(probs,labels))/len(labels)
    counts={k:0 for k in ('tp','fp','fn','tn')}
    for s,y in zip(scores,labels):
        counts[('tp' if y else 'fp') if s>=0 else ('fn' if y else 'tn')]+=1
    result.update(class_rates(counts))
    result.update(dict(zip(COLS[12:],scalar_retrieval(labels,scores))))
    return np.array([result[k] for k in COLS]),counts

def main():
    src=P/'scripts/step28_continual_population_evaluate.py'
    env={'np':np,'math':math,'KS':K,'CLASS_KEYS':COLS[:12]}
    extract(src,['curve_metrics','confusion_metrics','classification','retrieval','group_metrics'],env)
    ids=[i//3 for i in range(12)]+[4+(i-12)//2 for i in range(12,28)]
    # Explicitly handmade truth. These IDs have no relationship to formal input.
    labels=np.array([int(ids[i]==ids[j]) for i in range(28) for j in range(i+1,28)],dtype=np.uint8)
    assert int(labels.sum())==20
    scores={
        'all_tied_zero':np.zeros(378,dtype=np.float64),
        'mixed_ties':np.array([((i*17)%11-5)/3 for i in range(378)],dtype=np.float64),
        'all_negative_ordered':np.array([-4+i/1000 for i in range(378)],dtype=np.float64),
    }
    results=[]
    overall=0.0
    for name,x in scores.items():
        expected,count=reference(labels,x)
        got,actual_count=env['group_metrics'](labels[None],x[None])
        delta=float(np.max(abs(got[0]-expected)))
        assert delta<1e-12,(name,delta)
        assert actual_count==[count]
        # Check positive affine rank invariance in pure routines too.
        calibrated,_=env['group_metrics'](labels[None],(1.25*x+.75)[None])
        rank=[0,1,2,3,*range(12,22)]
        assert np.array_equal(calibrated[:,rank],got[:,rank])
        overall=max(overall,delta)
        results.append({'name':name,'maximum_error':delta,'reference':dict(zip(COLS,expected.tolist())),'confusion':count})
    calenv={'np':np,'BOUNDS':((.001,100.0),(-100.,100.))}
    extract(P/'scripts/step28_alias_calibration.py',['loss_gradient','projected_gradient'],calenv)
    x=np.array([-4.0,-1.0,0.0,.5,3.0])
    y=np.array([0.0,1.0,0.0,1.0,1.0])
    theta=np.array([.7,-.2])
    loss,grad=calenv['loss_gradient'](theta,x,y)
    def f(p):
        z=p[0]*x+p[1]
        return math.fsum((math.log1p(math.exp(float(v)))-float(t)*float(v)) for v,t in zip(z,y))/len(x)
    step=1e-5
    numeric=[]
    for axis in range(2):
        plus=theta.copy(); plus[axis]+=step
        minus=theta.copy(); minus[axis]-=step
        numeric.append((f(plus)-f(minus))/(2*step))
    assert abs(f(theta)-loss)<1e-12
    assert max(abs(grad-numeric))<1e-9
    bounds_cases=[([.001,-100],[1,1],[0,0]),([100,100],[-1,-1],[0,0]),([.001,100],[-1,1],[-1,1])]
    for point,gradient,expected in bounds_cases:
        assert np.array_equal(calenv['projected_gradient'](np.array(point),np.array(gradient)),expected)
    out={'status':'PASS_HANDMADE_METRIC_AND_CALIBRATION_REFERENCE','cases':results,'maximum_22_metric_error':overall,
         'calibration_case':{'loss':loss,'gradient':grad.tolist(),'finite_difference':numeric,
                             'maximum_gradient_error':float(max(abs(grad-numeric))),'projected_boundary_cases':3},
         'sources':SOURCES,
         'boundary':'Three handmade metric cases plus one handmade calibration objective, not formal label evaluation or proof of all corner cases.'}
    (OUT/'handmade_numeric_results.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'status':out['status'],'case_names':list(scores),'maximum_22_metric_error':overall,'calibration':out['calibration_case']},ensure_ascii=False,indent=2))

if __name__=='__main__':
    main()
