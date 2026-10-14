"""Web-review handwritten checks only; not project Linux py310 qualification.
Usage: python independent_checks.py INPUT_EXTRACT_DIR EVIDENCE_DIR
No model archive, formal data, network, or Memory payload is accessed.
"""
import sys, json, math, platform, traceback, time
from pathlib import Path
import numpy as np
import torch
root, out = map(Path, sys.argv[1:3]); sys.path.insert(0,str(root/'scripts'))
import step28_balanced_pooling as method
import step28_balanced_pooling_run as run
from step28_continual_population_evaluate import classification

torch.set_num_threads(1); started=time.monotonic()
result={'environment':{'python':sys.version,'platform':platform.platform(),'torch':torch.__version__,'numpy':np.__version__,'device':'cpu','project_py310_qualification':False},'cases':[],'expected_rejections':[]}

def ref(k):
    k=np.asarray(k,dtype=np.float64)
    if k.shape==(2,2):
        d=(k[0,0]+k[1,1]-k[0,1]-k[1,0])/4
        z=.5/(1+math.exp(-2*d));p=np.array([[z,.5-z],[.5-z,z]])
        return float(k.mean()+np.logaddexp(d,-d)-math.log(2)),p
    d=k[0]-k[1];lo=float(d.min());hi=float(d.max())
    for _ in range(200):
        if hi-lo<=1e-13:break
        mid=(lo+hi)/2
        if np.mean(1/(1+np.exp(mid-d)))>.5:lo=mid
        else:hi=mid
    p0=1/(1+np.exp((lo+hi)/2-d))/k.shape[1];p=np.stack([p0,1/k.shape[1]-p0]);nz=p>0
    return float((p*k).sum()-(p[nz]*np.log(p[nz]*k.size)).sum()),p

cases=[('constant',[[3,3],[3,3]]),('cover',[[2,0],[0,2]]),('isolated',[[4,0],[0,0]]),('row',[[2,2],[0,0]]),('large20',[[20,0],[0,0]]),('hard23',[[8,1,-4],[2,-2,3]])]
cases += [(f'2x{m}',np.stack([np.arange(m)*.7-2,np.sin(np.arange(m))*3+.2]).tolist()) for m in range(2,9)]
for name,k in cases:
    for dtype in (torch.float64,torch.float32):
        x=torch.tensor(k,dtype=dtype).unsqueeze(0).requires_grad_();rounded=x.detach()[0].double().numpy()
        fv=method.block_scores(x,'F');grad,=torch.autograd.grad(fv.sum(),x);rv,rp=ref(rounded)
        vt,pt,stats=method.solve(x.detach().transpose(1,2));value_error=abs(float(fv.detach())-rv);plan_error=float(np.abs(grad[0].numpy()-rp).sum())
        assert value_error <= (1e-8 if dtype==torch.float64 else 1e-8+1e-7*max(1,abs(rv)))
        assert plan_error<=2e-4 and abs(float(vt)-rv)<=1e-8 and float(np.abs(pt[0].numpy()-rp.T).sum())<=2e-4
        result['cases'].append({'name':name,'dtype':str(dtype),'input':rounded.tolist(),'reference_value':rv,'reference_plan':rp.tolist(),'actual_value':float(fv.detach()),'actual_gradient':grad[0].tolist(),'value_error':value_error,'plan_L1_error':plan_error,'transpose_diagnostics':stats})
# Independent symmetric 8x8 also exercises FP32 envelope-plan conversion.
x=(torch.eye(8)*3).unsqueeze(0).requires_grad_();fv=method.block_scores(x,'F');g,=torch.autograd.grad(fv.sum(),x)
z=math.exp(3)+7;p=np.full((8,8),1/(8*z));np.fill_diagonal(p,math.exp(3)/(8*z));rv=math.log(z/8)
assert abs(float(fv.detach())-rv)<=1e-8+1e-7*max(1,abs(rv)) and np.abs(g[0].numpy()-p).sum()<=2e-4
result['symmetric_8x8_fp32']={'input':'3*I_8','reference_value':rv,'actual_value':float(fv.detach()),'plan_L1_error':float(np.abs(g[0].numpy()-p).sum())}
# Fixed additive rectangular point; no hyperparameter or random search.
k=torch.tensor([[[2.,4.,7.],[5.,7.,10.]]],dtype=torch.float64)
v,p,stats=method.solve(k);assert abs(float(v)-float(k.mean()))<=1e-8
result['pure_additive_2x3']={'input':k[0].tolist(),'value':float(v),'mean':float(k.mean()),'plan':p[0].tolist(),'diagnostics':stats}
for name,x,kw in [('forced_one_iteration',torch.tensor([[[8.,1.,-4.],[2.,-2.,3.]]]),{'maximum_iterations':1}),('nan',torch.full((1,2,3),float('nan')),{}),('inf',torch.full((1,2,3),float('inf')),{}),('shape_1x2',torch.zeros(1,1,2),{})]:
    stats={}
    try:method.solve(x,stats,**kw)
    except ValueError as e:result['expected_rejections'].append({'case':name,'exception':type(e).__name__,'message':str(e),'stats':stats})
    else:raise AssertionError('Missing expected rejection: '+name)
# Genuine handwritten K2/K3 ground truth: no supplied formal label is opened.
controllers=[i for i,size in enumerate([3]*4+[2]*8) for _ in range(size)]
y=np.array([int(controllers[i]==controllers[j]) for i in range(28) for j in range(i+1,28)],np.uint8)
result['undefined_metric_reproduction']={}
for name,score in [('all_negative',-1.),('all_positive',1.)]:
    raw=classification(y,np.full(378,score))
    result['undefined_metric_reproduction'][name]={'input':{'controller_sizes':[3]*4+[2]*8,'all_logits':score,'pairs':378,'positives':int(y.sum())},'actual':raw,
        'contract_NA_columns':['precision','mcc'] if score<0 else ['mcc']}
# Exercise exact new saved-result entry, not only helper function.
sellers=lambda uid:tuple(f'{uid}_s{i:02}' for i in range(28))
groups=[run.data.Group(f'{d}_hand_{i:02}',sellers(f'{d}_hand_{i:02}'),tuple(tuple((f'{d}_{i}_{j}_{k}','手写标题','手写描述') for k in range(2)) for j in range(28)),tuple(int(x) for x in y)) for d in 'AB' for i in range(20)]
partition={'development':[{'group_uid':g.uid,'domain':g.uid[0]} for g in groups]}
raw=np.full((40,378),-1.,np.float32);scores={name:{'raw':raw.copy(),'stage-cal':raw.astype(np.float64),'first-cal':raw.astype(np.float64)} for name in run.POINTS}
collection=out/'na_pipeline';saved=run.collect(collection,scores,groups,partition);summary=run.finalize(collection)
result['saved_pipeline']={'status':summary['status'],'B_F_raw_A_precision':summary['absolute']['B_F']['raw']['by_domain']['A']['precision'],'B_F_raw_A_mcc':summary['absolute']['B_F']['raw']['by_domain']['A']['mcc'],'F_minus_S_raw_A_precision':summary['comparisons']['F-S']['raw']['A']['precision'],'pooled':summary['absolute']['B_F']['raw']['fixed_half_classification']['all']}
# Check exact new objective failure context from a handmade bad table.
t=torch.zeros(1540);t[0]=float('nan');st={}
try:method.objective(t,groups[0],29,'F',stats=st)
except ValueError as e:result['objective_failure_context']={'group_uid':groups[0].uid,'injected_nonfinite_upper_triangle_index':0,'exception':str(e),'stats':st,'group_uid_in_exception':groups[0].uid in str(e)}
else:raise AssertionError('Nonfinite objective must reject')
result['elapsed_seconds']=time.monotonic()-started
result['status']='EXPECTED_BEHAVIORS_AND_METRIC_DEFECT_REPRODUCED'
(out/'independent_checks.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False))
print(json.dumps({k:v for k,v in result.items() if k!='cases'},ensure_ascii=False,indent=2,allow_nan=False))
print('solver_cases_checked=',len(result['cases']))
