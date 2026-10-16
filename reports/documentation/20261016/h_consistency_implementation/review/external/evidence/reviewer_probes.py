"""Isolated reviewer handwritten checks, NOT project Linux qualification.
No model, tokenizer, original saved scores, formal data or labels are loaded.
Input source root is read-only; all outputs are written below this script's outputs/.
"""
from __future__ import annotations
import copy, hashlib, itertools, json, math, os, pathlib, sys
from unittest import mock
import numpy as np
import torch
HERE=pathlib.Path(__file__).resolve().parent
INPUT=HERE.parent/'input'/'sources'
sys.path.insert(0,str(INPUT/'scripts'))
import step28_h_consistency as cons
import step28_h_consistency_run as run
import step28_g_half as old_h
OUT=HERE/'probe_outputs'
OUT.mkdir(exist_ok=False)
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
torch.use_deterministic_algorithms(True)

def write(name, value):
    p=OUT/name
    p.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')

pairs=list(itertools.combinations(range(28),2))
edge_index={pair:i for i,pair in enumerate(pairs)}
controllers=[i for i,n in enumerate([3]*4+[2]*8) for _ in range(n)]
uid='reviewer_handwritten_28x2'
group=cons.data.Group(uid,tuple(f'reviewer_s{i:02d}' for i in range(28)),
    tuple(tuple((f'reviewer_r{i:02d}_{j}','手写标题'+str(i)+'_'+str(j),'手写描述'+str(i)+'_'+str(j)) for j in range(2)) for i in range(28)),
    tuple(int(controllers[i]==controllers[j]) for i,j in pairs))
group.validate()
slots=[cons.record.assignment(group),cons.record.assignment(group,17)]
k_index={p:i for i,p in enumerate(itertools.combinations(range(56),2))}
idx=np.arange(len(k_index),dtype=np.float64)
k_base=np.sin(.13*idx)+.35*np.cos(.071*idx)
write('handwritten_inputs.json',{
    'scope':'Reviewer-created labels and strings only; not project examples or data',
    'group':{'uid':group.uid,'seller_ids':list(group.seller_ids) if hasattr(group,'seller_ids') else [f'reviewer_s{i:02d}' for i in range(28)],
             'records':group.records,'labels':group.labels},
    'assignment_seed':17,'slots':[s.tolist() for s in slots],
    'K_formula':'sin(0.13*i)+0.35*cos(0.071*i), i=0..1539',
    'K_float64':k_base.tolist(),
    'dtype_runs':['float64','float32'],
})

# Independent enumeration; neither package literal_pool nor table_reference is used.
def pool_reference(k, slot):
    hs=[]; blocks=[]
    for a,b in pairs:
        indices=np.array([k_index[tuple(sorted((int(i),int(j))))]
                          for i in np.flatnonzero(slot==a) for j in np.flatnonzero(slot==b)],dtype=np.int64)
        v=k[indices].astype(np.float64)
        peak=float(v.max()); ex=np.exp(v-peak); sm=ex/ex.sum()
        hs.append(.5*float(v.mean())+.5*(peak+math.log(float(ex.sum()))-math.log(len(v))))
        blocks.append((indices,.5/len(v)+.5*sm))
    return np.array(hs),blocks

def queries(h):
    return np.array([[h[edge_index[tuple(sorted((i,j)))]] for j in range(28) if j!=i] for i in range(28)])

def analytic_js(a,b):
    # Independently normalized probabilities and arithmetic mixture, not logaddexp.
    probs=[]
    for u in (a,b):
        z=u-u.max(axis=1,keepdims=True); p=np.exp(z); p/=p.sum(axis=1,keepdims=True); probs.append(p)
    p,q=probs; m=(p+q)/2
    lp=np.log(p/m); lq=np.log(q/m)
    klp=np.sum(p*lp,axis=1,keepdims=True); klq=np.sum(q*lq,axis=1,keepdims=True)
    value=float((klp+klq).mean()/2)
    return value,[p*(lp-klp)/56,q*(lq-klq)/56]

summary={'scope':'ISOLATED_REVIEWER_CPU_HANDWRITTEN_NOT_PROJECT_QUALIFICATION','torch':torch.__version__,'numpy':np.__version__,'shared_K':{},'off_C_equals_old_H':{}}
for dtype in (torch.float64,torch.float32):
    key=str(dtype).split('.')[-1]
    t=torch.tensor(k_base,dtype=dtype,requires_grad=True)
    h=[cons.account_logits(t,s,'H_cons') for s in slots]
    query=[cons.query_logits(v) for v in h]
    loss=cons.consistency(*query)
    actual_query=[torch.autograd.grad(loss,v,retain_graph=True)[0].detach().numpy() for v in query]
    actual_k=torch.autograd.grad(loss,t)[0].detach().numpy()
    hd=[v.detach().numpy().astype(np.float64) for v in h]
    value,qgrad=analytic_js(*[queries(v) for v in hd])
    expected_k=np.zeros(len(k_base),np.float64)
    pool_error=[]
    for branch,slot in enumerate(slots):
        href,blocks=pool_reference(t.detach().numpy(),slot)
        pool_error.append(float(np.max(np.abs(hd[branch]-href))))
        edge_grad=np.zeros(378,np.float64)
        for a in range(28):
            js=[b for b in range(28) if b!=a]
            for col,b in enumerate(js):
                edge_grad[edge_index[tuple(sorted((a,b)))]]+=qgrad[branch][a,col]
        for weight,(indices,jac) in zip(edge_grad,blocks):
            np.add.at(expected_k,indices,weight*jac)
    err_l1=float(np.abs(actual_k-expected_k).sum())
    val_error=abs(float(loss.detach())-value)
    query_l1=[float(np.abs(x-y).sum()) for x,y in zip(actual_query,qgrad)]
    limit=5e-7 if dtype==torch.float32 else 1e-12
    assert err_l1<=limit,(key,err_l1,limit)
    assert all(v<2e-7 if dtype==torch.float32 else v<1e-12 for v in query_l1)
    assert val_error <= (1e-8+1e-7*abs(value) if dtype==torch.float32 else 1e-12+1e-12*abs(value))
    np.savez(OUT/f'shared_K_{key}.npz',K=t.detach().numpy(),H0=hd[0],H1=hd[1],
             query_gradient_actual_0=actual_query[0],query_gradient_actual_1=actual_query[1],
             query_gradient_reference_0=qgrad[0],query_gradient_reference_1=qgrad[1],
             K_gradient_actual=actual_k,K_gradient_reference=expected_k)
    summary['shared_K'][key]={'actual_C':float(loss.detach()),'reference_C':value,'value_abs_error':val_error,
        'query_gradient_L1':query_l1,'shared_K_total_gradient_L1':err_l1,'H_abs_max_before_output_cast':pool_error}
    for role in ('current','history'):
        target=None if role=='current' else (k_base+2*np.sin(.31*np.arange(len(k_base)))).astype(np.float32)
        a=torch.tensor(k_base,dtype=dtype,requires_grad=True)
        b=a.detach().clone().requires_grad_(True)
        j0,_=old_h.objective(a,group,17,'G_half',target)
        j1,_=cons.objective(b,group,17,'H_cons',target,qualification_without_c=True)
        ga=torch.autograd.grad(j0,a)[0]; gb=torch.autograd.grad(j1,b)[0]
        assert torch.equal(j0,j1) and torch.equal(ga,gb),(key,role)
        summary['off_C_equals_old_H'][key+'_'+role]={'objective_bitwise_equal':True,'K_gradient_bitwise_equal':True,'objective':float(j0.detach())}

# Three DISTINCT roots, intentionally COLLIDING relative filenames and DIFFERENT H/S values.
# Only the baseline metadata supplier is mocked. Real finalize, loading, matching and bootstrap run.
roots={a:OUT/('distinct_root_'+a) for a in ('H_cons','S','G_half')}
columns=list(run.parent.metrics.COLUMNS)
ids=[f'reviewer_{d}_{i:02d}' for d in 'AB' for i in range(20)]
domains=[d for d in 'AB' for _ in range(20)]
map_values={'H_cons':{'A':.6,'B':.7},'S':{'A':.4,'B':.4},'G_half':{'A':.5,'B':.55}}
loss_values={'H_cons':{'A':.3,'B':.4},'S':{'A':.3,'B':.3},'G_half':{'A':.5,'B':.45}}
collections={}
for arm,root in roots.items():
    origin=root if arm=='H_cons' else root/'evaluation'
    origin.mkdir(parents=True)
    col={'group_ids':ids,'domains':domains,'columns':columns,'points':{},'source_files':run.sources(),
         'status':'ALL_6_H_METRIC_COUNT_SETS_SAVED'}
    for stage in 'AB':
        col['points'][stage+'_'+arm]={}
        for role in run.parent.ROLES:
            name=f'{stage}_{role}' # exactly the same in all three roots
            vals=np.full((40,22),.5)
            vals[:,columns.index('map')]=map_values[arm][stage]
            for key in ('brier','log_loss'):vals[:,columns.index(key)]=loss_values[arm][stage]
            counts=[dict(tp=10,fp=2,fn=10,tn=356) for _ in range(40)]
            # One required A-domain group has undefined precision/MCC, never remove it.
            if arm=='S' and stage=='B':
                counts[0]=dict(tp=0,fp=0,fn=20,tn=358)
                vals[0,columns.index('precision')]=np.nan;vals[0,columns.index('mcc')]=np.nan
            values=[[None if not np.isfinite(v) else float(v) for v in row] for row in vals]
            mp=origin/(name+'_metrics.json');cp=origin/(name+'_counts.json');qp=origin/(name+'_queries.npy')
            run.data.write_json(mp,{'columns':columns,'values':values})
            run.data.write_json(cp,counts)
            q=np.repeat(vals[:,None,12:],28,axis=1)
            qr=run.previous.save_array(qp,q,origin)
            col['points'][stage+'_'+arm][role]={'matrix':run.data.record(mp,origin),'counts':run.data.record(cp,origin),'queries':qr}
    run.data.write_json(origin/'collected.json',col)
    collections[arm]=col
baseline={a:(roots[a],collections[a],{}) for a in ('S','G_half')}
with mock.patch.object(run,'baseline_runs',return_value=baseline):
    result=run.finalize(roots['H_cons'])
for domain in 'AB':
    assert math.isclose(result['comparisons']['H_cons-S']['raw'][domain]['map']['mean'],.3,abs_tol=1e-12)
    assert math.isclose(result['comparisons']['H_cons-G_half']['raw'][domain]['map']['mean'],.15,abs_tol=1e-12)
assert result['success']['passed'] is True and result['success']['direct_H_new_B_positive'] is True
assert math.isclose(result['forgetting_comparisons']['H_cons-S']['raw']['map']['mean'],-.1,abs_tol=1e-12)
assert math.isclose(result['forgetting_comparisons']['H_cons-S']['raw']['log_loss']['mean'],.1,abs_tol=1e-12)
na=result['comparisons']['H_cons-S']['raw']['A']['precision']
assert na['mean'] is None and na['conditional_95pct_interval'] is None and na['undefined_groups']==1 and na['total_groups']==20
# Direct comparison is NOT derived by subtracting S comparison endpoints.
summary['distinct_roots']={
    'roots':{a:str(r.relative_to(OUT)) for a,r in roots.items()},'colliding_relative_filenames':True,
    'MAP_H_cons_minus_S_B':result['comparisons']['H_cons-S']['raw']['B']['map'],
    'MAP_H_cons_minus_H_B':result['comparisons']['H_cons-G_half']['raw']['B']['map'],
    'MAP_forgetting_difference':result['forgetting_comparisons']['H_cons-S']['raw']['map'],
    'log_loss_forgetting_difference':result['forgetting_comparisons']['H_cons-S']['raw']['log_loss'],
    'fixed_group_NA':na,
    'meaning':'Saved algebra/routing fixture only; synthetic true booleans are NOT model results.'}
write('probe_summary.json',summary)
print(json.dumps(summary,ensure_ascii=False,indent=2,allow_nan=False))
