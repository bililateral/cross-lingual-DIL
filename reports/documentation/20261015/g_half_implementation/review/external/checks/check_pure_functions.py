"""Auditor's CPU-only table algebra checks. No models, Memory, formal data, or project entrypoints.
Actual production functions are AST-extracted unchanged; source identities are recorded.
This is NOT project Linux py310/native qualification and consumes no project authorization.
"""
from pathlib import Path
import ast, hashlib, itertools, json, math, os, platform, random, sys, time, traceback
import numpy as np
import torch
R=Path('/mnt/data/g_half_input');O=Path('/mnt/data/g_half_review/checks');O.mkdir(exist_ok=True)
started=time.monotonic();torch.set_num_threads(1)
source_records=[]
def extract(path,names,namespace):
 text=(R/path).read_text();tree=ast.parse(text)
 nodes=[x for x in tree.body if isinstance(x,(ast.FunctionDef,ast.ClassDef)) and x.name in names]
 assert set(n.name for n in nodes)==set(names)
 code='from __future__ import annotations\n\n'+'\n\n'.join(ast.get_source_segment(text,n) for n in nodes)+'\n'
 name=Path(path).stem+'_extracted.py';(O/name).write_text(code)
 source_records.append({'source':path,'source_sha256':hashlib.sha256((R/path).read_bytes()).hexdigest(),'functions':[{'name':n.name,'lines':[n.lineno,n.end_lineno]} for n in nodes],'extracted':name})
 exec(compile(code,str(O/name),'exec'),namespace)
ns={'torch':torch,'np':np,'math':math}
extract('scripts/step28_g_half.py',['identity','block_scores','account_logits'],ns)
extract('scripts/step28_alias_ranking.py',['hard_candidates'],ns)
extract('scripts/step28_g_half_run.py',['projection'],ns)
inputs={};outputs={};rows=[]
blocks=[np.zeros((2,2)),np.array([[4.,0.],[0.,0.]]),np.array([[-12.,0.],[0.,0.]])]+[np.sin(np.arange(n*m).reshape(n,m)*.7)*5 for n,m in ((2,8),(8,2),(8,8))]
for ci,k in enumerate(blocks):
 for dtype in (torch.float32,torch.float64):
  tag=f'b{ci}_{str(dtype).split(".")[-1]}'
  x=torch.tensor(k,dtype=dtype)[None].requires_grad_();v=x.detach().double().numpy().ravel().tolist();peak=max(v)
  weights=[math.exp(z-peak) for z in v];den=math.fsum(weights)
  ref=.5*math.fsum(v)/len(v)+.5*(peak+math.log(den/len(v)))
  q=np.array([.5/len(v)+.5*w/den for w in weights]).reshape(k.shape)
  y=ns['block_scores'](x);grad,=torch.autograd.grad(y.sum(),x)
  err=abs(y.item()-ref);ge=np.abs(grad[0].numpy()-q).sum()
  if dtype==torch.float32:assert err<=1e-8+1e-7*max(1,abs(ref)) and ge<=2e-7
  else:
   np.testing.assert_allclose(y.detach().numpy(),[ref],atol=1e-12,rtol=1e-12)
   np.testing.assert_allclose(grad[0].numpy(),q,atol=1e-12,rtol=1e-12)
  inputs[tag]=x.detach().numpy();outputs[tag+'_value']=y.detach().numpy();outputs[tag+'_gradient']=grad.numpy();outputs[tag+'_reference_gradient']=q
  rows.append({'tag':tag,'value':y.item(),'reference':ref,'value_error':err,'gradient_l1_error':float(ge)})
# Complete 140-record table, 28 unequal accounts with each shape2..8 repeated four times.
counts=np.tile(np.arange(2,9),4);slots=np.repeat(np.arange(28),counts);n=len(slots)
controllers=np.repeat(np.arange(12),[3]*4+[2]*8)
a1=slots.copy();rng=random.Random(17)
for c in range(12):
 accounts=np.flatnonzero(controllers==c);ids=np.flatnonzero(np.isin(slots,accounts)).tolist();rng.shuffle(ids);offset=0
 for a in accounts:a1[ids[offset:offset+counts[a]]]=a;offset+=counts[a]
table=(np.sin(np.arange(n*(n-1)//2)*.19)+(np.arange(n*(n-1)//2)%23-11)*.13).astype(np.float32)
coef=np.cos(np.arange(378)*.17).astype(np.float32);lookup={p:i for i,p in enumerate(itertools.combinations(range(n),2))}
complete=[]
for label,s in [('A0',slots),('A1',a1)]:
 x=torch.tensor(table,requires_grad=True);scores=ns['account_logits'](x,s,'G_half');loss=(scores*torch.from_numpy(coef)).sum();grad,=torch.autograd.grad(loss,x)
 expected_values=[];expected_grad=np.zeros_like(table,dtype=np.float64)
 for k,(a,b) in enumerate(itertools.combinations(range(28),2)):
  ids=[lookup[tuple(sorted((int(i),int(j))))] for i in np.flatnonzero(s==a) for j in np.flatnonzero(s==b)]
  v=table[ids].astype(np.float64).tolist();peak=max(v);w=[math.exp(z-peak) for z in v];den=math.fsum(w)
  expected_values.append(.5*math.fsum(v)/len(v)+.5*(peak+math.log(den/len(v))))
  expected_grad[ids]+=float(coef[k])*np.array([.5/len(v)+.5*z/den for z in w])
 expected_values=np.array(expected_values)
 err=np.abs(scores.detach().numpy()-expected_values)
 assert np.all(err<=1e-8+1e-7*np.maximum(1,np.abs(expected_values)))
 np.testing.assert_allclose(grad.numpy(),expected_grad,atol=2e-6,rtol=2e-5)
 inputs[label+'_slots']=s;outputs[label+'_scores']=scores.detach().numpy();outputs[label+'_reference_scores']=expected_values
 outputs[label+'_gradient']=grad.numpy();outputs[label+'_reference_gradient']=expected_grad
 complete.append({'view':label,'records':n,'pairs':len(table),'score_count':378,'max_value_error':float(err.max()),'max_gradient_error':float(np.abs(grad.numpy()-expected_grad).max())})
inputs['complete_table']=table;inputs['upstream_weights']=coef
# Actual H-dependent top5 selector, with hand labels. No group/model/Memory objects.
i,j=np.triu_indices(56,1);t=np.full(1540,-3.,np.float32)
for b in (3,4,5,6):t[(i//2==0)&(j//2==b)]=5.
t[(i//2==0)&(j//2==7)]=[4.,0.,0.,0.];t[(i//2==0)&(j//2==8)]=2.
s=np.repeat(np.arange(28),2);truth=np.array([controllers[a]==controllers[b] for a,b in itertools.combinations(range(28),2)],np.float32)
x=torch.tensor(t,requires_grad=True);h=ns['account_logits'](x,s,'G_half');m,positive,selected=ns['hard_candidates'](h,torch.from_numpy(truth))
assert selected[0].tolist()==[3,4,5,6,8]
terms=[]
for q in range(28):terms.append(torch.nn.functional.softplus(m[q,selected[q]][None,:]-m[q,positive[q]][:,None]).mean())
hard=torch.stack(terms).mean();hg,=torch.autograd.grad(hard,x)
assert hg[torch.from_numpy((i//2==0)&(j//2==8))].abs().sum().item()>0
inputs['top5_table']=t;inputs['top5_labels']=truth;outputs['top5_selected']=selected.numpy();outputs['top5_hard_gradient']=hg.numpy()
rejections=[]
for name,t in [('bad_shape',torch.zeros(1,1,2)),('nan',torch.full((1,2,2),float('nan'))),('inf',torch.full((1,2,2),float('inf')))]:
 try:ns['block_scores'](t)
 except ValueError as e:rejections.append({'case':name,'expected_exception':type(e).__name__,'message':str(e)})
 else:raise AssertionError('Invalid input accepted: '+name)
measurements={k:1. for k in ('t_A','t_B','v','h','L','s_F','r_F','s_I','r_I')};measurements.update(W_F=1000,W_I=500)
projection=ns['projection'](measurements);assert projection['projected_total_seconds']==4833.75
assert ns['projection']({**measurements,'t_B':1000})['within_budget'] is False
np.savez_compressed(O/'handwritten_inputs.npz',**inputs);np.savez_compressed(O/'raw_numeric_outputs.npz',**outputs)
result={'status':'PASS_ISOLATED_TABLE_CHECKS_ONLY','environment':{'python':sys.version,'platform':platform.platform(),'numpy':np.__version__,'torch':torch.__version__,'cuda_initialized':torch.cuda.is_initialized(),'torch_threads':torch.get_num_threads()},'source_extraction':source_records,'local_blocks':rows,'complete_table':complete,'top5_query0':selected[0].tolist(),'expected_rejections':rejections,'projection_handwritten_input':measurements,'projection':projection,'model_instances':0,'memory_instances':0,'optimizer_calls':0,'formal_asset_reads':0,'elapsed_seconds':time.monotonic()-started}
(O/'pure_checks_result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n');print(json.dumps(result,ensure_ascii=False,indent=2))
