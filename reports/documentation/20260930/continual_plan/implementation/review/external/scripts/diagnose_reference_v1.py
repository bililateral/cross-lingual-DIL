"""Actual diagnosis of failed independent reference, no formal assets."""
import sys, importlib.util, math, json
from pathlib import Path
import numpy as np
import torch
from unittest import mock
R=Path('/mnt/data/bge_continual_external_audit')
sys.argv=['independent_checks_v1.py','--out',str(R/'outputs/diagnosis_import_fixture')]
spec=importlib.util.spec_from_file_location('ref_v1',R/'scripts/independent_checks_v1.py');v=importlib.util.module_from_spec(spec);spec.loader.exec_module(v)
out={}
z=np.linspace(-100,100,378);vals,grad=v.scalar_objective(z);t=torch.tensor(z,dtype=torch.float64,requires_grad=True);got=v.m.ranking.objectives(t,torch.tensor(v.Y,dtype=torch.float64),.5)
rows={}
for k in vals:
 g=torch.autograd.grad(got[k],t,retain_graph=True)[0].numpy();rows[k]={'value_reference':vals[k],'value_project':float(got[k].detach()),'value_diff':float(got[k].detach())-vals[k],'gradient_max_diff':float(np.max(abs(g-grad[k])))}
# Verify the exact source of hard loss differences, no tolerance changes.
ss,positive,negative=v.m.ranking.hard_candidates(t,torch.tensor(v.Y,dtype=torch.float64)); exact=[];piecewise=[];all_diff=[];count=0
for q in range(28):
 d=ss[q,negative[q]][None,:]-ss[q,positive[q]][:,None]
 true=torch.logaddexp(torch.zeros_like(d),d); implemented=torch.nn.functional.softplus(d)
 exact.append(true.mean());piecewise.append(torch.where(d>20,d,true).mean());all_diff.extend((true-implemented).detach().flatten().tolist());count+=int((d>20).sum())
rows['explanation']={'hard_exact':float(torch.stack(exact).mean().detach()),'hard_threshold20':float(torch.stack(piecewise).mean().detach()),'actual_hard':float(got['hard'].detach()),'threshold_branch_count':count,'max_individual_value_difference':max(all_diff),'maximum_expected_bound':math.log1p(math.exp(-20))}
out['softplus_extreme']=rows
# dtypes and explicit rounding of BCE output for the double-valued handmade model.
c=v.m.config(v.m.contract()); current,old=v.handmade('new_B'),v.handmade('old_A');model=v.SmallScore().double();opt=v.m.core.make_optimizer(model,c)
with mock.patch.object(v.m.base,'logits',side_effect=lambda md,g,*a:md(g)):v.m.update(model,opt,old,None,None,c,'seq',1,1,510,511)
model.train();torch.manual_seed(520);p=model(current); y32=torch.tensor(v.Y,dtype=torch.float32);y64=y32.double()
got32=v.m.ranking.objectives(p,y32,.5);got64=v.m.ranking.objectives(p,y64,.5)
per=torch.logaddexp(torch.zeros_like(p),p)-p*y64
variants={'exact_double_mean':per.mean(),'per_cast32_then_mean':per.float().mean(),'double_mean_cast32':per.mean().float(),'actual32':got32['bce'],'actual64':got64['bce']}
out['BCE_mixed_dtype']={'prediction_dtype':str(p.dtype),'target_dtype':str(y32.dtype),'terms':{k:{'dtype':str(x.dtype),'value':float(x.detach())} for k,x in variants.items()},'term32_vs64':{k:float(got32[k].detach()-got64[k].detach()) for k in got32},'actual_total':float(got32['total'].detach()),'prior_reference_total':float(v.torch_reference(p).detach()),'grad_max_diffs':{k:float((torch.autograd.grad(x,p,retain_graph=True)[0]-torch.autograd.grad(got32['bce'],p,retain_graph=True)[0]).abs().max()) for k,x in variants.items()}}
out['fixture_api']={'budget_supplied_has_state':hasattr(v.submitted.HandmadeBudget(),'state'),'runner_requires_state':True,'actual_calibration_function':'loss_gradient','nonexistent_function_used_by_v1':'objective'}
path=R/'outputs/reference_v1_diagnosis.json';path.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
