"""Hand-derived per-edge derivatives, VJP, clip and independent NumPy AdamW.
Only the submitted handmade data/tiny encoder are reused as fixtures.
No project objective or update is used to construct the expected derivatives.
"""
from __future__ import annotations
import copy, itertools, json, math, os
from pathlib import Path
from unittest import mock
import numpy as np
import torch
import step28_er_weight as m
import step28_bge_continual as parent
import test_step28_bge_continual_contracts as fixture
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'evidence/math_reference'; OUT.mkdir(exist_ok=False)
torch.set_num_threads(1)
if hasattr(os,'sched_setaffinity'): os.sched_setaffinity(0,{min(os.sched_getaffinity(0))})
PAIRS=list(itertools.combinations(range(28),2))
def oracle(z,y):
 z=np.asarray(z,np.float64); y=np.asarray(y,np.float64)
 terms={'bce':float(np.mean(np.logaddexp(0,z)-y*z)),'rank':0.,'hard':0.}
 grad={'bce':(np.exp(-np.logaddexp(0,-z))-y)/378,'rank':np.zeros(378),'hard':np.zeros(378)}
 for q in range(28):
  incident=[(k,b if a==q else a) for k,(a,b) in enumerate(PAIRS) if a==q or b==q]
  ids=np.array([k for k,_ in incident]); pos=[k for k,_ in incident if y[k]==1]
  x=z[ids]; mx=float(max(x)); weights=np.exp(x-mx); den=float(sum(weights))
  terms['rank']+=(mx+math.log(den)-sum(float(z[k]) for k in pos)/len(pos))/28
  grad['rank'][ids]+=weights/den/28
  grad['rank'][pos]-=1/(28*len(pos))
  neg=sorted([(k,other) for k,other in incident if y[k]==0],key=lambda x:(-z[x[0]],x[1]))[:5]
  for p in pos:
   for n,_ in neg:
    d=float(z[n]-z[p]); scale=1/(28*len(pos)*5)
    terms['hard']+=float(np.logaddexp(0,d))*scale
    g=math.exp(-float(np.logaddexp(0,-d)))*scale
    grad['hard'][n]+=g; grad['hard'][p]-=g
 terms['total']=terms['bce']+terms['rank']+.5*terms['hard']
 grad['total']=grad['bce']+grad['rank']+.5*grad['hard']
 return terms,grad

def maxdiff(a,b): return float(np.max(np.abs(np.asarray(a)-np.asarray(b))))
def npcopy(x):return x.detach().cpu().double().numpy().copy()
report={'scope':'Handwritten inputs and tiny encoder ONLY; initial Adam counter explicitly rebased after one real warm update','scenarios':[], 'oracle':{}}
y=fixture.handmade_group('independent_label_fixture').labels
z=np.sin(np.arange(378)*.41)*1.4+np.arange(378)*.000013
terms,grads=oracle(z,y)
t=torch.tensor(z,dtype=torch.float64,requires_grad=True); truth=torch.tensor(y,dtype=torch.float64)
observed=parent.ranking.objectives(t,truth,.5)
for key in ('bce','rank','hard','total'):
 value_error=abs(float(observed[key].detach())-terms[key])
 actual=torch.autograd.grad(observed[key],t,retain_graph=True)[0].detach().numpy()
 error=maxdiff(actual,grads[key]); assert value_error<1e-12 and error<1e-12,(key,value_error,error)
 report['oracle'][key]={'value':terms[key],'value_error':value_error,'gradient_error':error}
finite=[]
for index in (0,1,17,100,251,377):
 plus=z.copy();minus=z.copy();plus[index]+=1e-6;minus[index]-=1e-6
 derivative=(oracle(plus,y)[0]['total']-oracle(minus,y)[0]['total'])/(2e-6)
 finite.append(abs(derivative-grads['total'][index]))
assert max(finite)<2e-8
report['oracle']['finite_difference_max_error']=max(finite)
for label,step,limit in [('unclipped',31,1e6),('active_clip',31,.01),('last_step',288,1e6)]:
 c=fixture.config(); c['optimizer']['clip_norm']=limit
 model=fixture.tiny_model();optimizer=m.core.make_optimizer(model,c)
 current=fixture.handmade_group('independent_current',7); history=fixture.handmade_group('independent_history',2)
 fixture.toy_prior_step(model,optimizer,history,c,count=288+step-1)
 names=[n for n,_ in model.named_parameters()];params=list(model.parameters())
 initial=[npcopy(v) for v in params];states=[copy.deepcopy(optimizer.state[v]) for v in params]
 target=np.linspace(-.9,1.3,378,dtype=np.float32);original_target=target.copy()
 cs,hs=4411,7733
 model.train();torch.manual_seed(cs);predc=m.base.logits(model,current,c,'split_rank')
 tc,gc=oracle(npcopy(predc),current.labels)
 current_grad=torch.autograd.grad(predc,params,grad_outputs=torch.tensor(gc['total'],dtype=predc.dtype))
 torch.manual_seed(hs);predh=m.base.logits(model,history,c,'split_rank')
 th,gh=oracle(npcopy(predh),history.labels)
 history_grad=torch.autograd.grad(predh,params,grad_outputs=torch.tensor(gh['total'],dtype=predh.dtype),retain_graph=True)
 residual=npcopy(predh)-target.astype(np.float64)
 mse=float(np.mean(residual**2));mse_derivative=2*residual/378
 mse_grad=torch.autograd.grad(predh,params,grad_outputs=torch.tensor(mse_derivative,dtype=predh.dtype))
 combined=[npcopy(a)+.25*npcopy(b)+.5*npcopy(d) for a,b,d in zip(current_grad,history_grad,mse_grad)]
 norm=math.sqrt(sum(float(np.sum(x*x)) for x in combined));factor=min(1.,limit/(norm+1e-6))
 expected=[x*factor for x in combined]
 wrong=[npcopy(a)+.25*npcopy(b)+.125*npcopy(d) for a,b,d in zip(current_grad,history_grad,mse_grad)]
 assert max(maxdiff(a,b) for a,b in zip(wrong,combined))>1e-3
 captured=[];original_clip=torch.nn.utils.clip_grad_norm_;counts={'forward':0,'zero_grad':0,'clip':0,'step':0}
 real_logits=m.base.logits;real_zero=optimizer.zero_grad;real_step=optimizer.step
 def count_logits(*args,**kw):counts['forward']+=1;return real_logits(*args,**kw)
 def count_zero(*args,**kw):counts['zero_grad']+=1;return real_zero(*args,**kw)
 def count_clip(ps,*args,**kw):
  counts['clip']+=1;ps=list(ps);captured.extend([npcopy(p.grad) for p in ps]);return original_clip(ps,*args,**kw)
 def count_step(*args,**kw):counts['step']+=1;return real_step(*args,**kw)
 with mock.patch.object(m.base,'logits',side_effect=count_logits),mock.patch.object(optimizer,'zero_grad',side_effect=count_zero),mock.patch.object(optimizer,'step',side_effect=count_step),mock.patch.object(torch.nn.utils,'clip_grad_norm_',side_effect=count_clip):
  actual=m.update(model,optimizer,current,history,c,.25,2,step,cs,hs,reference=target,logit_weight=.5,observe=True)
 assert counts=={'forward':2,'zero_grad':1,'clip':1,'step':1},counts
 assert np.array_equal(target,original_target)
 loss_expected=tc['total']+.25*th['total']+.5*mse
 preclip_error=max(maxdiff(a,b) for a,b in zip(captured,combined))
 postclip_error=max(maxdiff(p.grad.numpy(),e) for p,e in zip(params,expected))
 assert abs(actual['total']-loss_expected)<3e-6
 assert preclip_error<3e-6 and postclip_error<3e-6,(preclip_error,postclip_error)
 parameter_rows=[];offset=0
 for group in optimizer.param_groups:
  lr=1e-5*(step/29 if step<=29 else (288-step)/259) if offset==0 else .001
  assert group['lr']==lr
  b1,b2=group['betas'];eps=group['eps'];wd=group['weight_decay']
  for param in group['params']:
   j=next(j for j,p in enumerate(params) if p is param);old=states[j];g=expected[j]
   age=int(old['step'])+1
   exp_avg=b1*npcopy(old['exp_avg'])+(1-b1)*g
   exp_avg_sq=b2*npcopy(old['exp_avg_sq'])+(1-b2)*g*g
   update=lr*(exp_avg/(1-b1**age))/(np.sqrt(exp_avg_sq/(1-b2**age))+eps)
   prediction=initial[j]*(1-lr*wd)-update
   pe=maxdiff(npcopy(param),prediction);me=maxdiff(npcopy(optimizer.state[param]['exp_avg']),exp_avg);ve=maxdiff(npcopy(optimizer.state[param]['exp_avg_sq']),exp_avg_sq)
   assert pe<3e-7 and me<3e-7 and ve<3e-8,(names[j],pe,me,ve)
   parameter_rows.append({'name':names[j],'parameter_error':pe,'first_moment_error':me,'second_moment_error':ve,'step':age,'lr':lr})
  offset+=len(group['params'])
 report['scenarios'].append({'name':label,'current':tc,'history':th,'mse':mse,'loss_expected':loss_expected,'loss_actual':actual['total'],'loss_error':abs(actual['total']-loss_expected),'gradient_norm_expected':norm,'gradient_norm_actual':actual['gradient_norm'],'clip_factor':factor,'pre_clip_gradient_error':preclip_error,'post_clip_gradient_error':postclip_error,'wrong_mse_0125_gradient_difference':max(maxdiff(a,b) for a,b in zip(wrong,combined)),'calls':counts,'parameters':parameter_rows,'actual_update':actual})
 np.savez(OUT/(label+'_components.npz'),**{f'{role}_{j}':value for role,vs in [('current',[npcopy(v) for v in current_grad]),('history',[npcopy(v) for v in history_grad]),('mse',[npcopy(v) for v in mse_grad]),('combined',combined),('clipped',expected)] for j,value in enumerate(vs)})
# Compatibility of the old lambda=1 LOGIT on an independently chosen fixture.
c=fixture.config();a=fixture.tiny_model();oa=m.core.make_optimizer(a,c);h=fixture.handmade_group('legacy_history',3);cur=fixture.handmade_group('legacy_current',8)
fixture.toy_prior_step(a,oa,h,c);b=copy.deepcopy(a);ob=m.core.make_optimizer(b,c);ob.load_state_dict(copy.deepcopy(oa.state_dict()))
ref=np.linspace(.8,-.7,378,dtype=np.float32)
parent.update(a,oa,cur,h,ref,c,'logit',2,1,5021,6501)
m.update(b,ob,cur,h,c,1.,2,1,5021,6501,reference=ref,logit_weight=.5)
assert m.core.state_digest(a.state_dict())==m.core.state_digest(b.state_dict())
assert m.core.state_digest(oa.state_dict())==m.core.state_digest(ob.state_dict())
report['legacy_logit_lambda1']={'model_equal_bitwise':True,'optimizer_equal_bitwise':True}
report['status']='PASS';report['torch_threads']=torch.get_num_threads();report['cpu_affinity']=sorted(os.sched_getaffinity(0))
(OUT/'results.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(report,ensure_ascii=False,indent=2))
