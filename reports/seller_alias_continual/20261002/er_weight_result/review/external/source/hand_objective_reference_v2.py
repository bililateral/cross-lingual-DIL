"""Small synthetic reference; no project imports, no formal data, no model weights.
AST extracts only the actual pure metric/loss/update function bodies, then tests
against hand arithmetic / independent NumPy derivatives and an AdamW equation.
The tiny encoder/head are audit fixtures, never a native BGE execution.
"""
from __future__ import annotations
import ast,copy,hashlib,json,math,os,pathlib,types,typing
import numpy as np
import torch
R=pathlib.Path('/mnt/data/er_review_input');O=pathlib.Path('/mnt/data/er_review_evidence/results/hand_reference_v2');O.mkdir(exist_ok=False);os.sched_setaffinity(0,{0});torch.set_num_threads(1)
records=[];extracted=[]
def extract(file,names,env):
 source=(R/file).read_text();tree=ast.parse(source)
 nodes=[node for node in tree.body if isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef)) and node.name in names]
 assert {n.name for n in nodes}==set(names)
 for n in nodes:extracted.append(dict(file=file,function=n.name,line_start=n.lineno,line_end=n.end_lineno))
 obj=ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0),*nodes],type_ignores=[]);ast.fix_missing_locations(obj);exec(compile(obj,file,'exec'),env);return env

def check(name,a,b,tol=2e-12):
 x=np.asarray(a,dtype=float);y=np.asarray(b,dtype=float);assert x.shape==y.shape
 err=float(np.abs(x-y).max(initial=0));records.append(dict(name=name,numbers=x.size,max_absolute_error=err,tolerance=tol));assert np.isfinite(x).all() and err<=tol,(name,err,tol)

metric=extract('scripts/step28_continual_population_evaluate.py',['curve_metrics','confusion_metrics','classification','retrieval'],dict(np=np,math=math,KS=(1,3,5,10)))
y=np.array([1,0,1,0]);scores=np.array([4,3,2,1.]);v=metric['curve_metrics'](y,scores)
for k,e in dict(average_precision=5/6,trapezoidal_pr_auc=19/24,roc_auc=3/4,recall_at_fpr_1pct=.5).items():check('curve_untied/'+k,v[k],e)
v=metric['curve_metrics'](y,np.array([2,2,1,1.]));check('curve_tied/AP',v['average_precision'],.5);check('curve_tied/trapezoid',v['trapezoidal_pr_auc'],.625);check('curve_tied/ROC',v['roc_auc'],.5)
a=metric['classification'](y,np.array([0,-1,1,-2.]));assert a['confusion']==dict(tp=2,fp=0,fn=0,tn=2)
rv=metric['retrieval'](np.array([[1,0,0,0,0,1]]),np.array([[6,5,4,3,2,1.]]),4)
check('retrieval_4_sellers',rv[0],[2/3,2/3,.5,1,1,1,.5,.75,.75,.75])
# Deterministically handwritten eight pair-components and four triple-components.
components=[list(range(2*i,2*i+2)) for i in range(8)]+[list(range(16+3*i,19+3*i)) for i in range(4)]
owner={i:c for c,g in enumerate(components) for i in g};pairs=[(i,j) for i in range(28) for j in range(i+1,28)];truth=np.array([owner[i]==owner[j] for i,j in pairs],dtype=float);assert truth.sum()==20
base_loss=extract('scripts/step28_chinese_base.py',['objectives'],{})['objectives']
rankenv=extract('scripts/step28_alias_ranking.py',['hard_candidates','objectives'],dict(base=types.SimpleNamespace(objectives=base_loss)))

def independent_loss_gradient(z):
 """Analytic 378-edge gradient with explicit 28-query normalization."""
 z=np.asarray(z,dtype=float);M=np.zeros((28,28));Y=np.zeros((28,28),dtype=bool);index={}
 for e,(i,j) in enumerate(pairs):M[i,j]=M[j,i]=z[e];Y[i,j]=Y[j,i]=bool(truth[e]);index[i,j]=index[j,i]=e
 sigmoid=lambda x:np.exp(-np.logaddexp(0.,-x))
 bce=float(np.mean(np.logaddexp(0.,z)-truth*z));g=(sigmoid(z)-truth)/378;rank=hard=0.
 for q in range(28):
  candidates=[i for i in range(28) if i!=q];positive=[i for i in candidates if Y[q,i]];negative=sorted([i for i in candidates if not Y[q,i]],key=lambda i:(-M[q,i],i))[:5]
  vals=M[q,candidates];mx=max(vals);soft=np.exp(vals-mx);soft/=soft.sum();rank+=(mx+math.log(np.exp(vals-mx).sum())-float(M[q,positive].mean()))/28
  for i,p in zip(candidates,soft):g[index[q,i]]+=p/28
  for i in positive:g[index[q,i]]-=1/(28*len(positive))
  for pos in positive:
   for neg in negative:
    diff=M[q,neg]-M[q,pos];scale=1/(28*len(positive)*5);hard+=float(np.logaddexp(0.,diff))*scale;grad=.5*float(sigmoid(diff))*scale;g[index[q,neg]]+=grad;g[index[q,pos]]-=grad
 return dict(bce=bce,rank=rank,hard=hard,total=bce+rank+.5*hard),g

z=np.sin(np.arange(378)*.193)+np.arange(378)*.002
zt=torch.tensor(z,dtype=torch.float64,requires_grad=True);terms=rankenv['objectives'](zt,torch.tensor(truth,dtype=torch.float64),.5);terms['total'].backward();exp,g=independent_loss_gradient(z)
for k in exp:check('full_objective/'+k,float(terms[k].detach()),exp[k])
check('analytic_378_gradient',zt.grad.numpy(),g)
for i in (0,13,101,377):
 h=1e-5;plus=z.copy();minus=z.copy();plus[i]+=h;minus[i]-=h;fd=(independent_loss_gradient(plus)[0]['total']-independent_loss_gradient(minus)[0]['total'])/(2*h);check('finite_difference/'+str(i),g[i],fd,1e-9)

# Three-parameter fixture with nonzero Adam moments and step 288, not a restarted optimizer.
class Toy(torch.nn.Module):
 def __init__(self):
  super().__init__();self.encoder=torch.nn.Linear(2,1,bias=False,dtype=torch.float64);self.head=torch.nn.Linear(1,1,bias=False,dtype=torch.float64)
  with torch.no_grad():self.encoder.weight.copy_(torch.tensor([[.3,-.2]],dtype=torch.float64));self.head.weight.fill_(.7)
class Opt(torch.optim.AdamW):
 def __init__(self,model):
  super().__init__([{'params':model.encoder.parameters(),'lr':1e-5,'weight_decay':.01},{'params':model.head.parameters(),'lr':.001,'weight_decay':0.}],betas=(.9,.999),eps=1e-8,foreach=False);self.nsteps=0
  for p,m,v in zip(model.parameters(),([.03,-.02],[.01]),([.2,.3],[.1])):
   self.state[p]={'step':torch.tensor(288.),'exp_avg':torch.tensor(m,dtype=torch.float64).reshape_as(p),'exp_avg_sq':torch.tensor(v,dtype=torch.float64).reshape_as(p)}
 def step(self,*args,**kwargs):self.nsteps+=1;return super().step(*args,**kwargs)

def state_digest(st):return hashlib.sha256(b''.join(v.detach().cpu().numpy().tobytes() for k,v in sorted(st.items()))).hexdigest()
def adam_step(opt):
 steps={int(v['step']) for v in opt.state.values()};assert len(steps)==1;return next(iter(steps))
idx=np.arange(378,dtype=float)
current=types.SimpleNamespace(uid='HAND_CURRENT',labels=truth,features=np.column_stack([20*np.sin(idx*.173),30*np.cos(idx*.131)]),offset=np.linspace(-.1,.1,378))
history=types.SimpleNamespace(uid='HAND_HISTORY',labels=truth,features=np.column_stack([25*np.cos(idx*.157),17*np.sin(idx*.119)]),offset=np.linspace(.3,-.2,378))
calls=[]
def logits(model,group,c,arm,check):
 assert model.training and arm=='split_rank';calls.append(group.uid);return model.head(model.encoder(torch.tensor(group.features,dtype=torch.float64))).reshape(-1)+torch.tensor(group.offset,dtype=torch.float64)
parent=types.SimpleNamespace(adam_step=adam_step,stage_lr=lambda k:1e-5*(k/29 if k<=29 else (288-k)/259),ranking=types.SimpleNamespace(objectives=rankenv['objectives']))
columns=('current_bce','current_rank','current_hard','current_total','history_bce','history_rank','history_hard','history_total','weighted_history_total','total','encoder_lr','head_lr','gradient_norm','history_weight')
env=dict(np=np,parent=parent,base=types.SimpleNamespace(logits=logits),core=types.SimpleNamespace(state_digest=state_digest),STEP_COLUMNS=columns)
newupdate=extract('scripts/step28_er_weight.py',['update'],env)['update']
oldenv=dict(np=np,UPDATED=('seq','er','logit'),adam_step=adam_step,stage_lr=parent.stage_lr,base=types.SimpleNamespace(logits=logits),core=env['core'],ranking=parent.ranking,STEP_COLUMNS=('current_bce','current_rank','current_hard','current_total','history_bce','history_rank','history_hard','history_total','logit_mse','total','encoder_lr','head_lr','gradient_norm','logit_term_host_seconds'))
old_update=extract('scripts/step28_bge_continual.py',['update'],oldenv)['update']
conf=json.loads((R/'schema/step28_chinese_base_policy.json').read_text());optconf={'optimizer':conf['optimizer']}
pol=json.loads((R/'schema/step28_bge_continual_policy.json').read_text())
def seed(*items):b=(json.dumps(list(items),ensure_ascii=False,sort_keys=True,separators=(',',':'))+'\n').encode();return int.from_bytes(hashlib.sha256(b).digest()[:8],'big')%(2**63-1)
stream=seed(pol['schedule_seed'],'ABC',2,'current');cs=seed(stream,0,'dropout');hs=seed(pol['memory_seed'],'ABC',2,0,'history_dropout')
fixture_results=[]
for lam in (1.,.5,.25):
 model=Toy();opt=Opt(model);calls.clear();out=newupdate(model,opt,current,history,optconf,lam,2,1,cs,hs,observe=True)
 assert calls==['HAND_CURRENT','HAND_HISTORY'] and opt.nsteps==1 and adam_step(opt)==289
 theta=np.array([.3,-.2,.7]);parts=[];unweighted=[]
 for gr in (current,history):
  z=(gr.features@theta[:2])*theta[2]+gr.offset;terms,grad=independent_loss_gradient(z);unweighted.append(terms);parts.append(np.r_[gr.features.T@(grad*theta[2]),np.sum(grad*(gr.features@theta[:2]))])
 combined=parts[0]+lam*parts[1];norm=np.linalg.norm(combined);clipped=combined*min(1.,1/(norm+1e-6));m=.9*np.array([.03,-.02,.01])+.1*clipped;v=.999*np.array([.2,.3,.1])+.001*clipped**2;rates=np.array([1e-5/29,1e-5/29,.001]);wd=np.array([.01,.01,0.]);expected=theta*(1-rates*wd)-rates*(m/(1-.9**289))/(np.sqrt(v/(1-.999**289))+1e-8)
 actual=np.concatenate([p.detach().numpy().ravel() for p in model.parameters()]);grad_actual=np.concatenate([p.grad.numpy().ravel() for p in model.parameters()]);check(f'weighted_update/{lam}/gradient',grad_actual,clipped,1e-8);check(f'weighted_update/{lam}/parameters',actual,expected,2e-12);check(f'weighted_update/{lam}/gradient_norm',out['gradient_norm'],norm,2e-7);check(f'weighted_update/{lam}/total',out['total'],unweighted[0]['total']+lam*unweighted[1]['total'],3e-6)
 for p,expected_m,expected_v in zip(model.parameters(),(m[:2].reshape(1,2),m[2:].reshape(1,1)),(v[:2].reshape(1,2),v[2:].reshape(1,1))):check(f'weighted_update/{lam}/moment',opt.state[p]['exp_avg'].numpy(),expected_m,1e-8);check(f'weighted_update/{lam}/variance',opt.state[p]['exp_avg_sq'].numpy(),expected_v,1e-8)
 fixture_results.append(dict(lambda_value=lam,combined_norm=norm,clipped=bool(norm>1),parameter_values=actual.tolist(),adam_step=289,optimizer_calls=1))
 if lam==1.:
  old_model=Toy();old_opt=Opt(old_model);calls.clear();old_out=old_update(old_model,old_opt,current,history,None,optconf,'er',2,1,cs,hs,observe=True)
  check('lambda1_exact_old_ER_parameters',actual,np.concatenate([p.detach().numpy().ravel() for p in old_model.parameters()]),0.)
  check('lambda1_exact_old_ER_gradient',grad_actual,np.concatenate([p.grad.numpy().ravel() for p in old_model.parameters()]),0.)
summary={'status':'PASS_HAND_METRICS_LOSS_AND_CONTINUOUS_ADAM_REFERENCE','torch':torch.__version__,'numpy':np.__version__,'cpu_affinity':sorted(os.sched_getaffinity(0)),'checks':records,'extracted_function_bodies':extracted,'synthetic_fixtures':fixture_results,'scope':'Synthetic labels and three-parameter toy only; actual function bodies extracted without imports. No formal samples/labels/weights loaded. These are reference arithmetic checks, not new scientific training.'}
(O/'hand_reference.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2))
