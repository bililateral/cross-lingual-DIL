"""Independent bounded audit. All data and all weights made here are handmade.
No execute/train/native BGE entry; no installation, official assets or label recovery.
The submitted fixture builder is reused only for gate file layouts (explicit nonweights),
not for independent mathematical references. Project output functions are the objects tested.
"""
from __future__ import annotations
import argparse,copy,csv,hashlib,io,itertools,json,math,os,random,sys,time,traceback,unittest
from pathlib import Path
from unittest import mock
import numpy as np
import torch

ROOT=Path('/mnt/data/bge_continual_external_audit')
SRC=ROOT/'source'
sys.path.insert(0,str(SRC/'scripts'));sys.path.insert(0,str(SRC/'tests'))
import step28_bge_continual as m
import step28_bge_continual_run as r
import step28_bge_continual_evaluate as e
import test_step28_bge_continual_contracts as submitted

parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,required=True);args=parser.parse_args()
OUT=args.out
OUT.mkdir(parents=True,exist_ok=False)
torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
RESULT={}
PAIRS=list(itertools.combinations(range(28),2))
OWNERS=sum(([i]*n for i,n in enumerate([3]*4+[2]*8)),[])
Y=np.array([int(OWNERS[a]==OWNERS[b]) for a,b in PAIRS],dtype=np.uint8)
COLUMNS=list(m.metrics.COLUMNS)

def dump(path,obj):
 path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def sha(b):return hashlib.sha256(b).hexdigest()
def canon(x):return (json.dumps(x,ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n').encode()
def seed_for(seed,*parts):return int.from_bytes(hashlib.sha256(canon([seed,*parts])).digest()[:8],'big')%(2**63-1)
def handmade(uid,shift=0,labels=True):
 sellers=tuple(f'{uid}_account_{i:02}' for i in range(28))
 items=tuple(tuple((f'{uid}_item_{i:02}_{j}',f'人工商品 {i%5} 型号 {j}',f'手写示例：条件 {shift} 样式 {OWNERS[i]} 记录 {i} {j}') for j in range(2+i%3)) for i in range(28))
 g=m.data.Group(uid,sellers,items,tuple(int(y) for y in Y) if labels else None);g.validate();return g

def scalar_objective(z,y=Y,hard_threshold=None):
 """Independent Python/math evaluation and analytic dL/d(edge logit)."""
 z=[float(v) for v in z];y=[int(v) for v in y]
 def sp(v):return max(v,0.)+math.log1p(math.exp(-abs(v)))
 def sig(v):return 1/(1+math.exp(-v)) if v>=0 else math.exp(v)/(1+math.exp(v))
 vals={'bce':sum(sp(v)-t*v for v,t in zip(z,y))/378.,'rank':0.,'hard':0.}
 grad={'bce':np.array([(sig(v)-t)/378 for v,t in zip(z,y)],float),'rank':np.zeros(378),'hard':np.zeros(378)}
 for q in range(28):
  ix=[(k,b if a==q else a) for k,(a,b) in enumerate(PAIRS) if q in (a,b)]
  pos=[k for k,o in ix if y[k]];neg=sorted([(k,o) for k,o in ix if not y[k]],key=lambda v:(-z[v[0]],v[1]))[:5]
  maximum=max(z[k] for k,o in ix);ex=[math.exp(z[k]-maximum) for k,o in ix];den=sum(ex)
  vals['rank']+=(maximum+math.log(den)-sum(z[k] for k in pos)/len(pos))/28
  for (k,o),exv in zip(ix,ex):grad['rank'][k]+=exv/den/28
  for k in pos:grad['rank'][k]-=1/(28*len(pos))
  for p in pos:
   for n,o in neg:
    scale=1/(28*len(pos)*5);delta=z[n]-z[p]
    linear=hard_threshold is not None and delta>hard_threshold
    vals['hard']+=(delta if linear else sp(delta))*scale
    d=(1. if linear else sig(delta))*scale;grad['hard'][n]+=d;grad['hard'][p]-=d
 vals['total']=vals['bce']+vals['rank']+.5*vals['hard'];grad['total']=grad['bce']+grad['rank']+.5*grad['hard']
 return vals,grad

def torch_reference(z):
 """Independent edge-list differentiable objective, never project loss functions."""
 loss=torch.stack([torch.logaddexp(z.new_zeros(()),v)-int(y)*v for v,y in zip(z,Y)]).mean()
 qs=[];hs=[]
 for q in range(28):
  ix=[(k,b if a==q else a) for k,(a,b) in enumerate(PAIRS) if q in (a,b)]
  pos=[k for k,o in ix if Y[k]];neg=sorted([(k,o) for k,o in ix if not Y[k]],key=lambda v:(-float(z[v[0]].detach()),v[1]))[:5]
  qs.append(torch.logsumexp(torch.stack([z[k] for k,o in ix]),0)-torch.stack([z[k] for k in pos]).mean())
  hs.append(torch.stack([torch.logaddexp(z.new_zeros(()),z[n]-z[p]) for p in pos for n,o in neg]).mean())
 return loss+torch.stack(qs).mean()+.5*torch.stack(hs).mean()

def mixed_target_rounding_correction(z):
 """Value only: actual runner makes float32 labels even for this double toy.
 Exact Bernoulli derivative remains double (measured in diagnosis), whereas
 BCE values have two float32 stores before reduction. Do not fake gradients.
 """
 # Two explicit float32 stores are necessary for mixed-target BCE values.
 # The first rounds (1-y)*z; the second rounds its sum with softplus(-z).
 # Measured per-element equality in diagnose_reference_v2, not a fitted offset.
 y=torch.tensor(Y,dtype=z.dtype)
 exact=torch.logaddexp(torch.zeros_like(z),z)-y*z
 first_store=((1-y)*z).float().double()
 second_store=(first_store+torch.logaddexp(torch.zeros_like(z),-z)).float()
 return float(second_store.detach().mean())-float(exact.detach().mean())

class IndependentHandmadeBudget:
 """No formal budget assertion; fake monitoring interface for actual tiny path."""
 def check(self,*args,**kwargs): pass
 def state(self): return {'independent_scope':'handmade CPU; not formal resource measurement'}

class SmallScore(torch.nn.Module):
 def __init__(self):
  super().__init__();torch.manual_seed(3401)
  self.encoder=torch.nn.Sequential(torch.nn.Linear(4,7),torch.nn.Tanh(),torch.nn.Dropout(.15))
  self.head=torch.nn.Linear(7,1)
 def forward(self,group):
  s=sum(group.uid.encode())%101
  a=np.arange(378*4,dtype=np.float64).reshape(378,4)
  x=torch.tensor(np.sin(a*.11+s*.007),dtype=next(self.parameters()).dtype)
  return self.head(self.encoder(x)).flatten()

def independent_rate(step):return float(1e-5*(step/29 if step<30 else (288-step)/259))

def quantile_linear(arr):
 arr=np.sort(np.asarray(arr),axis=0)
 vals=[]
 for q in (.025,.975):
  f=(len(arr)-1)*q;i=int(math.floor(f));w=f-i
  vals.append((1-w)*arr[i]+w*arr[min(i+1,len(arr)-1)])
 return np.stack(vals)

def matrices():
 rng=np.random.Generator(np.random.PCG64(9152026));domains=list('ABC'*20)
 rng.shuffle(domains);arrays={}
 for i,name in enumerate(e.expected_points()):
  raw=.2+.03*rng.normal(size=(60,22))+.001*i
  arrays[name]={'raw':raw.copy()}
  for role,offset in (('stage-cal',.012),('first-cal',-.008)):
   val=raw.copy();val[:,4:12]+=offset+rng.normal(0,.002,(60,8));arrays[name][role]=val
 return arrays,domains

def manual_point(order,arm,t):return order+'_shared' if t==1 or arm=='frozen' else f'{order}_{arm}_stage{t}'
def manual_endpoints(arrays,domains,arm,role,indices=None):
 """First build R(t,arrival) by actual-domain means, then explicit formulas."""
 answers={k:[] for k in ('O','N','Z','F_first','F','G','final_all')}
 for order in ('ABC','BCA','CAB'):
  table={}
  for stage in (1,2,3):
   key=manual_point(order,arm,stage)
   values=arrays[key][role] if role!='primary' else arrays[key]['stage-cal'].copy()
   if role=='primary':
    rank=[COLUMNS.index(k) for k in [*m.metrics.CURVE_KEYS,*m.metrics.RETRIEVAL_KEYS]]
    values[:,rank]=arrays[key]['raw'][:,rank]
   for arrival,d in enumerate(order,1):
    rows=np.array([i for i,x in enumerate(domains) if x==d]); loc=rows if indices is None else rows[indices['ABC'.index(d)]]
    table[stage,arrival]=values[loc].mean(0)
  aa=table
  answers['O'].append((aa[3,1]+aa[3,2])/2)
  answers['N'].append((aa[2,2]+aa[3,3])/2)
  answers['Z'].append(aa[3,3])
  answers['F_first'].append(aa[1,1]-aa[3,1])
  answers['F'].append((aa[1,1]-aa[3,1]+aa[2,2]-aa[3,2])/2)
  answers['G'].append((aa[2,2]-aa[1,2]+aa[3,3]-aa[2,3])/2)
  answers['final_all'].append((aa[3,1]+aa[3,2]+aa[3,3])/3)
 result={k:np.stack(v) for k,v in answers.items()}
 for k in ('F_first','F','G'):result[k][:,[COLUMNS.index('brier'),COLUMNS.index('log_loss')]]*=-1
 return result

class IndependentAudit(unittest.TestCase):
 def test_01_scalar_analytic_gradient(self):
  rng=np.random.Generator(np.random.PCG64(951));records=[];maxloss=maxgrad=fdmax=0.
  for name,z in [('continuous',rng.normal(size=378)*2),('ties',np.round(rng.normal(size=378),1)),('extreme',np.linspace(-100,100,378))]:
   exact_vals,exact_grad=scalar_objective(z)
   vals,grad=scalar_objective(z,hard_threshold=20.);t=torch.tensor(z,dtype=torch.float64,requires_grad=True);got=m.ranking.objectives(t,torch.tensor(Y,dtype=torch.float64),.5)
   row={'case':name,'exact_mathematical_vs_runtime_softplus':{k:{'value_difference':vals[k]-exact_vals[k],'gradient_max_difference':float(np.max(abs(grad[k]-exact_grad[k])))} for k in vals},'runtime_threshold_explicit':20}
   for k in vals:
    g=torch.autograd.grad(got[k],t,retain_graph=True)[0].numpy();ld=abs(float(got[k].detach())-vals[k]);gd=float(np.max(abs(g-grad[k])));maxloss=max(maxloss,ld);maxgrad=max(maxgrad,gd)
    self.assertLessEqual(ld,1e-12);self.assertLessEqual(gd,1e-12);row[k]={'value':vals[k],'value_error':ld,'gradient_error':gd}
   if name=='continuous':
    for j in range(0,378,19):
     plus=z.copy();minus=z.copy();plus[j]+=1e-5;minus[j]-=1e-5
     fd=(scalar_objective(plus)[0]['total']-scalar_objective(minus)[0]['total'])/2e-5
     fdmax=max(fdmax,abs(fd-grad['total'][j]));self.assertLess(abs(fd-grad['total'][j]),1e-8)
   records.append(row)
  RESULT['loss_gradient']={'cases':records,'max_loss_error':maxloss,'max_gradient_error':maxgrad,'max_finite_difference_error':fdmax,'finite_difference_points':20}

 def test_02_independent_live_three_targets(self):
  c=m.config(m.contract()); current,old=handmade('new_B'),handmade('old_A');base=SmallScore().double();opt=m.core.make_optimizer(base,c)
  # Warm update is real; counter rebasing here is explicit and separate from test_04.
  with mock.patch.object(m.base,'logits',side_effect=lambda model,g,*a:model(g)):
   m.update(base,opt,old,None,None,c,'seq',1,1,510,511)
  for state in opt.state.values():state['step'].fill_(288)
  target=np.linspace(-.7,.9,378,dtype=np.float32); rows=[];captures=[]
  for arm in ('seq','er','logit'):
   model=copy.deepcopy(base); ref=copy.deepcopy(base); optimizer=m.core.make_optimizer(model,c); other=m.core.make_optimizer(ref,c)
   optimizer.load_state_dict(copy.deepcopy(opt.state_dict()));other.load_state_dict(copy.deepcopy(opt.state_dict()))
   seen=[]
   def logits(md,g,*args):
    v=md(g);seen.append({'group':g.uid,'training':md.training,'values':v.detach().numpy().copy()});return v
   origclip=torch.nn.utils.clip_grad_norm_
   with mock.patch.object(m.base,'logits',side_effect=logits),mock.patch.object(torch.nn.utils,'clip_grad_norm_',wraps=origclip) as cl,mock.patch.object(optimizer,'step',wraps=optimizer.step) as st:
    got=m.update(model,optimizer,current,None if arm=='seq' else old,target if arm=='logit' else None,c,arm,2,1,520,521,observe=True)
    self.assertEqual(cl.call_count,1);self.assertEqual(st.call_count,1)
   ref.train();other.zero_grad(set_to_none=True);torch.manual_seed(520);pc=ref(current);loss=torch_reference(pc);value_correction=mixed_target_rounding_correction(pc)
   if arm!='seq':
    torch.manual_seed(521);ph=ref(old);loss=loss+torch_reference(ph);value_correction+=mixed_target_rounding_correction(ph)
    if arm=='logit':loss=loss+.5*torch.stack([(ph[i]-float(target[i]))**2 for i in range(378)]).mean()
   loss.backward();refnorm=origclip(ref.parameters(),1.,error_if_nonfinite=True)
   for g,lr in zip(other.param_groups,(independent_rate(1),.001)):g['lr']=lr
   other.step();err=max(float((a-b).abs().max().detach()) for a,b in zip(model.parameters(),ref.parameters()))
   expected_logged=float(loss.detach())+value_correction
   self.assertLess(err,1e-10);self.assertLessEqual(abs(got['total']-expected_logged),1e-12)
   self.assertEqual(len(seen),1 if arm=='seq' else 2);self.assertTrue(all(v['training'] for v in seen));captures.append(seen)
   rows.append({'method':arm,'parameter_error':err,'total_error':abs(got['total']-expected_logged),'mathematical_double_total':float(loss.detach()),'float32_target_value_correction':value_correction,'clip_count':1,'step_count':1,'forward_count':len(seen),'optimizer_step':m.adam_step(optimizer),'gradient_norm_error':abs(got['gradient_norm']-float(refnorm))})
  for cap in captures[1:]:self.assertTrue(np.array_equal(cap[0]['values'],captures[0][0]['values']))
  self.assertTrue(np.array_equal(captures[1][1]['values'],captures[2][1]['values']))
  RESULT['live_targets']={'rows':rows,'shared_actual_warm_updates':1,'warm_counter_fixture':'explicit 1 -> 288; not 288 actual calls in this test','current_forward_identical':True,'history_forward_identical':True,'reference_uses_no_project_objective':True}

 def test_03_reservoir_independent_stream_and_budget(self):
  rows=[]
  for order in ('ABC','BCA','CAB'):
   seq=[handmade(f'{d}_{i:02}') for d in order[:2] for i in range(48)]
   rr=random.Random(seed_for(20260930,order,'retention'));expected=[];seen=0; snapshots={}
   for g in seq:
    seen+=1
    if seen<=6:expected.append(g.uid)
    else:
     index=rr.randrange(seen)
     if index<6:expected[index]=g.uid
    if seen in (48,96):snapshots[seen]=expected.copy()
   members=[]
   for logits in (False,True):
    mem=m.Memory(order,20260930,logits,{'a':.8,'b':.1});mem.auxiliary={'handmade_rng':r.rng_state(),'stage':1,'notes':'all serialised auxiliary included'}
    callback=[]
    def score(gs):
     callback.extend(g.uid for g in gs)
     return np.stack([np.full(378,1 if g.uid.startswith(order[0]) else 2,dtype=np.float32) for g in gs])
    mem.retain(seq[:48],1,score if logits else None);self.assertEqual(mem.summary()['members'],snapshots[48]);oldref={k:v.copy() for k,v in mem.references.items()}
    first_bytes=mem.to_bytes(); self.assertEqual(len(first_bytes),mem.summary()['serialized_bytes'])
    mem.begin_stage(2);indrng=random.Random(seed_for(20260930,order,2,'history_draws'))
    expected_draw=[snapshots[48][indrng.randrange(6)] for _ in range(288)]
    got=[mem.draw()[0].uid for _ in range(93)]; restored=m.Memory.from_bytes(mem.to_bytes()); got.extend(restored.draw()[0].uid for _ in range(195));self.assertEqual(got,expected_draw)
    restored.retain(seq[48:],2,score if logits else None);self.assertEqual(restored.summary()['members'],snapshots[96]);members.append(restored.summary()['members'])
    if logits:
     for uid,ref in restored.references.items():np.testing.assert_array_equal(ref,oldref[uid] if uid in oldref else np.full(378,2,dtype=np.float32))
     self.assertEqual(len(callback),6+len([uid for uid in members[-1] if uid not in oldref]))
    payload=restored.to_bytes();(OUT/f'memory_{order}_{logits}.json').write_bytes(payload)
    original=restored.to_bytes();restored.auxiliary['oversize']='x'*1048576
    with self.assertRaises(ValueError):restored.to_bytes()
    rows.append({'order':order,'logit':logits,'bytes_after48':len(first_bytes),'bytes_after96':len(payload),'reference_numeric_bytes':9072 if logits else 0,'callback_groups':callback,'restored_partial_draw_sequence':True})
   self.assertEqual(members[0],members[1])
  RESULT['memory']={'rows':rows,'metadata_overflow_refused':True,'no_archived_data_callback':True,'same_members_and_draws':True}

 def test_04_full_micro_stage_checkpoint_and_branch(self):
  c=m.config(m.contract());c['input']['encoder_bf16']=False; p=m.contract(); root=OUT/'actual_micro';root.mkdir()
  for f in ('branches','work','models','scores','maps','points','updates'): (root/f).mkdir()
  model=submitted.tiny_model();optimizer=m.core.make_optimizer(model,c);budget=IndependentHandmadeBudget()
  current=[handmade(f'fit_A_{i:02}',i%3) for i in range(48)]
  calls=[];original=m.update
  def wrapped(*a,**kw):
   ans=original(*a,**kw);calls.append((ans['stage'],ans['step'],ans['adam_step']));return ans
  with mock.patch.object(m,'update',side_effect=wrapped):training=r.train_stage(model,optimizer,current,None,c,p,'ABC',1,'seq',root,budget)
  self.assertEqual(len(calls),288);self.assertEqual(calls[-1],(1,288,288)); self.assertFalse(training['observations']['288']['encoder']['parameters_changed'])
  cal=[handmade(f'cal_A_{i:02}',i%4) for i in range(12)];valid=[handmade(f'blind_{i:02}',i%5,labels=False) for i in range(60)]
  before=r.rng_state();beforeopt=m.core.state_digest(optimizer.state_dict());point=r.checkpoint(root,'ABC_shared',model,optimizer,c,'ABC',1,cal,valid,None,budget)
  self.assertEqual(before,r.rng_state());self.assertEqual(beforeopt,m.core.state_digest(optimizer.state_dict()));self.assertTrue((root/point['full_checkpoint']['path']).exists())
  with mock.patch.object(m.base,'load_model',side_effect=lambda *a:submitted.tiny_model()):clone,other=r.restore_branch(root,point,c)
  self.assertEqual(m.core.state_digest(model.state_dict()),m.core.state_digest(clone.state_dict()))
  self.assertEqual(m.core.state_digest(optimizer.state_dict()),m.core.state_digest(other.state_dict()))
  new=handmade('fit_B_00',2)
  for md,op in ((model,optimizer),(clone,other)):m.update(md,op,new,None,None,c,'seq',2,1,119,120)
  self.assertEqual(m.core.state_digest(model.state_dict()),m.core.state_digest(clone.state_dict()));self.assertEqual(m.core.state_digest(optimizer.state_dict()),m.core.state_digest(other.state_dict()))
  RESULT['actual_micro']={'real_shared_stage_calls':288,'real_next_steps_on_original_and_restored':2,'full_Adam_before_after_checkpoint_equal':True,'RNG_restored':True,'next_model_and_Adam_exactly_equal':True,'score_roles':list(point['scores']),'calibration_status':m.data.read_json(root/point['mapping']['path'])['status'],'native_BGE':False,'source':'submitted TinyEncoder sample reused; actual project train_stage/checkpoint/restore_branch called'}

 def test_05_all_endpoint_formulas_and_bootstrap(self):
  arrays,domains=matrices();draws=e.bootstrap_draws(); rng=np.random.Generator(np.random.PCG64(20260930)); independent_draws=np.stack([rng.integers(0,20,(3,20)) for _ in range(5000)])
  np.testing.assert_array_equal(draws,independent_draws);np.save(OUT/'bootstrap_draws.npy',draws)
  counts=np.stack([[np.bincount(row,minlength=20) for row in ds] for ds in independent_draws])
  maxpoint=maxinterval=maxorder=0.; checked=0; results={}
  for arm in ('frozen','seq','er','logit'):
   for role in ('raw','stage-cal','first-cal','primary'):
    expected=manual_endpoints(arrays,domains,arm,role)
    # Finite linear functional basis from independently computed actual-domain R table.
    group_values={name:np.zeros((3,20,22)) for name in e.ENDPOINTS}
    for d in range(3):
     for g in range(20):
      select=np.tile(np.arange(20),(3,1));select[d]=g
      sel=manual_endpoints(arrays,domains,arm,role,select)
      for end in e.ENDPOINTS:
       # replacement difference plus equal allocation of the global point gives a
       # bootstrap representation with same total (other domains held at mean).
       group_values[end][d,g]=(sel[end].mean(0)-expected[end].mean(0))+expected[end].mean(0)/3
    for end in e.ENDPOINTS:
     field=e.endpoint_fields(arrays,domains,arm,role,end);got=e.summarize_field(field,draws)
     boot=np.einsum('bdg,dgk->bk',counts,group_values[end],optimize=False)/20
     interval=quantile_linear(boot); point=expected[end].mean(0)
     for k,name in enumerate(COLUMNS):
      pe=abs(point[k]-got[name]['mean']);oe=max(abs(expected[end][i,k]-got[name]['per_order'][o]) for i,o in enumerate(('ABC','BCA','CAB')));ie=float(np.max(abs(interval[:,k]-got[name]['conditional_95pct_interval'])))
      maxpoint=max(maxpoint,pe);maxorder=max(maxorder,oe);maxinterval=max(maxinterval,ie);self.assertLess(max(pe,oe,ie),1e-12);checked+=1
     if arm=='frozen' and end in ('F','F_first','G'):self.assertLess(np.max(abs(point)),1e-14)
     if arm=='seq' and role=='primary':results[end]=got
  RESULT['endpoints']={'arm_role_endpoint_metric_cases':checked,'max_point_error':maxpoint,'max_per_order_error':maxorder,'max_interval_error':maxinterval,'independent_draws_identical':True,'all_formula_branches_checked':True,'sequential_primary':results,'bootstrap_sha256':sha((OUT/'bootstrap_draws.npy').read_bytes()),'domains':domains}

 def test_06_twenty_three_single_condition_failures(self):
  def record(value):return {'mean':value,'conditional_95pct_interval':[.001,.04],'per_order':dict.fromkeys(('ABC','BCA','CAB'),.02)}
  template={end:{name:record(-.02 if name in ('brier','log_loss') else .02) for name in COLUMNS} for end in e.ENDPOINTS}
  raw=copy.deepcopy(template);good=e.comparison_checks(template,raw);self.assertEqual(len(good['checks']),23);self.assertTrue(good['pilot_observed_checks_pass'])
  cases=[]
  # Input field and value for each independent intended failure.
  mutations=[('old_map_improves','O','map','mean',0.,False),('old_map_interval_above_zero','O','map','conditional_95pct_interval',[0.,.01],False),('old_recall5_improves','O','recall_at_5','mean',0.,False),('new_map_non_decrease','N','map','mean',-1e-30,False),('new_recall5_non_decrease','N','recall_at_5','mean',-1e-30,False)]
  for end in ('O','N'):
   for name in ('average_precision','roc_auc','brier','log_loss'):mutations.append((f'{end}_{name}_non_degradation',end,name,'mean',1e-30 if name in ('brier','log_loss') else -1e-30,False))
   for name in ('brier','log_loss'):mutations.append((f'{end}_{name}_against_raw_reference',end,name,'mean',1e-30,True))
  for name in ('map','recall_at_5','average_precision','roc_auc','brier','log_loss'):mutations.append((f'Z_{name}_non_degradation','Z',name,'mean',1e-30 if name in ('brier','log_loss') else -1e-30,False))
  for key,end,name,field,value,toraw in mutations:
   d=copy.deepcopy(template);b=copy.deepcopy(raw);(b if toraw else d)[end][name][field]=value
   got=e.comparison_checks(d,b);self.assertEqual(got['failed'],[key]);cases.append({'key':key,'observed_failed':got['failed']})
  neutral=copy.deepcopy(template);nraw=copy.deepcopy(raw)
  for end in ('O','N','Z'):
   for name in ('map','recall_at_5','average_precision','roc_auc','brier','log_loss'):
    if not(end=='O' and name in ('map','recall_at_5')):neutral[end][name]['mean']=0.
   for name in ('brier','log_loss'):nraw[end][name]['mean']=0.
  self.assertTrue(e.comparison_checks(neutral,nraw)['pilot_observed_checks_pass'])
  self.assertEqual(good['future_three_seed_qualification'],'NOT_EVALUATED_SINGLE_SEED_PILOT')
  RESULT['acceptance']={'number':23,'individual_failures':cases,'guard_equalities_accepted':True,'performance_tolerance':0,'future_three_seed_not_evaluated':True}

 def test_07_gate_file_binding_and_draw_failures(self):
  root=OUT/'gate';root.mkdir();manifest,p=submitted.gate_fixture(root);baseline=r.blind_gate(root,manifest,p);self.assertEqual(len(baseline['points']),21)
  refused=[]
  files=[root/manifest['initial']['scores']['path']]
  for name in e.expected_points():
   point=m.data.read_json(root/manifest['points'][name]['path']);files.append(root/point['model']['path']);files.extend(root/x['path'] for x in point['scores'].values())
   if point['stage']==1:files.append(root/point['full_checkpoint']['path'])
  # First/last point plus each source class; full unique weight files covered.
  selected=[p for p in files if p.suffix=='.pt']+[files[0]]
  for name in ('ABC_shared','CAB_logit_stage3'):
   pt=m.data.read_json(root/manifest['points'][name]['path']); selected.extend(root/q['path'] for q in pt['scores'].values())
  for file in selected:
   original=file.read_bytes();bad=bytearray(original);bad[-1]^=1;file.write_bytes(bad)
   try:
    with self.assertRaises((ValueError,FileNotFoundError)):r.blind_gate(root,manifest,p)
    refused.append(str(file.relative_to(root)))
   finally:file.write_bytes(original)
  RESULT['gate']={'explicit_pseudo_weight_files':True,'not_native_weights':True,'complete_gate_baseline_points':21,'independently_mutated_files_refused':len(refused),'files':refused,'valid_loader_called':False}

 def test_08_collection_and_multi_failure_recovery(self):
  root=OUT/'collect'; groups=[handmade(f'v_{i:02}',labels=True) for i in range(60)];domains=list('BAC'*20); part={'development':[{'group_uid':g.uid,'domain':d} for g,d in zip(groups,domains)]}
  rng=np.random.Generator(np.random.PCG64(42));raw=rng.normal(-1,.4,size=(60,378)).astype(np.float32)
  scores={'initial':raw,'points':{}}
  for j,name in enumerate(e.expected_points()):
   val=(raw+j*.013+rng.normal(0,.03,(60,378))).astype(np.float32)
   scores['points'][name]={'raw':val,'stage-cal':.75*val.astype(np.float64)+.11,'first-cal':.9*val.astype(np.float64)-.03}
  r.collect(root,scores,groups,part,r.sources());self.assertEqual(len(list(root.glob('*.npy'))),64)
  immutable={p.name:sha(p.read_bytes()) for p in root.iterdir() if p.is_file()}
  faults=[]
  # Fault before and after first outputs, without changing any collected data.
  for field in ('bootstrap_draws','stage_table','comparison_checks','forgetting_checks'):
   with mock.patch.object(e,field,side_effect=RuntimeError('independent injected '+field)):
    with self.assertRaisesRegex(RuntimeError,'independent injected'):e.finalize(root)
   for name,digest in immutable.items():self.assertEqual(sha((root/name).read_bytes()),digest)
   faults.append(field)
  with mock.patch.object(r,'parse_once',side_effect=AssertionError('NO LABEL READ')),mock.patch.object(m.base,'load_model',side_effect=AssertionError('NO MODEL')),mock.patch.object(m.calibration,'fit',side_effect=AssertionError('NO FIT')):
   result=e.finalize(root);first=(root/'evaluation.json').read_bytes();again=e.finalize(root);self.assertEqual((root/'evaluation.json').read_bytes(),first)
  # A finished result may not be silently replaced by a different answer.
  (root/'evaluation.json').write_bytes(first+b' ')
  with self.assertRaises(FileExistsError):e.finalize(root)
  (root/'evaluation.json').write_bytes(first)
  partial=OUT/'partial_collect';original_group=m.metrics.group_metrics;calls=0
  def fail_second(*a,**kw):
   nonlocal calls;calls+=1
   if calls==2:raise RuntimeError('independent acquisition fault')
   return original_group(*a,**kw)
  with mock.patch.object(m.metrics,'group_metrics',side_effect=fail_second):
   with self.assertRaisesRegex(RuntimeError,'acquisition'):r.collect(partial,scores,groups,part,r.sources())
  self.assertEqual(len(list(partial.glob('*.npy'))),1);self.assertFalse((partial/'collected.json').exists())
  with self.assertRaises(FileNotFoundError):e.finalize(partial)
  # Counts -> rates independent checks; no scores-to-truth reverse inference.
  countchecks=0;maxerr=0.
  for pt,roles in result['absolute_stage_results'].items():
   for role,val in roles.items():
    info=m.data.read_json(root/'collected.json')['points'][pt][role];counts=m.data.read_json(root/info['counts']['path'])
    for dom,inds in [('pooled',range(60))]+[(d,[i for i,x in enumerate(domains) if x==d]) for d in 'ABC']:
     tp,fp,fn,tn=(sum(counts[i][k] for i in inds) for k in ('tp','fp','fn','tn'))
     expected={'tp':tp,'fp':fp,'fn':fn,'tn':tn,'precision':tp/(tp+fp) if tp+fp else 0.,'recall':tp/(tp+fn),'fpr':fp/(fp+tn),'f1':2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.}
     out=val['pooled_fixed_half_classification']['pooled' if dom=='pooled' else 'by_domain'];out=out if dom=='pooled' else out[dom]
     for k,v in expected.items():err=abs(v-out[k]);maxerr=max(maxerr,err);self.assertLessEqual(err,1e-12);countchecks+=1
  RESULT['recovery']={'matrix_count':64,'count_file_count':64,'injected_statistics_faults':faults,'all_collected_files_preserved':True,'finalize_twice_same_bytes':True,'refuses_different_existing_result':True,'partial_collection_matrix_count':1,'partial_not_claimed_complete':True,'count_rate_checks':countchecks,'max_rate_error':maxerr,'result_status':result['status'],'group_labels_handmade_only':True}

 def test_09_calibration_scalar_optimum_and_current_only(self):
  # Reproduce known interior optimum on handmade equally repeated score bins,
  # independent closed-form optimum in (a,b), never formal stage mapping.
  z=np.array([-1.]*200+[1.]*200,float);y=np.array([1]*20+[0]*180+[1]*80+[0]*120,dtype=np.uint8)
  optimum=np.array([(math.log(.4/.6)-math.log(.1/.9))/2,(math.log(.4/.6)+math.log(.1/.9))/2])
  val,grad=m.calibration.loss_gradient(optimum,z,y)
  t=optimum[0]*z+optimum[1];prob=np.exp(-np.logaddexp(0.,-t));ref=float(np.mean(np.maximum(t,0)+np.log1p(np.exp(-abs(t)))-y*t));g=np.array([np.dot(prob-y,z),sum(prob-y)])/len(y)
  self.assertLess(abs(val-ref),1e-12);np.testing.assert_allclose(grad,g,atol=1e-12,rtol=0)
  # Full-shape natural supervised handmade groups, no addition of past-domain labels.
  score=np.tile(np.linspace(-2,2,378),(12,1)).astype(np.float32);labels=np.tile(Y,(12,1))
  fitted=m.calibration.fit(score,labels,role='calibration');self.assertEqual(fitted['group_count'],12);self.assertEqual(fitted['positive_count'],240);self.assertEqual(fitted['pair_count'],4536);self.assertEqual(fitted['status'],'PASS_CALIBRATION_FIT')
  transformed=m.calibration.transform(score,fitted);m.calibration.preserve_order(score,transformed)
  # Current-map replacement may hurt old probability without any model/rank change.
  prior=20/378; pp=np.array([prior-.04,prior+.04]); pn=np.array([prior-.01,prior+.01]); br=float(np.mean((pn-pp)**2));loss=float(np.mean(pp*np.log(pp/pn)+(1-pp)*np.log((1-pp)/(1-pn))))
  RESULT['calibration']={'handmade_analytic_gradient_error':float(np.max(abs(g-grad))),'known_optimum':optimum.tolist(),'known_optimum_gradient':np.asarray(grad).tolist(),'fullshape_handmade_fit':fitted,'fixed_model_mapping_change_counterexample':{'both_domain_positive_rate':prior,'old_expected_brier_increase':br,'old_expected_log_loss_increase':loss,'unchanged_logits_and_ranking':True}}

 def test_10_access_and_visibility_scope(self):
  groups={};part={}
  for role,n in (('fit',48),('calibration',12)):
   groups[role]=[handmade(f'{role}_{d}_{i:02}') for d in 'ABC' for i in range(n)]
   part[role]=[{'group_uid':g.uid,'domain':g.uid.split('_')[1]} for g in groups[role]]
  supply=m.Supply(groups,part);received=[]
  for order in ('ABC','BCA','CAB'):
   first=order+'_shared';curr,cal=supply.current(first,order,1)
   self.assertTrue(all(g.uid.split('_')[1]==order[0] for g in curr+cal));received.append([first,1,order[0]])
   for arm in ('seq','er','logit'):
    path=order+'_'+arm;supply.branch_after_first(path,first)
    for stage in (2,3):
     curr,cal=supply.current(path,order,stage);self.assertTrue(all(g.uid.split('_')[1]==order[stage-1] for g in curr+cal));received.append([path,stage,order[stage-1]])
    with self.assertRaises(ValueError):supply.current(path,order,2)
  out=OUT/'access';out.mkdir();m.data.write_json(out/'access.json',dict(train=0,valid=0,heldout=0,owners=0));read=[]
  def fail(*a):
   read.append(m.data.read_json(out/'access.json'));raise RuntimeError('handmade parser failure')
  with self.assertRaises(RuntimeError):r.parse_once(out,[],{},'development',fail)
  with self.assertRaises(ValueError):r.parse_once(out,[],{},'development',fail)
  self.assertEqual(len(read),1);self.assertEqual(read[0]['valid'],1)
  RESULT['visibility']={'domain_requests':received,'count':len(received),'learner_functions_have_no_supply_or_archive_parameter':True,'parse_attempt_recorded_before_loader':True,'failed_attempt_not_retried':True,'formal_objects_accessed':False}

if __name__=='__main__':
 tests=unittest.defaultTestLoader.loadTestsFromTestCase(IndependentAudit)
 start=time.monotonic()
 with mock.patch.object(m.base.public,'public_inputs',side_effect=AssertionError('FORBIDDEN FORMAL INPUT')),mock.patch.object(m.base.public,'attach_labels',side_effect=AssertionError('FORBIDDEN CSV')),mock.patch.object(m.data,'Archive',side_effect=AssertionError('FORBIDDEN ARCHIVE')):
  tested=unittest.TextTestRunner(verbosity=2).run(tests)
 result={'status':'PASS' if tested.wasSuccessful() and not tested.skipped else 'FAILED','counts':{'run':tested.testsRun,'passed':tested.testsRun-len(tested.failures)-len(tested.errors)-len(tested.skipped),'failures':len(tested.failures),'errors':len(tested.errors),'skipped':len(tested.skipped)},'failures':[{'test':str(t),'traceback':tb} for t,tb in tested.failures+tested.errors],'details':RESULT,'runtime_seconds':time.monotonic()-start,'environment':{'python':sys.version,'numpy':np.__version__,'torch':torch.__version__,'cpu_affinity':sorted(os.sched_getaffinity(0)),'torch_threads':torch.get_num_threads(),'process_threads':[s for s in open('/proc/self/status').read().splitlines() if s.startswith('Threads:')][0]},'scope':{'formal_labels':False,'formal_text':False,'formal_arrays':False,'native_BGE':False,'formal_execute':False,'project_linux':False,'tiny_updates_and_handmade_fits':True,'project_sources_modified':False}}
 dump(OUT/'independent_results.json',result);print(json.dumps({'status':result['status'],'counts':result['counts'],'runtime_seconds':result['runtime_seconds']},indent=2));sys.exit(0 if result['status']=='PASS' else 1)
