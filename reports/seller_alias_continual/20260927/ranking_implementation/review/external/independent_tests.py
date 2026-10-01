"""Independent reviewer checks. All inputs are handmade; no formal labels/models.

Uses production math and persistence as subjects under test. The sole imported
submitted fixture builds explicitly fake receipts for file-boundary tests.
No formal execution entry or native BGE loader is called. GPU telemetry in the
one orchestration test is mocked and is not reported as GPU execution.
"""
from __future__ import annotations
import os,sys,json,math,hashlib,itertools,copy,tempfile,unittest,time,platform
from pathlib import Path
from unittest import mock
from contextlib import ExitStack
os.sched_setaffinity(0,{min(os.sched_getaffinity(0))})
sys.dont_write_bytecode=True
ROOT=Path('/mnt/data/ranking_review_source');OUT=Path('/mnt/data/ranking_reviewer_evidence')
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'tests')]
import numpy as np
import torch
from torch.utils.checkpoint import checkpoint as activation_checkpoint
import step28_alias_ranking as m
import step28_alias_ranking_run as r
from test_step28_alias_ranking_contracts import handmade_run_files

torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
EVIDENCE={};P=m.contract();CFG=m.reference_config(P,'s0')
EDGES=list(itertools.combinations(range(28),2));INDEX={p:k for k,p in enumerate(EDGES)}
COMP=[c for c,n in enumerate([2]*8+[3]*4) for _ in range(n)]
Y=np.array([int(COMP[a]==COMP[b]) for a,b in EDGES],dtype=np.int64)
def group(uid='reviewer_hand',labelled=True):
 items=tuple(tuple((f'row{i:02}_{j}',f'人工标题{i % 5}：{j}',f'人工描述{i}记录{j}。') for j in range(2+(i%2))) for i in range(28))
 g=m.data.Group(uid,tuple(f'account{i:02}' for i in range(28)),items,tuple(map(int,Y)) if labelled else None);g.validate();return g

def reference(s,y):
 s=np.asarray(s,dtype=np.float64);y=np.asarray(y,dtype=np.int64)
 vals={'bce':float(np.mean(np.logaddexp(0,s)-y*s)),'rank':0.,'hard':0.}
 grads={'bce':(np.exp(-np.logaddexp(0,-s))-y)/378,'rank':np.zeros(378),'hard':np.zeros(378)}
 chosen=[]
 for q in range(28):
  candidates=[j for j in range(28) if j!=q];ix=[INDEX[tuple(sorted((q,j)))] for j in candidates]
  pos=[i for i in ix if y[i]]
  neg=sorted([j for j in candidates if not y[INDEX[tuple(sorted((q,j)))]]],key=lambda j:(-float(s[INDEX[tuple(sorted((q,j)))]]),j))[:5]
  ns=[INDEX[tuple(sorted((q,j)))] for j in neg];chosen.append(neg)
  den=float(np.logaddexp.reduce(s[ix]));vals['rank']+=(den-float(s[pos].mean()))/28
  grads['rank'][ix]+=np.exp(s[ix]-den)/28;grads['rank'][pos]-=1/(28*len(pos))
  for a in pos:
   for b in ns:
    d=s[b]-s[a];vals['hard']+=float(np.logaddexp(0,d))/(28*5*len(pos));v=math.exp(-float(np.logaddexp(0,-d)))/(28*5*len(pos));grads['hard'][b]+=v;grads['hard'][a]-=v
 vals['total']=vals['bce']+vals['rank']+.5*vals['hard'];grads['total']=grads['bce']+grads['rank']+.5*grads['hard']
 return vals,grads,chosen

class Encoder(torch.nn.Module):
 def __init__(self,dropout=False):
  super().__init__();self.embed=torch.nn.Embedding(47,5);self.linear=torch.nn.Linear(5,5);self.dropout=dropout;self.forward_values=[]
 def tokenizer(self,texts,**kw):
  return {'input_ids':torch.tensor([[hashlib.sha256(t.encode()).digest()[j]%47 for j in range(4)] for t in texts])}
 def block(self,x):
  return self.linear(torch.nn.functional.dropout(x,p=.3,training=self.training and self.dropout)).tanh()
 def forward(self,f):
  x=self.embed(f['input_ids']).mean(1)
  if self.training and self.dropout:z=activation_checkpoint(self.block,x,use_reentrant=False,preserve_rng_state=True)
  else:z=self.block(x)
  if self.dropout and self.training:self.forward_values.append(z.detach().clone())
  return {'sentence_embedding':z}
def tiny(seed=199,dropout=False):
 torch.manual_seed(seed);return m.core.build_model(Encoder(dropout),20,9)

def quantile_linear(v,p):
 a=sorted(map(float,v));at=(len(a)-1)*p;i=math.floor(at);f=at-i
 return a[i] if i==len(a)-1 else a[i]*(1-f)+a[i+1]*f

class Checks(unittest.TestCase):
 def test_01_independent_losses_all_gradients_and_finite_difference(self):
  maxv=maxg=maxfd=0.;cases=[np.random.default_rng(610).normal(0,2,378),np.zeros(378),np.random.default_rng(611).integers(-2,3,378).astype(float)]
  for s in cases:
   ref,grad,chosen=reference(s,Y);x=torch.tensor(s,requires_grad=True);actual=m.objectives(x,torch.tensor(Y,dtype=torch.float64),.5)
   for name in ('bce','rank','hard','total'):
    val=float(actual[name].detach());g,=torch.autograd.grad(actual[name],x,retain_graph=True)
    maxv=max(maxv,abs(val-ref[name]));maxg=max(maxg,float(np.max(np.abs(g.numpy()-grad[name]))));self.assertLess(abs(val-ref[name]),2e-12);np.testing.assert_allclose(g.numpy(),grad[name],atol=2e-14,rtol=1e-12)
   live,positive,selected=m.hard_candidates(x,torch.tensor(Y));self.assertTrue(live.requires_grad);self.assertFalse(selected.requires_grad);self.assertEqual(selected.tolist(),chosen)
   self.assertLess(abs(grad['hard'].sum()),1e-14);self.assertLess(abs(grad['rank'].sum()),1e-14)
  s=cases[0];ref,grad,_=reference(s,Y)
  for idx in np.linspace(0,377,31,dtype=int):
   a=s.copy();b=s.copy();a[idx]+=1e-5;b[idx]-=1e-5
   diff=(reference(a,Y)[0]['total']-reference(b,Y)[0]['total'])/2e-5
   maxfd=max(maxfd,abs(diff-grad['total'][idx]));self.assertLess(abs(diff-grad['total'][idx]),2e-9)
  EVIDENCE['all_loss_gradients']={'score_cases':3,'gradient_components_per_case':4*378,'finite_difference_coordinates':31,'max_loss_error':maxv,'max_gradient_error':maxg,'max_finite_difference_error':maxfd}

 def test_02_dropout_checkpoint_random_streams_three_arms(self):
  models=[tiny(dropout=True) for _ in range(3)];first_states=[m.core.state_digest(x.state_dict()) for x in models];self.assertEqual(len(set(first_states)),1)
  logs=[];rngs=[]
  for kind,model in zip(m.ARMS,models):
   opt=m.core.make_optimizer(model,CFG);logs.append(m.update(model,opt,group(),CFG,P,kind,1,789,observe=True));rngs.append(torch.random.get_rng_state().clone())
  for model,rng in zip(models[1:],rngs[1:]):
   self.assertEqual(len(model.encoder.forward_values),len(models[0].encoder.forward_values));self.assertTrue(torch.equal(rng,rngs[0]))
   for x,y in zip(model.encoder.forward_values,models[0].encoder.forward_values):self.assertTrue(torch.equal(x,y))
  self.assertEqual(m.core.state_digest(models[0].head.state_dict()),m.core.state_digest(models[1].head.state_dict()))
  EVIDENCE['dropout_checkpoint']={'arms':3,'first_forward_microbatches_per_arm':len(models[0].encoder.forward_values),'same_initial_state':True,'all_first_microbatch_outputs_equal':True,'rng_after_backward_equal':True,'d_schedule_first_head_update_equal':True,'scope':'Tiny CPU encoder with actual dropout and non-reentrant checkpoint; not BGE or CUDA.'}

 def test_03_lr_full_series_and_real_adam_zero_step(self):
  expected=[1e-5*t/87 if t<=87 else 1e-5*(864-t)/777 for t in range(1,865)]
  for arm in ('schedule','hard'):np.testing.assert_allclose([m.encoder_lr(P,arm,t) for t in range(1,865)],expected,rtol=1e-15,atol=1e-21)
  self.assertAlmostEqual(math.fsum(expected),.00432,places=17);self.assertAlmostEqual(math.fsum([2e-5]*864),.01728,places=17)
  model=tiny();opt=m.core.make_optimizer(model,CFG)
  # Actual tiny Adam history. Loss is a controlled algebraic function, not formal training.
  for t in range(1,865):
   opt.zero_grad(set_to_none=True);loss=sum((p**2).sum() for p in model.parameters());loss.backward();m.set_learning_rate(opt,P,'hard',t)
   if t==864:before=m.core.state_digest(model.encoder.state_dict());head=m.core.state_digest(model.head.state_dict());states=m.core.state_digest(opt.state_dict())
   opt.step()
  self.assertEqual(before,m.core.state_digest(model.encoder.state_dict()));self.assertNotEqual(head,m.core.state_digest(model.head.state_dict()));self.assertNotEqual(states,m.core.state_digest(opt.state_dict()));self.assertTrue(all(float(v['step'])==864 for v in opt.state.values()))
  EVIDENCE['schedule']={'lr_sum_schedule':math.fsum(expected),'lr_sum_d':.01728,'ratio':math.fsum(expected)/.01728,'positive_encoder_lr_calls':sum(t>0 for t in expected),'last_encoder_unchanged':True,'last_head_changed':True,'last_adam_state_changed':True,'scope':'864 actual tiny algebraic-loss Adam calls; not formal training.'}

 def test_04_all_54_scores_rejected_before_label_access(self):
  with tempfile.TemporaryDirectory(dir=ROOT/'reports') as tmp:
   path=Path(tmp);selected,partition,weights=handmade_run_files(path,P);files=sorted(path.glob('*/scores/epoch*_*.npy'));files=[p for p in files if not p.name.endswith('_metrics.npy')];self.assertEqual(len(files),54)
   rejected=[]
   with mock.patch.object(m.base,'partition',return_value=(selected,partition)),mock.patch.object(m.base.public,'public_inputs',return_value=({},[],{'handmade':True})),mock.patch.object(m.base.public,'attach_labels',side_effect=AssertionError('Labels must not be reached')) as parser:
    arrays,_,run=r.validate_run(path,P,{},[],{'handmade':True});self.assertEqual(len(arrays),9)
    for f in files:
     original=f.read_bytes();modified=original[:-1]+bytes([original[-1]^1]);f.write_bytes(modified)
     try:
      with self.assertRaises(ValueError):r.evaluate(path,path/'eval',P,mock.Mock())
      self.assertFalse((path/'eval').exists());rejected.append(f.relative_to(path).as_posix())
     finally:f.write_bytes(original)
    parser.assert_not_called()
   EVIDENCE['score_corruption']={'count':len(rejected),'rejected_before_label_parser':len(rejected),'files':rejected,'fixtures_are_not_models':True}

 def test_05_statistics_faults_preserve_eighteen_matrices(self):
  rng=np.random.default_rng(712);truth=np.tile(Y,(60,1)).astype(np.uint8)
  arrays={run:{'points':{e:rng.normal(0,1,(60,378)).astype(np.float32) for e in ('3','6')},'threshold':.35} for run in m.RUNS}
  partition={'development':[{'group_uid':f'hand{i}','domain':'ABC'[i//20]} for i in range(60)]}
  checked=[]
  with tempfile.TemporaryDirectory() as td:
   reference=None
   for k,(owner,name) in enumerate([(None,None),(m,'paired_summary'),(m,'acceptance'),(m.base,'automatic_report'),(m.base,'metric_comparison'),(m.base,'automatic_comparison')]):
    dest=Path(td)/f'case{k}';dest.mkdir();r.collect(truth,arrays,partition,dest,P,{'handmade':True});raw=(dest/'collected.json').read_bytes();self.assertEqual(len(list(dest.glob('*/epoch*_metrics.npy'))),18)
    if owner:
     with mock.patch.object(owner,name,side_effect=RuntimeError('injected saved-statistics error')):
      with self.assertRaisesRegex(RuntimeError,'injected'):r.finalize(dest)
     self.assertFalse((dest/'evaluation.json').exists());self.assertEqual((dest/'collected.json').read_bytes(),raw)
    with mock.patch.object(m.base.public,'public_inputs',side_effect=AssertionError('NO input')),mock.patch.object(m.base.public,'attach_labels',side_effect=AssertionError('NO labels')),mock.patch.object(m,'load_model',side_effect=AssertionError('NO model')):
     result=r.finalize(dest)
    # Path-independent payload fields; identical collected files have identical relative records.
    if reference is None:reference=result
    else:self.assertEqual(result,reference);checked.append(name)
   EVIDENCE['postcollection_faults']={'fault_sites':checked,'matrices_preserved_each_case':18,'collection_bytes_unchanged':True,'recovered_results_equal':True,'recovery_formal_reads':0}

 def test_06_shuffled_domain_nonconstant_paired_bootstrap(self):
  rng=np.random.default_rng(42);domains=np.array(list('ABC')*20);d=rng.normal(size=(3,60,22))*.02
  d+=rng.normal(size=(1,60,1))*.05;d+=np.arange(3)[:,None,None]*.03
  summary=m.paired_summary(d,domains.tolist(),P['evaluation']);draws=np.random.default_rng(20260927).integers(0,20,(5000,3,20));weights=np.zeros((5000,60))
  for j,key in enumerate('ABC'):
   rows=np.flatnonzero(domains==key)
   for b in range(5000):weights[b,rows]=np.bincount(draws[b,j],minlength=20)
  boot=weights@(d.sum(0)/3)/60;maxerr=0.
  for j,name in enumerate(m.metrics.COLUMNS):
   expected=[quantile_linear(boot[:,j],q) for q in (.025,.975)];actual=summary['metrics'][name]
   maxerr=max(maxerr,float(np.max(np.abs(np.array(expected)-actual['conditional_95pct_interval']))));np.testing.assert_allclose(actual['conditional_95pct_interval'],expected,rtol=0,atol=1e-14);np.testing.assert_allclose(actual['per_seed'],d[:,:,j].mean(1),rtol=0,atol=1e-14)
  EVIDENCE['independent_bootstrap']={'replicates':5000,'metric_count':22,'domain_order':'interleaved A B C, not contiguous','method':'independent frequency weights and hand-written linear interpolation','max_interval_error':maxerr}

 def test_07_each_acceptance_condition_and_no_auxiliary_selection(self):
  rows={k:{'mean':.02,'per_seed':[.02]*3,'conditional_95pct_interval':[.001,.05]} for k in m.metrics.COLUMNS}
  for name in ('brier','log_loss'):rows[name]={'mean':-.02,'per_seed':[-.02]*3,'conditional_95pct_interval':[-.05,-.001]}
  base={'metrics':rows};self.assertTrue(m.acceptance(base,P['evaluation'])['passed']);self.assertEqual(len(m.acceptance(base,P['evaluation'])['checks']),13)
  mutations=[('map','mean',.00999),('map','conditional_95pct_interval',[0,.05]),('map','per_seed',[.02,.02,0]),('recall_at_5','mean',0),('recall_at_5','per_seed',[0,.02,.02])]
  for n,sign in m.GUARDS.items():mutations.extend([(n,'mean',-sign*.00001),(n,'per_seed',[-sign*.00001,.02,.02])])
  failed=[]
  for n,f,v in mutations:
   test=copy.deepcopy(base);test['metrics'][n][f]=v;a=m.acceptance(test,P['evaluation']);self.assertFalse(a['passed']);self.assertEqual(len(a['failed']),1);failed.extend(a['failed'])
  test=copy.deepcopy(base);test['metrics']['map']['mean']=.01
  for n in m.GUARDS:test['metrics'][n]['mean']=0;test['metrics'][n]['per_seed'][0]=0
  self.assertTrue(m.acceptance(test,P['evaluation'])['passed']);self.assertEqual(len(set(failed)),13)
  EVIDENCE['acceptance']={'independent_single_condition_failures':failed,'gain_equality_allowed_at_001':True,'identification_guard_equality_allowed':True,'recall_equality_rejected':True}

 def test_08_all22_metrics_against_scalar_reference(self):
  cases=[np.zeros(378),np.random.default_rng(413).integers(-3,4,378).astype(float),np.random.default_rng(414).normal(0,2,378),np.where(Y,4.,-4.),np.where(Y,-4.,4.)]
  maxerr=0.
  for s in cases:
   actual,_=m.metrics.group_metrics(Y[None,:].astype(np.uint8),s[None,:]);expected={}
   # ROC by positive-negative pair concordance, AP by distinct-score thresholds.
   pos=s[Y==1];neg=s[Y==0];expected['roc_auc']=float(((pos[:,None]>neg[None,:])+.5*(pos[:,None]==neg[None,:])).mean())
   prev_r=0.;prev_p=1.;ap=pr=rfpr=0.
   for threshold in sorted(set(map(float,s)),reverse=True):
    pred=s>=threshold;tp=int(np.sum(pred&(Y==1)));fp=int(np.sum(pred&(Y==0)));rec=tp/20;prec=tp/(tp+fp)
    ap+=(rec-prev_r)*prec;pr+=(rec-prev_r)*(prec+prev_p)/2;prev_r=rec;prev_p=prec
    if fp/358<=.01:rfpr=max(rfpr,rec)
   expected.update(average_precision=ap,trapezoidal_pr_auc=pr,recall_at_fpr_1pct=rfpr)
   prob=np.exp(-np.logaddexp(0,-s));clipped=np.clip(prob,1e-15,1-1e-15);expected['brier']=float(((prob-Y)**2).mean());expected['log_loss']=float(np.mean(-Y*np.log(clipped)-(1-Y)*np.log(1-clipped)))
   pred=s>=0;tp=int(np.sum(pred&(Y==1)));fp=int(np.sum(pred&(Y==0)));fn=20-tp;tn=358-fp
   den=math.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn));expected.update(precision=tp/(tp+fp) if tp+fp else 0,recall=tp/20,f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0,specificity=tn/358,balanced_accuracy=(tp/20+tn/358)/2,mcc=(tp*tn-fp*fn)/den if den else 0)
   qs=[]
   for q in range(28):
    order=sorted([j for j in range(28) if j!=q],key=lambda j:(-float(s[INDEX[tuple(sorted((q,j)))]]),j));rel=[int(Y[INDEX[tuple(sorted((q,j)))]] ) for j in order];hits=[i+1 for i,v in enumerate(rel) if v];n=len(hits)
    row={'map':math.fsum((j+1)/rank for j,rank in enumerate(hits))/n,'mrr':1/hits[0]}
    for k in (1,3,5,10):row[f'recall_at_{k}']=sum(rel[:k])/n;row[f'ndcg_at_{k}']=math.fsum(v/math.log2(i+2) for i,v in enumerate(rel[:k]))/math.fsum(1/math.log2(i+2) for i in range(min(k,n)))
    qs.append(row)
   for k in qs[0]:expected[k]=math.fsum(v[k] for v in qs)/28
   for j,name in enumerate(m.metrics.COLUMNS):maxerr=max(maxerr,abs(expected[name]-actual[0,j]));self.assertLess(abs(expected[name]-actual[0,j]),2e-14,name)
  EVIDENCE['all22_scalar_metrics']={'handmade_score_cases':len(cases),'comparisons':len(cases)*22,'max_absolute_error':maxerr,'official_labels_read':0}

 def test_09_new_checkpoint_path_e3_e6_and_adam_continuation(self):
  model=tiny();opt=m.core.make_optimizer(model,CFG);g=group();groups={role:[g if role!='development' else group('blind_hand',False)] for role in m.base.ROLES}
  # One group per role here: true new checkpoint path and real score recomputation,
  # not a claim of formal 144/36/60 scale. Full array-shape boundaries tested separately.
  held={};points={};budget=mock.Mock();budget.check.return_value=None
  with tempfile.TemporaryDirectory() as td:
   out=Path(td)
   for sub in ('models','scores','work'):(out/sub).mkdir()
   original_remove=m.base.persistence.remove_work_file
   def restore_fresh_before_remove(path,root):
    saved=torch.load(path,map_location='cpu',weights_only=True);digest=m.core.state_digest(saved)
    other=tiny();otheropt=m.core.make_optimizer(other,CFG);meta=m.core.restore_state(path,other,otheropt,digest)
    self.assertEqual(m.core.state_digest(opt.state_dict()),m.core.state_digest(otheropt.state_dict()));held[meta['completed_updates']]=(other,otheropt)
    return original_remove(path,root)
   for step in range(1,865):
    if step==864:encoder_before=m.core.state_digest(model.encoder.state_dict());adam_before=m.core.state_digest(opt.state_dict())
    log=m.update(model,opt,g,CFG,P,'hard',step,4000+step,observe=step in (1,433,864))
    if step==433:
     other,otheropt=held[432];log2=m.update(other,otheropt,g,CFG,P,'hard',step,4000+step,observe=True);self.assertEqual(log,log2);self.assertEqual(m.core.state_digest(model.state_dict()),m.core.state_digest(other.state_dict()));self.assertEqual(m.core.state_digest(opt.state_dict()),m.core.state_digest(otheropt.state_dict()))
    if step in (432,864):
     with mock.patch.object(m.base.persistence,'remove_work_file',side_effect=restore_fresh_before_remove):point=r.checkpoint(model,opt,P,CFG,'s0_hard',step//144,groups,out,budget)
     points[str(step)]=point;self.assertTrue(point['full_model_and_adam_reloaded']);self.assertEqual(point['metadata']['completed_updates'],step)
     inference=tiny();m.core.restore_state(out/point['model']['path'],inference,None,point['model']['state_sha256']);np.testing.assert_array_equal(m.score(inference,groups['development'],CFG),np.load(out/point['scores']['development']['path']))
   self.assertEqual(encoder_before,m.core.state_digest(model.encoder.state_dict()));self.assertNotEqual(adam_before,m.core.state_digest(opt.state_dict()));self.assertEqual(points['864']['metadata']['next_encoder_lr'],None);self.assertFalse(list((out/'work').glob('*.pt')))
   EVIDENCE['actual_new_checkpoint']={'scope':'Tiny CPU encoder, 864 handmade supervised update calls plus a fresh-instance 433rd-step comparison; 1 handmade group per role, not formal scale','points':points,'fresh_adam_next_step_exact':True,'encoder_last_step_unchanged':True,'native_model_loads':0}

 def test_10_nine_run_orchestration_and_one_shared_manual_label_attachment(self):
  public={};metadata=[]
  for split,n in [('train',180),('development',60)]:
   public[split]=[group(f'fixture_{split}_{i:03}',False) for i in range(n)]
   metadata.extend({'group_uid':g.uid,'domain':'ABC'[i//(n//3)],'split':split} for i,g in enumerate(public[split]))
  order=[];labels=[];preflights=[]
  def preflight(policy,run,groups,budget,tokenize=False):preflights.append((run,tokenize));return {'initial_state_sha256':run.split('_')[0]}
  def attach(gs,config,split):
   self.assertEqual(split,'train');labels.append(split);return [m.data.Group(g.uid,g.sellers,g.items,tuple(map(int,Y))) for g in gs]
  def one(out,run,policy,groups,partition,budget,before):
   self.assertEqual(len(labels),1);self.assertTrue(all(g.labels is not None for g in groups['fit']));self.assertTrue(all(g.labels is None for g in groups['development']));order.append(run);out.mkdir();m.data.write_json(out/'manifest.json',{'fixture_not_training':True});return {'updates':864,'formal_training_seconds':0}
  with tempfile.TemporaryDirectory(dir=ROOT/'reports') as td,ExitStack() as stack:
   for target,name,kw in [(m.base.public,'public_inputs',{'return_value':(public,metadata,{'handmade':True})}),(m.base.public,'attach_labels',{'side_effect':attach}),(m.core,'model_files',{'return_value':copy.deepcopy(CFG['models']['split_rank'])}),(r,'preflight',{'side_effect':preflight}),(r,'train_one',{'side_effect':one}),(torch.cuda,'get_device_name',{'return_value':'HANDMADE MOCK NO GPU'})]:stack.enter_context(mock.patch.object(target,name,**kw))
   budget=mock.Mock();budget.state.return_value={'fixture':True}
   result=r.train(Path(td)/'run',P,budget,ROOT/'reports/seller_alias_continual/20260927/ranking_implementation/cpu/20260927_105450/audit.json',{'fixture_not_authorized':True})
   self.assertEqual(order,list(m.RUNS));self.assertEqual(labels,['train']);self.assertEqual(result['physical_updates'],7776);self.assertEqual([b for _,b in preflights],[True]+[False]*8)
  EVIDENCE['nine_run_orchestration']={'run_order':order,'handmade_label_attachment_calls':len(labels),'preflight_order':preflights,'declared_budget_sum':9*864,'scope':'Control-flow fixture: train_one, native preflight, archive lookup, and GPU name explicitly mocked. Not 7776 actual model updates.'}

if __name__=='__main__':
 print(json.dumps({'python':platform.python_version(),'torch':torch.__version__,'numpy':np.__version__,'affinity':sorted(os.sched_getaffinity(0)),'threads':torch.get_num_threads(),'formal_reads':0,'native_model_loads':0},indent=2));start=time.monotonic()
 with mock.patch.object(m.base.public,'public_inputs',side_effect=AssertionError('Forbidden formal inputs')),mock.patch.object(m.base.public,'attach_labels',side_effect=AssertionError('Forbidden formal labels')),mock.patch.object(m.data,'Archive',side_effect=AssertionError('Forbidden archive')),mock.patch.object(m.base,'load_model',side_effect=AssertionError('Forbidden native model load')):
  suite=unittest.defaultTestLoader.loadTestsFromTestCase(Checks);result=unittest.TextTestRunner(verbosity=2).run(suite)
 summary={'tests_run':result.testsRun,'failures':len(result.failures),'errors':len(result.errors),'skipped':len(result.skipped),'seconds':time.monotonic()-start,'evidence':EVIDENCE}
 (OUT/'independent_results.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2));print(json.dumps({k:v for k,v in summary.items() if k!='evidence'},indent=2));sys.exit(0 if result.wasSuccessful() and not result.skipped else 1)
