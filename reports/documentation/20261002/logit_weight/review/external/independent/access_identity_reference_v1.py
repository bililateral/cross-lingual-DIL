"""Byte/source closure and handwritten one-attempt bookkeeping, no formal input loader."""
from __future__ import annotations
import hashlib,json,os,platform,sys
from pathlib import Path
from unittest import mock
import numpy as np
import torch
import step28_er_weight as m
import step28_er_weight_run as run
import test_step28_bge_continual_contracts as fixture
ROOT=Path(__file__).resolve().parents[1];SRC=ROOT/'submission';OUT=ROOT/'evidence/access_identity_reference';OUT.mkdir(exist_ok=False)
torch.set_num_threads(1)
if hasattr(os,'sched_setaffinity'):os.sched_setaffinity(0,{min(os.sched_getaffinity(0))})
p=m.contract('logit');freeze=m.data.read_json(SRC/'reports/documentation/20261002/logit_weight/freeze.json')
assert m.sources(p)==freeze['source_files'] and len(m.sources(p))==34
oldjob=SRC/p['baseline']['local_small_job'];oldmanifest=m.data.read_json(oldjob/'run/manifest.json')
original=m.parent.POLICY_SHA256
assert len(run.prior.sources())==18 and run.prior.sources()==oldmanifest['source_files']
results={'source_closure_current':34,'source_closure_original_unchanged':18,'policy_sha256':m.policy_sha256(p),'ledger_probes':[]}
for split in ['train','development']:
 job=OUT/split;job.mkdir();m.data.write_json(job/'access.json',{'train':0,'valid':0,'heldout':0,'owners':0});calls=[]
 def fail_loader(groups,c,role):
  ledger=m.data.read_json(job/'access.json');calls.append({'split':role,'ledger_observed_inside_loader':ledger});raise RuntimeError('deliberate HANDWRITTEN loader failure')
 try:run.prior.parse_once(job,[],{},split,loader=fail_loader)
 except RuntimeError as exc:assert str(exc)=='deliberate HANDWRITTEN loader failure'
 else:raise AssertionError('Missing injected failure')
 try:run.prior.parse_once(job,[],{},split,loader=fail_loader)
 except ValueError as exc:rejection=str(exc)
 else:raise AssertionError('Repeated parse accepted')
 assert len(calls)==1 and calls[0]['ledger_observed_inside_loader']['train' if split=='train' else 'valid']==1
 results['ledger_probes'].append({'split':split,'actual_handwritten_loader_calls':1,'attempt_recorded_before_loader':True,'second_attempt_rejected':rejection,'final_access':m.data.read_json(job/'access.json')})
for forbidden in ['heldout','test','owners']:
 try:run.prior.parse_once(OUT,[],{},forbidden,loader=lambda *a:(_ for _ in ()).throw(AssertionError('Forbidden loader called')))
 except ValueError:pass
 else:raise AssertionError(forbidden)
# Observe the actual fresh target tensor on a handwritten update, without changing its construction.
c=fixture.config();model=fixture.tiny_model();optimizer=m.core.make_optimizer(model,c)
h=fixture.handmade_group('target_history',3);cur=fixture.handmade_group('target_current',5);fixture.toy_prior_step(model,optimizer,h,c)
ref=np.linspace(-.4,.5,378,dtype=np.float32);before=ref.copy();real_tensor=torch.tensor;captured=[]
def tensor(*args,**kwargs):
 result=real_tensor(*args,**kwargs)
 if args and args[0] is ref:captured.append(result)
 return result
with mock.patch.object(torch,'tensor',side_effect=tensor):
 m.update(model,optimizer,cur,h,c,.25,2,1,151,152,reference=ref,logit_weight=.5)
assert len(captured)==1 and captured[0].is_leaf and not captured[0].requires_grad and captured[0].grad is None and np.array_equal(ref,before)
results['target_gradient']={'fresh_target_tensor_count':1,'is_leaf':True,'requires_grad':False,'grad_is_none':True,'input_reference_unchanged':True}
# Invalid reference types/dtypes/nonfinite values are refused before zero_grad, forward or Adam.
invalid=[('infinity',np.full(378,np.inf,np.float32)),('requires_grad_tensor',torch.zeros(378,requires_grad=True)),('python_list',[0.]*378)]
invalid_results=[]
for name,reference in invalid:
 states=(m.core.state_digest(model.state_dict()),m.core.state_digest(optimizer.state_dict()))
 with mock.patch.object(m.base,'logits',side_effect=AssertionError('No forward for invalid target')),mock.patch.object(optimizer,'zero_grad',side_effect=AssertionError('No zero_grad for invalid target')),mock.patch.object(optimizer,'step',side_effect=AssertionError('No optimizer step for invalid target')):
  try:m.update(model,optimizer,cur,h,c,.25,2,2,151,152,reference=reference,logit_weight=.5)
  except ValueError as exc:invalid_results.append({'case':name,'error':str(exc)})
  else:raise AssertionError(name)
 assert states==(m.core.state_digest(model.state_dict()),m.core.state_digest(optimizer.state_dict()))
results['invalid_targets']=invalid_results
results['environment']={'python':sys.version,'executable':sys.executable,'platform':platform.platform(),'numpy':np.__version__,'torch':torch.__version__,'torch_cpu_threads':torch.get_num_threads(),'cpu_affinity':sorted(os.sched_getaffinity(0)),'cuda_available':torch.cuda.is_available(),'cuda_visible_devices':os.getenv('CUDA_VISIBLE_DEVICES'),'native_BGE_loaded':False,'dependency_installed':False}
results['status']='PASS';(OUT/'results.json').write_text(json.dumps(results,ensure_ascii=False,indent=2)+'\n');print(json.dumps(results,ensure_ascii=False,indent=2))
