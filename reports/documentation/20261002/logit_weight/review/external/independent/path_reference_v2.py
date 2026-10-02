"""Independent real-ID schedules, bounded tiny 288-update continuation, cache and gate probes.
All model operations and serialized cache payloads here are HANDWRITTEN fixtures.
Reads only original small manifests/partition/training JSON for paired schedule comparisons.
"""
from __future__ import annotations
import copy,hashlib,json,os,random,shutil
from pathlib import Path
from unittest import mock
import numpy as np
import torch
import step28_er_weight as m
import step28_er_weight_run as run
import test_step28_bge_continual_contracts as fixture
import test_step28_er_weight_contracts as gates
ROOT=Path(__file__).resolve().parents[1];SRC=ROOT/'submission';OUT=ROOT/'evidence/path_reference';OUT.mkdir(exist_ok=False)
torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
if hasattr(os,'sched_setaffinity'):os.sched_setaffinity(0,{min(os.sched_getaffinity(0))})
p=m.contract('logit');parent=m.parent.contract();report={'scope':'Tiny handwritten models/caches ONLY. Real IDs and old schedules are nonlabel JSON evidence.','schedule':[],'gate_mutations':[]}
def load(path):return json.loads(path.read_text())
def save(path,value):path.write_text(json.dumps(value,ensure_ascii=False,sort_keys=True,separators=(',',':'))+'\n')
def seed(n,*parts):return int.from_bytes(hashlib.sha256((json.dumps([n,*parts],ensure_ascii=False,sort_keys=True,separators=(',',':'))+'\n').encode()).digest()[:8],'big')%(2**63-1)
def sequence(ids,n):
 r=random.Random(n);out=[]
 for _ in range(6):
  a=sorted(ids);r.shuffle(a);out+=a
 return out
def reservoir_append(members,ids,rng,seen):
 for uid in sorted(ids):
  seen+=1;k=len(members) if len(members)<6 else rng.randrange(seen)
  if k<6:
   if k==len(members):members.append(uid)
   else:members[k]=uid
 return seen
oldjob=SRC/p['baseline']['local_small_job'];weightjob=SRC/p['weight_reference']['local_small_job'];ref=m.baseline(p,oldjob,weightjob)
fit={d:[r['group_uid'] for r in ref['partition']['fit'] if r['domain']==d] for d in 'ABC'}
schedule_csv=['order,stage,step,current_id,history_id,current_seed,history_seed']
for order in ('ABC','BCA','CAB'):
 rng=random.Random(seed(parent['memory_seed'],order,'retention'));members=[];seen=reservoir_append(members,fit[order[0]],rng,0)
 initial_members=members.copy();assert members==ref['manifest']['memories'][order+'_logit_after1']['members']==ref['manifest']['memories'][order+'_er_after1']['members']
 stage_records=[]
 for stage in (2,3):
  stream=seed(parent['schedule_seed'],order,stage,'current');current=sequence(fit[order[stage-1]],stream)
  dr=random.Random(seed(parent['memory_seed'],order,stage,'history_draws'));history=[members[dr.randrange(6)] for _ in range(288)]
  oldlog=run.old_training(ref,order,stage)
  assert oldlog['current_ids']==current and oldlog['history_ids']==history and oldlog['current_dropout_stream']==stream
  wr=ref['weight_reference'];name=f'{order}_quarter_stage{stage}';rec=wr['manifest']['training'][name];quarter=load(wr['job']/'run'/rec['path'])
  assert quarter['current_ids']==current and quarter['history_ids']==history and quarter['current_dropout_stream']==stream
  assert oldlog['adam_step']==quarter['adam_step']==288*stage
  stage_records.append({'stage':stage,'members':members.copy(),'current_pairs':288,'history_pairs':288,'current_stream':stream,'adam_final':288*stage})
  for i,(c,h) in enumerate(zip(current,history)):
   cs=seed(stream,i,'dropout');hs=seed(parent['memory_seed'],order,stage,i,'history_dropout')
   assert cs==m.data.seed_for(stream,i,'dropout') and hs==m.data.seed_for(parent['memory_seed'],order,stage,i,'history_dropout')
   schedule_csv.append(f'{order},{stage},{i+1},{c},{h},{cs},{hs}')
  if stage==2:
   seen=reservoir_append(members,fit[order[1]],rng,seen)
   assert members==ref['manifest']['memories'][order+'_er_stage2']['members']
 report['schedule'].append({'order':order,'first_members':initial_members,'stages':stage_records,'stage3_domain_counts':{d:sum(uid in fit[d] for uid in members) for d in 'ABC'}})
(OUT/'paired_schedule_1728.csv').write_text('\n'.join(schedule_csv)+'\n')
# Independent tiny continuation uses independently drawn current/history IDs.
root=OUT/'tiny';root.mkdir()
for folder in ('updates','work','models','scores','maps','points','memory'):(root/folder).mkdir()
c=fixture.config();model=fixture.tiny_model();optimizer=m.core.make_optimizer(model,c)
history=[fixture.handmade_group(f'fit_A_{i:02}',3) for i in range(48)]
current=[fixture.handmade_group(f'fit_B_{i:02}',5) for i in range(48)]
fixture.toy_prior_step(model,optimizer,history[0],c,count=288)
mem=m.parent.Memory('ABC',parent['memory_seed'],True,{'a':1.,'b':0.})
mem.retain(history,1,lambda gs:m.parent.ranking.score(model,gs,c));initial_targets={k:v.copy() for k,v in mem.references.items()}
(root/'memory/handwritten_first.json').write_bytes(mem.to_bytes())
stream=seed(parent['schedule_seed'],'ABC',2,'current');ids=sequence([g.uid for g in current],stream)
dr=random.Random(seed(parent['memory_seed'],'ABC',2,'history_draws'));members=[g.uid for g in mem.reservoir.groups];hid=[members[dr.randrange(6)] for _ in range(288)]
expectedlog={'current_ids':ids,'history_ids':hid,'memory_after_training':{'members':members}}
class Budget:
 def check(self,reserve=0):pass
 def state(self):return {'handwritten_independent_fixture':True}
budget=Budget();actual_calls={'update':0,'group_forward':0};real_update=m.update;real_logits=m.base.logits
def track_update(*args,**kwargs):actual_calls['update']+=1;return real_update(*args,**kwargs)
def track_logits(*args,**kwargs):actual_calls['group_forward']+=1;return real_logits(*args,**kwargs)
with mock.patch.object(m,'update',side_effect=track_update),mock.patch.object(m.base,'logits',side_effect=track_logits):
 log=run.train_stage(model,optimizer,current,mem,c,'ABC',2,'logit_quarter',root,expectedlog,budget,p)
assert actual_calls=={'update':288,'group_forward':576}
assert m.parent.adam_step(optimizer)==576 and log['current_ids']==ids and log['history_ids']==hid
beforemodel=m.core.state_digest(model.state_dict());beforeopt=m.core.state_digest(optimizer.state_dict());beforerng=run.rng_state()
cal=[fixture.handmade_group(f'ind_cal_{i:02}',4) for i in range(12)];valid=[fixture.handmade_group(f'ind_valid_{i:02}',6) for i in range(60)]
point=run.checkpoint(root,'ABC_logit_quarter_stage2',model,optimizer,c,'ABC',2,cal,valid,mem.first_map,budget,.25,p)
assert m.core.state_digest(model.state_dict())==beforemodel and m.core.state_digest(optimizer.state_dict())==beforeopt and run.rng_state()==beforerng
assert point['full_model_adam_and_rng_restore_verified'] and not (root/point['full_checkpoint']['path']).exists()
mem.auxiliary=point['learner_auxiliary'];scored={};teacher_modes=[]
def score(gs):
 result=m.parent.ranking.score(model,gs,c);teacher_modes.append({'training':model.training,'grad_enabled_inside_score_contract':'score() uses torch.inference_mode; returned numpy float32'})
 for g,row in zip(gs,result):scored[g.uid]=row.copy()
 return result
mem.retain(current,2,score)
assert set(scored)==set(mem.references)-set(initial_targets)
for uid,row in mem.references.items():assert np.array_equal(row,initial_targets[uid] if uid in initial_targets else scored[uid])
assert all(not e['training'] for e in teacher_modes)
retained_bytes=mem.to_bytes();(root/'memory/handwritten_after2.json').write_bytes(retained_bytes)
restored=m.parent.Memory.from_bytes(retained_bytes);assert restored.to_bytes()==retained_bytes and len(retained_bytes)<=1048576 and len(mem.reservoir.groups)==6
# Deliberately include all auxiliary fields in the byte cap, not just numeric targets.
over=m.parent.Memory.from_bytes(retained_bytes);over.auxiliary['independent_padding']='x'*1048576
try:over.to_bytes()
except ValueError as exc:budget_error=str(exc)
else:raise AssertionError('Complete byte cap not enforced')
# Save a tiny full state independently, perturb then restore clone; verify next actual Adam update and same cache draw.
full=root/'work/independent_tiny_full.pt';meta={'handwritten':True,'rng':run.rng_state()};rec=m.core.save_state(full,model,optimizer,meta)
other=copy.deepcopy(model);other_opt=m.core.make_optimizer(other,c)
with torch.no_grad():next(other.parameters()).add_(.01)
assert m.core.restore_state(full,other,other_opt,rec['state_sha256'])==meta
mem.begin_stage(3);restored.begin_stage(3);g1,t1=mem.draw();g2,t2=restored.draw();assert g1.uid==g2.uid and np.array_equal(t1,t2)
cur3=fixture.handmade_group('independent_stage3_current',9)
a=m.update(model,optimizer,cur3,g1,c,.25,3,1,8301,9301,reference=t1,logit_weight=.5)
b=m.update(other,other_opt,cur3,g2,c,.25,3,1,8301,9301,reference=t2,logit_weight=.5)
assert a==b and m.core.state_digest(model.state_dict())==m.core.state_digest(other.state_dict()) and m.core.state_digest(optimizer.state_dict())==m.core.state_digest(other_opt.state_dict())
report['tiny_continuation']={'initial_genuine_updates':1,'initial_counter_rebase':[1,288],'actual_stage2_calls':actual_calls,'adam_after_stage2':576,'new_targets':list(scored),'old_survivors':[u for u in mem.references if u in initial_targets],'surviving_targets_equal':True,'new_targets_evaluated_on_own_model':True,'checkpoint_model_optimizer_rng_exact':True,'checkpoint_complete_replayed_groups':72,'inference_restore_exact':True,'retained_cache_bytes':len(retained_bytes),'auxiliary_bytes':len(m.data.json_bytes(mem.auxiliary)),'budget_overflow_rejected':budget_error,'next_actual_stage3_update_equal':True,'adam_after_next_update':577,'next_history_id':g1.uid,'model_digest_after_next_update':m.core.state_digest(model.state_dict()),'optimizer_digest_after_next_update':m.core.state_digest(optimizer.state_dict())}
# Gate fixtures are explicitly fake model identities; they do not prove native training.
gatehome=OUT/'gate_fixture';gatehome.mkdir();groot,manifest,greference=gates.gate_fixture(gatehome,p)
original_files={f.relative_to(groot):f.read_bytes() for f in groot.rglob('*') if f.is_file()}
original_manifest=copy.deepcopy(manifest)
qualified=run.blind_gate(groot,manifest,greference,p);assert len(qualified)==6 and all(len(roles)==3 for roles in qualified.values())
report['complete_gate']={'endpoints':6,'score_files_including_calibration':24,'returned_metric_roles':18,'fake_weights_only':True}
def reset():
 for relative,blob in original_files.items():(groot/relative).write_bytes(blob)
 return copy.deepcopy(original_manifest)
def edit_json_record(record,change):
 path=groot/record['path'];value=load(path);change(value);m.data.write_json(path,value);return m.data.record(path,groot)
def mutate_score(mf,name,role):
 def change(pt):
  path=groot/pt['scores'][role]['path'];a=np.load(path,allow_pickle=False).copy();a.flat[0]=np.nan;np.save(path,a,allow_pickle=False);pt['scores'][role]=m.data.record(path,groot)
 mf['points'][name]=edit_json_record(mf['points'][name],change)
def mutate_log(mf,name,change):mf['training'][name]=edit_json_record(mf['training'][name],change)
def mutate_values(mf,name,change):
 def edit(log):
  path=groot/log['update_file']['path'];a=np.load(path,allow_pickle=False).copy();change(a);np.save(path,a,allow_pickle=False);log['update_file']=m.data.record(path,groot)
 mutate_log(mf,name,edit)
def reject(label,fn):
 mf=reset();fn(mf)
 try:
  with mock.patch.object(run.prior,'parse_once',side_effect=AssertionError('gate must be before labels')):run.blind_gate(groot,mf,greference,p)
 except ValueError as exc:report['gate_mutations'].append({'case':label,'rejected':True,'error':str(exc)})
 else:raise AssertionError('Mutation accepted: '+label)
for name in m.expected_points(p):
 for role in ('calibration','development','stage-cal','first-cal'):
  reject(name+'/'+role+'/nonfinite_with_updated_hash',lambda mf,n=name,r=role:mutate_score(mf,n,r))
name='ABC_logit_quarter_stage2';cols={key:i for i,key in enumerate(m.step_columns(p))}
reject('MSE_wrong_0.125_even_if_total_recomputed',lambda mf:mutate_values(mf,name,lambda a:(a.__setitem__((slice(None),cols['logit_weight']),.125),a.__setitem__((slice(None),cols['total']),a[:,cols['current_total']]+a[:,cols['weighted_history_total']]+.125*a[:,cols['logit_mse']]))))
reject('only_BCE_history_weighted_not_rank_hard',lambda mf:mutate_values(mf,name,lambda a:a.__setitem__((slice(None),cols['weighted_history_total']),.25*a[:,cols['history_bce']]+a[:,cols['history_rank']]+.5*a[:,cols['history_hard']])))
reject('wrong_own_stage2_model_source',lambda mf:mf['memories'][name]['reference_update'].__setitem__('model_state_sha256','OLD_LOGIT1_STAGE2_IS_NOT_OWN_MODEL'))
reject('teacher_training_mode',lambda mf:mf['memories'][name]['reference_update'].__setitem__('mode','train'))
reject('new_target_list_missing',lambda mf:mf['memories'][name]['reference_update'].__setitem__('new_target_ids',[]))
reject('first_LOGIT_cache_replaced_by_ER',lambda mf:mf['restored_starts']['ABC_logit_quarter'].__setitem__('memory_source',greference['manifest']['memories']['ABC_er_after1']['file']))
reject('old_first_reference_summary_refreshed',lambda mf:mf['restored_starts']['ABC_logit_quarter']['memory_summary']['references'].__setitem__(next(iter(mf['restored_starts']['ABC_logit_quarter']['memory_summary']['references'])),'fresh_wrong_digest'))
def refresh_survivor(mf):
 rec=mf['memories'][name];memory=m.parent.Memory.from_bytes((groot/rec['file']['path']).read_bytes());first=greference['manifest']['memories']['ABC_logit_after1']['references'];uid=next(u for u in first if u in memory.references)
 memory.references[uid]=memory.references[uid]+np.float32(.1);blob=memory.to_bytes();path=groot/rec['file']['path'];path.write_bytes(blob);rec.update(memory.summary());rec['file']=m.data.record(path,groot)
reject('actual_old_survivor_values_refreshed_with_consistent_cache_hash',refresh_survivor)
reject('stage3_consumes_first_cache_targets',lambda mf:mutate_log(mf,'ABC_logit_quarter_stage3',lambda x:x['memory_after_training'].__setitem__('references',greference['manifest']['memories']['ABC_logit_after1']['references'])))
reject('unpaired_current_dropout_stream',lambda mf:mutate_log(mf,name,lambda x:x.__setitem__('current_dropout_stream',-7)))
reject('Adam_reset',lambda mf:mutate_log(mf,name,lambda x:x.__setitem__('adam_step',288)))
reject('encoder_last_step_lr_nonzero',lambda mf:mutate_values(mf,name,lambda a:a.__setitem__((287,cols['encoder_lr']),1e-5)))
reject('missing_sixth_endpoint',lambda mf:mf['points'].pop(m.expected_points(p)[-1]))
reject('old_study_policy_hash',lambda mf:mf.__setitem__('policy_sha256',m.policy_sha256(m.contract('low'))))
reject('wrong_physical_update_total',lambda mf:mf.__setitem__('physical_updates',1727))
manifest=reset();assert len(run.blind_gate(groot,manifest,greference,p))==6
save(OUT/'gate_fixture_manifest.json',manifest)
report['invalid_gate_cases']=len(report['gate_mutations']);report['status']='PASS';report['cpu_affinity']=sorted(os.sched_getaffinity(0))
save(OUT/'results.json',report);print(json.dumps(report,ensure_ascii=False,indent=2))
