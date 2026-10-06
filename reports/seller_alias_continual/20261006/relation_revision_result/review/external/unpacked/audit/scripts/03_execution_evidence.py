"""Independent frozen-source/evidence/schedule verification. No training or labels.
Actual module import below only checks binding/source inventory; no execute,
train, score, load_model, calibration fitting or project verification entry is called.
"""
from pathlib import Path
import json,hashlib,random,sys,datetime,ast,collections
import numpy as np
ROOT=Path(__file__).resolve().parents[2];IN=ROOT/'input';OUT=ROOT/'audit/outputs'
RR=IN/'reports/seller_alias_continual/20261006/relation_revision_result';JOB=RR/'job';SRC=RR/'source';RUN=JOB/'run'
def read(p):return json.loads(p.read_text())
def sh(p):return hashlib.sha256(p.read_bytes()).hexdigest()
verified=[]
def verify(root,rec):
 p=root/rec['path']; assert p.stat().st_size==rec['bytes'] and sh(p)==rec['sha256'],p
 verified.append(str(p.relative_to(ROOT)));return p
# Sources across approval, real job, collection, final statistics, submitted integration CPU.
gate=read(RR/'authorization.json');s=gate['source_files'];assert len(s)==31
for rec in s: verify(SRC,rec)
assert len(set(x['path'] for x in s))==31
ex=read(JOB/'execution.json');mf=read(RUN/'manifest.json');co=read(JOB/'evaluation/collected.json');ev=read(JOB/'evaluation/evaluation.json');cpu=read(RR/'admission/cpu.json');native=read(RR/'admission/native.json')
for records in [ex['sources'],mf['source_files'],co['source_files'],ev['source_files'],cpu['source_files']]:assert records==s
# Compare to exactly the prior reviewed source bytes.
prior=ROOT/'history/pilot_input'
for rec in s:verify(prior,rec)
assert ex['gate']['sha256']==sh(RR/'authorization.json') and ex['gate']['bytes']==(RR/'authorization.json').stat().st_size
for kind,name in [('native','native.json'),('integration_cpu','cpu.json')]:
 r=gate[kind];assert r['sha256']==sh(RR/'admission'/name) and r['bytes']==(RR/'admission'/name).stat().st_size
for n,sha in native['scientific_sources'].items():assert sh(SRC/n)==sha,n
assert cpu['mode']=='cpu' and native['mode']=='native'
assert cpu['status']==native['status']=='PASS_HANDWRITTEN_ONLY'
# Actual import/global binding validation (no data/model execution).
sys.path.insert(0,str(SRC/'scripts'))
import step28_relation_revision_run as new
assert new.sources()==s
binding={}
for n in ['train','checkpoint','blind_gate','collect','read_roles','comparisons','finalize','complete','recover_statistics','execute']:
 fn=getattr(new.runner,n);binding[n]=fn.__globals__ is new.runner.__dict__;assert binding[n]
assert new.runner.method.update is new.revision.update
assert new.prior.method.update is not new.revision.update
assert new.runner.POLICY!=new.prior.POLICY
# Persistent completion and access chain.
cm=read(JOB/'completion.json');bv=read(JOB/'before_valid.json');access=read(JOB/'access.json');resource=read(JOB/'resource.json')
assert access==cm['access']=={'train':1,'valid':1,'heldout':0,'owners':0}
assert bv['access']=={'train':1,'valid':0,'heldout':0,'owners':0}
assert bv['points']==9 and bv['status']=='PASS_COMPLETE_BLIND_GATE'
verify(JOB,bv['manifest']);verify(JOB,cm['evaluation']);verify(JOB/'evaluation',ev['collected'])
assert cm['status']=='COMPLETE_RELATION_FIXED_POINT' and cm['worth_matched_replay'] is False
assert ev['status']=='STATISTICS_COMPLETE_REQUIRES_VALID_COMPLETION'
assert mf['status']=='COMPLETE_RELATION_2592_UPDATES_9_ENDPOINTS_VALID_BLIND'
assert cm['physical_updates']==mf['physical_updates']==2592 and mf['gradient_group_presentations']==4320
assert not list(JOB.glob('failure.json')) and not list(JOB.glob('recovery*.json'))
lim={'elapsed_seconds':86400,'peak_observed_bytes':24*2**30,'peak_cuda_reserved_bytes':28*2**30,'peak_rss_bytes':64*2**30}
for k,v in lim.items():assert 0<=cm['budget'][k]<=resource[k]<=v,(k,cm['budget'][k],resource[k])
wrapper=dict(x.split('=',1) for x in (RR/'job.wrapper.txt').read_text().splitlines())
start=datetime.datetime.fromisoformat(wrapper['started_at']);end=datetime.datetime.fromisoformat(wrapper['ended_at'])
assert int((end-start).total_seconds())==int(wrapper['total_wall_seconds'])==13808
assert wrapper['exit_code']=='0';assert datetime.datetime.fromisoformat(gate['issued_at'])<start
assert 13808-resource['elapsed_seconds']>=0 and 13808-resource['elapsed_seconds']<2
startup=read(IN/'reports/seller_alias_continual/20261006/relation_revision_execution/20261006_135937/observation/startup_environment.json')
assert startup['formal_labels_read'] is False and startup['formal_sources']==31
assert startup['resources']['compute_apps']==ex['resources']['compute_apps']==''
assert ex['resources']['project_jobs']==[]
assert ex['resources']['host_available_bytes']>=gate['runtime']['minimum_free_host_bytes']
# Uploaded inventory only: model/cache bodies intentionally NOT available.
inv=read(RR/'inventory.json');assert len(inv['files'])==168
for rec in inv['files']:verify(RR,rec)
part=read(verify(RUN,mf['partition']));allids=[]
for role,n in [('fit',48),('calibration',12),('development',20)]:
 assert len(part[role])==3*n and all(sum(x['domain']==d for x in part[role])==n for d in 'ABC')
 allids.extend(x['group_uid'] for x in part[role])
assert len(set(allids))==len(allids)==240
assert [x['group_uid'] for x in part['development']]==co['group_ids']
assert [x['domain'] for x in part['development']]==co['domains']
fit_domain={x['group_uid']:x['domain'] for x in part['fit']}
def seed(base,*parts):
 b=(json.dumps([base,*parts],ensure_ascii=False,sort_keys=True,separators=(',',':'))+'\n').encode()
 return int.from_bytes(hashlib.sha256(b).digest()[:8],'big')%(2**63-1)
state={};update_count=history_count=0;errors={'B_float32':0.,'QplusR_float32':0.,'total':0.,'lr':0.}
for order in ('ABC','BCA','CAB'):
 members=[];seen=0;rrng=random.Random(seed(20260930,order,'retention'))
 for stage in (1,2,3):
  name=f'{order}_relation_revision_stage{stage}'
  pt=read(verify(RUN,mf['points'][name]));tr=read(verify(RUN,mf['training'][name]))
  fit=sorted(u for u,d in fit_domain.items() if d==order[stage-1]);rng=random.Random(seed(20260918,order,stage,'current'));schedule=[]
  for epoch in range(6):tmp=fit.copy();rng.shuffle(tmp);schedule.extend(tmp)
  assert tr['current_ids']==schedule and len(tr['updates'])==288
  assert set(tr['current_ids'])==set(fit) and set(collections.Counter(tr['current_ids']).values())=={6}
  drng=random.Random(seed(20260930,order,stage,'history_draws'));old_members=members.copy()
  expected=[members[drng.randrange(6)] for _ in range(288)] if stage>1 else [None]*288
  assert tr['history_ids']==expected
  assert set(fit).isdisjoint(members)
  for i,u in enumerate(tr['updates'],1):
   assert u['stage']==stage and u['step']==i and u['adam_step']==(stage-1)*288+i
   assert all(np.isfinite(v) for v in u.values())
   b=float(np.float32(np.float32(u['current_bce'])+np.float32(u['current_rank']))+np.float32(.5)*np.float32(u['current_hard']))
   h=u['history_compressed']+float(np.float32(.1)*np.float32(u['history_rank']))
   errors['B_float32']=max(errors['B_float32'],abs(b-u['current_total']))
   errors['QplusR_float32']=max(errors['QplusR_float32'],abs(h-u['history_total']))
   errors['total']=max(errors['total'],abs(u['current_total']+u['history_total']-u['total']))
   lr=1e-5*(i/29 if i<=29 else (288-i)/259)
   errors['lr']=max(errors['lr'],abs(lr-u['encoder_lr']))
   assert u['history_rank_weight']==.1 and u['head_lr']==.001
   if stage==1:assert u['history_compressed']==u['history_rank']==u['history_total']==0
   update_count+=1;history_count+=stage>1
  assert tr['adam_step']==pt['adam_step']==stage*288 and pt['full_restore_verified'] is True
  if stage<3:
   for uid in fit:
    seen+=1;k=len(members) if len(members)<6 else rrng.randrange(seen)
    if k<6:
     if k==len(members):members.append(uid)
     else:members[k]=uid
   ret=pt['retention']; assert ret['old_count']==(stage-1)*48 and ret['new_count']==stage*48
   assert ret['old_members']==old_members and ret['new_members']==members and ret['reservoir_seen']==seen
   assert ret['h_min_eigenvalue']>=-ret['h_psd_tolerance']
   assert ('transport_singular_values' in ret)==(stage==2)
  else:assert pt['retention'] is None
  assert pt['memory_summary']=={'count':seen,'seen':seen,'stage':min(stage,2),'members':members}
  assert pt['memory_bytes']==pt['memory']['bytes']<=1048576
  assert pt['intermediate_deleted_bytes']==pt['full_state']['bytes']
  cal=[x['group_uid'] for x in part['calibration'] if x['domain']==order[stage-1]];assert pt['calibration_ids']==cal
  mapping=read(verify(RUN,pt['mapping']));assert mapping['status']=='PASS_CALIBRATION_FIT' and mapping['optimizer_success']
  assert mapping['group_count']==12 and mapping['pair_count']==4536 and mapping['positive_count']==240
  assert mapping['a']==pt['maps']['stage']['a'] and mapping['b']==pt['maps']['stage']['b']
  assert mapping['a']>0 and mapping['final_nll']<=mapping['initial_nll']
  if stage==1:first_map=pt['maps']['first'];assert first_map==pt['maps']['stage']
  else:assert pt['maps']['first']==first_map
  for kind,key in [('retained_model_files','model'),('retained_memory_files','memory')]:
   rec=next(x for x in inv[kind] if Path(x['path']).name==Path(pt[key]['path']).name)
   assert rec['bytes']==pt[key]['bytes'] and rec['sha256']==pt[key]['sha256']
  vals=np.array([u['gradient_norm'] for u in tr['updates']])
  state[name]={'members_by_domain':dict(collections.Counter(fit_domain[u] for u in members)),
   'memory_bytes':pt['memory_bytes'],'adam_step':pt['adam_step'],'calibration':{k:mapping[k] for k in ('a','b','initial_nll','final_nll','raw_brier','calibrated_brier','projected_gradient_max')},
   'gradient_min':float(vals.min()),'gradient_median':float(np.median(vals)),'gradient_max':float(vals.max()),'clipped_steps':int((vals>1).sum()),
   'first_update':tr['updates'][0],'last_update':tr['updates'][-1],'retention':pt['retention']}
assert update_count==2592 and history_count==1728
assert errors['B_float32']==errors['QplusR_float32']==errors['total']==errors['lr']==0,errors
# Console is independently consistent with a single monotonically progressing job.
logs=[]
for line in (RR/'job.console.txt').read_text().splitlines():
 try:j=json.loads(line)
 except json.JSONDecodeError:continue
 if isinstance(j,dict) and {'order','stage','step'}<=j.keys():logs.append(j)
assert len(logs)==108,len(logs)
expected=[(o,s,t) for o in ('ABC','BCA','CAB') for s in (1,2,3) for t in range(24,289,24)]
assert [(x['order'],x['stage'],x['step']) for x in logs]==expected
elapsed=[x['elapsed_seconds'] for x in logs];assert all(b>a for a,b in zip(elapsed,elapsed[1:]))
# Reference qualification receipts are historical, not a new baseline run.
reference_job=IN/'reports/seller_alias_continual/20261004/logit_low_result/job'
reference_receipts={k:read(reference_job/(k+'.json')) for k in ('before_valid','access','completion')}
assert reference_receipts['before_valid']['label_parses']=={'train':1,'valid':0,'heldout':0,'owners':0}
assert reference_receipts['access']==reference_receipts['completion']['label_parses']=={'train':1,'valid':1,'heldout':0,'owners':0}
assert reference_receipts['completion']['physical_updates']==1728
assert reference_receipts['completion']['status']=='COMPLETE_LOGIT_WEIGHT_DEVELOPMENT_COMPARISON'
result={'scope':'Frozen source/bindings and submitted execution records; no label parser, training, BGE or model/cache-body loading.',
 'source_count':31,'matching_source_lists':6,'prior_review_source_bytes_identical':True,'native_scientific_source_count':len(native['scientific_sources']),
 'actual_module_global_binding':binding,'inventory_payloads_verified':168,
 'formal_updates':update_count,'historical_updates':history_count,'gradient_group_presentations':update_count+history_count,
 'log_combination_max_errors':errors,'console_progress_records':len(logs),'access':access,'before_valid':bv,
 'reference_qualification_receipts':reference_receipts,'completion':cm,'resource':resource,'wrapper':wrapper,'seconds_of_unused_wall_cap':86400-13808,
 'model_record_total_bytes':sum(x['bytes'] for x in inv['retained_model_files']),
 'memory_record_min_bytes':min(x['bytes'] for x in inv['retained_memory_files']),
 'memory_record_max_bytes':max(x['bytes'] for x in inv['retained_memory_files']),
 'missing_bodies_not_rehashed':{'models':9,'memory':9},'startup_environment':startup,'points':state,
 'verified_available_records':verified}
(OUT/'execution_evidence.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k not in ('points','verified_available_records','startup_environment')},ensure_ascii=False,indent=2))
