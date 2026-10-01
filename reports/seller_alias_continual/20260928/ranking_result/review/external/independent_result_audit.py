"""Independent saved-output audit. Standard library + NumPy ONLY.

No project module is imported. No label reconstruction, raw text, model loading,
training, threshold refitting, or new evaluation is performed. The bootstrap
conditions on the existing 3 paired models and saved group metric matrices.
"""
from __future__ import annotations
import argparse, collections, csv, datetime, hashlib, json, math, os, platform, random, re, sys
from pathlib import Path
import numpy as np

COLS=('average_precision','trapezoidal_pr_auc','roc_auc','recall_at_fpr_1pct','brier','log_loss','precision','recall','f1','specificity','balanced_accuracy','mcc','map','mrr','recall_at_1','recall_at_3','recall_at_5','recall_at_10','ndcg_at_1','ndcg_at_3','ndcg_at_5','ndcg_at_10')
SEEDS=('s0','s1','s2'); ARMS=('d','schedule','hard'); RUNS=tuple(f'{s}_{a}' for s in SEEDS for a in ARMS)
RATE_KEYS=('tp','fp','fn','tn','fpr','recall','precision','f1')
CLS=('precision','recall','f1','specificity','balanced_accuracy','mcc')
COMPS=(('hard','d'),('schedule','d'),('hard','schedule'))
NUM=0; MAXDIFF=0.; READ=set(); BLIND=0; METRICS=0; SCOREREADS=set(); NOTES=[]

def canonical(x):return (json.dumps(x,ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n').encode()
def digest(b):return hashlib.sha256(b).hexdigest()
def eq(a,b,name,atol=1e-12):
 global NUM,MAXDIFF
 aa=np.asarray(a,dtype=np.float64);bb=np.asarray(b,dtype=np.float64)
 if aa.shape!=bb.shape or not np.isfinite(aa).all() or not np.isfinite(bb).all():raise AssertionError(('shape/finiteness',name,aa.shape,bb.shape))
 diff=float(np.max(np.abs(aa-bb))) if aa.size else 0.
 if diff>atol:raise AssertionError(('difference',name,diff))
 NUM+=aa.size;MAXDIFF=max(MAXDIFF,diff)
def mapping(actual,expected,name):
 for k,v in expected.items():
  if isinstance(v,dict):mapping(actual[k],v,name+'/'+k)
  else:eq(actual[k],v,name+'/'+k)
def rates(c):
 """Explicit confusion algebra. Last axis: TP, FP, FN, TN; arbitrary batch dims."""
 c=np.asarray(c,dtype=np.float64);tp,fp,fn,tn=np.moveaxis(c,-1,0)
 def div(a,b):return np.divide(a,b,out=np.zeros_like(a,dtype=np.float64),where=b!=0)
 return dict(tp=tp,fp=fp,fn=fn,tn=tn,fpr=div(fp,fp+tn),recall=div(tp,tp+fn),precision=div(tp,tp+fp),f1=div(2*tp,2*tp+fp+fn),specificity=div(tn,tn+fp),balanced_accuracy=(div(tp,tp+fn)+div(tn,tn+fp))/2,mcc=div(tp*tn-fp*fn,np.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))))
def counts(raw):
 if isinstance(raw[0],dict):raw=[[x[k] for k in ('tp','fp','fn','tn')] for x in raw]
 a=np.asarray(raw)
 assert a.dtype.kind in 'iu' and a.ndim==2 and a.shape[1]==4 and (a>=0).all()
 assert (a[:,0]+a[:,2]==20).all() and (a[:,1]+a[:,3]==358).all()
 return a

def main(root:Path,out:Path):
 global BLIND,METRICS
 if out.exists():raise FileExistsError(out)
 inventory=json.loads((root/'source_inventory.json').read_text())
 allowed={r['path']:r for r in inventory['files']}
 def loadbytes(path):
  path=path.resolve();n=path.relative_to(root).as_posix()
  assert n in allowed and path.suffix not in ('.pt','.bin','.safetensors','.jsonl'),n
  content=path.read_bytes();rec=allowed[n]
  assert len(content)==rec['bytes'] and digest(content)==rec['sha256'],n
  READ.add(n);return content
 def read(p):return json.loads(loadbytes(p))
 def record(base,rec):
  p=(base/rec['path']).resolve();content=loadbytes(p)
  assert len(content)==rec['bytes'] and digest(content)==rec['sha256'],str(p)
  return p
 def array(base,rec,shape,dtype):
  import io
  p=record(base,rec);a=np.load(io.BytesIO(loadbytes(p)),allow_pickle=False)
  assert a.shape==shape and a.dtype==np.dtype(dtype) and np.isfinite(a).all(),str(p)
  return a
 job=root/'reports/seller_alias_continual/20260927/ranking_execution/20260927_114646/job'
 res=root/'reports/seller_alias_continual/20260928/ranking_result'
 policy=read(root/'schema/step28_alias_ranking_policy.json');base=read(root/'schema/step28_chinese_base_policy.json')
 manifest=read(job/'run/manifest.json');evaluation=read(job/'evaluation/evaluation.json');collected=read(job/'evaluation/collected.json');complete=read(job/'completion.json')
 auth=read(record(root,manifest['authorization']));cpu=read(record(root,manifest['native_verification']))
 partition=read(record(job/'run',manifest['partition']));preparse=read(job/'evaluation/preparse_verification.json')
 assert policy==manifest['policy']==evaluation['policy']==collected['policy']
 assert tuple(policy['runs'])==RUNS and set(manifest['runs'])==set(evaluation['runs'])==set(collected['runs'])==set(RUNS)
 assert auth['status']=='AUTHORIZED_RANKING_TRAIN_AND_VALID' and auth['runs']==list(RUNS) and auth['physical_updates']==7776
 assert not auth['test_access'] and not auth['owners_access'] and not auth['automatic_retry']
 assert auth['policy_sha256']==digest(loadbytes(root/'schema/step28_alias_ranking_policy.json'))
 assert auth['source_files']==manifest['source_files']==preparse['sources']==cpu['source_files'] and len(manifest['source_files'])==18
 for r in manifest['source_files']:record(root,r)
 assert complete['formal_updates']==manifest['physical_updates']==7776
 assert complete['label_parses']=={'train':1,'development':1,'heldout':0,'owners':0}
 assert read(job/'run/access.json')=={'train_parse_attempts':1,'development':0,'heldout':0,'owners':0}
 assert read(job/'evaluation/access.json')=={'development_parse_attempts':1,'train':0,'heldout':0,'owners':0}
 assert loadbytes(job/'exit_status.txt').strip()==b'0' and not any(p.name=='failure.json' for p in job.rglob('*'))
 assert preparse['manifest_sha256']==read(job/'run/completion.json')['manifest_sha256']==digest(loadbytes(job/'run/manifest.json'))
 record(root,evaluation['provenance']['training_manifest']);record(job/'evaluation',evaluation['provenance']['preparse_verification'])
 assert list(COLS)==evaluation['columns']==collected['columns']
 assert evaluation['group_ids']==collected['group_ids']==[r['group_uid'] for r in partition['development']]
 assert evaluation['domains']==collected['domains']==[r['domain'] for r in partition['development']]
 ids=[r['group_uid'] for role in partition.values() for r in role];assert len(ids)==len(set(ids))==240
 rows={d:np.flatnonzero(np.asarray(evaluation['domains'])==d) for d in 'ABC'}
 for role,n in [('fit',48),('calibration',12),('development',20)]:
  assert collections.Counter(r['domain'] for r in partition[role])=={d:n for d in 'ABC'}
 for d in 'ABC':
  all_train=[r['group_uid'] for role in ('fit','calibration') for r in partition[role] if r['domain']==d]
  chosen=set(sorted(all_train,key=lambda u:digest(canonical([base['partition']['seed'],d,u])))[:48])
  assert chosen=={r['group_uid'] for r in partition['fit'] if r['domain']==d}
 # Draw sequentially, independently of both project batch-index and frequency-weight routines.
 rng=np.random.default_rng(20260927);draw=np.empty((5000,3,20),dtype=np.int64)
 for b in range(5000):
  for j in range(3):draw[b,j]=rng.integers(0,20,size=20)
 idx=np.concatenate([rows[d][draw[:,j]] for j,d in enumerate('ABC')],axis=1)
 assert idx.shape==(5000,60)
 for j,d in enumerate('ABC'):assert np.isin(idx[:,j*20:(j+1)*20],rows[d]).all()
 def boot(v):return np.asarray(v,dtype=np.float64)[idx].sum(axis=1)/60.
 def ci(v):return np.quantile(v,[.025,.975],axis=0,method='linear')
 def sampled_counts(c):return {**{d:c[rows[d][draw[:,j]]].sum(axis=1) for j,d in enumerate('ABC')},'pooled':c[idx].sum(axis=1)}
 def audit_counts(c,score,threshold,name,matrix=None):
  global BLIND
  # Python scalar threshold comparisons avoid accidental float32 threshold rounding.
  pred=np.asarray([sum(float(v)>=float(threshold) for v in row) for row in score])
  eq(pred,c[:,0]+c[:,1],name+'/predicted_positive_total',atol=0);BLIND+=len(c)
  if matrix is not None:
   rr=rates(c)
   eq(matrix[:,[COLS.index(k) for k in CLS]],np.column_stack([rr[k] for k in CLS]),name+'/six_confusion_metrics')
 matrices={};automatic={};fixed={};auto_boot={};point_rows=[];trajectory=[];training={};identity_rows=[];max_loss=0.;loss_rows=0;all_scores=set();all_matrices=set();epoch_records=[];checkpoint_bytes=[];logs={r:[] for r in RUNS}
 for line in loadbytes(job/'train.log').decode().splitlines():
  if line.startswith('{'):
   r=json.loads(line)
   if r.get('event')=='updates':logs[r['run_id']].append(r)
 assert sum(len(v) for v in logs.values())==324
 for run_id in RUNS:
  seed,arm=run_id.split('_');aroot=job/'run'/run_id
  a=read(record(job/'run',manifest['runs'][run_id]['manifest']))
  assert a['run_id']==run_id and a['updates']==manifest['runs'][run_id]['updates']==864 and a['label_parses']==0
  assert a['parameter_count']==a['trainable_parameter_count']==326571265
  cfg=dict(base);cfg.update(policy['seeds'][seed]);cfg['runtime']=policy['runtime'];assert cfg==a['reference_config']
  assert a['preflight']==manifest['preflights'][run_id] and a['preflight']['updates']==a['preflight']['label_reads']==0
  assert a['fit_group_ids']==[r['group_uid'] for r in partition['fit']]
  assert a['calibration_group_ids']==[r['group_uid'] for r in partition['calibration']]
  stream=int.from_bytes(hashlib.sha256(canonical([cfg['schedule_seed'],'BASE_JOINT','current'])).digest()[:8],'big')%(2**63-1)
  rr=random.Random(stream);schedule=[]
  for epoch in range(6):
   tmp=sorted(a['fit_group_ids']);rr.shuffle(tmp);schedule.extend(tmp)
  assert digest(canonical(schedule))==a['group_schedule_sha256'] and stream==a['dropout_stream']
  assert collections.Counter(schedule)=={g:6 for g in a['fit_group_ids']}
  u=array(aroot,a['update_log'],(864,6),'float64')
  lr=np.asarray([2e-5 if arm=='d' else 1e-5*(t/87 if t<=87 else (864-t)/777) for t in range(1,865)])
  eq(u[:,4],lr,run_id+'/lr',0);eq(u[:,5],np.full(864,.001),run_id+'/head_lr',0)
  assert np.all(u[:,:4]>=0) and (arm=='hard' or np.all(u[:,2]==0))
  assert a['encoder_positive_lr_updates']==int(np.count_nonzero(lr>0))
  weight=.5 if arm=='hard' else 0.
  total64=u[:,0]+u[:,1]+weight*u[:,2];max_loss=max(max_loss,float(abs(total64-u[:,3]).max()))
  # Reconstruct the actual float32 arithmetic tree, not just a permissive tolerance.
  first=(u[:,0].astype('float32')+u[:,1].astype('float32')).astype('float32')
  expected=(first+np.float32(weight)*u[:,2].astype('float32')).astype('float32')
  assert np.array_equal(expected.astype('float64'),u[:,3]),run_id+'/float32 loss composition'
  loss_rows+=len(u);losses=u[:,:4].reshape(6,144,4).mean(axis=1)
  assert len(a['training'])==2
  for segment_i,segment in enumerate(a['training']):
   assert (segment['start'],segment['stop'],segment['updates'])==(432*segment_i,432*(segment_i+1),432)
   expected_obs={str(t) for t in (1,2,87,88,433) if segment['start']<t<=segment['stop']}
   assert set(segment['observations'])==expected_obs
   for obs in segment['observations'].values():
    assert obs=={k:{'finite_nonzero_gradient':True,'parameters_changed':True} for k in ('encoder','head')}
   for j,k in enumerate(('bce','rank','hard','total')):eq(segment['mean_losses_by_epoch'][k],losses[3*segment_i:3*(segment_i+1),j],run_id+'/loss_summary/'+k)
  assert [r['completed'] for r in logs[run_id]]==list(range(24,865,24))
  for r in logs[run_id]:eq(r['encoder_lr'],lr[r['completed']-1],run_id+'/logged_lr',0)
  epoch_records.extend({'run_id':run_id,'epoch':i+1,**dict(zip(('bce','rank','hard','total'),row.tolist()))} for i,row in enumerate(losses))
  training[run_id]={'initial_state':a['preflight']['initial_state_sha256'],'schedule_sha256':a['group_schedule_sha256'],'dropout_stream':stream,'lr_sum':float(lr.sum()),'positive_encoder_lr_steps':int((lr>0).sum()),'resources':a['resources'],'training_seconds':a['formal_training_seconds'],'observed_steps':[1,2,87,88,433]}
  for ep in ('3','6'):
   p=a['points'][ep];saved=evaluation['runs'][run_id]['points'][ep]
   assert saved==collected['runs'][run_id]['points'][ep]
   md=p['metadata'];step=int(ep)*144
   assert p['run_id']==md['run_id']==run_id and p['epoch']==md['epoch']==int(ep) and md['completed_updates']==step
   assert md['reference_config']==cfg and md['policy_sha256']==auth['policy_sha256']
   eq(md['optimizer_lrs'],[lr[step-1],.001],run_id+'/saved_lrs',0)
   assert md['next_encoder_lr']==(lr[step] if step<864 else None)
   assert p['full_model_and_adam_reloaded'] and p['model']['actual_reload_verified']
   assert p['model']['path']==f'models/epoch{ep}.pt'
   checkpoint_bytes.append(p['checkpoint']['bytes'])
   identity_rows.append({'path':(aroot/p['model']['path']).relative_to(root).as_posix(),'bytes':p['model']['bytes'],'sha256':p['model']['sha256'],'run_id':run_id,'epoch':int(ep),'state_sha256':p['model']['state_sha256'],'model_state_sha256':p['model_state_sha256']})
   for role,n in (('development',60),('fit',144),('calibration',36)):
    rec=p['scores'][role];assert rec['path']==f'scores/epoch{ep}_{role}.npy'
    score=array(aroot,rec,(n,378),'float32');all_scores.add((run_id,ep,role))
    if role=='development':
     mrec=saved['file'];mat=array(job/'evaluation',mrec,(n,22),'float64');c=counts(saved['counts_at_logit_zero']);matrices[run_id,ep]=mat
     assert mrec['path']==f'{run_id}/epoch{ep}_metrics.npy'
     for scope,index in {'mean':np.arange(60),**rows}.items():
      mm=mat[index].mean(axis=0);mapping(saved['mean'] if scope=='mean' else saved['by_domain'][scope],dict(zip(COLS,mm)),run_id+ep+'/'+scope)
      point_rows.extend({'run_id':run_id,'epoch':ep,'scope':scope,'metric':k,'value':float(v)} for k,v in zip(COLS,mm))
     for scope,index in {'pooled':np.arange(60),**rows}.items():
      cr=rates(c[index].sum(axis=0));mapping(saved['fixed_classification']['pooled'] if scope=='pooled' else saved['fixed_classification']['by_domain'][scope],{k:cr[k] for k in RATE_KEYS},run_id+'/fixed/'+scope)
     fixed[run_id,ep]=c
    else:
     mrec=p['train_metrics'][role]['file'];mat=array(aroot,mrec,(n,22),'float64');c=counts(p['train_metrics'][role]['counts'])
     assert mrec['path']==f'scores/epoch{ep}_{role}_metrics.npy'
    METRICS+=1;all_matrices.add((run_id,ep,role));audit_counts(c,score,0.,run_id+ep+role,mat)
    trajectory.append({'run_id':run_id,'epoch':ep,'split':'valid' if role=='development' else role,**dict(zip(COLS,mat.mean(axis=0).tolist()))})
  cal=read(record(aroot,a['calibration']));auto=evaluation['runs'][run_id]['automatic_classification']
  assert cal['source_role']=='train_calibration_only' and cal['epoch']==6 and cal['prediction']=='float64(logit)>=float64(threshold)'
  assert cal['model_state_sha256']==a['points']['6']['model_state_sha256'] and cal['score_sha256']==a['points']['6']['scores']['calibration']['sha256']
  assert cal['threshold']==auto['threshold']==max(v['threshold'] for v in cal['bounds'].values())
  ccal=counts(cal['counts_by_group']);cauto=counts(auto['counts_by_group']);automatic[run_id]=cauto
  for role,c in [('calibration',ccal),('development',cauto)]:
   score=array(aroot,a['points']['6']['scores'][role],(len(c),378),'float32');audit_counts(c,score,cal['threshold'],run_id+'/threshold/'+role)
  for d in 'ABC':
   which=[i for i,r in enumerate(partition['calibration']) if r['domain']==d]
   eq(ccal[which].sum(axis=0),cal['counts_by_domain'][d],run_id+'/cal_domain/'+d,0)
   bound=cal['bounds'][d];assert bound['negative_pairs']==4296 and bound['allowed_false_positives']==4
   assert bound['threshold']==np.nextafter(np.float64(bound['next_negative_logit']),np.inf) and ccal[which,1].sum()<=4
  boots=sampled_counts(cauto);auto_boot[run_id]={scope:rates(v) for scope,v in boots.items()}
  for scope,index in {'pooled':np.arange(60),**rows}.items():
   ar=auto['pooled'] if scope=='pooled' else auto['by_domain'][scope];cr=rates(cauto[index].sum(axis=0))
   expected={k:cr[k] for k in RATE_KEYS};expected['conditional_95pct_intervals']={k:ci(auto_boot[run_id][scope][k]) for k in ('fpr','recall','precision')}
   mapping(ar,expected,run_id+'/automatic/'+scope)
   if scope!='pooled':assert ar['passes_point_gates']==bool(cr['fp']*1000<=7160 and cr['tp']*2>=400)
  assert auto['all_domains_pass']==all(auto['by_domain'][d]['passes_point_gates'] for d in 'ABC')
 for seed in SEEDS:
  for k in ('initial_state','schedule_sha256','dropout_stream'):
   assert len({training[f'{seed}_{a}'][k] for a in ARMS})==1,(seed,k)
 assert len(all_scores)==54 and len(all_matrices)==54 and BLIND==5184 and loss_rows==7776
 # Reconcile four independent stages of the eighteen weight-file records, without opening any weight.
 w=read(res/'weight_inventory.json');retain=read(res/'weight_retention.json');post=read(res/'retention_verification.json')
 identity={r['path']:r for r in identity_rows};before={r['path']:r for r in preparse['files']};fresh={r['path']:r for r in w['files']};after={r['path']:r for r in post['files']}
 assert len(identity)==len(before)==len(fresh)==len(after)==18 and identity.keys()==before.keys()==fresh.keys()==after.keys()
 for n,r in identity.items():
  for k in ('bytes','sha256'):assert r[k]==before[n][k]==fresh[n][k],(n,k)
  assert r['bytes']==after[n]['bytes'] and after[n]['retained']
 total_weight=sum(r['bytes'] for r in identity.values());assert total_weight==23516221872==w['total_bytes']==retain['total_bytes']==post['total_bytes']
 assert w['deletion_performed'] is False and retain['deleted_files']==post['deleted_files']==0
 assert post['nonweight_files_unchanged']==161
 assert datetime.datetime.fromisoformat(w['checked_at'])<datetime.datetime.fromisoformat(retain['recorded_at'])<datetime.datetime.fromisoformat(post['observed_at'])
 comparisons={};perseed={};comparison_rows=[];perseed_rows=[];domain_rows=[];primary_boot=None
 for cand,ref in COMPS:
  name=cand+'_minus_'+ref
  delta=np.asarray([matrices[s+'_'+cand,'6']-matrices[s+'_'+ref,'6'] for s in SEEDS]);average=delta.sum(axis=0)/3
  bs=boot(average);limits=ci(bs);seedlimits=np.asarray([ci(boot(delta[i])) for i in range(3)])
  if name=='hard_minus_d':primary_boot=bs
  comparisons[name]={};perseed[name]={}
  for j,k in enumerate(COLS):
   values={'mean':float(average[:,j].mean()),'per_seed':delta[:,:,j].mean(axis=1).tolist(),'by_domain':{d:float(average[index,j].mean()) for d,index in rows.items()},'conditional_95pct_interval':limits[:,j].tolist()}
   mapping(evaluation['comparisons'][name]['metrics'][k],values,name+'/'+k)
   v={**values,'reference_mean':float(np.asarray([matrices[s+'_'+ref,'6'][:,j] for s in SEEDS]).mean()),'candidate_mean':float(np.asarray([matrices[s+'_'+cand,'6'][:,j] for s in SEEDS]).mean())}
   comparisons[name][k]=v;comparison_rows.append({'comparison':name,'metric':k,'reference':v['reference_mean'],'candidate':v['candidate_mean'],'delta':values['mean'],'lower':limits[0,j],'upper':limits[1,j]})
   for d,index in rows.items():domain_rows.append({'comparison':name,'domain':d,'metric':k,'delta':values['by_domain'][d]})
   for i,s in enumerate(SEEDS):
    vv={'mean':values['per_seed'][i],'conditional_95pct_interval':seedlimits[i,:,j].tolist()}
    mapping(evaluation['per_seed_comparisons'][name][s]['metrics'][k],vv,name+s+k)
    perseed[name].setdefault(s,{})[k]=vv
    perseed_rows.append({'comparison':name,'seed':s,'metric':k,'delta':vv['mean'],'lower':vv['conditional_95pct_interval'][0],'upper':vv['conditional_95pct_interval'][1]})
  for s in SEEDS:
   c=automatic[s+'_'+cand];r=automatic[s+'_'+ref]
   for scope,index in {'pooled':np.arange(60),**rows}.items():
    cpoint,rpoint=rates(c[index].sum(axis=0)),rates(r[index].sum(axis=0))
    for k in ('fpr','recall','precision'):
     vv={'difference':cpoint[k]-rpoint[k],'conditional_95pct_interval':ci(auto_boot[s+'_'+cand][scope][k]-auto_boot[s+'_'+ref][scope][k])}
     mapping(evaluation['per_seed_comparisons'][name][s]['automatic'][k][scope],vv,name+s+'/automatic/'+k+scope)
 orig_boot=np.load(res/'analysis/primary_bootstrap.npy',allow_pickle=False);eq(primary_boot,orig_boot,'all_110000_primary_bootstrap_values')
 pr=comparisons['hard_minus_d'];mp=pr['map'];r5=pr['recall_at_5']
 checks={'map_minimum_observed_gain':mp['mean']>=.01,'map_interval_above_zero':mp['conditional_95pct_interval'][0]>0,'map_improves_each_seed':all(v>0 for v in mp['per_seed']),'recall_at_5_mean_strictly_improves':r5['mean']>0,'recall_at_5_s0_strictly_improves':r5['per_seed'][0]>0}
 for k,sgn in [('average_precision',1),('roc_auc',1),('brier',-1),('log_loss',-1)]:
  checks[k+'_mean_non_degradation']=sgn*pr[k]['mean']>=0;checks[k+'_s0_non_degradation']=sgn*pr[k]['per_seed'][0]>=0
 assert len(checks)==13 and checks==evaluation['acceptance']['checks']==complete['acceptance']['checks']
 assert not all(checks.values()) and sum(checks.values())==9 and not evaluation['acceptance']['passed']
 assert {k for k,v in checks.items() if not v}==set(evaluation['acceptance']['failed'])
 # Extra descriptive decomposition uses only already-saved aggregate confusion counts.
 fixed_table=[]
 for run_id in RUNS:
  cc=fixed[run_id,'6'];rr=rates(cc.sum(axis=0));mr=rates(cc)
  fixed_table.append({'run_id':run_id,**{k:float(v) for k,v in rr.items()},'macro_precision':float(mr['precision'].mean()),'macro_f1':float(mr['f1'].mean()),'groups_without_positive_prediction':int(((cc[:,0]+cc[:,1])==0).sum())})
 for arm in ARMS:
  cc=np.concatenate([fixed[s+'_'+arm,'6'] for s in SEEDS]);rr=rates(cc.sum(axis=0));mr=rates(cc)
  fixed_table.append({'run_id':'all3_'+arm,**{k:float(v) for k,v in rr.items()},'macro_precision':float(mr['precision'].mean()),'macro_f1':float(mr['f1'].mean()),'groups_without_positive_prediction':int(((cc[:,0]+cc[:,1])==0).sum())})
 started=datetime.datetime.fromisoformat(loadbytes(job/'started.txt').decode().strip());finished=datetime.datetime.fromisoformat(loadbytes(job/'finished.txt').decode().strip())
 assert started>=datetime.datetime.fromisoformat(auth['recorded_at'])
 elapsed=(finished-started).total_seconds();assert elapsed<86400 and complete['budget']['elapsed_seconds']<86400
 job_bytes=sum(r['bytes'] for r in inventory['files'] if r['path'].startswith(job.relative_to(root).as_posix()+'/'))
 bound=total_weight+max(checkpoint_bytes)+job_bytes
 assert job_bytes==8894602 and bound==27436077439==post['conservative_job_coexistence_bytes'] and bound<policy['runtime']['maximum_output_bytes']
 out.mkdir(parents=True)
 outputs=[('comparisons_22.csv',comparison_rows),('per_seed_comparisons_22.csv',perseed_rows),('by_domain_comparisons_22.csv',domain_rows),('point_metrics_22.csv',point_rows),('trajectories_22.csv',trajectory),('epoch_losses.csv',epoch_records),('fixed_zero_confusion.csv',fixed_table),('weight_record_crosscheck.csv',identity_rows)]
 for filename,records in outputs:
  with (out/filename).open('w',newline='',encoding='utf8') as f:
   writer=csv.DictWriter(f,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
 np.save(out/'independent_primary_bootstrap.npy',primary_boot,allow_pickle=False)
 assert not any(x=='torch' or x.startswith('step28_') or x=='sentence_transformers' for x in sys.modules)
 result={'status':'PASS_INDEPENDENT_SAVED_RESULT_AUDIT','acceptance_passed':False,'checks_passed':9,'checks_total':13,
 'counts':{'numerical_values':NUM,'max_numeric_difference':MAXDIFF,'blind_positive_count_rows':BLIND,'unique_score_arrays':len(all_scores),'metric_matrices':METRICS,'loss_rows':loss_rows,'max_loss_float64_roundoff':max_loss,'float32_loss_compositions_exact':True,'log_update_events':324,'weight_record_chains':18,'observed_update_steps':45,'module_observations':90},
 'versions':{'python':platform.python_version(),'numpy':np.__version__,'platform':platform.platform(),'cpu_affinity':sorted(os.sched_getaffinity(0)),'proc_threads':[x for x in Path('/proc/self/status').read_text().splitlines() if x.startswith('Threads:')]},
 'operations':{'formal_label_reads':0,'formal_text_reads':0,'model_loads':0,'training_updates':0,'project_modules_imported':False,'new_packages_installed':False,'remote_server_access':False},
 'bootstrap':{'replicates':5000,'seed':20260927,'unit':'whole group stratified by domain','fixed_paired_seeds_averaged_first':True,'group_draws':300000,'primary_values_compared':110000,'conditional_only':True},
 'acceptance':checks,'comparisons':comparisons,'perseed':perseed,'training':training,'fixed_zero':fixed_table,
 'resource':{'started':started.isoformat(),'finished':finished.isoformat(),'timestamp_seconds':elapsed,'budget':complete['budget'],'conservative_coexistence_bytes':bound,'weights_bytes':total_weight,'max_temporary_full_checkpoint_bytes':max(checkpoint_bytes),'job_small_bytes':job_bytes,'peak_allocated_bytes':max(v['resources']['peak_allocated_bytes'] for v in training.values()),'peak_reserved_bytes':max(v['resources']['peak_reserved_bytes'] for v in training.values())},
 'weight_retention':{'three_hash_record_stages_match':True,'after_instruction_metadata_match':True,'file_count':18,'total_bytes':total_weight,'deletions_recorded':0,'remote_file_bytes_independently_accessed':False},
 'files_read':sorted(READ),'scope':'No true-label metric recomputation. Saved matrices/counts/score thresholds only; actual GPU, Adam and remote file custody supported by submitted records + static execution paths.'}
 (out/'independent_results.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
 print(json.dumps({k:result[k] for k in ('status','acceptance_passed','checks_passed','checks_total','counts','versions','operations','resource')},ensure_ascii=False,indent=2))
 print('PRIMARY',json.dumps({k:pr[k] for k in ('map','recall_at_5','average_precision','roc_auc','brier','log_loss')},ensure_ascii=False,indent=2))

if __name__=='__main__':
 parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',type=Path,required=True);parser.add_argument('--out',type=Path,required=True);args=parser.parse_args()
 main(args.root.resolve(),args.out.resolve())
