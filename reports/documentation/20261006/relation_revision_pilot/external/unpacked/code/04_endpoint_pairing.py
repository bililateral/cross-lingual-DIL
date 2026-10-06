"""Independent saved-matrix ROUTING/STATISTICS fixture; NOT effect measurements.
Matrices contain distinct symbolic metric values, not MAP/AP recomputed from labels.
Counts are legal handwritten counts. No project labels/models/baseline are loaded.
"""
import os,sys,json,copy,hashlib
from pathlib import Path
from unittest import mock
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
R=Path(__file__).resolve().parents[1];C=R/'sources/current'
sys.dont_write_bytecode=True;sys.path[:0]=[str(C/'scripts'),str(C/'tests')]
import numpy as np
import step28_relation_revision_run as a
run=a.runner; ev=run.evaluation; d=run.data
root=R/'outputs/04_fixture'; root.mkdir(exist_ok=False)
cr=root/'candidate'; br=root/'baseline'; cr.mkdir();br.mkdir();(br/'reference').mkdir()
domains=list('ABC')*20; gids=[f'WEB_SYMBOLIC_{dom}_{i//3:02}' for i,dom in enumerate(domains)]
columns=list(run.parent.metrics.COLUMNS); rank_cols=set(ev.RANK_COLUMNS)
nonrank=[i for i in range(22) if i not in rank_cols]
counts=[dict(tp=10,fn=10,fp=20,tn=338) for _ in range(60)]
allarr={}; candidate_col=dict(group_ids=gids,domains=domains,metric_columns=columns,points={},source_files=a.sources(),status='ALL_28_RELATION_METRIC_COUNT_SETS_SAVED')
refs=[dict(group_ids=gids,domains=domains,metric_columns=columns,points={}) for _ in range(2)]
def save_role(folder,name,role,arr):
 p=folder/(name+'_'+role+'.npy');np.save(p,arr,allow_pickle=False)
 q=folder/(name+'_'+role+'_counts.json');d.write_json(q,counts)
 return {'matrix':d.record(p,folder),'counts':d.record(q,folder)}
for arm in ('candidate','baseline'):
 allarr[arm]={}
 for oi,o in enumerate(('ABC','BCA','CAB')):
  for s in (1,2,3):
   symbol=run.point_name(o,s) if arm=='candidate' else o+'_shared' if s==1 else f'{o}_logit_tenth_stage{s}'
   raw=np.empty((60,22),np.float64)
   for row,dom in enumerate(domains):
    for col in range(22):
     raw[row,col]=(0.26 if arm=='candidate' else 0.42)+(0.058 if arm=='candidate' else 0.021)*s+.006*oi+.004*'ABC'.index(dom)+.0003*(row//3)+.0001*col
   roles={'raw':raw,'stage-cal':raw.copy(),'first-cal':raw.copy()}
   roles['stage-cal'][:,nonrank]+=.017 if arm=='candidate' else .011
   roles['first-cal'][:,nonrank]+=.031 if arm=='candidate' else .023
   allarr[arm][(o,s)]=roles
   folder=cr if arm=='candidate' else br/'reference' if s==1 else br
   record={role:save_role(folder,symbol,role,arr) for role,arr in roles.items()}
   if arm=='candidate':candidate_col['points'][symbol]=record
   else:refs[1 if s==1 else 0]['points'][symbol]=record
candidate_col['points']['initial']={'raw':save_role(cr,'initial','raw',np.full((60,22),.1))}
d.write_json(cr/'collected.json',candidate_col)
for folder,col in ((br,refs[0]),(br/'reference',refs[1])):d.write_json(folder/'collected.json',col)
p=a.policy();p['reference']={'linux_root':str(br),'collections':[d.record(br/'collected.json',br),d.record(br/'reference/collected.json',br)]}
with mock.patch.object(run,'policy',return_value=p):
 result=run.finalize(cr)
assert result['status']=='STATISTICS_COMPLETE_REQUIRES_VALID_COMPLETION'
assert 'worth_matched_replay' not in result
# Independent direct endpoint algebra, indexed by actual domain not row block/order.
def independent_field(arm,role,endpoint):
 out=np.zeros((3,3,20,22))
 for oi,o in enumerate(('ABC','BCA','CAB')):
  def M(s,arrival):
   dom=o[arrival-1];rows=[i for i,dd in enumerate(domains) if dd==dom]
   vals=allarr[arm][(o,s)][role if role!='primary' else 'stage-cal'].copy()
   if role=='primary':vals[:,sorted(rank_cols)]=allarr[arm][(o,s)]['raw'][:,sorted(rank_cols)]
   v=np.zeros((3,20,22));v['ABC'.index(dom)]=vals[rows];return v
  if endpoint=='O':out[oi]=(M(3,1)+M(3,2))/2
  elif endpoint=='N':out[oi]=(M(2,2)+M(3,3))/2
  elif endpoint=='Z':out[oi]=M(3,3)
  elif endpoint=='F_first':out[oi]=M(1,1)-M(3,1)
  elif endpoint=='F':out[oi]=(M(1,1)-M(3,1)+M(2,2)-M(3,2))/2
  elif endpoint=='G':out[oi]=(M(2,2)-M(1,2)+M(3,3)-M(2,3))/2
  elif endpoint=='final_all':out[oi]=(M(3,1)+M(3,2)+M(3,3))/3
 if endpoint in ('F_first','F','G'):
  out[:,:,:,[columns.index('brier'),columns.index('log_loss')]]*=-1
 return out
rng=np.random.Generator(np.random.PCG64(20260930));draws=rng.integers(0,20,(5000,3,20))
assert np.array_equal(draws,ev.bootstrap_draws())
def independent_summary(field):
 per=field.sum(axis=1).mean(axis=1)
 mean=per.mean(axis=0)
 pooled=field.mean(axis=0)
 samples=np.zeros((5000,22))
 for domain in range(3): samples+=np.take(pooled[domain],draws[:,domain],axis=0).mean(axis=1)
 ci=np.quantile(samples,[.025,.975],axis=0,method='linear')
 return mean,per,ci
error=0.;compared=0;fingerprints={}
for arm,armkey in [('candidate','relation'),('baseline','logit0.1')]:
 for role in ('raw','stage-cal','first-cal','primary'):
  for endpoint in ('O','N','Z','F_first','F','G','final_all'):
   mean,per,ci=independent_summary(independent_field(arm,role,endpoint))
   for i,c in enumerate(columns):
    got=result['endpoints'][armkey][role][endpoint][c]
    pairs=[(got['mean'],mean[i]),*zip(got['per_order'].values(),per[:,i]),*zip(got['conditional_95pct_interval'],ci[:,i])]
    error=max(error,max(abs(float(x)-float(y)) for x,y in pairs));compared+=len(pairs)
for endpoint in ('O','N','Z','F_first','F','G','final_all'):
 mean,per,ci=independent_summary(independent_field('candidate','primary',endpoint)-independent_field('baseline','primary',endpoint))
 for i,c in enumerate(columns):
  got=result['delta'][endpoint][c]
  pairs=[(got['mean'],mean[i]),*zip(got['per_order'].values(),per[:,i]),*zip(got['conditional_95pct_interval'],ci[:,i])]
  error=max(error,max(abs(float(x)-float(y)) for x,y in pairs));compared+=len(pairs)
 fingerprints[endpoint]=result['delta'][endpoint]['map']
assert error<1e-12
# Verify mismatched paired IDs/domains/columns are rejected BEFORE metric aggregation.
reference_path=br/'collected.json';original=d.read_json(reference_path)
rejections={}
for key in ('group_ids','domains','metric_columns'):
 altered=copy.deepcopy(original);altered[key][0],altered[key][1]=altered[key][1],altered[key][0]
 d.write_json(reference_path,altered)
 pp=copy.deepcopy(p);pp['reference']['collections'][0]=d.record(reference_path,br)
 with mock.patch.object(run,'policy',return_value=pp):
  try:run.finalize(cr)
  except ValueError as exc:assert 'pairing' in str(exc);rejections[key]=True
  else:raise AssertionError(key)
d.write_json(reference_path,original)
# Observation rule is intentionally not equivalent to a positive interval rule.
base={ev.point_name(o,'er',s):{role:np.full((60,22),.5) for role in ('raw','stage-cal','first-cal')} for o in ('ABC','BCA','CAB') for s in (1,2,3)}
new=copy.deepcopy(base);mi=columns.index('map');ai=columns.index('average_precision')
for roles in new.values():
 for arr in roles.values():arr[:,mi]+=.001+.01*(np.arange(60)//3-9.5)
case=run.comparisons(new,base,domains)
assert all(case['continuation_checks'].values()) and case['worth_matched_replay']
assert not case['old_map_conditional_positive'] and not case['automatic_followup']
unchanged=run.comparisons(base,base,domains)
assert sum(unchanged['continuation_checks'].values())==5 and not unchanged['worth_matched_replay']
summary={'origin':'web_symbolic_saved_metric_fixture','matrices_are_effect_results':False,'formal_labels_read':False,'actual_baseline_read':False,'numerical_values_compared':compared,'maximum_absolute_error':error,'paired_axis_mismatch_rejected':rejections,'own_first_endpoint_delta_map':fingerprints['F_first'],'all_seven_map_deltas':fingerprints,'observation_positive_interval_crosses_zero':{'checks':case['continuation_checks'],'interval':case['delta']['O']['map']['conditional_95pct_interval'],'mean':case['delta']['O']['map']['mean'],'positive_interval':case['old_map_conditional_positive'],'automatic_followup':case['automatic_followup']},'equal_candidates_fail_strict_O_rule':unchanged['continuation_checks'],'statistics_status':result['status']}
(R/'outputs/04_endpoint_pairing.result.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False)+'\n');print(json.dumps(summary,indent=2,ensure_ascii=False))
