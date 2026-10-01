"""Label-free independent audit of saved outputs.

Does NOT import project modules or the supplied auditor. No labels inferred,
loaded or reconstructed. AP/MAP/etc are INPUTS, not recomputed from labels.
Bootstrap is evaluated by resampling actual-domain matrices first, then forming
R(t,j) endpoints directly; unlike the supplied order/domain coefficient fields.
"""
from __future__ import annotations
import csv, hashlib, json, math, pathlib, resource, time
import numpy as np
ROOT=pathlib.Path('/mnt/data/bge_input')
JOB=ROOT/'reports/seller_alias_continual/20260930/bge_continual_execution/20260930_143700/job'
E=JOB/'evaluation'; RUN=JOB/'run'; OUT=pathlib.Path('/mnt/data/bge_review_evidence/outputs')
ORDERS=('ABC','BCA','CAB'); ARMS=('frozen','seq','er','logit'); ROLES=('raw','stage-cal','first-cal','primary')
METRICS=('average_precision','trapezoidal_pr_auc','roc_auc','recall_at_fpr_1pct','brier','log_loss','precision','recall','f1','specificity','balanced_accuracy','mcc','map','mrr','recall_at_1','recall_at_3','recall_at_5','recall_at_10','ndcg_at_1','ndcg_at_3','ndcg_at_5','ndcg_at_10')
RANK_IDX=[0,1,2,3,*range(12,22)]; CLASS_IDX=list(range(6,12)); ENDPOINTS=('O','N','Z','F_first','F','G','final_all')
J=lambda p:json.loads(p.read_text())
def save(n,obj): (OUT/n).write_text(json.dumps(obj,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
class Check:
 def __init__(self): self.count=0;self.max_abs=0.;self.max_at='';self.exact_assertions=0
 def close(self,a,b,name):
  x=np.asarray(a,dtype=np.float64);y=np.asarray(b,dtype=np.float64)
  assert x.shape==y.shape,(name,x.shape,y.shape)
  assert np.isfinite(x).all() and np.isfinite(y).all(),name
  diff=float(np.max(np.abs(x-y))) if x.size else 0.
  self.count+=x.size
  if diff>self.max_abs: self.max_abs=diff;self.max_at=name
  assert diff<=1e-12,(name,diff)
 def exact(self,a,b,name):
  assert a==b,(name,a,b);self.exact_assertions+=1
C=Check(); start=time.perf_counter()
def verified(base,ref):
 p=base/ref['path'];v=p.read_bytes();C.exact(len(v),ref['bytes'],str(p)+' size');C.exact(hashlib.sha256(v).hexdigest(),ref['sha256'],str(p)+' hash');return p
col=J(E/'collected.json'); official=J(E/'evaluation.json'); manifest=J(RUN/'manifest.json'); partition=J(RUN/'partition.json')
C.exact(col['metric_columns'],list(METRICS),'22 metric column definitions')
C.exact(len(set(col['group_ids'])),60,'60 unique groups')
C.exact(list(zip(col['group_ids'],col['domains'])),[(r['group_uid'],r['domain']) for r in partition['development']],'group/domain alignment')
rows={d:np.flatnonzero(np.asarray(col['domains'])==d) for d in 'ABC'}
for d in 'ABC': C.exact(len(rows[d]),20,'20 groups '+d)
expected={o+'_shared' for o in ORDERS}|{f'{o}_{a}_stage{t}' for o in ORDERS for a in ARMS[1:] for t in (2,3)}
C.exact(set(col['points']),expected,'21 point set')
mat={}; counts={}; scores={}; maxima=[]; predchecks=0; tie_groups=0; tied_adjacencies=0; query_orders=0
for point,entry in [('initial',{'raw':col['initial']}),*col['points'].items()]:
 mat[point]={};counts[point]={};scores[point]={}
 info=None if point=='initial' else J(RUN/'points'/f'{point}.json')
 for role,item in entry.items():
  a=np.load(verified(E,item['matrix']),allow_pickle=False);ct=J(verified(E,item['counts']))
  C.exact(a.shape,(60,22),'matrix shape '+point+role);C.exact(str(a.dtype),'float64','matrix dtype');assert np.isfinite(a).all()
  C.exact(len(ct),60,'count rows')
  score_ref=manifest['initial']['scores'] if point=='initial' else info['scores']['development' if role=='raw' else role]
  s=np.load(verified(RUN,score_ref),allow_pickle=False)
  C.exact(s.shape,(60,378),'score shape');assert np.isfinite(s).all()
  C.exact(str(s.dtype),'float32' if role=='raw' else 'float64','score dtype')
  c=np.array([[r[k] for k in ('tp','fp','fn','tn')] for r in ct],dtype=np.int64)
  assert all(type(r[k]) is int and r[k]>=0 for r in ct for k in ('tp','fp','fn','tn'))
  assert np.all(c[:,0]+c[:,2]==20) and np.all(c[:,1]+c[:,3]==358)
  assert np.array_equal(np.count_nonzero(s>=0,axis=1),c[:,0]+c[:,1]);predchecks+=60
  for i,(tp,fp,fn,tn) in enumerate(c.tolist()):
   den=math.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))
   v=[tp/(tp+fp) if tp+fp else 0.,tp/(tp+fn),2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.,tn/(tn+fp),(tp/(tp+fn)+tn/(tn+fp))/2,(tp*tn-fp*fn)/den if den else 0.]
   C.close(v,a[i,CLASS_IDX],f'six classification {point}/{role}/{i}')
  mat[point][role]=a;counts[point][role]=c;scores[point][role]=s
  maxima.append({'point':point,'role':role,'min_score':float(s.min()),'max_score':float(s.max()),'positive_predictions':int((s>=0).sum()),'tp':int(c[:,0].sum()),'fp':int(c[:,1].sum()),'fn':int(c[:,2].sum()),'tn':int(c[:,3].sum())})
 if point!='initial':
  raw=scores[point]['raw'].astype(np.float64);mapping=J(verified(RUN,info['mapping']))
  cs=np.load(verified(RUN,info['scores']['calibration']),allow_pickle=False)
  C.exact(cs.shape,(12,378),'calibration score shape');C.exact(str(cs.dtype),'float32','calibration dtype');assert np.isfinite(cs).all()
  C.exact(mapping['actual_domain'],info['actual_domain'],'calibration actual domain')
  C.exact(mapping['calibration_group_ids'],[r['group_uid'] for r in partition['calibration'] if r['domain']==info['actual_domain']],'current calibration only')
  C.exact(mapping['score_source'],info['scores']['calibration'],'calibration score binding');C.exact(mapping['model_state_sha256'],info['model_state_sha256'],'calibration model binding')
  for k,v in [('group_count',12),('pair_count',4536),('positive_count',240)]: C.exact(mapping[k],v,k)
  first=J(RUN/'maps'/f'{point[:3]}_shared.json');C.exact(info['first_map_parameters'],{k:first[k] for k in ('a','b')},'frozen first calibration')
  for role,use in [('stage-cal',mapping),('first-cal',first)]:
   assert 0<use['a']<=100; trans=use['a']*raw+use['b'];C.close(trans,scores[point][role],point+'/'+role+' affine');assert np.array_equal(trans,scores[point][role])
   C.close(mat[point]['raw'][:,RANK_IDX],mat[point][role][:,RANK_IDX],point+' 14 invariant ranking/curve columns');assert np.array_equal(mat[point]['raw'][:,RANK_IDX],mat[point][role][:,RANK_IDX])
   # All 378 edge order relations and ties, then each 27-candidate query order.
   for i in range(60):
    ix=np.argsort(-raw[i],kind='stable');iy=np.argsort(-trans[i],kind='stable')
    assert np.array_equal(ix,iy)
    dr=np.diff(raw[i,ix]);dt=np.diff(trans[i,iy]);assert np.array_equal(dr==0,dt==0);assert np.array_equal(np.sign(dr),np.sign(dt))
    tie_groups+=1;tied_adjacencies+=int((dr==0).sum())
    for q in range(28):
     edges=[(min(q,j),max(q,j)) for j in range(28) if j!=q]
     # Closed upper-triangular row-major edge index, independent of project matrix construction.
     inds=[i0*(55-i0)//2+j0-i0-1 for i0,j0 in edges]
     assert np.array_equal(np.argsort(-raw[i,inds],kind='stable'),np.argsort(-trans[i,inds],kind='stable'));query_orders+=1
# Draw actual domains once, never treat the 3 repetitions as 180 independent groups.
draws=np.random.Generator(np.random.PCG64(20260930)).integers(0,20,(5000,3,20))
pubdraw=np.load(verified(E,official['draws']),allow_pickle=False);assert np.array_equal(draws,pubdraw);C.exact(str(pubdraw.dtype),'int64','draw dtype');np.save(OUT/'independent_bootstrap_draws.npy',draws,allow_pickle=False)
# Compute domain sample means first. All endpoint arithmetic below uses R(t,j).
cache={}
def physical(o,a,t):return o+'_shared' if t==1 or a=='frozen' else f'{o}_{a}_stage{t}'
def domain_stats(point,role):
 key=(point,role)
 if key not in cache:
  if role=='primary':
   vals=mat[point]['stage-cal'].copy();vals[:,RANK_IDX]=mat[point]['raw'][:,RANK_IDX]
  else:vals=mat[point][role]
  pointvals=np.stack([vals[rows[d]].mean(axis=0) for d in 'ABC'])
  resamples=np.stack([vals[rows[d]][draws[:,di],:].mean(axis=1) for di,d in enumerate('ABC')],axis=1)
  cache[key]=(pointvals,resamples)
 return cache[key]
def by_order(o,a,role):
 R={}
 for t in (1,2,3):
  mu,b=domain_stats(physical(o,a,t),role)
  for j,d in enumerate(o,1):R[t,j]=(mu['ABC'.index(d)],b[:,'ABC'.index(d)])
 output={}
 # Written directly from the frozen endpoint definitions, not project coefficient arrays.
 for component in (0,1):
  r=lambda t,j:R[t,j][component]
  values={'O':(r(3,1)+r(3,2))/2,'N':(r(2,2)+r(3,3))/2,'Z':r(3,3),'F_first':r(1,1)-r(3,1),'F':((r(1,1)-r(3,1))+(r(2,2)-r(3,2)))/2,'G':((r(2,2)-r(1,2))+(r(3,3)-r(2,3)))/2,'final_all':(r(3,1)+r(3,2)+r(3,3))/3}
  for name,v in values.items():
   v=v.copy()
   if name in ('F_first','F','G'):v[...,4:6]*=-1
   output.setdefault(name,[]).append(v)
 return output
def interval(b):
 x=np.sort(b,axis=0);result=[]
 for q in (.025,.975):
  loc=(len(x)-1)*q;low=int(math.floor(loc));hi=int(math.ceil(loc));w=loc-low
  result.append((1-w)*x[low]+w*x[hi])
 return np.stack(result)
def summarize(mus,boots):
 mean=np.mean(mus,axis=0);ci=interval(np.mean(boots,axis=0))
 return {m:{'mean':float(mean[k]),'per_order':{o:float(mus[i][k]) for i,o in enumerate(ORDERS)},'conditional_95pct_interval':ci[:,k].tolist()} for k,m in enumerate(METRICS)}
def compare_stats(actual,expected,name):
 for m in METRICS:
  C.close(actual[m]['mean'],expected[m]['mean'],name+'/'+m+'/mean')
  C.close([actual[m]['per_order'][o] for o in ORDERS],[expected[m]['per_order'][o] for o in ORDERS],name+'/'+m+'/orders')
  C.close(actual[m]['conditional_95pct_interval'],expected[m]['conditional_95pct_interval'],name+'/'+m+'/CI')
stats={};trajectories={}
for arm in ARMS:
 stats[arm]={}
 for role in ROLES:
  traces=[by_order(o,arm,role) for o in ORDERS];trajectories[arm,role]=traces;stats[arm][role]={}
  for endpoint in ENDPOINTS:
   value=summarize([tr[endpoint][0] for tr in traces],[tr[endpoint][1] for tr in traces]);stats[arm][role][endpoint]=value
   compare_stats(value,official['endpoints'][arm][role][endpoint],f'{arm}/{role}/{endpoint}')
  for m in METRICS:C.close(stats[arm][role]['final_all'][m]['mean'],(2*stats[arm][role]['O'][m]['mean']+stats[arm][role]['Z'][m]['mean'])/3,'final=(2O+Z)/3')
comparisons={}; allchecks=[]
for cand,ref in [('er','seq'),('logit','er')]:
 d={};vr={}
 for role,ends,target in [('primary',ENDPOINTS,d),('raw',('O','N'),vr)]:
  for ep in ends:
   mus=[trajectories[cand,'primary'][i][ep][0]-trajectories[ref,role][i][ep][0] for i in range(3)]
   boots=[trajectories[cand,'primary'][i][ep][1]-trajectories[ref,role][i][ep][1] for i in range(3)]
   target[ep]=summarize(mus,boots)
   compare_stats(target[ep],official['comparisons'][cand+'_minus_'+ref]['primary' if role=='primary' else 'against_raw_reference'][ep],f'{cand}-{ref}/{role}/{ep}')
 v=lambda ep,m:d[ep][m]['mean']; checks={}
 items=[('old_map_improves',v('O','map'),'>'),('old_map_interval_above_zero',d['O']['map']['conditional_95pct_interval'][0],'>'),('old_recall5_improves',v('O','recall_at_5'),'>'),('new_map_non_decrease',v('N','map'),'>='),('new_recall5_non_decrease',v('N','recall_at_5'),'>=')]
 for ep in ('O','N'):
  items += [(f'{ep}_{m}_non_degradation',v(ep,m),'<=' if m in ('brier','log_loss') else '>=') for m in ('average_precision','roc_auc','brier','log_loss')]
  items += [(f'{ep}_{m}_against_raw_reference',vr[ep][m]['mean'],'<=') for m in ('brier','log_loss')]
 items += [(f'Z_{m}_non_degradation',v('Z',m),'<=' if m in ('brier','log_loss') else '>=') for m in ('map','recall_at_5','average_precision','roc_auc','brier','log_loss')]
 C.exact(len(items),23,'23 frozen conditions')
 for idx,(name,value,operator) in enumerate(items,1):
  ok=value>0 if operator=='>' else value>=0 if operator=='>=' else value<=0
  checks[name]=bool(ok);allchecks.append({'comparison':cand+'_minus_'+ref,'index':idx,'condition':name,'value':value,'operator':operator+'0','pass':bool(ok)})
 C.exact(checks,official['comparisons'][cand+'_minus_'+ref]['interpretation']['checks'],'all conditions')
 C.exact(all(checks.values()),official['comparisons'][cand+'_minus_'+ref]['interpretation']['pilot_observed_checks_pass'],'all-pass')
 comparisons[cand+'_minus_'+ref]={'primary':d,'against_raw_reference':vr,'checks':checks,'passes':sum(checks.values()),'positive_old_map_orders':sum(x>0 for x in d['O']['map']['per_order'].values())}
# No post-hoc LOGIT-SEQ gate or CI; descriptive endpoint differences only.
descriptive={ep:{m:stats['logit']['primary'][ep][m]['mean']-stats['seq']['primary'][ep][m]['mean'] for m in METRICS} for ep in ENDPOINTS}
f=stats['seq']['raw']['F_first']['map'];g=stats['seq']['raw']['G']['map'];matched=[o for o in ORDERS if f['per_order'][o]>0 and g['per_order'][o]>0]
forget={'checks':{'seq_first_MAP_loss_positive':f['mean']>0,'seq_first_MAP_loss_interval_above_zero':f['conditional_95pct_interval'][0]>0,'same_order_path_forgetting_and_new_learning':len(matched)>=2},'matched_orders':matched}
C.exact(forget['checks'],official['seq_forgetting']['checks'],'forgetting checks');C.exact(matched,official['seq_forgetting']['matched_orders'],'matched paths')
# Recompute every table key and every saved group-macro value, including frozen aliases.
expected_table={}
for o in ORDERS:
 for j,d in enumerate(o,1):
  for k,m in enumerate(METRICS):expected_table[(m,'raw',d,str(j),'0',o,'s0','initial','initial')]=float(mat['initial']['raw'][rows[d],k].mean())
  for a in ARMS:
   for t in (1,2,3):
    pt=physical(o,a,t)
    for role in ROLES[:3]:
     for k,m in enumerate(METRICS):expected_table[(m,role,d,str(j),str(t),o,'s0',a,pt)]=float(mat[pt][role][rows[d],k].mean())
tab=list(csv.DictReader((E/'stage_metrics.csv').open()));C.exact(len(tab),7326,'7326 data rows');seen=set()
for row in tab:
 key=tuple(row[k] for k in ('metric','output_role','actual_domain','arrival_index','stage','order','seed','method','source_point'))
 assert key not in seen;seen.add(key);C.close(float(row['group_macro']),expected_table[key],'stage table '+'|'.join(key))
C.exact(seen,set(expected_table),'complete stage-table keys')
# Absolute matrix means and saved pooled classification (counts summed before rates).
for pt in expected:
 for role in ROLES[:3]:
  pub=official['absolute_stage_results'][pt][role];a=mat[pt][role]
  C.close([pub['macro_all'][m] for m in METRICS],a.mean(0),'macro all')
  for d in 'ABC':C.close([pub['macro_by_domain'][d][m] for m in METRICS],a[rows[d]].mean(0),'macro by domain')
  for dom,ix in [('pooled',np.arange(60)),*[(d,rows[d]) for d in 'ABC']]:
   tp,fp,fn,tn=map(int,counts[pt][role][ix].sum(0));v={'tp':tp,'fp':fp,'fn':fn,'tn':tn,'fpr':fp/(fp+tn),'recall':tp/(tp+fn),'precision':tp/(tp+fp) if tp+fp else 0.,'f1':2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.}
   p=pub['pooled_fixed_half_classification']['pooled'] if dom=='pooled' else pub['pooled_fixed_half_classification']['by_domain'][dom]
   C.close([v[k] for k in v],[p[k] for k in v],'pooled classification')
# Initial-to-first learning, including Recall@5 omitted from the narrative table.
first_learning={}
for o in ORDERS:
 ix=rows[o[0]];a=mat['initial']['raw'][ix];b=mat[o+'_shared']['raw'][ix]
 result={'initial_raw':dict(zip(METRICS,a.mean(0).tolist())),'first_raw':dict(zip(METRICS,b.mean(0).tolist())),'raw_first_minus_initial':dict(zip(METRICS,(b-a).mean(0).tolist()))}
 for role,vals in result.items():C.close(list(vals.values()),[official['initial_to_first_learning'][o][role][m] for m in METRICS],'first learning')
 first_learning[o]=result
save('independent_endpoints.json',stats);save('independent_comparisons.json',comparisons);save('descriptive_logit_minus_seq.json',descriptive);save('independent_first_learning.json',first_learning);save('independent_score_and_count_summary.json',maxima)
with (OUT/'independent_23_conditions.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=list(allchecks[0]));w.writeheader();w.writerows(allchecks)
with (OUT/'independent_endpoints.csv').open('w',newline='') as f:
 w=csv.writer(f);w.writerow(['method','role','endpoint','metric','mean','CI_low','CI_high',*ORDERS])
 for a in ARMS:
  for role in ROLES:
   for ep in ENDPOINTS:
    for m,s in stats[a][role][ep].items():w.writerow([a,role,ep,m,s['mean'],*s['conditional_95pct_interval'],*[s['per_order'][o] for o in ORDERS]])
report={'status':'PASS_SAVED_OUTPUTS_INDEPENDENT_RECONSTRUCTION','matrix_count':sum(len(v) for v in mat.values()),'confusion_count_sets':sum(len(v) for v in counts.values()),'score_arrays':len(list((RUN/'scores').glob('*.npy'))),'threshold_prediction_group_checks':predchecks,'full_edge_order_and_tie_checks':tie_groups,'tied_adjacencies_observed_across_both_transforms':tied_adjacencies,'full_query_candidate_order_checks':query_orders,'stage_table_rows':len(tab),'bootstrap_shape':list(draws.shape),'bootstrap_regeneration_exact':True,'distinct_endpoint_metric_cases':4*4*7*22,'numeric_values_compared':C.count,'maximum_absolute_difference':C.max_abs,'maximum_at':C.max_at,'exact_checks':C.exact_assertions,'formal_comparison_passes':{k:v['passes'] for k,v in comparisons.items()},'forgetting':forget,'formal_labels_read':0,'formal_models_loaded':0,'project_modules_imported':0,'fitting_or_training':False,'elapsed_seconds':time.perf_counter()-start,'maximum_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'unverified':['Recomputation of AP/MAP, curve, retrieval, Brier or log_loss from formal labels','Actual tensor checkpoint reloading','Full memory-text/label byte remeasurement','Calibration optimization optimality from formal calibration labels']}
save('independent_result_audit.json',report);print(json.dumps(report,indent=2,ensure_ascii=False))
for a in ARMS:print(a,{e:{m:round(stats[a]['primary'][e][m]['mean'],9) for m in ['map','recall_at_5','brier','log_loss']} for e in ['O','N','Z','final_all']})
