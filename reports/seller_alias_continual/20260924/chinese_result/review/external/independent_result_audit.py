"""Independent result audit. Never reads labels, original texts, or model weights.

Checks archived arrays/counts/provenance and frozen group bootstrap. Toy labels
are constructed only in manual_metric_tests(); they are unrelated to real data.
Usage: python independent_result_audit.py SOURCE_DIRECTORY OUTPUT_JSON
"""
from __future__ import annotations
import sys, json, math, hashlib, random, zipfile, time
from pathlib import Path
from collections import Counter
import numpy as np

ROOT=Path(sys.argv[1]).resolve(); OUTPUT=Path(sys.argv[2]).resolve()
RUN=ROOT/'reports/seller_alias_continual/20260924/chinese_execution/20260924_152642/job/run'
EV=ROOT/'reports/seller_alias_continual/20260924/chinese_evaluation/20260924_202700/run'
OLD=ROOT/'reports/seller_alias_continual/20260924/base_evaluation/20260924_113500/evaluation'
ARMS=['mean_bce','mean_rank','split_bce','split_rank']; ALL=ARMS+['historical_labse']
COLS=['average_precision','trapezoidal_pr_auc','roc_auc','recall_at_fpr_1pct','brier','log_loss','precision','recall','f1','specificity','balanced_accuracy','mcc','map','mrr','recall_at_1','recall_at_3','recall_at_5','recall_at_10','ndcg_at_1','ndcg_at_3','ndcg_at_5','ndcg_at_10']
N=0; MAXERR=0.; BLIND=0; HASHES=[]; CHECKS=[]; SUMMARY={}; start=time.monotonic()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def canonical(x):return (json.dumps(x,ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n').encode()
def close(a,b,name):
 global N,MAXERR
 x,y=np.asarray(a,dtype=float),np.asarray(b,dtype=float)
 assert x.shape==y.shape and np.isfinite(x).all() and np.isfinite(y).all(),name
 err=float(np.max(np.abs(x-y),initial=0));N+=x.size;MAXERR=max(MAXERR,err)
 assert err<1e-12,(name,err)
def verify(base,row):
 p=base/row['path'];assert p.stat().st_size==row['bytes'] and sha(p)==row['sha256'],str(p)
 HASHES.append(p.relative_to(ROOT).as_posix());return p
def load(base,row,shape):
 a=np.load(verify(base,row),allow_pickle=False);assert a.shape==shape and np.isfinite(a).all()
 assert a.dtype==(np.float64 if shape[-1]==22 else np.float32)
 return a
def counts(rows):
 a=np.array([[r[k] for k in ('tp','fp','fn','tn')] for r in rows],dtype=np.int64) if isinstance(rows[0],dict) else np.array(rows,dtype=np.int64)
 assert a.ndim==2 and a.shape[1]==4 and np.all(a>=0)
 assert np.all(a[:,0]+a[:,2]==20) and np.all(a[:,1]+a[:,3]==358)
 return a
def ratio(t):
 t=np.asarray(t,dtype=float);tp,fp,fn,tn=np.moveaxis(t,-1,0)
 def div(n,d):return np.divide(n,d,out=np.zeros_like(np.asarray(n,dtype=float)),where=d!=0)
 p=div(tp,tp+fp);r=div(tp,tp+fn);fpr=div(fp,fp+tn);spec=div(tn,tn+fp)
 return {'precision':p,'recall':r,'f1':div(2*tp,2*tp+fp+fn),'fpr':fpr,'specificity':spec,'balanced_accuracy':(r+spec)/2,'mcc':div(tp*tn-fp*fn,np.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn)))}
def verify_rates(rows,saved,scope):
 t=np.asarray(rows).sum(0)
 close(t,[saved[k] for k in ('tp','fp','fn','tn')],scope+' counts')
 rr=ratio(t)
 for k in ('precision','recall','f1','fpr'):close(rr[k],saved[k],scope+' '+k)
def blind(s,t,c):
 global BLIND
 close(np.sum(s.astype(np.float64)>=float(t),axis=1),c[:,0]+c[:,1],'blind positives only')
 BLIND+=len(s)
def percentile(x):
 """Independently implement linear empirical 2.5/97.5 percentiles."""
 x=np.sort(np.asarray(x),axis=0);out=[]
 for p in (.025,.975):
  h=(len(x)-1)*p;lo=math.floor(h);hi=math.ceil(h)
  out.append(x[lo]+(x[hi]-x[lo])*(h-lo))
 return np.array(out)

inv=read(ROOT/'source_inventory.json');assert len(inv['files'])==127
for row in inv['files']:verify(ROOT,row)
CHECKS.append('127_payload_hashes')
policy=read(ROOT/'schema/step28_chinese_base_policy.json');m=read(RUN/'manifest.json');e=read(EV/'evaluation.json');c=read(EV/'collected.json');part=read(RUN/'partition.json')
assert m['config']==policy==e['config']==c['config'];assert e['columns']==c['columns']==COLS
assert m['label_parses']=={'train':1,'development':0,'heldout':0,'owners':0}
assert e['label_parses']=={'train':0,'development':1,'heldout':0,'owners':0}
assert e['manifest_sha256']==c['manifest_sha256']==sha(RUN/'manifest.json')
for row in m['source_files']:verify(ROOT,row)
ex=read(EV.parent/'execution.json')
for row in ex['frozen_sources_verified']:verify(ROOT,row)
assert len(ex['frozen_sources_verified'])==11 and ex['exit_status']==0
verify(RUN,m['partition']);assert e['partition']==m['partition']==c['partition']
oldpart=read(ROOT/policy['historical_reference']['files']['partition.json']['path']);assert part==oldpart
ids=[r['group_uid'] for role in part for r in part[role]];assert len(set(ids))==240
for role,n in [('fit',48),('calibration',12),('development',20)]:
 assert Counter(r['domain'] for r in part[role])==Counter({d:n for d in 'ABC'})
for d in 'ABC':
 tr=[r['group_uid'] for role in ('fit','calibration') for r in part[role] if r['domain']==d]
 chosen=sorted(tr,key=lambda uid:hashlib.sha256(canonical([20260918,d,uid])).hexdigest())[:48]
 assert set(chosen)=={r['group_uid'] for r in part['fit'] if r['domain']==d}
fitids=[r['group_uid'] for r in part['fit']]
seed=int.from_bytes(hashlib.sha256(canonical([20260918,'BASE_JOINT','current'])).digest()[:8],'big')%(2**63-1)
rng=random.Random(seed);schedule=[]
for _ in range(6):
 row=sorted(fitids);rng.shuffle(row);schedule+=row
schedule_sha=hashlib.sha256(canonical(schedule)).hexdigest()
assert Counter(schedule)==Counter({i:6 for i in fitids})
CHECKS.append('partition_hash_rule_original_48_12_20_and_exact_864_schedule')
assert e['primary_comparison']==['split_rank','mean_bce']==policy['primary_comparison']
assert c['status']=='CHINESE_BASE_METRICS_COLLECTED_BEFORE_UNCERTAINTY' and c['comparisons']=={}
assert e['status']=='CHINESE_BASE_DEVELOPMENT_EVALUATED_NO_AUTOMATIC_WINNER'
assert e['domains']==c['domains']==[r['domain'] for r in part['development']]
assert e['group_ids']==c['group_ids']==[r['group_uid'] for r in part['development']]
assert not (EV/'failure.json').exists() and not (RUN/'failure.json').exists()
dom=np.array(e['domains']);scopes={d:np.flatnonzero(dom==d) for d in 'ABC'};scopes['pooled']=np.arange(60)
matrices={};autoc={};ams={};table={};gaps={};cal_summary={};nmat=0
for a in ALL:
 saved=e['arms'][a] if a in ARMS else e[a];raw=c['arms'][a] if a in ARMS else c[a]
 if a in ARMS:
  am=read(verify(RUN,m['arms'][a]['manifest']));ams[a]=am
  assert am['updates']==864 and am['label_parses']==0 and am['group_schedule_sha256']==schedule_sha and am['dropout_stream']==seed
  assert am['fit_group_ids']==fitids and am['calibration_group_ids']==[r['group_uid'] for r in part['calibration']]
  assert am['parameter_count']==am['trainable_parameter_count']
  close(am['formal_training_seconds'],sum(x['seconds'] for x in am['training']),'arm duration')
  for log,st,en in zip(am['training'],[0,432],[432,864]):
   assert log['start']==st and log['stop']==en and log['updates']==432
   assert all(v=={'finite_nonzero_gradient':True,'parameters_changed':True} for v in log['first_update_modules'].values())
   loss=log['losses_by_epoch'];w=policy['interventions'][a]['rank_weight']
   # Production sums rounded float32 loss per step. Tolerance is relative.
   assert np.allclose(loss['total'],np.array(loss['bce'])+w*np.array(loss['rank']),rtol=1e-6,atol=1e-7)
  pfs=am['preflight'];assert pfs==m['preflights'][a] and pfs['max_tokens']<=256 and pfs['updates']==pfs['supervision_reads']==0
  assert pfs['records_checked']==24103*(1 if a.startswith('mean') else 2)
  for k in ['file_count','total_size_bytes','content_sha256']:assert pfs['pretrained'][k]==policy['models'][a][k]
  assert saved['points']==raw['points']
  points=saved['points'];gaps[a]={}
 else:
  for k in ('file','counts_at_logit_zero','fixed_classification','counts_by_group','threshold'):assert saved[k]==raw[k]
  points={'6':saved}
 threshold=saved['automatic_classification']['threshold'] if a in ARMS else saved['threshold']
 ac=counts(saved['automatic_classification']['counts_by_group'] if a in ARMS else saved['counts_by_group']);autoc[a]=ac
 assert ac.tolist()==(raw['automatic_classification']['counts_by_group'] if a in ARMS else raw['counts_by_group'])
 for epoch,p in points.items():
  mat=load(EV,p['file'],(60,22));nmat+=1;z=counts(p['counts_at_logit_zero'])
  computed=ratio(z)
  for k in COLS[6:12]:close(mat[:,COLS.index(k)],computed[k],a+epoch+' fixed '+k)
  if a in ARMS:
   pp=am['points'][epoch];assert pp['full_model_and_adam_reloaded'] and pp['model']['actual_reload_verified']
   ss=load(RUN/a,pp['scores']['development'],(60,378))
  else:ss=load(ROOT,policy['historical_reference']['files']['labse/scores/epoch6_development.npy'],(60,378))
  blind(ss,0,z)
  if epoch=='6':
   matrices[a]=mat;blind(ss,threshold,ac);table[a]={k:float(mat[:,j].mean()) for j,k in enumerate(COLS)}
  for scope,ix in scopes.items():
   if a in ARMS:close(mat[ix].mean(0),[p['mean' if scope=='pooled' else 'by_domain'][k] if scope=='pooled' else p['by_domain'][scope][k] for k in COLS],a+epoch+scope+' means')
   verify_rates(z[ix],p['fixed_classification']['pooled'] if scope=='pooled' else p['fixed_classification']['by_domain'][scope],a+epoch+scope)
  # Label-free consistency inequalities only; no inferred formal pair truth.
  assert np.all((mat[:,[i for i in range(22) if i not in (5,11)]]>=0)&(mat[:,[i for i in range(22) if i not in (5,11)]]<=1))
  assert np.all(mat[:,5]>=0) and np.all(abs(mat[:,11])<=1)
  assert np.all(np.diff(mat[:,14:18],axis=1)>=-1e-14)
  assert np.all(mat[:,18]+1e-14>=mat[:,14]) and np.all(mat[:,18]<=2*mat[:,14]+1e-14)
  if a in ARMS:
   gaps[a][epoch]={'valid':{k:float(mat[:,COLS.index(k)].mean()) for k in ('average_precision','map','mrr','log_loss')}}
   for role,n in [('fit',144),('calibration',36)]:
    pm=pp['train_metrics'][role];mm=load(RUN/a,pm['file'],(n,22));nmat+=1;cc=counts(pm['counts']);rr=ratio(cc)
    for k in COLS[6:12]:close(mm[:,COLS.index(k)],rr[k],a+epoch+role+k)
    ssr=load(RUN/a,pp['scores'][role],(n,378));blind(ssr,0,cc)
    gaps[a][epoch][role]={k:float(mm[:,COLS.index(k)].mean()) for k in ('average_precision','map','mrr','log_loss')}
 if a in ARMS:
  cal=read(verify(RUN/a,am['calibration']));cc=counts(cal['counts_by_group']);cd=np.array([r['domain'] for r in part['calibration']])
  assert cal['score_sha256']==am['points']['6']['scores']['calibration']['sha256'] and cal['model_state_sha256']==am['points']['6']['model_state_sha256']
  assert cal['threshold']==threshold==max(x['threshold'] for x in cal['bounds'].values())
  assert cal['source_role']=='train_calibration_only' and cal['prediction']=='float64(logit)>=float64(threshold)'
  for d in 'ABC':
   b=cal['bounds'][d];assert b['allowed_false_positives']==4 and b['negative_pairs']==4296
   assert b['threshold']==float(np.nextafter(np.float64(b['next_negative_logit']),np.inf))
   assert cc[cd==d].sum(0).tolist()==cal['counts_by_domain'][d] and cal['counts_by_domain'][d][1]<=4
  ss=load(RUN/a,am['points']['6']['scores']['calibration'],(36,378));blind(ss,threshold,cc)
  cal_summary[a]={'threshold':threshold,'threshold_domain_bounds':cal['bounds'],'calibration_counts':cc.sum(0).tolist(),'calibration_recall':float(ratio(cc.sum(0))['recall']),'valid_counts':ac.sum(0).tolist(),'valid_recall':float(ratio(ac.sum(0))['recall'])}
CHECKS.append('all_25_matrices_and_R1_snapshot_R2_macro_and_pooled_R3_records')
assert ams['mean_bce']['preflight']['initial_model_state_sha256']==ams['mean_rank']['preflight']['initial_model_state_sha256']
assert ams['split_bce']['preflight']['initial_model_state_sha256']==ams['split_rank']['preflight']['initial_model_state_sha256']
close(m['formal_training_seconds'],sum(v['formal_training_seconds'] for v in ams.values()),'total duration')
assert m['physical_updates']==sum(a['updates'] for a in ams.values())==3456
assert m['budget']['elapsed_seconds']<28800 and m['budget']['peak_observed_bytes']<32*1024**3
receipt=read(RUN/'model_verification.json');assert len(receipt['files'])==8 and receipt['manifest_sha256']==sha(RUN/'manifest.json')
for a in ARMS:
 for ep in ('3','6'):
  rec=ams[a]['points'][ep]['model'];row={'path':a+'/'+rec['path'],'bytes':rec['bytes'],'sha256':rec['sha256']};assert row in receipt['files']
assert sum(row['bytes'] for row in receipt['files'])==10439069376
transfer=read(ROOT/'reports/seller_alias_continual/20260924/chinese_result/transfer_inventory.json')
assert len(transfer['files'])==80 and sum(x['bytes'] for x in transfer['files'])==3776103
for row in transfer['files']:verify(ROOT,row)
# Conditional group bootstrap. Explicit group-frequency matrix; no evaluator imports.
draws=np.random.default_rng(20260918).integers(0,20,size=(5000,3,20));weights=np.zeros((5000,60),dtype=np.int16)
for b in range(5000):
 for di,d in enumerate('ABC'):weights[b,scopes[d]]=np.bincount(draws[b,di],minlength=20)
assert np.all(weights.sum(1)==60)
auto_intervals=0;auto_comparisons=0
boots={a:{scope:weights[:,ix]@autoc[a][ix] for scope,ix in scopes.items()} for a in ALL}
for a in ALL:
 auto=e['arms'][a]['automatic_classification'] if a in ARMS else e[a]['automatic_classification'];passed=[]
 for scope,ix in scopes.items():
  sp=auto['pooled'] if scope=='pooled' else auto['by_domain'][scope]
  verify_rates(autoc[a][ix],sp,a+scope+' calibrated')
  rr=ratio(boots[a][scope])
  for metric in ['fpr','recall','precision']:
   close(percentile(rr[metric]),sp['conditional_95pct_intervals'][metric],a+scope+metric+' interval');auto_intervals+=1
  if scope!='pooled':
   t=autoc[a][ix].sum(0);gate=bool(t[1]<=7 and t[0]>=200);assert sp['passes_point_gates']==gate;passed.append(gate)
 assert auto['all_domains_pass']==all(passed)==False
comp_summary={};retrieval_keys=COLS[12:];directions={}
assert len(policy['comparisons'])==7 and set(e['comparisons'])=={a+'_minus_'+b for a,b in policy['comparisons']}
for a,b in policy['comparisons']:
 name=a+'_minus_'+b;delta=matrices[a]-matrices[b];sample=weights@delta/60;ci=percentile(sample);means=np.array([math.fsum(delta[:,j])/60 for j in range(22)])
 sp=e['comparisons'][name];assert set(sp['metrics'])==set(COLS)
 for j,key in enumerate(COLS):close([means[j],*ci[:,j]],[sp['metrics'][key]['mean'],*sp['metrics'][key]['conditional_95pct_interval']],name+key)
 for scope,ix in scopes.items():
  ap,bp=ratio(autoc[a][ix].sum(0)),ratio(autoc[b][ix].sum(0));ab,bb=ratio(boots[a][scope]),ratio(boots[b][scope])
  for metric in ['fpr','recall','precision']:
   sm=sp['automatic'][metric][scope];close([ap[metric]-bp[metric],*percentile(ab[metric]-bb[metric])],[sm['difference'],*sm['conditional_95pct_interval']],name+scope+metric);auto_comparisons+=1
 comp_summary[name]={k:sp['metrics'][k] for k in ['average_precision','map','mrr','recall_at_1','recall_at_3','recall_at_5','recall_at_10','ndcg_at_1','ndcg_at_3','ndcg_at_5','ndcg_at_10','log_loss','brier']}
 comp_summary[name]['frozen_pooled_recall']=sp['automatic']['recall']['pooled']
 directions[name]={'interval_above_zero':[k for j,k in enumerate(COLS) if ci[0,j]>0],'interval_below_zero':[k for j,k in enumerate(COLS) if ci[1,j]<0],'covers_zero':[k for j,k in enumerate(COLS) if ci[0,j]<=0<=ci[1,j]]}
assert auto_intervals==60 and auto_comparisons==84
old=read(OLD/'evaluation.json');om=load(OLD,old['arms']['labse']['points']['6']['file'],(60,22));assert np.array_equal(om,matrices['historical_labse'])
assert old['arms']['labse']['automatic_classification']['counts_by_group']==autoc['historical_labse'].tolist()
CHECKS.append('154_metric_and_84_classification_difference_intervals_plus_60_classification_intervals')
# Hand-constructed production metric tests, with no real labels.
def manual_metric_tests():
 sys.path.insert(0,str(ROOT/'scripts'))
 import step28_continual_population_evaluate as prod
 assert list(prod.COLUMNS)==COLS
 components=[list(range(2*i,2*i+2)) for i in range(8)]+[list(range(16+3*i,19+3*i)) for i in range(4)]
 pairs=[(i,j) for i in range(28) for j in range(i+1,28)]
 component={q:n for n,group in enumerate(components) for q in group}
 y=np.array([int(component[i]==component[j]) for i,j in pairs],dtype=np.uint8)
 scenarios=[np.zeros(378),np.array([random.Random(i+7).randrange(-3,4) for i in range(378)],dtype=float),np.linspace(-7,4,378),np.where(y,3.,-3.),np.where(y,-3.,3.)]
 errors=[]
 for idx,s in enumerate(scenarios):
  actual,cc=prod.group_metrics(y[None],s[None]);rp=0.;prevp=1.;ap=0.;trap=0.;rec1=0.
  for t in sorted(set(s),reverse=True):
   selected=s>=t;tp=sum(int(v) for v in y[selected]);fp=int(selected.sum())-tp;r=tp/20;p=tp/(tp+fp)
   ap+=(r-rp)*p;trap+=(r-rp)*(prevp+p)/2;rp,prevp=r,p
   if fp/358<=.01:rec1=max(rec1,r)
  auc=math.fsum(float(a>b)+.5*float(a==b) for a in s[y==1] for b in s[y==0])/(20*358)
  probs=1/(1+np.exp(-s));br=float(np.mean((probs-y)**2));ll=float(np.mean(np.logaddexp(0,s)-y*s))
  tps=int(np.count_nonzero((s>=0)&(y==1)));fps=int(np.count_nonzero((s>=0)&(y==0)));r=ratio(np.array([tps,fps,20-tps,358-fps]))
  classrow=[ap,trap,auc,rec1,br,ll]+[float(r[k]) for k in COLS[6:12]]
  edges={pair:(float(s[j]),int(y[j])) for j,pair in enumerate(pairs)};queries=[]
  for q in range(28):
   cand=[n for n in range(28) if n!=q];cand.sort(key=lambda n:(-edges[tuple(sorted((q,n)))][0],n))
   rel=[edges[tuple(sorted((q,n)))][1] for n in cand];places=[i+1 for i,x in enumerate(rel) if x];total=len(places)
   avg=math.fsum((j+1)/rank for j,rank in enumerate(places))/total;rr=1/places[0]
   vals=[avg,rr]+[sum(rel[:k])/total for k in [1,3,5,10]]
   vals += [math.fsum(x/math.log2(i+2) for i,x in enumerate(rel[:k]))/math.fsum(1/math.log2(i+2) for i in range(min(k,total))) for k in [1,3,5,10]]
   queries.append(vals)
  manual=np.r_[classrow,np.array(queries).mean(0)]
  close(actual[0],manual,'manual metric scenario '+str(idx));errors.append(float(np.max(np.abs(actual[0]-manual))))
  # Bounded monotone shift/scale preserves rank metrics, not fixed classification.
  transformed,_=prod.group_metrics(y[None],(2*s+1)[None])
  close(transformed[:,0:4],actual[:,0:4],'rank transformation curves')
  close(transformed[:,12:],actual[:,12:],'rank transformation retrieval')
 return {'scenarios':len(scenarios),'metrics_each':22,'max_absolute_error':max(errors),'formal_labels_used':False,'case_types':['all_ties','discrete_ties','continuous','perfect','reversed']}
manual=manual_metric_tests();CHECKS.append('five_handcrafted_full_22_metric_and_monotone_transform_checks')
SUMMARY={'status':'PASS_INDEPENDENT_SAVED_RESULT_REVIEW','checks':CHECKS,'numeric_comparisons':int(N),'max_absolute_error':MAXERR,'blind_group_counts':BLIND,'unique_hashed_files':len(set(HASHES)),'metric_matrices':nmat,'formal_labels_read':0,'model_weights_read':0,'formal_updates':0,'bootstrap':{'replicates':5000,'metric_comparisons':154,'classification_difference_intervals':84,'classification_intervals':60,'unit':'paired whole group within each fixed domain','percentiles':'explicit sorted linear interpolation'},'manual_metric_tests':manual,'epoch6_metrics':table,'comparison_summary':comp_summary,'interval_signs_not_multiplicity_corrected':directions,'training_gaps':gaps,'calibration_valid':cal_summary,'resources':{a:ams[a]['resources'] for a in ARMS},'preflight':{a:{k:ams[a]['preflight'][k] for k in ['max_tokens','records_checked','actual_forward_shape']} for a in ARMS},'elapsed_seconds':time.monotonic()-start,'limits':'Formal curve/probability/retrieval rows are received saved results, not reconstructed from labels; no model weights or original text inspected. Saved provenance/receipts are not a fresh server verification.'}
OUTPUT.write_text(json.dumps(SUMMARY,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps({k:SUMMARY[k] for k in ['status','numeric_comparisons','max_absolute_error','blind_group_counts','metric_matrices','manual_metric_tests','elapsed_seconds']},ensure_ascii=False))
