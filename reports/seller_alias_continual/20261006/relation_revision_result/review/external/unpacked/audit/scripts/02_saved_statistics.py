"""Independent saved-matrix audit. No imports from project source; no labels/models.
Endpoint aggregation uses explicit terms and bootstrap frequency matrices rather
than the production endpoint_fields/summarize_field implementation.
"""
from pathlib import Path
import json,hashlib,csv,math,time
import numpy as np
ROOT=Path(__file__).resolve().parents[2]; IN=ROOT/'input'; OUT=ROOT/'audit/outputs'; OUT.mkdir(exist_ok=True)
RR=IN/'reports/seller_alias_continual/20261006/relation_revision_result'; EV=RR/'job/evaluation'
ORDERS=('ABC','BCA','CAB'); ROLES=('raw','stage-cal','first-cal','primary')
TERMS={'O':[(3,1,.5),(3,2,.5)],'N':[(2,2,.5),(3,3,.5)],'Z':[(3,3,1)],
'F_first':[(1,1,1),(3,1,-1)],'F':[(1,1,.5),(3,1,-.5),(2,2,.5),(3,2,-.5)],
'G':[(2,2,.5),(1,2,-.5),(3,3,.5),(2,3,-.5)],'final_all':[(3,1,1/3),(3,2,1/3),(3,3,1/3)]}
checks=[]; verified=[]; errs=[]
def read(p):return json.loads(p.read_text())
def verify(base,rec):
 p=base/rec['path'];b=p.read_bytes(); assert len(b)==rec['bytes'] and hashlib.sha256(b).hexdigest()==rec['sha256'],p
 verified.append({'path':str(p.relative_to(IN)),**{k:rec[k] for k in ['bytes','sha256']}});return p
def near(a,b,label,tol=1e-12):
 a=np.asarray(a,dtype=float);b=np.asarray(b,dtype=float);assert a.shape==b.shape,(label,a.shape,b.shape)
 err=float(np.max(np.abs(a-b))) if a.size else 0.;errs.append((label,err,a.size)); assert err<=tol,(label,err)
def div(a,b):return float(a/b) if b else 0.
def rates(c):
 tp,fp,fn,tn=[int(c[k]) for k in ('tp','fp','fn','tn')]
 precision=div(tp,tp+fp); recall=div(tp,tp+fn); specificity=div(tn,tn+fp)
 return dict(precision=precision,recall=recall,f1=div(2*tp,2*tp+fp+fn),specificity=specificity,
 balanced_accuracy=(recall+specificity)/2,mcc=div(tp*tn-fp*fn,math.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))),fpr=1-specificity)
c=read(EV/'collected.json');saved=read(EV/'evaluation.json'); policy=read(RR/'source/schema/step28_relation_revision_policy.json')
COLS=c['metric_columns'];assert len(COLS)==22 and len(set(COLS))==22
assert len(c['group_ids'])==len(set(c['group_ids']))==60
assert c['status']=='ALL_28_RELATION_METRIC_COUNT_SETS_SAVED'
assert sum(map(len,c['points'].values()))==28
DOM=np.array(c['domains']);ROWS={d:np.flatnonzero(DOM==d) for d in 'ABC'}
assert all(len(r)==20 for r in ROWS.values()); RANK=[0,1,2,3,*range(12,22)]
REF=IN/policy['reference']['local_root'];refcol=[]
for r in policy['reference']['collections']:
 rc=read(verify(REF,r)); refcol.append(rc)
 for k in ('metric_columns','domains','group_ids'):assert rc[k]==c[k],k
A={'relation':{},'logit0.1':{}}; C={'relation':{},'logit0.1':{}}
def load(base,rec):
 mat=np.load(verify(base,rec['matrix']),allow_pickle=False); counts=read(verify(base,rec['counts']))
 assert mat.shape==(60,22) and mat.dtype==np.float64 and np.isfinite(mat).all()
 assert len(counts)==60
 for i,ct in enumerate(counts):
  assert set(ct)=={'tp','fp','fn','tn'} and all(type(v)==int and v>=0 for v in ct.values())
  assert ct['tp']+ct['fn']==20 and ct['fp']+ct['tn']==358
  expected=rates(ct)
  for name in ('precision','recall','f1','specificity','balanced_accuracy','mcc'):near(mat[i,COLS.index(name)],expected[name],f'group-count:{name}')
 return mat,counts
initial,initial_counts=load(EV,c['points']['initial']['raw'])
for o in ORDERS:
 for s in (1,2,3):
  cur=c['points'][f'{o}_relation_revision_stage{s}'];rb=REF/'reference' if s==1 else REF
  other=refcol[1]['points'][f'{o}_shared'] if s==1 else refcol[0]['points'][f'{o}_logit_tenth_stage{s}']
  for arm,base,roles in [('relation',EV,cur),('logit0.1',rb,other)]:
   assert set(roles)==set(ROLES[:3]);a={};ct={}
   for role,rec in roles.items():a[role],ct[role]=load(base,rec)
   for role in ROLES[1:3]:assert np.array_equal(a[role][:,RANK],a['raw'][:,RANK])
   a['primary']=a['stage-cal'].copy();a['primary'][:,RANK]=a['raw'][:,RANK]
   A[arm][o,s]=a;C[arm][o,s]=ct
# Reproduce the declared random draw generator, using independent frequency-weighted statistics.
draws=np.random.Generator(np.random.PCG64(20260930)).integers(0,20,size=(5000,3,20))
freq=np.stack([np.stack([np.bincount(row,minlength=20) for row in draws[:,d]]) for d in range(3)],axis=1)/20
np.save(OUT/'bootstrap_draws.npy',draws,allow_pickle=False)
fields={};summaries={};stat_rows=[]
# Field per order/domain is needed to preserve use of each actual group across all paths.
def coefficients(arm,role,ep):
 f=np.zeros((3,3,20,22))
 for oi,o in enumerate(ORDERS):
  for stage,arrival,weight in TERMS[ep]:
   d=o[arrival-1]; f[oi,'ABC'.index(d)]+=weight*A[arm][o,stage][role][ROWS[d]]
 if ep in ('F_first','F','G'):f[:,:,:,COLS.index('brier')]*=-1;f[:,:,:,COLS.index('log_loss')]*=-1
 return f

def summary(f):
 per=f.sum(axis=1).mean(axis=1)
 # Average fixed order paths; sum contributions of actual domains, never average them again.
 coefficient=f.mean(axis=0)
 boot=np.zeros((5000,22))
 for d in range(3):boot+=freq[:,d,:]@coefficient[d]
 mean=coefficient.sum(axis=0).mean(axis=0)
 q=np.quantile(boot,[.025,.975],axis=0,method='linear')
 return {name:{'mean':float(mean[i]),'per_order':{o:float(per[j,i]) for j,o in enumerate(ORDERS)},'conditional_95pct_interval':q[:,i].tolist()} for i,name in enumerate(COLS)}
def compare_stats(x,y,label):
 for name in COLS:
  near([x[name]['mean'],*[x[name]['per_order'][o] for o in ORDERS],*x[name]['conditional_95pct_interval']],
       [y[name]['mean'],*[y[name]['per_order'][o] for o in ORDERS],*y[name]['conditional_95pct_interval']],label+':'+name)
for arm in A:
 summaries[arm]={}; fields[arm]={}
 for role in ROLES:
  summaries[arm][role]={}; fields[arm][role]={}
  for ep in TERMS:
   f=coefficients(arm,role,ep);sm=summary(f);fields[arm][role][ep]=f;summaries[arm][role][ep]=sm
   compare_stats(sm,saved['endpoints'][arm][role][ep],f'endpoints/{arm}/{role}/{ep}')
delta={}
for ep in TERMS:
 sm=summary(fields['relation']['primary'][ep]-fields['logit0.1']['primary'][ep]);delta[ep]=sm
 compare_stats(sm,saved['delta'][ep],f'delta/{ep}')
 for n in COLS:
  stat_rows.append({'endpoint':ep,'metric':n,'candidate':summaries['relation']['primary'][ep][n]['mean'],
   'reference':summaries['logit0.1']['primary'][ep][n]['mean'],'delta':sm[n]['mean'],
   **{o:sm[n]['per_order'][o] for o in ORDERS},'ci_low':sm[n]['conditional_95pct_interval'][0],'ci_high':sm[n]['conditional_95pct_interval'][1]})
checks={'O_map_positive':delta['O']['map']['mean']>0,
 **{f'{ep}_map_non_decrease':delta[ep]['map']['mean']>=0 for ep in ('N','Z')},
 **{f'{ep}_AP_non_decrease':delta[ep]['average_precision']['mean']>=0 for ep in ('O','N','Z')}}
assert checks==saved['continuation_checks']
assert saved['observed_continuation_checks_pass']==all(checks.values())
assert saved['old_map_conditional_positive']==(delta['O']['map']['conditional_95pct_interval'][0]>0)
assert saved['automatic_followup'] is False
# Absolute stage macro and pooled counts are independently checked.
def pool(ct):
 s={k:sum(x[k] for x in ct) for k in ('tp','fp','fn','tn')};return {**s,**{k:v for k,v in rates(s).items() if k in ('precision','recall','f1','fpr')}}
for point,roles in c['points'].items():
 for role in roles:
  if point=='initial':a,ct=initial,initial_counts
  else:
   o=point[:3];s=int(point[-1]);a=A['relation'][o,s][role];ct=C['relation'][o,s][role]
  entry=saved['absolute_stage_results'][point][role]
  near(a.mean(0),[entry['macro_all'][n] for n in COLS],point+'/'+role+'/macro_all')
  for d,rr in ROWS.items():near(a[rr].mean(0),[entry['macro_by_domain'][d][n] for n in COLS],point+'/'+role+'/'+d)
  pc=entry['pooled_fixed_half_classification'];assert pc['threshold']==0.0
  for k,v in pool(ct).items():near(v,pc['pooled'][k],point+'/'+role+'/pool/'+k)
  for d,rr in ROWS.items():
   for k,v in pool([ct[i] for i in rr]).items():near(v,pc['by_domain'][d][k],point+'/'+role+'/'+d+'/pool/'+k)
# Counts pooled across observed endpoint appearances, not new independent groups.
pooled={};stage=[]
for arm in A:
 pooled[arm]={}
 for ep in ('O','N','Z','final_all'):
  entries=[]
  for o in ORDERS:
   for s,arrival,w in TERMS[ep]:entries.extend(C[arm][o,s]['stage-cal'][i] for i in ROWS[o[arrival-1]])
  pooled[arm][ep]={'group_appearances':len(entries),**pool(entries)}
 for o in ORDERS:
  for s in (1,2,3):
   for d,rr in ROWS.items():
    stage.append({'arm':arm,'order':o,'stage':s,'actual_domain':d,
       **{n:float(A[arm][o,s]['primary'][rr,COLS.index(n)].mean()) for n in COLS},
       'pooled':pool([C[arm][o,s]['stage-cal'][i] for i in rr])})
result={'scope':'Saved matrices/counts only; no truth labels, BGE, weights, retraining, or external source functions used.',
 'matrix_count_sets':len(verified)//2-1,'file_identities_checked':len(verified),'groups':60,'columns':COLS,
 'bootstrap':{'draws':5000,'seed':20260930,'generator':'PCG64','resampling_unit':'complete group stratified by actual A/B/C domain; shared across fixed orders and arms','quantile':'linear','draws_sha256':hashlib.sha256((OUT/'bootstrap_draws.npy').read_bytes()).hexdigest()},
 'numeric_comparisons':sum(x[2] for x in errs),'max_abs_error':max(x[1] for x in errs),
 'endpoint_statistics':1386,'endpoint_scalar_values':8316,'continuation_checks':checks,
 'endpoints':summaries,'delta':delta,'pooled_stage_cal':pooled,'stage_rows':stage,
 'verified_inputs':verified}
# files include two collection identities, plus 110 matrix/count files
result['matrix_count_sets']=55
(OUT/'saved_statistics.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
with (OUT/'primary_7x22.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=list(stat_rows[0]));w.writeheader();w.writerows(stat_rows)
lines=['# 独立复算：七端点全部22主指标','', '差值＝修订候选−LOGIT0.1。条件区间为固定s0、三个到达顺序、按实际域整群配对bootstrap；不是多指标同时区间。', '']
for ep in TERMS:
 lines+=['## '+ep,'','|指标|候选|LOGIT0.1|差值|条件95%区间|ABC差|BCA差|CAB差|','|---|---:|---:|---:|---|---:|---:|---:|']
 for r in stat_rows:
  if r['endpoint']==ep:lines.append('|'+r['metric']+'|'+ '|'.join(f"{r[k]:.9f}" for k in ['candidate','reference','delta'])+f"|[{r['ci_low']:.9f}, {r['ci_high']:.9f}]|"+'|'.join(f"{r[o]:.9f}" for o in ORDERS)+'|')
 lines+=['']
(OUT/'primary_7x22.zh.md').write_text('\n'.join(lines)+'\n')
print(json.dumps({k:result[k] for k in ('scope','matrix_count_sets','file_identities_checked','numeric_comparisons','max_abs_error','endpoint_statistics','endpoint_scalar_values','continuation_checks','pooled_stage_cal')},ensure_ascii=False,indent=2))
