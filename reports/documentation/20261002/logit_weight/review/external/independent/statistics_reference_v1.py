"""Independent metric definitions, direct R(t,j) endpoint/paired bootstrap,
and save-before-statistics recovery on handwritten scores plus 45 frozen sets.
Does not read formal labels/text, load models, call execute, or contact a server.
"""
from __future__ import annotations
import copy,csv,hashlib,itertools,json,math,os
from pathlib import Path
from unittest import mock
import numpy as np
import step28_er_weight as m
import step28_er_weight_run as run
import step28_er_weight_evaluate as evaluation
import test_step28_bge_continual_contracts as fixture
ROOT=Path(__file__).resolve().parents[1];SRC=ROOT/'submission';OUT=ROOT/'evidence/statistics_reference';OUT.mkdir(exist_ok=False)
if hasattr(os,'sched_setaffinity'):os.sched_setaffinity(0,{min(os.sched_getaffinity(0))})
p=m.contract('logit');COLS=list(m.metrics.COLUMNS);ORDERS=['ABC','BCA','CAB'];ROLES=['raw','stage-cal','first-cal'];EPNAMES=['O','N','Z','F_first','F','G','final_all']
PAIRS=list(itertools.combinations(range(28),2));rankingcols=list(range(4))+list(range(12,22))
report={'scope':'All new scores and relevance in this reference are HANDWRITTEN, not new LOGIT results. Frozen 45 sets used as received.','numeric_comparisons':0,'maximum_absolute_error':0.,'checks':[]}
def equal(a,b,tol=2e-12,label=''):
 x=np.asarray(a,dtype=float);y=np.asarray(b,dtype=float);assert x.shape==y.shape,(label,x.shape,y.shape)
 error=float(np.max(np.abs(x-y))) if x.size else 0.
 assert error<=tol,(label,error,tol)
 report['numeric_comparisons']+=x.size;report['maximum_absolute_error']=max(report['maximum_absolute_error'],error);return error
def load(path):return json.loads(path.read_text())
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
oldjob=SRC/p['baseline']['local_small_job'];weightjob=SRC/p['weight_reference']['local_small_job']
reference=m.baseline(p,oldjob,weightjob);domains=reference['collected']['domains'];gids=reference['collected']['group_ids'];assert len(gids)==60 and len(set(gids))==60
rows={d:np.array([i for i,x in enumerate(domains) if x==d]) for d in 'ABC'};assert all(len(r)==20 for r in rows.values())
# Handwritten complete cliques, no relationship to the actual formal labels sharing these public IDs.
groups=[fixture.handmade_group(uid,2) for uid in gids]
scores={}
for j,name in enumerate(m.expected_points(p)):
 raw=np.stack([np.round(np.sin(np.arange(378)*(.213+.003*j)+g*.027)*1.2 + np.cos(np.arange(378)*.047+g*.09)*.6-.5+.03*j,2) for g in range(60)]).astype(np.float32)
 scores[name]={'raw':raw,'stage-cal':1.2*raw.astype(np.float64)-.4,'first-cal':.8*raw.astype(np.float64)+.2}
collection=OUT/'handwritten_collection'
# Label/model entry guards remain active during collecting, injected failure, and saved-only recovery.
with mock.patch.object(m.base,'load_model',side_effect=AssertionError('FORBIDDEN model entry')),mock.patch.object(m.base.public,'attach_labels',side_effect=AssertionError('FORBIDDEN label entry')),mock.patch.object(run.prior,'parse_once',side_effect=AssertionError('FORBIDDEN formal parse')):
 run.collect(collection,scores,groups,reference['partition'],m.sources(p),p)
 try:
  with mock.patch.object(evaluation.previous,'bootstrap_draws',side_effect=RuntimeError('independent failure AFTER saving 18 new plus 45 reused sets')):
   evaluation.finalize(collection,oldjob/'evaluation',p,weightjob/'evaluation')
 except RuntimeError as exc:
  assert str(exc).startswith('independent failure AFTER');(OUT/'injected_failure.txt').write_text(str(exc)+'\n')
 else:raise AssertionError('Failure injection did not occur')
 assert len(load(collection/'collected.json')['points'])*3==18
 assert len(load(collection/'reference/collected.json')['points'])*3==45
 assert not (collection/'evaluation.json').exists()
 saved_before={str(f.relative_to(collection)):sha(f) for f in collection.rglob('*') if f.is_file()}
 actual=evaluation.finalize(collection,oldjob/'evaluation',p,weightjob/'evaluation')
 for rel,digest in saved_before.items():assert sha(collection/rel)==digest
 published=(collection/'evaluation.json').read_bytes()
 evaluation.finalize(collection,oldjob/'evaluation',p,weightjob/'evaluation')
 assert (collection/'evaluation.json').read_bytes()==published
report['recovery']={'new_sets_saved_before_failure':18,'reused_sets_saved_before_failure':45,'saved_only_recovery_pass':True,'all_saved_inputs_unchanged':True,'repeated_finalization_byte_identical':True,'formal_parser_calls':0,'model_loader_calls':0}
# Read matrices/counts independently with exact record hashes, not project read_points.
arrays={};counts={};frozen_count=0
for base,coll in [(collection,load(collection/'collected.json')),(collection/'reference',load(collection/'reference/collected.json'))]:
 for name,roles in coll['points'].items():
  arrays[name]={};counts[name]={}
  for role,entries in roles.items():
   for rec in entries.values():
    path=base/rec['path'];assert path.stat().st_size==rec['bytes'] and sha(path)==rec['sha256']
   arrays[name][role]=np.load(base/entries['matrix']['path'],allow_pickle=False)
   counts[name][role]=load(base/entries['counts']['path'])
   assert arrays[name][role].shape==(60,22) and arrays[name][role].dtype==np.float64
   if base!=collection:frozen_count+=1
assert frozen_count==45 and len(arrays)==21

def confusion(tp,fp,fn,tn):
 precision=tp/(tp+fp) if tp+fp else 0.;recall=tp/(tp+fn);spec=tn/(tn+fp)
 den=math.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))
 return {'precision':precision,'recall':recall,'f1':2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.,'specificity':spec,'balanced_accuracy':(recall+spec)/2,'mcc':(tp*tn-fp*fn)/den if den else 0.}
def handwritten_metrics(y,z):
 y=np.asarray(y);z=np.asarray(z,np.float64);n=int(sum(y));neg=len(y)-n
 order=sorted(range(len(y)),key=lambda i:-z[i]);tp=fp=0;pr=1.;rec=0.;ap=pr_area=low=0.;start=0
 while start<len(order):
  end=start+1
  while end<len(order) and z[order[end]]==z[order[start]]:end+=1
  tp+=sum(int(y[k]) for k in order[start:end]);fp+=(end-start)-sum(int(y[k]) for k in order[start:end])
  r=tp/n;q=tp/(tp+fp);ap+=(r-rec)*q;pr_area+=(r-rec)*(pr+q)/2
  if fp/neg<=.01:low=max(low,r)
  pr,rec=q,r;start=end
 pos=z[y==1];negative=z[y==0]
 auc=float(np.mean((pos[:,None]>negative[None,:])+.5*(pos[:,None]==negative[None,:])))
 probability=1/(1+np.exp(-z));prob=np.clip(probability,1e-15,1-1e-15)
 tp=int(sum((z>=0)&(y==1)));fp=int(sum((z>=0)&(y==0)));fn=n-tp;tn=neg-fp
 result={'average_precision':ap,'trapezoidal_pr_auc':pr_area,'roc_auc':auc,'recall_at_fpr_1pct':low,'brier':float(np.mean((probability-y)**2)),'log_loss':float(np.mean(-y*np.log(prob)-(1-y)*np.log(1-prob)))}
 result.update(confusion(tp,fp,fn,tn));queries=[]
 for q in range(28):
  incident=[(i,b if a==q else a) for i,(a,b) in enumerate(PAIRS) if q in (a,b)]
  ranking=sorted(incident,key=lambda e:(-z[e[0]],e[1]));hits=[r for r,(e,_) in enumerate(ranking,1) if y[e]==1]
  values={'map':sum((j+1)/r for j,r in enumerate(hits))/len(hits),'mrr':1/min(hits)}
  for k in (1,3,5,10):
   values[f'recall_at_{k}']=sum(r<=k for r in hits)/len(hits)
   dcg=sum(1/math.log2(r+1) for r in hits if r<=k);ideal=sum(1/math.log2(r+1) for r in range(1,min(k,len(hits))+1))
   values[f'ndcg_at_{k}']=dcg/ideal
  queries.append(values)
 for key in queries[0]:result[key]=sum(x[key] for x in queries)/28
 return np.array([result[key] for key in COLS]),dict(tp=tp,fp=fp,fn=fn,tn=tn)
metric_errors=[]
for name,roles in scores.items():
 for role,z in roles.items():
  errors=[]
  for g in range(60):
   expected,cts=handwritten_metrics(groups[g].labels,z[g]);errors.append(equal(expected,arrays[name][role][g],label=f'{name}/{role}/{g}'))
   assert cts==counts[name][role][g]
  metric_errors.append({'point':name,'role':role,'groups':60,'max_error':max(errors)})
report['handwritten_metric_checks']={'sets':18,'groups':1080,'metrics_per_group':22,'max_error':max(r['max_error'] for r in metric_errors)}
# AP deliberately differs from trapezoidal PR, using four hand-selected edges, unrelated to group constraints.
small_y=np.array([1,0,1,0]);small_z=np.array([4.,3.,2.,1.]);simple=m.metrics.curve_metrics(small_y,small_z)
equal(simple['average_precision'],5/6);equal(simple['trapezoidal_pr_auc'],19/24)
report['ap_vs_pr_hand_example']=simple
# All saved count-derived group ratios, including the frozen observations, independently recomputed.
for name,roles in counts.items():
 for role,cs in roles.items():
  for g,row in enumerate(cs):
   assert row['tp']+row['fn']==20 and row['fp']+row['tn']==358
   for key,value in confusion(**row).items():equal(value,arrays[name][role][g,COLS.index(key)],label='saved counts '+key)
report['count_derived_values']=63*60*6
# PCG64 actual-domain draws -> per-domain frequency weights. No production field/summarize helper used.
draws=np.random.Generator(np.random.PCG64(20260930)).integers(0,20,(5000,3,20))
assert np.array_equal(draws,np.load(collection/'bootstrap_draws.npy',allow_pickle=False))
np.save(OUT/'independent_draws.npy',draws,allow_pickle=False)
freq={d:np.stack([np.bincount(row,minlength=20) for row in draws[:,di]]).astype(np.float64)/20 for di,d in enumerate('ABC')}
def R(method,role,order,t,j):
 name=order+'_shared' if t==1 else f'{order}_{method}_stage{t}'
 mat=arrays[name]['stage-cal' if role=='primary' else role].copy()
 if role=='primary':mat[:,rankingcols]=arrays[name]['raw'][:,rankingcols]
 v=mat[rows[order[j-1]]];return v.mean(0),freq[order[j-1]]@v

def direct_endpoint(method,role,ep):
 individual=[];bs=[]
 for order in ORDERS:
  values={(t,j):R(method,role,order,t,j) for t,j in ((1,1),(1,2),(2,2),(2,3),(3,1),(3,2),(3,3))}
  def formula(index):
   r=lambda t,j:values[t,j][index]
   if ep=='O':return (r(3,1)+r(3,2))/2
   if ep=='N':return (r(2,2)+r(3,3))/2
   if ep=='Z':return r(3,3)
   if ep=='F_first':return r(1,1)-r(3,1)
   if ep=='F':return ((r(1,1)-r(3,1))+(r(2,2)-r(3,2)))/2
   if ep=='G':return ((r(2,2)-r(1,2))+(r(3,3)-r(2,3)))/2
   if ep=='final_all':return (r(3,1)+r(3,2)+r(3,3))/3
   raise AssertionError(ep)
  point,boot=formula(0),formula(1)
  if ep in ('F_first','F','G'):
   point[[4,5]]*=-1;boot[:,[4,5]]*=-1
  individual.append(point);bs.append(boot)
 return np.stack(individual),np.mean(bs,axis=0)
def summary(per_order,boot):
 sorted_boot=np.sort(boot,axis=0);limits=[]
 for q in (.025,.975):
  at=4999*q;lo=int(math.floor(at));hi=int(math.ceil(at));limits.append(sorted_boot[lo]*(hi-at)+sorted_boot[hi]*(at-lo))
 return {key:{'mean':float(per_order[:,k].mean()),'per_order':{o:float(per_order[i,k]) for i,o in enumerate(ORDERS)},'conditional_95pct_interval':[float(v[k]) for v in limits]} for k,key in enumerate(COLS)}
def compare_summary(expected,observed,label):
 for key in COLS:
  equal(expected[key]['mean'],observed[key]['mean'],label=label+'/'+key+'/mean')
  equal(list(expected[key]['per_order'].values()),list(observed[key]['per_order'].values()),label=label+'/'+key+'/order')
  equal(expected[key]['conditional_95pct_interval'],observed[key]['conditional_95pct_interval'],label=label+'/'+key+'/interval')
cache={};independent={'endpoints':{},'comparisons':{}}
for arm in ['seq','quarter','logit_quarter']:
 independent['endpoints'][arm]={}
 for role in ROLES+['primary']:
  independent['endpoints'][arm][role]={}
  for ep in EPNAMES:
   x,b=direct_endpoint(arm,role,ep);cache[arm,role,ep]=(x,b);s=summary(x,b);independent['endpoints'][arm][role][ep]=s
   compare_summary(s,actual['endpoints'][arm][role][ep],f'{arm}/{role}/{ep}')

def check23(delta,raw):
 def v(ep,key):return delta[ep][key]['mean']
 checks={'old_map_improves':v('O','map')>0,'old_map_interval_above_zero':delta['O']['map']['conditional_95pct_interval'][0]>0,'old_recall5_improves':v('O','recall_at_5')>0,'new_map_non_decrease':v('N','map')>=0,'new_recall5_non_decrease':v('N','recall_at_5')>=0}
 for ep in ('O','N'):
  for key in ('average_precision','roc_auc'):checks[f'{ep}_{key}_non_degradation']=v(ep,key)>=0
  for key in ('brier','log_loss'):checks[f'{ep}_{key}_non_degradation']=v(ep,key)<=0
  for key in ('brier','log_loss'):checks[f'{ep}_{key}_against_raw_reference']=raw[ep][key]['mean']<=0
 for key in ('map','recall_at_5','average_precision','roc_auc'):checks[f'Z_{key}_non_degradation']=v('Z',key)>=0
 for key in ('brier','log_loss'):checks[f'Z_{key}_non_degradation']=v('Z',key)<=0
 assert len(checks)==23;return checks
for arm,ref in [('logit_quarter','quarter'),('quarter','seq'),('logit_quarter','seq')]:
 key=arm+'_minus_'+ref;delta={};raw={}
 for mode,epnames,reference_role in [(delta,EPNAMES,'primary'),(raw,['O','N'],'raw')]:
  for ep in epnames:
   a,b=cache[arm,'primary',ep];c,d=cache[ref,reference_role,ep];s=summary(a-c,b-d);mode[ep]=s
   compare_summary(s,actual['comparisons'][key]['primary' if mode is delta else 'against_raw_reference'][ep],key+'/'+ep)
 checks=check23(delta,raw);assert checks==actual['comparisons'][key]['interpretation']['checks']
 independent['comparisons'][key]={'primary':delta,'against_raw_reference':raw,'checks':checks}
 report['checks'].append({'comparison':key,'passed':sum(checks.values()),'total':23,'O_MAP':delta['O']['map'],'N_MAP':delta['N']['map'],'Z_MAP':delta['Z']['map'],'note':'Only quarter_minus_seq is a historical observed comparison; new LOGIT is handwritten.'})
assert 'selection' not in actual
assert actual['method_checks']['increment_against_matched_er_passes']==all(independent['comparisons']['logit_quarter_minus_quarter']['checks'].values())
assert actual['method_checks']['all_guards_against_seq_pass']==all(independent['comparisons']['logit_quarter_minus_seq']['checks'].values())
# Previously published quarter-SEQ check identity and numeric summaries remain unchanged.
published_old=load(weightjob/'evaluation/evaluation.json')['comparisons']['quarter_minus_seq']
assert published_old['interpretation']==actual['comparisons']['quarter_minus_seq']['interpretation']
for ep in EPNAMES:compare_summary(independent['comparisons']['quarter_minus_seq']['primary'][ep],published_old['primary'][ep],'published quarter-SEQ '+ep)
# A nonconstant diagnostic where LOGIT beats matched ER but still loses to SEQ.
new={};old={}
for order in ORDERS:
 shared=np.tile(np.linspace(.21,.25,22),(60,1));old[order+'_shared']={r:shared.copy() for r in ROLES}
 for stage in (2,3):
  for arm,value in [('seq',.6),('quarter',.5),('logit_quarter',.55)]:
   mat=value+np.arange(60)[:,None]*.0001+np.arange(22)[None,:]*.00001;mat[:,[4,5]]=1-mat[:,[4,5]]
   (new if arm=='logit_quarter' else old)[f'{order}_{arm}_stage{stage}']={r:mat.copy() for r in ROLES}
logic=evaluation.evaluate_matrices(new,old,domains,p)
assert logic['method_checks']['increment_against_matched_er_passes'] and not logic['method_checks']['all_guards_against_seq_pass'] and 'selection' not in logic
report['better_ER_still_worse_SEQ_fixture']={'method_checks':logic['method_checks'],'matched_ER_checks':sum(logic['comparisons']['logit_quarter_minus_quarter']['interpretation']['checks'].values()),'SEQ_checks':sum(logic['comparisons']['logit_quarter_minus_seq']['interpretation']['checks'].values()),'matched_ER_O_MAP':logic['comparisons']['logit_quarter_minus_quarter']['primary']['O']['map']['mean'],'SEQ_O_MAP':logic['comparisons']['logit_quarter_minus_seq']['primary']['O']['map']['mean']}
# Independently break each of the 23 conjuncts without breaking any other conjunct.
delta=copy.deepcopy(logic['comparisons']['logit_quarter_minus_quarter']['primary']);raw=copy.deepcopy(logic['comparisons']['logit_quarter_minus_quarter']['against_raw_reference'])
base_checks=check23(delta,raw);assert all(base_checks.values());mutations=[]
for name in base_checks:
 d=copy.deepcopy(delta);r=copy.deepcopy(raw)
 mapping={'old_map_improves':('O','map','mean',0.),'old_map_interval_above_zero':('O','map','interval',0.),'old_recall5_improves':('O','recall_at_5','mean',0.),'new_map_non_decrease':('N','map','mean',-.001),'new_recall5_non_decrease':('N','recall_at_5','mean',-.001)}
 if name in mapping:
  ep,metric,field,value=mapping[name]
  if field=='interval':d[ep][metric]['conditional_95pct_interval'][0]=value
  else:d[ep][metric][field]=value
 else:
  ep=name[0];suffix='_against_raw_reference' if name.endswith('_against_raw_reference') else '_non_degradation';metric=name[2:-len(suffix)]
  (r if suffix=='_against_raw_reference' else d)[ep][metric]['mean']=.001 if metric in ('brier','log_loss') else -.001
 expected=check23(d,r);observed=evaluation.previous.comparison_checks(d,r)['checks']
 assert expected==observed and [k for k,v in expected.items() if not v]==[name],(name,expected)
 mutations.append(name)
report['independent_23_guard_mutations']=mutations
report['stage_table_rows']=len((collection/'stage_metrics.csv').read_text().splitlines())-1;assert report['stage_table_rows']==5346
report['endpoint_summaries']=3*4*7*22;report['comparison_summaries']=3*9*22;report['status']='PASS';report['cpu_affinity']=sorted(os.sched_getaffinity(0))
(OUT/'independent_statistics.json').write_text(json.dumps(independent,ensure_ascii=False,indent=2)+'\n')
(OUT/'metrics_per_set.json').write_text(json.dumps(metric_errors,ensure_ascii=False,indent=2)+'\n')
(OUT/'results.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n');print(json.dumps(report,ensure_ascii=False,indent=2))
