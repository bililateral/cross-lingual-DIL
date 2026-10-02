"""Web-container result audit. Reads only supplied JSON/NPY/CSV/ZIP; no project imports.
Independent route: sample each actual-domain matrix first, then use explicit
stage/arrival formulas. No production endpoint, bootstrap, or audit functions.
"""
from __future__ import annotations
import argparse,collections,csv,datetime,hashlib,json,math,os,pathlib,platform,sys,time,zipfile
import numpy as np

METRICS=('average_precision','trapezoidal_pr_auc','roc_auc','recall_at_fpr_1pct','brier','log_loss','precision','recall','f1','specificity','balanced_accuracy','mcc','map','mrr','recall_at_1','recall_at_3','recall_at_5','recall_at_10','ndcg_at_1','ndcg_at_3','ndcg_at_5','ndcg_at_10')
ORDERS=('ABC','BCA','CAB'); ROLES=('raw','stage-cal','first-cal','primary'); ARMS={'half':.5,'quarter':.25}; EPS=('O','N','Z','F_first','F','G','final_all')
RANK=list(range(4))+list(range(12,22))
NEW='reports/seller_alias_continual/20261001/er_weight_execution/20261001_144452/job'
OLD='reports/seller_alias_continual/20260930/bge_continual_execution/20260930_143700/job'
RES='reports/seller_alias_continual/20261002/er_weight_result'

class Checks:
 def __init__(self): self.categories=collections.Counter();self.numeric=collections.Counter();self.maxerr={};self.rows=[]
 def require(self, condition, name):
  self.categories[name.split('/')[0]]+=1
  if not condition: raise AssertionError(name)
 def close(self, actual, expected, name, atol=3e-12):
  a=np.asarray(actual,dtype=float);b=np.asarray(expected,dtype=float)
  self.require(a.shape==b.shape,name+'/shape')
  diff=np.abs(a-b);m=float(diff.max(initial=0));cat=name.split('/')[0]
  self.numeric[cat]+=a.size;self.maxerr[cat]=max(self.maxerr.get(cat,0.),m)
  self.rows.append((name,a.size,m,atol))
  self.require(np.isfinite(diff).all() and m<=atol,name+f'/error={m}')
 def tree(self,a,b,name,atol=3e-12):
  if isinstance(b,dict):
   self.require(set(a)==set(b),name+'/keys')
   for k in b:self.tree(a[k],b[k],name+'/'+str(k),atol)
  elif isinstance(b,(tuple,list)):
   self.require(len(a)==len(b),name+'/length')
   for i,v in enumerate(b):self.tree(a[i],v,name+'/'+str(i),atol)
  elif isinstance(b,(int,float)) and not isinstance(b,bool):self.close(a,b,name,atol)
  else:self.require(a==b,name+'/value')

def read(p):return json.loads(p.read_text(encoding='utf-8'))
def hashfile(p):
 b=p.read_bytes();return dict(bytes=len(b),sha256=hashlib.sha256(b).hexdigest())
def dump(p,o):p.write_text(json.dumps(o,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def conf(c):
 tp,fp,fn,tn=(c[k] for k in ('tp','fp','fn','tn'))
 div=lambda a,b:a/b if b else 0.
 rec=div(tp,tp+fn);spe=div(tn,tn+fp)
 return dict(precision=div(tp,tp+fp),recall=rec,f1=div(2*tp,2*tp+fp+fn),specificity=spe,balanced_accuracy=(rec+spe)/2,mcc=div(tp*tn-fp*fn,math.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))))
def point(order,arm,stage):return order+'_shared' if stage==1 else f'{order}_{arm}_stage{stage}'

def formulas(stage_domain,order):
 """stage_domain[stage-1, actual_domain, replicate, metric]."""
 a,b,c=map('ABC'.index,order);x=stage_domain
 out={'O':(x[2,a]+x[2,b])/2,'N':(x[1,b]+x[2,c])/2,'Z':x[2,c].copy(),
      'F_first':x[0,a]-x[2,a], 'F':((x[0,a]-x[2,a])+(x[1,b]-x[2,b]))/2,
      'G':((x[1,b]-x[0,b])+(x[2,c]-x[1,c]))/2, 'final_all':(x[2,0]+x[2,1]+x[2,2])/3}
 for ep in ('F_first','F','G'):out[ep][...,4:6]*=-1
 return out

def manual_quantile(samples):
 ordered=np.sort(samples,axis=0)
 vals=[]
 for q in (.025,.975):
  h=(len(ordered)-1)*q;i=int(math.floor(h));f=h-i
  vals.append(ordered[i]*(1-f)+ordered[min(i+1,len(ordered)-1)]*f)
 return np.stack(vals)

def summarize_by_order(v):
 # index 0 is the unresampled value; indices 1..5000 share actual-domain draws.
 mean=v.mean(axis=0);ci=manual_quantile(mean[1:])
 return {m:dict(mean=float(mean[0,k]),per_order={o:float(v[i,0,k]) for i,o in enumerate(ORDERS)},conditional_95pct_interval=ci[:,k].tolist()) for k,m in enumerate(METRICS)}

def manual_tests(C):
 x=np.zeros((3,3,1,22));x[:,:,:,:]=np.array([[10,20,30],[11,22,33],[8,19,36]])[:,:,None,None]
 expected={'ABC':[13.5,29,36,2,2.5,2.5,21],'BCA':[27.5,20.5,8,1,-1,0,21],'CAB':[22,15,19,-6,-1.5,-1,21]}
 rows=[]
 for o in ORDERS:
  f=formulas(x,o)
  for ep,target in zip(EPS,expected[o],strict=True):
   C.close(f[ep][0,12],target,'manual/formula/'+o+'/'+ep,0.)
   C.close(f[ep][0,4],-target if ep in ('F_first','F','G') else target,'manual/loss_sign/'+o+'/'+ep,0.)
   rows.append(dict(order=o,endpoint=ep,map_expected=target,brier_expected=-target if ep in ('F_first','F','G') else target))
 C.tree(conf(dict(tp=2,fp=1,fn=2,tn=5)),dict(precision=2/3,recall=.5,f1=4/7,specificity=5/6,balanced_accuracy=2/3,mcc=8/math.sqrt(504)),'manual/confusion')
 C.close(manual_quantile(np.arange(5000.)[:,None])[:,0],[124.975,4874.025],'manual/linear_quantile',1e-12)
 C.close(conf(dict(tp=0,fp=0,fn=20,tn=358))['mcc'],0,'manual/zero_denominator',0.)
 return rows

def main():
 p=argparse.ArgumentParser();p.add_argument('--project',type=pathlib.Path,required=True);p.add_argument('--zip',type=pathlib.Path,required=True);p.add_argument('--output',type=pathlib.Path,required=True);p.add_argument('--cpu',type=int,default=0);a=p.parse_args()
 os.sched_setaffinity(0,{a.cpu});a.output.mkdir(parents=True,exist_ok=False);t=time.monotonic();C=Checks();r=a.project;out=a.output;j=r/NEW;old=r/OLD;ev=j/'evaluation';run=j/'run'
 manual=manual_tests(C);dump(out/'hand_reference_results.json',manual)
 # Complete supplied archive and inventory, not only result-file count.
 zin=zipfile.ZipFile(a.zip);names=zin.namelist();inv=read(r/'source_inventory.json')['files']
 C.require(len(names)==len(set(names))==365,'inventory/unique_zip_members');C.require(zin.testzip() is None,'inventory/crc')
 C.require(set(names)=={v['path'] for v in inv}|{'source_inventory.json'},'inventory/exact_members')
 file_rows=[]
 for v in inv:
  h=hashfile(r/v['path']);C.tree(h,{k:v[k] for k in h},'inventory/'+v['path']);file_rows.append({'path':v['path'],**h})
 returned=read(r/RES/'return_inventory.json')
 C.require(len(returned['files'])==returned['file_count']==288,'returned/count')
 C.require(sum(v['bytes'] for v in returned['files'])==returned['total_bytes']==11609562,'returned/bytes')
 for v in returned['files']:C.tree(hashfile(r/v['path']),{k:v[k] for k in ('bytes','sha256')},'returned/'+v['path'])
 manifest=read(run/'manifest.json');sm=read(old/'run/manifest.json');saved=read(ev/'evaluation.json');col=read(ev/'collected.json');ref=read(ev/'reference/collected.json');oc=read(old/'evaluation/collected.json');part=read(run/'partition.json');policy=read(r/'schema/step28_er_weight_policy.json')
 freeze=read(r/'reports/documentation/20261001/er_weight/freeze.json');ex=read(j/'execution.json');obs=read(r/RES/'observation.json')
 src={v['path']:v for v in freeze['source_files']};C.require(len(src)==28,'sources/28')
 for label,items in [('execution',ex['source_files']),('manifest',manifest['source_files']),('collection',col['source_files']),('observation',obs['source_files_verified'])]:
  C.tree({v['path']:v for v in items},src,'sources/'+label)
 for v in src.values():C.tree(hashfile(r/v['path']),{k:v[k] for k in ('bytes','sha256')},'sources/bytes/'+v['path'])
 for n,v in policy['baseline']['records'].items():C.tree(hashfile(old/n),{k:v[k] for k in ('bytes','sha256')},'baseline/'+n)
 def bound(base,v,category):
  path=base/v['path'];C.tree(hashfile(path),{k:v[k] for k in ('bytes','sha256')},category+'/'+v['path']);return path
 gate=read(j/'before_valid.json');completion=read(j/'completion.json')
 C.tree(read(j/'access.json'),dict(train=1,valid=1,heldout=0,owners=0),'execution/access');C.tree(gate['label_parses'],dict(train=1,valid=0,heldout=0,owners=0),'execution/blind_access')
 C.require(gate['status']=='PASS_ER_WEIGHT_COMPLETE_BLIND_GATE','execution/blind_status');bound(j,gate['manifest'],'execution');bound(j,completion['evaluation'],'execution')
 C.require(manifest['status']=='COMPLETE_3456_ER_WEIGHT_UPDATES_VALID_BLIND','execution/status');C.require(manifest['physical_updates']==completion['physical_updates']==3456 and manifest['gradient_group_presentations']==6912,'execution/counts')
 C.require((j/'exit_status.txt').read_text().strip()=='0' and not (j/'failure.json').exists(),'execution/exit')
 C.tree(completion,obs['completion'],'execution/observation_completion')
 C.tree({k:ex[k] for k in policy['baseline']['environment']},policy['baseline']['environment'],'execution/paired_environment')
 started=datetime.datetime.fromisoformat((j/'started.txt').read_text().strip());finished=datetime.datetime.fromisoformat((j/'finished.txt').read_text().strip())
 C.close((finished-started).total_seconds(),38369,'execution/wall_seconds',0.)
 C.require('10:39:29' in (j/'resource_usage.log').read_text() and 'Exit status: 0' in (j/'resource_usage.log').read_text(),'execution/gnu')
 C.tree(part,read(old/'run/partition.json'),'identity/partition')
 for collection in (col,ref,oc):
  C.tree(collection['group_ids'],[v['group_uid'] for v in part['development']],'identity/groups');C.tree(collection['domains'],[v['domain'] for v in part['development']],'identity/domains');C.tree(collection['metric_columns'],list(METRICS),'identity/columns')
 C.require(len(set(col['group_ids']))==60,'identity/unique_groups')
 rows={d:np.where(np.array(col['domains'])==d)[0] for d in 'ABC'}
 C.require(all(len(v)==20 for v in rows.values()),'identity/domain20')
 expected={point(o,m,s) for o in ORDERS for m in ARMS for s in (2,3)};expected_old={point(o,m,s) for o in ORDERS for m in ('seq','er') for s in (1,2,3)}
 for name,v in [('points',manifest['points']),('training',manifest['training']),('collected',col['points'])]:C.require(set(v)==expected,'identity/'+name)
 C.require(set(ref['points'])==expected_old,'identity/reference15')
 scores={};diagnostics={};domain={g['group_uid']:g['domain'] for g in part['fit']};observations=0
 for name in sorted(expected):
  P=read(bound(run,manifest['points'][name],'point_hash'));T=read(bound(run,manifest['training'][name],'training_hash'));o,m,s=P['order'],T['arm'],P['stage'];lam=ARMS[m]
  C.require(name==point(o,m,s)==P['name']==T['name'],'training/point_identity');C.require(P['actual_domain']==T['actual_domain']==o[s-1],'training/domain')
  oldt=read(bound(old/'run',sm['training'][point(o,'er',s)],'baseline_training_hash'))
  C.require(P['history_weight']==T['history_weight']==lam and T['updates']==288 and T['adam_step']==P['completed_updates']==288*s,'training/updates')
  for k in ('current_ids','history_ids','current_dropout_stream','adam_step','updates'):C.tree(T[k],oldt[k],'paired_schedule/'+name+'/'+k)
  fit={u for u,d in domain.items() if d==o[s-1]}
  C.require(len(fit)==48,'training/48fit')
  for epoch in range(6):C.require(set(T['current_ids'][48*epoch:48*(epoch+1)])==fit,'training/each_epoch')
  mem=read(run/'memory'/f'{name}_budget.json');before=T['memory_after_training']
  C.require(len(mem['members'])==6 and mem['members']==before['members']==oldt['memory_after_training']['members'],'memory/members')
  C.require(set(T['history_ids'])<=set(mem['members']) and not fit.intersection(mem['members']),'memory/history_only')
  C.require(mem['seen']==48*(s-1) and mem['draw_count']==before['draw_count']==288,'memory/seen_draw_count');C.require(not mem['with_logits'] and not before['with_logits'],'memory/no_logit')
  C.require(0<mem['serialized_bytes']<=1048576,'memory/budget')
  U=np.load(bound(run,T['update_file'],'update_array_hash'),allow_pickle=False);C.require(U.shape==(288,14) and np.isfinite(U).all(),'updates/shape')
  columns=('current_bce','current_rank','current_hard','current_total','history_bce','history_rank','history_hard','history_total','weighted_history_total','total','encoder_lr','head_lr','gradient_norm','history_weight')
  C.tree(T['update_columns'],list(columns),'updates/column_identity');u={k:U[:,i] for i,k in enumerate(columns)}
  for role in ('current','history'):C.close(u[role+'_total'],u[role+'_bce']+u[role+'_rank']+.5*u[role+'_hard'],'loss_decomposition/'+name+'/'+role,3e-6)
  C.close(u['history_weight'],np.full(288,lam),'updates/lambda/'+name,0.);C.close(u['weighted_history_total'],lam*u['history_total'],'updates/weighted/'+name,0.);C.close(u['total'],u['current_total']+lam*u['history_total'],'updates/total/'+name,0.)
  lr=np.array([1e-5*(k/29 if k<=29 else (288-k)/259) for k in range(1,289)])
  C.close(u['encoder_lr'],lr,'updates/encoder_lr/'+name);C.close(u['head_lr'],np.full(288,.001),'updates/head_lr/'+name,0.)
  C.require(set(T['observations'])=={'1','29','30','288'},'observations/steps')
  for step in (1,29,30,288):
   for module in ('encoder','head'):
    ob=T['observations'][str(step)][module];C.require(ob['finite_nonzero_combined_gradient'] is True and ob['parameters_changed']==(module=='head' or step!=288),'observations/gradient_update');observations+=1
  C.require(P['full_model_adam_and_rng_restore_verified'] is True,'restore/record')
  mp=read(bound(run,P['mapping'],'mapping_hash'))
  C.require(mp['a']>0 and mp['status']=='PASS_CALIBRATION_FIT' and mp['optimizer_success'],'calibration/solver')
  C.require(mp['calibration_group_ids']==[g['group_uid'] for g in part['calibration'] if g['domain']==o[s-1]],'calibration/current_ids')
  C.require(mp['actual_domain']==o[s-1] and (mp['group_count'],mp['pair_count'],mp['positive_count'])==(12,4536,240),'calibration/counts')
  C.require(mp['model_state_sha256']==P['model_state_sha256'] and mp['score_source']==P['scores']['calibration'],'calibration/source')
  shared=read(bound(old/'run',sm['points'][o+'_shared'],'shared_hash'))
  C.tree(P['first_map_parameters'],shared['first_map_parameters'],'calibration/first_map')
  values={k:np.load(bound(run,v,'scores_hash'),allow_pickle=False) for k,v in P['scores'].items()}
  C.require(set(values)=={'calibration','development','stage-cal','first-cal'},'scores/roles')
  for role,v in values.items():C.require(v.shape==((12,378) if role=='calibration' else (60,378)) and v.dtype==(np.float32 if role in ('calibration','development') else np.float64) and np.isfinite(v).all(),'scores/shape_dtype')
  raw=values['development'].astype(float)
  for role,mapping in [('stage-cal',mp),('first-cal',P['first_map_parameters'])]:
   C.close(values[role],mapping['a']*raw+mapping['b'],'affine_scores/'+name+'/'+role,0.)
   ord1=np.argsort(raw,axis=1,kind='stable');ord2=np.argsort(values[role],axis=1,kind='stable');C.require(np.array_equal(ord1,ord2),'affine_order/'+name+'/'+role)
   C.require(np.array_equal(np.diff(np.take_along_axis(raw,ord1,1),axis=1)==0,np.diff(np.take_along_axis(values[role],ord2,1),axis=1)==0),'affine_ties/'+name+'/'+role)
  scores[name]=values
  diagnostics[name]={'lambda':lam,'order':o,'stage':s,'memory_domains':dict(collections.Counter(domain[v] for v in mem['members'])),'memory_bytes':mem['serialized_bytes'],'history_draws_by_domain':dict(collections.Counter(domain[v] for v in T['history_ids'])),'history_draw_count_range':[min(collections.Counter(T['history_ids']).values()),max(collections.Counter(T['history_ids']).values())],'gradient_gt_one':int((u['gradient_norm']>1).sum()),'gradient_norm_mean':float(u['gradient_norm'].mean()),'gradient_norm_min':float(u['gradient_norm'].min()),'gradient_norm_max':float(u['gradient_norm'].max()),'training_seconds':T['training_seconds'],'calibration':{k:mp[k] for k in ('a','b','initial_nll','final_nll')},'native_model_record':P['model'],'cpu_observation_cells':8}
  for ob in obs['model_weights_verified']:
   if ob['name']==name:C.tree({k:ob[k] for k in ('bytes','sha256')},{k:P['model'][k] for k in ('bytes','sha256')},'native_weight_record/'+name)
 for o in ORDERS:
  sh=read(old/'run'/sm['points'][o+'_shared']['path'])
  for m in ARMS:
   st=manifest['restored_starts'][o+'_'+m];C.tree(st['full_checkpoint'],sh['full_checkpoint'],'restore/checkpoint');C.tree(st['model_state_sha256'],sh['model_state_sha256'],'restore/model');C.tree(st['first_map'],sh['first_map_parameters'],'restore/map')
   C.require(st['adam_step']==288 and st['first_scores_replayed_exactly'],'restore/adam_scores');C.tree(st['memory_source'],sm['memories'][o+'_er_after1']['file'],'restore/cache_source');C.tree(st['memory_summary']['members'],sm['memories'][o+'_er_after1']['members'],'restore/cache_members')
   rt=manifest['memories'][point(o,m,2)];C.tree(rt['members'],sm['memories'][point(o,'er',2)]['members'],'memory/after2');C.require(rt['seen']==96 and not rt['with_logits'] and rt['serialized_bytes']<=1048576,'memory/after2_budget')
 C.require(len(obs['model_weights_verified'])==12 and sum(v['bytes'] for v in obs['model_weights_verified'])==15677731710,'native_weight_record/counts_bytes')
 matrices={};counts={};nsets=0
 for collection,folder in [(ref,ev/'reference'),(col,ev)]:
  for name,roles in collection['points'].items():
   C.require(set(roles)==set(ROLES[:3]),'matrix/role_identity');matrices[name]={};counts[name]={}
   for role,recs in roles.items():
    mat=np.load(bound(folder,recs['matrix'],'matrix_hash'),allow_pickle=False);ct=read(bound(folder,recs['counts'],'counts_hash'));C.require(mat.shape==(60,22) and mat.dtype==np.float64 and np.isfinite(mat).all(),'matrix/shape_dtype');C.require(len(ct)==60,'counts/60')
    for i,v in enumerate(ct):
     C.require(all(type(v[k])==int and v[k]>=0 for k in ('tp','fp','fn','tn')) and v['tp']+v['fn']==20 and v['fp']+v['tn']==358,'counts/totals')
     for metric,expect in conf(v).items():C.close(mat[i,METRICS.index(metric)],expect,'counts_algebra/'+name+'/'+role+'/'+metric)
     if name in scores:C.require(int((scores[name]['development' if role=='raw' else role][i]>=0).sum())==v['tp']+v['fp'],'counts/predicted_positives')
    if name in expected_old:C.tree(recs,oc['points'][name][role],'reference/record_identity')
    matrices[name][role]=mat;counts[name][role]=ct;nsets+=1
   for role in ROLES[1:3]:C.close(matrices[name][role][:,RANK],matrices[name]['raw'][:,RANK],'rank_invariance/'+name+'/'+role,0.)
   primary=matrices[name]['stage-cal'].copy();primary[:,RANK]=matrices[name]['raw'][:,RANK];matrices[name]['primary']=primary
 C.require(nsets==81,'matrix/81sets')
 draws=np.random.Generator(np.random.PCG64(20260930)).integers(0,20,size=(5000,3,20));saved_draws=np.load(bound(ev,saved['draws'],'bootstrap_hash'),allow_pickle=False);C.close(draws,saved_draws,'bootstrap/draws',0.);np.save(out/'independent_draws.npy',draws,allow_pickle=False)
 fields={};ends={}
 for arm in ('seq','er','half','quarter'):
  fields[arm]={};ends[arm]={}
  for role in ROLES:
   fo={ep:[] for ep in EPS}
   for order in ORDERS:
    sd=np.empty((3,3,5001,22))
    for stage in (1,2,3):
     mat=matrices[point(order,arm,stage)][role]
     for d,domain_name in enumerate('ABC'):
      block=mat[rows[domain_name]];sd[stage-1,d,0]=block.mean(axis=0)
      sd[stage-1,d,1:]=block[draws[:,d]].mean(axis=1)
    for ep,v in formulas(sd,order).items():fo[ep].append(v)
   fields[arm][role]={ep:np.stack(v) for ep,v in fo.items()};ends[arm][role]={ep:summarize_by_order(v) for ep,v in fields[arm][role].items()}
   C.tree(saved['endpoints'][arm][role],ends[arm][role],'endpoints/'+arm+'/'+role)
  print('INDEPENDENT_ENDPOINTS',arm,flush=True)
 # Identical method minus itself must have zero paired uncertainty, not two marginal intervals.
 C.close(fields['er']['primary']['O']-fields['er']['primary']['O'],np.zeros((3,5001,22)),'bootstrap/self_difference',0.)
 comparisons={};check_details=[];bootout={}
 for cand in ARMS:
  for refarm in ('er','seq'):
   name=cand+'_minus_'+refarm
   delta={ep:summarize_by_order(fields[cand]['primary'][ep]-fields[refarm]['primary'][ep]) for ep in EPS}
   raw={ep:summarize_by_order(fields[cand]['primary'][ep]-fields[refarm]['raw'][ep]) for ep in ('O','N')}
   C.tree(saved['comparisons'][name]['primary'],delta,'comparisons/'+name+'/primary');C.tree(saved['comparisons'][name]['against_raw_reference'],raw,'comparisons/'+name+'/raw')
   checks={};details=[]
   def put(key,val,op,reference='stage-cal',endpoint='',metric=''):
    ok=val>0 if op=='gt0' else val>=0 if op=='ge0' else val<=0
    checks[key]=bool(ok);details.append(dict(comparison=name,check=key,value=float(val),operator=op,passed=bool(ok),reference=reference,endpoint=endpoint,metric=metric))
   put('old_map_improves',delta['O']['map']['mean'],'gt0',endpoint='O',metric='map')
   put('old_map_interval_above_zero',delta['O']['map']['conditional_95pct_interval'][0],'gt0',endpoint='O',metric='map_CI_low')
   put('old_recall5_improves',delta['O']['recall_at_5']['mean'],'gt0',endpoint='O',metric='recall_at_5')
   put('new_map_non_decrease',delta['N']['map']['mean'],'ge0',endpoint='N',metric='map');put('new_recall5_non_decrease',delta['N']['recall_at_5']['mean'],'ge0',endpoint='N',metric='recall_at_5')
   for ep in ('O','N'):
    for metric in ('average_precision','roc_auc','brier','log_loss'):put(f'{ep}_{metric}_non_degradation',delta[ep][metric]['mean'],'le0' if metric in ('brier','log_loss') else 'ge0',endpoint=ep,metric=metric)
    for metric in ('brier','log_loss'):put(f'{ep}_{metric}_against_raw_reference',raw[ep][metric]['mean'],'le0',reference='raw',endpoint=ep,metric=metric)
   for metric in ('map','recall_at_5','average_precision','roc_auc','brier','log_loss'):put(f'Z_{metric}_non_degradation',delta['Z'][metric]['mean'],'le0' if metric in ('brier','log_loss') else 'ge0',endpoint='Z',metric=metric)
   verdict=saved['comparisons'][name]['interpretation'];C.tree(verdict['checks'],checks,'decisions/'+name);C.require(len(checks)==23 and verdict['pilot_observed_checks_pass']==all(checks.values()),'decisions/all');C.require(verdict['failed']==[k for k,v in checks.items() if not v],'decisions/failed_order')
   C.require(verdict['positive_old_map_orders']==sum(v>0 for v in delta['O']['map']['per_order'].values()),'decisions/order_count')
   comparisons[name]=dict(primary=delta,against_raw_reference=raw,checks=checks,passed=sum(checks.values()),failed=[k for k,v in checks.items() if not v]);check_details.extend(details)
   for ep in EPS:bootout[name+'__'+ep]=(fields[cand]['primary'][ep]-fields[refarm]['primary'][ep]).mean(axis=0)[1:]
   print('INDEPENDENT_COMPARISON',name,sum(checks.values()),'of 23','O_MAP',delta['O']['map'],flush=True)
 eligible=[a for a in ARMS if all(comparisons[a+'_minus_er']['checks'].values())];winner=sorted(eligible,key=lambda a:(ends[a]['primary']['O']['map']['mean'],ends[a]['primary']['O']['recall_at_5']['mean'],ARMS[a]),reverse=True)[0] if eligible else 'er'
 C.require(saved['selection']['selected']==winner and saved['selection']['eligible']==eligible and saved['selection']['history_weight']==ARMS.get(winner,1.) and saved['selection']['fallback_used']==(not eligible),'selection/ordered_rule');C.tree(saved['selection'],completion['selection'],'selection/completion')
 # Recheck every saved absolute domain/macro value and pooled count ratio.
 for name,roles in saved['absolute_stage_results'].items():
  for role,obj in roles.items():
   mat=matrices[name][role];ct=counts[name][role]
   C.tree(obj['macro_all'],dict(zip(METRICS,mat.mean(0).tolist())),'absolute/'+name+'/'+role+'/all')
   for d,idx in rows.items():C.tree(obj['macro_by_domain'][d],dict(zip(METRICS,mat[idx].mean(0).tolist())),'absolute/'+name+'/'+role+'/'+d)
   C.require(obj['pooled_fixed_half_classification']['threshold']==0,'pooled/threshold_logit')
   for d,idx in {**rows,'pooled':np.arange(60)}.items():
    total={k:sum(ct[i][k] for i in idx) for k in ('tp','fp','fn','tn')};cf=conf(total);v={**total,**{k:cf[k] for k in ('precision','recall','f1')},'fpr':total['fp']/(total['fp']+total['tn'])}
    actual=obj['pooled_fixed_half_classification']['pooled'] if d=='pooled' else obj['pooled_fixed_half_classification']['by_domain'][d];C.tree(actual,v,'pooled/'+name+'/'+role+'/'+d)
 with bound(ev,saved['stage_metrics'],'csv_hash').open(newline='') as f:
  records=list(csv.DictReader(f));keys=set()
  for row in records:
   key=tuple(row[k] for k in ('order','method','stage','role','actual_domain','metric'));C.require(key not in keys,'stage_csv/unique');keys.add(key)
   n=point(row['order'],row['method'],int(row['stage']));C.require(row['source_point']==n,'stage_csv/source_point');C.close(float(row['history_weight']),ARMS.get(row['method'],1. if row['method']=='er' else 0.),'stage_csv/lambda',0.)
   C.close(float(row['group_macro']),matrices[n][row['role']][rows[row['actual_domain']],METRICS.index(row['metric'])].mean(),'stage_csv/value')
  C.require(len(keys)==7128,'stage_csv/7128')
 # Immutable previous report match for reused SEQ and ER, all roles/endpoints.
 old_eval=read(old/'evaluation/evaluation.json')
 for arm in ('seq','er'):C.tree(old_eval['endpoints'][arm],ends[arm],'old_endpoint_identity/'+arm)
 dump(out/'independent_endpoints.json',ends);dump(out/'independent_comparisons.json',comparisons);dump(out/'diagnostics.json',diagnostics);dump(out/'verified_input_files.json',file_rows)
 with (out/'all_92_checks.csv').open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(check_details[0]));w.writeheader();w.writerows(check_details)
 with (out/'numeric_comparisons.csv').open('w',newline='') as f:
  w=csv.writer(f);w.writerow(['comparison','numbers','maximum_absolute_error','tolerance']);w.writerows(C.rows)
 np.savez_compressed(out/'independent_paired_bootstrap_samples.npz',**bootout)
 result={'status':'PASS_INDEPENDENT_SAVED_RESULT_AUDIT','environment':{'python':sys.version,'numpy':np.__version__,'platform':platform.platform(),'cpu_affinity':sorted(os.sched_getaffinity(0)),'project_linux':False},'input_zip':hashfile(a.zip),'inventory_payloads':len(inv),'returned_files':len(returned['files']),'frozen_sources':len(src),'matrix_sets':nsets,'matrix_cells':nsets*60*22,'observed_module_step_records':observations,'stage_csv_rows':len(records),'numeric_comparisons':sum(C.numeric.values()),'numeric_by_category':dict(C.numeric),'max_absolute_error_by_category':C.maxerr,'checks_by_category':dict(C.categories),'selection':winner,'comparisons':{k:{'passed':v['passed'],'failed':v['failed']} for k,v in comparisons.items()},'elapsed_seconds':time.monotonic()-t,'new_label_access':0,'project_module_imports':[],'native_models_loaded':False,'endpoint_route':'resample each actual domain first; direct stage/arrival equations; average same paired draws over orders; independent sorted linear percentiles','scope':'Saved metric algebra, not label-dependent AP/MAP recomputation or Linux reenactment'}
 dump(out/'summary.json',result);print(json.dumps(result,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
