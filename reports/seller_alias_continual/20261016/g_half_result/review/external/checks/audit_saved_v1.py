#!/usr/bin/env python3
"""Independent external audit of supplied H result bytes only.
Does not import project modules, open labels/models/Memory, or execute training.
Reconstructs ONLY the already registered resampling and schedule choices.
"""
from pathlib import Path
import argparse,csv,hashlib,json,random,sys,platform,time,traceback,math
from collections import Counter
import numpy as np

P=argparse.ArgumentParser();P.add_argument('--input',type=Path,required=True);P.add_argument('--output',type=Path,required=True)
args=P.parse_args();R=args.input.resolve();O=args.output.resolve();O.mkdir(parents=True,exist_ok=True)
started=time.perf_counter();reads={};checks=[];maxdiff={};counts=Counter();results={}
def log(msg): print(msg,flush=True)
def read_bytes(p):
 p=p.resolve(); b=p.read_bytes();reads[str(p.relative_to(R))]={'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()};return b
def J(path): return json.loads(read_bytes(R/path))
def A(path):
 p=R/path;read_bytes(p)
 return np.load(p,allow_pickle=False)
def save(name,obj): (O/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def require(ok,why):
 if not ok:raise AssertionError(why)
 checks.append(why)
def equalnum(a,b,why,atol=1e-12):
 a=np.asarray(a,dtype=np.float64);b=np.asarray(b,dtype=np.float64)
 require(a.shape==b.shape,why+':shape');require(np.array_equal(np.isnan(a),np.isnan(b)),why+':NA')
 require(not np.isinf(a).any() and not np.isinf(b).any(),why+':finite')
 d=float(np.max(np.abs(a[~np.isnan(a)]-b[~np.isnan(b)]))) if (~np.isnan(a)).any() else 0.
 maxdiff[why]=d;counts['numeric_scalars']+=a.size;require(d<=atol,why+':value')
def verify(base,rec):
 name=str(Path(base)/rec['path']);b=read_bytes(R/name)
 require(len(b)==rec['bytes'] and hashlib.sha256(b).hexdigest()==rec['sha256'],'identity:'+name);return name
def seed(*parts):
 b=(json.dumps(list(parts),ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n').encode()
 return int.from_bytes(hashlib.sha256(b).digest()[:8],'big')%(2**63-1)
def mark(name):counts['sections']+=1;log('PASS '+name)

try:
 m=J('result/job/run/manifest.json');old=J('baseline/run/manifest.json');c=J('result/job/evaluation/collected.json');oc=J('baseline/evaluation/collected.json');e=J('result/job/evaluation/evaluation.json')
 before=J('result/job/before_valid.json');done=J('result/job/completion.json');access=J('result/job/access.json');ex=J('result/job/execution.json');gate=J('context/formal_gate.json');bs=J('context/baseline_sources.json')
 split=J('result/job/run/partition.json');osplit=J('baseline/run/partition.json');native=J('context/native_result.json');cpu=J('context/cpu_receipt.json')
 inventory=J('result/inventory.json');ainv=J('result/analysis_inventory.json');receipt=J('result/receive_verification.json')
 for rec in inventory['files']:verify('result',rec)
 for rec in ainv:verify('result',rec)
 require(len(inventory['files'])==147 and sum(x['bytes'] for x in inventory['files'])==4102971,'147 returned files 4102971 bytes')
 require(sum(x['bytes'] for x in ainv)==receipt['analysis']['bytes'],'analysis inventory total')
 require(len(bs['files'])==43,'43 baseline records')
 for rec in bs['files']:
  rel=Path(rec['path']).relative_to(bs['root']);verify('baseline',dict(rec,path=str(rel)))
 require(sum(x['bytes'] for x in bs['files'])==2873124,'baseline bytes 2873124')
 sources=done['source_files'];require(len(sources)==33 and len({r['path'] for r in sources})==33,'33 unique frozen sources')
 for obj in [m,c,e,before,gate,native,cpu]:require(obj['source_files']==sources,'consistent source list')
 require(ex['sources']==sources,'execution sources')
 for rec in sources:verify('result/job/run/sources',rec)
 verify('result/job',before['manifest'])
 verify('context',dict(ex['gate'],path='formal_gate.json'))
 for key,name in [('external_review','implementation_external.zh.txt'),('primary_disposition','implementation_primary.zh.txt'),('cpu','cpu_receipt.json'),('gpu','native_result.json')]:verify('context',dict(gate[key],path=name))
 require(m['baseline_identity']['sha256']==reads['context/baseline_sources.json']['sha256'],'frozen baseline identity')
 for name in ['step28_balanced_pooling.py','step28_balanced_pooling_run.py','step28_record_replay.py','step28_record_replay_numerics.py','step28_alias_ranking.py','step28_replay_diagnostics.py']:
  prior=next(z for z in old['source_files'] if z['path']=='scripts/'+name);verify('result/job/run/sources',prior)
 require(m['public_inputs']==old['public_inputs'],'formal input identities H vs old')
 require(m['pretrained_archive']==old['pretrained_archive'],'pretrained archive identities H vs old')
 require(split==osplit,'exact split equality')
 require(m['initials']['G_half']==bs['initial_identity']==old['initials']['S']==old['initials']['G']==native['native']['initial_state'],'same independent initial model and Adam')
 require(m['starts']['G_half']==m['points']['A_G_half']['continuity'],'own A to B full continuity record')
 require(before['access']==dict(train=1,valid=0,heldout=0,owners=0),'before-valid access')
 require(access==done['access']==dict(train=1,valid=1,heldout=0,owners=0),'completion access')
 require(before['status']=='ALL_BLIND_GATES_PASS','blind gate receipt')
 require(done['physical_updates']==m['physical_updates']==576 and done['gradient_group_presentations']==m['gradient_group_presentations']==864,'manifest counts')
 require(set(m['points'])==set(m['training'])==set(m['diagnostics'])=={'A_G_half','B_G_half'},'two H points only')
 for key in ['group_ids','domains','columns']:require(c[key]==oc[key],'H old '+key+' alignment')
 require(c['group_ids']==[x['group_uid'] for x in split['development']],'group IDs follow split')
 require(c['domains']==[x['domain'] for x in split['development']],'domains follow split')
 require(len(set(c['group_ids']))==40 and Counter(c['domains'])==Counter(A=20,B=20),'40 unique dev groups 20+20')
 require(set(c['group_ids']).isdisjoint({x['group_uid'] for x in split['fit']+split['calibration']}),'fit/calibration disjoint from dev')
 results['identity']={'baseline_files':43,'baseline_bytes':2873124,'frozen_sources':33,'returned_files':147,'initials':m['initials'],'same_initials':True,'own_continuity':True,'access':access,'before_access':before['access']}
 mark('identities and provenance')

 members=m['points']['A_G_half']['memory_summary']['members'];rr=random.Random(seed(20260930,'ABC','retention'));ret=[]
 for seen,uid in enumerate(sorted(x['group_uid'] for x in split['fit'] if x['domain']=='A'),1):
  k=seen-1 if seen<=6 else rr.randrange(seen)
  if k<6:
   if k==len(ret):ret.append(uid)
   else:ret[k]=uid
 require(ret==members,'independent Algorithm R members')
 tr={};train_stats={};presentations=0;steps=0
 for stage,domain in enumerate('AB',1):
  point=domain+'_G_half';t=J(verify('result/job/run',m['training'][point]));tr[point]=t
  ids=[x['group_uid'] for x in split['fit'] if x['domain']==domain];rng=random.Random(seed(20260918,'ABC',stage,'current'));schedule=[]
  for epoch in range(6):chunk=sorted(ids);rng.shuffle(chunk);schedule+=chunk
  require(t['current_ids']==schedule,'independent current schedule '+point)
  require(len(t['rows'])==t['updates']==288 and t['adam_step']==stage*288,'actual step rows '+point)
  rng=random.Random(seed(20260930,'ABC',stage,'history_draws'));hs=[rng.choice(members) for _ in range(288)] if stage==2 else []
  require(t['history_ids']==hs,'independent history schedule '+point)
  for other in ['S','G']:
   ot=J('baseline/run/updates/'+domain+'_'+other+'.json')
   require(ot['current_ids']==t['current_ids'] and ot['history_ids']==t['history_ids'],'external schedule pair '+point+other)
   require(len(ot['rows'])==288,'old row count')
   for i,(a,b) in enumerate(zip(t['rows'],ot['rows'])):
    for role in ['current']+(['history'] if stage==2 else []):require(a[role]['assignment_sha256']==b[role]['assignment_sha256'],'A1 pair '+point+other+str(i)+role)
  errors=[];clipactive=0
  for i,row in enumerate(t['rows'],1):
   require(row['step']==i and row['current_uid']==schedule[i-1],'row/schedule alignment '+point+str(i))
   require(row['history_uid']==(hs[i-1] if stage==2 else None),'row/history alignment '+point+str(i))
   require(row['encoder_lr']==1e-5*(i/29 if i<=29 else (288-i)/259) and row['head_lr']==.001,'LR '+point+str(i))
   require(row['pooling']==gate['pooling'],'pooling identity '+point+str(i))
   require(math.isfinite(row['gradient_norm']) and row['gradient_norm']>0,'finite nonzero combined gradient '+point+str(i));clipactive+=row['gradient_norm']>1
   cur=row['current'];require(cur['mse0']==cur['distillation']==0 and cur['mse1'] is None,'current no teacher')
   require(cur['weighted']==cur['supervised'] and cur['record_gradient_norm']>0,'current weighted/record gradient')
   expected=cur['supervised'];presentations+=1
   if stage==2:
    h=row['history'];require(h['mse0']==h['distillation'] and h['mse0']>0 and h['mse1'] is None and h['record_gradient_norm']>0,'active D0/no D1 training')
    require(abs(h['weighted']-(.1*h['supervised']+.5*h['mse0']))<=2e-6*(1+abs(h['weighted'])),'history weighting')
    expected+=.1*h['supervised']+.5*h['mse0'];presentations+=1
   errors.append(abs(row['total']-expected));require(errors[-1]<=2e-6*(1+abs(expected)),'row total formula');steps+=1
  if stage==2:
   for key in ['members','seen','tables','origins']:
    require(t['memory_before'][key]==t['memory_after'][key]==m['points']['A_G_half']['memory_summary'][key]==m['points']['B_G_half']['memory_summary'][key],'birth memory invariant '+key)
   require(t['memory_before']['count']==0 and t['memory_after']['count']==288,'history count continuity')
  else:require(t['memory_before'] is None and t['memory_after'] is None,'no history during A')
  train_stats[point]={'updates':288,'gradient_presentations':288*stage,'max_total_formula_abs_diff':max(errors),'current_supervised_mean':float(np.mean([x['current']['supervised'] for x in t['rows']])),'nonzero_record_gradients':288*stage,'clip_active_logged_steps':int(clipactive),'encoder_lr_zero_steps':[x['step'] for x in t['rows'] if x['encoder_lr']==0]}
  if stage==2:train_stats[point].update(history_supervised_mean=float(np.mean([x['history']['supervised'] for x in t['rows']])),mean_D0=float(np.mean([x['history']['mse0'] for x in t['rows']])),positive_D0_steps=288)
 require(steps==576 and presentations==864,'independent physical and group count')
 results['training']=train_stats;mark('576 updates, 864 group presentations, schedules and objective logs')

 columns=c['columns'];require(len(columns)==22,'22 metrics');roles=['raw','stage-cal','first-cal'];points=['A_G_half','B_G_half','A_S','B_S','A_G','B_G'];mat={};classifications={}
 domrows={d:np.flatnonzero(np.array(c['domains'])==d) for d in 'AB'}
 draws=np.random.Generator(np.random.PCG64(20260930)).integers(0,20,size=(5000,2,20),dtype=np.int64);np.save(O/'registered_draws.npy',draws,allow_pickle=False)
 def get_stats(x,di):
  if np.isnan(x).any():return [None,None,None],int(np.isnan(x).sum())
  boot=x[draws[:,di,:]].mean(axis=1)
  bounds=np.quantile(boot,[.025,.975],method='linear');return [float(x.mean()),*map(float,bounds)],0
 statistics=[]
 def record_stats(kind,point,role,domain,values,target):
  for j,key in enumerate(columns):
   vals,missing=get_stats(values[:,j],0 if domain=='A' else 1);rec=target[key]
   if missing:require(rec['mean'] is None and rec['conditional_95pct_interval'] is None and rec['undefined_groups']==missing and rec['total_groups']==20,'strict N/A '+str((kind,point,role,domain,key)))
   else:equalnum(vals,[rec['mean'],*rec['conditional_95pct_interval']],'stat '+str((kind,point,role,domain,key)))
   statistics.append([kind,point,role,domain,key,*vals])
 rankcols=[i for i,key in enumerate(columns) if key not in ['brier','log_loss','precision','recall','f1','specificity','balanced_accuracy','mcc']]
 for point in points:
  origin='result/job/evaluation' if point.endswith('G_half') else 'baseline/evaluation';col=c if point.endswith('G_half') else oc
  require(set(col['points'][point])==set(roles),'role set '+point);classifications[point]={}
  for role in roles:
   rec=col['points'][point][role];s=J(verify(origin,rec['matrix']));require(s['columns']==columns,'matrix columns');v=np.array(s['values'],dtype=float);mat[(point,role)]=v;require(v.shape==(40,22),'40x22 metric matrix')
   q=A(verify(origin,rec['queries']));require(q.shape==(40,28,10) and q.dtype==np.float64,'query shape/dtype');equalnum(q.mean(axis=1),v[:,12:],'query average '+point+role)
   ct=J(verify(origin,rec['counts']));ct=np.array([[x[k] for k in ['tp','fp','fn','tn']] for x in ct],dtype=np.int64)
   require(ct.shape==(40,4) and (ct>=0).all() and np.all(ct[:,0]+ct[:,2]==20) and np.all(ct[:,1]+ct[:,3]==358),'count coverage '+point+role)
   rates=[]
   for tp,fp,fn,tn in ct:
    a,b,c0,d=map(int,[tp,fp,fn,tn]);den=(a+b)*(a+c0)*(d+b)*(d+c0)
    rates.append([a/(a+b) if a+b else None,a/(a+c0),2*a/(2*a+b+c0),d/(d+b),(a/(a+c0)+d/(d+b))/2,(a*d-b*c0)/math.sqrt(den) if den else None])
   equalnum(v[:,6:12],rates,'confusion-derived rates '+point+role)
   mask=np.zeros((40,22),bool);mask[:,6]=(ct[:,0]+ct[:,1])==0;mask[:,11]=mask[:,6]
   require(np.array_equal(np.isnan(v),mask),'all-metric missing mask '+point+role)
   target=e['absolute'][point][role]
   for d in 'AB':record_stats('absolute',point,role,d,v[domrows[d]],target['by_domain'][d])
   equalnum(v.mean(0),[target['macro_all'][k] for k in columns],'strict all-domain macro '+point+role)
   require(target['macro_undefined_groups']==dict(zip(columns,np.isnan(v).sum(0).tolist())),'undefined counts '+point+role)
   classifications[point][role]={}
   for domain,rows in [('all',np.arange(40)),*domrows.items()]:
    tp,fp,fn,tn=map(int,ct[rows].sum(0));pooled={'tp':tp,'fp':fp,'fn':fn,'tn':tn,'precision':tp/(tp+fp) if tp+fp else None,'recall':tp/(tp+fn),'f1':2*tp/(2*tp+fp+fn),'fpr':fp/(fp+tn)}
    for k,val in pooled.items():equalnum(val,target['fixed_half_classification'][domain][k],'pooled '+point+role+domain+k)
    classifications[point][role][domain]=pooled
   if role!='raw':equalnum(v[:,rankcols],mat[(point,'raw')][:,rankcols],'rank invariant '+point+role)
   if point.endswith('G_half'):
    p=m['points'][point];sc=A(verify('result/job/run',p['scores'][role]));require(sc.shape==(40,378),'score shape')
    require(np.array_equal((sc>=0).sum(1),ct[:,0]+ct[:,1]),'threshold predicted positive count independent of labels '+point+role)
    if role=='raw':
     require(sc.dtype==np.float32,'raw f32');raw=sc
     for typ in ['raw','calibration']:
      orig=A(verify('result/job/run',p['scores'][typ]));require(orig.shape==((40,378) if typ=='raw' else (12,378)),'restore score shape')
      for pref in ['full','inference']:require(np.array_equal(orig,A(verify('result/job/run',p['scores'][pref+'-'+typ]))),'exact saved restore '+point+pref+typ)
    else:
     mapping=p['maps']['stage' if role=='stage-cal' else 'first'];require(mapping['a']>0,'positive calibration slope');require(np.array_equal(sc,mapping['a']*raw.astype(np.float64)+mapping['b']),'exact positive affine '+point+role)
     require(np.array_equal(np.argsort(sc,kind='stable',axis=1),np.argsort(raw,kind='stable',axis=1)),'stable order preserved '+point+role)
     require(np.array_equal(np.diff(np.sort(sc,axis=1),axis=1)==0,np.diff(np.sort(raw,axis=1),axis=1)==0),'ties preserved '+point+role)
  if point.endswith('G_half'):
   p=m['points'][point];mapping=J(verify('result/job/run',p['mapping']));require(mapping['a']==p['maps']['stage']['a'] and mapping['b']==p['maps']['stage']['b'],'saved map params')
   require(p['maps']['first']==m['points']['A_G_half']['maps']['stage'],'own first map')
   require(p['calibration_ids']==[x['group_uid'] for x in split['calibration'] if x['domain']==point[0]],'calibration IDs')
   require(p['memory_summary']['bytes']<=1048576 and p['memory_summary']['sha256']==p['memory']['sha256']==p['continuity']['memory_sha256'],'memory size and identity')
   require(J('result/job/run/points/'+point+'.json')==p,'point sidecar consistency '+point)
 for other in ['S','G']:
  con='G_half-'+other
  for role in roles:
   for st,kind,key in [('B','final_contrast','comparisons'),('A','initial_contrast','initial_comparisons')]:
    delta=mat[(st+'_G_half',role)]-mat[(st+'_'+other,role)]
    for d in 'AB':record_stats(kind,con,role,d,delta[domrows[d]],e[key][con][role][d])
   f=(mat[('A_G_half',role)]-mat[('B_G_half',role)])-(mat[('A_'+other,role)]-mat[('B_'+other,role)])
   f[:,[columns.index('brier'),columns.index('log_loss')]]*=-1
   record_stats('forgetting_contrast',con,role,'A',f[domrows['A']],e['forgetting_comparisons'][con][role])
 for arm in ['G_half','S','G']:
  for role in roles:record_stats('old_change',arm,role,'A',(mat[('B_'+arm,role)]-mat[('A_'+arm,role)])[domrows['A']],e['old_changes'][arm][role])
 require(len(statistics)==1650,'1650 recomputed rows')
 with (O/'recomputed_statistics.csv').open('w',newline='') as f:
  w=csv.writer(f);w.writerow(['kind','point_or_contrast','role','domain','metric','mean','lower_95','upper_95']);w.writerows(statistics)
 supplied=list(csv.reader((R/'result/analysis/all_metrics.csv').open()));require(len(supplied)==1651,'supplied 1650 CSV rows');sd={tuple(x[:5]):x[5:] for x in supplied[1:]};require(len(sd)==1650,'unique supplied CSV rows')
 for row in statistics:
  equalnum(row[5:],[float(z) if z else None for z in sd[tuple(row[:5])]],'Linux CSV '+str(row[:5]))
 require(e['success']['passed']==all(e['comparisons']['G_half-S']['raw'][d]['map']['conditional_95pct_interval'][0]>0 for d in 'AB'),'dual MAP gate')
 require(e['success']['direct_G_new_B_positive']==(e['comparisons']['G_half-G']['raw']['B']['map']['conditional_95pct_interval'][0]>0),'direct G new B gate')
 results['statistics']={'rows':1650,'NA_rows':sum(x[5] is None for x in statistics),'draw_shape':list(draws.shape),'draw_sha256':hashlib.sha256(draws.tobytes()).hexdigest(),'success':e['success'],'counts_by_kind':dict(Counter(x[0] for x in statistics))}
 results['classification']=classifications;mark('all 1650 statistics, query means, confusion counts, N/A, calibration and saved restores')

 over=J('result/job/diagnostics_evaluation/overfitting.json');diagnoses=[];curves=[];residuals=[]
 for stage,d in enumerate('AB',1):
  point=d+'_G_half';entries=[];fit=[];cache=[];valid=[]
  require(len(m['diagnostics'][point])==6,'six diagnostic entries '+point)
  for epoch,rec in enumerate(m['diagnostics'][point],1):
   entry=J(verify('result/job/run/diagnostics',rec));entries.append(entry)
   require(entry['point']==point and entry['stage']==stage and entry['epoch']==epoch and entry['stage_step']==48*epoch and entry['adam_step']==288*(stage-1)+48*epoch,'diagnostic clocks '+point+str(epoch))
   require(entry['valid_labels_used'] is False and entry['mode']=='A0_eval_no_autograd' and entry['order']=='ABC' and entry['arm']=='G_half','diagnostic mode')
   require(entry['valid_ids']==c['group_ids'],'diagnostic dev group IDs')
   require(entry['fit_ids']==[x['group_uid'] for x in split['fit'] if x['domain']==d] and entry['fit_domains']==[d]*48,'diagnostic fit group IDs')
   require(entry['cache_ids']==(members if stage==2 else []) and entry['cache_domains']==(['A']*6 if stage==2 else []),'diagnostic cache IDs')
   f=A(verify('result/job/run/diagnostics',entry['fit']));k=A(verify('result/job/run/diagnostics',entry['cache']));v=A(verify('result/job/diagnostics_evaluation',over['group_metrics'][point+'_epoch'+str(epoch)]));scores=A(verify('result/job/run/diagnostics',entry['valid_blind']))
   require(f.shape==(48,3) and k.shape==((6 if stage==2 else 0),3) and v.shape==(40,3) and scores.shape==(40,378),'diagnostic array shapes');require(all(np.isfinite(x).all() for x in [f,k,v,scores]),'diagnostic finite')
   fit.append(f);cache.append(k);valid.append(v)
   for role,domain,data_ in [('fit',d,f),*([('cache','A',k)] if len(k) else []),('valid','A',v[domrows['A']]),('valid','B',v[domrows['B']])]:
    for j,metric in enumerate(['map','recall_at_5','log_loss']):curves.append([point,epoch,role,domain,metric,float(data_[:,j].mean())])
   require([z['uid'] for z in entry['teacher_errors']]==entry['cache_ids'],'residual IDs')
   for z in entry['teacher_errors']:
    equalnum(z['residual_difference_mse'],z['D0']+z['D1']-2*z['residual_cross_mean'],'residual identity '+point+str(epoch)+z['uid'])
    require(z['D0']>=0 and z['D1']>=0 and z['residual_difference_mse']>=0 and abs(z['residual_cross_mean'])<=math.sqrt(z['D0']*z['D1'])+1e-12,'residual Cauchy bound')
   if entry['teacher_errors']:
    residuals.append({'point':point,'epoch':epoch,**{key:float(np.mean([z[key] for z in entry['teacher_errors']])) for key in ['D0','D1','residual_cross_mean','residual_difference_mse']}})
  equalnum(valid[-1],mat[(point,'raw')][:,[12,16,5]],'last diagnostic metrics equal endpoint '+point)
  require(np.array_equal(scores,A('result/job/run/'+m['points'][point]['scores']['raw']['path'])),'last diagnostic scores equal endpoint '+point)
  for role,domain,x in [('fit',d,np.stack(fit)),*([('cache','A',np.stack(cache))] if stage==2 else [])]:
   y=np.stack(valid)[:,domrows[domain]];rng=np.random.Generator(np.random.PCG64(seed(20260930,'ABC',stage,role,domain,'diagnostics')))
   ti=rng.integers(0,x.shape[1],size=(5000,x.shape[1]));vi=rng.integers(0,20,size=(5000,20));sign=np.array([1.,1.,-1.])
   tx=(x[-1]-x[0])*sign;vy=(y[-1]-y[0])*sign
   tboot=tx[ti].mean(1);vboot=vy[vi].mean(1);gap=(x.mean(1)-y.mean(1))*sign
   gboot=(x[:,ti].mean(2)-y[:,vi].mean(2))*sign;gb=gboot[-1]-gboot[0];tci=np.quantile(tboot,[.025,.975],axis=0).T;vci=np.quantile(vboot,[.025,.975],axis=0).T;gci=np.quantile(gb,[.025,.975],axis=0).T
   target=over['points'][point][role][domain]
   equalnum(x.mean(1),target['train_raw_curve'],'train curve '+point+role);equalnum(y.mean(1),target['valid_raw_curve'],'valid curve '+point+role);equalnum(gap,target['benefit_gap_curve'],'gap curve '+point+role)
   equalnum(np.quantile(gboot,[.025,.975],axis=1).transpose(1,2,0),target['benefit_gap_conditional_intervals'],'gap CI curve '+point+role)
   for values,ci,key in [(tx.mean(0),tci,'epoch1_to_6_train_benefit'),(vy.mean(0),vci,'epoch1_to_6_valid_benefit'),(gap[-1]-gap[0],gci,'epoch1_to_6_gap_widening')]:equalnum(values,target[key]['mean'],key+point+role);equalnum(ci,target[key]['conditional_95pct_interval'],key+'CI'+point+role)
   for j,metric in enumerate(['map','recall_at_5','log_loss']):
    ts=float(tx[:,j].mean());vs=float(vy[:,j].mean());gs=float(gap[-1,j]-gap[0,j]);label='有相应迹象' if tci[j,0]>0 and vci[j,1]<0 and gci[j,0]>0 else ('证据不足' if ts>0 and vs<0 and gs>0 else '未见明确迹象')
    require(label==target['interpretation'][metric],'diagnostic decision '+point+role+metric)
    diagnoses.append({'point':point,'role':role,'domain':domain,'metric':metric,'train_benefit':ts,'train_ci':tci[j].tolist(),'valid_benefit':vs,'valid_ci':vci[j].tolist(),'gap_widening':gs,'gap_ci':gci[j].tolist(),'interpretation':label})
 require(len(diagnoses)==9 and len(curves)==126,'nine diagnoses and 126 curve values');require(all(x['gap_ci'][0]>0 for x in diagnoses),'nine gap intervals positive')
 save('diagnostics.json',diagnoses);save('residuals.json',residuals)
 with (O/'diagnostic_curves.csv').open('w',newline='') as f:w=csv.writer(f);w.writerow(['point','epoch','role','domain','metric','mean']);w.writerows(curves)
 results['diagnostics']={'sequences':12,'decisions':9,'curve_values':126,'all_nine_gap_intervals_positive':True,'decision_counts':dict(Counter(x['interpretation'] for x in diagnoses))}
 mark('12 diagnostic sequences, 9 decisions, all curve components and residual identities')

 wrapper=(R/'result/job.wrapper.txt').read_text().splitlines();from datetime import datetime
 wall=(datetime.fromisoformat(wrapper[2])-datetime.fromisoformat(wrapper[0])).total_seconds();require(wall==6048 and wrapper[1]=='exit_code=0','formal outer wallclock 6048 exit0')
 aw=J('result/analysis_wallclock.json');nw=J('context/native_wallclock.json');require(aw['charged_seconds']==2 and aw['cumulative_charged_seconds']==6050 and nw['charged_seconds']==320 and cpu['cumulative_charged_seconds']==17,'separate conservative compute ledgers')
 require(aw['source_sha256']==hashlib.sha256(read_bytes(R/'result/step28_g_half_result_audit.py')).hexdigest(),'Linux audit executed-source identity')
 budget=done['budget'];require(budget['elapsed_seconds']<=wall and budget['peak_cuda_reserved_bytes']<=28*2**30 and budget['peak_rss_bytes']<=64*2**30 and budget['peak_output_bytes']<=10*2**30,'formal recorded resources under budget')
 measure=native['native']['measurements'];pr=native['native']['projection'];t=288*(measure['t_A']+measure['t_B']);ev=1446*measure['v']+36*measure['h'];io=measure['L']+2*sum(measure[k] for k in ['s_F','r_F','s_I','r_I']);sec=1.25*(t+ev+io+1800);size=math.ceil(1.1*(measure['W_F']+2*measure['W_I']+2*2**30))
 equalnum([t,ev,io,sec,size],[pr['train_seconds'],pr['forward_seconds'],pr['io_seconds'],pr['projected_total_seconds'],pr['projected_peak_output_bytes']],'qualification projection',atol=1e-9)
 close=J('result/closeout.json');require(not close['failure_exists'] and close['original_pids']=='' and close['related_processes']=='','recorded closeout no active jobs/failure')
 results['budget']={'formal_outer_seconds':wall,'analysis_charged_seconds':2,'cumulative_formal_seconds':6050,'formal_cap_seconds':28800,'cpu_charged_seconds':17,'native_charged_seconds':320,'projection_seconds':sec,'projection_bytes':size,'actual_peak_output_bytes':budget['peak_output_bytes'],'actual_peak_cuda_reserved_bytes':budget['peak_cuda_reserved_bytes'],'actual_peak_rss_bytes':budget['peak_rss_bytes']}
 mark('resource accounting and source-bound qualification projection')
 results['status']='PASS_EXTERNAL_SAVED_EVIDENCE_CHECKS';results['numeric_comparisons']=len(maxdiff);results['numeric_scalars']=counts['numeric_scalars'];results['max_numeric_abs_difference']=max(maxdiff.values(),default=0);results['assertion_count']=len(checks);results['files_read_by_program']=len(reads)
except Exception as exc:
 results['status']='FAIL_EXTERNAL_AUDIT_SCRIPT';results['error']=str(exc);traceback.print_exc();raise
finally:
 results['environment']={'python':platform.python_version(),'numpy':np.__version__,'platform':platform.platform(),'project_native':False,'no_project_modules_imported':True,'no_labels_models_memory_opened':True};results['elapsed_seconds']=time.perf_counter()-started
 save('audit_result.json',results);save('program_read_files.json',reads);save('numeric_differences.json',maxdiff)
 log(json.dumps({k:v for k,v in results.items() if k not in ['classification','training']},ensure_ascii=False))
