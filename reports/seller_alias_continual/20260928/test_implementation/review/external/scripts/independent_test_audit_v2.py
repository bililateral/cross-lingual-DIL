"""Independent handwritten reference audit of the fixed-s0 test implementation.

Only toy CSV/text/tensors are generated. Never calls execute, accesses native model
assets, infers sealed labels, or fits a calibration. Project routines are the
system under test; expected metrics/bootstrap/guards are separately implemented.
The frozen float64 probability expression is deliberately shared as a numerical
interface; all classification, curve, retrieval and resampling algebra is separate.
"""
from __future__ import annotations
import copy,csv,hashlib,itertools,json,math,os,platform,shutil,sys,time,traceback
from contextlib import ExitStack
from pathlib import Path
from unittest import mock
import numpy as np

B=Path('/mnt/data/test_pretest_audit_20260928'); R=B/'source'; OUT=B/'evidence/outputs/independent_v2'
OUT.mkdir(parents=True,exist_ok=False)
sys.path.insert(0,str(R/'scripts'))
import step28_alias_test as target
import step28_alias_test_run as runner
import torch
torch.set_num_threads(1)
try: torch.set_num_interop_threads(1)
except RuntimeError: pass
D=runner.data
COLS=('average_precision','trapezoidal_pr_auc','roc_auc','recall_at_fpr_1pct','brier','log_loss','precision','recall','f1','specificity','balanced_accuracy','mcc','map','mrr','recall_at_1','recall_at_3','recall_at_5','recall_at_10','ndcg_at_1','ndcg_at_3','ndcg_at_5','ndcg_at_10')
ROLES=('A_raw','A_cal','C_raw','C_cal'); COMPS=(('C_cal','A_cal'),('C_cal','A_raw'),('C_cal','C_raw'),('A_cal','A_raw'))
EDGES=list(itertools.combinations(range(28),2)); K=(1,3,5,10)
RESULTS={}; NUMERICS={}; fixture_state={}

def jwrite(path,value):
 path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False))
def fhash(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def check_close(label,actual,expected,tolerance=1e-12):
 a,b=np.asarray(actual,dtype=float),np.asarray(expected,dtype=float)
 assert a.shape==b.shape,(label,a.shape,b.shape)
 assert np.isfinite(a).all() and np.isfinite(b).all(),label
 diff=float(np.max(np.abs(a-b))) if a.size else 0.
 n=NUMERICS.setdefault(label,{'values':0,'max_abs_difference':0.,'tolerance':tolerance})
 n['values']+=a.size;n['max_abs_difference']=max(n['max_abs_difference'],diff)
 assert diff<=tolerance,(label,diff,tolerance)
def rejected(fn,exception=(ValueError,FileNotFoundError,FileExistsError)):
 try: fn()
 except exception as e:return {'type':type(e).__name__,'message':str(e)}
 raise AssertionError('Invalid hand-made case was not rejected')
def quantile_linear(v):
 a=np.sort(np.asarray(v),axis=0);ans=[]
 for q in (.025,.975):
  h=(len(a)-1)*q;lo=math.floor(h);hi=math.ceil(h)
  ans.append(a[lo]+(a[hi]-a[lo])*(h-lo))
 return np.array(ans)
def ref_counts(y,z,threshold=0.):
 tp=fp=fn=tn=0
 for yy,zz in zip(y,z,strict=True):
  if float(zz)>=float(threshold):
   if yy:tp+=1
   else:fp+=1
  else:
   if yy:fn+=1
   else:tn+=1
 return {'tp':tp,'fp':fp,'fn':fn,'tn':tn}
def ref_rates(c):
 t,f,n,u=(float(c[k]) for k in ('tp','fp','fn','tn'))
 p=t/(t+f) if t+f else 0.;r=t/(t+n) if t+n else 0.;s=u/(u+f) if u+f else 0.
 den=math.sqrt((t+f)*(t+n)*(u+f)*(u+n))
 return {'precision':p,'recall':r,'specificity':s,'f1':2*t/(2*t+f+n) if 2*t+f+n else 0.,'balanced_accuracy':(r+s)/2,'mcc':(t*u-f*n)/den if den else 0.,'fpr':f/(f+u) if f+u else 0.}
def ref_metrics(y,z):
 y=[int(v) for v in y];z=[float(v) for v in z];P=sum(y);N=len(y)-P
 order=sorted(range(len(z)),key=lambda i:(-z[i],i));tp=fp=0;last_r=0.;last_p=1.;ap=pr=low=0.
 for score,block in itertools.groupby(order,key=lambda i:z[i]):
  ix=list(block);tp+=sum(y[i] for i in ix);fp+=len(ix)-sum(y[i] for i in ix)
  recall=tp/P;precision=tp/(tp+fp)
  ap+=(recall-last_r)*precision;pr+=(recall-last_r)*(precision+last_p)/2
  if fp/N<=.01:low=max(low,recall)
  last_r,last_p=recall,precision
 pos=[z[i] for i in range(len(y)) if y[i]];neg=[z[i] for i in range(len(y)) if not y[i]]
 auc=math.fsum(1. if p>n else .5 if p==n else 0. for p in pos for n in neg)/(P*N)
 # Explicitly keep the frozen float64 probability primitive; not a new sigmoid.
 probs=np.exp(-np.logaddexp(0.,-np.asarray(z,np.float64)))
 clipped=np.clip(probs,1e-15,1-1e-15)
 brier=math.fsum((float(p)-yy)**2 for p,yy in zip(probs,y))/len(y)
 nll=-math.fsum(math.log(float(p)) if yy else math.log1p(-float(p)) for p,yy in zip(clipped,y))/len(y)
 counts=ref_counts(y,z); rates=ref_rates(counts)
 score_map={e:s for e,s in zip(EDGES,z)};label_map={e:v for e,v in zip(EDGES,y)}
 qr=[];hits=[]
 for query in range(28):
  other=[i for i in range(28) if i!=query]
  other.sort(key=lambda j:(-score_map[tuple(sorted((query,j)))],j))
  rel=[label_map[tuple(sorted((query,j)))] for j in other];total=sum(rel);assert total in (1,2)
  relevant=[i+1 for i,x in enumerate(rel) if x]
  row=[math.fsum((j+1)/rank for j,rank in enumerate(relevant))/total,1./relevant[0]]
  row.extend(sum(rel[:k])/total for k in K)
  row.extend(math.fsum(rel[i]/math.log2(i+2) for i in range(k))/math.fsum(1./math.log2(i+2) for i in range(min(total,k))) for k in K)
  qr.append(row);hits.append([int(any(rel[:k])) for k in K])
 values={'average_precision':ap,'trapezoidal_pr_auc':pr,'roc_auc':auc,'recall_at_fpr_1pct':low,'brier':brier,'log_loss':nll,**rates}
 rvalues=[math.fsum(row[i] for row in qr)/28 for i in range(10)]
 return np.array([values[k] for k in COLS[:12]]+rvalues),counts,np.asarray(hits).mean(0)
def truth_for(g):
 controllers=[i for i in range(8) for _ in range(2)]+[i for i in range(8,12) for _ in range(3)]
 np.random.Generator(np.random.PCG64(1103+g)).shuffle(controllers)
 return np.array([controllers[a]==controllers[b] for a,b in EDGES],np.uint8)
def record(path,root):return {'path':path.relative_to(root).as_posix(),'bytes':path.stat().st_size,'sha256':fhash(path)}

def make_fixture(root):
 dataset=root/'data';(dataset/'heldout/supervision').mkdir(parents=True)
 meta=[];groups=[];items=[];rows=[];truth=[]
 for split,count in [('train',60),('development',20),('heldout',40)]:
  for domain in 'ABC':
   for i in range(count):
    uid=f'toy_{split}_{domain}_{i:03d}'; sellers=tuple(f'{uid}_s{j:02d}' for j in range(28))
    nitems=sum(2+(j+i)%7 for j in range(28))
    meta.append({'domain':domain,'split':split,'group_uid':uid,'group_index':i,'accounts':28,'items':nitems})
    if split!='heldout':continue
    allitems=[]
    for j,s in enumerate(sellers):
     account=[]
     for t in range(2+(j+i)%7):
      title='手工标题'+str((j*3+t+i)%13);desc='手工描述'+str((j+t*7+i)%17)
      item=(f'{s}_i{t:02d}',title,desc);account.append(item)
      items.append(dict(group_uid=uid,seller_uid=s,item_uid=item[0],title=title,description=desc))
     allitems.append(tuple(account))
    groups.append(D.Group(uid,sellers,tuple(allitems)))
    yy=truth_for(len(groups));truth.append(yy)
    for (a,b),v in zip(EDGES,yy):rows.append(dict(group_uid=uid,seller_uid_left=sellers[a],seller_uid_right=sellers[b],label=int(v)))
 with (dataset/'groups.csv').open('w',newline='',encoding='utf-8') as f:
  w=csv.DictWriter(f,fieldnames=['domain','split','group_uid','group_index','accounts','items']);w.writeheader();w.writerows(reversed(meta))
 with (dataset/'heldout/items.jsonl').open('w',encoding='utf-8') as f:
  for it in reversed(items): f.write(json.dumps(it,ensure_ascii=False)+'\n')
 with (dataset/'heldout/supervision/pairs.csv').open('w',newline='',encoding='utf-8') as f:
  w=csv.DictWriter(f,fieldnames=['group_uid','seller_uid_left','seller_uid_right','label']);w.writeheader();w.writerows(reversed(rows))
 p=copy.deepcopy(runner.contract());p['data']['root']='data'
 p['data']['inputs']={name:record(dataset/name,root) for name in ('groups.csv','heldout/items.jsonl','heldout/supervision/pairs.csv')}
 jwrite(dataset/'manifest.json',{'study':p['data']['study'],'root_seed':p['data']['root_seed'],'files':{n:{k:r[k] for k in ('bytes','sha256')} for n,r in p['data']['inputs'].items()}})
 jwrite(dataset/'validation.json',{'status':'PASS_GENERATION_CONTRACT_NOT_MODEL_QUALIFICATION'})
 p['data']['manifest']=record(dataset/'manifest.json',root);p['data']['validation']=record(dataset/'validation.json',root)
 truth=np.stack(truth); rng=np.random.Generator(np.random.PCG64(1903))
 A=rng.normal(-2.,1.2,truth.shape).astype(np.float32);C=(A+truth*.45+rng.normal(0,.12,truth.shape)).astype(np.float32)
 A[0]=0.;C[0]=0.;A[1]=np.round(A[1],1);C[1]=np.round(C[1],1)
 # Include saturation without introducing nonfinite raw scores.
 A[2,0]=-1000.;A[2,1]=1000.;C[2,0]=-999.;C[2,1]=1001.
 info={'A':{'map':{'a':.625,'b':-.25},'threshold':float(np.nextafter(np.float64(.1),np.inf))},'C':{'map':{'a':1.25,'b':.5},'threshold':.25}}
 out=root/'job';out.mkdir()
 blind={'policy_sha256':D.sha256(runner.POLICY),'group_ids':[g.uid for g in groups],'domains':['ABC'[i//40] for i in range(120)],'models':{}}
 for name,raw in [('A',A),('C',C)]:
  p['models'][name]['payload']={'handmade':name};p['models'][name]['calibration']={'handmade_map':name}
  blind['models'][name]=runner.save_scores(out,name,raw,info[name]['map'],{'model':p['models'][name]['payload'],'map':p['models'][name]['calibration']})
 D.write_json(out/'blind.json',blind)
 return dict(root=root,p=p,groups=groups,info=info,blind=blind,truth=truth,out=out)

def ref_guard(comps):
 x=comps['C_cal_minus_A_cal'];r=comps['C_cal_minus_A_raw']
 vals=[x['map']['mean'],x['map']['conditional_95pct_interval'][0],x['recall_at_5']['mean'],x['average_precision']['mean'],x['roc_auc']['mean'],x['brier']['mean'],x['log_loss']['mean'],r['brier']['mean'],r['log_loss']['mean']]
 assert all(math.isfinite(v) for v in vals)
 tests=[vals[0]>=.01,vals[1]>0,vals[2]>0,vals[3]>=0,vals[4]>=0,vals[5]<=0,vals[6]<=0,vals[7]<=0,vals[8]<=0]
 return [f'T{i+1}' for i,t in enumerate(tests) if not t]

def test_01_criteria():
 def passing():
  metrics={m:{'mean':0.,'conditional_95pct_interval':[0.,0.]} for m in COLS}
  metrics['map']={'mean':.01,'conditional_95pct_interval':[.001,.02]};metrics['recall_at_5']['mean']=float(np.nextafter(0.,np.inf))
  return {'C_cal_minus_A_cal':copy.deepcopy(metrics),'C_cal_minus_A_raw':copy.deepcopy(metrics)}
 fields=[('C_cal_minus_A_cal','map','mean'),('C_cal_minus_A_cal','map','lower'),('C_cal_minus_A_cal','recall_at_5','mean'),('C_cal_minus_A_cal','average_precision','mean'),('C_cal_minus_A_cal','roc_auc','mean'),('C_cal_minus_A_cal','brier','mean'),('C_cal_minus_A_cal','log_loss','mean'),('C_cal_minus_A_raw','brier','mean'),('C_cal_minus_A_raw','log_loss','mean')]
 observed=[]
 assert target.acceptance(passing())['failed']==[]
 for i,(c,m,s) in enumerate(fields):
  v=passing();new=np.nextafter(.01,-np.inf) if i==0 else 0. if i in (1,2) else np.nextafter(0.,-np.inf if i in (3,4) else np.inf)
  if s=='lower':v[c][m]['conditional_95pct_interval'][0]=float(new)
  else:v[c][m]['mean']=float(new)
  got=target.acceptance(v);assert got['failed']==ref_guard(v)==[f'T{i+1}'];observed.append(got)
 v=passing();v['C_cal_minus_A_cal']['brier']['mean']=-.01;v['C_cal_minus_A_cal']['log_loss']['mean']=-.1
 v['C_cal_minus_A_raw']['brier']['mean']=.001;v['C_cal_minus_A_raw']['log_loss']['mean']=.01
 assert target.acceptance(v)['failed']==ref_guard(v)==['T8','T9']
 v['C_cal_minus_A_cal']['map']['mean']=float('nan');rejected(lambda:target.acceptance(v))
 jwrite(OUT/'nine_criteria.json',observed)
 return {'single_failure_cases':9,'zero_tolerance_nextafter_checked':True,'bad_calibrated_control_rejected':True}

def test_02_metrics_scalar():
 ys=np.stack([truth_for(i) for i in range(8)])
 rng=np.random.Generator(np.random.PCG64(2803));zs=rng.normal(-1,3,ys.shape)
 zs[0]=0.;zs[1]=np.round(zs[1]);zs[2]=np.where(ys[2],5.,-5.);zs[3]=-zs[2]
 zs[4]=np.resize(np.array([-1000.,-40.,-1.,0.,1.,40.,1000.]),378)
 zs[5]=np.linspace(-2,2,378);zs[6]=np.nextafter(np.zeros(378),np.inf);zs[7]=np.full(378,-2.)
 got,counts=target.metrics.group_metrics(ys,zs);expected=[];hits=[]
 for i,(y,z) in enumerate(zip(ys,zs)):
  ref,c,h=ref_metrics(y,z);expected.append(ref);hits.append(h);assert c==counts[i]
 check_close('22_metrics_scalar',got,expected)
 assert abs(got[0,COLS.index('recall_at_1')]-hits[0][0])>0
 assert abs(got[0,COLS.index('average_precision')]-got[0,COLS.index('trapezoidal_pr_auc')])>0
 np.savez(OUT/'handmade_metric_reference.npz',truth=ys,scores=zs,expected=expected,actual=got,hit_at_k=hits)
 return {'handmade_cases':8,'metrics':22,'probability_interface':'frozen numpy float64 exp(-logaddexp), independently aggregated','r1_not_hit1':True,'ap_not_trapezoid':True}

def test_03_bootstrap():
 # Domain order intentionally interleaved to detect accidental contiguous slicing.
 domains=list('ABC'*40);rng=np.random.Generator(np.random.PCG64(811))
 mats={r:rng.uniform(.05,.85,(120,22)) for r in ROLES}
 for i,r in enumerate(ROLES):mats[r]+=(np.arange(120)[:,None]%5)*.0003*i
 actual_draws=target.bootstrap_draws()
 rr=np.random.Generator(np.random.PCG64(20260928));draws=np.empty((5000,3,40),dtype=np.int64);freq=np.zeros((5000,120),dtype=np.int64)
 indices=[[i for i,x in enumerate(domains) if x==d] for d in 'ABC']
 for b in range(5000):
  for d in range(3):
   ix=rr.integers(0,40,size=40);draws[b,d]=ix
   for j in ix:freq[b,indices[d][int(j)]]+=1
 assert np.array_equal(draws,actual_draws)
 assert (freq[:,np.array(indices)].sum(2)==40).all()
 out=target.summarize(mats,domains,actual_draws);saved={}
 for a,c in COMPS:
  key=a+'_minus_'+c;delta=mats[a]-mats[c];boot=freq@delta/120.;interval=quantile_linear(boot)
  for k,m in enumerate(COLS):
   rec=out['comparisons'][key][m]
   check_close('bootstrap_intervals',rec['conditional_95pct_interval'],interval[:,k])
   check_close('bootstrap_means',rec['mean'],math.fsum(delta[:,k])/120.)
   for d,ix in zip('ABC',indices):check_close('bootstrap_domain',rec['by_domain'][d],math.fsum(delta[ix,k])/40.)
  saved[key]=boot
 assert out['acceptance']['failed']==ref_guard(out['comparisons'])
 np.save(OUT/'independent_draws.npy',draws);np.savez_compressed(OUT/'independent_bootstrap.npz',**saved)
 jwrite(OUT/'bootstrap_comparisons.json',out)
 rejected(lambda:target.compare(mats['A_raw'],mats['C_raw'],domains,draws[:3000]))
 return {'replicates':5000,'seed':20260928,'shape':[5000,3,40],'comparisons':4,'metrics':22,'interleaved_domains':True,'draws_equal':True}

def test_04_public_and_pipeline():
 global fixture_state
 f=make_fixture(OUT/'toy_pipeline');fixture_state=f
 groups,metadata=runner.public_inputs(f['p'],f['root']);assert groups==f['groups']
 assert all(g.labels is None for g in groups)
 truth,alignment=runner.parse_once(f['out'],f['p'],groups,f['blind'],f['info'],f['root'])
 assert np.array_equal(truth,f['truth']);assert alignment['positive_pairs']==2400 and alignment['pairs']==45360
 rejected(lambda:runner.parse_once(f['out'],f['p'],groups,f['blind'],f['info'],f['root']),FileExistsError)
 scores=runner.restore_scores(f['out'],f['blind'],f['p'],f['info']);snapshot=runner.sources()
 captured=runner.collect(f['out'],truth,scores,f['blind'],f['info'],snapshot)
 assert len(captured['points'])==4
 # Full reference of every metric, not just aggregate values of the target.
 reference_mats={}
 for role in ROLES:
  references=[];counts=[]
  for y,z in zip(truth,scores[role]):
   ref,c,_=ref_metrics(y,z);references.append(ref);counts.append(c)
  got=np.load(f['out']/'evaluation'/captured['points'][role]['file']['path'],allow_pickle=False)
  check_close('full_handmade_metrics',got,references);assert counts==captured['points'][role]['counts']
  reference_mats[role]=np.array(references)
 f['scores']=scores;f['snapshot']=snapshot
 shutil.copytree(f['out'],OUT/'captured_before_statistics')
 result=runner.finalize(f['out'],snapshot);f['result']=result
 rejected(lambda:runner.finalize(f['out'],snapshot),FileExistsError)
 assert result['acceptance']['failed']==ref_guard(result['comparisons'])
 # Fixed classifications and original-threshold counts independently reconstructed.
 for role in ROLES:
  cs=[ref_counts(y,z) for y,z in zip(truth,scores[role])]
  pooled={k:sum(c[k] for c in cs) for k in ('tp','fp','fn','tn')};rates=ref_rates(pooled)
  rec=result['points'][role]['fixed_classification']['pooled'];assert {k:rec[k] for k in pooled}==pooled
  for k in ('fpr','recall','precision','f1'):check_close('pooled_counts',rec[k],rates[k])
 for name in ('A','C'):
  cs=[ref_counts(y,z,f['info'][name]['threshold']) for y,z in zip(truth,scores[name+'_raw'])]
  assert result['automatic'][name]['counts']==[[c[k] for k in ('tp','fp','fn','tn')] for c in cs]
 np.savez_compressed(OUT/'full_handmade_reference_matrices.npz',**reference_mats)
 jwrite(OUT/'toy_pipeline_summary.json',{'alignment':alignment,'acceptance':result['acceptance'],'note':'Handmade only; not formal test results'})
 return {'matrices':4,'shape':[120,22],'full_metrics':10560,'toy_label_parse':1,'formal_label_parse':0,'account_item_counts':'2 through 8','shuffled_rows_aligned':True}

def test_05_prelabel_rejections():
 f=fixture_state;out=f['out'];errors={}
 access=out/'heldout_access.json';old_access=access.read_bytes();access.unlink()
 try:
  for role in ROLES:
   path=out/(role+'_scores.npy');old=path.read_bytes();changed=bytearray(old);changed[-1]^=1;path.write_bytes(changed)
   try:errors['byte_flip_'+role]=rejected(lambda:runner.parse_once(out,f['p'],f['groups'],f['blind'],f['info'],f['root']));assert not access.exists()
   finally:path.write_bytes(old)
  blind_path=out/'blind.json';old_blind=blind_path.read_bytes()
  for case in ('missing_A','missing_C','reversed_groups','missing_cal_role','map_binding'):
   bad=copy.deepcopy(f['blind'])
   if case.startswith('missing_') and case[-1] in 'AC': del bad['models'][case[-1]]
   elif case=='reversed_groups':bad['group_ids'].reverse()
   elif case=='missing_cal_role':del bad['models']['C']['files']['C_cal']
   else:bad['models']['C']['origin']['map']={'wrong':'mapping'}
   D.write_json(blind_path,bad)
   try:errors[case]=rejected(lambda:runner.parse_once(out,f['p'],f['groups'],bad,f['info'],f['root']));assert not access.exists()
   finally:blind_path.write_bytes(old_blind)
  # A coherently rehashed but wrong calibrated array must fail map replay.
  path=out/'C_cal_scores.npy';old=path.read_bytes();z=np.load(path,allow_pickle=False);z+=.01;np.save(path,z)
  bad=copy.deepcopy(f['blind']);bad['models']['C']['files']['C_cal']=record(path,out);D.write_json(blind_path,bad)
  try:errors['wrong_map_replay']=rejected(lambda:runner.parse_once(out,f['p'],f['groups'],bad,f['info'],f['root']));assert not access.exists()
  finally:path.write_bytes(old);blind_path.write_bytes(old_blind)
 finally:access.write_bytes(old_access)
 jwrite(OUT/'prelabel_rejections.json',errors)
 return {'cases':len(errors),'all_rejected_before_access_record':True}

def test_06_supervision_malformed_boundary():
 f=fixture_state;path=f['root']/'data/heldout/supervision/pairs.csv';old=path.read_bytes()
 with path.open(newline='',encoding='utf-8') as stream:rows=list(csv.DictReader(stream))
 cases={};access=f['out']/'heldout_access.json';old_access=access.read_bytes()
 for case in ('duplicate_pair','missing_group','unknown_account','invalid_label','wrong_positive_degree'):
  bad=copy.deepcopy(rows)
  if case=='duplicate_pair':bad[-1]=bad[-2]
  elif case=='missing_group':bad=[r for r in bad if r['group_uid']!=bad[0]['group_uid']]
  elif case=='unknown_account':bad[0]['seller_uid_right']='not_in_group'
  elif case=='invalid_label':bad[0]['label']='2'
  else:
   # Keep 20 positive edges, but give one query no positive neighbor.
   uid=f['groups'][0].uid;s=f['groups'][0].sellers[0]
   edge=next(r for r in bad if r['group_uid']==uid and s in (r['seller_uid_left'],r['seller_uid_right']) and r['label']=='1');edge['label']='0'
   replacement=next(r for r in bad if r['group_uid']==uid and s not in (r['seller_uid_left'],r['seller_uid_right']) and r['label']=='0');replacement['label']='1'
  try:
   with path.open('w',newline='',encoding='utf-8') as stream:
    w=csv.DictWriter(stream,fieldnames=['group_uid','seller_uid_left','seller_uid_right','label']);w.writeheader();w.writerows(bad)
   p=copy.deepcopy(f['p']);p['data']['inputs']['heldout/supervision/pairs.csv']=record(path,f['root'])
   access.unlink(missing_ok=True)
   cases[case]=rejected(lambda:runner.parse_once(f['out'],p,f['groups'],f['blind'],f['info'],f['root']))
   assert access.exists();cases[case]['retry']=rejected(lambda:runner.parse_once(f['out'],p,f['groups'],f['blind'],f['info'],f['root']),FileExistsError)
  finally:path.write_bytes(old);access.write_bytes(old_access)
 jwrite(OUT/'malformed_supervision.json',cases)
 return {'malformed_cases':len(cases),'exclusive_attempts_not_success_count':True,'no_retry':True}

def test_07_postcapture_recovery():
 f=fixture_state;cases={}
 patches=[('fixed_classification',runner.base,'fixed_classification'),('comparison',target,'compare'),('acceptance',target,'acceptance'),('automatic_report',runner.base,'automatic_report')]
 for key,obj,name in patches:
  out=OUT/('fault_'+key);shutil.copytree(OUT/'captured_before_statistics',out)
  with mock.patch.object(obj,name,side_effect=RuntimeError('handmade injected '+key)):
   cases[key]=rejected(lambda:runner.finalize(out,f['snapshot']),RuntimeError)
  assert len(list((out/'evaluation').glob('*_metrics.npy')))==4
  assert not (out/'evaluation/evaluation.json').exists()
  with ExitStack() as s:
   for o,n in [(runner,'parse_once'),(runner,'public_inputs'),(runner,'inference_one'),(runner.core,'restore_state'),(target.calibration,'fit')]:
    s.enter_context(mock.patch.object(o,n,side_effect=AssertionError('forbidden recovery action '+n)))
   result=runner.finalize(out,f['snapshot'])
  assert result==f['result'];cases[key]['recovered_identical']=True
 # Fail DURING collection: explicitly demonstrate the boundary, not claim preservation.
 out=OUT/'fault_inside_collection';out.mkdir();shutil.copy2(f['out']/'blind.json',out/'blind.json')
 original=target.metrics.group_metrics;calls=[0]
 def fail_second(*args,**kwargs):
  calls[0]+=1
  if calls[0]==2:raise RuntimeError('handmade incomplete collection')
  return original(*args,**kwargs)
 with mock.patch.object(target.metrics,'group_metrics',side_effect=fail_second):
  cases['inside_collection']=rejected(lambda:runner.collect(out,f['truth'],f['scores'],f['blind'],f['info'],f['snapshot']),RuntimeError)
 assert len(list((out/'evaluation').glob('*_metrics.npy')))==1
 assert not (out/'evaluation/collected.json').exists()
 jwrite(OUT/'recovery_cases.json',cases)
 return {'postcapture_failures':4,'identical_label_free_recovery':4,'incomplete_collection_saved_matrices':1,'incomplete_not_claimed_recoverable':True}

def test_08_order_and_threshold():
 raw=np.array([[0.,0.,1.,2.],[-1000.,-999.,1000.,1001.]],dtype=np.float32)
 mapping={'a':.75,'b':10.};cal=target.calibration.transform(raw,mapping)
 independent=np.array([[float(mapping['a'])*float(v)+mapping['b'] for v in row] for row in raw],np.float64)
 assert np.array_equal(cal,independent);target.calibration.preserve_order(raw,cal)
 assert np.all(np.exp(-np.logaddexp(0.,-cal[1,2:]))==1.)
 collapse=np.array([[0.,np.nextafter(0.,1.)]],np.float64)
 rejected(lambda:target.calibration.preserve_order(collapse,target.calibration.transform(collapse,{'a':.001,'b':1.})))
 f=fixture_state;root=OUT/'threshold_boundary';root.mkdir();shutil.copy2(f['out']/'blind.json',root/'blind.json')
 scores={k:v.copy() for k,v in f['scores'].items()};info=copy.deepcopy(f['info'])
 info['A']['threshold']=float(np.nextafter(np.float64(1.),np.inf));info['A']['map']=mapping
 scores['A_raw'][0,0]=1.;scores['A_cal']=target.calibration.transform(scores['A_raw'],mapping)
 cap=runner.collect(root,f['truth'],scores,f['blind'],info,f['snapshot'])
 threshold=info['A']['threshold'];rawpred=scores['A_raw'].astype(np.float64)>=threshold;transpred=scores['A_cal']>=cap['automatic']['A']['mapped_threshold']
 assert not rawpred[0,0] and transpred[0,0]
 expected=[ref_counts(y,z,threshold) for y,z in zip(f['truth'],scores['A_raw'])]
 assert cap['automatic']['A']['counts']==[[c[k] for k in ('tp','fp','fn','tn')] for c in expected]
 jwrite(OUT/'threshold_boundary.json',{'raw_value':1.,'raw_threshold':threshold,'mapped_value':float(scores['A_cal'][0,0]),'mapped_threshold':cap['automatic']['A']['mapped_threshold'],'raw_positive':False,'wrong_mapped_positive':True,'production_uses_original_space':True})
 return {'positive_affine':True,'float64_collapse_rejected':True,'saturation_keeps_logit_order':True,'original_threshold_semantics_verified':True}

class ToyEncoder(torch.nn.Module):
 def __init__(self):
  super().__init__();self.weight=torch.nn.Parameter(torch.tensor([[.3,.2],[-.4,.6],[.7,-.1]],dtype=torch.float32));self.records=[];self.flags=[]
 def tokenizer(self,texts,**kwargs):
  self.records.extend(texts)
  assert kwargs=={'padding':True,'truncation':False,'return_tensors':'pt'}
  def val(t):return [(sum(t.encode('utf-8'))%101)/101.,len(t)/20.,1.]
  return {'input_ids':torch.tensor([val(t) for t in texts],dtype=torch.float32)}
 def forward(self,batch):
  self.flags.append({'training':self.training,'grad_enabled':torch.is_grad_enabled(),'inference_mode':torch.is_inference_mode_enabled()})
  return {'sentence_embedding':batch['input_ids']@self.weight}

def toy_model():
 enc=ToyEncoder();m=runner.core.build_model(enc,8,5)
 with torch.no_grad():
  m.head[0].weight.copy_(torch.arange(80,dtype=torch.float32).reshape(5,16)/100.-.4)
  m.head[0].bias.copy_(torch.linspace(-.1,.1,5));m.head[2].weight.copy_(torch.tensor([[.2,-.3,.4,-.1,.5]]));m.head[2].bias.fill_(.12)
 return m

def reference_forward(model,group):
 def normalize(x):return x/np.maximum(np.linalg.norm(x,axis=-1,keepdims=True),1e-12)
 accounts=[];w=model.encoder.weight.detach().numpy().astype(float)
 for rows in group.items:
  parts=[]
  for channel in (1,2):
   x=np.array([[(sum(it[channel].encode('utf-8'))%101)/101.,len(it[channel])/20.,1.] for it in rows]);v=normalize(x@w)
   avg=v.mean(0);std=np.sqrt(((v-avg)**2).mean(0)+1e-8);parts.extend([avg,std])
  accounts.append(np.concatenate(parts))
 accounts=normalize(np.stack(accounts));out=[]
 w1=model.head[0].weight.detach().numpy().astype(float);b1=model.head[0].bias.detach().numpy().astype(float);w2=model.head[2].weight.detach().numpy().astype(float);b2=model.head[2].bias.detach().numpy().astype(float)
 for i,j in EDGES:
  features=np.r_[np.abs(accounts[i]-accounts[j]),accounts[i]*accounts[j]]
  out.append(float((w2@np.maximum(w1@features+b1,0.)+b2).item()))
 return out

def test_09_real_tiny_restore_full_inherited_forward():
 root=OUT/'tiny_checkpoint';root.mkdir();original=toy_model();metadata={'run_id':'s0_d','epoch':6,'handmade':True}
 payload=runner.core.save_state(root/'tiny.pt',original,None,metadata);parameter_digest=runner.core.state_digest(original.state_dict())
 p={'models':{'A':{'payload':{**payload,'path':'tiny.pt','payload_state_sha256':payload['state_sha256'],'model_parameters_sha256':parameter_digest}}}}
 config={'interventions':{'split_rank':{'representation':'separate_moments'}},'input':{'microbatch':4,'token_budget':256,'encoder_bf16':False},'representation':{'variance_epsilon':1e-8}}
 info={'A':{'config':config,'point':{'metadata':metadata}}};groups=fixture_state['groups'][:3];models=[];budget_calls=[0]
 def fresh(*args):
  m=toy_model()
  with torch.no_grad():
   for p1 in m.parameters():p1.add_(1.)
  models.append(m);return m
 def check():budget_calls[0]+=1
 with mock.patch.object(D,'ROOT',root),mock.patch.object(runner.base,'load_model',side_effect=fresh):
  actual,rec=runner.inference_one(p,'A',info,groups,check)
 expected=np.asarray([reference_forward(original,g) for g in groups])
 check_close('tiny_full_forward',actual,expected,2e-6)
 assert runner.core.state_digest(models[0].state_dict())==parameter_digest
 flags=models[0].encoder.flags;assert flags and all(not f['training'] and not f['grad_enabled'] and f['inference_mode'] for f in flags)
 texts=[t for g in groups for channel in (1,2) for account in g.items for it in account for t in [it[channel]]]
 assert models[0].encoder.records==texts
 assert budget_calls[0]==sum(math.ceil(sum(map(len,g.items))*2/4) for g in groups)
 errors={}
 for case in ('payload_digest','parameter_digest','metadata','file_sha'):
  badp=copy.deepcopy(p);badi=copy.deepcopy(info)
  if case=='payload_digest':badp['models']['A']['payload']['payload_state_sha256']='0'*64
  elif case=='parameter_digest':badp['models']['A']['payload']['model_parameters_sha256']='0'*64
  elif case=='metadata':badi['A']['point']['metadata']['epoch']=3
  else:badp['models']['A']['payload']['sha256']='0'*64
  with mock.patch.object(D,'ROOT',root),mock.patch.object(runner.base,'load_model',side_effect=fresh),mock.patch.object(runner.base,'score',side_effect=AssertionError('must reject before forward')):
   errors[case]=rejected(lambda:runner.inference_one(badp,'A',badi,groups,lambda:None))
 np.savez(OUT/'tiny_forward_reference.npz',actual=actual,expected=expected)
 jwrite(OUT/'tiny_restore.json',{'payload':p,'record':rec,'expected_parameters_digest':parameter_digest,'budget_calls':budget_calls[0],'flags':flags,'rejections':errors,'scope':'Real CPU torch tensors; patched constructor only, inherited base.score/tokenization/pooling/pair head actually run; not BGE'})
 return {'real_torch':True,'groups':3,'pairs':1134,'constructor_replaced_only':True,'native_bge':False,'full_inherited_forward':True,'mismatch_rejections':4,'model_updates':0}

def test_10_automatic_bootstrap_and_budget():
 f=fixture_state;result=f['result'];draws=target.bootstrap_draws();domains=f['blind']['domains']
 freq=np.zeros((5000,120),np.int64)
 for b in range(5000):
  for d in range(3):
   for i in draws[b,d]:freq[b,d*40+int(i)]+=1
 for name in ('A','C'):
  counts=np.asarray(result['automatic'][name]['counts']);report=result['automatic'][name]['report']
  for d,scope in enumerate(['A','B','C','pooled']):
   ix=np.arange(d*40,(d+1)*40) if d<3 else np.arange(120)
   boot=freq[:,ix]@counts[ix];sums=counts[ix].sum(0);cs=dict(zip(('tp','fp','fn','tn'),map(int,sums)));rr=ref_rates(cs)
   observed=report['by_domain'][scope] if d<3 else report['pooled']
   for key in ('fpr','recall','precision'):
    numerator,other={'fpr':(1,3),'recall':(0,2),'precision':(0,1)}[key]
    den=boot[:,numerator]+boot[:,other];v=np.divide(boot[:,numerator],den,out=np.zeros(5000),where=den!=0)
    check_close('automatic_intervals',observed['conditional_95pct_intervals'][key],quantile_linear(v))
    check_close('automatic_point',observed[key],rr[key])
   if d<3:assert observed['passes_point_gates']==(cs['fp']*1000<=cs['fp']+cs['tn'] and cs['tp']*2>=cs['tp']+cs['fn'])
 tmp=OUT/'budget_toy';tmp.mkdir();(tmp/'toy.bin').write_bytes(b'12345')
 rejected(lambda:runner.Budget(tmp,100,4).check(),RuntimeError)
 return {'automatic_scopes':8,'interval_metrics':3,'original_diagnostic_not_acceptance':True,'sampled_file_budget_check':True}

TESTS=[test_01_criteria,test_02_metrics_scalar,test_03_bootstrap,test_04_public_and_pipeline,test_05_prelabel_rejections,test_06_supervision_malformed_boundary,test_07_postcapture_recovery,test_08_order_and_threshold,test_09_real_tiny_restore_full_inherited_forward,test_10_automatic_bootstrap_and_budget]
started=time.monotonic()
for fn in TESTS:
 t=time.monotonic();print('RUN',fn.__name__,flush=True)
 try:detail=fn();RESULTS[fn.__name__]={'passed':True,'seconds':time.monotonic()-t,'details':detail};print('PASS',fn.__name__,flush=True)
 except Exception as e:
  RESULTS[fn.__name__]={'passed':False,'seconds':time.monotonic()-t,'error_type':type(e).__name__,'message':str(e),'traceback':traceback.format_exc()};traceback.print_exc()
report={'tests':RESULTS,'passed':sum(r['passed'] for r in RESULTS.values()),'failed':sum(not r['passed'] for r in RESULTS.values()),'numeric_checks':NUMERICS,'elapsed_seconds':time.monotonic()-started,'environment':{'python':sys.version,'numpy':np.__version__,'torch':torch.__version__,'platform':platform.platform(),'cpu_affinity':sorted(os.sched_getaffinity(0)),'torch_threads':torch.get_num_threads(),'torch_interop_threads':torch.get_num_interop_threads(),'process_threads':next(l.strip() for l in Path('/proc/self/status').read_text().splitlines() if l.startswith('Threads:'))},'formal_labels_read':0,'formal_text_read':0,'formal_scores_read':0,'native_model_loads':0,'formal_execute_called':False,'optimizer_steps':0,'calibration_fits':0,'reference_revisions':[{'version':1,'failed_test':'test_08_order_and_threshold','diagnosis':'NumPy float32 array / NumPy scalar comparison with Python float implicitly rounded the reference threshold; production explicitly casts logits to float64.','fix':'Use Python floats in scalar reference and explicit float64 raw reference comparison; no project changes or tolerance changes.','diagnostic':'outputs/threshold_reference_diagnosis.json'}]}
jwrite(OUT/'independent_results.json',report)
print(json.dumps(report,ensure_ascii=False,indent=2),flush=True)
sys.exit(0 if report['failed']==0 else 1)
