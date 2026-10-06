"""Audit saved scores, transforms, log diagnostics, published tables, SVG points.
No labels are available or inferred. Synthetic labels in the last section are
newly hand-constructed solely to illustrate a mathematical limitation.
"""
from pathlib import Path
import json,hashlib,re,math,xml.etree.ElementTree as ET
import numpy as np
ROOT=Path(__file__).resolve().parents[2];IN=ROOT/'input';OUT=ROOT/'audit/outputs'
RR=IN/'reports/seller_alias_continual/20261006/relation_revision_result';RUN=RR/'job/run';EV=RR/'job/evaluation'
OLD=IN/'reports/seller_alias_continual/20261006/relation_result'
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def verify(root,r):
 p=root/r['path'];assert p.stat().st_size==r['bytes'] and sha(p)==r['sha256'];return p
s=read(OUT/'saved_statistics.json');state=read(OUT/'execution_evidence.json');original=read(RR/'analysis/diagnostics.json');mf=read(RUN/'manifest.json');co=read(EV/'collected.json');old=read(OLD/'job/evaluation/evaluation.json')
dom=np.array(co['domains']);roles=('raw','stage-cal','first-cal');orders=('ABC','BCA','CAB')
checks=[];transforms=[];distribution=[];logs={};identities=0
# Initial blind scores and prediction-count constraint without recovering labels.
initial=np.load(verify(RUN,mf['initial']),allow_pickle=False); assert initial.shape==(60,378) and initial.dtype==np.float32 and np.isfinite(initial).all();identities+=1
ct=read(verify(EV,co['points']['initial']['raw']['counts']));assert np.array_equal((initial>=0).sum(1),[x['tp']+x['fp'] for x in ct])
for o in orders:
 for stage in (1,2,3):
  name=f'{o}_relation_revision_stage{stage}';pt=read(RUN/'points'/f'{name}.json')
  arrays={role:np.load(verify(RUN,rec),allow_pickle=False) for role,rec in pt['scores'].items()};identities+=len(arrays)
  for role,a in arrays.items():
   assert np.isfinite(a).all()
   assert a.shape==((12,378) if role=='calibration' else (60,378))
   assert a.dtype==(np.float32 if role in ('raw','calibration') else np.float64)
  for role,key in [('stage-cal','stage'),('first-cal','first')]:
   m=pt['maps'][key];a,b=m['a'],m['b'];assert a>0
   expected=arrays['raw'].astype('f8')*a+b
   assert np.array_equal(expected,arrays[role])
   sort=np.argsort(arrays['raw'],axis=1,kind='stable');sort2=np.argsort(arrays[role],axis=1,kind='stable');assert np.array_equal(sort,sort2)
   dif=np.diff(np.take_along_axis(arrays['raw'],sort,1),axis=1);dif2=np.diff(np.take_along_axis(arrays[role],sort2,1),axis=1)
   assert np.array_equal(dif==0,dif2==0)
   transforms.append({'point':name,'role':role,'a':a,'b':b,'raw_threshold_for_p_half':-b/a,'max_abs_transform_error':float(np.abs(expected-arrays[role]).max()),'sort_and_ties_preserved':True})
  for role in roles:
   ct=read(verify(EV,co['points'][name][role]['counts']));pred=(arrays[role]>=0).sum(1);assert np.array_equal(pred,[x['tp']+x['fp'] for x in ct])
   for d in 'ABC':
    r=np.flatnonzero(dom==d);a=arrays[role][r]
    distribution.append({'point':name,'role':role,'domain':d,'score_min':float(a.min()),'score_max':float(a.max()),'score_mean':float(a.mean()),'score_std':float(a.std()),'predicted_positive':int(pred[r].sum()),
       **{k:sum(ct[i][k] for i in r) for k in ('tp','fp','fn','tn')}})
  tr=read(RUN/'updates'/f'{name}.json')['updates'];otr=read(OLD/'job/run/updates'/f'{o}_relation_stage{stage}.json')['updates'];summary={}
  for key in ['current_bce','current_rank','current_hard','current_total','history_compressed','history_rank','history_total','gradient_norm']:
   a=np.array([x[key] for x in tr]); st={'mean':float(a.mean()),'median':float(np.median(a)),'first':float(a[0]),'last':float(a[-1]),'first24_mean':float(a[:24].mean()),'last24_mean':float(a[-24:].mean())}
   assert st==original['training'][name][key];summary[key]=st
  assert sum(x['gradient_norm']>1 for x in tr)==original['training'][name]['clip_active_count']
  assert otr[0]['history']==original['training'][name]['old_history_first']
  oldmedian=float(np.median([x['gradient_norm'] for x in otr]));assert oldmedian==original['training'][name]['old_gradient_median']
  summary['weighted_R_mean']=float(np.mean([float(np.float32(.1)*np.float32(x['history_rank'])) for x in tr]))
  summary['old_Q_first']=otr[0]['history'];summary['old_gradient_median']=oldmedian;summary['clip_count']=sum(x['gradient_norm']>1 for x in tr)
  logs[name]=summary
  if stage==2:
   pooled={k:sum(x[k] for x in distribution if x['point']==name and x['role']=='stage-cal') for k in ('tp','fp','fn','tn')}
   assert pooled==original['stage2_predictions'][name]
# Current diagnostic hashes accurately refer to three distinct originals.
hashes=list(original['input_hashes'].values())
assert sorted(hashes)==sorted([sha(EV/'evaluation.json'),sha(OLD/'job/evaluation/evaluation.json'),sha(RR/'verification/result.json')])
for ep,v in original['old_configuration'].items():
 for name,value in v.items():assert old['endpoints']['relation']['primary'][ep][name]['mean']==value
for o in orders:assert old['absolute_stage_results'][f'{o}_relation_stage1']['raw']['macro_by_domain'][o[0]]['map']==original['old_first_stage_map'][o]
# Compare the 9-decimal report table to independently recalculated statistics.
ntable=0;table_error=0.
for line in (RR/'analysis/metrics.zh.md').read_text().splitlines():
 if not line.startswith('|'):continue
 row=line.strip('|').split('|')
 if len(row)!=9 or row[0] not in s['delta']:continue
 ep,name=row[:2]; d=s['delta'][ep][name]
 vals=[float(row[i]) for i in (2,3,4)]+[float(x) for x in row[5].strip('[]').split(',')]+[float(row[i]) for i in (6,7,8)]
 expected=[s['endpoints']['relation']['primary'][ep][name]['mean'],s['endpoints']['logit0.1']['primary'][ep][name]['mean'],d['mean'],*d['conditional_95pct_interval'],*[d['per_order'][o] for o in orders]]
 err=float(np.max(np.abs(np.array(vals)-expected)));assert err<=5.01e-10,(ep,name,err);table_error=max(table_error,err);ntable+=1
assert ntable==154
# Read 18 actual SVG trajectory polylines, invert the plotted fixed y-axis transform.
svg=ET.parse(RR/'analysis/stage_map.svg').getroot();ns={'s':'http://www.w3.org/2000/svg'}
clips={n.get('id'):n.find('s:rect',ns).attrib for n in svg.findall('.//s:clipPath',ns)}
colors={'#277a8a':'A','#d38530':'B','#9764a0':'C'};svg_rows=[]
for ax in svg.findall('.//s:g',ns):
 id=ax.get('id','')
 if id not in ('axes_1','axes_2','axes_3'):continue
 o=orders[int(id[-1])-1]
 for group in list(ax):
  if not group.get('id','').startswith('line2d_'):continue
  for p in group.findall('s:path',ns):
   sty=p.get('style','');color=next((c for c in colors if 'stroke: '+c in sty),None)
   if color is None:continue
   nums=np.array([float(n) for n in re.findall(r'-?\d+(?:\.\d+)?(?:e[+-]?\d+)?',p.attrib['d'])]).reshape(-1,2);assert nums.shape==(3,2)
   c=clips[p.get('clip-path')[5:-1]];y0=float(c['y']);height=float(c['height'])
   data_y=.52-(nums[:,1]-y0)/height*(.52-.28)
   arm='logit0.1' if 'stroke-dasharray' in sty else 'relation';d=colors[color]
   exp=np.array([next(x['map'] for x in s['stage_rows'] if x['arm']==arm and x['order']==o and x['stage']==stage and x['actual_domain']==d) for stage in (1,2,3)])
   err=float(np.max(np.abs(data_y-exp)));assert err<=1e-8,(o,arm,d,err)
   svg_rows.append({'order':o,'arm':arm,'domain':d,'decoded_map':data_y.tolist(),'expected_map':exp.tolist(),'max_abs_error':err})
assert len(svg_rows)==18
# Synthetic complete-group score example: high Q need not mean ranking is wrong.
components=[list(range(3*k,3*k+3)) for k in range(4)]+[[12+2*k,13+2*k] for k in range(8)]
lab=np.zeros((28,28),dtype=bool)
for group in components:
 for i in group:
  for j in group:
   if i!=j:lab[i,j]=True
left,right=np.triu_indices(28,1);truth=lab[left,right]; assert truth.sum()==20
pairs=[(np.flatnonzero(((left==q)|(right==q)) & truth),np.flatnonzero(((left==q)|(right==q)) & ~truth)) for q in range(28)]
def objective(pos,neg):
 scores=np.where(truth,pos,neg); target=truth*2-1
 bce=np.mean(np.logaddexp(0,scores)-truth*scores)
 query=[];hard=[];rank=[];rankq=[]
 for p,n in pairs:
  ids=np.r_[p,n];vals=scores[ids];mx=max(vals)
  query.append(mx+np.log(np.exp(vals-mx).sum())-scores[p].mean())
  hard.append(np.logaddexp(0,scores[n[:5]][None,:]-scores[p,None]).mean())
  rank.append(np.logaddexp(0,scores[n][None,:]-scores[p,None]).mean())
  rankq.append(((scores[p,None]-scores[n][None,:]-2)**2).mean())
 return {'positive_score':pos,'negative_score':neg,'B':float(bce+np.mean(query)+.5*np.mean(hard)),
 'bce':float(bce),'query':float(np.mean(query)),'hard':float(np.mean(hard)),
 'Q':float(np.mean((scores-target)**2)+np.mean(rankq)), 'R':float(np.mean(rank)),
 'all_positive_above_all_negative':bool(pos>neg),'synthetic_MAP_AP':1.0 if pos>neg else None}
toy=[objective(1,-1),objective(2,-4)]
assert toy[1]['B']<toy[0]['B'] and toy[1]['Q']>toy[0]['Q'] and toy[1]['R']<toy[0]['R']
# Same X=Y, rank-deficient finite 32-dimensional canonical feature matrix; identity reduction.
z=np.zeros((32,378));z[0]=np.where(truth,.4,-.8);z[-1]=1.;w=np.zeros(32);w[0]=5.
k=z@z.T/378+.001*np.eye(32);v=np.linalg.solve(k,z@(z.T@w)/378+.001*w)
assert np.max(np.abs(v-w))<1e-10
r={'scope':'Independent blind-score transforms/prediction counts, saved diagnostics, table and SVG; synthetic score example not a trained model or formal data.',
 'score_files_checked':identities,'affine_transforms':transforms,'score_distributions_and_counts':distribution,
 'training':logs,'old_configuration':original['old_configuration'],'old_first_stage_map':original['old_first_stage_map'],
 'published_full_table_rows':ntable,'published_table_max_rounding_error':table_error,
 'svg_polylines':svg_rows,'svg_point_count':54,'svg_max_coordinate_error':max(x['max_abs_error'] for x in svg_rows),
 'synthetic_example':toy,'synthetic_identity_v_max_error':float(np.max(np.abs(v-w)))}
(OUT/'diagnostics_audit.json').write_text(json.dumps(r,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({k:r[k] for k in ('scope','score_files_checked','published_full_table_rows','published_table_max_rounding_error','svg_point_count','svg_max_coordinate_error','synthetic_example','synthetic_identity_v_max_error')},ensure_ascii=False,indent=2))
print('stage2 stage-cal current-domain predictions:',json.dumps([x for x in distribution if 'stage2' in x['point'] and x['role']=='stage-cal' and x['domain']==x['point'][1]],ensure_ascii=False,indent=2))
