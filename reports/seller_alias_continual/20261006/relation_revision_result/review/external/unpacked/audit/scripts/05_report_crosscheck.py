"""Cross-check displayed report tables and derive bounded interpretation highlights.
No new statistical comparison, training, threshold selection, or model inference.
"""
from pathlib import Path
import json,re
import numpy as np
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'audit/outputs';IN=ROOT/'input'
s=json.loads((OUT/'saved_statistics.json').read_text());d=json.loads((OUT/'diagnostics_audit.json').read_text())
def stage(arm,o,st,dom):return next(x for x in s['stage_rows'] if (x['arm'],x['order'],x['stage'],x['actual_domain'])==(arm,o,st,dom))
text=(IN/'docs/SELLER_ALIAS_RELATION_REVISION_RESULT.zh.md').read_text();section=None;checked=[]
def checkrow(no,label,a,b,tols):
 errs=[abs(x-y) for x,y in zip(a,b,strict=True)]; assert all(e<=tol for e,tol in zip(errs,tols,strict=True)),(no,label,a,b)
 checked.append({'line':no,'label':label,'observed':a,'expected':b,'max_abs_error':max(errs)})
for no,line in enumerate(text.splitlines(),1):
 m=re.match(r'## (\d+)\.',line)
 if m:section=int(m[1])
 if not line.startswith('|'):continue
 row=[cell.strip() for cell in line.strip('|').split('|')]
 if section==3 and row[0] in ['O MAP','N MAP','Z MAP','O AP','N AP','Z AP','final_all MAP']:
  ep,met=row[0].split();met='map' if met=='MAP' else 'average_precision';x=s['delta'][ep][met]
  a=[float(v) for v in row[1:4]]+[float(v) for v in row[4].strip('[]').split(',')]
  b=[s['endpoints']['relation']['primary'][ep][met]['mean'],s['endpoints']['logit0.1']['primary'][ep][met]['mean'],x['mean'],*x['conditional_95pct_interval']]
  checkrow(no,row[0],a,b,[5.01e-7]*5)
 elif section==4 and row[0] in ['ABC/A','BCA/B','CAB/C']:
  o,dom=row[0].split('/')
  if len(row)==4:
   a=list(map(float,row[1:]));b=[stage('relation',o,1,dom)['map'],d['old_first_stage_map'][o],stage('logit0.1',o,1,dom)['map']]
  else:
   a=[float(v) for cell in row[1:] for v in cell.split('→')];b=[stage(arm,o,st,dom)['map'] for arm in ['relation','logit0.1'] for st in (1,2,3)]
  checkrow(no,row[0],a,b,[5.01e-7]*len(a))
 elif section==5 and row[0] in ['ABC','BCA','CAB']:
  k=d['training'][row[0]+'_relation_revision_stage2'];a=list(map(float,row[1:]));b=[k['history_compressed']['first'],k['old_Q_first'],k['history_compressed']['mean'],k['weighted_R_mean'],k['gradient_norm']['median']]
  checkrow(no,row[0]+' Q diagnostic',a,b,[5.01e-7]*4+[5.01e-4])
 elif section==6 and row[0].endswith((' 修订',' LOGIT0.1')):
  ep,arm=row[0].split();arm='relation' if arm=='修订' else 'logit0.1';a=list(map(float,row[1:]));p=s['pooled_stage_cal'][arm][ep];b=[p[k] for k in ['tp','fp','fn','tn','precision','recall','f1']]
  checkrow(no,row[0],a,b,[0.]*4+[5.01e-5]*3)
assert len(checked)==22,len(checked)
extra={}
for ep in ('O','N','Z'):
 favor=[]
 for met,x in s['delta'][ep].items():
  sign=-1 if met in ('brier','log_loss') else 1
  if sign*x['mean']>0:favor.append({'metric':met,**x})
 extra[ep]=favor
# Point-estimate identity, not a new acceptance endpoint or new CI.
levels={}
for arm in ['relation','logit0.1']:
 old_acquired=np.mean([(stage(arm,o,1,o[0])['map']+stage(arm,o,2,o[1])['map'])/2 for o in ['ABC','BCA','CAB']])
 O=s['endpoints'][arm]['primary']['O']['map']['mean'];F=s['endpoints'][arm]['primary']['F']['map']['mean']
 assert abs((old_acquired-F)-O)<1e-15
 levels[arm]={'old_domains_mean_when_acquired':float(old_acquired),'O_MAP':O,'F_MAP':F}
levels['delta_acquired']=levels['relation']['old_domains_mean_when_acquired']-levels['logit0.1']['old_domains_mean_when_acquired']
levels['delta_forgetting']=s['delta']['F']['map']['mean']
levels['delta_final_old']=s['delta']['O']['map']['mean']
out={'scope':'Published table rounding checks and arithmetic interpretation; no additional old-new CI or acceptance rule.',
 'main_report_rows_checked':checked,'local_favorable_metrics':extra,'old_domain_point_identity':levels,
 'first_domain_trajectories':{arm:{o:[stage(arm,o,st,o[0])['map'] for st in (1,2,3)] for o in ['ABC','BCA','CAB']} for arm in ['relation','logit0.1']},
 'current_stage2_roles': [x for x in d['score_distributions_and_counts'] if x['point'].endswith('stage2') and x['domain']==x['point'][1]],
 'forgetting_gain_MAP':{ep:{'candidate':s['endpoints']['relation']['primary'][ep]['map'],'reference':s['endpoints']['logit0.1']['primary'][ep]['map'],'delta':s['delta'][ep]['map']} for ep in ['F_first','F','G']}}
(OUT/'report_crosscheck.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({k:out[k] for k in ['scope','local_favorable_metrics','old_domain_point_identity','forgetting_gain_MAP']},ensure_ascii=False,indent=2))
print('Main report rows matched:',len(checked))
