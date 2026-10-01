"""Read selected submitted metadata and raw log edges, never load model/data files."""
from pathlib import Path
import json
r=Path('/mnt/data/ranking_result_reaudit_20260928/submitted');j=r/'reports/seller_alias_continual/20260927/ranking_execution/20260927_114646/job';s=r/'reports/seller_alias_continual/20260928/ranking_result'
for p in [j.parent.parent/'authorization.json',j/'completion.json',s/'weight_retention.json',s/'retention_verification.json',s/'analysis_execution.json',s/'sync.json']:
 d=json.loads(p.read_text());print('\n====',p.relative_to(r),'====')
 for k,v in d.items():
  if k in ('files','source_files','frozen_sources','weights') and isinstance(v,list):print(k,'COUNT',len(v),'FIRST',v[:1])
  else:print(k,':',json.dumps(v,ensure_ascii=False))
print('\n==== JOB LOG EDGES ====')
ls=(j/'train.log').read_text().splitlines();print('\n'.join(ls[:4]));print('\n'.join(ls[-8:]));print('lines',len(ls))
for f in ['started.txt','finished.txt','exit_status.txt','resource_usage.log']:
 p=j/f;print('\n====',f,'====');print(p.read_text() if p.exists() else 'NOT PRESENT')
print('==== SINGLE RUN RAW SAVE POINT & TRAIN ===')
d=json.loads((j/'run/s0_hard/manifest.json').read_text())
for k in ['preflight','resources','training','points']:
 if k=='points':
  for ep,v in d[k].items():
   vv={kk:val for kk,val in v.items() if kk not in ('train_metrics','scores')}
   vv['metadata']={kk:val for kk,val in vv['metadata'].items() if kk!='reference_config'}
   print(ep,json.dumps(vv,ensure_ascii=False,indent=2))
 else:print(k,json.dumps(d[k],ensure_ascii=False,indent=2))
