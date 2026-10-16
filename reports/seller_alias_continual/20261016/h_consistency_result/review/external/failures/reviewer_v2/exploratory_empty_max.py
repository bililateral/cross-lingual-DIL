# Preserved from the inline command already executed during v2 diagnosis.
# This file is a transcript copy, not a second execution.
from pathlib import Path
import json, numpy as np
R=Path('/mnt/data/hcons_review/input');E=R/'evidence';j=E/'reports/job';ev=json.loads((j/'evaluation/evaluation.json').read_text());base=json.loads((E/'reports/documentation/20261016/h_consistency/baseline_sources.json').read_text());roots={'H_cons':j,**{v['arm']:E/v['root'] for v in base['baselines'].values()}}; out=[]
for arm,root in roots.items():
 col=json.loads((root/'evaluation/collected.json').read_text());rows=[np.flatnonzero(np.array(col['domains'])==d) for d in 'AB']
 for stage in 'AB':
  p=stage+'_'+arm
  for role in ['raw','stage-cal','first-cal']:
   cc=json.loads((root/'evaluation'/col['points'][p][role]['counts']['path']).read_text());ct=np.array([[r[k] for k in ('tp','fp','fn','tn')] for r in cc])
   for d,sel in [('all',np.arange(40)),('A',rows[0]),('B',rows[1])]:
    t,f,nf,nt=map(int,ct[sel].sum(0)); ours=2*t/(2*t+f+nf); old=ev['absolute'][p][role]['fixed_half_classification'][d]['f1']
    if ours!=old:out.append(dict(point=p,role=role,domain=d,direct_count_f1=ours,saved_f1=old,difference=ours-old))
print('F1 roundoff',len(out),out[:5],'maxabs',max(abs(r['difference']) for r in out))
Path('/mnt/data/hcons_review/return/failures/reviewer_v2/count_diagnosis.json').write_text(json.dumps(out,indent=2)+'\n')
