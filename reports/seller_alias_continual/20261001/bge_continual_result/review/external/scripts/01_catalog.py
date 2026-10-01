from pathlib import Path
import json,numpy as np
r=Path('/mnt/data/bge_input'); job=r/'reports/seller_alias_continual/20260930/bge_continual_execution/20260930_143700/job'
files=['execution.json','before_valid.json','completion.json','run/manifest.json','run/partition.json','run/maps/ABC_er_stage3.json','run/memory/ABC_er_stage3_budget.json','run/updates/ABC_er_stage3.json','run/points/ABC_shared.json','evaluation/collected.json','evaluation/evaluation.json']
for n in files:
 d=json.loads((job/n).read_text());print('\n===',n,'===');print('keys:',list(d))
 if n.endswith('evaluation.json'): print('comparisons',json.dumps({k:v['interpretation'] for k,v in d['comparisons'].items()},indent=2)); print('endpoint schema',d['endpoints']['seq']['primary']['O']['map']);print('absolute schema',str(d['absolute_stage_results']['ABC_er_stage3'])[:1000])
 elif n.endswith('collected.json'): print('columns',d['metric_columns']); print('ids/domains',list(zip(d['group_ids'],d['domains']))[:3]);print('initial',d['initial']);print('sample point',d['points']['ABC_shared'])
 elif n.endswith('partition.json'):print('counts',{k:len(v) for k,v in d.items()});print('samples',{k:v[:1] for k,v in d.items()})
 elif n.endswith('manifest.json'):print({k:d[k] for k in ['status','physical_updates','gradient_group_presentations','budget','initial']});print('memory',list(d['memories'].items())[:1]);print('sources',d['source_files'])
 elif 'updates/' in n: print({k:v for k,v in d.items() if k not in ['current_ids','history_ids']})
 elif 'points/' in n:print({k:v for k,v in d.items() if k!='learner_auxiliary'})
 else:print(json.dumps(d,indent=2))
for sub in ['run/scores','run/updates','evaluation']:
 for p in sorted((job/sub).glob('*.npy'))[:2]:
  a=np.load(p,allow_pickle=False);print(p.name,a.shape,str(a.dtype),float(a.min()),float(a.max()))
print('counts example',json.loads((job/'evaluation/ABC_er_stage3_stage-cal_counts.json').read_text())[:2])
print('stage csv header and first row','\n'.join((job/'evaluation/stage_metrics.csv').read_text().splitlines()[:2]))
