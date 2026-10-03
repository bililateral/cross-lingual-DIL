from pathlib import Path
import json
import platform
import sys
import numpy as np

P = Path('/workspace/scratch/ca13b63db3f0/review_work/input/project')
roots = {
    'low': P / 'reports/seller_alias_continual/20261002/er_low_execution/20261002_124018/job',
    'logit': P / 'reports/seller_alias_continual/20261002/logit_weight_execution/20261002_153200/workspace/reports/job',
    'base': P / 'reports/seller_alias_continual/20260930/bge_continual_execution/20260930_143700/job',
    'weight': P / 'reports/seller_alias_continual/20261001/er_weight_execution/20261001_144452/job',
}

def read(p):
    return json.loads(p.read_text())

print(json.dumps({'python':sys.version,'numpy':np.__version__,'platform':platform.platform()}))
for k,r in roots.items():
    c = read(r/'evaluation/collected.json')
    m = read(r/'run/manifest.json')
    sample = next(iter(c['points']))
    ans = {'job':k,'collection_keys':list(c),'manifest_keys':list(m),'points':list(c['points']), 'sample_point':sample}
    if k in ('low','logit'):
        point = read(r/'run/points'/f'{sample}.json')
        mapping = read(r/'run'/point['mapping']['path'])
        ans.update({'point_keys':list(point),'map':mapping, 'manifest_starts':m.get('starts')})
        ans['array_shape']={role:{'shape':list(np.load(r/'run'/rec['path'],allow_pickle=False).shape),'dtype':str(np.load(r/'run'/rec['path'],allow_pickle=False).dtype)} for role,rec in point['scores'].items()}
        ref=read(r/'evaluation/reference/collected.json')
        ans['reference_keys']=list(ref)
        ans['reference_points']=list(ref['points'])
    print(json.dumps(ans,ensure_ascii=False))
