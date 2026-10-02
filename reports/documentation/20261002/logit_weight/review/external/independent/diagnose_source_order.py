import json
from pathlib import Path
import step28_er_weight as m
root=Path(__file__).resolve().parents[1];p=m.contract('logit');actual=m.sources(p);frozen=m.data.read_json(root/'submission/reports/documentation/20261002/logit_weight/freeze.json')['source_files']
a={r['path']:r for r in actual};b={r['path']:r for r in frozen};different=[{'path':k,'actual':a.get(k),'frozen':b.get(k)} for k in sorted(set(a)|set(b)) if a.get(k)!=b.get(k)]
print(json.dumps({'counts':[len(actual),len(frozen)],'duplicate_actual':len(actual)-len(a),'duplicate_frozen':len(frozen)-len(b),'path_keyed_equal':a==b,'list_equal':actual==frozen,'different_records':different,'actual_order':[r['path'] for r in actual],'freeze_order':[r['path'] for r in frozen]},indent=2))
