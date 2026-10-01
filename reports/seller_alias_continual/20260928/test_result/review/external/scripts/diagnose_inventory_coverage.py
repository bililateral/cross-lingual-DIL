from pathlib import Path
import json,hashlib
r=Path('/mnt/data/test_result_audit_20260928/project');b=r/'reports/seller_alias_continual/20260928'
current=json.loads((r/'source_inventory.json').read_text());named={x['path'] for x in current['files']};results={}
for rel in ('test_execution/monitoring/return_sync.json','test_result/return_sync.json','test_result/analysis_return_sync.json'):
 d=json.loads((b/rel).read_text());present=[];absent=[];bad=[]
 for rec in d['files']:
  p=r/rec['path']
  if not p.exists():absent.append({**rec,'declared_in_current_inventory':rec['path'] in named})
  elif p.stat().st_size!=rec['bytes'] or hashlib.sha256(p.read_bytes()).hexdigest()!=rec['sha256']:bad.append(rec)
  else:present.append(rec)
 result={'recorded_count':d['count'],'present_verified_count':len(present),'present_verified_bytes':sum(x['bytes'] for x in present),'missing':absent,'mismatch':bad}
 results[rel]=result
 print(rel,json.dumps(result,ensure_ascii=False,indent=2))
 assert not bad
out=Path('/mnt/data/test_result_audit_20260928/evidence/outputs/inventory_coverage_diagnosis.json');out.write_text(json.dumps(results,ensure_ascii=False,indent=2)+'\n')
