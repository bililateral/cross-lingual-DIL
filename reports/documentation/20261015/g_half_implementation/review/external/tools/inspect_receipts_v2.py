"""Read receipt bodies; v2 fixes a metadata-summary type assumption, not project code."""
from pathlib import Path
import json
R=Path('/mnt/data/g_half_input');p=R/'reports/documentation/20261015/g_half_implementation/qualification'
for n in ['baseline_identity.json','receive_verification.json','duplicate_cleanup.json']:
 o=json.loads((p/n).read_text());print(n,json.dumps({k:v for k,v in o.items() if k not in ('files','removed','omitted_duplicate_sources')},ensure_ascii=False,indent=2))
 for k in ('files','removed','omitted_duplicate_sources'):
  if k in o:
   v=o[k]
   print(k, {'count':len(v),'first':v[0] if v else None} if isinstance(v,list) else v)
for n in ['actual_capture_restore','independent_full_graph','local_precision','objective_witnesses']:
 print(n,(p/f'cpu/attempt01/result/{n}.json').read_text())
for n in ['comparison_and_permissions','orchestration']:
 print(n,(p/f'cpu/attempt02/result/{n}.json').read_text())
