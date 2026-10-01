import json
from pathlib import Path
r=Path('/mnt/data/test_result_audit_20260928/project');b=r/'reports/seller_alias_continual/20260928'
p=json.loads((r/'schema/step28_alias_test_policy.json').read_text())
a=json.loads(json.loads((b/'test_execution/monitoring/repair_verification.json').read_text())['stdout'])['inputs']
e=[p['data']['manifest'],p['data']['validation'],*p['data']['inputs'].values()]
print('ACTUAL',json.dumps(a,ensure_ascii=False,indent=2));print('EXPECTED',json.dumps(e,ensure_ascii=False,indent=2))
print('actual paths',[x['path'] for x in a]);print('expected paths',[x['path'] for x in e])
def identity(rows):return {x['path']:{k:x[k] for k in ('path','bytes','sha256')} for x in rows}
print('BY_PATH_IDENTITY_EQUAL',identity(a)==identity(e))
print('CURRENT_SOURCE_INPUT_COUNT',len(p['data']['inputs']))
assert identity(a)==identity(e)
