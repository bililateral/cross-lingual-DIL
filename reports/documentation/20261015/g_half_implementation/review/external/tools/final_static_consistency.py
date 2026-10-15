"""Final review-artifact consistency checks only; no project execution or data."""
from pathlib import Path
import hashlib,json,zipfile
R=Path('/mnt/data/g_half_input');O=Path('/mnt/data/g_half_review')
identity=json.loads((O/'evidence/input_identity.json').read_text())
assert identity['archive']['bytes']==322949
assert identity['archive']['sha256']=='f0f09f50daddbcf948bf9717edcb9b204fc19ea8f8a6d1ecc74b353f09f403e9'
assert identity['archive']['members']==78 and identity['payload_count']==77 and identity['all_payloads_ok']
scope=json.loads((O/'reading_scope.json').read_text());assert len(scope['files'])==78
for r in scope['files']:
 p=R/r['path'];assert p.stat().st_size==r['bytes'] and hashlib.sha256(p.read_bytes()).hexdigest()==r['sha256']
 for lo,hi in r['read_ranges']:assert 1<=lo<=hi<=r['total_text_lines']
json_files=list(O.rglob('*.json'))
for p in json_files:json.loads(p.read_text())
f=json.loads((O/'findings.json').read_text())
assert f['open_demonstrated_scientific_blockers']==[] and f['open_necessary_reproducibility_defects']==[]
assert f['formal_launch_qualified_now'] is False and f['project_code_modified_by_reviewer'] is False
p=json.loads((O/'checks/pure_checks_result.json').read_text())
assert [p[k] for k in ['model_instances','memory_instances','optimizer_calls','formal_asset_reads']]==[0,0,0,0]
assert p['environment']['cuda_initialized'] is False
for name in ['handwritten_inputs.npz','raw_numeric_outputs.npz']:
 with zipfile.ZipFile(O/'checks'/name) as z:assert z.testzip() is None
assert not list(O.rglob('review_input*.zip'))
print(json.dumps({'status':'PASS_REVIEW_ARTIFACT_CONSISTENCY_ONLY','input_members_rehashed':78,'json_files_parsed':len(json_files),'reading_ranges_within_files':True,'handwritten_npz_crc_ok':True,'formal_launch_not_qualified':True,'project_sources_modified':False,'unexpected_errors':0},ensure_ascii=False,indent=2))
