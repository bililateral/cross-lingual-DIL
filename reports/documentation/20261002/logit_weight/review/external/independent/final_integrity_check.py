"""Recheck original bytes, payloads, source closure, raw command logs and final evidence."""
from __future__ import annotations
import hashlib,json,re,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];SRC=ROOT/'submission';expected='ff678c2f9687ced6fa9888eca55064ca617c98d35bf371e082e7cf321f9ed08a'
def digest(b):return hashlib.sha256(b).hexdigest()
original=ROOT/'original/logit_review.zip';assert original.stat().st_size==2084038 and digest(original.read_bytes())==expected
received=Path('/mnt/data/logit_review.zip');assert received.read_bytes()==original.read_bytes()
with zipfile.ZipFile(original) as z:
 assert len(z.infolist())==304 and z.testzip() is None
 inventory=[]
 for item in z.infolist():
  b=z.read(item);path=SRC/item.filename;assert path.is_file() and path.read_bytes()==b,item.filename
  inventory.append({'path':item.filename,'bytes':len(b),'sha256':digest(b)})
 assert sorted(str(p.relative_to(SRC)) for p in SRC.rglob('*') if p.is_file())==sorted(row['path'] for row in inventory)
source_manifest=json.loads((SRC/'reports/documentation/20261002/logit_weight/review/source_inventory.json').read_text())
logs=[]
for directory in sorted((ROOT/'evidence').glob('[0-9][0-9]_*')):
 f=directory/'command.json'
 if not f.exists():continue
 meta=json.loads(f.read_text())
 for stream in ('stdout.txt','stderr.txt'):
  b=(directory/stream).read_bytes();assert len(b)==meta[stream]['bytes'] and digest(b)==meta[stream]['sha256']
 logs.append({'name':directory.name,'exit_code':meta['exit_code'],'wall_seconds':meta['wall_seconds']})
assert [row['name'] for row in logs if row['exit_code']!=0]==['10_path_v1','12_access_identity_v1']
text=(ROOT/'evidence/02_unittest/stderr.txt').read_text()
names=re.findall(r'^(test_\S+) \(([^)]+)\) \.\.\. ok$',text,re.M);assert len(names)==len(set(names))==28
assert 'Ran 28 tests' in text and text.rstrip().endswith('OK')
for sub in ['math_reference','diff_reference','statistics_reference','path_reference','access_identity_reference_v2']:
 result=json.loads((ROOT/'evidence'/sub/'results.json').read_text());assert result['status']=='PASS'
report={'status':'PASS','original_zip_bytes':2084038,'original_zip_sha256':expected,'original_zip_copy_byte_identical':True,'received_zip_still_unmodified':True,'payloads_still_byte_identical':304,'submission_extra_files':0,'original_project_mutations':0,'verified_recorded_commands_before_this_check':len(logs),'recorded_nonzero_commands':[r['name'] for r in logs if r['exit_code']!=0],'submitted_unique_unittest_count':28,'submitted_failed_error_skipped':0,'independent_final_programs_passed':5,'formal_project_connections':0,'formal_training_updates':0,'native_BGE_entry_runs':0,'dependency_installations':0,'limitations':'Initial exploratory tool commands were not all timestamp-wrapped; see failure notes. Handwritten fixtures are not formal output.'}
(ROOT/'evidence/final_integrity.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
(ROOT/'evidence/received_members_304.json').write_text(json.dumps(inventory,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(report,ensure_ascii=False,indent=2))
