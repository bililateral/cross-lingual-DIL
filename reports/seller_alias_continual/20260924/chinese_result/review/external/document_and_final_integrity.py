from pathlib import Path
import json, hashlib, zipfile, re
ROOT=Path('/mnt/data/chinese_result_review/source')
OUT=Path('/mnt/data/chinese_result_review/evidence')
a=json.loads((OUT/'independent_result_audit.json').read_text())
doc=(ROOT/'docs/SELLER_ALIAS_CHINESE_RESULT.zh.md').read_text()
order=['historical_labse','mean_bce','mean_rank','split_bce','split_rank']
nums=0
for line in doc.splitlines():
 if not line.startswith('|'):continue
 cells=[x.strip() for x in line.strip('|').split('|')]
 if cells[0] in a['epoch6_metrics']['split_rank']:
  assert len(cells)==6
  for arm,text in zip(order,cells[1:]):
   assert abs(float(text)-a['epoch6_metrics'][arm][cells[0]])<=.500001e-6
   nums+=1
 if cells[0] in a['comparison_summary']:
  v=a['comparison_summary'][cells[0]];truth=[v['average_precision']['mean'],*v['average_precision']['conditional_95pct_interval'],v['frozen_pooled_recall']['difference'],*v['frozen_pooled_recall']['conditional_95pct_interval']]
  numbers=[float(x) for text in cells[1:] for x in re.findall(r'[-+]?\d+\.\d+',text)]
  assert len(numbers)==6 and all(abs(x-y)<=.500001e-6 for x,y in zip(numbers,truth))
  nums+=6
assert nums==152
inv=json.loads((ROOT/'source_inventory.json').read_text()); verified=[]
for f in inv['files']:
 p=ROOT/f['path'];payload=p.read_bytes();assert len(payload)==f['bytes'] and hashlib.sha256(payload).hexdigest()==f['sha256'];verified.append(f['path'])
source_paths=json.loads((ROOT/'reports/seller_alias_continual/20260924/chinese_evaluation/20260924_202700/execution.json').read_text())['frozen_sources_verified']
with zipfile.ZipFile('/mnt/data/chinese_supplement.zip') as z:
 unchanged=[]
 for f in source_paths:
  assert z.read(f['path'])==(ROOT/f['path']).read_bytes();unchanged.append(f['path'])
with zipfile.ZipFile('/mnt/data/package.zip') as z:
 assert len(z.infolist())==128 and len(set(z.namelist()))==128 and z.testzip() is None
 assert set(z.namelist())==set(verified)|{'source_inventory.json'}
b=Path('/mnt/data/package.zip').read_bytes();assert len(b)==3138095 and hashlib.sha256(b).hexdigest()=='8be3427e2e1574dff3d3317aa6e7680fee32377e1373cd6707196b63948e2796'
r={'status':'PASS','document_rounding_cells_checked':nums,'all_payloads_unchanged_after_review':len(verified),'sources_byte_identical_to_supplement':unchanged,'package_members':128,'package_bytes':len(b),'package_sha256':hashlib.sha256(b).hexdigest(),'source_runner_sha256':hashlib.sha256((ROOT/'scripts/step28_chinese_base.py').read_bytes()).hexdigest()}
(OUT/'document_and_final_integrity.json').write_text(json.dumps(r,ensure_ascii=False,indent=2)+'\n');print(json.dumps(r,ensure_ascii=False))
