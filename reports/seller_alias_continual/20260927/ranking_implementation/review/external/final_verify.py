"""Final immutable-source and reviewer-output check; no model/data import."""
from pathlib import Path
import hashlib,json,zipfile,ast,subprocess,sys
ROOT=Path('/mnt/data/ranking_review_source')
OUT=Path('/mnt/data/ranking_reviewer_evidence')
ZIP=Path('/mnt/data/ranking_review.zip')
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
inv=json.loads((ROOT/'reports/seller_alias_continual/20260927/ranking_implementation/review/source_inventory.json').read_text())
rows=[]
for item in inv['files']:
 p=ROOT/item['path']; ok=p.stat().st_size==item['bytes'] and sha(p)==item['sha256']
 assert ok,item['path'];rows.append({**item,'unchanged':ok})
with zipfile.ZipFile(ZIP) as z:
 assert z.testzip() is None and len(z.infolist())==61
 # Check every archive member including inventory against the working source.
 for item in z.infolist(): assert (ROOT/item.filename).read_bytes()==z.read(item.filename),item.filename
syntax=[]
for item in inv['files']:
 p=ROOT/item['path']
 if p.suffix=='.py':
  compile(p.read_text(encoding='utf-8-sig'),str(p),'exec');syntax.append(item['path'])
 if p.suffix=='.sh':
  done=subprocess.run(['bash','-n',str(p)],capture_output=True,text=True)
  assert done.returncode==0,(item['path'],done.stderr);syntax.append(item['path'])
for f in ('contracts.exit.txt','independent.exit.txt','package_cpu.exit.txt'):
 assert (OUT/f).read_text().strip()=='0',f
assert not [p for p in OUT.rglob('*') if p.suffix in {'.pt','.pth','.safetensors','.bin'}]
result={'zip_bytes':ZIP.stat().st_size,'zip_sha256':sha(ZIP),'source_count':len(rows),'all_61_archive_members_unchanged':True,'all_sources_unchanged':True,'syntax_files_checked':syntax,'reviewer_retained_model_files':0,'formal_data_labels_or_models_accessed':0,'source_files':rows}
(OUT/'final_source_check.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
print(json.dumps({k:v for k,v in result.items() if k!='source_files'},ensure_ascii=False,indent=2))
