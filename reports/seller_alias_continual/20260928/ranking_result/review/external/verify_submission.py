"""Verify the actual re-upload, then extract only its listed non-secret review files."""
import hashlib,json,zipfile,sys,platform
from pathlib import Path, PurePosixPath
base=Path('/mnt/data/ranking_result_reaudit_20260928'); zpath=Path('/mnt/data/ranking_result.zip')
h=lambda b:hashlib.sha256(b).hexdigest()
assert zpath.stat().st_size==7579246
assert h(zpath.read_bytes())=='6d01ada467bc705044d31008302fb6a96aa6fd9fbb7817db4166dd06254c9474'
with zipfile.ZipFile(zpath) as z:
 names=z.namelist(); assert len(names)==253 and len(set(names))==253 and z.testzip() is None
 inv=json.loads(z.read('source_inventory.json')); assert len(inv['files'])==252
 assert set(names)=={x['path'] for x in inv['files']}|{'source_inventory.json'}
 for item in inv['files']:
  data=z.read(item['path']); assert len(data)==item['bytes'] and h(data)==item['sha256'],item['path']
 for name in names:
  q=PurePosixPath(name);assert not q.is_absolute() and '..' not in q.parts
  p=base/'submitted'/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(z.read(name))
 records=[{'path':n,'bytes':len(z.read(n)),'sha256':h(z.read(n))} for n in names]
 job_prefix='reports/seller_alias_continual/20260927/ranking_execution/20260927_114646/job/'
 jobs=[r for r in records if r['path'].startswith(job_prefix)]
 result={'zip_bytes':zpath.stat().st_size,'zip_sha256':h(zpath.read_bytes()),'zip_members':len(names),'verified_sources':252,'job_members':len(jobs),'job_bytes':sum(r['bytes'] for r in jobs),'python':sys.version,'platform':platform.platform(),'files':records}
 (base/'evidence/submission_verification.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
 print(json.dumps({k:v for k,v in result.items() if k!='files'},indent=2))
 print('ALL MEMBER PATHS:');print('\n'.join(names))
