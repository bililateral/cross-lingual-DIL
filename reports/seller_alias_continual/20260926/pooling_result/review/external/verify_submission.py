from pathlib import Path
import hashlib,json,zipfile,platform,sys,datetime
root=Path('/mnt/data/pooling_result_audit_workspace')
archive=Path('/mnt/data/pooling_result_review.zip')
invpath=root/'reports/seller_alias_continual/20260926/pooling_result/review/source_inventory.json'
inv=json.loads(invpath.read_text())
records=[]
for r in inv['files']:
 p=root/r['path'];actual={'path':r['path'],'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
 actual['matches']=actual['bytes']==r['bytes'] and actual['sha256']==r['sha256'];records.append(actual)
assert len(records)==165 and all(x['matches'] for x in records)
assert sum(x['bytes'] for x in records)==inv['total_bytes']
with zipfile.ZipFile(archive) as z:
 assert z.testzip() is None
 archive_names=set(z.namelist())
 assert archive_names==set(r['path'] for r in records)|{invpath.relative_to(root).as_posix()}
previous={}
with zipfile.ZipFile('/mnt/data/pooling_review(1).zip') as z:
 for name in z.namelist():
  if (root/name).is_file() and (name.startswith('scripts/') or name.startswith('schema/') or name.startswith('tests/') or name.startswith('docs/')):
   previous[name]={'previous_sha256':hashlib.sha256(z.read(name)).hexdigest(),'current_sha256':hashlib.sha256((root/name).read_bytes()).hexdigest(),'unchanged':z.read(name)==(root/name).read_bytes()}
result={'archive':{'bytes':archive.stat().st_size,'sha256':hashlib.sha256(archive.read_bytes()).hexdigest(),'members':len(archive_names)},'manifest_files':len(records),'manifest_bytes':sum(x['bytes'] for x in records),'files':records,'previous_submission_comparison':previous,'environment':{'platform':platform.platform(),'python':sys.version},'checked_at':datetime.datetime.now(datetime.timezone.utc).isoformat()}
Path('/mnt/data/pooling_result_reviewer_evidence/submission_integrity.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k!='files'},ensure_ascii=False,indent=2))
