from pathlib import Path
import hashlib,json,zipfile
R=Path('/mnt/data/bge_continual_external_audit');zpath=Path('/mnt/data/bge_continual_review.zip');source=R/'source'
rows=[]
with zipfile.ZipFile(zpath) as z:
 assert z.testzip() is None
 names=z.namelist()
 for name in names:
  b=(source/name).read_bytes();original=z.read(name);assert b==original,name
  rows.append({'path':name,'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest(),'unchanged':True})
 extras=sorted(p.relative_to(source).as_posix() for p in source.rglob('*') if p.is_file() and p.relative_to(source).as_posix() not in names)
 assert not extras,extras
out={'status':'PASS','zip_bytes':zpath.stat().st_size,'zip_sha256':hashlib.sha256(zpath.read_bytes()).hexdigest(),'members':len(rows),'all_original_bytes_unchanged':True,'source_directory_new_files':extras,'files':rows}
(R/'outputs/final_source_verification.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({k:v for k,v in out.items() if k!='files'},indent=2))
