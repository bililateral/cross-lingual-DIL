"""Build and validate this audit bundle; does not run project code."""
from pathlib import Path
import hashlib, json, zipfile, sys
R=Path(__file__).resolve().parents[2]
target=Path(sys.argv[1]) if len(sys.argv)>1 else R.parent/'FUNCTION_MEMORY_IMPLEMENTATION.audit_evidence.zip'
files=[R/'FUNCTION_MEMORY_IMPLEMENTATION.external_review.zh.md',R/'README.zh.md',R/'original_input.zip']
for folder in ('audit/scripts','audit/outputs','input'):
    for p in (R/folder).rglob('*'):
        rel=p.relative_to(R)
        if not p.is_file() or '__pycache__' in p.parts or 'micro_temp' in p.parts:continue
        if rel.parts[:2]==('input','history'):continue
        files.append(p)
files=sorted(set(files),key=lambda p:p.relative_to(R).as_posix())
rows=[{'path':p.relative_to(R).as_posix(),'bytes':p.stat().st_size,
       'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in files]
manifest={'kind':'FUNCTION_MEMORY_INDEPENDENT_IMPLEMENTATION_AUDIT',
          'scope':'WEB_CPU_HANDWRITTEN_AND_STATIC_ONLY',
          'source_archive_sha256':'5e11c03cc89a66466b122f4ad8e503e02b481f5d773bdb74cb6319c81f4bfa1e',
          'manifest_excludes_itself':True,'files':rows}
m=(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n').encode('utf-8')
(R/'manifest.json').write_bytes(m)
with zipfile.ZipFile(target,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
    for p in files:z.write(p,p.relative_to(R).as_posix())
    z.writestr('manifest.json',m)
with zipfile.ZipFile(target) as z:
    assert z.testzip() is None
    assert len(z.namelist())==len(rows)+1==len(set(z.namelist()))
    for row in rows:
        b=z.read(row['path']);assert len(b)==row['bytes']
        assert hashlib.sha256(b).hexdigest()==row['sha256']
print(json.dumps({'file':str(target),'bytes':target.stat().st_size,
    'sha256':hashlib.sha256(target.read_bytes()).hexdigest(),
    'files':len(rows)+1,'payloads':len(rows),'validated':True},indent=2))
