"""Materialize and verify the one exact authorized archive for offline review."""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile

OUT=Path(__file__).resolve().parent
ap=argparse.ArgumentParser();ap.add_argument('archive',type=Path);args=ap.parse_args()
raw=args.archive.read_bytes();digest=hashlib.sha256(raw).hexdigest()
assert len(raw)==47913772 and digest=='5b410fa7b80fb9fd89a02555a898014f3877e53620dfc170a2ffcc02870da2d1'
z=zipfile.ZipFile(args.archive);assert len(z.infolist())==1629
target=OUT.parent/'input';target.mkdir(exist_ok=True)
for item in z.infolist():
    path=target/item.filename
    assert path.resolve().is_relative_to(target.resolve())
    if item.is_dir():path.mkdir(parents=True,exist_ok=True);continue
    payload=z.read(item.filename)
    if path.exists():assert path.read_bytes()==payload
    else:path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(payload)
m=json.loads((target/'package_manifest.json').read_text())
fail=[]
for r in m['files']:
    p=target/r['path'];b=p.read_bytes()
    if len(b)!=r['bytes'] or hashlib.sha256(b).hexdigest()!=r['sha256']:fail.append(r['path'])
expected={x['path'] for x in m['files']}|{'package_manifest.json'}
actual={str(p.relative_to(target)) for p in target.rglob('*') if p.is_file()}
check={'payload_count':len(m['files']),'payload_bytes':sum(x['bytes'] for x in m['files']),
       'hash_or_size_failures':fail,'missing':sorted(expected-actual),'extra':sorted(actual-expected)}
assert not(fail or check['missing'] or check['extra'])
identity={'uploaded_name':args.archive.name,'bytes':len(raw),'sha256':digest,'members':len(z.infolist()),
          'uncompressed_bytes':sum(x.file_size for x in z.infolist())}
(OUT/'input_identity.json').write_text(json.dumps(identity,ensure_ascii=False,indent=2)+'\n')
(OUT/'package_check.json').write_text(json.dumps(check,ensure_ascii=False,indent=2)+'\n')
(OUT/'input_inventory.json').write_text(json.dumps([{'path':x.filename,'bytes':x.file_size,'compressed_bytes':x.compress_size} for x in z.infolist()],ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'status':'PASS','identity':identity,'manifest':check,'scope':'identity verification only; not semantic acceptance'},ensure_ascii=False,indent=2))
