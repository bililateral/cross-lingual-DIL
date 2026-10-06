from pathlib import Path, PurePosixPath
import hashlib,json,zipfile,sys
R=Path(__file__).resolve().parents[2]
P=R/'original_input.zip'
b=P.read_bytes(); d={'file':P.name,'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()}
assert d['bytes']==12917110 and d['sha256']=='5e11c03cc89a66466b122f4ad8e503e02b481f5d773bdb74cb6319c81f4bfa1e'
with zipfile.ZipFile(P) as z:
 names=z.namelist(); d['entries']=len(names);d['files']=sum(not v.is_dir() for v in z.infolist());d['crc_error']=z.testzip();assert d['crc_error'] is None
 assert len(names)==len(set(names))
 for n in names:
  p=PurePosixPath(n);assert not p.is_absolute() and '..' not in p.parts
 z.extractall(R/'input')
 d['names']=names
print(json.dumps(d,ensure_ascii=False,indent=2))
(R/'audit/outputs/package_identity.json').write_text(json.dumps(d,ensure_ascii=False,indent=2))
print('MANIFEST:')
print((R/'input/manifest.json').read_text()[:10000])
