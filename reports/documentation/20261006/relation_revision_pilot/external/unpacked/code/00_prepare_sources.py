"""Offline reconstruction of the supplied current and nested-history archives.
This utility extracts files only; it does not run a test, model, or project command.
"""
from pathlib import Path,PurePosixPath
import zipfile,hashlib,json
R=Path(__file__).resolve().parents[1]
p=R/'input/relation_revision_pilot_review.zip'
if not p.exists():p=R.parent/'relation_revision_pilot_review.zip'
def unpack(path,destination,expected):
 b=path.read_bytes();assert hashlib.sha256(b).hexdigest()==expected
 with zipfile.ZipFile(path) as z:
  assert z.testzip() is None
  names=z.namelist();assert len(names)==len(set(names))
  for item in z.infolist():
   name=PurePosixPath(item.filename)
   assert not name.is_absolute() and '..' not in name.parts
   assert ((item.external_attr>>16)&0o170000)!=0o120000
  z.extractall(destination)
unpack(p,R/'sources/current','992990d9c1b5f9c5053b6fc675509506ff4d66c5da22247d8c156e818ac2de84')
unpack(R/'sources/current/history/relation_revision_review.zip',R/'sources/history','480371343ea0be50e4f8fcc614ec1aee5b35b20593151b7fbb3932e75669a9b5')
print('Original current and historical archives reconstructed; no project commands executed.')
