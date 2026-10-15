#!/usr/bin/env python3
"""Package generated audit deliverables and create exact per-member size/SHA manifests."""
import ast,csv,hashlib,io,json,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
DEST=ROOT.parents[1]
BASE='sgf_result_external_review_20261015'
def sha(b):return hashlib.sha256(b).hexdigest()
def tsv(rows):
    s=io.StringIO(newline='');w=csv.DictWriter(s,fieldnames=['path','bytes','sha256'],delimiter='\t',lineterminator='\n');w.writeheader();w.writerows(rows);return s.getvalue().encode('utf-8')
for p in ROOT.rglob('*.py'):ast.parse(p.read_text())
for p in ROOT.rglob('*.json'):json.loads(p.read_text())
# Exclude only the internal manifest itself; it cannot contain its own SHA.
items=sorted(p for p in ROOT.rglob('*') if p.is_file() and p.name!='payload_manifest.tsv' and '__pycache__' not in p.parts)
rows=[dict(path=p.relative_to(ROOT).as_posix(),bytes=p.stat().st_size,sha256=sha(p.read_bytes())) for p in items]
(ROOT/'payload_manifest.tsv').write_bytes(tsv(rows))
items.append(ROOT/'payload_manifest.tsv');items.sort()
zpath=DEST/(BASE+'.zip')
with zipfile.ZipFile(zpath,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=9) as z:
    for p in items:
        info=zipfile.ZipInfo(p.relative_to(ROOT).as_posix(),date_time=(2026,10,15,12,0,0));info.compress_type=zipfile.ZIP_DEFLATED;info.external_attr=0o100644<<16
        z.writestr(info,p.read_bytes(),compress_type=zipfile.ZIP_DEFLATED,compresslevel=9)
with zipfile.ZipFile(zpath) as z:
    assert z.testzip() is None
    external=[]
    for n in z.namelist():
        b=z.read(n);external.append(dict(path=n,bytes=len(b),sha256=sha(b)))
    internal=list(csv.DictReader(io.StringIO(z.read('payload_manifest.tsv').decode()),delimiter='\t'))
    assert len(internal)+1==len(external)
    for r in internal:
        b=z.read(r['path']);assert int(r['bytes'])==len(b) and r['sha256']==sha(b)
manifest=DEST/(BASE+'_members.tsv');manifest.write_bytes(tsv(external))
identity={'zip_filename':zpath.name,'zip_bytes':zpath.stat().st_size,'zip_members':len(external),'zip_sha256':sha(zpath.read_bytes()),'zip_uncompressed_member_bytes':sum(r['bytes'] for r in external),'member_manifest_filename':manifest.name,'member_manifest_bytes':manifest.stat().st_size,'member_manifest_sha256':sha(manifest.read_bytes()),'internal_manifest_excludes_only_itself':True,'original_input_bytes':6123032,'original_input_members':376,'original_input_sha256':'07f941b2d689c4f9e2cd024364b57e3607bd9bb762868378a854b5be97ef83f6','input_archive_repackaged':False}
(DEST/(BASE+'_identity.json')).write_text(json.dumps(identity,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(identity,ensure_ascii=False,indent=2))
