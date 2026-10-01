"""Package immutable evidence after all scientist process logs have closed."""
from pathlib import Path
import hashlib,json,zipfile,sys,platform,datetime
root=Path(sys.argv[1]).resolve(); dest=Path(sys.argv[2]); receipt=Path(sys.argv[3])
manifest=root/'evidence_sha256.json'
def hash_bytes(b): return hashlib.sha256(b).hexdigest()
rows=[]
for p in sorted(root.rglob('*')):
    if p.is_file() and p!=manifest:
        b=p.read_bytes();rows.append({'path':p.relative_to(root).as_posix(),'bytes':len(b),'sha256':hash_bytes(b)})
manifest.write_text(json.dumps({'scope':'all evidence files except this manifest itself','count':len(rows),'files':rows},ensure_ascii=False,indent=2))
with zipfile.ZipFile(dest,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=9) as z:
    for r in rows:z.write(root/r['path'],r['path'])
    z.write(manifest,manifest.name)
with zipfile.ZipFile(dest) as z:
    assert z.testzip() is None;assert len(z.namelist())==len(rows)+1
    for r in rows:
        b=z.read(r['path']);assert len(b)==r['bytes'] and hash_bytes(b)==r['sha256'];assert b==(root/r['path']).read_bytes()
    assert z.read(manifest.name)==manifest.read_bytes()
out={'status':'PASS','created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'zip_path':str(dest),'zip_bytes':dest.stat().st_size,'zip_sha256':hash_bytes(dest.read_bytes()),'zip_members':len(rows)+1,'manifested_files':len(rows),'uncompressed_bytes':sum(r['bytes'] for r in rows)+manifest.stat().st_size,'manifest_sha256':hash_bytes(manifest.read_bytes()),'zip_crc_passed':True,'all_member_hashes_verified':True,'command':sys.argv,'python':sys.version,'platform':platform.platform()}
receipt.write_text(json.dumps(out,ensure_ascii=False,indent=2));print(json.dumps(out,ensure_ascii=False,indent=2))
