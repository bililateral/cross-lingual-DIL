"""Build and verify delivery inventory; hashes only, no scientific recomputation."""
from pathlib import Path
import csv,hashlib,json,zipfile
O=Path('/mnt/data/hcons_review/return');D=Path('/mnt/data')
manifest_path=O/'return_manifest.json'
files=[p for p in sorted(O.rglob('*')) if p.is_file() and p!=manifest_path]
rows=[dict(path=p.relative_to(O).as_posix(),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in files]
manifest=dict(kind='H_cons_fixed_result_independent_review_return',payload_count=len(rows),payload_bytes=sum(x['bytes'] for x in rows),files=rows,self_excluded='return_manifest.json is excluded to avoid circular self-hash; complete ZIP member inventory including this manifest is provided externally.',original_input_sha256='588e79e3a265319b9025fe94d16b4cc583f2d35cf4a51aa7fbc9b1fe54e4fb15')
manifest_path.write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
allfiles=sorted([*files,manifest_path]);zpath=D/'h_cons_result_independent_review_return.zip'
with zipfile.ZipFile(zpath,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=9) as z:
 for p in allfiles:z.write(p,p.relative_to(O).as_posix())
with zipfile.ZipFile(zpath) as z:
 assert z.testzip() is None and len(z.namelist())==len(set(z.namelist()))==len(allfiles)
 stored=json.loads(z.read('return_manifest.json'))
 for r in stored['files']:
  b=z.read(r['path']);assert len(b)==r['bytes'] and hashlib.sha256(b).hexdigest()==r['sha256']
 assert hashlib.sha256(z.read('input_original.zip')).hexdigest()==manifest['original_input_sha256']
 member_rows=[dict(path=i.filename,bytes=i.file_size,sha256=hashlib.sha256(z.read(i.filename)).hexdigest()) for i in z.infolist()]
tsv=D/'h_cons_review_return_members.tsv'
with tsv.open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=['path','bytes','sha256'],delimiter='\t');w.writeheader();w.writerows(member_rows)
meta=dict(archive=zpath.name,bytes=zpath.stat().st_size,sha256=hashlib.sha256(zpath.read_bytes()).hexdigest(),zip_members=len(member_rows),uncompressed_bytes=sum(r['bytes'] for r in member_rows),payloads_excluding_manifest=len(rows),all_members_verified=True,original_input_unchanged=True,member_inventory=tsv.name,member_inventory_sha256=hashlib.sha256(tsv.read_bytes()).hexdigest())
(D/'h_cons_review_delivery.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2)+'\n')
(D/'h_cons_review_SHA256SUMS.txt').write_text(meta['sha256']+'  '+zpath.name+'\n'+meta['member_inventory_sha256']+'  '+tsv.name+'\n'+manifest['original_input_sha256']+'  h_cons_result_review_input.zip\n')
print(json.dumps(meta,ensure_ascii=False,indent=2))
