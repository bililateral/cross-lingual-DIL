"""Package completed evidence read-only, excluding two self-referential inventories."""
from pathlib import Path
import hashlib,json,zipfile,time
R=Path('/mnt/data/bge_continual_external_audit');out=Path('/mnt/data/bge_continual_local_implementation_evidence.zip')
rows=[]
for p in sorted(R.rglob('*')):
 if not p.is_file() or p.name in ('artifact_inventory.json','SHA256SUMS.txt'):continue
 b=p.read_bytes();rows.append({'path':p.relative_to(R).as_posix(),'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()})
(R/'artifact_inventory.json').write_text(json.dumps({'scope':'All payloads excluding these two self-referential inventory files','files':rows,'payload_count':len(rows)},ensure_ascii=False,indent=2)+'\n')
(R/'SHA256SUMS.txt').write_text(''.join(f"{x['sha256']}  {x['path']}\n" for x in rows))
with zipfile.ZipFile(out,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
 for p in sorted(R.rglob('*')):
  if p.is_file():z.write(p,p.relative_to(R).as_posix())
with zipfile.ZipFile(out) as z:
 assert z.testzip() is None
 expected={x['path'] for x in rows}|{'artifact_inventory.json','SHA256SUMS.txt'}
 assert set(z.namelist())==expected
 for row in rows:
  b=z.read(row['path']);assert len(b)==row['bytes'] and hashlib.sha256(b).hexdigest()==row['sha256']
 receipt={'file':out.name,'bytes':out.stat().st_size,'sha256':hashlib.sha256(out.read_bytes()).hexdigest(),'members':len(z.namelist()),'payload_files':len(rows),'all_payloads_verified':True,'crc':'PASS','formal_assets_included':False,'all_source_originals_unchanged':'outputs/final_source_verification.json'}
Path('/mnt/data/bge_continual_local_evidence_receipt.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n');print(json.dumps(receipt,ensure_ascii=False,indent=2))
