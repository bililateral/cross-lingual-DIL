"""Package only the completed current re-audit evidence; manifest excludes itself."""
from pathlib import Path
import hashlib, json, zipfile
B=Path('/mnt/data/ranking_result_reaudit_20260928');E=B/'evidence'
manifest=E/'evidence_inventory.json';out=Path('/mnt/data/ranking_result_resubmission_evidence.zip')
files=[]
for p in sorted(E.rglob('*')):
 if p.is_file() and p!=manifest:
  b=p.read_bytes();files.append({'path':p.relative_to(E).as_posix(),'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()})
manifest.write_text(json.dumps({'scope':'Current attachment re-audit; original result ZIP not duplicated; manifest excludes itself','files':files,'files_count':len(files),'total_bytes_excluding_manifest':sum(x['bytes'] for x in files)},ensure_ascii=False,indent=2)+'\n')
with zipfile.ZipFile(out,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
 for p in sorted(E.rglob('*')):
  if p.is_file():z.write(p,p.relative_to(E).as_posix())
with zipfile.ZipFile(out) as z:
 assert z.testzip() is None and len(z.namelist())==len(files)+1
 for r in files:
  b=z.read(r['path']);assert len(b)==r['bytes'] and hashlib.sha256(b).hexdigest()==r['sha256']
receipt={'status':'PACKAGED_AND_VERIFIED','archive':str(out),'members':len(files)+1,'bytes':out.stat().st_size,'sha256':hashlib.sha256(out.read_bytes()).hexdigest(),'all_member_hashes_match':True,'original_submission_preserved':True}
(B/'packaging_receipt.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(receipt,ensure_ascii=False,indent=2))
