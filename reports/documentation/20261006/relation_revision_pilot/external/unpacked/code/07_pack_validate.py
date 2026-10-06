"""Package this review's original input, selected source mirror, and actual outputs.
The per-file manifest excludes only itself; outer hash is published separately.
"""
from pathlib import Path
import json,hashlib,shutil,zipfile
R=Path(__file__).resolve().parents[1];parent=R.parent
stage=parent/'relation_revision_pilot_delivery'
if stage.exists():shutil.rmtree(stage)
stage.mkdir()
for name in ('README.zh.md','RELATION_REVISION_PILOT.external_review.zh.md'):
 shutil.copy2(R/name,stage/name)
for name in ('code','outputs'):
 shutil.copytree(R/name,stage/name)
(stage/'input').mkdir();inp=parent/'relation_revision_pilot_review.zip'
shutil.copy2(inp,stage/'input'/inp.name)
for p in (R/'sources/current').rglob('*'):
 if not p.is_file() or p.relative_to(R/'sources/current').as_posix()=='history/relation_revision_review.zip':continue
 dest=stage/'sources/current'/p.relative_to(R/'sources/current');dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dest)
records=[]
for p in sorted(stage.rglob('*')):
 if p.is_file():
  b=p.read_bytes();records.append({'path':p.relative_to(stage).as_posix(),'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()})
manifest={'schema':'independent-review-per-file-sha256-v1','scope':'all delivered files except manifest.json itself','original_input_sha256':hashlib.sha256(inp.read_bytes()).hexdigest(),'files':records}
(stage/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
out=parent/'RELATION_REVISION_PILOT.audit_evidence.zip'
with zipfile.ZipFile(out,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
 for p in sorted(stage.rglob('*')):
  if p.is_file():z.write(p,p.relative_to(stage).as_posix())
with zipfile.ZipFile(out) as z:
 assert z.testzip() is None
 rows=json.loads(z.read('manifest.json'))['files']
 assert len(z.namelist())==len(set(z.namelist()))==len(rows)+1
 assert set(z.namelist())=={r['path'] for r in rows}|{'manifest.json'}
 for row in rows:
  b=z.read(row['path']);assert len(b)==row['bytes'] and hashlib.sha256(b).hexdigest()==row['sha256']
shutil.copy2(R/'RELATION_REVISION_PILOT.external_review.zh.md',parent/'RELATION_REVISION_PILOT.external_review.zh.md')
identity={'path':str(out),'bytes':out.stat().st_size,'sha256':hashlib.sha256(out.read_bytes()).hexdigest(),'files':len(records)+1,'manifest_payloads':len(records),'all_payloads_verified':True}
(parent/'RELATION_REVISION_PILOT.delivery_identity.json').write_text(json.dumps(identity,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(identity,ensure_ascii=False,indent=2))
