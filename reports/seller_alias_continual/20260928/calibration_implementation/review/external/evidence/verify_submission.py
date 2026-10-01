from pathlib import Path
import json, hashlib, zipfile, difflib
ROOT=Path(__file__).resolve().parents[1]
SRC=ROOT/'submission'; E=ROOT/'evidence'
def digest(p):
 b=p.read_bytes(); return {'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()}
def verify(rows, root=SRC):
 checked=[]
 for row in rows:
  d=digest(root/row['path']); assert d['bytes']==row['bytes'] and d['sha256']==row['sha256'],row['path']
  checked.append({'path':row['path'],**d})
 return checked
inv=json.loads((SRC/'source_inventory.json').read_text())
rows=verify(inv['files']); assert len(rows)==102 and sum(r['bytes'] for r in rows)==inv['total_bytes']
zip_path=ROOT.parent/'calibration_review.zip'
with zipfile.ZipFile(zip_path) as z:
 assert len(z.infolist())==103 and z.testzip() is None
 assert set(z.namelist())=={r['path'] for r in rows}|{'source_inventory.json'}
 assert all((SRC/n).read_bytes()==z.read(n) for n in z.namelist())
prefix='reports/seller_alias_continual/20260928/calibration_implementation/'
primary=json.loads((SRC/prefix/'primary_check.json').read_text()); verify(primary['source_files']); verify(primary['evidence_files'])
sync=json.loads((SRC/prefix/'cpu_sync.json').read_text()); verify(sync['files'])
assert len(sync['files'])==sync['count'] and sum(r['bytes'] for r in sync['files'])==sync['bytes']
old_prefix='reports/seller_alias_continual/20260927/ranking_execution/20260927_114646/job/'
manifest=json.loads((SRC/old_prefix/'run/manifest.json').read_text()); verify(manifest['source_files'])
changed=[]; cpus={}
for stamp in ('20260928_121500','20260928_121900'):
 d=SRC/prefix/'cpu'/stamp
 aud=json.loads((d/'evidence/audit.json').read_text()); verify([aud['handmade']],d/'evidence')
 for row in aud['source_files']:
  current=digest(SRC/row['path'])
  if current!={'bytes':row['bytes'],'sha256':row['sha256']}:
   snap=d/'source'/row['path']; assert digest(snap)=={'bytes':row['bytes'],'sha256':row['sha256']}
   changed.append({'stamp':stamp,'path':row['path'],'snapshot_verified':True})
   name=row['path'].replace('/','__')+'.diff'
   (E/name).write_text(''.join(difflib.unified_diff(snap.read_text().splitlines(True),(SRC/row['path']).read_text().splitlines(True),fromfile='initial/'+row['path'],tofile='current/'+row['path'])))
 cpus[stamp]={'contracts':aud['contracts'],'environment':aud['environment'],'exit_status':(d/'exit_status.txt').read_text().strip(),'seconds':aud['seconds'],'handmade':aud['handmade']}
result={'archive':digest(zip_path),'members':103,'sources':102,'source_bytes':sum(r['bytes'] for r in rows),'all_inventory_verified':True,'old_frozen_files_verified':len(manifest['source_files']),'current_stage_source_files':len(primary['source_files']),'cpu_sync':{'files':sync['count'],'bytes':sync['bytes'],'verified':True},'cpu_records':cpus,'initial_changed_snapshots':changed,'formal_arrays_loaded':0,'formal_labels_read':0}
(E/'submission_verification.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)); print(json.dumps(result,ensure_ascii=False,indent=2))
