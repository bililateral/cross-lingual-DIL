from pathlib import Path
import json,hashlib,zipfile,sys
R=Path('/mnt/data/h_review_input'); O=Path('/mnt/data/h_external_review')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
p=Path('/mnt/data/review_input(20261016-014443).zip'); m=json.loads((R/'manifest.json').read_text()); rows=m['payloads']; defects=[]
for x in rows:
 f=R/x['path']
 if not f.is_file() or f.stat().st_size!=x['bytes'] or sha(f)!=x['sha256']: defects.append(x['path'])
actual={str(f.relative_to(R)) for f in R.rglob('*') if f.is_file()}
report={'input_name':p.name,'bytes':p.stat().st_size,'sha256':sha(p),'zip_members':len(zipfile.ZipFile(p).infolist()),'manifest_payloads':len(rows),'payload_failures':defects,'unlisted':sorted(actual-{x['path'] for x in rows}-{'manifest.json'}),'manifest_metadata':{k:v for k,v in m.items() if k!='payloads'}}
(O/'checks/input_identity.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
print(json.dumps(report,ensure_ascii=False,indent=2))
for n in ['result/job/before_valid.json','result/job/completion.json','result/job/execution.json','result/job/resource.json','result/analysis/audit.json','result/analysis_wallclock.json','result/closeout.json','context/baseline_sources.json','context/formal_gate.json','result/analysis_inventory.json']:
 print('\n########',n);t=(R/n).read_text();print(t)
