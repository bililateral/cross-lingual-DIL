"""Mechanical ZIP/manifest verification only. Never import uploaded project code."""
from pathlib import Path
import hashlib, json, sys, zipfile
p=Path(sys.argv[1]); out=Path(sys.argv[2])
def h(b): return hashlib.sha256(b).hexdigest()
with zipfile.ZipFile(p) as z:
 names=z.namelist(); mname='reports/documentation/20261014/balanced_pooling/review/package_manifest.json'
 m=json.loads(z.read(mname)); rows=[]
 for f in m['files']:
  b=z.read(f['path']); rows.append({'path':f['path'],'actual_bytes':len(b),'actual_sha256':h(b),'match':len(b)==f['bytes'] and h(b)==f['sha256']})
 report={'filename':p.name,'bytes':p.stat().st_size,'sha256':h(p.read_bytes()),'zip_members':len(names),
 'zip_crc_first_bad_member':z.testzip(),'duplicate_members':len(names)!=len(set(names)),
 'manifest_path':mname,'manifest_self_excluded':m['manifest_self_excluded'],
 'declared_payload_count':m['file_count'],'actual_manifest_entries':len(rows),
 'manifest_exact_coverage':set(names)=={f['path'] for f in m['files']}|{mname},'rows':rows,
 'all_pass':len(names)==18 and len(rows)==17 and all(r['match'] for r in rows) and set(names)=={f['path'] for f in m['files']}|{mname}}
 out.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
 print(json.dumps({k:v for k,v in report.items() if k!='rows'},ensure_ascii=False,indent=2))
 assert report['all_pass'] and not report['duplicate_members'] and report['zip_crc_first_bad_member'] is None
