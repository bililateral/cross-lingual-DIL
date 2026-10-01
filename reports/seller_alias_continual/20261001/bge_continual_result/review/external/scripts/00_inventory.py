"""Independent archive integrity verification. Does not load labels or project modules."""
import zipfile,pathlib,json,hashlib,stat,sys,platform,importlib.metadata,datetime
p=pathlib.Path('/mnt/data/bge_continual_result_review.zip'); dest=pathlib.Path('/mnt/data/bge_input'); out=pathlib.Path('/mnt/data/bge_review_evidence/outputs')
h=lambda b:hashlib.sha256(b).hexdigest()
with zipfile.ZipFile(p) as z:
 names=z.namelist(); inv=json.loads(z.read('source_inventory.json')); expected=[r['archive_path'] for r in inv['files']]
 assert len(names)==len(set(names));assert len(expected)==len(set(expected)); assert set(names)==set(expected)|{'source_inventory.json'}
 assert z.testzip() is None
 rows=[]
 for r in inv['files']:
  n=r['archive_path']; d=z.read(n); q=pathlib.PurePosixPath(n)
  assert not q.is_absolute() and '..' not in q.parts; assert not stat.S_ISLNK(z.getinfo(n).external_attr>>16)
  assert len(d)==r['bytes'] and h(d)==r['sha256'], n
  rows.append({'path':n,'bytes':len(d),'sha256':h(d),'source_path':r['source_path']})
 assert len(rows)==inv['file_count']; assert sum(r['bytes'] for r in rows)==inv['total_source_bytes']
 dest.mkdir(exist_ok=True)
 for n in names:
  q=dest/n;q.parent.mkdir(parents=True,exist_ok=True);q.write_bytes(z.read(n))
 result={'archive':str(p),'archive_bytes':p.stat().st_size,'archive_sha256':h(p.read_bytes()),'zip_members':len(names),'payload_files':len(rows),'payload_bytes':sum(r['bytes'] for r in rows),'uncompressed_total':sum(z.getinfo(n).file_size for n in names),'all_payload_hashes_match':True,'crc_ok':True,'member_set_exact':True,'inventory_created_at_utc':inv['created_at_utc'],'rows':rows}
(out/'inventory_verification.json').write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')
(out/'archive_members.txt').write_text('\n'.join(names)+'\n')
 print(json.dumps({k:v for k,v in result.items() if k!='rows'},indent=2));print('Extracted to',dest)
env={'python':sys.version,'executable':sys.executable,'platform':platform.platform(),'time_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'packages':{}}
for name in ['numpy','scipy','pandas','torch','scikit-learn','sentence-transformers','transformers']:
 try: env['packages'][name]=importlib.metadata.version(name)
 except importlib.metadata.PackageNotFoundError: env['packages'][name]=None
(out/'environment.json').write_text(json.dumps(env,indent=2)+'\n');print(json.dumps(env,indent=2))
