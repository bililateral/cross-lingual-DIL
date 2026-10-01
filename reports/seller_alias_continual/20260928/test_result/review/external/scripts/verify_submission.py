import hashlib,json,pathlib,zipfile
BASE=pathlib.Path('/mnt/data/test_result_audit_20260928');ZIP=pathlib.Path('/mnt/data/test_result_review(1).zip');ROOT=BASE/'project';OUT=BASE/'evidence/outputs'
sha=lambda b:hashlib.sha256(b).hexdigest()
with zipfile.ZipFile(ZIP) as z:
 assert z.testzip() is None
 names=z.namelist();assert len(names)==len(set(names))
 inv=json.loads(z.read('source_inventory.json'));recs=inv['files']
 assert set(names)=={r['path'] for r in recs}|{'source_inventory.json'}
 assert len(recs)==inv['source_count'];assert sum(r['bytes'] for r in recs)==inv['total_bytes']
 checks=[]
 for r in recs:
  b=z.read(r['path']);assert len(b)==r['bytes'] and sha(b)==r['sha256'],r['path'];checks.append(r)
 ROOT.mkdir(exist_ok=False)
 for n in names:
  rel=pathlib.PurePosixPath(n);assert not rel.is_absolute() and '..' not in rel.parts
  dest=ROOT/n;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(z.read(n))
 result={'zip_name':ZIP.name,'zip_bytes':ZIP.stat().st_size,'zip_sha256':sha(ZIP.read_bytes()),'members':len(names),'sources':len(recs),'source_bytes':inv['total_bytes'],'inventory_verified':True,'zip_integrity':True,'extraction_root':str(ROOT),'files':checks}
 (OUT/'source_verification.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
 print(json.dumps({k:v for k,v in result.items() if k!='files'},ensure_ascii=False,indent=2))
