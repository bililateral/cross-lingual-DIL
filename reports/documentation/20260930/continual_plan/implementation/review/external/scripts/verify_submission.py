import hashlib,json,zipfile,pathlib,sys
root=pathlib.Path('/mnt/data/bge_continual_external_audit'); target=root/'source'
p=pathlib.Path('/mnt/data/bge_continual_review.zip'); h=lambda b:hashlib.sha256(b).hexdigest()
raw=p.read_bytes(); expected='d0917adce08dcb8a836af7cd2a14ec1414c92d96d314b7149770828303dae6db'
assert len(raw)==466624 and h(raw)==expected
with zipfile.ZipFile(p) as z:
 assert len(z.infolist())==53 and z.testzip() is None
 inv=json.loads(z.read('source_inventory.json')); rows=[]
 assert len(inv['files'])==52
 assert set(z.namelist())=={r['path'] for r in inv['files']}|{'source_inventory.json'}
 for r in inv['files']:
  b=z.read(r['path']); assert len(b)==r['bytes'] and h(b)==r['sha256'],r['path']; rows.append(dict(r,verified=True))
 for n in z.namelist():
  rp=pathlib.PurePosixPath(n); assert not rp.is_absolute() and '..' not in rp.parts
  out=target/n; out.parent.mkdir(parents=True,exist_ok=True); out.write_bytes(z.read(n))
  if out.suffix in ('.py','.md','.json','.sh'):
   number=root/'numbered_sources'/str(n+'.txt'); number.parent.mkdir(parents=True,exist_ok=True)
   text=out.read_text(); number.write_text(''.join(f'{i:5d} {line}\n' for i,line in enumerate(text.splitlines(),1)))
result={'zip_path':str(p),'bytes':len(raw),'sha256':h(raw),'expected_match':True,'members':53,'source_count':52,'all_source_files':rows,'crc_ok':True}
(root/'outputs/submission_verification.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k!='all_source_files'},ensure_ascii=False,indent=2))
