from pathlib import Path
import hashlib,zipfile,json,shutil
root=Path('/mnt/data/s_pos_static_review'); p=Path('/mnt/data/s_improvement_review_input.zip')
sha=lambda b:hashlib.sha256(b).hexdigest()
res={'expected':{'bytes':152565,'members':22,'sha256':'09fc72f89bb351ad1ad092ce18bd010eec0171e373b0cf84100ca95bd05b78ba'},'actual':{'bytes':p.stat().st_size,'sha256':sha(p.read_bytes())},'payloads':[]}
with zipfile.ZipFile(p) as z:
 ns=z.namelist(); res['actual']['members']=len(ns); assert len(ns)==len(set(ns))
 m=json.loads(z.read('input_manifest.json')); declared={i['member'] for i in m['files']}
 assert set(ns)==declared|{'input_manifest.json'}
 assert z.testzip() is None
 for i in m['files']:
  b=z.read(i['member']); assert len(b)==i['bytes'] and sha(b)==i['sha256']
  res['payloads'].append({'member':i['member'],'bytes':len(b),'sha256':sha(b),'match':True})
 for n in ns:
  assert not Path(n).is_absolute() and '..' not in Path(n).parts
  dest=root/'input'/n; dest.parent.mkdir(parents=True,exist_ok=True); dest.write_bytes(z.read(n))
 res['uncompressed_payload_bytes']=sum(i['bytes'] for i in m['files'])
 res['uncompressed_all_bytes']=sum(i.file_size for i in z.infolist())
assert res['actual']==res['expected']
res['attachments']=[]
for name in ['s_improvement_review_input.zip','review.zh(7).md','primary_disposition.zh.txt']:
 q=Path('/mnt/data')/name; shutil.copy2(q,root/'originals'/name)
 res['attachments'].append({'name':name,'bytes':q.stat().st_size,'sha256':sha(q.read_bytes())})
res['status']='PASS: exact requested ZIP identity and all 21 payload identities'
(root/'checks/input_identity.json').write_text(json.dumps(res,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(res,ensure_ascii=False,indent=2))
