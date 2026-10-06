from pathlib import Path
import sys,json,hashlib,zipfile,platform,importlib.metadata
ROOT=Path(__file__).resolve().parents[2]
IN=ROOT/'input'; OUT=ROOT/'audit/outputs'; OUT.mkdir(parents=True,exist_ok=True)
def sha(b):return hashlib.sha256(b).hexdigest()
def unpack_check(path,dst,manname='manifest.json'):
 z=zipfile.ZipFile(path); names=[x.filename for x in z.infolist() if not x.is_dir()]
 assert len(names)==len(set(names)); assert z.testzip() is None
 for name in z.namelist(): assert not Path(name).is_absolute() and '..' not in Path(name).parts
 dst.mkdir(parents=True,exist_ok=True); z.extractall(dst)
 manifests=[manname] if manname in names else [n for n in names if Path(n).name=='manifest.json']
 findings=[]
 for mn in manifests:
  m=json.loads(z.read(mn)); entries=m if isinstance(m,list) else m.get('files',m.get('payloads',m.get('entries',[])))
  if isinstance(entries,dict):entries=[dict(path=k,**v) for k,v in entries.items()]
  if not isinstance(entries,list):continue
  used=[]
  for e in entries:
   if not isinstance(e,dict): continue
   p=e.get('path',e.get('name'))
   if p is None:continue
   choices=[p,str(Path(mn).parent/p)]
   member=next((c for c in choices if c in names),None)
   assert member is not None,(mn,p)
   b=z.read(member); assert b==(dst/member).read_bytes()
   expected=e.get('sha256',e.get('sha')); size=e.get('bytes',e.get('size'))
   if expected: assert sha(b)==expected,(mn,p)
   if size is not None: assert len(b)==size,(mn,p)
   used.append(member)
  findings.append({'path':mn,'verified_payloads':len(used),'unique':len(set(used)), 'unlisted':sorted(set(names)-set(used)-{mn})[:20]})
 return {'archive':str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path),'bytes':path.stat().st_size,'sha256':sha(path.read_bytes()),'entries':len(z.infolist()),'files':len(names),'directories':len(z.infolist())-len(names),'manifest_checks':findings}
a=[]
a.append(unpack_check(Path('/mnt/data/review_input.zip'),IN))
assert a[0]['bytes']==11461459 and a[0]['sha256']=='3df2d543301bfc4ad80e1d9bf33a7a3f04123d71b0068188308a4b7a34bfd9d3'
p=IN/'history/relation_revision_pilot_external_evidence.zip'; dst=ROOT/'history/pilot_external'; a.append(unpack_check(p,dst))
p=dst/'input/relation_revision_pilot_review.zip'; dst=ROOT/'history/pilot_input'; a.append(unpack_check(p,dst))
p=dst/'history/relation_revision_review.zip'; dst=ROOT/'history/revision_input'; a.append(unpack_check(p,dst))
env={'python':sys.version,'platform':platform.platform()}
for mod in ['numpy','torch','matplotlib','scipy']:
 try: env[mod]=importlib.metadata.version(mod)
 except importlib.metadata.PackageNotFoundError:env[mod]=None
result={'archives':a,'environment':env,'scope':'Extraction and byte identities only; not runtime or scientific validation.'}
(OUT/'identity.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(result,ensure_ascii=False,indent=2))
