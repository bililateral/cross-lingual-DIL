from pathlib import Path,PurePosixPath
import zipfile,json,hashlib
R=Path(__file__).resolve().parents[2]; I=R/'input'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def validate(root,manifest_name):
 obj=json.loads((root/manifest_name).read_text())
 print(manifest_name,type(obj).__name__, list(obj)[:3] if isinstance(obj,dict) else len(obj))
 entries=obj if isinstance(obj,list) else obj.get('files',obj.get('payloads',obj.get('entries')))
 if isinstance(entries,dict):entries=[dict(path=k,**v) for k,v in entries.items()]
 if entries is None:raise ValueError('unrecognized manifest')
 checks=[]
 for x in entries:
  p=root/x['path'];assert p.is_file(),str(p)
  s=x.get('sha256');n=x.get('bytes',x.get('size_bytes',x.get('size')))
  assert s is None or sha(p)==s,str(p)
  assert n is None or p.stat().st_size==n,str(p)
  checks.append(x['path'])
 return {'count':len(checks),'checked_paths':checks}
def extract(p,d):
 with zipfile.ZipFile(p) as z:
  assert z.testzip() is None
  for n in z.namelist():
   assert not PurePosixPath(n).is_absolute() and '..' not in PurePosixPath(n).parts
  z.extractall(d)
 return {'bytes':p.stat().st_size,'sha256':sha(p),'files':sum(f.is_file() for f in d.rglob('*'))}
out={};out['payloads']=validate(I,'manifest.json')
inv=json.loads((I/'source_inventory.json').read_text());print('inventory',type(inv),str(inv)[:200])
entries=inv if isinstance(inv,list) else inv.get('files',inv.get('sources'))
actual=json.loads((I/'cpu_20261006_220500/result/result.json').read_text())
for r in actual['source_files']:
 p=I/r['path'];assert p.stat().st_size==r['bytes'] and sha(p)==r['sha256']
out['cpu_sources']={'count':len(actual['source_files']),'records':actual['source_files']}
if entries is not None: assert entries==actual['source_files'] or sorted(entries,key=lambda v:v['path'])==sorted(actual['source_files'],key=lambda v:v['path'])
H=R/'history/discussion';out['discussion']=extract(I/'history/discussion_audit.zip',H)
print('history listing:');print('\n'.join(str(p.relative_to(H)) for p in H.rglob('*') if p.is_file()))
for p in H.glob('*manifest*'):
 out['discussion_manifest']=validate(H,p.name)
orig=list(H.rglob('original_input.zip'))
print('original',orig)
if orig:
 out['previous_input']=extract(orig[0],R/'history/previous_input')
 print('previous input:', '\n'.join(str(p.relative_to(R/'history/previous_input')) for p in (R/'history/previous_input').rglob('*') if p.is_file()))
(R/'audit/outputs/source_history_identity.json').write_text(json.dumps(out,ensure_ascii=False,indent=2))
