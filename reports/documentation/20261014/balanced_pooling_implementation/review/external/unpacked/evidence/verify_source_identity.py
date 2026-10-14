from pathlib import Path
import json, hashlib, math, sys, difflib
r=Path(sys.argv[1]); out=Path(sys.argv[2]); b=r/'reports/documentation/20261014/balanced_pooling_implementation'
def norm(rows):
 return sorted([{'path':x['path'],'bytes':x.get('bytes',x.get('size_bytes')),'sha256':x['sha256']} for x in rows],key=lambda x:x['path'])
dep=norm(json.loads((b/'deployment_manifest.json').read_text())); dep1=norm(json.loads((b/'qualification/cpu/attempt01/deployment_manifest.json').read_text()))
a1=json.loads((b/'qualification/cpu/attempt01/result/result.json').read_text()); a2=json.loads((b/'qualification/cpu/attempt02/result/result.json').read_text())
def check(x,p):return p.stat().st_size==x['bytes'] and hashlib.sha256(p.read_bytes()).hexdigest()==x['sha256']
changes=[]
for x,y in zip(dep1,dep,strict=True):
 assert x['path']==y['path']
 if x!=y:
  old=b/'qualification/cpu/attempt01/sources'/x['path'];new=r/y['path']
  changes.append({'path':x['path'],'old':x,'new':y,'archived_old_matches':check(x,old)})
  (out/('diff_'+new.name+'.txt')).write_text(''.join(difflib.unified_diff(old.read_text().splitlines(True),new.read_text().splitlines(True),fromfile='attempt01/'+x['path'],tofile='current/'+y['path'])))
wr=[json.loads((b/f'qualification/cpu/attempt0{i}/wrapper.json').read_text()) for i in (1,2)]
result={'normalization':'deployment uses size_bytes; result uses bytes. Both normalized to bytes, no source edit.',
'final_deployment_count':len(dep),'current_files_all_match':all(check(x,r/x['path']) for x in dep),
'attempt02_matches_final_deployment':norm(a2['source_files'])==dep,'attempt01_matches_old_deployment':norm(a1['source_files'])==dep1,'changes':changes,
'wrapper_wall_seconds':[w['ended_epoch']-w['started_epoch'] for w in wr],
'charged_seconds':sum(math.ceil(w['ended_epoch']-w['started_epoch']) for w in wr),
'maximum_rss_bytes':max(a1['peak_rss_bytes'],a2['peak_rss_bytes']),
'receive_verification_sources_match':norm(json.loads((b/'receive_verification.json').read_text())['source_files'])==dep}
for name in ('receive_manifest.json','receive_attempt02.json'):
 obj=json.loads((b/name).read_text()); result[name+'_structure']={'type':type(obj).__name__,'keys':list(obj) if isinstance(obj,dict) else None}
(out/'source_cpu_verification.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
print(json.dumps(result,ensure_ascii=False,indent=2))
