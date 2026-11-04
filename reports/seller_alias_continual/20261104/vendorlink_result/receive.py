"""File/metadata custody only; run in the authenticated Windows SFTP session."""
import json, pathlib, hashlib, time, math
from datetime import datetime, timezone, timedelta
ROOT = pathlib.Path('reports/seller_alias_continual/20261104/vendorlink_result').resolve()
REMOTE = '/home/yongpeng/cross-lingual/reports/documentation/20261101/formal_chain'
def save(name, value):
    p = ROOT/name
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')

REMOTE_METADATA = r'''
import json,hashlib,time,math,os
from pathlib import Path
from datetime import datetime,timezone,timedelta
import psutil
started=time.monotonic()
b=Path('/home/yongpeng/cross-lingual/reports/documentation/20261101/formal_chain')
c=b/'vendorlink/training'
h=Path('/home/yongpeng/cross-lingual/reports/documentation/20261103/history_wait')
read=lambda p:json.loads(p.read_text())
g=read(b/'vendorlink/formal_gate.json')
done=read(c/'completion.json')
expected={r['id'] for r in g['runs']}|{'scoring','analysis'}
assert done['status']=='ALL_FORMAL_ACTIONS_COMPLETE'
assert len(done['completed'])==38 and {r['id'] for r in done['completed']}==expected
assert all(r['success'] for r in done['completed']) and not done['active'] and not done['retired_slots']
assert done['sources']==g['sources'] and all(v<=done['limits'][k] for k,v in done['used'].items())
assert not list(c.rglob('failure.json'))
train=read(c/'training_complete.json')
assert len(train['completed'])==36 and all(r['success'] for r in train['completed'])
blind=read(c/'scoring/all_blind_complete.json')
assert blind['sources']==g['sources'] and blind['forwards']==4104 and set(blind['runs'])=={r['id'] for r in g['runs']}
assert all(set(v)=={'1_1','2_1','2_2','3_1','3_2','3_3'} for v in blind['runs'].values())
assert [x['role'] for x in read(c/'evaluation_access.json')['events']]==['heldout_blind_text_scoring','heldout_qrels_joint_opening']
files=[]
def record(p,local):
 raw=p.read_bytes()
 r=dict(path=str(p),local=local,bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
 files.append(r)
 return r
for p in sorted(c.rglob('*')):
 if p.is_file() and not p.is_symlink() and p.suffix in ('.json','.jsonl','.txt'):
  assert not any(x in p.parts for x in ('private_custody','memory','owners'))
  record(p,'formal/'+p.relative_to(c).as_posix())
for name in ('scope.zh.txt','review_ready.json','vendorlink/formal_gate.json','state/completion.json','state/process.json'):
 p=b/name
 if p.exists():record(p,'inputs/chain/'+name)
source_checks=[]
for r in g['sources']:
 p=Path(r['path']); actual=record(p,'source/'+p.relative_to(Path(g['workspace'])).as_posix())
 assert actual['bytes']==r['bytes'] and actual['sha256']==r['sha256']
 source_checks.append(r)
record(Path(g['dependency_metadata']['path']),'inputs/dependencies.json')
models=[]
for spec in g['runs']:
 run=read(c/spec['id']/'run/completion.json')
 assert run['identity']==dict(spec=spec,epochs=40,sources=g['sources'],task='observed_account_views_not_verified_aliases')
 assert run['updates']==10280 and run['inference_endpoints'] and set(run['endpoint_files'])=={'1','2','3'}
 assert run['access']['fit_domains']==[dict(A='AlphaBay',B='Dream',C='Agora')[x] for x in spec['order']]
 assert all(run['access'][x]==0 for x in ('dev','heldout','qrels','owners'))
 for rec in run['endpoint_files'].values():
  p=Path(rec['path']); st=p.stat()
  assert st.st_size==rec['bytes'] and p.is_file() and not p.is_symlink()
  models.append(dict(**rec,uid=st.st_uid,nlink=st.st_nlink,inode=st.st_ino,verification='stat_only; saved SHA retained, payload not read'))
 assert not (c/spec['id']/'run/latest.pt').exists()
record(h/'state/start.json','inputs/history/start.json')
record(h/'state/first_update.json','inputs/history/first_update.json')
record(h/'state/process.json','inputs/history/process.json')
record(h/'state/formal_gate.json','inputs/history/formal_gate.json')
hist=[]
for p in psutil.process_iter(['pid','username','cmdline','create_time']):
 try:
  a=p.info; cmd=a['cmdline'] or []
  if a['username']=='yongpeng' and any(str(h) in x for x in cmd):
   hist.append(dict(pid=a['pid'],owner=a['username'],created=a['create_time'],command=cmd,cwd=p.cwd(),status=p.status(),children=[x.pid for x in p.children()]))
 except (psutil.NoSuchProcess,psutil.AccessDenied):pass
histformal=h/'workspace/formal'
history=dict(processes=hist,active_listener_exists=(h/'active_history_wait.py').exists(),formal_exists=histformal.exists(),top_files=[p.name for p in histformal.iterdir()] if histformal.exists() else [])
if (histformal/'budget.json').exists():history['budget']=read(histformal/'budget.json')
summary=dict(server_time=datetime.now(timezone(timedelta(hours=8))).isoformat(),completion_mtime=datetime.fromtimestamp((c/'completion.json').stat().st_mtime,timezone(timedelta(hours=8))).isoformat(),gate_passed=True,run_count=36,endpoint_count=len(models),cells=sum(len(v) for v in blind['runs'].values()),forwards=blind['forwards'],sources_count=len(source_checks),files=files,models=models,history=history,used=done['used'],remote_metadata_seconds=time.monotonic()-started)
print(json.dumps(summary))
'''

def collect(client, sftp, initial):
    started=time.monotonic()
    save('observation.json', initial)
    _,stdout,stderr=client.exec_command('/home/yongpeng/miniconda3/envs/py310/bin/python -B - <<\'PY\'\n'+REMOTE_METADATA+'\nPY')
    raw=stdout.read(); err=stderr.read().decode(); rc=stdout.channel.recv_exit_status()
    if rc: raise RuntimeError((rc,err))
    manifest=json.loads(raw)
    save('receive_manifest.json',manifest)
    received=[]
    for row in manifest['files']:
        dest=ROOT/row['local'];dest.parent.mkdir(parents=True,exist_ok=True)
        assert dest.resolve().is_relative_to(ROOT)
        sftp.get(row['path'],str(dest))
        content=dest.read_bytes()
        assert len(content)==row['bytes'] and hashlib.sha256(content).hexdigest()==row['sha256'],row['local']
        received.append(dict(local=row['local'],bytes=len(content),sha256=row['sha256']))
    receipt=dict(status='ALL_REQUESTED_FILES_SIZE_SHA_MATCH',client_time=datetime.now(timezone(timedelta(hours=8))).isoformat(),count=len(received),bytes=sum(x['bytes'] for x in received),active_seconds=time.monotonic()-started,files=received,exclusions='No text, qrels, owners, model or Memory payloads; history tuning effects unopened')
    save('receive.json',receipt)
    print(json.dumps(dict(count=receipt['count'],bytes=receipt['bytes'],seconds=receipt['active_seconds'],server_time=manifest['server_time'],completion_mtime=manifest['completion_mtime'],history=manifest['history']),ensure_ascii=False))
    return manifest,receipt

