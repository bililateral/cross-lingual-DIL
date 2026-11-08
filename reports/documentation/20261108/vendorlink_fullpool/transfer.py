"""Fixed-action SSH/SFTP for this approved run only; no arbitrary command/path input."""
import argparse
import getpass
import hashlib
import json
from pathlib import Path
import shlex
import sys

import paramiko

LOCAL = Path(__file__).resolve().parents[4]
REL = Path('reports/documentation/20261108/vendorlink_fullpool')
REMOTE = '/home/yongpeng/cross-lingual/reports/documentation/20261108/vendorlink_fullpool'
PYTHON = '/home/yongpeng/miniconda3/envs/py310/bin/python'
SOURCES = ('step32_vendorlink_fullpool.py', 'step31_evolution_registered_evaluate.py',
           'step31_evolution_registered_policy.py', 'step30_evolution_control.py',
           'step30_evolution_io.py', 'step30_evolution_policy.py', 'step30_evolution_data.py',
           'step30_evolution_evaluate.py', 'step30_evolution_run.py',
           'step29_vendorlink_data.py', 'step29_vendorlink_methods.py',
           'step29_vendorlink_run.py', 'step28_formal_queue.py')
TASKS = [f'ABC_s{s}_{m}' for s in (4, 5, 6) for m in ('H_SEQ', 'VendorLink_SEQ')]
PUBLIC = ['gate.json', 'source_manifest.json', 'observation.json', 'final_observation.json', 'cleanup_linux.json', 'final_sync.json', 'preparation_budget.json', 'launch.json',
          'cpu01/receipt.json', 'cpu01/console.txt', 'cpu02/receipt.json', 'cpu02/console.txt', 'campaign/batch.json', 'campaign/budget.json',
          'campaign/completion.json', 'campaign/dispatch.json', 'campaign/failure.json',
          'campaign/scoring_complete.json', 'campaign/console.txt',
          'campaign/analysis/result.json', 'campaign/analysis/tables.zh.md',
          'campaign/analysis/dispatch_receipt.json', 'campaign/analysis/process.json',
          'campaign/analysis/failure.json', 'campaign/analysis/console.txt']
for task in TASKS:
    PUBLIC += [f'campaign/scoring/{task}/{name}' for name in
               ('blind_complete.json', 'dispatch_receipt.json', 'process.json', 'admission.json',
                'progress.json', 'failure.json', 'console.txt')]

parser = argparse.ArgumentParser()
parser.add_argument('action', choices=('prepare', 'test', 'test_repair', 'start', 'status', 'observe_final', 'collect', 'deliver'))
args = parser.parse_args()
client = paramiko.SSHClient()
client.load_system_host_keys()
client.connect('10.201.109.111', username='yongpeng', password=getpass.getpass('SSH password: '),
               look_for_keys=False, allow_agent=False, timeout=15)


def fixed_remote(code, timeout=60):
    stdin, stdout, stderr = client.exec_command(PYTHON+' -B -c '+shlex.quote(code), timeout=timeout)
    out, err = stdout.read().decode('utf-8'), stderr.read().decode('utf-8')
    status = stdout.channel.recv_exit_status()
    if err:
        print(err, file=sys.stderr)
    if status:
        print(out)
        raise SystemExit(status)
    return out


prefix = "from pathlib import Path\nimport json,sys,time,os,math,subprocess\nroot=Path("+repr(REMOTE)+")\n"
prefix += "sys.path.insert(0,str(root/'workspace/scripts'))\n"
try:
    if args.action == 'prepare':
        fixed_remote(prefix+"(root/'workspace/scripts').mkdir(parents=True,exist_ok=True)\n(root/'workspace/tests').mkdir(exist_ok=True)\n")
        files = [(LOCAL/'scripts'/name, 'workspace/scripts/'+name) for name in SOURCES]
        files += [(LOCAL/'tests/step32_fullpool_increment.py', 'workspace/tests/step32_fullpool_increment.py')]
        files += [(LOCAL/REL/name, name) for name in ('inputs.json', 'scope.zh.md', 'authorization.json')]
        with client.open_sftp() as sftp:
            for local, relative in files:
                sftp.put(str(local), REMOTE+'/'+relative)
        text = fixed_remote(prefix+r'''
import datetime, shutil, psutil
from step30_evolution_io import record,write
start=time.monotonic()
assert not (root/'campaign').exists(), 'Never overwrite a launched campaign'
assert not (root/'preparation_budget.json').exists(), 'Preparation is single-use'
sources=[record(p) for p in sorted((root/'workspace/scripts').glob('*.py'))]
write(root/'source_manifest.json',sources)
pause=Path('/home/yongpeng/cross-lingual/reports/documentation/20261107/h_ablation_prepare/workspace/formal/training/dispatch.json')
d=json.loads(pause.read_text()); processes=[]
for p in psutil.process_iter(['pid','username','create_time','cmdline']):
    try:
        if p.info['username']=='yongpeng' and any(Path(x).name=='step28_h_ablation_pause.py' for x in (p.info['cmdline'] or [])):
            processes.append(dict(**p.info,cwd=p.cwd()))
    except (psutil.NoSuchProcess,psutil.AccessDenied): pass
observation=dict(observed=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
    pause_status=d.get('status'),pause_active=d.get('active'),pause_pending=len(d.get('pending',[])),pause_processes=processes,
    gpu=subprocess.check_output(['nvidia-smi','--query-gpu=name,memory.total,memory.free','--format=csv,noheader']).decode(),
    gpu_processes=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader']).decode(),
    ram_available=psutil.virtual_memory().available,disk_free=shutil.disk_usage(root).free)
write(root/'observation.json',observation)
elapsed=time.monotonic()-start
write(root/'preparation_budget.json',dict(used_seconds=dict(gpu=0,auxiliary=300+math.ceil(elapsed)),cpu_test_seconds=0,
    events=[dict(operation='conservative_local_transport_observation_and_closeout',charged_seconds=300),
            dict(operation='remote_preparation',actual_seconds=elapsed,charged_seconds=math.ceil(elapsed))]))
print(json.dumps(dict(sources=len(sources),observation=observation)))
''')
        print(text)
    elif args.action in ('test', 'test_repair'):
        with client.open_sftp() as sftp:
            if args.action == 'test_repair':
                sftp.put(str(LOCAL/REL/'cpu01/test_source.py'), REMOTE+'/cpu01/test_source.py')
            sftp.put(str(LOCAL/'tests/step32_fullpool_increment.py'), REMOTE+'/workspace/tests/step32_fullpool_increment.py')
        selection = "test_name='cpu02'\nselection=['Increment.test_self_ties_and_wrong_labels']\n" if args.action == 'test_repair' else "test_name='cpu01'\nselection=[]\n"
        print(fixed_remote(prefix+selection+r'''
from step30_evolution_io import read,write,record
import step32_vendorlink_fullpool as f
target=root/test_name; target.mkdir()
prior=read(root/'preparation_budget.json'); start=time.monotonic()
env=dict(os.environ,PYTHONPATH=str(root/'workspace/scripts'),CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
code=-1; error=None
try:
    with (target/'console.txt').open('x') as log:
        run=subprocess.run([sys.executable,'-B','-u',str(root/'workspace/tests/step32_fullpool_increment.py'),*selection],
            cwd=root/'workspace',env=env,stdout=log,stderr=subprocess.STDOUT,timeout=180)
    code=run.returncode
except BaseException as exc: error=repr(exc)
elapsed=time.monotonic()-start; charged=math.ceil(elapsed)
prior['used_seconds']['auxiliary']+=charged; prior['cpu_test_seconds']+=charged
prior['events'].append(dict(operation=test_name,seconds=elapsed,charged_seconds=charged,exit=code,error=error))
write(root/'preparation_budget.json',prior)
receipt=dict(passed=code==0 and charged<=180,exit=code,error=error,seconds=elapsed,
    sources=[record(root/'workspace/scripts'/name) for name in f.SOURCES],
    test_source=record(root/'workspace/tests/step32_fullpool_increment.py'),selection=selection,pretrained=False,gpu=False,formal_text=False)
write(target/'receipt.json',receipt)
print((target/'console.txt').read_text()); print(json.dumps(dict(passed=receipt['passed'],seconds=elapsed,used=prior)))
''',timeout=210))
    elif args.action == 'start':
        print(fixed_remote(prefix+r'''
import psutil
from step30_evolution_io import read,write,record,verify
import step32_vendorlink_fullpool as f
assert not (root/'campaign').exists() and not (root/'launch.json').exists(), 'Single launch only'
first=read(root/'cpu01/receipt.json')
receipt=read(root/'cpu02/receipt.json') if (root/'cpu02/receipt.json').exists() else first
assert receipt['passed']
if not first['passed']:
    assert receipt['sources']==first['sources'], 'Production source changed; old evidence cannot be reused automatically'
    assert receipt['selection']==['Increment.test_self_ties_and_wrong_labels']
    assert record(root/'cpu01/test_source.py')['sha256']==first['test_source']['sha256']
    original=(root/'cpu01/console.txt').read_text()
    assert 'test_actual_six_scores_and_joint_analysis (__main__.Increment) ... ok' in original
    assert 'test_real_two_slot_failure_and_limits (__main__.Increment) ... ok' in original
for source in receipt['sources']: verify(source)
gate=dict(protocol=f.PROTOCOL,approved=True,review_exempt_by_user=True,caps=f.CAPS,limits=f.LIMITS,
    automatic_retry=False,automatic_next_phase=False,automatic_wait=False,other_gpu_commitments=[],
    experiment_root=str(root),workspace=str(root/'workspace'),sources=receipt['sources'],
    authorization=record(root/'authorization.json'),scope=record(root/'scope.zh.md'),inputs=record(root/'inputs.json'),
    preparation_budget=record(root/'preparation_budget.json'))
write(root/'gate.json',gate)
command=[sys.executable,'-B','-u',str(root/'workspace/scripts/step32_vendorlink_fullpool.py'),'--gate',str(root/'gate.json'),'--action','campaign']
env=dict(os.environ,OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',TOKENIZERS_PARALLELISM='false',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',CUBLAS_WORKSPACE_CONFIG=':4096:8')
with (root/'controller_console.txt').open('x') as log:
    process=subprocess.Popen(command,cwd=root/'workspace',env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,umask=0o077)
p=psutil.Process(process.pid)
launch=dict(pid=p.pid,created=p.create_time(),owner=p.username(),cwd=p.cwd(),command=command,started=time.time())
write(root/'launch.json',launch);print(json.dumps(launch))
'''))
    elif args.action == 'status':
        print(fixed_remote(prefix+r'''
import psutil
from step30_evolution_io import read
out={}
for name in ('launch.json','campaign/dispatch.json','campaign/budget.json','campaign/completion.json','campaign/failure.json'):
    if (root/name).exists(): out[name]=read(root/name)
if 'launch.json' in out:
    try:
        p=psutil.Process(out['launch.json']['pid']);out['controller']=dict(pid=p.pid,owner=p.username(),created=p.create_time(),command=p.cmdline(),cwd=p.cwd())
    except (psutil.NoSuchProcess,psutil.ZombieProcess): out['controller']=None
out['progress']={p.parent.name:read(p) for p in (root/'campaign/scoring').glob('*/progress.json')}
if (root/'campaign/failure.json').exists() or out.get('controller') is None:
    out['controller_log']=(root/'controller_console.txt').read_text()[-5000:]
print(json.dumps(out,ensure_ascii=False))
'''))
    elif args.action == 'observe_final':
        print(fixed_remote(prefix+r'''
import datetime,shutil,psutil
from step30_evolution_io import read,write,verify,directory_bytes
complete=read(root/'campaign/completion.json');assert complete['status']=='DESCRIPTIVE_DEV_DIAGNOSTIC_COMPLETE'
launch=read(root/'launch.json');owned=[launch]+[read(p) for p in (root/'campaign').glob('**/process.json')]
alive=[]
for r in owned:
    try:
        p=psutil.Process(r['pid'])
        if p.status()!=psutil.STATUS_ZOMBIE and p.create_time()==r['created'] and p.username()==r['owner']:
            alive.append(dict(pid=p.pid,created=p.create_time(),owner=p.username(),command=p.cmdline(),cwd=p.cwd()))
    except (psutil.NoSuchProcess,psutil.ZombieProcess):pass
assert not alive,'Owned processes still running'
gate=read(root/'gate.json')
for r in gate['sources']:verify(r)
dispatch=read(Path('/home/yongpeng/cross-lingual/reports/documentation/20261107/h_ablation_prepare/workspace/formal/training/dispatch.json'))
pause=[]
for p in psutil.process_iter(['pid','username','create_time','cmdline']):
    try:
        if p.info['username']=='yongpeng' and any(Path(x).name=='step28_h_ablation_pause.py' for x in (p.info['cmdline'] or [])):
            pause.append(dict(**p.info,cwd=p.cwd()))
    except (psutil.NoSuchProcess,psutil.AccessDenied):pass
end=(root/'campaign/completion.json').stat().st_mtime
out=dict(observed=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
    finished=datetime.datetime.fromtimestamp(end,datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
    elapsed_seconds=end-launch['started'],own_processes_alive=alive,production_sources_unchanged=True,
    directory_bytes=directory_bytes(root),disk_free=shutil.disk_usage(root).free,
    gpu_processes=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader']).decode(),
    chinese_pause_status=dispatch.get('status'),chinese_active=dispatch.get('active'),chinese_pending=len(dispatch.get('pending',[])),chinese_processes=pause)
write(root/'final_observation.json',out);print(json.dumps(out,ensure_ascii=False))
'''))
    elif args.action == 'deliver':
        names = ['report.zh.md', 'implementation.zh.md', 'saved_check.json',
                 'deployment_check.json', 'cleanup_request.json']
        delivered = []
        with client.open_sftp() as sftp:
            for name in names:
                local = LOCAL/REL/name
                delivered.append(dict(relative=name, bytes=local.stat().st_size,
                                      sha256=hashlib.sha256(local.read_bytes()).hexdigest()))
                sftp.put(str(local), REMOTE+'/'+name)
        print(fixed_remote(prefix+'delivered='+repr(delivered)+'\n'+r'''
import datetime,psutil,stat
from step30_evolution_io import read,write,record,verify
assert read(root/'campaign/completion.json')['status']=='DESCRIPTIVE_DEV_DIAGNOSTIC_COMPLETE'
assert not (root/'cleanup_linux.json').exists(), 'Cleanup is single-use'
launch=read(root/'launch.json')
for r in [launch]+[read(p) for p in (root/'campaign').glob('**/process.json')]:
    try:
        p=psutil.Process(r['pid'])
        assert p.create_time()!=r['created'] or p.status()==psutil.STATUS_ZOMBIE,'Own process still active'
    except (psutil.NoSuchProcess,psutil.ZombieProcess):pass
gate=read(root/'gate.json')
for r in gate['sources']:verify(r)
for r in delivered:verify(dict(path=str(root/r['relative']),bytes=r['bytes'],sha256=r['sha256']))
request=read(root/'cleanup_request.json')
targets={'workspace/tests/step32_fullpool_increment.py','cpu01/test_source.py'}
assert {r['relative'] for r in request['linux']}==targets and len(request['linux'])==2
assert request['git_original_bytes_verified'] is True
deleted=[]
for r in request['linux']:
    p=root/r['relative']
    assert not p.is_symlink() and p.resolve()==p and p.parent.is_relative_to(root)
    st=p.stat();assert stat.S_ISREG(st.st_mode) and st.st_uid==os.getuid()
    assert str(p) not in {x['path'] for x in gate['sources']}
    actual=record(p)
    assert actual['bytes']==r['bytes'] and actual['sha256']==r['sha256']
    deleted.append(dict(**actual,restore=r['restore']))
for r in deleted:Path(r['path']).unlink()
for r in gate['sources']:verify(r)
out=dict(at=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
    deleted=deleted,count=len(deleted),bytes=sum(x['bytes'] for x in deleted),
    production_sources_unchanged=True,weights_untouched=True,restore_commit=request['commit'])
write(root/'cleanup_linux.json',out)
write(root/'final_sync.json',dict(at=out['at'],files=delivered,all_matched=True,root_docs_synced=False))
print(json.dumps(out))
'''))
    if args.action in ('collect', 'deliver'):
        paths = PUBLIC+['controller_console.txt']
        output = fixed_remote(prefix+'names='+repr(paths)+'\n'+r'''
from step30_evolution_io import record
print(json.dumps([dict(relative=n,**record(root/n)) for n in names if (root/n).is_file()]))
''')
        members = json.loads(output)
        with client.open_sftp() as sftp:
            for member in members:
                relative = member['relative']
                if relative not in paths or member['path'] != REMOTE+'/'+relative:
                    raise ValueError('Only the explicit aggregate/receipt allowlist may be downloaded')
                local = LOCAL/REL/relative
                local.parent.mkdir(parents=True,exist_ok=True)
                sftp.get(member['path'],str(local))
                if local.stat().st_size != member['bytes'] or hashlib.sha256(local.read_bytes()).hexdigest() != member['sha256']:
                    raise ValueError('Transferred bytes differ')
        (LOCAL/REL/'receive.json').write_text(json.dumps(dict(files=members,all_matched=True),indent=2)+'\n',encoding='utf-8')
        print(json.dumps(dict(received=len(members),bytes=sum(x['bytes'] for x in members),all_matched=True)))
finally:
    client.close()
