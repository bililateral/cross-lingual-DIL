"""One fixed read-only server observation for the authorized VendorLink diagnostic."""
import getpass
import paramiko
import shlex

client = paramiko.SSHClient()
client.load_system_host_keys()
client.connect('10.201.109.111', username='yongpeng', password=getpass.getpass('SSH password: '),
               look_for_keys=False, allow_agent=False, timeout=15)
code = r'''
import datetime, json, os, shutil, subprocess, time
from pathlib import Path
import psutil
start = time.monotonic()
base = Path('/home/yongpeng/cross-lingual')
old = base/'reports/documentation/20261101/formal_chain/vendorlink/training'
rows = []
for seed in (4, 5, 6):
    for method in ('H_SEQ', 'VendorLink_SEQ'):
        run = old/f'ABC_s{seed}_{method}'/'run'
        complete = json.loads((run/'completion.json').read_text())
        rows.append(dict(run=run.parent.name, status=complete['status'],
            endpoints=[dict(path=str(run/f'stage{s}.pt'), bytes=(run/f'stage{s}.pt').stat().st_size)
                       for s in (1, 2, 3)]))
pause = base/'reports/documentation/20261107/h_ablation_prepare/workspace/formal/training/dispatch.json'
dispatch = json.loads(pause.read_text())
processes = []
for p in psutil.process_iter(['pid', 'username', 'create_time', 'cmdline']):
    try:
        argv = p.info['cmdline'] or []
        if p.info['username'] == 'yongpeng' and any('step28_h_ablation_pause.py' in x for x in argv):
            processes.append(dict(**p.info, cwd=p.cwd()))
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        pass
out = dict(observed=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
    endpoints=rows, pause_status=dispatch.get('status'), pause_active=dispatch.get('active'),
    pause_pending=len(dispatch.get('pending', [])), pause_processes=processes,
    gpu=subprocess.check_output(['nvidia-smi', '--query-gpu=name,memory.total,memory.free', '--format=csv,noheader']).decode(),
    gpu_processes=subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,process_name,used_memory', '--format=csv,noheader']).decode(),
    ram_available=psutil.virtual_memory().available, disk_free=shutil.disk_usage(base).free,
    seconds=time.monotonic()-start)
print(json.dumps(out, ensure_ascii=False))
'''
stdin, stdout, stderr = client.exec_command('/home/yongpeng/miniconda3/envs/py310/bin/python -c '+shlex.quote(code), timeout=60)
print(stdout.read().decode('utf-8'))
print(stderr.read().decode('utf-8'))
print('EXIT', stdout.channel.recv_exit_status())
client.close()
