"""Five bounded handwritten GPU workers; resource shortage launches nothing.

Three serial pooling checks, then two simultaneous worst-pooling workers.
Each invocation resumes only an unstarted phase; a failed worker is never retried.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import psutil
import step28_formal_queue as q


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--permit',type=Path,required=True)
    args=p.parse_args()
    permit=json.loads(args.permit.read_text())
    if (sys.platform!='linux' or permit.get('operation')!='english_abc_handwritten_gpu'
            or permit.get('approved') is not True or permit.get('formal_experiment') is not False
            or permit.get('total_worker_seconds')!=1800):
        raise ValueError('Only explicitly authorized handwritten qualification')
    for row in permit['sources']: q.verify(row)
    if str(Path(__file__).resolve()) not in {r['path'] for r in permit['sources']}:
        raise ValueError('Parent source is not pinned')
    root=Path(permit['root']);root.mkdir(exist_ok=True)
    ledger_path=root/'gpu_ledger.json'
    ledger=json.loads(ledger_path.read_text()) if ledger_path.exists() else dict(status='PREPARED',used=0.,completed=[])
    if ledger['status'] in ('FAILED_NO_RETRY','HANDWRITTEN_GPU_COMPLETE') or ledger.get('active'):
        raise ValueError('Finished/failed/in-flight qualification must not be relaunched')
    monitor=q.GpuMemory()
    active=[]
    try:
        while len(ledger['completed'])<5:
            count=len(ledger['completed'])
            if count<3:
                names=[('mean_last4_sum','mean_last','cls')[count]]
                reserve=permit.get('single_process_gpu_bytes',20*2**30)
            elif count==3:
                worst=max(ledger['completed'],key=lambda r:r['reserve'])
                names=[worst['pooling']]*2;reserve=worst['reserve']
            else:
                raise ValueError('Incomplete concurrent phase is not resumable')
            snap=monitor.snapshot()
            resource=dict(epoch=time.time(),**snap,required_bytes=reserve*len(names),
                          ram_available=psutil.virtual_memory().available)
            q.write(root/'gpu_resource.json',resource)
            if resource['required_bytes']>snap['total']:
                ledger.update(status='TWO_SLOTS_DO_NOT_FIT',reason='Measured reservation exceeds whole GPU')
                q.write(ledger_path,ledger);return
            if snap['free']<resource['required_bytes'] or resource['ram_available']<32*2**30:
                ledger.update(status='WAIT_RESOURCE',reason='Shared resource precondition not met; no new worker')
                q.write(ledger_path,ledger);return
            if ledger['used']+300*len(names)>1800:
                raise TimeoutError('Insufficient remaining qualification allowance')
            env={**os.environ,'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1',
                 'TOKENIZERS_PARALLELISM':'false','CUBLAS_WORKSPACE_CONFIG':':4096:8',
                 'CUDA_VISIBLE_DEVICES':'0','HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1',
                 'PYTHONPATH':str(Path(permit['workspace'])/'scripts')}
            cores=psutil.Process().cpu_affinity()
            if len(cores)<len(names): raise RuntimeError('Distinct CPU cores unavailable')
            for index,name in enumerate(names):
                # Refresh, including the full not-yet-observed commitment of prior starts.
                snap=monitor.snapshot()
                if not q.admission(snap,[r['process'].pid for r in active],reserve)['ready']:
                    raise RuntimeError('Concurrent admission changed after first launch')
                job=root/f'probe{count+index:02d}_{name}'
                log=(root/f'{job.name}.stdout.txt').open('x')
                start=time.monotonic()
                argv=['taskset','-c',str(cores[index]),sys.executable,'-B',str(Path(__file__).with_name('gpu_probe.py')),
                      '--permit',str(args.permit.resolve()),'--job',str(job),'--pooling',name,
                      '--deadline',str(start+300)]
                process=subprocess.Popen(argv,env=env,cwd=permit['workspace'],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                active.append(dict(process=process,job=job,log=log,started=start,pooling=name,nvml_peak=0))
            ledger.update(status='RUNNING',active=[dict(pid=r['process'].pid,job=str(r['job'])) for r in active])
            q.write(ledger_path,ledger)
            while any(r['process'].poll() is None for r in active):
                snap=monitor.snapshot()
                for item in active:
                    process=item['process'];value=snap['processes'].get(process.pid,0)
                    if process.poll() is not None:
                        item.setdefault('ended',time.monotonic())
                    item['nvml_peak']=max(item['nvml_peak'],value)
                    if value>reserve or time.monotonic()-item['started']>=295:
                        raise RuntimeError('Probe memory/deadline limit reached')
                    if process.poll() not in (None,0): raise RuntimeError('Probe failed; no retry')
                time.sleep(.1)
            batch=[]
            for item in active:
                elapsed=item.get('ended',time.monotonic())-item['started']
                item['charged']=True;ledger['used']+=elapsed;item['log'].close()
                if item['process'].returncode!=0: raise RuntimeError('Probe failed; no retry')
                result=json.loads((item['job']/'result.json').read_text())
                peak=max(item['nvml_peak'],result['peak']['reserved'])
                batch.append(dict(pooling=item['pooling'],job=str(item['job']),elapsed=elapsed,
                                  nvml_peak=item['nvml_peak'],reserve=int(peak+max(2**30,.1*peak)),result=result))
            if len(batch)==2:
                intervals=[row['result']['intervals'] for row in batch]
                overlap=any(max(a['begin'],b['begin'])<min(a['end'],b['end']) for a in intervals[0] for b in intervals[1])
                if not overlap: raise RuntimeError('No observed overlap of actual update intervals')
                ledger['concurrent_update_overlap']=True
            ledger['completed'].extend(batch);active=[];ledger['active']=[]
            ledger['status']='PHASE_COMPLETE';q.write(ledger_path,ledger)
        ledger['status']='HANDWRITTEN_GPU_COMPLETE';q.write(ledger_path,ledger)
    except BaseException as error:
        for item in active:
            q.stop(item['process'])
            if not item.get('charged'): ledger['used']+=time.monotonic()-item['started']
            item['log'].close()
        ledger.update(status='FAILED_NO_RETRY',active=[],error=repr(error));q.write(ledger_path,ledger)
        raise
    finally:
        monitor.close()
        print(json.dumps({k:v for k,v in ledger.items() if k!='completed'}),flush=True)


if __name__=='__main__':main()
