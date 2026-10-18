#!/usr/bin/env python3
"""Fetch and read fixed official C++ sources as text only; never build/import them."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import resource
import sys
import time
import traceback
import urllib.request

BASE=Path('/workspace/scratch/2456d4553fd5')
WORK=BASE/'cublas_audit_work/primary'
OUT=BASE/'cublas_audit_output/evidence/primary'
SOURCES=[
 {'name':'CublasHandlePool_v2.9.1.cpp',
  'url':'https://raw.githubusercontent.com/pytorch/pytorch/v2.9.1/aten/src/ATen/cuda/CublasHandlePool.cpp',
  'patterns':['cublas_handle_stream_to_workspace','cublaslt_handle_stream_to_workspace','clearCublasWorkspaces','getNewWorkspace','getCurrentCUDABlasHandle','getCurrentCUDABlasLtHandle','cublasSetWorkspace','CUDACachingAllocator::get()->allocate']},
 {'name':'Module_v2.9.1.cpp',
  'url':'https://raw.githubusercontent.com/pytorch/pytorch/v2.9.1/torch/csrc/cuda/Module.cpp',
  'patterns':['THCPModule_clearBlasWorkspaces','_cuda_clearCublasWorkspaces','clearCublasWorkspaces']}
]

def sha(data): return hashlib.sha256(data).hexdigest()
def save(path,obj): path.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')

def main():
    started=time.perf_counter(); WORK.mkdir(parents=True,exist_ok=True); OUT.mkdir(parents=True,exist_ok=True)
    pre={'timestamp_utc':datetime.now(timezone.utc).isoformat(),'command_argv':[sys.executable,*sys.argv],
         'source_sha256':sha(Path(__file__).read_bytes()),'source_bytes':Path(__file__).stat().st_size,
         'python':sys.version,'platform':platform.platform(),'cwd':os.getcwd(),'inputs':SOURCES,
         'purpose':'Only fixed-version CUDA workspace ownership and Python binding, directly required by current repair.',
         'scientific_computation':False,'project_code_executed':False,'project_linux_connection':False,
         'network_installation':False}
    save(OUT/'pre_execution.json',pre)
    result={'operation':pre,'sources':[]}; displayed=[]; errors=[]; status=0
    for spec in SOURCES:
        step=time.perf_counter(); row={'url':spec['url'],'name':spec['name']}
        try:
            req=urllib.request.Request(spec['url'],headers={'User-Agent':'Static-source-review'})
            with urllib.request.urlopen(req,timeout=20) as response:
                content=response.read(1024*1024+1); assert len(content)<=1024*1024
                row.update(http_status=response.status,final_url=response.url)
            p=WORK/spec['name']; p.write_bytes(content); lines=content.decode('utf-8').splitlines()
            selected=set()
            for index,line in enumerate(lines):
                if any(pattern in line for pattern in spec['patterns']):
                    selected.update(range(max(0,index-7),min(len(lines),index+15)))
            assert selected
            ranges=[]
            for index in sorted(selected):
                if ranges and index+1==ranges[-1][1]+1: ranges[-1][1]=index+1
                else: ranges.append([index+1,index+1])
            displayed.append('\nSOURCE '+spec['url'])
            for lo,hi in ranges:
                displayed.append(f'RANGE {lo}-{hi}')
                for index in range(lo-1,hi): displayed.append(f'{index+1:5d} {lines[index]}')
            row.update(status='READ_ONLY_SOURCE_FETCHED',bytes=len(content),sha256=sha(content),lines=len(lines),
                       read_ranges=ranges,local_path=str(p),executed=False)
        except BaseException:
            status=1; row['status']='READ_FAILURE'; row['traceback']=traceback.format_exc(); errors.append(row['traceback'])
        row['wall_seconds']=time.perf_counter()-step; result['sources'].append(row)
    result.update(exit_code=status,wall_seconds=time.perf_counter()-started,
                  process_maxrss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
    save(OUT/'result.json',result)
    # Full official sources remain read-only references outside the delivered project evidence.
    # Deliver identities, exact requested/read ranges, source of this mechanical operation, and logs.
    (OUT/'read_log.json').write_text(json.dumps({'source_read_ranges':result['sources'],
        'full_text_reference_policy':'Official URLs and SHA identify temporary text. No full upstream source is repackaged.'},ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    (OUT/'stderr.txt').write_text('\n'.join(errors),encoding='utf-8')
    (WORK/'displayed_source.txt').write_text('\n'.join(displayed)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k!='operation'},ensure_ascii=False,indent=2))
    print('\n'.join(displayed))
    return status

if __name__=='__main__': raise SystemExit(main())
