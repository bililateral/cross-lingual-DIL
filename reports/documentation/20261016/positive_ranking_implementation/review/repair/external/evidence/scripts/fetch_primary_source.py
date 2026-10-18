#!/usr/bin/env python3
"""Read one version-pinned upstream source file as text. Never executes that code."""
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
OUT=BASE/'repair_audit_work/upstream'
EVIDENCE=BASE/'repair_audit_output/evidence/upstream'
URL='https://raw.githubusercontent.com/pytorch/pytorch/v2.9.1/aten/src/ATen/cuda/CublasHandlePool.cpp'
OUT.mkdir(parents=True,exist_ok=True);EVIDENCE.mkdir(parents=True,exist_ok=True)
def save(path,value):path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
start=time.perf_counter()
pre={'timestamp_utc':datetime.now(timezone.utc).isoformat(),'command_argv':[sys.executable,*sys.argv],
     'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
     'url':URL,'purpose':'Only direct cuBLAS workspace lifetime semantics required by repaired zero-allocation preflight.',
     'python':sys.version,'platform':platform.platform(),'cwd':os.getcwd(),
     'project_code_executed':False,'scientific_computation':False,'network_installation':False}
save(EVIDENCE/'fetch_pre_execution.json',pre)
result=dict(pre);code=0
try:
    with urllib.request.urlopen(URL,timeout=20) as r:
        b=r.read(256*1024+1)
        if len(b)>256*1024:raise ValueError('Unexpected source size')
        result.update(http_status=r.status,final_url=r.url)
    p=OUT/'CublasHandlePool_v2.9.1.cpp';p.write_bytes(b)
    result.update(status='READ_ONLY_SOURCE_FETCHED',local_path=str(p),bytes=len(b),
                  sha256=hashlib.sha256(b).hexdigest(),lines=len(b.splitlines()))
except BaseException:
    code=1;result.update(status='FETCH_FAILED',traceback=traceback.format_exc())
result.update(exit_code=code,wall_seconds=time.perf_counter()-start,
              process_maxrss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
save(EVIDENCE/'fetch_result.json',result)
print(json.dumps(result,ensure_ascii=False,indent=2))
raise SystemExit(code)
