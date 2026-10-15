#!/usr/bin/env python3
"""Execute this isolated audit once and preserve real stdout, stderr and exit code."""
from pathlib import Path
import datetime, hashlib, json, platform, subprocess, sys, time
root=Path(__file__).resolve().parent.parent
cmd=[sys.executable,'checks/check_static.py','--input-zip','input/review_input.zip',
     '--toy-input','checks/inputs.json','--output','checks/output.json']
started=datetime.datetime.now(datetime.timezone.utc).isoformat()
t0=time.monotonic()
with (root/'checks/stdout.txt').open('wb') as out, (root/'checks/stderr.txt').open('wb') as err:
    result=subprocess.run(cmd,cwd=root,stdout=out,stderr=err,check=False)
status={'started_at_utc':started,'ended_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'elapsed_seconds':time.monotonic()-t0,'command':cmd,'cwd':str(root),
        'exit_code':result.returncode,'python_version':sys.version,'platform':platform.platform(),
        'qualification':'web-isolated scalar and saved-summary arithmetic; NOT project Linux qualification',
        'source_sha256':hashlib.sha256((root/'checks/check_static.py').read_bytes()).hexdigest(),
        'input_sha256':hashlib.sha256((root/'checks/inputs.json').read_bytes()).hexdigest()}
(root/'checks/status.json').write_text(json.dumps(status,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps(status,ensure_ascii=False,indent=2))
sys.exit(result.returncode)
