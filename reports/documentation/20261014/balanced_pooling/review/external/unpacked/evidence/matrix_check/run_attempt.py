"""Capture the actual isolated reference process; refuse to overwrite an attempt."""
from pathlib import Path
import datetime, hashlib, json, os, platform, subprocess, sys, time
root=Path(__file__).resolve().parent
attempt=root/'attempt01'
if (attempt/'execution.json').exists():raise FileExistsError('Attempt already executed')
env={'python_version':sys.version,'executable':sys.executable,'platform':platform.platform(),
     'machine':platform.machine(),'implementation':platform.python_implementation(),
     'libraries_used':'Python standard library only; math/json/pathlib, no numpy/torch/project imports',
     'cwd':str(root),'OMP_NUM_THREADS':os.environ.get('OMP_NUM_THREADS'),
     'scope':'ChatGPT isolated environment; NOT the user project Linux/py310 or native qualification'}
(attempt/'environment.json').write_text(json.dumps(env,ensure_ascii=False,indent=2)+'\n')
cmd=[sys.executable,'-I',str(root/'check_math.py'),str(root/'inputs.json'),str(attempt/'output.json')]
start=datetime.datetime.now(datetime.timezone.utc).isoformat();t=time.monotonic()
with (attempt/'stdout.txt').open('wb') as stdout,(attempt/'stderr.txt').open('wb') as stderr:
    result=subprocess.run(cmd,cwd=root,stdout=stdout,stderr=stderr,timeout=35)
record={'started_utc':start,'finished_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'elapsed_seconds':time.monotonic()-t,'argv':cmd,'exit_code':result.returncode,
        'source_sha256':hashlib.sha256((root/'check_math.py').read_bytes()).hexdigest(),
        'input_sha256':hashlib.sha256((root/'inputs.json').read_bytes()).hexdigest(),
        'attempt':1,'project_code_executed':False,'formal_data_read':False,'model_loaded':False}
(attempt/'execution.json').write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(record,ensure_ascii=False,indent=2))
sys.exit(result.returncode)
