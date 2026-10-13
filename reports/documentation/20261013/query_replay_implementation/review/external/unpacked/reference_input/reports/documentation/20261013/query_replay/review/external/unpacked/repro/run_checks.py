#!/usr/bin/env python3
"""Capture the one mathematical-check execution; does not run project code."""
import datetime, hashlib, json, os, platform, subprocess, sys, time
from pathlib import Path
import numpy as np
root=Path(__file__).resolve().parent
stamp=lambda: datetime.datetime.now(datetime.timezone.utc).isoformat()
env={'scope':'ChatGPT网页工作区；不是项目Linux py310资格', 'python_version':sys.version,
     'python_executable':sys.executable,'platform':platform.platform(),'machine':platform.machine(),
     'numpy_version':np.__version__,'working_directory':str(root), 'gpu_used':False,
     'torch_imported':False,'project_code_imported':False,'network_used_by_math_script':False,
     'input_files':['inputs.json'], 'numerical_dtype':'numpy.float64; exact fractions.Fraction for enumeration'}
(root/'environment.json').write_text(json.dumps(env,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
started=stamp(); begin=time.perf_counter()
proc=subprocess.run([sys.executable,str(root/'check_math.py')],cwd=root,capture_output=True,text=True,
                    encoding='utf-8',timeout=40,env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1'})
(root/'stdout.txt').write_text(proc.stdout,encoding='utf-8')
(root/'stderr.txt').write_text(proc.stderr,encoding='utf-8')
meta={'command':[sys.executable,'check_math.py'],'started_utc':started,'ended_utc':stamp(),
      'elapsed_seconds':time.perf_counter()-begin,'exit_code':proc.returncode,
      'attempt':1,'previous_failed_attempts':0,
      'source_sha256':hashlib.sha256((root/'check_math.py').read_bytes()).hexdigest(),
      'inputs_sha256':hashlib.sha256((root/'inputs.json').read_bytes()).hexdigest(),
      'stdout_bytes':len(proc.stdout.encode()),'stderr_bytes':len(proc.stderr.encode())}
(root/'run.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps(meta,ensure_ascii=False,indent=2)); print(proc.stdout); print(proc.stderr)
raise SystemExit(proc.returncode)
