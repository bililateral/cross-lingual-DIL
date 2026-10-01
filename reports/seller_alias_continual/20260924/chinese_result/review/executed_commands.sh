CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /home/yongpeng/miniconda3/envs/py310/bin/python -B - <<'PY'
from pathlib import Path
import hashlib, json, zipfile, subprocess, time, sys
base=Path('reports/seller_alias_continual/20260924/chinese_result/review').resolve()
pkg=base/'package.zip'
assert pkg.stat().st_size==3138095 and hashlib.sha256(pkg.read_bytes()).hexdigest()=='8be3427e2e1574dff3d3317aa6e7680fee32377e1373cd6707196b63948e2796'
script=base/'external/independent_result_audit.py'
assert hashlib.sha256(script.read_bytes()).hexdigest()=='6e81285b667e0aa55dd0d44b1301f55218dd39b2d0ed02c9577b039cf88c4072'
source=base/'temporary_input'
source.mkdir(exist_ok=False)
with zipfile.ZipFile(pkg) as z:
    assert len(z.infolist())==128 and len(set(z.namelist()))==128 and z.testzip() is None
    for n in z.namelist():
        assert (source/n).resolve().is_relative_to(source)
    z.extractall(source)
cmd=[sys.executable,'-B',str(script),str(source),str(base/'primary_runtime.json')]
start=time.monotonic()
with (base/'primary_runtime.log').open('wb') as log:
    proc=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,timeout=120)
record={'command':cmd,'exit_status':proc.returncode,'elapsed_seconds':time.monotonic()-start,'python':sys.version,'threads':1,'gpu_used':False,'new_formal_label_reads':0,'model_loads':0,'script_sha256':hashlib.sha256(script.read_bytes()).hexdigest(),'package_sha256':hashlib.sha256(pkg.read_bytes()).hexdigest()}
(base/'primary_execution.json').write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps(record,ensure_ascii=False))
print((base/'primary_runtime.log').read_text())
PY
