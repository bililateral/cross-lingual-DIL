"""Re-run only the supplied/independent handwritten web CPU audits.
No native(), formal execute(), BGE weights, project data, downloads, or project SSH.
Use a fresh extracted evidence directory so that original outputs remain archived.
"""
from pathlib import Path
import os, sys, json, subprocess, time
R = Path(__file__).resolve().parents[2]
O = R / 'audit/outputs'
O.mkdir(parents=True, exist_ok=True)
env = dict(os.environ, CUDA_VISIBLE_DEVICES='', TOKENIZERS_PARALLELISM='false',
           OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
           PYTHONDONTWRITEBYTECODE='1')
names = ['verify_package.py', 'verify_sources_history.py', 'check_reused_sources.py',
         'run_supplied_cpu_tests.py', 'independent_checks.py', 'check_lifecycle.py',
         'check_evaluation.py']
records=[]
for name in names:
    start=time.monotonic()
    with (O / ('replay_'+name+'.txt')).open('w',encoding='utf-8') as out:
        result=subprocess.run([sys.executable,str(R/'audit/scripts'/name)],cwd=R,
                              env=env,stdout=out,stderr=subprocess.STDOUT,timeout=120)
    records.append({'script':name,'returncode':result.returncode,
                    'wall_seconds':time.monotonic()-start})
    (O/'replay_receipt.json').write_text(json.dumps({'scope':'WEB_CPU_HANDWRITTEN_ONLY',
        'records':records},ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    if result.returncode:
        raise SystemExit(result.returncode)
print(json.dumps({'all_passed':True,'records':records},ensure_ascii=False,indent=2))
