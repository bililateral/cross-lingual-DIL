/home/yongpeng/miniconda3/envs/py310/bin/python -B - <<'PY'
import datetime, hashlib, json, os, pathlib, shutil, subprocess, time
root = pathlib.Path('/home/yongpeng/cross-lingual')
source_list = json.loads((root / 'reports/seller_alias_continual/20260924/chinese_implementation/final_inputs.json').read_text())
for row in source_list['files']:
    p = root / row['path']
    if p.stat().st_size != row['bytes'] or hashlib.sha256(p.read_bytes()).hexdigest() != row['sha256']:
        raise RuntimeError('Frozen source mismatch: ' + row['path'])
job = root / 'reports/seller_alias_continual/20260924/chinese_execution/20260924_152642/job'
if (job / 'exit_status.txt').read_text().strip() != '0' or not (job / 'run/completion.json').is_file():
    raise RuntimeError('Training incomplete')
phase = root / 'reports/seller_alias_continual/20260924/chinese_evaluation/20260924_202700'
phase.mkdir(parents=True, exist_ok=False)
env = os.environ.copy()
env.update(CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', PYTHONDONTWRITEBYTECODE='1', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', TOKENIZERS_PARALLELISM='false')
argv = ['/home/yongpeng/miniconda3/envs/py310/bin/python', '-B', 'scripts/step28_chinese_base.py', 'evaluate', '--out', str(job / 'run'), '--evaluation', str(phase / 'run')]
started = datetime.datetime.now().astimezone().isoformat()
record = {'status': 'APPROVED_ONCE_ONLY_EVALUATION_STARTED', 'started_at': started, 'argv': argv, 'cpu_threads': 1, 'visible_gpu': '', 'frozen_sources_verified': source_list['files'], 'free_disk_bytes': shutil.disk_usage(root).free, 'meminfo': [x for x in pathlib.Path('/proc/meminfo').read_text().splitlines() if x.startswith(('MemAvailable:', 'MemFree:'))], 'authorization': 'reports/seller_alias_continual/20260924/chinese_execution/authorization.json'}
(phase / 'execution.json').write_text(json.dumps(record, ensure_ascii=False, indent=2) + '\n')
print(json.dumps({'started_at': started, 'phase': str(phase.relative_to(root)), 'sources_verified': len(source_list['files'])}), flush=True)
t0 = time.monotonic()
with (phase / 'evaluation.log').open('xb') as log:
    completed = subprocess.run(['/usr/bin/time', '-v', '-o', str(phase / 'resource_usage.log'), *argv], cwd=root, env=env, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
record.update(status='EVALUATION_PROCESS_FINISHED', exit_status=completed.returncode, finished_at=datetime.datetime.now().astimezone().isoformat(), elapsed_seconds=time.monotonic()-t0)
(phase / 'execution.json').write_text(json.dumps(record, ensure_ascii=False, indent=2) + '\n')
print(json.dumps({k: record[k] for k in ('status', 'exit_status', 'finished_at', 'elapsed_seconds')}), flush=True)
print((phase / 'evaluation.log').read_text()[-2500:])
PY

CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
/home/yongpeng/miniconda3/envs/py310/bin/python -B \
scripts/step28_chinese_result.py \
--run reports/seller_alias_continual/20260924/chinese_execution/20260924_152642/job/run \
--evaluation reports/seller_alias_continual/20260924/chinese_evaluation/20260924_202700/run \
--out reports/seller_alias_continual/20260924/chinese_result \
--historical reports/seller_alias_continual/20260924/base_evaluation/20260924_113500/evaluation \
> reports/seller_alias_continual/20260924/chinese_result/analysis.log 2>&1

