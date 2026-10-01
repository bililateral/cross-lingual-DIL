# Actual commands executed over the existing authenticated SSH session; no credentials.
# CPU, offline, one thread; CHINESE_CPU_OUT and deadline were defined at authorized startup.

/home/yongpeng/miniconda3/envs/py310/bin/python -B - <<'PY'
from pathlib import Path
import hashlib,json,datetime
r=json.loads(Path('reports/seller_alias_continual/20260924/chinese_implementation/collection_inputs.json').read_text())
for item in r['files']:
 p=Path(item['path'])
 assert p.stat().st_size==item['bytes'] and hashlib.sha256(p.read_bytes()).hexdigest()==item['sha256'],item['path']
out=Path('reports/seller_alias_continual/20260924/chinese_implementation/cpu_initial')
(out/'collection_sources.json').write_text(json.dumps({'verified_files':r['files'],'observed_at':datetime.datetime.now().astimezone().isoformat()},indent=2)+'\n')
print('COLLECTION_SOURCES_VERIFIED',len(r['files']))
PY
/usr/bin/time -v -o "$CHINESE_CPU_OUT/collection_resources.log" timeout 120s /home/yongpeng/miniconda3/envs/py310/bin/python -B -m unittest discover -s tests -p test_step28_chinese_base_contracts.py -v > "$CHINESE_CPU_OUT/collection_contracts.log" 2>&1
printf '%s\n' "$?" > "$CHINESE_CPU_OUT/collection_exit.txt"
cat "$CHINESE_CPU_OUT/collection_exit.txt"
tail -n 5 "$CHINESE_CPU_OUT/collection_contracts.log"
/home/yongpeng/miniconda3/envs/py310/bin/python -B - <<'PY'
from pathlib import Path
import hashlib,json,datetime
r=json.loads(Path('reports/seller_alias_continual/20260924/chinese_model/download.json').read_text())
base=Path(r['relative_path']); actual=[]
for e in r['files']:
 p=base/e['path']; h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''): h.update(b)
 item={'path':e['path'],'size_bytes':p.stat().st_size,'sha256':h.hexdigest()}
 assert item==e,(item,e)
 actual.append(item)
assert sorted(p.relative_to(base).as_posix() for p in base.rglob('*') if p.is_file())==sorted(e['path'] for e in actual)
content=hashlib.sha256(json.dumps(actual,sort_keys=True,separators=(',',':')).encode()).hexdigest()
assert content==r['content_sha256'],content
receipt={'status':'LINUX_ACTUAL_FILES_VERIFIED','observed_at':datetime.datetime.now().astimezone().isoformat(),'archive_path':str(base.resolve()),'model_id':r['model_id'],'revision':r['revision'],'files':actual,'file_count':len(actual),'total_size_bytes':sum(e['size_bytes'] for e in actual),'content_sha256':content,'download_manifest_sha256':hashlib.sha256(Path('reports/seller_alias_continual/20260924/chinese_model/download.json').read_bytes()).hexdigest()}
p=Path('reports/seller_alias_continual/20260924/chinese_model/linux_verification.json'); assert not p.exists(); p.write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n')
print('MODEL_ACTUAL_VERIFIED',len(actual),receipt['total_size_bytes'],content)
print('RECEIPT_SHA256',hashlib.sha256(p.read_bytes()).hexdigest())
PY

remaining=$((CHINESE_CPU_DEADLINE - $(date +%s)))
printf 'NATIVE_REMAINING_SECONDS=%s\n' "$remaining"
date --iso-8601=seconds > "$CHINESE_CPU_OUT/native_started.txt"
if (( remaining > 0 )); then
  /usr/bin/time -v -o "$CHINESE_CPU_OUT/native_resources.log" timeout --signal=TERM --kill-after=30s "${remaining}s" bash -euo pipefail -c '
    for arm in mean_bce mean_rank split_bce split_rank; do
      /home/yongpeng/miniconda3/envs/py310/bin/python -u -B scripts/step28_chinese_base_audit.py --arm "$arm" --out "$CHINESE_CPU_OUT/$arm.json"
    done
  ' > "$CHINESE_CPU_OUT/native_runtime.log" 2>&1
  printf '%s\n' "$?" > "$CHINESE_CPU_OUT/native_exit.txt"
else
  printf '124\n' > "$CHINESE_CPU_OUT/native_exit.txt"
fi
date --iso-8601=seconds > "$CHINESE_CPU_OUT/native_finished.txt"
cat "$CHINESE_CPU_OUT/native_exit.txt"
tail -n 15 "$CHINESE_CPU_OUT/native_runtime.log"
/home/yongpeng/miniconda3/envs/py310/bin/python -B - <<'PY'
from pathlib import Path
import hashlib,json,datetime
r=json.loads(Path('reports/seller_alias_continual/20260924/chinese_implementation/final_inputs.json').read_text())
for item in r['files']:
 p=Path(item['path'])
 assert p.stat().st_size==item['bytes'] and hashlib.sha256(p.read_bytes()).hexdigest()==item['sha256'],item['path']
out=Path('reports/seller_alias_continual/20260924/chinese_implementation/cpu_initial')
(out/'final_sources.json').write_text(json.dumps({'verified_files':r['files'],'observed_at':datetime.datetime.now().astimezone().isoformat()},indent=2)+'\n')
print('FINAL_SOURCES_VERIFIED',len(r['files']))
PY
/usr/bin/time -v -o "$CHINESE_CPU_OUT/final_resources.log" timeout 120s /home/yongpeng/miniconda3/envs/py310/bin/python -B -m unittest discover -s tests -p test_step28_chinese_base_contracts.py -v > "$CHINESE_CPU_OUT/final_contracts.log" 2>&1
printf '%s\n' "$?" > "$CHINESE_CPU_OUT/final_exit.txt"
cat "$CHINESE_CPU_OUT/final_exit.txt"
tail -n 5 "$CHINESE_CPU_OUT/final_contracts.log"
/home/yongpeng/miniconda3/envs/py310/bin/python -B - <<'PY'
from pathlib import Path
import ast,hashlib,json,datetime,inspect
import torch
root=Path('reports/seller_alias_continual/20260924/chinese_implementation')
old=root/'native_source/step28_chinese_base.py'; new=Path('scripts/step28_chinese_base.py')
def functions(p):
 return {n.name:ast.dump(n,include_attributes=False) for n in ast.parse(p.read_text()).body if isinstance(n,ast.FunctionDef)}
a,b=functions(old),functions(new)
changed=sorted(k for k in a if a[k]!=b.get(k)); added=sorted(set(b)-set(a))
assert changed==['evaluate','train_arm'] and added==['fixed_classification'],(changed,added)
for key in ['load_model','model_config','record_texts','pool_channels','objectives','logits','update','score','pooling_modes','contract']:
 assert a[key]==b[key]
api_names=['reset_peak_memory_stats','memory_allocated','memory_reserved','max_memory_allocated','max_memory_reserved','synchronize','current_device','get_device_name']
apis={name:{'callable':callable(getattr(torch.cuda,name)),'source_file':inspect.getsourcefile(getattr(torch.cuda,name))} for name in api_names}
assert all(v['callable'] for v in apis.values())
result={'status':'NATIVE_COMPUTATION_UNCHANGED_AFTER_REPORTING_CORRECTIONS','old_sha256':hashlib.sha256(old.read_bytes()).hexdigest(),'final_sha256':hashlib.sha256(new.read_bytes()).hexdigest(),'changed_functions':changed,'added_functions':added,'unchanged_functions':sorted(k for k in a if a[k]==b.get(k)),'gpu_api_definitions':apis,'gpu_execution':False,'torch':torch.__version__,'scope':'AST equality verifies unchanged native computation; actual CUDA resource measurement remains pending formal execution.'}
(root/'cpu_initial/final_semantics.json').write_text(json.dumps(result,indent=2)+'\n')
out=root/'cpu_initial'
assert (out/'final_exit.txt').read_text().strip()=='0'
assert (out/'native_exit.txt').read_text().strip()=='0'
assert not list((out/'work').iterdir())
(out/'finished.txt').write_text(datetime.datetime.now().astimezone().isoformat()+'\n')
files=[{'path':p.relative_to(out).as_posix(),'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in sorted(out.rglob('*')) if p.is_file() and p.name!='evidence_inventory.json']
assert sum(r['bytes'] for r in files)<256*1024*1024
receipt={'status':'CPU_EVIDENCE_COMPLETE','files':files,'file_count':len(files),'total_bytes':sum(r['bytes'] for r in files),'native_updates':4,'formal_label_reads':0,'gpu_used':False,'work_empty':True}
(out/'evidence_inventory.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps({'changed_functions':changed,'added_functions':added,'final_sha256':result['final_sha256'],'evidence_file_count':len(files),'evidence_bytes':receipt['total_bytes'],'inventory_sha256':hashlib.sha256((out/'evidence_inventory.json').read_bytes()).hexdigest()}))
PY
