"""Compare numerical audit outputs from a fresh-directory web replay."""
from pathlib import Path
import json,sys,shutil
R=Path(__file__).resolve().parents[2]
P=Path(sys.argv[1])
files=['independent_checks.json','lifecycle_checks.json','evaluation_checks.json','reused_source_identity.json']
checks=[]
def compare(a,b,path=''):
 if isinstance(a,dict):
  assert a.keys()==b.keys(),path
  return max([compare(a[k],b[k],path+'/'+k) for k in a]+[0.0])
 if isinstance(a,list):
  assert len(a)==len(b),path
  return max([compare(x,y,path+'/'+str(i)) for i,(x,y) in enumerate(zip(a,b))]+[0.0])
 if isinstance(a,float):
  diff=abs(a-b)
  assert diff <= 1e-11, (path,a,b)
  return diff
 assert a==b,(path,a,b)
 return 0.0
for name in files:
 a=json.loads((R/'audit/outputs'/name).read_text());b=json.loads((P/'audit/outputs'/name).read_text())
 error=compare(a,b);checks.append({'output':name,'exact_json_equal':a==b,'max_float_absolute_difference':error,'within_1e_11':True})
src=json.loads((P/'audit/outputs/replay_receipt.json').read_text())
assert len(src['records'])==7 and all(r['returncode']==0 for r in src['records'])
d=R/'audit/outputs/portable_replay';d.mkdir(exist_ok=True)
for p in (P/'audit/outputs').glob('replay_*'):
 if p.is_file():shutil.copyfile(p,d/p.name)
for name in ['web_cpu_result.json','web_cpu_unittest.txt','independent_checks.json','lifecycle_checks.json','evaluation_checks.json','reused_source_identity.json']:
 shutil.copyfile(P/'audit/outputs'/name,d/name)
shutil.copyfile(P/'replay.stdout.txt',d/'all_stdout.txt')
out={'scope':'FRESH_DIRECTORY_WEB_CPU_REPLAY_ONLY','all_seven_scripts_passed':True,
     'numeric_output_comparisons':checks,'project_runs':0,'gpu_runs':0}
(R/'audit/outputs/portable_replay_check.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(out,indent=2))
