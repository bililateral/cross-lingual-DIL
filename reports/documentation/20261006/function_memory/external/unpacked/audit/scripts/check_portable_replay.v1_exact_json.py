"""Compare numerical audit outputs from a fresh-directory web replay."""
from pathlib import Path
import json,sys,shutil
R=Path(__file__).resolve().parents[2]
P=Path(sys.argv[1])
files=['independent_checks.json','lifecycle_checks.json','evaluation_checks.json','reused_source_identity.json']
checks=[]
for name in files:
 a=json.loads((R/'audit/outputs'/name).read_text());b=json.loads((P/'audit/outputs'/name).read_text())
 equal=a==b;checks.append({'output':name,'exact_json_equal':equal});assert equal,name
src=json.loads((P/'audit/outputs/replay_receipt.json').read_text())
assert len(src['records'])==7 and all(r['returncode']==0 for r in src['records'])
d=R/'audit/outputs/portable_replay';d.mkdir(exist_ok=True)
for p in (P/'audit/outputs').glob('replay_*'):
 if p.is_file():shutil.copyfile(p,d/p.name)
for name in ['web_cpu_result.json','web_cpu_unittest.txt']:
 shutil.copyfile(P/'audit/outputs'/name,d/name)
shutil.copyfile(P/'replay.stdout.txt',d/'all_stdout.txt')
out={'scope':'FRESH_DIRECTORY_WEB_CPU_REPLAY_ONLY','all_seven_scripts_passed':True,
     'numeric_output_comparisons':checks,'project_runs':0,'gpu_runs':0}
(R/'audit/outputs/portable_replay_check.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(out,indent=2))
