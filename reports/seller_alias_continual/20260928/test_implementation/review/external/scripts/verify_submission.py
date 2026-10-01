"""No model/data arrays loaded: verify this upload, source binding and CPU records."""
from pathlib import Path
import hashlib,zipfile,json,sys,platform,os,importlib.metadata,difflib
B=Path('/mnt/data/test_pretest_audit_20260928'); R=B/'source'; E=B/'evidence/outputs'
sha=lambda x:hashlib.sha256(x).hexdigest()
Z=Path('/mnt/data/test_review.zip'); z=zipfile.ZipFile(Z)
assert z.testzip() is None
inventory=json.loads(z.read('source_inventory.json')); sources=inventory['files']
assert len(sources)==inventory['source_count']==116
assert sum(x['bytes'] for x in sources)==inventory['total_bytes']
assert set(z.namelist())=={r['path'] for r in sources}|{'source_inventory.json'}
for row in sources:
 payload=(R/row['path']).read_bytes()
 assert len(payload)==row['bytes'] and sha(payload)==row['sha256']
 assert payload==z.read(row['path'])
p=json.loads((R/'schema/step28_alias_test_policy.json').read_text())
for rec in p['sources']['inherited_scientific_sources']:
 b=(R/rec['path']).read_bytes();assert len(b)==rec['bytes'] and sha(b)==rec['sha256']
sys.path.insert(0,str(R/'scripts'))
import step28_alias_test_run as runner
info,partition=runner.historical_models(p)
current=runner.sources();assert len(current)==34
cpu_root=R/'reports/seller_alias_continual/20260928/test_implementation/cpu'
cpus={}
for d in sorted(cpu_root.iterdir()):
 a=json.loads((d/'evidence/audit.json').read_text()); h=json.loads((d/'evidence/handmade.json').read_text())
 cpus[d.name]={'audit':a,'handmade':h,'exit':(d/'exit_status.txt').read_text().strip(),'started':(d/'started.txt').read_text().strip(),'finished':(d/'finished.txt').read_text().strip(),'resource_usage':(d/'resource_usage.log').read_text(),'stderr':(d/'stderr.log').read_text()}
 assert cpus[d.name]['exit']=='0' and a['contracts']=={'run':15,'passed':15,'errors':0,'failures':0,'skipped':0}
assert cpus['20260928_155900']['audit']['source_files']==current
old=R/'reports/seller_alias_continual/20260928/test_implementation/coverage/before/test_step28_alias_test_contracts.py'
new=R/'tests/test_step28_alias_test_contracts.py'
diff=''.join(difflib.unified_diff(old.read_text().splitlines(True),new.read_text().splitlines(True),fromfile='before.py',tofile='current.py'))
(E/'project_test_coverage_change.diff').write_text(diff)
old_snapshot=cpus['20260928_155700']['audit']['source_files']; olddict={r['path']:r for r in old_snapshot}
changed=[r['path'] for r in current if olddict[r['path']]!=r];assert changed==['tests/test_step28_alias_test_contracts.py']
packages={}
for name in ['numpy','scipy','torch','sentence-transformers','transformers']:
 try: packages[name]=importlib.metadata.version(name)
 except importlib.metadata.PackageNotFoundError: packages[name]=None
res={'upload':{'path':str(Z),'bytes':Z.stat().st_size,'sha256':sha(Z.read_bytes()),'members':len(z.infolist()),'source_count':len(sources),'source_bytes':inventory['total_bytes'],'all_match':True},'frozen_source_count':len(current),'inherited_sources':len(p['sources']['inherited_scientific_sources']),'sources':current,'history_metadata_match':True,'models':{k:{'run_id':v['point']['run_id'],'epoch':v['point']['epoch'],'map':{'a':v['map']['a'],'b':v['map']['b']},'threshold':v['threshold'],'payload_state_sha256':p['models'][k]['payload']['payload_state_sha256'],'model_parameters_sha256':p['models'][k]['payload']['model_parameters_sha256']} for k,v in info.items()},'project_cpu':cpus,'coverage_changed_files':changed,'environment':{'python':sys.version,'platform':platform.platform(),'packages':packages,'cpu_affinity':sorted(os.sched_getaffinity(0)),'threads':next((l.strip() for l in Path('/proc/self/status').read_text().splitlines() if l.startswith('Threads:')),None)},'formal_inputs_opened':0,'model_loads':0,'new_training':0}
(E/'submission_verification.json').write_text(json.dumps(res,ensure_ascii=False,indent=2))
print(json.dumps({k:res[k] for k in ['upload','frozen_source_count','inherited_sources','history_metadata_match','models','coverage_changed_files','environment']},ensure_ascii=False,indent=2)); print(diff)
