"""Check retained failures, final totals, unchanged reference versions, and delivery syntax."""
import ast,hashlib,json,re,sys,subprocess
from pathlib import Path
B=Path('/mnt/data/er_low_audit');E=B/'evidence'
results=json.loads((E/'test_results.json').read_text());assert results['latest_distinct_cases']==34 and results['latest_passed']==34
assert results['all_version_execution_totals']==dict(tests=41,passed=38,failures=1,errors=2,skipped=0)
for filename,stem in [('statistics_corrections.json','test_independent_statistics'),('paths_corrections.json','test_independent_paths')]:
 r=json.loads((B/'revisions'/filename).read_text())
 assert hashlib.sha256((B/'independent'/(stem+'.py')).read_bytes()).hexdigest()==r['original_source_sha256']
 assert hashlib.sha256((B/'independent'/(stem+'_v2.py')).read_bytes()).hexdigest()==r['revised_source_sha256']
for path in [B/'reproduce_cpu.py',*sorted((B/'independent').glob('*.py'))]:ast.parse(path.read_text(),filename=str(path))
for path in B.rglob('*.json'):json.loads(path.read_text())
assert '原 ZIP 及全部 294 个解压文件大小/SHA 未变' in (B/'REVIEW.zh.md').read_text()
assert (B/'logs/08_independent_statistics_v1.stderr.log').read_text().count('FAILED (failures=1, errors=1)')==1
assert (B/'logs/10_independent_paths_v1.stderr.log').read_text().count('FAILED (errors=1)')==1
cmd=[sys.executable,str(B/'reproduce_cpu.py'),'--help'];r=subprocess.run(cmd,capture_output=True,text=True)
print('REPRODUCER_HELP_COMMAND',json.dumps(cmd));print('REPRODUCER_HELP_STDOUT\n'+r.stdout);print('REPRODUCER_HELP_STDERR\n'+r.stderr);print('REPRODUCER_HELP_EXIT',r.returncode);assert r.returncode==0
print(json.dumps(dict(status='DELIVERY_INTERNAL_CHECKS_PASS',latest_distinct_tests=34,all_version_executions=41,all_version_failures=1,all_version_errors=2,initial_reference_sources_preserved=True,project_sources_changed=False,scope='No native or formal execution'),indent=2))
