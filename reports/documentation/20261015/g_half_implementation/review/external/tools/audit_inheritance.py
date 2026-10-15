"""Mechanical source/evidence comparison, no imports of project code."""
from pathlib import Path
import ast,hashlib,json,difflib
R=Path('/mnt/data/g_half_input');O=Path('/mnt/data/g_half_review')
Q=R/'reports/documentation/20261015/g_half_implementation/qualification'
def load(p):return json.loads(p.read_text())
def record(path):
 b=(R/path).read_bytes();return {'path':path,'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()}
a,b=[load(Q/f'cpu/attempt{x}/source_files.json') for x in ('01','02')]
d={x['path']:x for x in a};changed=[x['path'] for x in b if d.get(x['path'])!=x]
current=[record(x['path']) for x in b]
receipt=load(Q/'cpu_receipt.json')
closure={f'{s}/{n}' for s,n in [('scripts','step28_g_half_run.py'),('scripts','step28_g_half_verify.py'),('tests','test_step28_g_half.py'),('tests','test_step28_record_replay.py')]};todo=list(closure)
while todo:
 path=todo.pop()
 for node in ast.walk(ast.parse((R/path).read_text())):
  ns=[n.name for n in node.names] if isinstance(node,ast.Import) else [node.module] if isinstance(node,ast.ImportFrom) else []
  for name in ns:
   if name and name.startswith('step28_'):
    p='scripts/'+name+'.py'
    if p not in closure:closure.add(p);todo.append(p)
closure.update('schema/'+n for n in ('step28_g_half_policy.json','step28_record_replay_policy.json','step28_bge_continual_policy.json','step28_alias_ranking_policy.json','step28_chinese_base_policy.json'))
closure.update(('docs/SELLER_ALIAS_G_HALF.zh.md','scripts/run_step28_g_half_linux_20261015.sh','reports/documentation/20261015/g_refinement/baseline_sources.json','reports/documentation/20261015/g_half_implementation/authorization.json'))
archive01_checks=[];diffs=[]
for path in changed:
 old=(Q/'cpu/attempt01/sources'/path).read_bytes()
 archive01_checks.append({'path':path,'matches_attempt01':len(old)==d[path]['bytes'] and hashlib.sha256(old).hexdigest()==d[path]['sha256']})
 diffs.extend(difflib.unified_diff(old.decode().splitlines(True),(R/path).read_text().splitlines(True),fromfile='attempt01/'+path,tofile='current/'+path))
(O/'evidence/actual_attempt_delta.diff').write_text(''.join(diffs))
a_tree=ast.parse((Q/'cpu/attempt01/sources/tests/test_step28_g_half.py').read_text());b_tree=ast.parse((R/'tests/test_step28_g_half.py').read_text())
def funcs(t):return {n.name:ast.get_source_segment('',n) if False else ast.dump(n,include_attributes=False) for n in ast.walk(t) if isinstance(n,ast.FunctionDef)}
fa,fb=funcs(a_tree),funcs(b_tree)
functions={n:fa[n]==fb[n] for n in receipt['attempts'][0]['passed']}
raw=[]
for i in ('01','02'):
 result=load(Q/f'cpu/attempt{i}/result/result.json');wall=load(Q/f'cpu/attempt{i}/wallclock.json')
 raw.append({'attempt':i,'result':{k:v for k,v in result.items() if k!='source_files'},'source_list_matches':result['source_files']==(a if i=='01' else b),'wallclock':wall})
base=load(R/'reports/documentation/20261015/g_refinement/baseline_sources.json');ident=load(Q/'baseline_identity.json')
print('BASELINE fields',list(base));print('IDENT sample',ident['files'][0]);print('BASE sample',base['files'][0])
# Preserve the recorded comparison without pretending the 43 absent payloads were opened here.
base_rows={x['path']:x for x in base['files']}
basechecks=[]
for row in ident['files']:
 expected=base_rows.get(row['path']);basechecks.append({'path':row['path'],'same_expected_bytes_sha256':bool(expected and all(row.get(k)==expected.get(k) for k in ('bytes','sha256'))),'row':row})
inherit=[]
for path in ['reports/documentation/20261014/balanced_pooling_implementation/qualification/cpu/attempt04/result/result.json','reports/documentation/20261014/balanced_pooling_implementation/qualification/gpu/attempt01/result/result.json']:
 obj=load(R/path);rows=obj.get('source_files',[])
 if not rows:print('OLD_EVIDENCE_KEYS',path,list(obj))
 comp=[]
 for row in rows:
  if (R/row['path']).is_file():comp.append({'path':row['path'],'same':record(row['path'])==row})
 inherit.append({'evidence':path,'comparisons':comp})
r={'source_count':len(current),'computed_source_closure_matches':closure==set(x['path'] for x in current),'attempt02_matches_current':current==b,'receipt_matches_current':receipt['source_files']==current,'changed_files':changed,'attempt01_snapshots_verified':archive01_checks,'four_test_functions_ast_unchanged':functions,'attempts':raw,'baseline_count':len(base['files']),'baseline_total_bytes':sum(x['bytes'] for x in base['files']),'baseline_expected_vs_receipt':basechecks,'baseline_payloads_present':sum((R/x['path']).exists() for x in base['files']),'old_qualification_source_comparisons':inherit}
(O/'evidence/source_inheritance.json').write_text(json.dumps(r,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({k:v for k,v in r.items() if k not in ('baseline_expected_vs_receipt','old_qualification_source_comparisons')},ensure_ascii=False,indent=2))
print('Old differing present source files:',[(x['evidence'],[v['path'] for v in x['comparisons'] if not v['same']]) for x in inherit])
