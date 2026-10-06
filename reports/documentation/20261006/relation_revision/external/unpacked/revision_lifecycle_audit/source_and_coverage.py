"""Independent attachment-only source identity and test-coverage inspection.

No project imports, NumPy, Torch, GPU, Linux server, formal inputs or training.
Mutation observations are static coverage findings, not executed mutant tests.
"""
import ast
import hashlib
import importlib.util
import itertools
import json
from pathlib import Path
import re

ROOT=Path('/workspace/scratch/f4d639b3b473/relation_revision_review')
OLD=ROOT/'reports/seller_alias_continual/20261006/relation_result/source'
PRIOR=Path('/workspace/scratch/f4d639b3b473/relation_memory_result_review/reports/seller_alias_continual/20261006/relation_result/source')
CPU=ROOT/'reports/documentation/20261006/relation_revision/cpu'
OUT=Path(__file__).parent
def read(path): return json.loads(Path(path).read_text(encoding='utf-8'))
def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def rec(path,base):return {'path':str(path.relative_to(base)),'bytes':path.stat().st_size,'sha256':digest(path)}

failures=[]
manifest=read(CPU/'manifest.json')
for r in manifest:
    p=ROOT/r['path']
    if not p.is_file() or p.stat().st_size!=r['bytes'] or digest(p)!=r['sha256']:
        failures.append({'kind':'cpu_source_binding','record':r})
recovery=read(CPU/'recovery_manifest.json')
for name,r in recovery.items():
    p=CPU/name
    if not p.is_file() or p.stat().st_size!=r['bytes'] or digest(p)!=r['sha256']:
        failures.append({'kind':'cpu_recovery_binding','path':name})

# Rebuild the deployed Python dependency closure without executing any module.
closure={'scripts/step28_relation_revision.py','scripts/step28_relation_revision_verify.py',
         'tests/test_step28_relation_revision.py'}
todo=list(closure)
while todo:
    name=todo.pop()
    for node in ast.walk(ast.parse((ROOT/name).read_text(encoding='utf-8'))):
        modules=[alias.name for alias in node.names] if isinstance(node,ast.Import) else [node.module] if isinstance(node,ast.ImportFrom) else []
        for module in modules:
            child=('scripts/'+module+'.py' if module and module.startswith('step28_') else
                   'tests/'+module+'.py' if module and module.startswith('test_step28_') else None)
            if child is not None and child not in closure:
                closure.add(child);todo.append(child)
closure.update(['scripts/run_step28_relation_revision_cpu.sh',
                'schema/step28_alias_ranking_policy.json','schema/step28_bge_continual_policy.json',
                'schema/step28_chinese_base_policy.json'])
if closure!={r['path'] for r in manifest}:failures.append({'kind':'ast_closure','actual':sorted(closure)})

gate=read(ROOT/'reports/seller_alias_continual/20261006/relation_result/gate.json')
frozen=[]
for r in gate['source_files']:
    p=OLD/r['path']
    actual=rec(p,OLD)
    row={'record':actual,'matches_original_gate':actual==r,
         'matches_previous_reviewed_attachment':(PRIOR/r['path']).is_file() and p.read_bytes()==(PRIOR/r['path']).read_bytes()}
    if not row['matches_original_gate'] or not row['matches_previous_reviewed_attachment']:
        failures.append({'kind':'old_frozen_changed','path':r['path']})
    frozen.append(row)
root_unchanged=[]
for r in manifest:
    p=OLD/r['path']
    if p.is_file():
        same=(ROOT/r['path']).read_bytes()==p.read_bytes()
        root_unchanged.append({'path':r['path'],'same_as_original_frozen':same,'sha256':digest(p)})
        if not same:failures.append({'kind':'reused_root_changed','path':r['path']})

testpath=ROOT/'tests/test_step28_relation_revision.py'
test_ast=ast.parse(testpath.read_text(encoding='utf-8'))
tests=[n.name for n in ast.walk(test_ast) if isinstance(n,ast.FunctionDef) and n.name.startswith('test_')]
log=(CPU/'evidence/unittest.log').read_text(encoding='utf-8')
logged=re.findall(r'^(test_\w+) \(test_step28_relation_revision\.RevisionTests\)',log,flags=re.M)
if set(tests)!=set(logged) or len(tests)!=5 or 'Ran 5 tests' not in log or not log.rstrip().endswith('OK'):
    failures.append({'kind':'test_log_identity'})

# All current/history helper labels are UID-independent. Provide one minimal
# valid differentiated alternative, without constructing or importing Group.
edges=list(itertools.combinations(range(28),2))
controls=[i for i,n in enumerate([3]*4+[2]*8) for _ in range(n)]
labels=[int(controls[i]==controls[j]) for i,j in edges]
alternative=list(controls);alternative[0],alternative[3]=alternative[3],alternative[0]
permuted=[int(alternative[i]==alternative[j]) for i,j in edges]
degrees=lambda truth:[sum(y for e,y in zip(edges,truth) if q in e) for q in range(28)]
label_counterexample={'same_labels_for_all_handmade_group_UIDs':True,
    'original_positive_edges':sum(labels),'permuted_positive_edges':sum(permuted),
    'changed_labels_after_swapping_membership_positions_0_3':sum(a!=b for a,b in zip(labels,permuted)),
    'original_query_positive_degrees':degrees(labels),'permuted_query_positive_degrees':degrees(permuted),
    'both_have_positive_and_negative_per_query':all(0<d<27 for truth in [labels,permuted] for d in degrees(truth)),
    'purpose':'Shows a 20-positive, 28-account routing-discriminating fixture exists; no model/training was executed.'}

# Explain why a first Adam comparison can hide a gradient-scale error. With
# initial moments zero, bias-corrected Adam update is lr*g/(abs(g)+eps).
lr,eps=0.001,1e-8
step=lambda g:lr*g/(abs(g)+eps)
adam_example={'scope':'Scalar illustrative first Adam moment, no weight decay; not a reproduced model mutant.',
    'learning_rate':lr,'epsilon':eps,'gradient_a':1.,'gradient_b':2.,
    'gradient_difference':1.,'update_difference':abs(step(1.)-step(2.)),
    'test_parameter_atol':2e-6,
    'note':'Global clip and first-step normalization can conceal changed gradient magnitude; compare pre-clip gradients.'}

result={'scope':'Attachment-only stdlib execution; AST/source/receipt checks and scalar examples. No project imports or training.',
    'status':'PASS_SOURCE_BINDINGS_WITH_STATIC_TEST_COVERAGE_LIMITS' if not failures else 'ISSUES',
    'torch_module_available_in_this_environment':importlib.util.find_spec('torch') is not None,
    'torch_imported_or_tests_executed_here':False,
    'cpu_source_records_verified':len(manifest),'cpu_AST_closure_count':len(closure),
    'cpu_recovery_records_verified':len(recovery),'old_frozen_records_verified':len(frozen),
    'old_frozen_comparison':frozen,'current_reused_files_comparison':root_unchanged,
    'new_test_names':tests,'actual_Linux_test_names_from_log':logged,
    'linux_evidence':read(CPU/'evidence/result.json'),
    'label_routing_counterexample':label_counterexample,'first_Adam_discrimination_example':adam_example,
    'static_coverage_limits':[
      {'location':'new tests lines 88-99 and revision lines 32-39',
       'finding':'Joint and serial both consume production history_objective.total. No independent assertion fixes total to Q+0.1R; changing only aggregation or coefficient can be shared by both.',
       'not_claimed':'No Torch mutation run; this is an absent-oracle observation.'},
      {'location':'new tests lines 101-103',
       'finding':'Only post-first-Adam parameters are compared, not pre-clip full gradients or optimizer moment state.'},
      {'location':'old test helper lines 19-21 and new tests lines 80,107,131',
       'finding':'All helper groups share exactly the same labels. Passing current.labels to historical R would not be discriminated by these fixtures.'},
      {'location':'new tests lines 127-141',
       'finding':'Two stages and two consolidations execute; roundtrip is checked, but restored Memory object is discarded and not used for stage2. No new model/Adam/RNG restore test.'},
      {'location':'new tests lines 36-42,105-125',
       'finding':'count1/stage0/empty-reservoir kernel fixture cannot be serialized as a valid stage Memory. Alternative H/b from bounded random_z is a lawful quadratic-statistic fixture, not a demonstrated reachable same-cache accumulated history.'}
    ],'failures':failures}
(OUT/'source_and_coverage.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps({k:v for k,v in result.items() if k not in ['old_frozen_comparison','current_reused_files_comparison']},ensure_ascii=False,indent=2))
raise SystemExit(bool(failures))
