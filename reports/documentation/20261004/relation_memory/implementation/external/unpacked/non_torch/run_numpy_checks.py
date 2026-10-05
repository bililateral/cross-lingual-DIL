"""Web CPU audit. Executes real non-torch functions; no substitute torch package."""
from __future__ import annotations
import ast
import hashlib
import importlib.util
import itertools
import json
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import time
import traceback

START = time.monotonic()
ROOT = Path(__file__).resolve().parent
ORIGINAL = Path('/workspace/scratch/f4d639b3b473/relation_memory_impl_review')
SNAPSHOT = ROOT / 'snapshot'
for directory in ('scripts', 'schema', 'tests'):
    shutil.copytree(ORIGINAL / directory, SNAPSHOT / directory, dirs_exist_ok=True)
SOURCES = [p for d in ('scripts', 'schema', 'tests') for p in (ORIGINAL/d).glob('*') if p.is_file()]
hashes_before = {str(p.relative_to(ORIGINAL)): hashlib.sha256(p.read_bytes()).hexdigest() for p in SOURCES}
report = {'environment': {'python_executable': sys.executable, 'python_version': sys.version,
    'runtime_python': os.environ.get('CODEX_PRIMARY_RUNTIME_PYTHON'),
    'runtime_root': os.environ.get('CODEX_PRIMARY_RUNTIME_ROOT'),
    'torch_spec': str(importlib.util.find_spec('torch')),
    'CUDA_VISIBLE_DEVICES': os.environ.get('CUDA_VISIBLE_DEVICES'),
    'thread_env': {k: os.environ.get(k) for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS')}},
    'scope': 'Web Python 3.12 only; no project Linux, CUDA, BGE, dependency installation, or formal data.',
    'actual_non_torch_checks': [], 'independent_numpy_formula_checks': [], 'static_checks': []}
env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1')
original_cmd = [sys.executable, '-m', 'unittest', 'discover', '-s', str(SNAPSHOT/'tests'), '-p', 'test_step28_relation_memory.py', '-v']
attempt = subprocess.run(original_cmd, cwd=SNAPSHOT, env=env, capture_output=True, text=True)
(ROOT/'original_six_attempt.stdout.txt').write_text(attempt.stdout)
(ROOT/'original_six_attempt.stderr.txt').write_text(attempt.stderr)
report['original_six_attempt'] = {'command':original_cmd, 'returncode':attempt.returncode,
    'status':'NOT_EXECUTED_MISSING_TORCH', 'reason':'Test module import fails before any of the six tests execute.'}
assert attempt.returncode != 0 and "No module named 'torch'" in attempt.stderr
sys.path.insert(0, str(SNAPSHOT/'scripts'))
import numpy as np
import step28_relation_memory as method
report['environment']['numpy_version'] = np.__version__
assert 'torch' not in sys.modules

def record(name, detail):
    report['actual_non_torch_checks'].append({'name':name,'result':'PASS','detail':detail})

def reject(name, fn):
    try:
        fn()
    except Exception as exc:
        record(name, {'exception':type(exc).__name__,'message':str(exc)})
    else:
        raise AssertionError(name+' was accepted')

CONTROLS = [i for i, n in enumerate([3]*4+[2]*8) for _ in range(n)]
LABELS = tuple(int(CONTROLS[i]==CONTROLS[j]) for i,j in itertools.combinations(range(28),2))
def handmade(uid):
    return method.data.Group(uid, tuple(f'{uid}_s{i:02}' for i in range(28)),
        tuple(tuple((f'{uid}_i{i:02}_{j}', f'手写标题{i}-{j}', f'手写描述{i}-{j}')
                    for j in range(2)) for i in range(28)), LABELS)
FIRST = [handmade(f'first{i:02}') for i in range(48)]
SECOND = [handmade(f'second{i:02}') for i in range(48)]
for group in FIRST+SECOND: group.validate()
empty = method.Memory('ABC')
empty_payload=empty.to_bytes()
assert method.Memory.from_bytes(empty_payload).to_bytes()==empty_payload
record('empty_memory_exact_restore', {'bytes':len(empty_payload)})
seed = method.data.seed_for(20260930,'ABC','retention')
actual = method.data.Memory(seed)
expected_rng = random.Random(seed)
expected = []
for batch_number, groups in enumerate((FIRST,SECOND),1):
    actual.add_stage(groups)
    for number, group in enumerate(groups, 1+(batch_number-1)*48):
        if number<=6: expected.append(group.uid)
        else:
            index=expected_rng.randrange(number)
            if index<6: expected[index]=group.uid
    assert [g.uid for g in actual.groups]==expected
    assert actual.seen==batch_number*48
    assert actual.rng.getstate()==expected_rng.getstate()
record('algorithm_r_two_stage_independent_stream', {'seen':actual.seen,'members':expected})

# A hand-constructed serialization fixture, NOT a simulated trained first stage.
memory = method.Memory('ABC')
memory.reservoir.add_stage(FIRST)
memory.stage=1; memory.count=48; memory.constant=240.
memory.h=np.eye(32,dtype=np.float64)*48
memory.b=np.zeros(32,dtype=np.float64)
for i,g in enumerate(memory.reservoir.groups):
    z=np.random.default_rng(i+10).uniform(-.8,.8,(32,378)).astype(np.float32); z[-1]=1
    memory.references[g.uid]=z
    memory.reference_keys[g.uid]=memory.group_key(g)
memory.maps={'current':{'a':1.,'b':0.}}
payload=memory.to_bytes()
restored=method.Memory.from_bytes(payload)
assert payload==restored.to_bytes()
for uid,z in memory.references.items():
    assert z.tobytes()==restored.references[uid].tobytes()
    assert z.dtype==restored.references[uid].dtype
record('populated_memory_exact_reference_and_rng_restore', {'bytes':len(payload),'count':restored.count})
memory.begin_stage(2)
for _ in range(7): memory.draw()
restored_mid=method.Memory.from_bytes(memory.to_bytes())
uids=[]
for _ in range(281):
    a,x=memory.draw(); b,y=restored_mid.draw()
    assert a.uid==b.uid and x.tobytes()==y.tobytes(); uids.append(a.uid)
assert memory.to_bytes()==restored_mid.to_bytes()
record('midstage_draw_7_restore_and_remaining_281', {'last_count':memory.draw_count, 'draws':len(uids)})
reject('draw_289_rejected', memory.draw)
reject('trailing_byte_rejected', lambda:method.Memory.from_bytes(payload+b'x'))
reject('truncated_reference_rejected', lambda:method.Memory.from_bytes(payload[:-1]))
bad=method.Memory.from_bytes(payload); bad.reservoir.seen=96
reject('statistic_count_vs_reservoir_seen_rejected', bad.to_bytes)
bad=method.Memory.from_bytes(payload); bad.maps={'oversized':'x'*1048576}
reject('complete_container_metadata_budget_rejected', bad.to_bytes)
bad=method.Memory.from_bytes(payload); bad.begin_stage(2)
bad.reference_keys={uid:'wrong' for uid in bad.reference_keys}
reject('changed_raw_group_correspondence_rejected_before_draw',bad.draw)
z=next(iter(memory.references.values()))
reject('canonical_reference_float64_rejected', lambda:method.canonical_reference(z.astype(np.float64)))
bad_z=z.copy(); bad_z[-1,0]=0
reject('canonical_reference_bias_row_rejected',lambda:method.canonical_reference(bad_z))
bad_z=z.copy(); bad_z[0,0]=np.nan
reject('canonical_reference_nonfinite_rejected',lambda:method.canonical_reference(bad_z))
bad_z=z.copy(); bad_z[0,0]=1.01
reject('canonical_reference_out_of_tanh_range_rejected',lambda:method.canonical_reference(bad_z))
fortran=np.asfortranarray(z)
assert method.canonical_reference(fortran).flags.c_contiguous
assert method.canonical_reference(fortran).tobytes()==z.tobytes()
record('canonical_reference_layout_normalization_preserves_values',{'output_c_contiguous':True})

# This is an independent NumPy check of the equations used in source, not execution of torch functions.
edges=list(itertools.combinations(range(28),2))
rng=np.random.default_rng(29)
x=rng.uniform(-.8,.8,(32,378)); x[-1]=1
y=rng.uniform(-.8,.8,(32,378)); y[-1]=1
w=np.linspace(-.3,.2,32)
target=2*np.array(LABELS)-1
queries=[([j for j,e in enumerate(edges) if q in e and LABELS[j]],
          [j for j,e in enumerate(edges) if q in e and not LABELS[j]]) for q in range(28)]
h=x@x.T/378; b=x@target/378
rank_raw=[]
for pos,neg in queries:
    diffs=np.array([x[:,p]-x[:,n] for p in pos for n in neg]).T
    h+=diffs@diffs.T/(28*diffs.shape[1]); b+=2*diffs.mean(axis=1)/28
    rank_raw.append(np.array([(x[:,p]@w-x[:,n]@w-2)**2 for p in pos for n in neg]))
score=x.T@w
explicit=np.mean((score-target)**2)+np.mean([a.mean() for a in rank_raw])
quadratic=w@h@w-2*b@w+5
wrong_global=np.mean(np.concatenate(rank_raw))
proper=np.mean([a.mean() for a in rank_raw])
assert np.isclose(explicit,quadratic,atol=1e-10,rtol=1e-9)
assert abs(wrong_global-proper)>1e-5
report['independent_numpy_formula_checks'].append({'name':'explicit_query_reduction_vs_quadratic',
    'loss':float(explicit),'quadratic':float(quadratic),'wrong_global_minus_proper':float(wrong_global-proper),
    'ranking_removed_delta':float(proper),'comparison_count':sum(len(a) for a in rank_raw)})
k=x@x.T/378+.001*np.eye(32)
t=np.linalg.solve(k,(y@x.T/378+.001*np.eye(32)).T).T
def old_loss(yv,wv):
    v=np.linalg.solve(k,x@(yv.T@wv)/378+.001*wv)
    return v@h@v-2*b@v+5
v=np.linalg.solve(k,x@(y.T@w)/378+.001*w)
a=2*(h@v-b)
gy=np.outer(w,x.T@np.linalg.solve(k,a))/378
gw=t@a
fd_w=np.array([(old_loss(y,w+np.eye(32)[j]*1e-5)-old_loss(y,w-np.eye(32)[j]*1e-5))/2e-5 for j in range(32)])
fd_y=[]
for row,col in ((0,0),(14,129),(31,377)):
    step=np.zeros_like(y);step[row,col]=1e-5
    fd_y.append((old_loss(y+step,w)-old_loss(y-step,w))/2e-5)
assert np.allclose(fd_w,gw,atol=2e-7,rtol=2e-5)
assert np.allclose(fd_y,[gy[r,c] for r,c in ((0,0),(14,129),(31,377))],atol=2e-7,rtol=2e-5)
report['independent_numpy_formula_checks'].append({'name':'full_Y_w_gradient_formula_sensitivity',
    'max_w_finite_difference_error':float(np.max(abs(fd_w-gw))),
    'Y_selected_finite_difference_error':float(np.max(abs(np.array(fd_y)-[gy[r,c] for r,c in ((0,0),(14,129),(31,377))]))),
    'omitting_epsilon_w_derivative_norm':float(np.linalg.norm(.001*np.linalg.solve(k,a))),
    'detaching_y_would_remove_nonzero_norm':float(np.linalg.norm(gy))})

tree=ast.parse((SNAPSHOT/'scripts/step28_relation_memory.py').read_text())
funcs={node.name:node for node in ast.walk(tree) if isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef))}
kernel=funcs['optimization_step']
attrs=[node.func.attr for node in ast.walk(kernel) if isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute)]
report['static_checks'].append({'name':'actual_kernel_call_sites','clip_grad_norm_call_sites':attrs.count('clip_grad_norm_'),
    'optimizer_step_call_sites':sum(isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='step' and isinstance(n.func.value,ast.Name) and n.func.value.id=='optimizer' for n in ast.walk(kernel)),
    'backward_call_sites':attrs.count('backward'),'note':'AST/static evidence only; not autograd execution.'})
hashes_after={str(p.relative_to(ORIGINAL)):hashlib.sha256(p.read_bytes()).hexdigest() for p in SOURCES}
assert hashes_before==hashes_after
report['original_sources_unchanged']=True
report['original_source_hashes']=hashes_before
report['torch_present_at_exit']='torch' in sys.modules
report['elapsed_seconds']=time.monotonic()-START
report['status']='COMPLETED_NON_TORCH_ONLY'
(ROOT/'numpy_check_results.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps({k:report[k] for k in ('environment','original_six_attempt','actual_non_torch_checks','independent_numpy_formula_checks','static_checks','original_sources_unchanged','torch_present_at_exit','elapsed_seconds','status')},ensure_ascii=False,indent=2))
