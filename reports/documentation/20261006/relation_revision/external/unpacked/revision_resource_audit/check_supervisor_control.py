"""Stdlib-only execution of the submitted supervisor's main-function AST.

Popen, psutil and time are controlled substitutes. No actual process, Torch,
project test, GPU, model or formal input is started. Tiny temporary evidence
files exercise the supervisor's real file and exception control flow.
"""
from pathlib import Path
from types import SimpleNamespace
import ast
import json
import tempfile

SOURCE = Path('/workspace/scratch/f4d639b3b473/relation_revision_review/scripts/step28_relation_revision_verify.py')
OUT = Path(__file__).parent
tree = ast.parse(SOURCE.read_text())
main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'main')
code = compile(ast.Module(body=[main], type_ignores=[]), '<submitted supervisor main AST>', 'exec')

def run_case(root, name):
    root.mkdir()
    calls = {'popen':0, 'kill':0, 'wait':0}
    class Child:
        def poll(self):
            return 1 if name == 'test_failure' else None
        def kill(self):
            calls['kill'] += 1
        def wait(self):
            calls['wait'] += 1
            return 1 if name == 'test_failure' else -9
    def popen(*args, **kwargs):
        calls['popen'] += 1
        if name == 'evidence_limit':
            kwargs['stdout'].truncate(16*2**20)
        return Child()
    def memory_info():
        if name == 'monitor_exception':
            raise OSError('controlled monitor read error')
        return SimpleNamespace(rss=3*2**30 if name == 'rss_limit' else 1024)
    process = SimpleNamespace(cpu_affinity=lambda:[0], memory_info=memory_info,
                              children=lambda recursive:[])
    env = dict(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
    namespace = dict(__file__=str(root/'scripts/verifier.py'), Path=Path,
        os=SimpleNamespace(environ=env), psutil=SimpleNamespace(Process=lambda:process,NoSuchProcess=ProcessLookupError),
        subprocess=SimpleNamespace(Popen=popen,STDOUT=-2),
        sys=SimpleNamespace(executable='CONTROLLED_NO_PROCESS',version='CONTROLLED_NOT_PROJECT_PYTHON'),
        time=SimpleNamespace(monotonic=lambda:586. if name == 'time_limit' else 1.,sleep=lambda seconds:None),
        STARTED=0., json=json, print=lambda *a,**k:None)
    exec(code,namespace)
    caught = None
    try:
        namespace['main']()
    except (SystemExit,OSError) as exc:
        caught = {'type':type(exc).__name__,'value':str(exc)}
    path = root/'evidence/result.json'
    record = json.loads(path.read_text()) if path.exists() else None
    return {'case':name,'calls':calls,'caught':caught,
            'record_status':record['status'] if record else None,
            'record_limit_failure':record['limit_failure'] if record else None}

with tempfile.TemporaryDirectory(prefix='revision_supervisor_ast_') as temp:
    cases = [run_case(Path(temp)/name,name) for name in
             ('time_limit','rss_limit','evidence_limit','test_failure','monitor_exception')]
for case in cases[:3]:
    assert case['calls'] == dict(popen=1,kill=1,wait=1)
    assert case['record_status'] == 'FAIL' and case['record_limit_failure']
assert cases[3]['calls'] == dict(popen=1,kill=0,wait=1)
assert cases[3]['record_status'] == 'FAIL'
assert cases[4]['calls'] == dict(popen=1,kill=0,wait=0)
assert cases[4]['record_status'] is None
report = {'scope':'Web stdlib AST control-flow probes with fake process/psutil/time; no Torch, test, Linux project process or GPU execution',
    'cases':cases,
    'interpretation':'Normal resource-limit and failed-child routes stop without retry. Unexpected monitor exceptions currently bypass child cleanup; this was not observed in the submitted all-PASS run and is a limited robustness consideration for future reuse.'}
(OUT/'supervisor_control_checks.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(report,ensure_ascii=False,indent=2))
