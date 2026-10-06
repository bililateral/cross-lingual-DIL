"""Execute two unchanged frozen function bodies with explicit stdlib fault injection.
No import of project runner/dependencies, no Torch shim, training, GPU or formal data.
This probes exception/control semantics only, not the resource monitor measurement.
"""
from pathlib import Path
from types import SimpleNamespace
import ast, hashlib, json, tempfile, threading
source=Path('/workspace/scratch/f4d639b3b473/relation_memory_result_review/reports/seller_alias_continual/20261006/relation_result/source/scripts/step28_relation_memory_run.py')
tree=ast.parse(source.read_text())
names={'complete','watchdog'}
nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names]
assert {n.name for n in nodes}==names
module=ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0),*nodes],type_ignores=[])
ast.fix_missing_locations(module)
def write_json(p,d): p.write_text(json.dumps(d))
def read_json(p): return json.loads(p.read_text())
def record(p,root): return {'path':p.relative_to(root).as_posix(),'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
namespace={'Path':Path,'data':SimpleNamespace(write_json=write_json,read_json=read_json,record=record),
           'threading':threading}
exec(compile(module,str(source), 'exec'),namespace)
class BudgetInjection:
    def __init__(self,fail_check=None):
        self.calls=[]; self.fail_check=fail_check; self.lock=threading.RLock(); self.progress={}; self.persisted=False
    def mark(self,**kwargs): self.progress.update(kwargs)
    def check(self,reserve=0):
        self.calls.append(reserve)
        if len(self.calls)==self.fail_check: raise RuntimeError('Injected exhausted approved resource')
    def snapshot(self): return {'elapsed_seconds':86401 if self.fail_check else 100,'fixture_only':True}
    def persist(self): self.persisted=True
result={'scope':'Web stdlib unchanged complete/watchdog bodies with explicit hand-injected boundaries; not PyTorch/BGE/project Linux or full resource measurement','runner_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'checks':{}}
with tempfile.TemporaryDirectory(prefix='relation_repair_probe_') as t:
    root=Path(t)
    for name,fail_at in [('pre_receipt_resource_rejection',1),('post_receipt_resource_rejection',2),('successful_receipt',None)]:
        job=root/name; (job/'evaluation').mkdir(parents=True)
        write_json(job/'access.json',dict(train=1,valid=1,heldout=0,owners=0))
        def statistical_only(folder):
            value={'status':'STATISTICS_COMPLETE_REQUIRES_VALID_COMPLETION','observed_continuation_checks_pass':True}
            write_json(folder/'evaluation.json',value); return value
        namespace['finalize']=statistical_only
        budget=BudgetInjection(fail_at)
        if fail_at:
            try: namespace['complete'](job,budget)
            except RuntimeError: pass
            else: raise AssertionError('Failure was swallowed')
            assert not (job/'completion.json').exists()
            result['checks'][name]={'statistics_saved':True,'completion_exists':False,'check_reserves':budget.calls}
        else:
            completed=namespace['complete'](job,budget)
            assert completed['status']=='COMPLETE_RELATION_FIXED_POINT' and completed['worth_matched_replay'] is True
            assert budget.persisted and (job/'completion.json').exists()
            result['checks'][name]={'status':completed['status'],'check_reserves':budget.calls,'persisted':budget.persisted}
    # Reproduce P2's exact failed-ledger branch without killing this evidence process.
    class ExitObserved(BaseException): pass
    exit_codes=[]
    def exit_observer(code): exit_codes.append(code); raise ExitObserved()
    def fail_write(*args): raise OSError('Injected ledger disk-write failure')
    namespace['data']=SimpleNamespace(write_json=fail_write)
    namespace['os']=SimpleNamespace(_exit=exit_observer)
    try: namespace['watchdog'](root,BudgetInjection(1),SimpleNamespace(wait=lambda timeout:False))
    except ExitObserved: pass
    assert exit_codes==[2]
    result['checks']['watchdog_ledger_failure_still_calls_exit']={'exit_codes':exit_codes}
    # Evaluate the exact P3 paths expression from the frozen monitor, with no Torch.
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='Budget')
    check=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='check')
    expr=next(n.value for n in ast.walk(check) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='paths' for t in n.targets))
    dotted=root/'job.v1'; dotted.mkdir()
    for suffix in ('.console.txt','.wrapper.txt'): dotted.with_name(dotted.name+suffix).write_bytes(b'0123456789')
    paths=eval(compile(ast.Expression(expr),str(source),'eval'),{'self':SimpleNamespace(root=dotted)})
    size=sum(p.stat().st_size for p in paths if p.is_file())
    assert size==20
    result['checks']['dotted_job_path_expression']={'bytes':size,'file_names':[p.name for p in paths if p.is_file()]}
print(json.dumps(result,ensure_ascii=False,indent=2))
