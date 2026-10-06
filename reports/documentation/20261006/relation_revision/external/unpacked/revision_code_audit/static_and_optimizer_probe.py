"""Web stdlib AST/number check. No Torch import, tensor emulation or training."""
from pathlib import Path
import ast,json,math,hashlib
root=Path('/workspace/scratch/f4d639b3b473/relation_revision_review')
code=root/'scripts/step28_relation_revision.py'; tests=root/'tests/test_step28_relation_revision.py'
module=ast.parse(code.read_text()); tm=ast.parse(tests.read_text())
functions={n.name:n for n in module.body if isinstance(n,ast.FunctionDef)}
def calls(node):
 out=[]
 for n in ast.walk(node):
  if isinstance(n,ast.Call):
   if isinstance(n.func,ast.Name): out.append((n.lineno,n.func.id))
   elif isinstance(n.func,ast.Attribute): out.append((n.lineno,n.func.attr))
 return sorted(out)
serial_calls=calls(functions['optimization_step'])
assert sum(n=='backward' for _,n in serial_calls)==2
assert sum(n=='clip_grad_norm_' for _,n in serial_calls)==1
assert sum(n=='step' for _,n in serial_calls)==1
assert sum(n=='zero_grad' for _,n in serial_calls)==1
history_return=next(n for n in ast.walk(functions['history_objective']) if isinstance(n,ast.Return))
coefficient=next(n.value.value for n in module.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='HISTORY_RANK_WEIGHT' for t in n.targets))
assert coefficient==.1
# Real-valued first Adam step after bias correction. This is arithmetic, not a
# replay of the project's tensor model or any unit test.
g=[.2,-.5,1.0]; norm=math.sqrt(sum(x*x for x in g)); factor=1/(norm+1e-6); eps=1e-8; lr=.001
def adam_first(values): return [-lr*x/(abs(x)+eps) for x in values]
raw=adam_first(g); clipped=adam_first([x*factor for x in g]); scaled=adam_first([x*.1 for x in g])
checks={'clip_removed_max_first_step_difference':max(abs(a-b) for a,b in zip(raw,clipped)), 'global_scale_point1_max_first_step_difference':max(abs(a-b) for a,b in zip(raw,scaled)), 'new_test_parameter_absolute_tolerance':2e-6}
assert checks['clip_removed_max_first_step_difference']<2e-6 and checks['global_scale_point1_max_first_step_difference']<2e-6
relevant={}
for cls in tm.body:
 if isinstance(cls,ast.ClassDef):
  for fn in cls.body:
   if isinstance(fn,ast.FunctionDef):
    relevant[fn.name]=[(line,name) for line,name in calls(fn) if name in ('history_objective','optimization_step','historical_ranking','current_objective','train_stage','grad','assertAlmostEqual','assertEqual','assertTrue')]
result={'scope':'web Python stdlib AST and independent scalar first-Adam arithmetic; no Torch, BGE, gradients or training executed','source_sha256':hashlib.sha256(code.read_bytes()).hexdigest(),'test_sha256':hashlib.sha256(tests.read_bytes()).hexdigest(),'production_optimizer_calls':serial_calls,'history_weight':coefficient,'history_return_ast':ast.unparse(history_return),'test_calls':relevant,'first_adam_counterexample':checks,'interpretation':'Production body has B backward then history-total backward, one clip and one step. Shared history_objective in expected joint and actual serial leaves missing R/incorrect history-weight assembly unindependently checked; this is a static test-coverage conclusion, not a claim that five Torch mutants were run.'}
print(json.dumps(result,ensure_ascii=False,indent=2))
