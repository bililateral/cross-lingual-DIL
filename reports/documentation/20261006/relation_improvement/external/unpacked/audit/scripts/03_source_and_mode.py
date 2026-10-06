"""Read-only source inventory and an independent checkpoint/dropout-mode toy.
No import of project modules; no BGE or formal inputs.
"""
from __future__ import annotations
import ast, hashlib, json
from pathlib import Path
import torch
from torch.utils.checkpoint import checkpoint
ROOT=Path(__file__).resolve().parents[2];SRC=ROOT/'input/active_source'
files=['scripts/step28_continual_population.py','scripts/step28_chinese_base.py',
'scripts/step28_alias_ranking.py','scripts/step28_bge_continual.py',
'scripts/step28_bge_continual_run.py','scripts/step28_er_weight.py',
'scripts/step28_relation_memory.py','scripts/step28_relation_memory_run.py',
'scripts/step28_continual_expression_run.py','schema/step28_relation_memory_policy.json']
records=[]
for name in files:
    p=SRC/name
    if not p.exists():continue
    b=p.read_bytes();s=b.decode('utf-8');r={'path':'active_source/'+name,'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest(),'lines':len(s.splitlines())}
    if p.suffix=='.py':
        tree=ast.parse(s)
        r['top_level_definitions']=[{'name':n.name,'line':n.lineno,'end_line':n.end_lineno,'kind':type(n).__name__} for n in tree.body if isinstance(n,(ast.ClassDef,ast.FunctionDef))]
        if name.endswith('step28_relation_memory.py'):
            terms=['DIMENSION =','5 * self.count','5.0 *','MAGIC','np.float32','np.float64','1.000001','def canonical_reference']
            r['static_schema_hits']=[{'line':i,'text':line.strip()} for i,line in enumerate(s.splitlines(),1) if any(t in line for t in terms)]
    records.append(r)

torch.set_num_threads(1);torch.manual_seed(3)
# Comparing a deterministic direct graph with checkpoint recomputation inside a
# matching mode interval. This is not intended to simulate every BERT layer.
a=torch.nn.Linear(5,7);drop=torch.nn.Dropout(.5);xx=torch.randn(4,5)
def fn(x):return torch.tanh(drop(a(x)))
def go(checkpointed,restore_early):
    a.zero_grad(set_to_none=True);x=xx.clone().requires_grad_();drop.p=0.
    out=checkpoint(fn,x,use_reentrant=False) if checkpointed else fn(x)
    if restore_early:drop.p=.5
    try:
        out.square().sum().backward()
        return {'completed':True,'gradient':a.weight.grad.clone().detach()}
    except Exception as e:
        return {'completed':False,'exception_type':type(e).__name__,'exception':str(e)}
    finally:drop.p=.5
reference=go(False,False);held=go(True,False);early=go(True,True)
assert reference['completed'] and held['completed']
error=float((reference['gradient']-held['gradient']).abs().max())
assert error==0
if early['completed']:
    bad_error=float((reference['gradient']-early['gradient']).abs().max())
    assert bad_error>0
    early={k:v for k,v in early.items() if k!='gradient'}|{'gradient_diff_max':bad_error}
# Keep errors from intentional bad mode as expected negative examples.
result={'kind':'source_AST_and_independent_toy_only','sources':records,
'mode_checkpoint_toy':{'reference_vs_same_mode_checkpoint_gradient_max_abs':error,'restored_before_backward':early,'dropout_restored_final':drop.p,'not_a_project_BGE_test':True},
'no_project_module_imported':True,'no_checkpoint_or_formal_data_loaded':True}
(ROOT/'audit/outputs/source_and_mode.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(result,ensure_ascii=False,indent=2))
