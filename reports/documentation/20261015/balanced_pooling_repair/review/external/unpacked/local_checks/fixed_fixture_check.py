"""Replay only the supplied first/current terminal integration fixture.
Observation hooks delegate unchanged production operations. No project server,
BGE loading, formal data, qualification main(), or formal execute() is called.
"""
from pathlib import Path
import os,sys,json,hashlib,platform,time,unittest,traceback,itertools
root=Path(sys.argv[1]).resolve(); out=Path(sys.argv[2]).resolve();out.mkdir(parents=True,exist_ok=True)
os.environ['BALANCED_POOLING_TEST_ROOT']=str(out)
sys.path[:0]=[str(root/'scripts'),str(root/'tests')]
start=time.perf_counter()
import torch,numpy as np
import test_step28_balanced_pooling as h
m,r=h.method,h.record
from unittest import mock
torch.set_num_threads(1);torch.set_num_interop_threads(1)
h.run.parent.configure_numerics()
captures=[];steps=[]
original_pool=m.account_logits
original_step=torch.optim.AdamW.step
def observe_pool(table,slots,*args,**kw):
 v=original_pool(table,slots,*args,**kw)
 captures.append({'table':table.detach().cpu().clone(),'slots':np.array(slots),'scores':v.detach().cpu().clone()})
 return v
def observe_step(opt,*args,**kw):
 before=[[p.detach().clone() for p in g['params']] for g in opt.param_groups]
 v=original_step(opt,*args,**kw)
 diffs=[]
 for gs,g in zip(before,opt.param_groups):
  deltas=[float((p.detach()-b).abs().max()) for b,p in zip(gs,g['params'])]
  diffs.append({'lr':g['lr'],'max_displacement_from_pre_step':max(deltas),'parameter_displacements':deltas,
                'changed_elements':sum(int((p.detach()!=b).sum()) for b,p in zip(gs,g['params']))})
 steps.append(diffs);return v
res={}
with (out/'unittest.txt').open('w') as f:
 with mock.patch.object(m,'account_logits',side_effect=observe_pool),mock.patch.object(torch.optim.AdamW,'step',observe_step):
  result=unittest.TextTestRunner(stream=f,verbosity=2).run(unittest.TestSuite([h.TerminalPoolingTests('test_terminal_real_chunked_update')]))
res.update(test_success=result.wasSuccessful(),failures=len(result.failures),errors=len(result.errors),skipped=len(result.skipped),actual_adamw_step_calls=len(steps),optimizer_step_observations=steps,production_pool_calls=len(captures))
# This post-test analysis only reads captured fixed handwritten tensors.
if len(captures)==5:
 (cg,cc),(hg,hc)=[h.terminal_group('terminal_'+role,role) for role in ('current','history')]
 lookup={p:i for i,p in enumerate(itertools.combinations(range(84),2))}
 a=[int(np.flatnonzero(cc[:3]==j)[0]) for j in range(3)]
 b=[int(np.flatnonzero(cc[3:6]==j)[0])+3 for j in range(3)]
 base=np.array([[float(captures[0]['table'][lookup[min(i,j),max(i,j)]]) for j in b] for i in a])
 ref,plan=h.rounded_reference(base);analysis=[];arrays={}
 for ix,cap in enumerate(captures):
  colors=cc if ix<2 else hc; group=cg if ix<2 else hg
  table,slots,actual=cap['table'],cap['slots'],cap['scores']; expected=[];max_residual=0.;error_bound=0.
  for aa,bb in itertools.combinations(range(28),2):
   rows=np.flatnonzero(slots==aa);cols=np.flatnonzero(slots==bb)
   vals=torch.stack([table[lookup[min(i,j),max(i,j)]] for i in rows for j in cols])
   ps=torch.tensor([plan[colors[i],colors[j]] for i in rows for j in cols],dtype=table.dtype)
   expected.append((vals*ps).sum()-(ps*torch.log(9*ps)).sum())
   k=vals.double().numpy().reshape(3,3);k0=base[colors[rows,None],colors[None,cols]]
   E=k-k0;E=E-E.mean(axis=1,keepdims=True)-E.mean(axis=0,keepdims=True)+E.mean()
   max_residual=max(max_residual,float(np.abs(E).max()))
   # Pinsker + symmetric KL identity: exact plan L1 <= ||E||_infinity.
  expected=torch.stack(expected)
  adjA=np.zeros((28,28));adjR=np.zeros((28,28));labels=np.zeros((28,28),bool)
  pi,pj=np.triu_indices(28,1)
  adjA[pi,pj]=adjA[pj,pi]=actual.numpy();adjR[pi,pj]=adjR[pj,pi]=expected.numpy();labels[pi,pj]=labels[pj,pi]=np.array(group.labels,bool)
  mismatches=[];border=[]
  for q in range(28):
   neg=[j for j in range(28) if j!=q and not labels[q,j]]
   ta=sorted(neg,key=lambda j:(-adjA[q,j],j))[:5];tr=sorted(neg,key=lambda j:(-adjR[q,j],j))[:5]
   if set(ta)!=set(tr):mismatches.append({'query':q,'actual_top5':ta,'independent_top5':tr})
   sr=sorted([adjR[q,j] for j in neg],reverse=True);border.append(sr[4]-sr[5])
  analysis.append({'call':ix,'role':['current_A0','current_A1','history_A0','history_A1','teacher_A0'][ix],
   'max_value_delta':float((actual-expected).abs().max()),'hard_negative_set_mismatch_queries':len(mismatches),
   'mismatch_examples':mismatches[:3],'minimum_reference_fifth_sixth_margin':min(border),
   'maximum_nonadditive_FP32_residual':max_residual,
   'exact_optimum_plan_L1_perturbation_bound':max_residual,
   'fixed_plan_value_suboptimality_bound':max_residual**2})
  arrays[f'actual_scores_{ix}']=actual.numpy();arrays[f'independent_scores_{ix}']=expected.numpy()
 res.update(actual_symmetric_reference_input=base.tolist(),reference_value=ref,reference_plan=plan.tolist(),initial_pool_analysis=analysis)
 np.savez_compressed(out/'handwritten_scores.npz',**arrays)
res.update(elapsed_seconds=time.perf_counter()-start,environment={'python':sys.version,'platform':platform.platform(),'torch':torch.__version__,'numpy':np.__version__,'cpu_threads':torch.get_num_threads(),'gpu_visible':os.environ.get('CUDA_VISIBLE_DEVICES'),'project_linux':False},scope='Only one supplied handwritten integration test per source version; includes production AdamW and independent reference AdamW; observer-only delegation; not project Linux qualification')
res['source_files']=[{'path':str(p.relative_to(root)),'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in sorted(root.rglob('*')) if p.is_file() and p.suffix in ('.py','.json','.sh','.md') and ('reports' not in p.relative_to(root).parts or 'qualification' not in p.relative_to(root).parts)]
(out/'result.json').write_text(json.dumps(res,indent=2,ensure_ascii=False)+'\n')
print(json.dumps({k:v for k,v in res.items() if k not in ('source_files','reference_plan')},ensure_ascii=False,indent=2))
