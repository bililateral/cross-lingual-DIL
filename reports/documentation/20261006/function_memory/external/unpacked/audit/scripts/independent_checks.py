"""Independent web CPU checks of supplied code and narrow failure examples.
No BGE, no official data, no project execution or actual approval-gate issuance.
"""
from pathlib import Path
import os,sys,json,ast,copy,hashlib,tempfile,itertools
from unittest import mock
R=Path(__file__).resolve().parents[2]; I=R/'input'; O=R/'audit/outputs'
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[k]='1'
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['TOKENIZERS_PARALLELISM']='false'
sys.dont_write_bytecode=True; sys.path[:0]=[str(I/'scripts'),str(I/'tests')]
import numpy as np,torch
import step28_function_memory as m
import step28_function_memory_run as run
import test_step28_function_memory as fixtures
torch.set_num_threads(1);torch.set_num_interop_threads(1)
out={'scope':'WEB_CPU_HANDWRITTEN_ONLY','formal_data':False,'gpu':False,'torch':torch.__version__,'numpy':np.__version__}
# Full independently enumerated query geometry, no production query_indices.
controls=np.repeat(np.arange(12),[3]*4+[2]*8);edges=list(itertools.combinations(range(28),2))
labels=np.array([controls[a]==controls[b] for a,b in edges]);G=np.eye(378)/378
per_query=[]
for q in range(28):
 pos=[i for i,e in enumerate(edges) if q in e and labels[i]];neg=[i for i,e in enumerate(edges) if q in e and not labels[i]]
 per_query.append(len(pos)*len(neg))
 for a,b in itertools.product(pos,neg):
  r=1/(28*len(pos)*len(neg));G[a,a]+=r;G[b,b]+=r;G[a,b]-=r;G[b,a]-=r
rng=np.random.default_rng(7301)
x=rng.uniform(0,.4,(129,378)).astype(np.float16).astype(np.float64);x[-1]=1
y=rng.uniform(0,.4,(129,378)).astype(np.float16).astype(np.float64);y[-1]=1
w=rng.normal(0,.06,129);v=rng.normal(0,.06,129)
H=x@G@x.T;b=H@w;c=float(w@b)
hp,bp,cp=m.statistics(torch.from_numpy(x),labels.astype(np.uint8),torch.from_numpy(w))
# Explicit error before transport
err=x.T@(v-w);explicit=float(err@G@err/3);stat=float((v@H@v-2*b@v+c)/3)
xt=torch.from_numpy(x);yt=torch.from_numpy(y).requires_grad_();wt=torch.from_numpy(v).requires_grad_()
loss=m.historical_loss(yt,wt,xt,torch.from_numpy(H),torch.from_numpy(b),c,1)
dy,dw=torch.autograd.grad(loss,(yt,wt))
K=x@x.T/378+.001*np.eye(129);vv=np.linalg.solve(K,x@(y.T@v)/378+.001*v)
t=np.linalg.solve(K.T,2*(H@vv-b)/3);g=x.T@t/378
f=lambda Y,W:float((lambda V: (V@H@V-2*b@V+c)/3)(np.linalg.solve(K,x@(Y.T@W)/378+.001*W)))
vdir=rng.normal(size=129);vdir/=np.linalg.norm(vdir);ydir=rng.normal(size=(129,378));ydir/=np.linalg.norm(ydir)
eps=1e-5
fdw=(f(y,v+eps*vdir)-f(y,v-eps*vdir))/(2*eps);fdy=(f(y+eps*ydir,v)-f(y-eps*ydir,v))/(2*eps)
z0=torch.from_numpy(x).requires_grad_();w0=torch.from_numpy(w).requires_grad_();loss0=m.historical_loss(z0,w0,xt,H,b,c,1);g0=torch.autograd.grad(loss0,(z0,w0))
out['mathematics']={'positive_edges':int(labels.sum()),'query_comparisons':per_query,'geometry_trace':float(np.trace(G)),
'H_max_error':float(np.max(np.abs(hp.numpy()-H))),'b_max_error':float(np.max(np.abs(bp.numpy()-b))),'c_error':float(abs(cp-c)),
'explicit_loss':explicit,'statistic_loss':stat,'dy_max_error':float(np.max(np.abs(dy.numpy()-v[:,None]*g[None,:]))),
'dw_max_error':float(np.max(np.abs(dw.numpy()-(y@g+.001*t)))),'w_direct_epsilon_norm':float(np.linalg.norm(.001*t)),
'finite_diff_w_error':float(abs(fdw-np.dot(dw.numpy(),vdir))),'finite_diff_y_error':float(abs(fdy-np.sum(dy.numpy()*ydir))),
'creation_loss':float(loss0.detach()),'creation_grad_y_norm':float(g0[0].norm()),'creation_grad_w_norm':float(g0[1].norm())}
assert np.trace(G)-3<1e-12
assert out['mathematics']['H_max_error']<1e-12 and out['mathematics']['dy_max_error']<1e-11
# Actual gate function accepts an authentic CPU receipt in BOTH slots. No gate written, no execute invoked.
p=run.policy();cpu_path=I/'cpu_20261006_220500/result/result.json';cpu_record=run.data.record(cpu_path,I)
job=I/'reports/NOT_A_REAL_JOB_NEVER_EXECUTED';sentinel=O/'NOT_AN_AUTHORIZATION_TEST_INPUT.json'
gate={'status':'APPROVED_FUNCTION_MEMORY_PILOT','source_files':run.sources(),'job':job.relative_to(I).as_posix(),
      'runtime':p['runtime'],'supervision':p['supervision'],'native':cpu_record,'integration_cpu':cpu_record,'review_disposition':'NO_OPEN_BLOCKERS'}
read_json=run.data.read_json
with mock.patch.object(run.data,'read_json',side_effect=lambda path:gate if path==sentinel else read_json(path)):
 accepted=run.validate_gate(job,sentinel)
assert accepted==p
out['formal_gate_cpu_as_native']={'gate_object_handwritten_not_issued':True,'validate_gate_returned_policy':True,
'evidence_path':cpu_record,'both_slots_same_cpu':True,'provided_receipt_mode':read_json(cpu_path)['mode'],
'provided_receipt_has_native': 'native' in read_json(cpu_path),'execute_called':False,'job_created':job.exists()}
# Isolated exact AST watchdog: record writer failure prevents os._exit. No actual exit/threads.
src=(I/'scripts/step28_function_memory_verify.py').read_text();tree=ast.parse(src)
main=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='main')
watch=next(n for n in main.body if isinstance(n,ast.FunctionDef) and n.name=='watchdog')
class Stop:
 def wait(self,t):return False
class FakeOS:
 def __init__(self):self.exits=[]
 def _exit(self,n):self.exits.append(n);raise SystemExit(n)
def fail_check():raise RuntimeError('HANDWRITTEN_LIMIT_FAILURE')
def fail_write():raise OSError('HANDWRITTEN_DISK_FAILURE')
fake=FakeOS();env={'stopped':Stop(),'check':fail_check,'report':{},'write':fail_write,'os':fake}
exec(compile(ast.Module(body=[watch],type_ignores=[]),'<exact_supplied_watchdog_AST>','exec'),env)
try:env['watchdog']()
except BaseException as e: outcome=type(e).__name__+':'+str(e)
out['verification_watchdog_write_failure']={'exit_calls':fake.exits,'exception':outcome,'report':env['report'],
'function_source_lines':[watch.lineno,watch.end_lineno],'production_code_modified':False,'live_process_used':False}
# One label-attempt failure is consumed before loader. Fresh handwritten ledger only.
with tempfile.TemporaryDirectory(dir=O) as tmp:
 folder=Path(tmp);run.data.write_json(folder/'access.json',dict(train=0,valid=0,heldout=0,owners=0));n=[0]
 def fake_loader(*args):n[0]+=1;raise OSError('fixture loader failure')
 attempts=[]
 for split in ('train','development'):
  for _ in range(2):
   try:run.previous.parse_once(folder,[],m.config(),split,loader=fake_loader)
   except Exception as e:attempts.append({'split':split,'error':type(e).__name__})
 out['label_attempt_guard']={'loader_calls':n[0],'ledger':run.data.read_json(folder/'access.json'),'attempts':attempts}
 assert n[0]==2
# Preserve exact head parameter identities & straight-through gradient dtype/value.
model=fixtures.tiny_model();optimizer=m.make_optimizer(model,m.config())
params=list(model.parameters());registered=[q for group in optimizer.param_groups for q in group['params']]
assert len(registered)==len({id(q) for q in registered})==len(params)
whead=m.weight(model);test_grad=torch.arange(129,dtype=torch.float32);(whead*test_grad).sum().backward()
h=torch.linspace(.001,1.1,378*128).reshape(378,128).requires_grad_();z=m.canonical_features(h);incoming=torch.full_like(h,1.000123);(z[:-1].T*incoming).sum().backward()
out['parameter_and_precision']={'model_parameters':len(params),'optimizer_unique_parameters':len({id(q) for q in registered}),
'last_weight_gradient_exact':torch.equal(model.head[2].weight.grad.reshape(-1),test_grad[:-1]),
'last_bias_gradient_exact':torch.equal(model.head[2].bias.grad,test_grad[-1:]),
'quantized_forward_exact':torch.equal(z[:-1].T,h.detach().half().float()),'ste_gradient_exact':torch.equal(h.grad,incoming),'ste_gradient_dtype':str(h.grad.dtype)}
(O/'independent_checks.json').write_text(json.dumps(out,ensure_ascii=False,indent=2))
print(json.dumps(out,ensure_ascii=False,indent=2))
