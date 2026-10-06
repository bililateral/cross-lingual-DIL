"""Two-stage handwritten statistics/transport/container, without training or model weights."""
from pathlib import Path
import os,sys,json,itertools,hashlib
R=Path(__file__).resolve().parents[2];I=R/'input';O=R/'audit/outputs'
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[k]='1'
os.environ['CUDA_VISIBLE_DEVICES']='';sys.dont_write_bytecode=True;sys.path[:0]=[str(I/'scripts'),str(I/'tests')]
import numpy as np,torch
import step28_function_memory as m
import test_step28_function_memory as fixtures
torch.set_num_threads(1)
old=[fixtures.handmade_group(f'a{i:02}') for i in range(48)]
new=[fixtures.handmade_group(f'b{i:02}') for i in range(48)]
edges=list(itertools.combinations(range(28),2));labs=old[0].labels;G=np.eye(378)/378
for q in range(28):
 pos=[i for i,e in enumerate(edges) if q in e and labs[i]];neg=[i for i,e in enumerate(edges) if q in e and not labs[i]]
 for a,b in itertools.product(pos,neg):
  c=1/(28*len(pos)*len(neg));G[a,a]+=c;G[b,b]+=c;G[a,b]-=c;G[b,a]-=c
rng=np.random.default_rng(66);cache=[]
for i in range(8):
 z=(rng.integers(1,20,(129,378))/64).astype(np.float16);z[-1]=1;cache.append(z)
def initial(g):return cache[int(g.uid[1:])%4]
def changed(g):
 i=int(g.uid[1:]);z=cache[(i%4)+(4 if g.uid[0]=='b' else 0)].copy()
 if g.uid[0]=='a':z[:-1]*=np.float16([.5,1.,1.5,2.][i%4])
 return z
w1=np.linspace(-.06,.09,129);w2=np.linspace(.10,-.04,129)
mem=m.Memory('ABC');e1=mem.consolidate(old,1,initial,torch.from_numpy(w1));first=mem.to_bytes()
H1=sum(z.astype(float)@G@z.astype(float).T for z in (initial(g) for g in old));B1=H1@w1;C1=float(w1@B1)
assert np.max(abs(mem.h-H1))<1e-10
retained=list(mem.reservoir.groups);X=np.concatenate([initial(g).astype(float) for g in retained],axis=1);Y=np.concatenate([changed(g).astype(float) for g in retained],axis=1)
K=X@X.T/X.shape[1]+.001*np.eye(129);T=np.linalg.solve(K,(Y@X.T/X.shape[1]+.001*np.eye(129)).T).T
x1=initial(retained[0]).astype(float);y1=changed(retained[0]).astype(float);Tfirst=np.linalg.solve(x1@x1.T/378+.001*np.eye(129),(y1@x1.T/378+.001*np.eye(129)).T).T
mem.begin_stage(2)
for _ in range(288):mem.draw()
e2=mem.consolidate(new,2,changed,torch.from_numpy(w2))
Hnew=sum(z.astype(float)@G@z.astype(float).T for z in (changed(g) for g in new));Bnew=Hnew@w2;Cnew=float(w2@Bnew)
Hwant=T@H1@T.T+Hnew;Bwant=T@B1+Bnew;Cwant=C1+Cnew
errors={'H':float(np.max(abs(mem.h-Hwant))),'b':float(np.max(abs(mem.b-Bwant))),'c':float(abs(mem.constant-Cwant))}
assert max(errors.values())<1e-8
payload=mem.to_bytes();restored=m.Memory.from_bytes(payload);assert restored.to_bytes()==payload
assert all(np.array_equal(mem.references[g.uid],changed(g)) for g in mem.reservoir.groups)
rejected=[]
for name,p in [('tail',payload+b'x'),('truncated',payload[:-1]),('old_magic',b'RELATION'+payload[8:])]:
 try:m.Memory.from_bytes(p)
 except Exception as e:rejected.append({'case':name,'exception':type(e).__name__})
assert len(rejected)==3
mem.begin_stage(3);restored.begin_stage(3)
for _ in range(7):
 a,x=mem.draw();b,y=restored.draw();assert a.uid==b.uid and np.array_equal(x,y)
mid=m.Memory.from_bytes(mem.to_bytes())
for _ in range(281):
 a,x=mem.draw();b,y=mid.draw();assert a.uid==b.uid and np.array_equal(x,y)
assert mem.to_bytes()==mid.to_bytes()
out={'scope':'HANDWRITTEN_FEATURES_NOT_TRAINING','new_teacher_groups_each':48,'actual_optimizer_steps':0,
'initial_count':e1['new_count'],'second_count':e2['new_count'],'second_seen':mem.reservoir.seen,
'first_bytes':len(first),'second_bytes':len(payload),'mathematical_numeric_bytes':6*378*129*2+129*129*8+129*8+16,
'joint_T_vs_first_group_max_difference':float(np.max(abs(T-Tfirst))),'expected_statistic_max_errors':errors,
'old_target_not_recentered_norm':float(np.linalg.norm(mem.b-mem.h@w2)),
'reference_refresh_exact':True,'seven_then_281_draw_restore_exact':True,'rejected_containers':rejected,
'first_members':e1['new_members'],'second_members':e2['new_members'],'second_H_constant_diagonal':float(mem.h[-1,-1])}
(O/'lifecycle_checks.json').write_text(json.dumps(out,ensure_ascii=False,indent=2));print(json.dumps(out,ensure_ascii=False,indent=2))
