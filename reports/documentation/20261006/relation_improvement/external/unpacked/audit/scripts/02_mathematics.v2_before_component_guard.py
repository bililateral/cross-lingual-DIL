"""Independent mathematical examples only; no project imports, text, labels, or BGE.
The 28-account labels below are explicitly handwritten synthetic group structure.
The straight-through estimator is a proposed update convention, not a true derivative
of FP16 rounding. Full-precision coordinate derivatives are checked separately.
"""
from __future__ import annotations
import json, math, sys, time
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[2]
torch.set_num_threads(1)
rng=np.random.default_rng(20261006)
start=time.monotonic()
n,m=28,378; d=129; eps=.001
owners=np.repeat(np.arange(12),[3,3,3,3,2,2,2,2,2,2,2,2])
edges=[(i,j) for i in range(n) for j in range(i+1,n)]
idx={e:k for k,e in enumerate(edges)}
y=np.array([owners[i]==owners[j] for i,j in edges])
queries=[]
G=np.eye(m)/m
for q in range(n):
    pos=[idx[tuple(sorted((q,j)))] for j in range(n) if j!=q and owners[q]==owners[j]]
    neg=[idx[tuple(sorted((q,j)))] for j in range(n) if j!=q and owners[q]!=owners[j]]
    queries.append((pos,neg))
    a=1/(n*len(pos)*len(neg))
    for p in pos:
        for k in neg:
            G[p,p]+=a;G[k,k]+=a;G[p,k]-=a;G[k,p]-=a
assert len(edges)==378 and y.sum()==20
assert sum(len(p)*len(k) for p,k in queries)==1016
assert np.allclose(np.trace(G),3,atol=1e-13)
assert np.allclose(G@np.ones(m),np.ones(m)/m,atol=1e-14)

def explicit(residual):
    return float(np.mean(residual**2)+sum(np.mean((residual[np.array(p)][:,None]-residual[np.array(k)][None,:])**2) for p,k in queries)/28)

def Q_v(v,H,b,c,N=1):
    return float((v@H@v-2*b@v+c)/(3*N))

def transported(X,Y,w,H,b,c,N=1):
    K=X@X.T/m+eps*np.eye(len(w)); v=np.linalg.solve(K,X@(Y.T@w)/m+eps*w)
    return Q_v(v,H,b,c,N),v,K

# Exact sufficient statistics in one canonical FP16->FP32 coordinate system.
raw=rng.uniform(.3,1.6,size=(128,m)).astype(np.float32)
X=np.vstack([raw.astype(np.float16).astype(np.float64),np.ones(m)])
w0=rng.normal(scale=.2,size=d)
H=X@G@X.T;b=H@w0;c=float(w0@b)
w=w0+rng.normal(scale=.05,size=d)
res=X.T@(w-w0)
q_exp=explicit(res)/3;q_stat=Q_v(w,H,b,c)
assert abs(q_exp-q_stat)<1e-11
q0,v0,K=transported(X,X,w0,H,b,c)
assert abs(q0)<1e-11 and np.max(np.abs(v0-w0))<1e-10
Yraw=np.vstack([raw.astype(np.float64),np.ones(m)])
q_bad,v_bad,_=transported(X,Yraw,w0,H,b,c)
# Positive mismatch diagnosed with residual expression rather than cancellation.
bad_resid=(v_bad-w0)@H@(v_bad-w0)/3
assert bad_resid>0
rawteacher=Yraw.T@w0
quant_teacher=X.T@w0
qt_err=explicit(quant_teacher-rawteacher)/3
assert qt_err>0

# Full-precision gradient independent reference plus Torch automatic differentiation.
dg=5
Xg=np.vstack([rng.uniform(.3,1.8,size=(dg-1,m)),np.ones(m)])
Z2=np.vstack([rng.uniform(.2,2.1,size=(dg-1,m)),np.ones(m)])
t1=rng.normal(size=dg);t2=rng.normal(size=dg)
Hg=Xg@G@Xg.T+Z2@G@Z2.T
bg=(Xg@G@Xg.T)@t1+(Z2@G@Z2.T)@t2
cg=float(t1@(Xg@G@Xg.T)@t1+t2@(Z2@G@Z2.T)@t2)
Yg=Xg+rng.normal(scale=.12,size=Xg.shape);Yg[-1]=1
wg=rng.normal(size=dg)
qg,vg,Kg=transported(Xg,Yg,wg,Hg,bg,cg,N=2)
a=2*(Hg@vg-bg)/6;t=np.linalg.solve(Kg.T,a);g=Xg.T@t/m
gY=np.outer(wg,g);gw=Yg@g+eps*t
xt=torch.tensor(Xg);yt=torch.tensor(Yg,requires_grad=True);wt=torch.tensor(wg,requires_grad=True)
ht=torch.tensor(Hg);bt=torch.tensor(bg)
kt=xt@xt.T/m+eps*torch.eye(dg,dtype=torch.float64)
vt=torch.linalg.solve(kt,xt@(yt.T@wt)/m+eps*wt)
loss=(vt@ht@vt-2*bt@vt+cg)/6
ay,aw=torch.autograd.grad(loss,(yt,wt))
errY=float(np.max(np.abs(ay.detach().numpy()-gY)));errW=float(np.max(np.abs(aw.detach().numpy()-gw)))
assert max(errY,errW)<1e-10
DY=rng.normal(size=Yg.shape);DY[-1]=0;DY/=np.linalg.norm(DY)
DW=rng.normal(size=wg.shape);DW/=np.linalg.norm(DW)
h=1e-5
fdY=(transported(Xg,Yg+h*DY,wg,Hg,bg,cg,2)[0]-transported(Xg,Yg-h*DY,wg,Hg,bg,cg,2)[0])/(2*h)
fdW=(transported(Xg,Yg,wg+h*DW,Hg,bg,cg,2)[0]-transported(Xg,Yg,wg-h*DW,Hg,bg,cg,2)[0])/(2*h)
fd_err=max(abs(fdY-np.sum(gY*DY)),abs(fdW-gw@DW))
assert fd_err<1e-7

# Cross-stage teacher semantics: no reset of old b/c to the newest head.
Hs=np.array([[2.]]);bs=np.array([3.]);cs=5.;ws=np.array([2.])
kept=Q_v(ws,Hs,bs,cs,2);kept_grad=2*(Hs@ws-bs)/6
wrong=Q_v(ws,Hs,Hs@ws,float(ws@Hs@ws),2)
assert abs(kept-1/6)<1e-14 and wrong==0 and abs(kept_grad[0]-1/3)<1e-14

# Exact congruence identity (any supplied T, no claim that ridge finds the true map).
A=np.eye(dg)+rng.normal(scale=.1,size=(dg,dg));A[-1]=np.eye(dg)[-1]
left=Q_v(A.T@wg,Hg,bg,cg,2)
right=Q_v(wg,A@Hg@A.T,A@bg,cg,2)
assert abs(left-right)<1e-12
# Same live cache/current w, different eliminated history changes the actual gradient.
_,va,ka=transported(Xg,Yg,wg,Hg,bg,cg,2)
Ha=Xg@G@Xg.T;ba=Ha@t1
aga=2*(Ha@va-ba)/3
wa=(Yg@Xg.T/m+eps*np.eye(dg))@np.linalg.solve(ka,aga)
assert np.linalg.norm(wa-gw)>1e-4

# A true 129x378 cache-output null direction survives function recentering.
U,S,Vh=np.linalg.svd(X,full_matrices=True)
rank=int(np.linalg.matrix_rank(X));delta=Vh[-1]
null_err=float(np.linalg.norm(X@delta))
ww=np.zeros(d);ww[0]=1
lim=.5*np.min(X[0]/np.maximum(np.abs(delta),1e-30))
chosen=None
for qi,(p,nn) in enumerate(queries):
    for ip in p:
        for inn in nn:
            margin=X[0,ip]-X[0,inn];slope=delta[ip]-delta[inn]
            if margin>0 and abs(slope)>1e-8:
                alpha=-1.5*margin/slope
                if abs(alpha)<lim:
                    chosen=(qi,ip,inn,margin,alpha);break
        if chosen:break
    if chosen:break
assert chosen is not None
qi,ip,inn,margin,alpha=chosen
Yblind=X.copy();Yblind[0]+=alpha*delta
assert np.min(Yblind[:-1])>=0
bb=H@ww;cc=float(ww@bb)
qb,vb,_=transported(X,Yblind,ww,H,bb,cc)
blind_res=(vb-ww)@H@(vb-ww)/3
assert blind_res<1e-18 and Yblind[0,ip]-Yblind[0,inn]<0

# Exact binary16 construction: all reference AND live values lie on the FP16 grid.
# It removes the continuous-shadow qualification from a second, independent example.
X16=X.copy(); ia=idx[(0,3)];ib=idx[(0,1)];ic=idx[(0,4)]
X16[:,[ia,ib,ic]]=1.0;X16[0,[ia,ib,ic]]=[.5,1.,1.5]
D16=np.zeros(m);D16[[ia,ib,ic]]=[.25,-.5,.25]
Y16=X16.copy();Y16[0]+=D16
assert np.array_equal(X16,X16.astype(np.float16).astype(np.float64))
assert np.array_equal(Y16,Y16.astype(np.float16).astype(np.float64))
assert np.array_equal(X16@D16,np.zeros(d))
H16=X16@G@X16.T;b16=H16@ww;c16=float(ww@b16)
_,v16,_=transported(X16,Y16,ww,H16,b16,c16)
q16center=float((v16-ww)@H16@(v16-ww)/3)
rank16=int(np.linalg.matrix_rank(X16))
assert rank16==129 and q16center<1e-18
assert X16[0,ib]-X16[0,ia]==.5 and Y16[0,ib]-Y16[0,ia]==-.25

# Rounding has zero almost-everywhere true derivative; STE is explicitly a surrogate.
class Canonical16(torch.autograd.Function):
    @staticmethod
    def forward(ctx,x):
        q=x.to(torch.float16)
        if not torch.isfinite(q).all():raise FloatingPointError('FP16 overflow')
        return q.to(torch.float32)
    @staticmethod
    def backward(ctx,grad):return grad
x=torch.tensor([1.234],dtype=torch.float32,requires_grad=True)
l=(2*Canonical16.apply(x)-.8).square().sum();l.backward()
xd=float(x.detach()[0]);step=1e-6
f=lambda z:float((2*np.float16(z).astype(np.float64)-.8)**2)
true_fd=(f(xd+step)-f(xd-step))/(2*step)
ste=float(x.grad[0]);assert true_fd==0 and abs(ste)>1
# Overflow must not be silently clipped.
overflow_detected=False
try:Canonical16.apply(torch.tensor([70000.]))
except FloatingPointError:overflow_detected=True
assert overflow_detected

# d129 uses a view of the existing model, with no new Parameters.
torch.manual_seed(42)
head=torch.nn.Sequential(torch.nn.Linear(8192,128),torch.nn.ReLU(),torch.nn.Linear(128,1))
u=torch.randn(7,8192)
hh=head[1](head[0](u));zz=torch.cat((hh,torch.ones(7,1)),dim=1)
wview=torch.cat((head[2].weight.reshape(-1),head[2].bias))
out1=head[2](hh).reshape(-1);out2=zz@wview
params=tuple(head.parameters())
g1=torch.autograd.grad(out1.square().sum(),params,retain_graph=True)
g2=torch.autograd.grad(out2.square().sum(),params)
head_output_error=float(torch.max(torch.abs(out1-out2)).detach())
head_grad_error=max(float(torch.max(torch.abs(a-b))) for a,b in zip(g1,g2))
assert head_output_error<1e-6 and head_grad_error<1e-5
assert len(params)==4 and sum(p.numel() for p in params)==1048833

# Exact normalization/size accounting, and a criterion counterexample.
byte_sizes={'six_fp32_refs':6*378*129*4,'six_fp16_refs':6*378*129*2,'H64':129*129*8,'b64':129*8,'c64_N64':16}
num=byte_sizes['six_fp16_refs']+byte_sizes['H64']+byte_sizes['b64']+16
assert num==719320
other=[484442-298768,504581-298768]
est=[num+k for k in other]
# Delta N=.5(delta new_at_stage2 + delta Z).
dO,dZ,dnew2=.04,-.07,-.11;dfinal=(2*dO+dZ)/3;dN=(dnew2+dZ)/2
assert dO>0 and dfinal>0 and dN<-.01 and dZ<-.01
result={
 'kind':'independent_web_sandbox_math_no_project_execution',
 'environment':{'python':sys.version,'numpy':np.__version__,'torch':torch.__version__,'torch_threads':torch.get_num_threads()},
 'group':{'accounts':28,'edges':378,'positives':int(y.sum()),'query_comparisons':1016,'trace_G':float(np.trace(G)),'trace_G_over3':float(np.trace(G/3)),'score_nullity_at_least':378-129},
 'canonical_target':{'explicit_loss_div3':q_exp,'statistics_loss_div3':q_stat,'error':abs(q_exp-q_stat),'initial_expanded_Q_div3':q0,'initial_v_minus_teacher_max':float(np.max(np.abs(v0-w0))),'ref16_live32_centered_Q_div3':float(bad_resid),'ref16_live32_v_error_max':float(np.max(np.abs(v_bad-w0))),'raw_teacher_vs_quantized_teacher_Q_div3':qt_err,'gram_condition_number':float(np.linalg.cond(K))},
 'gradient':{'torch_vs_analytic_Y_max':errY,'torch_vs_analytic_w_max':errW,'directional_finite_difference_max':fd_err,'w_path_via_p_norm':float(np.linalg.norm(Yg@g)),'w_direct_epsilon_path_norm':float(np.linalg.norm(eps*t)),'same_cache_different_history_w_gradient_difference':float(np.linalg.norm(wa-gw))},
 'accumulation':{'kept_old_target_Q_div3':kept,'kept_old_target_grad':float(kept_grad[0]),'incorrect_recenter_Q':wrong,'congruence_identity_max_error':abs(left-right)},
 'function_target_blindness':{'cache_rank':rank,'cache_nullity':378-rank,'X_delta_norm':null_err,'query':qi,'positive_edge':edges[ip],'negative_edge':edges[inn],'alpha':alpha,'correct_margin_before':float(margin),'margin_after':float(Yblind[0,ip]-Yblind[0,inn]),'centered_function_Q_after_div3':float(blind_res),'live_Y_is_float_continuous_shadow_not_a_BGE_trajectory':True},
 'exact_fp16_blindness':{'cache_rank':rank16,'X_delta_is_exact_zero':True,'all_live_and_reference_values_fp16_exact':True,'margin_before':.5,'margin_after':-.25,'centered_function_Q_after_div3':q16center,'v_teacher_max_abs':float(np.max(np.abs(v16-ww))),'raw_history_labels_still_usable_by_B':True,'not_claimed_to_be_reachable_by_BGE':True},
 'quantization_surrogate':{'input':xd,'canonical':float(np.float16(xd)),'true_finite_difference':true_fd,'declared_STE_gradient':ste,'overflow_rejected':overflow_detected},
 'head_view':{'num_parameter_tensors':len(params),'num_parameters':sum(p.numel() for p in params),'output_max_abs_difference':head_output_error,'gradient_max_abs_difference':head_grad_error,'note':'ordinary full head forward remains unchanged; the concatenation is a live view construction, not another Parameter'},
 'bytes':{**byte_sizes,'numeric_total':num,'remaining_before_text_metadata':1048576-num,'old_non_numeric_range':other,'heuristic_new_total_range':est,'heuristic_headroom_at_high':1048576-est[1],'not_new_memory_serialization':True},
 'criterion_counterexample':{'delta_O':dO,'delta_Z':dZ,'delta_stage2_new':dnew2,'delta_N':dN,'delta_final_all':dfinal},
 'elapsed_seconds':time.monotonic()-start,
 'not_executed':['project Linux tests','BGE/GPU','formal data','model checkpoints','new candidate training','new serializer']}
(ROOT/'audit/outputs/mathematics.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(result,ensure_ascii=False,indent=2))
