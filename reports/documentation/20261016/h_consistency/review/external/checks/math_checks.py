#!/usr/bin/env python3
"""Independent handwritten algebraic checks. NEVER imports project modules.
CPU tensors only; no model, optimizer, training, project fixture or private payload.
The exact input is inputs.json. Expected values derive from independent numpy
formulas / analytic identities, not any submitted H/JS function.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
import sys, json, math, platform, traceback, time
from pathlib import Path
import numpy as np
import torch

torch.set_num_threads(1)
torch.set_num_interop_threads(1)
torch.set_default_dtype(torch.float64)
ROOT = Path(__file__).resolve().parent
F = json.loads((ROOT/'inputs.json').read_text(encoding='utf-8'))
R = {'scope':F['scope'], 'environment':{'python':sys.version,'numpy':np.__version__,
     'torch':torch.__version__, 'platform':platform.platform(), 'device':'cpu',
     'torch_num_threads':torch.get_num_threads(), 'CUDA_VISIBLE_DEVICES':os.environ['CUDA_VISIBLE_DEVICES']},
     'checks':[], 'numeric':{}, 'interpretation_limits':[
         'No project imports, encoder, scoring head, optimizer, labels, Memory, GPU or training.',
         'FP64 finite differences test real-valued algebra; they do not certify FP32 production casts.',
         'All generated K tables are abstract handwritten tables; BGE realizability was not tested.',
         'Deliberate bad formulas below are mutation checks, not observed production defects.']}
arrays={}

def check(name, condition, details=None):
    ok=bool(condition)
    item={'name':name,'status':'PASS' if ok else 'FAIL','details':details}
    R['checks'].append(item)
    print(json.dumps(item,ensure_ascii=False),flush=True)
    if not ok: raise AssertionError(name)

def lsm(x):
    a=np.asarray(x,dtype=np.float64)
    shift=a-a.max(axis=-1,keepdims=True)
    return shift-np.log(np.exp(shift).sum(axis=-1,keepdims=True))

def scalar_js(z0,z1):
    lp,lr=lsm(z0),lsm(z1)
    lm=np.logaddexp(lp,lr)-math.log(2)
    p,r=np.exp(lp),np.exp(lr)
    klp=(p*(lp-lm)).sum(axis=-1)
    klr=(r*(lr-lm)).sum(axis=-1)
    return (0.5*(klp+klr)).mean(),(lp,lr,lm,klp,klr)

def independent_grad(z0,z1):
    c,(lp,lr,lm,klp,klr)=scalar_js(z0,z1)
    rows=np.asarray(z0).shape[0]
    g0=0.5*np.exp(lp)*(lp-lm-klp[...,None])/rows
    g1=0.5*np.exp(lr)*(lr-lm-klr[...,None])/rows
    return c,g0,g1

def tensor_js(z0,z1,stop_m=False,stop_view1=False):
    if stop_view1: z1=z1.detach()
    lp=torch.log_softmax(z0,dim=-1); lr=torch.log_softmax(z1,dim=-1)
    lm=torch.logaddexp(lp,lr)-math.log(2)
    if stop_m: lm=lm.detach()
    return (0.5*(lp.exp()*(lp-lm)+lr.exp()*(lr-lm)).sum(dim=-1)).mean()

pairs=[tuple(x) for x in F['record_pair_order']]
pair_index={e:k for k,e in enumerate(pairs)}
slots0=F['view0']; slots1=F['view1_rotated']
account_pairs=[(a,b) for a in range(28) for b in range(a+1,28)]
row_candidates=[[j for j in range(28) if j!=q] for q in range(28)]
candidate_position=[{j:i for i,j in enumerate(c)} for c in row_candidates]

def blocks(slots):
    return [np.array([pair_index[tuple(sorted((i,j)))] for i in slots[a] for j in slots[b]],dtype=np.int64)
            for a,b in account_pairs]
B0=blocks(slots0); B1=blocks(slots1)
edge_idx={e:i for i,e in enumerate(account_pairs)}
row_edges=np.array([[edge_idx[tuple(sorted((q,j)))] for j in row_candidates[q]] for q in range(28)])

def project_np(k,bb):
    h=[]
    for ix in bb:
        v=k[ix]; mx=v.max()
        h.append(0.5*v.mean()+0.5*(mx+np.log(np.exp(v-mx).mean())))
    return np.array(h)[row_edges]

def project_torch(k,bb):
    h=[]
    for ix in bb:
        v=k[torch.as_tensor(ix)]
        h.append(0.5*v.mean()+0.5*(torch.logsumexp(v,0)-math.log(len(ix))))
    return torch.stack(h)[torch.as_tensor(row_edges)]

def pullback_np(k,bb,alpha):
    g=np.zeros_like(k)
    for (a,b),ix in zip(account_pairs,bb):
        beta=alpha[a,candidate_position[a][b]]+alpha[b,candidate_position[b][a]]
        v=k[ix]; prob=np.exp(v-v.max()); prob/=prob.sum()
        g[ix]+=beta*(0.5/len(ix)+0.5*prob)
    return g

def average_precision(z):
    owners={a:c for c,comp in enumerate(F['components']) for a in comp}
    aps=[]
    for q in range(28):
        order=sorted(range(27),key=lambda j:(-float(z[q,j]),row_candidates[q][j]))
        hits=0; total=0.0; n=sum(owners[j]==owners[q] for j in row_candidates[q])
        for rank,i in enumerate(order,1):
            if owners[row_candidates[q][i]]==owners[q]:
                hits+=1;total+=hits/rank
        aps.append(total/n)
    return np.array(aps)

try:
    # Legal handwritten population; no observed formal data.
    own={a:c for c,comp in enumerate(F['components']) for a in comp}
    valid=True
    for slots in (slots0,slots1,F['view1_swap_17_18']):
        valid &= sorted(r for s in slots for r in s)==list(range(56))
        valid &= all(len(s)==2 and all(own[r//2]==own[a] for r in s) for a,s in enumerate(slots))
    check('legal_28_slot_eight_K2_four_K3_handwritten_assignments',valid)
    # Binary analytic signs and JS bound.
    zp=np.log(np.array(F['binary']['p']))[None,:]; zr=np.log(np.array(F['binary']['r']))[None,:]
    c,g0,g1=independent_grad(zp,zr)
    expected=math.log(2)+0.9*math.log(0.9)+0.1*math.log(0.1)
    check('binary_value_and_derivative',abs(c-expected)<2e-15 and abs(g0[0,0]-.045*math.log(9))<2e-15)
    R['numeric']['binary']={'JS':float(c),'gradient_view0':g0.tolist(),'gradient_view1':g1.tolist()}
    # AP=1 positive swaps: numerical example and existing rank.
    p=np.array(F['AP1_query']['p']); r=np.array(F['AP1_query']['r'])
    cq,gp,gr=independent_grad(np.log(p)[None,:],np.log(r)[None,:])
    p_bal=np.r_[.475,.475,np.full(25,.002)]
    rank_swap=float(-.5*(np.log(p[0])+np.log(p[1])))
    rank_bal=float(-math.log(.475))
    check('AP1_nonzero_JS_and_original_rank_already_prefers_balanced_positives',cq>0 and rank_swap>rank_bal)
    R['numeric']['AP1_query']={'JS':float(cq),'rank_imbalanced':rank_swap,'rank_balanced':rank_bal,
        'rank_excess':rank_swap-rank_bal,'negative_logit_gradient_per_view':float(gp[0,2]),
        'rank_plus_JS_derivative_common_positive_shift':float((1-.95)*(cq-1))}
    # Embed example into a complete symmetric score graph using one shared K.
    kk=np.array(F['K_AP1_nonzero']); za=project_np(kk,B0); zb=project_np(kk,blocks(F['view1_swap_17_18']))
    cc,_=scalar_js(za,zb); ap0=average_precision(za); ap1=average_precision(zb)
    qjs,_=scalar_js(za[16:17,:],zb[16:17,:])
    check('shared_K_legal_full_graph_MAP1_both_views_but_C_positive',np.all(ap0==1) and np.all(ap1==1) and cc>0 and abs(qjs-cq)<1e-14)
    R['numeric']['legal_AP1_graph']={'group_C':float(cc),'query16_JS':float(qjs),'MAP_view0':float(ap0.mean()),'MAP_view1':float(ap1.mean())}
    arrays.update(AP1_z0=za,AP1_z1=zb)
    # Correct and zero; identical wrong distributions and uniform are also zero.
    kinv=np.array(F['K_correct_invariant']); zi0=project_np(kinv,B0); zi1=project_np(kinv,B1)
    ci,_=scalar_js(zi0,zi1)
    check('correct_full_graph_C_zero',abs(ci)<2e-13 and np.all(average_precision(zi0)==1) and np.all(average_precision(zi1)==1))
    bad=np.r_[0.01,0.99][None,:]
    cbad,_=scalar_js(np.log(bad),np.log(bad)); cunif,_=scalar_js(np.zeros((28,27)),np.zeros((28,27)))
    check('identically_wrong_or_uniform_C_zero',abs(cbad)<1e-14 and abs(cunif)<1e-14)
    # An explicit C-only adverse direction; NOT an optimizer/training or total-J result.
    f=F['correct_vs_wrong']; x0=np.log(np.array(f['p']))[None,:]; x1=np.log(np.array(f['r']))[None,:]
    cx,dx0,dx1=independent_grad(x0,x1)
    x0_after=x0-f['algebraic_logit_step']*dx0
    check('C_only_can_move_correct_student_across_margin_when_other_student_wrong',x0[0,0]>x0[0,1] and x0_after[0,0]<x0_after[0,1])
    R['numeric']['C_only_adverse_direction']={'margin_before':float(x0[0,0]-x0[0,1]),'margin_after':float(x0_after[0,0]-x0_after[0,1]),
        'algebraic_step':f['algebraic_logit_step'],'not_full_objective':True}
    # Full shared table derivative comparison, separately for both branches.
    k_np=np.array(F['K_general']); k=torch.tensor(k_np,requires_grad=True)
    z0=project_torch(k,B0); z1=project_torch(k,B1)
    loss=tensor_js(z0,z1)
    gz0,gz1=torch.autograd.grad(loss,(z0,z1),retain_graph=True)
    kg0=torch.autograd.grad(z0,k,grad_outputs=gz0,retain_graph=True)[0]
    kg1=torch.autograd.grad(z1,k,grad_outputs=gz1,retain_graph=True)[0]
    kg=torch.autograd.grad(loss,k,retain_graph=True)[0]
    z0_np=z0.detach().numpy(); z1_np=z1.detach().numpy()
    cref,a0,a1=independent_grad(z0_np,z1_np)
    kref0=pullback_np(k_np,B0,a0); kref1=pullback_np(k_np,B1,a1)
    atol=F['tolerances']['analytical_autograd_atol']
    error0=float(np.max(np.abs(kg0.numpy()-kref0))); error1=float(np.max(np.abs(kg1.numpy()-kref1)))
    errork=float(np.max(np.abs(kg.numpy()-kref0-kref1)))
    check('per_view_query_gradients_28_mean_and_zero_row_sums',np.max(np.abs(a0-gz0.numpy()))<atol and np.max(np.abs(a1-gz1.numpy()))<atol and np.max(np.abs(a0.sum(-1)))<atol)
    check('shared_K_both_independent_branch_pullbacks_and_sum',max(error0,error1,errork)<atol and np.linalg.norm(kref0)>0 and np.linalg.norm(kref1)>0)
    eps=F['tolerances']['finite_difference_step']; fd=[]
    for i in F['finite_difference_indices']:
        kp=k_np.copy();km=k_np.copy();kp[i]+=eps;km[i]-=eps
        cp,_=scalar_js(project_np(kp,B0),project_np(kp,B1)); cm,_=scalar_js(project_np(km,B0),project_np(km,B1))
        fd.append({'index':i,'finite_difference':float((cp-cm)/(2*eps)),'analytic':float(kref0[i]+kref1[i])})
    fd_error=max(abs(x['finite_difference']-x['analytic']) for x in fd)
    check('shared_K_independent_finite_difference',fd_error<F['tolerances']['finite_difference_atol'],{'max_abs_error':fd_error})
    arrays.update(general_z0=z0_np,general_z1=z1_np,alpha_view0=a0,alpha_view1=a1,
        K_gradient_view0=kref0,K_gradient_view1=kref1,K_gradient_sum=kref0+kref1,
        K_gradient_autograd=kg.numpy())
    R['numeric']['shared_K']={'C':float(cref),'query_gradient_max_error_view0':float(np.max(np.abs(a0-gz0.numpy()))),
       'K_gradient_max_error_view0':error0,'K_gradient_max_error_view1':error1,'K_gradient_max_error_sum':errork,
       'norm_branch0':float(np.linalg.norm(kref0)),'norm_branch1':float(np.linalg.norm(kref1)),
       'norm_sum':float(np.linalg.norm(kref0+kref1)),'finite_differences':fd}
    # Mutation: full student branch detach is detectable.
    ld=tensor_js(z0,z1,stop_view1=True)
    kd=torch.autograd.grad(ld,k,retain_graph=True)[0].numpy()
    check('student_view1_detach_loses_its_K_gradient',np.max(np.abs(kd-kref0))<atol and np.max(np.abs(kd-kg.numpy()))>1e-7)
    # Crucial exception: m-only detach leaves all first derivatives unchanged.
    lm=tensor_js(z0,z1,stop_m=True)
    m0,m1=torch.autograd.grad(lm,(z0,z1),retain_graph=True)
    km=torch.autograd.grad(lm,k,retain_graph=True)[0].numpy()
    m_err=float(np.max(np.abs(km-kg.numpy())))
    check('m_only_detach_first_order_equivalence',abs(float(lm.detach()-loss.detach()))<atol and m_err<atol and torch.max(torch.abs(m0-gz0))<atol and torch.max(torch.abs(m1-gz1))<atol)
    # Autodifferentiated Hessian, not finite differences through a refreshed detach.
    v=torch.zeros(4,requires_grad=True)
    hh_live=torch.autograd.functional.hessian(lambda t:tensor_js(t[:2][None,:],t[2:][None,:]),v).numpy()
    hh_stop=torch.autograd.functional.hessian(lambda t:tensor_js(t[:2][None,:],t[2:][None,:],stop_m=True),v).numpy()
    S=np.array([[.25,-.25],[-.25,.25]])
    live_expected=np.block([[.25*S,-.25*S],[-.25*S,.25*S]])
    stop_expected=np.block([[.5*S,np.zeros((2,2))],[np.zeros((2,2)),.5*S]])
    check('m_detach_autodiff_second_order_or_graph_not_first_order_distinguishes',np.max(np.abs(hh_live-live_expected))<atol and np.max(np.abs(hh_stop-stop_expected))<atol)
    R['numeric']['detach']={'m_only_max_K_first_gradient_difference':m_err,
        'student1_detach_max_K_gradient_difference':float(np.max(np.abs(kd-kg.numpy()))),
        'live_autodiff_Hessian_at_uniform':hh_live.tolist(),'m_detached_autodiff_Hessian_at_uniform':hh_stop.tolist()}
    arrays.update(K_gradient_student1_detached=kd,K_gradient_m_detached=km)
    check('omission_or_wrong_28_normalization_or_history_weight_is_numerically_detectable',
       np.linalg.norm(kref0+kref1)>1e-7 and np.max(np.abs(28*(kref0+kref1)-(kref0+kref1)))>1e-7 and
       np.max(np.abs(.1*(kref0+kref1)-(kref0+kref1)))>1e-7)
    # Local vs globally realizable offset invariance for symmetric account edges.
    off0=z0_np+np.arange(28)[:,None]*.25;off1=z1_np-np.arange(28)[:,None]*.125
    co,ao0,ao1=independent_grad(off0,off1)
    check('independent_query_row_offset_value_and_gradient_invariance',abs(co-cref)<atol and np.max(np.abs(ao0-a0))<atol and np.max(np.abs(ao1-a1))<atol)
    allshift=z0_np+2.0
    star=z0_np.copy()
    for q in range(28):
        for idx,j in enumerate(row_candidates[q]):
            if q==0 or j==0:star[q,idx]+=.75
    cglob,_=scalar_js(z0_np,allshift); cstar,_=scalar_js(z0_np,star)
    cstar_q0,_=scalar_js(z0_np[0:1],star[0:1])
    check('one_global_symmetric_offset_zero_but_single_query_star_not_group_invariant',abs(cglob)<atol and abs(cstar_q0)<atol and cstar>1e-6)
    R['numeric']['offset']={'global_C':float(cglob),'star_C':float(cstar),'star_query0_C':float(cstar_q0)}
    # Stable finite logarithms with probabilities underflowing; compare deliberate bad diagonal form.
    ex0=torch.zeros((28,27)); ex1=torch.zeros((28,27));ex0[:,0]=1000;ex0[:,1]=-1000;ex1[:,0]=-1000;ex1[:,1]=1000
    extreme=tensor_js(ex0,ex1)
    full=torch.zeros((28,28));full.fill_diagonal_(-torch.inf)
    logfull=torch.log_softmax(full,-1)
    logmfull=torch.logaddexp(logfull,logfull)-math.log(2)
    naive=(logfull.exp()*(logfull-logmfull)).sum()
    check('FP64_finite_extreme_log_space_and_rejected_diagonal_NaN',torch.isfinite(extreme) and abs(float(extreme)-math.log(2))<atol and torch.isnan(naive))
    R['numeric']['stable_extreme']={'JS':float(extreme),'naive_unremoved_diagonal_produces_NaN':bool(torch.isnan(naive))}
    R['status']='PASS_REVIEWER_ISOLATED_MATHEMATICS_ONLY'
except BaseException as exc:
    R['status']='FAILED_REVIEWER_CHECK_ATTEMPT_RETAINED'
    R['error']={'type':type(exc).__name__,'message':str(exc),'traceback':traceback.format_exc()}
    traceback.print_exc()
finally:
    (ROOT/'results.json').write_text(json.dumps(R,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    np.savez_compressed(ROOT/'gradient_evidence.npz',**arrays)
    print(json.dumps({'status':R['status'],'checks_completed':len(R['checks'])},ensure_ascii=False),flush=True)
if R['status']!='PASS_REVIEWER_ISOLATED_MATHEMATICS_ONLY':sys.exit(1)
