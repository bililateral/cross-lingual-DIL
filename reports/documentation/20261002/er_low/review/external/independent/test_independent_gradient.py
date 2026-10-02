"""Independent scalar derivatives, explicit gradient mixing, clipping and AdamW.

Only handwritten groups and a tiny CPU encoder are used. The model/optimizer
fixture makes one real warm update and explicitly rebases counters; that is NOT
288 warm updates. No formal data/model loading or remote connections occur.
"""
from __future__ import annotations
import copy,itertools,json,math,os
from pathlib import Path
import unittest
from unittest import mock
import numpy as np
import torch
import step28_er_weight as m
import test_step28_bge_continual_contracts as fixture
E=Path(os.environ.get('ER_AUDIT_EVIDENCE','/mnt/data/er_low_audit/evidence'))/'gradient'
E.mkdir(parents=True,exist_ok=True)
PAIRS=list(itertools.combinations(range(28),2))


def sigmoid(x):
    return np.exp(-np.logaddexp(0.,-np.asarray(x,dtype=np.float64)))


def scalar_and_derivative(scores, labels):
    """Explicit undirected-edge traversal; no submitted objectives/gradients."""
    z=np.asarray(scores,dtype=np.float64);y=np.asarray(labels,dtype=np.float64)
    assert z.shape==y.shape==(378,)
    bce=float(np.mean(np.logaddexp(0.,z)-y*z));gb=(sigmoid(z)-y)/378
    gr=np.zeros(378);gh=np.zeros(378);rank=0.;hard=0.
    for query in range(28):
        edges=[(k,v if u==query else u) for k,(u,v) in enumerate(PAIRS) if query in (u,v)]
        ids=[k for k,_ in edges];pos=[k for k in ids if y[k]==1]
        neg=sorted(((k,other) for k,other in edges if y[k]==0),key=lambda t:(-z[t[0]],t[1]))[:5]
        mx=float(max(z[ids]));ex=np.exp(z[ids]-mx);den=float(ex.sum())
        rank+=(mx+math.log(den)-float(z[pos].mean()))/28
        gr[ids]+=ex/den/28;gr[pos]-=1./(28*len(pos))
        scale=1./(28*len(pos)*5)
        for pi in pos:
            for ni,_ in neg:
                d=z[ni]-z[pi];hard+=float(np.logaddexp(0.,d))*scale
                value=float(sigmoid(d))*scale;gh[ni]+=value;gh[pi]-=value
    return dict(bce=bce,rank=rank,hard=hard,total=bce+rank+.5*hard),dict(bce=gb,rank=gr,hard=gh,total=gb+gr+.5*gh)


def prepared(count=288):
    model=fixture.tiny_model();c=fixture.config();old=fixture.handmade_group('audit_old',1);cur=fixture.handmade_group('audit_current',2)
    opt=m.core.make_optimizer(model,c);fixture.toy_prior_step(model,opt,old,c,count)
    return model,opt,c,cur,old


def reference_components(model,c,cur,old,seeds):
    model.train();params=list(model.parameters());grads=[];losses=[];scores=[]
    for group,seed in zip((cur,old),seeds):
        torch.manual_seed(seed);z=m.base.logits(model,group,c,'split_rank');a,b=scalar_and_derivative(z.detach().numpy(),group.labels)
        grad=torch.autograd.grad(z,params,grad_outputs=torch.as_tensor(b['total'],dtype=z.dtype))
        grads.append([g.detach().numpy().astype(np.float64) for g in grad]);losses.append(a);scores.append(z.detach().numpy())
    return grads,losses,scores


def verify_update(weight,clip,count=288,step=1):
    model,opt,c,cur,old=prepared(count);c=copy.deepcopy(c);c['optimizer']['clip_norm']=clip
    params=list(model.parameters());before=[x.detach().numpy().astype(np.float64).copy() for x in params]
    before_states=[{k:(v.detach().numpy().astype(np.float64).copy() if isinstance(v,torch.Tensor) else copy.deepcopy(v)) for k,v in opt.state[x].items()} for x in params]
    other=copy.deepcopy(model);components,losses,scores=reference_components(other,c,cur,old,(1401,1402))
    combined=[a+weight*b for a,b in zip(*components)]
    norm=math.sqrt(sum(float(np.sum(g*g)) for g in combined));scale=min(1.,clip/(norm+1e-6));after_clip=[g*scale for g in combined]
    rates=(1e-5*(step/29 if step<=29 else (288-step)/259),.001)
    meta=[]
    for group,rate in zip(opt.param_groups,rates):
        for _ in group['params']:meta.append((rate,group['weight_decay'],group['betas'],group['eps']))
    expected=[]
    for p0,st,g,(lr,wd,(b1,b2),eps) in zip(before,before_states,after_clip,meta):
        t=float(st['step'])+1;mu=b1*st['exp_avg']+(1-b1)*g;v=b2*st['exp_avg_sq']+(1-b2)*(g*g)
        value=p0*(1-lr*wd)-(lr/(1-b1**t))*mu/(np.sqrt(v)/math.sqrt(1-b2**t)+eps)
        expected.append((value,mu,v,t))
    capture={'backward':[],'preclip':[]};original_backward=torch.Tensor.backward;original_clip=torch.nn.utils.clip_grad_norm_
    def backward(tensor,*args,**kwargs):
        capture['backward'].append(float(tensor.detach()));return original_backward(tensor,*args,**kwargs)
    def clipping(parameters,*args,**kwargs):
        seq=list(parameters);capture['preclip']=[p.grad.detach().numpy().astype(np.float64).copy() for p in seq]
        return original_clip(seq,*args,**kwargs)
    with mock.patch.object(torch.Tensor,'backward',new=backward),mock.patch.object(torch.nn.utils,'clip_grad_norm_',side_effect=clipping) as cl,mock.patch.object(opt,'step',wraps=opt.step) as st,mock.patch.object(opt,'zero_grad',wraps=opt.zero_grad) as ze:
        observed=m.update(model,opt,cur,old,c,weight,2,step,1401,1402,observe=True)
        calls=dict(backward=len(capture['backward']),clip=cl.call_count,step=st.call_count,zero_grad=ze.call_count)
    assert calls==dict(backward=2,clip=1,step=1,zero_grad=1),calls
    np.testing.assert_allclose(capture['backward'],[losses[0]['total'],weight*losses[1]['total']],rtol=2e-6,atol=2e-6)
    np.testing.assert_allclose(observed['gradient_norm'],norm,rtol=3e-6,atol=2e-7)
    maxima={k:0. for k in ['unclipped_gradient','clipped_gradient','parameter','adam_exp_avg','adam_exp_avg_sq']};payload={}
    for i,(p,exp,pre,g) in enumerate(zip(params,expected,capture['preclip'],after_clip)):
        actual=[pre,p.grad.detach().numpy(),p.detach().numpy(),opt.state[p]['exp_avg'].numpy(),opt.state[p]['exp_avg_sq'].numpy()]
        wanted=[combined[i],g,exp[0],exp[1],exp[2]]
        for key,a,b in zip(maxima,actual,wanted):
            np.testing.assert_allclose(a,b,rtol=3e-5,atol=2e-7)
            maxima[key]=max(maxima[key],float(np.max(np.abs(a-b))))
            payload[f'p{i}_{key}_actual']=np.asarray(a);payload[f'p{i}_{key}_reference']=np.asarray(b)
        assert float(opt.state[p]['step'])==exp[3]
    key=f'weight{weight}_clip{clip}_step{step}'
    np.savez_compressed(E/(key+'.npz'),**payload)
    record=dict(case=key,history_weight=weight,clip_limit=clip,independent_unclipped_norm=norm,independent_clip_multiplier=scale,calls=calls,independent_objectives={'current':losses[0],'history':losses[1]},independent_total=losses[0]['total']+weight*losses[1]['total'],observed=observed,max_absolute_differences=maxima,actual_warm_updates=1,rebased_prior_adam_counter=count,actual_new_updates=1,formal_updates=0)
    (E/(key+'.json')).write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record,ensure_ascii=False))
    return capture,record


class IndependentGradient(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1);torch.use_deterministic_algorithms(True)

    def test_scalar_analytic_derivatives_against_autograd_and_finite_difference(self):
        labels=fixture.handmade_group('analytic_only').labels;z=np.random.default_rng(813).normal(0,.7,378)
        values,derivative=scalar_and_derivative(z,labels);tensor=torch.tensor(z,dtype=torch.float64,requires_grad=True)
        production=m.parent.ranking.objectives(tensor,torch.tensor(labels,dtype=torch.float64),.5)
        maximum=0.;count=0
        for term in ['bce','rank','hard','total']:
            np.testing.assert_allclose(float(production[term].detach()),values[term],rtol=0,atol=2e-14)
            actual=torch.autograd.grad(production[term],tensor,retain_graph=True)[0].numpy()
            np.testing.assert_allclose(actual,derivative[term],rtol=0,atol=2e-15)
            for k in range(0,378,23):
                plus=z.copy();minus=z.copy();plus[k]+=1e-5;minus[k]-=1e-5
                finite=(scalar_and_derivative(plus,labels)[0][term]-scalar_and_derivative(minus,labels)[0][term])/2e-5
                maximum=max(maximum,abs(finite-derivative[term][k]));count+=1
                self.assertLess(abs(finite-derivative[term][k]),2e-9)
        row=dict(scalar_losses=values,analytic_derivative_elements_compared=4*378,finite_difference_checks=count,finite_difference_max_absolute_difference=maximum)
        (E/'analytic_reference_self_check.json').write_text(json.dumps(row,indent=2)+'\n');print(json.dumps(row))

    def test_tenth_unclipped_gradient_parameter_and_manual_adam(self):
        _,r=verify_update(.1,1e6);self.assertEqual(r['independent_clip_multiplier'],1.)

    def test_tenth_active_clip_gradient_parameter_and_manual_adam(self):
        _,r=verify_update(.1,.01);self.assertLess(r['independent_clip_multiplier'],1.)

    def test_tenth_last_step_zero_encoder_lr_continues_adam(self):
        _,r=verify_update(.1,1.,575,288);self.assertEqual(r['observed']['adam_step'],576)
        self.assertFalse(r['observed']['modules']['encoder']['parameters_changed']);self.assertTrue(r['observed']['modules']['head']['parameters_changed'])

    def test_current_equal_history_tenth_quarter_ratio(self):
        a,opt,c,cur,old=prepared();b=copy.deepcopy(a);opb=m.core.make_optimizer(b,c);opb.load_state_dict(copy.deepcopy(opt.state_dict()))
        records=[];forward=[]
        for model,optimizer,weight in [(a,opt,.1),(b,opb,.25)]:
            hits={name:[] for name,_ in model.named_parameters()};out=[];hooks=[]
            for name,p in model.named_parameters():hooks.append(p.register_hook(lambda g,n=name:hits[n].append(g.detach().clone())))
            hooks.append(model.head.register_forward_hook(lambda mod,args,value:out.append(value.detach().clone())))
            try:m.update(model,optimizer,cur,old,c,weight,2,1,1501,1502)
            finally:
                for h in hooks:h.remove()
            records.append(hits);forward.append(out)
        maximum=0.;elements=0
        for name,pa in records[0].items():
            pb=records[1][name];self.assertEqual(len(pa),2);self.assertEqual(len(pb),2)
            torch.testing.assert_close(pa[0],pb[0],rtol=0,atol=0)
            torch.testing.assert_close(2.5*pa[1],pb[1],rtol=2e-5,atol=1e-8)
            maximum=max(maximum,float((2.5*pa[1]-pb[1]).abs().max()));elements+=pa[1].numel()
        for fa,fb in zip(*forward):torch.testing.assert_close(fa,fb,rtol=0,atol=0)
        row=dict(current_gradients_exact=True,current_history_forwards_exact=True,expected_quarter_over_tenth_ratio=2.5,history_max_absolute_difference=maximum,parameter_gradient_elements=elements,formal_updates=0)
        (E/'tenth_quarter_ratio.json').write_text(json.dumps(row,indent=2)+'\n');print(json.dumps(row))


if __name__=='__main__':unittest.main(verbosity=2)
