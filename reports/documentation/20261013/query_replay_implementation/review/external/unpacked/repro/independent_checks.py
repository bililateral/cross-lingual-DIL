"""External-review handwritten calculations ONLY. No formal entry/data/BGE/Memory load.
Run: python independent_checks.py --source-root PATH --out PATH
Creates tiny modules in RAM, never loads model weights or project Memory.
"""
from __future__ import annotations
import argparse, copy, hashlib, itertools, json, math, os, platform, sys, tempfile, time, traceback
from pathlib import Path
from fractions import Fraction
from unittest.mock import patch
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
os.environ['OMP_NUM_THREADS']=os.environ['MKL_NUM_THREADS']=os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['TOKENIZERS_PARALLELISM']='false'
os.environ['PYTHONDONTWRITEBYTECODE']='1'
import numpy as np
import torch

def dump(p, x):
    p.write_text(json.dumps(x, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--source-root',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);ap.add_argument('--only',default=None)
    args=ap.parse_args();args.out.mkdir(parents=True,exist_ok=True)
    sys.path.insert(0,str(args.source_root/'scripts'))
    import step28_query_replay as m
    import step28_query_replay_run as runner
    torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    components=[[2*k,2*k+1] for k in range(8)]+[list(range(16+3*k,19+3*k)) for k in range(4)]
    controller={a:k for k,g in enumerate(components) for a in g}
    labels=tuple(int(controller[a]==controller[b]) for a,b in itertools.combinations(range(28),2))
    def group(uid, counts):
        return m.data.Group(uid,tuple(f'{uid}_s{a:02}' for a in range(28)),
            tuple(tuple((f'{uid}_a{a:02}_r{r:02}',f'人工标题{uid}号{a}条{r}',f'人工描述{uid}组{a}项{r}') for r in range(n)) for a,n in enumerate(counts)),labels)
    g=group('external_math',[2+a%3 for a in range(28)])
    current=group('external_current',[2]*28);history=group('external_history',[2]*28)
    dump(args.out/'inputs.json',{'scope':'External handwritten inputs, no project data','groups':[g.payload(),current.payload(),history.payload()],
        'components':components,'table_recipe':'K=sin(arange(Npairs)/7)+.2*cos(arange(Npairs)/13); T=.7*cos(arange(Npairs)/11)',
        'target_update_recipe':'3*sin(arange(1540)/7) float32','stage':2,'step':49,'model_seed':42017})
    def idx(group):
        ends=np.cumsum([0]+[len(v) for v in group.items]);return [list(range(int(ends[a]),int(ends[a+1]))) for a in range(28)]
    def matrix(table):
        n=int((1+math.sqrt(1+8*table.numel()))/2);i,j=torch.triu_indices(n,n,1)
        x=table.new_zeros((n,n));x[i,j]=table;x[j,i]=table;return x
    def pooled(table, inds):
        x=matrix(table)
        return torch.stack([torch.stack([table.sum()*0 if q==j else x[inds[q]][:,inds[j]].mean() for j in range(28)]) for q in range(28)])
    def directed(table, group, rows):
        x=matrix(table);natural=idx(group)
        return torch.stack([torch.stack([table.sum()*0 if q==j else x[natural[q]][:,rows[q][j]].mean() for j in range(28)]) for q in range(28)])
    def rowloss(scores):
        bce=[];rank=[];hard=[]
        for q in range(28):
            js=[j for j in range(28) if j!=q];pos=[j for j in js if controller[q]==controller[j]];neg=[j for j in js if j not in pos]
            top=sorted(neg,key=lambda j:(-float(scores[q,j].detach()),j))[:5]
            bce.append(torch.stack([torch.nn.functional.softplus(scores[q,j])-int(j in pos)*scores[q,j] for j in js]).sum())
            rank.append(torch.logsumexp(scores[q,js],dim=0)-scores[q,pos].mean())
            hard.append(torch.stack([torch.nn.functional.softplus(scores[q,n]-scores[q,p]) for p in pos for n in top]).mean())
        return torch.stack(bce).sum()/756+torch.stack(rank).sum()/28+.5*torch.stack(hard).sum()/28
    def response(table, group, target):
        x=matrix(table)-matrix(torch.as_tensor(target,dtype=table.dtype).detach());inds=idx(group);terms=[]
        for q in range(28):
            for comp in components:
                accounts=[a for a in comp if a!=q];M=sum(len(inds[a]) for a in accounts)
                b=sum(Fraction(M-len(inds[a]),len(inds[a])*M*(M-1)) for a in accounts)
                for a in accounts:
                    e=x[inds[q]][:,inds[a]].mean(0);e=e-e.mean();terms.append(float(b)*e.square().sum())
        return torch.stack(terms).sum()/756
    tests=[];start=time.monotonic()
    def test(name, fn):
        if args.only is not None and name != args.only: return
        tick=time.monotonic()
        try:
            detail=fn();tests.append({'name':name,'status':'PASS','seconds':time.monotonic()-tick,'detail':detail});print('PASS',name,json.dumps(detail,ensure_ascii=False),flush=True)
        except Exception:
            detail=traceback.format_exc();tests.append({'name':name,'status':'FAIL','seconds':time.monotonic()-tick,'traceback':detail});print('FAIL',name,detail,flush=True)
    n=sum(map(len,g.items));k=n*(n-1)//2
    table=(torch.sin(torch.arange(k,dtype=torch.float64)/7)+.2*torch.cos(torch.arange(k,dtype=torch.float64)/13)).requires_grad_()
    target=.7*torch.cos(torch.arange(k,dtype=torch.float64)/11)
    rows=m.candidates(g,2,49,'query_replay')
    def sampling():
        natural=idx(g);owner={r:a for a,rs in enumerate(natural) for r in rs};before=runner.diag.rng_state()
        bootstrap=m.candidates(g,2,49,'within_account_bootstrap');again=m.candidates(g,2,49,'query_replay')
        assert before==runner.diag.rng_state() and again==rows
        repeats=0
        for q in range(28):
            assert not rows[q][q]
            assert sorted(sum(rows[q],[]))==sorted(r for a in range(28) if a!=q for r in natural[a])
            for j in range(28):
                if q==j:continue
                assert len(rows[q][j])==len(natural[j])
                assert all(controller[owner[r]]==controller[j] and owner[r]!=q for r in rows[q][j])
                assert len(bootstrap[q][j])==len(natural[j]) and set(bootstrap[q][j])<=set(natural[j])
                repeats+=len(set(bootstrap[q][j]))<len(bootstrap[q][j])
        assert repeats>0
        dump(args.out/'candidate_rows.json',{'query_replay':rows,'within_account_bootstrap':bootstrap})
        return {'queries':28,'directed_candidates':756,'positives':2*sum(labels),'records':n,'bootstrap_slots_with_repeats':repeats,'global_rng_unchanged':True}
    test('source_exclusion_and_multisets',sampling)
    def directed_test():
        actual=m.directed_scores(table,g,rows);expected=directed(table,g,rows)
        torch.testing.assert_close(actual,expected,atol=1e-10,rtol=1e-9)
        a=m.row_objective(actual,g);b=rowloss(expected)
        ga=torch.autograd.grad(a,table,retain_graph=True)[0];gb=torch.autograd.grad(b,table,retain_graph=True)[0]
        torch.testing.assert_close(a,b,atol=1e-10,rtol=1e-9);torch.testing.assert_close(ga,gb,atol=1e-10,rtol=1e-9)
        triangular=torch.triu(actual,1);wrong=triangular+triangular.T
        wrongloss=m.row_objective(wrong,g);assert abs(float(a.detach()-wrongloss.detach()))>1e-8
        ties=torch.zeros((28,28),dtype=torch.float64,requires_grad=True)
        tg=torch.autograd.grad(m.row_objective(ties,g),ties,retain_graph=True)[0]
        te=torch.autograd.grad(rowloss(ties),ties)[0]
        torch.testing.assert_close(tg,te,atol=1e-10,rtol=1e-9)
        return {'score_max_error':float((actual-expected).abs().max().detach()),'loss_error':float((a-b).abs().detach()),
                'gradient_max_error':float((ga-gb).abs().max()),'asymmetry':float((actual-actual.T).abs().max().detach()),
                'wrong_triangle_loss_delta':float((wrongloss-a).detach()),'tie_gradient_max_error':float((tg-te).abs().max())}
    test('directed_loss_gradients_and_wrong_triangle',directed_test)
    def response_test():
        actual_R=m.response_loss(table,g,target);literal_R=response(table,g,target)
        ga=torch.autograd.grad(actual_R,table,retain_graph=True)[0];gb=torch.autograd.grad(literal_R,table,retain_graph=True)[0]
        torch.testing.assert_close(actual_R,literal_R,atol=1e-10,rtol=1e-9);torch.testing.assert_close(ga,gb,atol=1e-10,rtol=1e-9)
        inds=idx(g);owner={r:a for a,rs in enumerate(inds) for r in rs};err=(matrix(table)-matrix(target)).detach().numpy()
        E={};weights={}
        for q in range(28):
            for comp in components:
                ac=[a for a in comp if a!=q];M=sum(len(inds[a]) for a in ac)
                w=sum(Fraction(M-len(inds[a]),len(inds[a])*M*(M-1)) for a in ac)
                for a in ac:
                    e=err[np.ix_(inds[q],inds[a])].mean(0);e=e-e.mean()
                    for r,v in zip(inds[a],e):E[q,r]=v;weights[q,r]=float(w)
        manual=[];one_side=[]
        for s,r in itertools.combinations(range(n),2):
            q,a=owner[s],owner[r]
            if q==a:manual.append(0.);one_side.append(0.);continue
            left=2/756*weights[q,r]*E[q,r]/len(inds[q]);right=2/756*weights[a,s]*E[a,s]/len(inds[a])
            manual.append(left+right);one_side.append(left)
        torch.testing.assert_close(ga,torch.tensor(manual),atol=1e-10,rtol=1e-9)
        missing=max(abs(x-y) for x,y in zip(manual,one_side));assert missing>1e-8
        wrong=m.response_loss(table,g,np.zeros(k));wrongg=torch.autograd.grad(wrong,table,retain_graph=True)[0];assert (wrongg-ga).abs().max()>1e-8
        table32=table.detach().float().requires_grad_();v32=m.response_loss(table32,g,target.float());torch.testing.assert_close(v32.double(),actual_R.detach(),atol=1e-6,rtol=1e-5)
        g32=torch.autograd.grad(v32,table32)[0];torch.testing.assert_close(g32.double(),ga,atol=1e-6,rtol=1e-5)
        finite=[]
        for j in np.linspace(0,k-1,19,dtype=int):
            p=table.detach().clone();mi=p.clone();p[j]+=1e-5;mi[j]-=1e-5
            est=(float(m.response_loss(p,g,target))-float(m.response_loss(mi,g,target)))/2e-5
            finite.append({'index':int(j),'manual':manual[j],'finite_difference':est,'error':abs(est-manual[j])})
        dump(args.out/'gradient_details.json',{'all_manual':manual,'all_autograd':ga.tolist(),'finite_differences':finite})
        return {'R':float(actual_R.detach()),'literal_R_error':float((actual_R-literal_R).abs().detach()),'gradient_max_error':float((ga-gb).abs().max()),
                'manual_dual_role_max_error':float(np.max(np.abs(ga.numpy()-manual))),'missing_role_max_error':missing,
                'wrong_teacher_gradient_max_delta':float((wrongg-ga).abs().max()),'finite_difference_max_error':max(x['error'] for x in finite),'fp32_gradient_error':float((g32.double()-ga).abs().max()),'fp32_value_error':abs(float(v32)-float(actual_R.detach()))}
    test('finite_population_R_and_symmetric_two_roles',response_test)
    def enumeration():
        e=[Fraction(-1),Fraction(1),Fraction(-2),Fraction(1,2),Fraction(3,2)];vals=[]
        for A in itertools.combinations(range(5),2):
            B=[r for r in range(5) if r not in A];v=(sum(e[r] for r in A)/2)**2+(sum(e[r] for r in B)/3)**2
            vals.append({'slot0':list(A),'slot1':B,'squared_sum_fraction':str(v)})
        empirical=sum(Fraction(v['squared_sum_fraction']) for v in vals)/len(vals)
        exact=(Fraction(5-2,2*5*4)+Fraction(5-3,3*5*4))*sum(x*x for x in e)
        assert empirical==exact
        dump(args.out/'enumeration.json',{'population':[str(x) for x in e],'rows':vals,'mean':str(empirical),'formula':str(exact)})
        return {'joint_partitions':len(vals),'mean_fraction':str(empirical),'exact_fraction':str(exact)}
    test('unequal_slot_joint_exact_enumeration',enumeration)
    def bounds_test():
        group=current;N=56;K=1540;zero=torch.zeros(K,dtype=torch.float64);pairs=list(itertools.combinations(range(N),2));lookup={p:i for i,p in enumerate(pairs)}
        def changed(block,qa,aa):
            val=zero.clone()
            for i in range(2):
                for j in range(2):val[lookup[tuple(sorted((2*qa+i,2*aa+j)))]]=block[i][j]
            return val
        W=changed([[1,-1],[-1,1]],0,2);K2=changed([[3,1],[-1,-3]],0,1)
        records={}
        for name,v in [('W',W),('K2',K2)]:
            for arm in ['Q','C_q']:
                a=m.objective(v,group,23,arm,zero.numpy(),stage=2,step=49)[0];b=m.objective(zero,group,23,arm,zero.numpy(),stage=2,step=49)[0]
                assert torch.equal(a,b);records[name+'_'+arm]=float(a-b)
        t=torch.tensor([-1.,1.]*28,dtype=torch.float64);bias=torch.stack([t[i]+t[j] for i,j in pairs]).requires_grad_()
        loss=m.objective(bias,group,23,'C_q',zero.numpy(),stage=2,step=49)[0];grad=torch.autograd.grad(loss,bias)[0]
        same=torch.tensor([i//2==j//2 for i,j in pairs]);assert grad[same].abs().sum()==0
        R=float(m.response_loss(bias,group,zero).detach());assert R>0 and m.response_loss(bias,group,bias).item()==0
        records.update(query_independent_bias_R=R,same_origin_gradient_max=float(grad[same].abs().max()),same_table_R=0.)
        return records
    test('blind_spaces_bias_and_source_gradient',bounds_test)
    class Encoder(torch.nn.Module):
        def __init__(self):
            super().__init__();self.embed=torch.nn.Embedding(101,5);self.linear=torch.nn.Linear(5,5);self.dropout=torch.nn.Dropout(.2)
        def tokenizer(self,texts,**kwargs):
            return {'input_ids':torch.tensor([[v%101 for v in hashlib.sha256(t.encode()).digest()[:6]] for t in texts])}
        def forward(self,batch):
            z=self.linear(self.embed(batch['input_ids'])).tanh();return {'sentence_embedding':self.dropout(z).mean(1)}
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__();self.encoder=Encoder();self.head=torch.nn.Sequential(torch.nn.Linear(20,7),torch.nn.ReLU(),torch.nn.Linear(7,1))
    def newmodel():
        torch.manual_seed(42017);return Model()
    def optimizer(model):
        return torch.optim.AdamW([{'params':model.encoder.parameters(),'lr':1e-5,'weight_decay':.01},{'params':model.head.parameters(),'lr':.001,'weight_decay':0.}],betas=(.9,.999),eps=1e-8,foreach=False)
    c={'input':{'microbatch':4,'token_budget':256}}
    target2=(3*np.sin(np.arange(1540)/7)).astype(np.float32)
    def live_table(model,group,role):
        torch.manual_seed(m.data.seed_for(20260918,'ABC',2,49,role,'dropout'))
        texts=[row[1] for x in group.items for row in x]+[row[2] for x in group.items for row in x]
        vectors=[]
        for start in range(0,len(texts),4):vectors.append(model.encoder(model.encoder.tokenizer(texts[start:start+4]))['sentence_embedding'].float())
        channels=torch.nn.functional.normalize(torch.cat(vectors),dim=1);z=torch.cat(channels.chunk(2),dim=1)/math.sqrt(2)
        i,j=torch.triu_indices(len(z),len(z),1);u,v=z[i],z[j]
        return model.head(torch.cat(((u-v).abs(),u*v),1)).flatten()
    def update_test():
        actual=newmodel();direct=copy.deepcopy(actual);oa=optimizer(actual);ob=optimizer(direct)
        before=copy.deepcopy(actual.state_dict());record=m.update(actual,oa,current,history,target2,c,'C_q',2,49)
        direct.train();ob.zero_grad(set_to_none=True);losses=[]
        for role,group,T in [('current',current,None),('history',history,target2)]:
            tab=live_table(direct,group,role);natural=pooled(tab,idx(group));B0=rowloss(natural)
            if T is None:
                seed=m.data.seed_for(20260918,'ABC',2,49,role,group.uid,'regroup');slots=m.record.assignment(group,seed)
                B1=rowloss(pooled(tab,[np.flatnonzero(slots==a).tolist() for a in range(28)]));loss=(B0+B1)/2
            else:
                Q=directed(tab,group,m.candidates(group,2,49,'query_replay'));old=pooled(torch.from_numpy(T),idx(group));i,j=torch.triu_indices(28,28,1)
                loss=.1*(B0+rowloss(Q))/2+.5*(natural[i,j]-old[i,j]).square().mean()+.25*response(tab,group,T)
            losses.append(float(loss.detach()));loss.backward()
        torch.nn.utils.clip_grad_norm_(direct.parameters(),1.)
        ob.param_groups[0]['lr']=m.record.parent.stage_lr(49);ob.param_groups[1]['lr']=.001;ob.step()
        maxp=maxg=0.
        for a,b in zip(actual.parameters(),direct.parameters()):
            torch.testing.assert_close(a,b,atol=1e-6,rtol=1e-5);torch.testing.assert_close(a.grad,b.grad,atol=1e-6,rtol=1e-5)
            maxp=max(maxp,float((a-b).abs().max().detach()));maxg=max(maxg,float((a.grad-b.grad).abs().max()))
        outputs={}
        for arm in ['S','Q','C_q','C_b']:
            model=newmodel();opt=optimizer(model);log=m.update(model,opt,current,history,target2,c,arm,2,49)
            outputs[arm]=(log,{k:v.detach().clone() for k,v in model.named_parameters()},{k:p.grad.detach().clone() for k,p in model.named_parameters()})
        effects={}
        for lhs,rhs in [('Q','S'),('C_q','Q')]:
            effect={'loss_delta':outputs[lhs][0]['total']-outputs[rhs][0]['total']}
            for part in ['encoder','head']:
                names=[k for k in outputs[lhs][1] if k.startswith(part+'.')]
                pd=max(float((outputs[lhs][1][k]-outputs[rhs][1][k]).abs().max()) for k in names)
                gd=max(float((outputs[lhs][2][k]-outputs[rhs][2][k]).abs().max()) for k in names)
                assert pd>0 and gd>0;effect[part]={'parameter_max_delta':pd,'clipped_gradient_max_delta':gd}
            effects[lhs+'-'+rhs]=effect
        assert outputs['Q'][0]['history']['assignment_sha256']==outputs['C_q'][0]['history']['assignment_sha256']
        assert outputs['C_b'][0]['history']['response']==outputs['C_q'][0]['history']['response']
        np.savez(args.out/'tiny_update_arrays.npz',**{arm+'_'+kind+'_'+k:v.numpy() for arm,(log,pars,grads) in outputs.items() for kind,d in [('parameter',pars),('gradient',grads)] for k,v in d.items()})
        dump(args.out/'tiny_update_logs.json',{k:v[0] for k,v in outputs.items()})
        return {'direct_loss_error':abs(record['total']-sum(losses)),'direct_parameter_max_error':maxp,'direct_clipped_gradient_max_error':maxg,'effects':effects,'Q_Cq_assignments_equal':True,'Cq_Cb_same_R_at_common_state':True,'scope':'Tiny in-RAM modules only; not native qualification or project training'}
    test('actual_chunked_VJP_vs_independent_graph_and_four_toy_paths',update_test)
    def gates():
        # Only the gate function is invoked, never execute/train/native. All files here
        # are explicitly synthetic and deleted after this witness.
        measurements=dict(t_A=1.,t_S=2.,t_Q=2.,t_q=2.,t_b=2.,f=.1,h=.1,L=1.,s_F=1.,r_F=1.,s_I=1.,r_I=1.,W_F=1000,W_I=500)
        with tempfile.TemporaryDirectory(prefix='external_gate_fixture_',dir=args.out) as tmp:
            root=Path(tmp);src=runner.sources()
            auth=root/'toy_authorization.json';review=root/'toy_review.json';cpu=root/'toy_cpu.json';gpu=root/'toy_native.json';gate=root/'toy_gate.json'
            dump(auth,{'EXTERNAL_HANDWRITTEN_FIXTURE_NOT_AUTHORIZATION':True});dump(review,{'EXTERNAL_HANDWRITTEN_FIXTURE_NOT_REVIEW_DISPOSITION':True})
            dump(cpu,{'mode':'cpu','status':'PASS_HANDWRITTEN_ONLY','source_files':src,'external_fixture':True})
            native={'mode':'gpu','status':'PASS_HANDWRITTEN_ONLY','source_files':src,'external_fixture':True,'native':{'actual_updates':6,'next_update_bitwise_equal':True,'new_term_encoder_head_effect':True,'measurements':measurements}}
            dump(gpu,native)
            def rec(p):return {'path':str(p),'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
            gdict={'status':'APPROVED_QUERY_REPLAY_FORMAL','source_files':src,'review_disposition':'NO_OPEN_BLOCKERS','user_scope_approved':True,
                'authorization':rec(auth),'review':rec(review),'cpu':rec(cpu),'gpu':rec(gpu),'job':'reports/external_nonexecuted_job','runtime':runner.policy()['runtime'],'supervision':runner.policy()['supervision']() if False else runner.policy()['supervision']}
            dump(gate,gdict)
            runner.validate_gate(m.data.ROOT/'reports/external_nonexecuted_job',gate)
            assert 'pretrained_archive' not in native
        # Contractual archive identity is a distinct check from the trust in the scoped
        # human gate, which is intentionally simulated here, not challenged as security.
        return {'native_record_without_model_archive_accepted_by_formal_gate':True,'formal_entry_invocations':0,'native_entry_invocations':0,'formal_file_reads':0,
                'scope':'Synthetic gate data. Demonstrates absence of archive binding, not a forged real approval or a native success.'}
    test('qualification_archive_binding_counterexample',gates)
    def counts():
        vals={'A_diagnostic_forwards':6*(48+40),'B_diagnostic_forwards':4*6*(48+6+40+6),'endpoint_forwards':5*2*(12+40),'birth_forwards':6}
        vals.update(total_forwards=sum(vals.values()),h_calls=4*6*6,updates=288+4*288,presentations=288+4*288*2,
            L=1,s_F=5,r_F=5+4,s_I=5,r_I=5,full_states_peak=2,inference_states_peak=5)
        assert vals['total_forwards']==3454 and vals['h_calls']==144
        return vals
    test('fixed_call_accounting',counts)
    result={'scope':'Independent external handwritten audit only','environment':{'python':sys.version,'platform':platform.platform(),'torch':torch.__version__,'numpy':np.__version__,'cuda_visible_devices':os.environ['CUDA_VISIBLE_DEVICES']},
            'tests':tests,'failures':sum(t['status']!='PASS' for t in tests),'elapsed_seconds':time.monotonic()-start}
    dump(args.out/'results.json',result)
    print('SUMMARY',json.dumps({'tests':len(tests),'failures':result['failures'],'elapsed_seconds':result['elapsed_seconds']}),flush=True)
    return 1 if result['failures'] else 0
if __name__=='__main__':sys.exit(main())
