"""Use authored texts to exercise the actual optimizer, checkpoint and scoring paths."""
import argparse
from contextlib import ExitStack
import gc
import hashlib
import io
import math
from pathlib import Path
import tempfile
from unittest.mock import patch
from zipfile import ZipFile

from operations import previous, read, verify, write
import torch
from step29_vendorlink_data import Ad, View, assignment
from step29_vendorlink_methods import Memory, h_pool
from step35_market_adapters import MarketH, restore_history
from step35_market_worker import market_tasks, paired_report
from step35_vendorlink_account_control import validate_training
from step35_vendorlink_account_evaluate import score_market
from step35_vendorlink_account_policy import CELLS, DOMAINS, SEEDS, summarize
import step35_vendorlink_account_run as run


def require(value, message):
    if not value:
        raise ValueError(message)


def accounts(domain, n, long=False):
    fill = '.!'*128 if long else ' paper'
    return [dict(account_uid=f'{domain}-author-{i:02}',records=[
            dict(uid=f'{domain}-{i:02}-{j:02}',title=f'Notebook {i} {j}'+fill,
                 description=f'Synthetic book {j} {i}'+fill) for j in range(12)]) for i in range(n)]


def digest(values):
    h=hashlib.sha256()
    for value in values:
        h.update(value.detach().contiguous().cpu().numpy().tobytes())
    return h.hexdigest()


def read_scores(model, stage):
    result={}
    for t,j in CELLS:
        if t!=stage:
            continue
        market=DOMAINS[j-1]
        vals=accounts(market,2)
        query=[View('q'+str(i),tuple(Ad('q'+r['uid'],'Query '+r['title'],r['description']) for r in v['records'][:2]))
               for i,v in enumerate(vals)]
        fit=[View('g'+str(i),tuple(Ad(**r) for r in v['records'])) for i,v in enumerate(vals)]
        q,f,s=score_market(model,market,query,fit,lambda:None)
        result[market]=(digest([q]),digest([f]),hashlib.sha256(s.tobytes()).hexdigest())
    return result


def objective_check(binding):
    task=next(t for t in market_tasks() if t['method']=='H_full')
    model=MarketH(run.make_model(binding,task,'cpu'))
    model.enter(1)
    group=run.groups_for(accounts('AlphaBay',4),51,'AlphaBay',0)[0]
    base=model.teacher(group,lambda:None).to('cpu')
    teacher=base+.25
    model.enter(2)
    model.eval()
    with torch.no_grad():
        z=model.records(group,lambda:None)
    ref=model.h.table(z)
    delta=h_pool(ref,assignment(group))-h_pool(teacher,assignment(group))
    loss=model.settings.beta*delta.square().mean()
    params=list(model.head.parameters())
    expected=torch.autograd.grad(loss,params)
    before=torch.get_rng_state().clone()
    model.train()
    log=model.backward_birth(group,teacher,lambda:None)
    error=max((p.grad-r).abs().max().item() for p,r in zip(params,expected))
    require(all(torch.allclose(p.grad,r,rtol=2e-4,atol=2e-6) for p,r in zip(params,expected)) and
            torch.equal(torch.get_rng_state(),before) and max(r.abs().max() for r in expected)>0 and
            all(p.grad is None for p in model.encoder.parameters()) and log['weighted']>0,
            'Deterministic birth gradient differs from direct reference or changed old encoder/RNG')
    from dataclasses import replace
    model.h.settings=replace(model.settings,method='H_ER',deterministic_history=False)
    model.zero_grad(set_to_none=True)
    label=model.backward_h(group,837,base,history=True)
    label_norm=sum(p.grad.norm().item() for p in params)
    require(label_norm>0 and label['weighted']>0 and all(p.grad is None for p in model.encoder.parameters()),
            'Old ER labels must reach only the shared head')
    return dict(birth_gradient_max_abs_error=error,birth_objective=log['weighted'],er_head_gradient=label_norm,
                deterministic_rng_preserved=True,old_encoder_gradients_absent=True)


def decision_check():
    def rows(old,pre=.60,final=.80):
        cells=dict(R01=.20,R11=.80,R12=pre,R21=.76,R22=final,R23=.60,R31=old,R32=old,R33=.83)
        return [dict(seed=s,**summarize(cells)) for s in SEEDS]
    trials=dict(H_full=rows(.79,.71),H_ER=rows(.73),H_SEQ=rows(.70),V_CLS=rows(.75),V_SET=rows(.75))
    result=paired_report(trials,{})
    require(result['full_application'] and result['new_learning']['H_ER/Dream']['net']<-.02,
            'A higher pre-learning score with protected acquisition should pass')
    trials['H_full']=rows(.79,.79)
    require(not paired_report(trials,{})['full_er'],'Relative alternative must not waive absolute learning')
    trials['H_full']=rows(.79)
    for r in trials['H_full']:
        r.update(summarize({**{k:r[k] for k in ('R01','R11','R12','R21','R22','R23','R31','R32','R33')},
                            'R31':.72,'R32':.86}))
    require(not paired_report(trials,{})['full_er_checks']['each_old_final'],'One falling old market must fail')
    return True


def exercise(binding,device,folder,output,used):
    native=device=='cuda'
    arm='H_full' if native else 'H_SEQ'
    target='cuda:0' if native else 'cpu'
    task=next(t for t in market_tasks() if t['method']==arm)
    task=dict(task,epochs=1)
    identity=dict(qualification='authored_synthetic',task=task)
    sizes=(19,11,11) if native else (4,4,4)
    rows={d:accounts(d,n,native) for d,n in zip(DOMAINS,sizes)}
    base=run.make_model(binding,task,target)
    live_model=[]
    stage_start={}
    endpoints={}
    probes={}
    counters=dict(steps=0,checkpoint_recomputations=0,old_supervision_calls=0,max_records=0,
                  max_tokens=0,longest_training_channels=0)
    orig_init=MarketH.__init__
    orig_start=MarketH.start_training_market
    orig_back=MarketH.backward_h
    orig_update=run.update_account
    orig_save=run.atomic_tensor
    def capture(model,original):
        orig_init(model,original)
        live_model.append(model)
    def stage_state(model):
        state=dict(head=digest(model.head.parameters()),base=digest(p for n,p in model.encoder.named_parameters()
                                                                  if '.banks.' not in n))
        for market in DOMAINS[1:model.stage]:
            state[market]=digest(v for k,v in model.adapter_tensors().items() if '.'+market+'.' in k)
        return state
    def start(model,stage,seed,optimizer):
        if optimizer is not None:
            head_params=list(model.head.parameters())
            before=[(id(optimizer.state[p]),digest(optimizer.state[p].values())) for p in head_params]
        state=torch.get_rng_state().clone()
        cuda_state=torch.cuda.get_rng_state() if native else None
        new=orig_start(model,stage,seed,optimizer)
        if native and stage==1:
            def entering_optimizer(opt,args,kwargs):
                require(used+counters['steps']<12,'Cumulative native optimizer allowance')
                write(output/'step_intents.json',dict(count=counters['steps']+1))
            new.register_step_pre_hook(entering_optimizer)
        require(torch.equal(state,torch.get_rng_state()) and
                (not native or torch.equal(cuda_state,torch.cuda.get_rng_state())), 'Adapter initialization changed pairing RNG')
        if stage>1:
            require(new is optimizer and before==[(id(new.state[p]),digest(new.state[p].values())) for p in head_params]
                    and not any(p in new.state for p in model.encoder.parameters()), 'Optimizer switch lost head or kept old moments')
        if stage==3:
            for p in model.projections.values():
                for a,b in zip(p.banks['Dream'].parameters(),p.banks['Agora'].parameters()):
                    require(torch.equal(a,b) and a.data_ptr()!=b.data_ptr() and not a.requires_grad and b.requires_grad,
                            'Agora must receive an independent copy of Dream')
        stage_start[stage]=stage_state(model)
        return new
    def backward(model,group,step_seed,teacher=None,history=False,check=lambda:None):
        routes=[]
        def hooked(layer,inputs):
            routes.append(layer.route.get())
            counters['max_tokens']=max(counters['max_tokens'],inputs[0].shape[1])
        hook=model.projections['8.query'].register_forward_pre_hook(hooked)
        try:
            value=orig_back(model,group,step_seed,teacher,history,check)
        finally:
            hook.remove()
        n=sum(len(v.records) for v in group.views)
        require(routes and set(routes)=={group.domain},'A forward or checkpoint recomputation used another market')
        if history:
            counters['old_supervision_calls']+=1
        else:
            forward=math.ceil(2*n/model.settings.microbatch)
            require(len(routes)>forward,'Current route was not witnessed during checkpoint recomputation')
            counters['checkpoint_recomputations']+=len(routes)-forward
            counters['max_records']=max(counters['max_records'],n)
            if native and n==44 and counters['max_tokens']==256:
                counters['longest_training_channels']=88
        return value
    def update(model,opt,current,history,teacher,seed,check):
        value=orig_update(model,opt,current,history,teacher,seed,check)
        require(value['optimizer_updates']==1 and value['current_encoder_gradient_norm']>0,
                'No actual current encoder/adapter optimization')
        counters['steps']+=1
        return value
    def save(path,value,check,replace=False):
        orig_save(path,value,check,replace)
        if path.name.startswith('stage'):
            k=value['stage']
            model=live_model[0]
            probes[k]=read_scores(model,k)
            if k:
                endpoints[k]=stage_state(model)
    with ExitStack() as ctx:
        for obj,key,func in ((MarketH,'__init__',capture),(MarketH,'start_training_market',start),
                             (MarketH,'backward_h',backward),(run,'update_account',update),(run,'atomic_tensor',save)):
            ctx.enter_context(patch.object(obj,key,func))
        result=run.execute(base,rows.__getitem__,folder/'run',task,identity,binding)
    counts={k:dict(eligible_accounts=len(v)) for k,v in rows.items()}
    validate_training(task,result,dict(counts=counts),identity)
    require(counters['steps']==(12 if native else 9) and
            all(endpoints[i]['head']!=stage_start[i]['head'] for i in (1,2,3)) and
            endpoints[1]['base']!=stage_start[1]['base'] and
            endpoints[1]['base']==endpoints[2]['base']==endpoints[3]['base'] and
            endpoints[2]['Dream']!=stage_start[2]['Dream'] and endpoints[3]['Dream']==endpoints[2]['Dream'] and
            endpoints[3]['Agora']!=stage_start[3]['Agora'],'Missing actual update or frozen parameter drift')
    if native:
        require(counters['max_records']==44 and counters['longest_training_channels']==88 and
                counters['old_supervision_calls']==6,'Native maximum group or replay path missing')
    del base
    live_model.clear()
    gc.collect()
    if native:
        torch.cuda.empty_cache()
    for k in range(4):
        restored,_=run.restore(binding,task,verify(result['endpoints'][str(k)]),identity,k,
                               factory=lambda b,t:run.make_model(b,t,target))
        require(read_scores(restored,k)==probes[k], 'Reloaded Q2/FIT12 vectors or real scores differ')
        if k:
            payload=(folder/'run'/f'market_state_{k}.zip').read_bytes()
            memory=Memory(123,restored.settings)
            restore_history(restored,payload,identity,memory)
            require(len(memory.entries)==(6 if native else 0),'Wrong retained group count')
            if native:
                with ZipFile(io.BytesIO(payload)) as z:
                    ref=torch.load(io.BytesIO(z.read('memory.pt')),map_location='cpu',weights_only=True)
                require(ref['seen']==memory.seen and ref['rng']==memory.rng.getstate() and
                        all(a['group']==b['group'] and torch.equal(a['teacher'],b['teacher']) for a,b in
                            zip(memory.entries,ref['entries'],strict=True)),'Replay or sampler failed to restore')
        del restored
    return dict(arm=arm,workload=result['workload'],counters=counters,restored_score_cells=len(CELLS),
                history_bytes=result['history_bytes'],old_parameters_unchanged=True,adam_history_preserved=True,
                dependency_and_gate_identity='local official pretrained plus authored synthetic inputs')


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--device',required=True,choices=('cpu','cuda'))
    p.add_argument('--output',required=True,type=Path)
    p.add_argument('--used-steps',required=True,type=int)
    args=p.parse_args()
    torch.set_num_threads(1 if args.device=='cpu' else 2)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    binding=previous()['model_binding']
    for item in binding['files']:
        verify(item)
    extra={}
    if args.device=='cpu':
        extra=dict(replay_gradients=objective_check(binding),policy_fixtures=decision_check())
        gc.collect()
    with tempfile.TemporaryDirectory(prefix='authored_',dir=args.output) as scratch:
        info=exercise(binding,args.device,Path(scratch),args.output,args.used_steps)
    write(args.output/'result.json',dict(passed=True,**extra,**info))
