"""One-off incremental CPU/GPU evidence; only handwritten inputs."""
import argparse
import copy
import json
import math
import os
from pathlib import Path
import resource
import sys
import time
from types import SimpleNamespace
from unittest.mock import patch

parser = argparse.ArgumentParser()
parser.add_argument('--root', type=Path, required=True)
parser.add_argument('--workspace', type=Path, required=True)
parser.add_argument('--mode', choices=['cpu', 'gpu'], required=True)
args = parser.parse_args()
sys.path.insert(0, str(args.workspace/'scripts'))
started = time.monotonic()
os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[-2:])
import torch
from torch import nn
from torch.nn import functional as F
import step35_vendorlink_head_adapt as h
import step35_vendorlink_head_control as c
from step35_vendorlink_account_io import record, write
from step35_vendorlink_account_policy import settings, AccountSettings
from step29_vendorlink_methods import h_objective
from step35_vendorlink_account_evaluate import analyze
from step35_vendorlink_account_run import restore, groups_for
from step29_vendorlink_data import seed_for

c.configure_compute()
torch.manual_seed(7)
args.root.mkdir(parents=True, exist_ok=False)
checks = {}


def reference(head, query, galleries):
    values = []
    for gallery in galleries:
        pairs = torch.stack([head(torch.cat(((u-v).abs(), u*v))).squeeze() for u in query for v in gallery]).double()
        values.append((.5*pairs.mean()+.5*torch.logsumexp(pairs, 0)-.5*math.log(8)).float())
    return torch.stack(values)


def accounts(domain, n=4):
    return [dict(account_uid=f'{domain}_a{i:02d}', records=[dict(uid=f'{domain}_{i}_{j}', title='sample title', description='sample description')
            for j in range(12)]) for i in range(n)]


try:
    if args.mode == 'cpu':
        head = nn.Sequential(nn.Linear(12, 128), nn.ReLU(), nn.Linear(128, 1))
        other = copy.deepcopy(head)
        q, g = torch.randn(2, 6), torch.randn(7, 4, 6)
        logits = h.full_pool_logits(head, q, g)
        expected = reference(other, q, g)
        torch.testing.assert_close(logits, expected, atol=2e-7, rtol=1e-6)
        F.cross_entropy(logits[None], torch.tensor([0])).backward()
        F.cross_entropy(expected[None], torch.tensor([0])).backward()
        error = max(float((a.grad-b.grad).abs().max()) for a, b in zip(head.parameters(), other.parameters()))
        assert error < 2e-6
        changed = g.clone(); changed[-1] += 10
        assert not torch.allclose(h.full_pool_logits(head, q, changed)[-1], logits[-1])
        checks['full_pool_independent_score_gradient'] = dict(max_error=error, outside_group_candidate_effect=True)

        a = accounts('AlphaBay', 12)
        group = groups_for(a, 41, 'AlphaBay', 40)[0]
        lookup = {r['uid']: i for i, r in enumerate(r for x in a for r in x['records'])}
        indices = h.gallery_indices(a, group, lookup, 123)
        queries = {lookup[r.uid] for v in group.views for r in v.records}
        assert len(indices) == 12 and all(len(x) == 4 for x in indices)
        assert not queries.intersection(i for row in indices for i in row)
        features = torch.randn(12*12, 6).detach()
        actual = copy.deepcopy(head); expected_head = copy.deepcopy(head)
        opt = torch.optim.SGD(actual.parameters(), lr=.01)
        refopt = torch.optim.SGD(expected_head.parameters(), lr=.01)
        refopt.zero_grad(set_to_none=True)
        s = AccountSettings(**settings('H_CE_RMS', 'H_SEQ'))
        logs = h.head_update(actual, opt, features, a, group, lookup, s, 123, True)
        z = features[[lookup[r.uid] for v in group.views for r in v.records]]
        base_loss, _ = h_objective(h.original_table(expected_head, z), group, seed_for(123, 'regroup'), s)
        ce = sum(F.cross_entropy(reference(expected_head, features[[lookup[r.uid] for r in v.records]], features[indices])[None],
            torch.tensor([[x['account_uid'] for x in a].index(v.identity)])) for v in group.views)/len(group.views)
        (base_loss+ce).backward()
        torch.nn.utils.clip_grad_norm_(expected_head.parameters(), 1.)
        refopt.step()
        update_error = max(float((p-q).abs().max()) for p, q in zip(actual.parameters(), expected_head.parameters()))
        assert update_error < 2e-7 and features.grad is None and logs['encoder_updates'] == 0
        checks['actual_update_and_disjoint_roles'] = dict(max_update_error=update_error, candidates=12, frozen_features=True)

        from transformers import BertModel, BertConfig
        from step35_vendorlink_account_methods import AccountModel
        config = BertConfig(vocab_size=32, hidden_size=4, num_hidden_layers=4, num_attention_heads=2, intermediate_size=8)
        class Tokenizer:
            def __call__(self, texts, **kwargs):
                ids = torch.tensor([[1, 2+(sum(map(ord, text)) % 20), 3, 4] for text in texts])
                return dict(input_ids=ids, attention_mask=torch.ones_like(ids))
        def factory(binding, task):
            return AccountModel(BertModel(config), Tokenizer(), AccountSettings(**task['settings']))
        manifest = dict(status='QUALIFIED', files={}, counts={})
        bases = {}
        baseline, rms = {}, {}
        for domain in ('AlphaBay', 'Dream'):
            data = accounts(domain)
            task = dict(id='D_'+domain+'_H_CE_RMS', phase='D', candidate='H_CE_RMS', method='H_SEQ',
                seed=41, epochs=40, stages=1, domains=[domain], settings=settings('H_CE_RMS', 'H_SEQ'))
            model = factory({}, task)
            endpoint = args.root/(domain+'_base.pt')
            torch.save(dict(identity={'handwritten': domain}, task=task, model_binding={'files': []}, stage=1,
                encoder=model.encoder.state_dict(), head=model.head.state_dict(), vendor_heads={}), endpoint)
            b = args.root/(domain+'_base.json')
            write(b, dict(task=task, identity={'handwritten': domain}, endpoints={'1':record(endpoint)}))
            bases[domain] = record(b)
            for role, count in [('fit',12),('gallery',4),('dev',2)]:
                rows = data if role == 'fit' else [dict(account_uid=x['account_uid'], records=[
                    dict(uid=f"{x['account_uid']}_{role}_{i}", title='sample title', description='sample description') for i in range(count)]) for x in data]
                p = args.root/(domain+'_'+role+'.json'); write(p, rows)
                manifest['files'][domain+':'+role] = record(p)
            manifest['counts'][domain] = dict(eligible_accounts=4)
            baseline['D_'+domain+'_V1'] = {'V_CLS':[{'R11':.9}]}
            rms['D_'+domain+'_H_CE_RMS'] = {'H_SEQ':[{'R11':.8}]}
        for name, value in [('manifest',manifest),('baseline',baseline),('rms',rms)]:
            write(args.root/(name+'.json'), value)
        gate = dict(campaign=str(args.root/'campaign'), experiment_root=str(args.root),
            workspace=str(args.workspace), bases=bases, manifest=record(args.root/'manifest.json'), model_binding={'files': []},
            baseline=record(args.root/'baseline.json'), rms_results=record(args.root/'rms.json'),
            prior_seconds=dict(training=0,gpu=0,auxiliary=0),cpu_test_seconds=0)
        path = args.root/'gate.json'; write(path,gate)
        calls = []
        class SynchronousRunner:
            def __init__(self,*a,**k): pass
            def run_many(self,jobs,slots,completed):
                for job in jobs:
                    calls.append(job['action'])
                    extra = dict(zip(job['extra'][::2],job['extra'][1::2]))
                    obj = SimpleNamespace(action=job['action'],target=job['target'],deadline=time.monotonic()+120,
                        gate=path,domain=extra.get('--domain'),task=extra.get('--task'))
                    c.worker(gate,obj); completed(job)
        actual_train = h.train_head
        with patch.object(h,'restore',side_effect=lambda *a,**k:restore(*a,**k,factory=factory)), \
             patch.object(h,'train_head',side_effect=lambda *a,**k:actual_train(*a,**k,device='cpu')), \
             patch.object(c,'configure_compute',return_value=None), patch.object(c,'Runner',SynchronousRunner):
            c.campaign(gate,path,time.monotonic())
        assert calls == ['cache']*2+['train']*4+['score']*4+['analyze']
        specs = c.tasks(gate)
        result = json.loads((args.root/'campaign/completion.json').read_text())
        assert result['next_phase_authorized'] is False
        comps = [json.loads((args.root/'campaign/D/scoring'/t['id']/'result/completion.json').read_text()) for t in specs]
        missing = args.root/'forbidden_open'
        try: analyze(comps[:-1],specs,missing)
        except ValueError: pass
        else: raise AssertionError('Incomplete batch opened')
        assert not missing.exists()
        m = json.loads((args.root/'campaign/D/analysis/result/metrics.json').read_text())
        assert len(m) == 4 and all(len(v)==13 for v in m.values())
        assert sum(j['seconds'] for j in c.jobs(gate,specs,'train'))==c.CAPS['training']
        assert sum(j['seconds'] for j in c.jobs(gate,specs,'cache')+c.jobs(gate,specs,'score'))+60 <= c.CAPS['gpu']
        checks['actual_cache_train_score_batch'] = dict(tasks=4, readings=4, metrics=13, missing_batch_rejected=True, no_next_phase=True)
    else:
        torch.cuda.reset_peak_memory_stats()
        head = nn.Sequential(nn.Linear(3072,128),nn.ReLU(),nn.Linear(128,1)).cuda()
        query = F.normalize(torch.randn(20,2,1536,device='cuda'),dim=-1).detach()
        galleries = F.normalize(torch.randn(546,4,1536,device='cuda'),dim=-1).detach()
        before = [p.detach().clone() for p in head.parameters()]
        optimizer = torch.optim.AdamW(head.parameters(),lr=3e-4)
        torch.cuda.synchronize(); begin=time.monotonic()
        for i in range(20):
            loss=F.cross_entropy(h.full_pool_logits(head,query[i],galleries)[None],torch.tensor([i],device='cuda'))/20
            loss.backward()
        torch.nn.utils.clip_grad_norm_(head.parameters(),1.,error_if_nonfinite=True)
        optimizer.step();torch.cuda.synchronize()
        elapsed=time.monotonic()-begin
        assert query.grad is None and galleries.grad is None
        assert any(not torch.equal(a,b) for a,b in zip(before,head.parameters()))
        assert all(torch.isfinite(p).all() for p in head.parameters())
        checks['real_full_shape_head_backward'] = dict(queries=20,candidates=546,dimension=1536,
            seconds=elapsed,peak_allocated=torch.cuda.max_memory_allocated(),peak_reserved=torch.cuda.max_memory_reserved(),
            actual_parameter_change=True,frozen_inputs=True)
    status='PASS'
except BaseException as exc:
    status='FAIL';checks['error']=repr(exc)
    raise
finally:
    elapsed=time.monotonic()-started
    write(args.root/'receipt.json',dict(status=status,tests=len(checks),failures=int(status!='PASS'),errors=0,
        checks=checks,actual_seconds=elapsed,charged_seconds=math.ceil(elapsed),source=record(Path(__file__).resolve()),
        runtime_sources=[record(args.workspace/'scripts'/n) for n in c.SOURCES],
        peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,cpu_affinity=sorted(os.sched_getaffinity(0))))
    print(json.dumps(dict(status=status,actual_seconds=elapsed,checks=checks)))
