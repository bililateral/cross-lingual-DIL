"""One-shot affected-path evidence; Linux py310 only, handwritten accounts."""
import copy
from dataclasses import asdict, replace
import json
import math
from pathlib import Path
import signal
import sys
import time
from types import SimpleNamespace
from unittest.mock import patch

started = time.monotonic()
signal.alarm(60)
import psutil
process = psutil.Process()
process.cpu_affinity(process.cpu_affinity()[-2:])
import torch
from torch.nn import functional as F
from transformers import BertConfig, BertModel
import step35_vendorlink_account_run as run
import step35_vendorlink_account_evaluate as evaluate
from step35_vendorlink_account_methods import AccountModel, update_account
from step35_vendorlink_account_policy import AccountSettings, tasks, CAPS, LIMITS
from step35_vendorlink_account_control import make_jobs, pipeline, validate_training
from step35_vendorlink_account_io import record, write
from step29_vendorlink_data import seed_for
from step29_vendorlink_methods import Model, h_objective, make_optimizer, update_h

torch.set_num_threads(2)
torch.set_num_interop_threads(2)
torch.use_deterministic_algorithms(True)
root = Path(sys.argv[1])
root.mkdir(exist_ok=False)
checks = {}


class Tokenizer:
    def __call__(self, texts, **kwargs):
        ids = [[1]+[3+ord(c)%55 for c in t[:5]]+[2] for t in texts]
        width = max(map(len, ids))
        return {'input_ids': torch.tensor([x+[0]*(width-len(x)) for x in ids]),
                'attention_mask': torch.tensor([[1]*len(x)+[0]*(width-len(x)) for x in ids])}


def factory(binding, task):
    torch.manual_seed(seed_for(task['seed'], ('AlphaBay', 'Dream', 'Agora'), 'initial'))
    encoder = BertModel(BertConfig(vocab_size=64, hidden_size=4, num_hidden_layers=4,
        num_attention_heads=2, intermediate_size=8, hidden_dropout_prob=.1,
        attention_probs_dropout_prob=.1, max_position_embeddings=32))
    return AccountModel(encoder, Tokenizer(), AccountSettings(**task['settings']))


def accounts(domain, role, count):
    return [dict(account_uid=f'hand-{k}', records=[dict(uid=f'{domain}-{role}-{k}-{j}',
        title=f't{k}{j}', description=f'd{k}{j}') for j in range(count)]) for k in range(4)]


try:
    specs = tasks('D')
    assert len(specs) == 2 and all(t['seed'] == 41 and t['epochs'] == 40 for t in specs)
    try:
        tasks('P', {'H': 'H_CE_RMS', 'V': 'V1'})
        raise AssertionError('P accepted')
    except ValueError:
        pass
    task = {**specs[0], 'epochs': 1}
    task['settings'] = {**task['settings'], 'channel_tokens': 8, 'microbatch': 32}
    model = factory({}, task)
    group = run.groups_for(accounts('AlphaBay', 'fit', 12), 41, 'AlphaBay', 0)[0]
    model.start_account_stage([f'hand-{k}' for k in range(4)], 77)
    ref = copy.deepcopy(model)
    seed = 173
    captures = []
    hook = model.auxiliary_head.register_forward_pre_hook(lambda module, args: captures.append(args[0].detach().clone()))
    torch.manual_seed(seed)
    before = model.records(group).detach()
    model.zero_grad(set_to_none=True)
    logs = model.backward_h(group, seed)
    hook.remove()
    torch.manual_seed(seed)
    z = ref.records(group)
    loss, _ = h_objective(ref.table(z), group, seed_for(seed, 'regroup'), ref.settings)
    labels = torch.tensor([int(v.identity.split('-')[1]) for v in group.views for _ in v.records])
    ce = F.cross_entropy(ref.auxiliary_head(z*math.sqrt(z.shape[-1])), labels)
    (loss+ce).backward()
    error = max(float((a.grad-b.grad).abs().max()) for a, b in zip(model.parameters(), ref.parameters()) if a.grad is not None)
    assert error < 2e-5, error
    assert torch.allclose(captures[0], before*math.sqrt(8), atol=1e-6)
    assert torch.allclose(before.norm(dim=1), torch.ones(len(before)), atol=1e-6)
    checks['scale_and_independent_autograd'] = {'max_gradient_error': error, 'dimension': 8,
        'scale': model.ce_scale(), 'correct_count': logs['ce_correct'], 'records': logs['ce_records']}

    zero = copy.deepcopy(model)
    zero.settings = replace(zero.settings, gamma=0.)
    old = copy.deepcopy(zero)
    old.backward_h = Model.backward_h.__get__(old, type(old))
    o1, o2 = make_optimizer(zero), make_optimizer(old)
    a = update_account(zero, o1, group, None, None, 88)
    rng1 = torch.get_rng_state().clone()
    b = update_h(old, o2, group, None, None, 88)
    rng2 = torch.get_rng_state().clone()
    assert torch.equal(rng1, rng2)
    assert a['current']['supervised'] == b['current']['supervised']
    delta = max(float((p-q).abs().max()) for p,q in zip(zero.parameters(), old.parameters()))
    assert delta == 0., delta
    checks['gamma_zero_original_update_rng'] = {'max_parameter_difference': delta}

    completions, tiny_tasks = [], []
    manifest = {'files': {}, 'counts': {}}
    for original in specs:
        t = {**original, 'epochs': 1, 'settings': task['settings']}
        tiny_tasks.append(t)
        domain = t['domains'][0]
        manifest['counts'][domain] = {'eligible_accounts': 4}
        for role, count in [('gallery', 4), ('dev', 2)]:
            p = root/f'{domain}_{role}.json'
            write(p, accounts(domain, role, count))
            manifest['files'][domain+':'+role] = record(p)
        identity = {'handwritten': True, 'task': t}
        completed = run.execute(factory({}, t), lambda d: accounts(d, 'fit', 12),
                                root/domain/'train', t, identity, {})
        validate_training(t, completed, manifest, identity)
        (root/domain/'train/latest.pt').unlink()
        with patch.object(evaluate, 'restore', side_effect=lambda binding, tt, path, ii, stage:
                          run.restore(binding, tt, path, ii, stage, factory=factory)):
            c = evaluate.score(t, completed, manifest, {}, root/domain/'score', 'dev', lambda: None)
        completions.append(c)
        payload = torch.load(completed['diagnostics']['1']['path'], weights_only=False)
        bad = {**payload, 'classes': list(reversed(payload['classes']))}
        try:
            evaluate.diagnostic_readings(torch.ones(4,2,8), torch.ones(4,4,8), bad, payload['classes'])
            raise AssertionError('Wrong classes accepted')
        except ValueError:
            pass
    bad = copy.deepcopy(completions)
    del bad[1]['files']['H_AUX:R11']
    try:
        evaluate.analyze(bad, tiny_tasks, root/'must_not_open')
        raise AssertionError('Missing reading accepted')
    except ValueError:
        assert not (root/'must_not_open').exists()
    results = evaluate.analyze(completions, tiny_tasks, root/'analysis')
    metric_values = json.loads((root/'analysis/metrics.json').read_text())
    assert len(metric_values) == 6 and all(len(v) == 13 for v in metric_values.values())

    # Independent nontrivial formulas catch omitted scale, wrong softmax order,
    # and replacement of the original set denominator with ordinary cosine.
    q = F.normalize(torch.arange(1., 65.).reshape(4,2,8), dim=-1)
    g = F.normalize(torch.arange(3., 131.).reshape(4,4,8).flip(-1), dim=-1)
    p = {'classes': list('abcd'), 'dimension': 8, 'scale': math.sqrt(8),
         'state': {'weight': torch.arange(32.).reshape(4,8)/9, 'bias': torch.tensor([.1,-.2,.3,-.4])}}
    read = evaluate.diagnostic_readings(q,g,p,list('abcd'))
    expected_aux = torch.stack([torch.stack([torch.softmax(p['state']['weight'] @ (r*math.sqrt(8))+p['state']['bias'], 0)
                                  for r in records]).mean(0) for records in q]).numpy()
    expected_set = torch.tensor([[2*torch.stack([a@b for a in qr for b in gr]).mean() /
        (torch.stack([a@b for a in qr for b in qr]).mean()+torch.stack([a@b for a in gr for b in gr]).mean())
        for gr in g] for qr in q]).numpy()
    import numpy as np
    assert np.allclose(read['H_AUX'], expected_aux, atol=1e-6)
    assert np.allclose(read['H_SET'], expected_set, atol=1e-6)
    checks['actual_save_restore_score_release'] = {'tasks': 2, 'readings': 6, 'metrics_per_reading': 13,
        'missing_reading_rejected_before_open': True, 'wrong_class_order_rejected': True,
        'latest_deleted_before_score': True}

    train = make_jobs(root, 'D', specs, 'train')
    score = make_jobs(root, 'D', specs, 'score')
    assert [x['seconds'] for x in train] == [10800,10800]
    assert [x['seconds'] for x in score] == [900,900]
    assert sum(x['seconds'] for x in train) == CAPS['training']
    assert sum(x['seconds'] for x in score) == CAPS['gpu']
    phases = []
    fake = {t['id']: {k: [{'seed':41, 'R11':.9}] for k in ('H_SEQ','H_AUX','H_SET')} for t in specs}
    baseline = {'D_'+d+'_'+c: {k: [{'seed':41,'R11':.8}] for k in ('H_SEQ','V_CLS')} for d in ('AlphaBay','Dream') for c in ('H_CE','H_base','V1')}
    result = pipeline(lambda phase, selection: (phases.append(phase) or fake), lambda *args: None, baseline)
    assert phases == ['D'] and result['status'] == 'PASS_D' and not result['next_phase_authorized']
    checks['two_task_budget_and_no_next_phase'] = {'phases':phases,'caps':CAPS,'train_caps':[10800,10800], 'score_caps':[900,900]}
    receipt = dict(status='PASS', tests=len(checks), failures=0, errors=0, checks=checks)
except BaseException as error:
    receipt = dict(status='FAIL', tests=len(checks), failures=1, errors=0, checks=checks, error=repr(error))
    raise
finally:
    receipt.update(actual_seconds=time.monotonic()-started, charged_seconds=math.ceil(time.monotonic()-started),
        source=record(Path(__file__)), peak_rss_bytes=process.memory_info().rss, cpu_affinity=process.cpu_affinity(),
        actual_entry_sources=[record(p) for p in sorted(Path(run.__file__).parent.glob('step35_vendorlink_account_*.py'))],
        no_pretrained_load=True, no_formal_data=True, no_gpu=True)
    write(root/'receipt.json', receipt)
    print(json.dumps(receipt))
