"""Actual twelve-ad training, fixed epoch-zero replay, and retained V heads."""
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import random

import torch

from step29_vendorlink_data import Ad, View, Group, group_chunks, seed_for
from step29_vendorlink_methods import Memory, make_optimizer, update_vendor
from step35_vendorlink_account_methods import (AccountModel, update_account, set_account_rates,
                                               save_account_state)
from step35_vendorlink_account_io import write, record
from step35_vendorlink_account_policy import AccountSettings, DOMAINS, task_domains


def groups_for(accounts, seed, domain, epoch):
    ids = [a['account_uid'] for a in accounts]
    if ids != sorted(set(ids)) or len(ids) < 4:
        raise ValueError('Sorted unique fit accounts required')
    records = {}
    for a in accounts:
        items = sorted((Ad(**r) for r in a['records']), key=lambda r: r.uid)
        if len(items) != 12 or len({r.uid for r in items}) != 12:
            raise ValueError('Twelve distinct fit records required')
        random.Random(seed_for(seed, domain, epoch, 'records', a['account_uid'])).shuffle(items)
        records[a['account_uid']] = items
    result = []
    for packet in range(3):
        shuffled = ids.copy()
        random.Random(seed_for(seed, domain, epoch, 'accounts', packet)).shuffle(shuffled)
        for index, chunk in enumerate(group_chunks(shuffled, 8)):
            views = []
            for account in chunk:
                for v in range(2):
                    ads = tuple(records[account][4*packet+2*v:4*packet+2*v+2])
                    uid = hashlib.sha256(json.dumps([r.uid for r in ads]).encode()).hexdigest()
                    views.append(View(uid, ads, account))
            key = [seed, domain, epoch, packet, index, [[r.uid for r in v.records] for v in views]]
            uid = hashlib.sha256(json.dumps(key, ensure_ascii=False, separators=(',', ':')).encode()).hexdigest()
            g = Group(uid, domain, 'fit', tuple(views))
            g.validate(training=True)
            result.append(g)
    all_ids = [r.uid for g in result for v in g.views for r in v.records]
    if len(all_ids) != 12*len(accounts) or len(set(all_ids)) != len(all_ids):
        raise ValueError('Every current advertisement must appear once per epoch')
    return result


def make_model(binding, task, device='cuda:0'):
    from transformers import BertModel, BertTokenizer
    torch.manual_seed(seed_for(task['seed'], DOMAINS, 'initial'))
    encoder = BertModel.from_pretrained(binding['directory'], local_files_only=True)
    if encoder.config.hidden_size != 768:
        raise ValueError('Bound BERT base required')
    if task['method'] != 'VendorLink_SEQ':
        # Recompute transformer activations during backward with the original
        # dropout RNG restored. Same inputs/objective; avoids retaining all
        # 88 full-length channel graphs. No artificial GPU allocation cap.
        encoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    tokenizer = BertTokenizer.from_pretrained(binding['directory'], local_files_only=True, do_lower_case=True)
    if not tokenizer.do_lower_case:
        raise ValueError('Keep the approved English tokenizer behavior')
    return AccountModel(encoder, tokenizer, AccountSettings(**task['settings'])).to(device)


def atomic_tensor(path, value, check, replace=False):
    if path.exists() and not replace:
        raise ValueError('No endpoint overwrite')
    check()
    temporary = path.with_suffix('.partial')
    torch.save(value, temporary)
    temporary.replace(path)
    check()


def execute(model, load_fit, job, task, identity, binding, check=lambda: None):
    routed = 'market_adapters' in task
    if routed:
        from step35_market_adapters import SHARED_FFN_FORMAT, MarketH, pack_history
        if not task.get('stage_end_only'):
            raise ValueError('Market routing requires stage-end endpoints')
        model = MarketH(model, adapter_format=task['market_adapters'])
    if job.exists() or model.settings != AccountSettings(**task['settings']) or task['stages'] not in (1, 3):
        raise ValueError('Fresh fixed task and actual configuration required')
    job.mkdir(parents=True)
    seed, s = task['seed'], model.settings
    vendor = s.method == 'VendorLink_SEQ'
    replay = s.method in ('H_ER', 'H_full')
    memory = Memory(seed_for(seed, DOMAINS, 'reservoir'), s)
    optimizer, heads, endpoints, updates = None, {}, {}, 0
    diagnostics = {}
    workload = dict(current_ads=0, current_updates=0, history_ads=0, history_groups=0,
                    teacher_ads=0, teacher_groups=0)
    if s.deterministic_history:
        workload['deterministic_history_groups'] = 0
    base = dict(identity=identity, task=task, model_binding=binding)
    current_stream, replay_stream = hashlib.sha256(), hashlib.sha256()
    history_sizes, survivors = {}, {}
    shared = None
    def save_endpoint(stage):
        nonlocal shared
        value = dict(**base, stage=stage, updates=updates,
                     head={k: v.detach().cpu() for k, v in model.head.state_dict().items()}, vendor_heads=heads)
        if stage and routed:
            if stage == 1:
                path = job/'shared_encoder.pt'
                atomic_tensor(path, model.shared_encoder_state(), check)
                shared = record(path)
            if stage >= 2 and model.adapter_format == SHARED_FFN_FORMAT:
                # Only the scorer reads this endpoint. Training carries one live FFN.
                value['shared_ffn'] = {k: p.detach().cpu() for k, p in model.shared_ffn_parameters().items()}
            path = job/f'market_state_{stage}.zip'
            payload = pack_history(model, memory, identity)
            with path.open('xb') as f:
                f.write(payload)
            value.update(shared_encoder=shared, market_state=record(path))
            history_sizes[f'end_{stage}'] = len(payload)
        elif stage:
            value['encoder'] = {k: v.detach().cpu() for k, v in model.encoder.state_dict().items()}
        path = job/f'stage{stage}.pt'
        atomic_tensor(path, value, check)
        endpoints[str(stage)] = record(path)
    save_endpoint(0)
    try:
        for stage, domain in enumerate(task_domains(task), 1):
            check()
            accounts = load_fit(domain)  # Only the current market; never any gallery/query.
            memory_candidates = groups_for(accounts, seed, domain, 0)
            classes = [a['account_uid'] for a in accounts]
            if vendor:
                model.start_vendor_stage(classes, seed_for(seed, domain, 'head'))
                optimizer = make_optimizer(model)  # Previous head is only in CPU snapshots.
            else:
                if routed:
                    optimizer = model.start_training_market(stage, seed_for(seed, domain, 'market_adapter'), optimizer)
                    if stage > 1:
                        history_sizes[f'enter_{stage}'] = len(pack_history(model, memory, identity))
                model.start_account_stage(classes, seed_for(seed, domain, 'auxiliary_head'))
                if optimizer is None:
                    optimizer = make_optimizer(model)
            per_epoch = (12*len(accounts)+31)//32 if vendor else len(memory_candidates)
            for epoch in range(task['epochs']):
                if vendor:
                    examples = [(Ad(**r), a['account_uid']) for a in accounts for r in a['records']]
                    random.Random(seed_for(seed, domain, epoch, 'current')).shuffle(examples)
                    batches = [examples[i:i+32] for i in range(0, len(examples), 32)]
                else:
                    batches = memory_candidates if epoch == 0 else groups_for(accounts, seed, domain, epoch)
                if len(batches) != per_epoch:
                    raise ValueError('Exposure/schedule count changed')
                for number, batch in enumerate(batches, 1):
                    check()
                    update = epoch*per_epoch+number
                    step_seed = seed_for(seed, domain, update, 'update')
                    history, teacher = (None, None)
                    if stage > 1 and replay:
                        history, teacher = memory.draw(seed_for(seed, domain, update, 'history'))
                    set_account_rates(model, optimizer, update, task['epochs']*per_epoch)
                    if vendor:
                        records, labels = zip(*batch)
                        logs = update_vendor(model, optimizer, records, labels, step_seed, check)
                        count = len(batch)
                        current = [r.uid for r in records]
                    else:
                        logs = update_account(model, optimizer, batch, history, teacher, step_seed, check)
                        count = sum(len(v.records) for v in batch.views)
                        current = [batch.uid]
                    updates += 1
                    workload['current_updates'] += 1
                    workload['current_ads'] += count
                    if routed:
                        current_stream.update((batch.uid+'\n').encode())
                        if history is not None:
                            replay_stream.update((history.uid+'\n').encode())
                    if history is not None:
                        workload['history_groups'] += 1
                        workload['history_ads'] += sum(len(v.records) for v in history.views)
                        if s.deterministic_history:
                            if not logs.get('deterministic_birth', {}).get('deterministic'):
                                raise ValueError('Deterministic birth branch was not executed')
                            workload['deterministic_history_groups'] += 1
                    with (job/'updates.jsonl').open('a', encoding='utf-8') as f:
                        f.write(json.dumps(dict(stage=stage, epoch=epoch, stage_update=update, updates=updates, total_updates=updates,
                            current_ids=current, history_group=None if history is None else history.uid,
                            rates=[g['lr'] for g in optimizer.param_groups], **logs), allow_nan=False)+'\n')
                    if task.get('stage_end_only') and (update == 1 or update % 25 == 0):
                        write(job/'workload.json', dict(workload, stage=stage, domain=domain, stage_update=update))
                write(job/'workload.json', dict(workload, stage=stage, domain=domain, stage_update=update)
                      if task.get('stage_end_only') else workload)
                if task.get('stage_end_only'):
                    continue
                cursor = dict(stage=stage, next_epoch=epoch+1, updates=updates, workload=workload,
                              vendor_heads=heads)
                check()
                save_account_state(job/'latest.pt', model, optimizer, memory, cursor, identity)
                check()
            if vendor:
                heads[domain] = dict(classes=list(model.classes),
                    state={k: v.detach().cpu().clone() for k, v in model.classifier.state_dict().items()})
            elif replay and stage < task['stages']:
                def teacher_done(group):
                    workload['teacher_groups'] += 1
                    workload['teacher_ads'] += sum(len(v.records) for v in group.views)
                memory.retain(memory_candidates, stage, model, check, record_teacher=teacher_done)
                if routed:
                    survivors[domain] = [r['group']['uid'] for r in memory.entries]
                write(job/f'memory_stage{stage}.json', dict(bytes=len(memory.bytes()), seen=memory.seen,
                    candidates=len(memory_candidates), groups=[dict(uid=r['group']['uid'], origin=r['origin']) for r in memory.entries]))
            save_endpoint(stage)
            if s.diagnostics:
                diagnostic = dict(**base, stage=stage, endpoint=endpoints[str(stage)],
                    dimension=2*model.encoder.config.hidden_size, scale=model.ce_scale(),
                    classes=list(model.auxiliary_classes),
                    state={k: v.detach().cpu().clone() for k, v in model.auxiliary_head.state_dict().items()})
                diagnostic_path = job/f'diagnostic_stage{stage}.pt'
                atomic_tensor(diagnostic_path, diagnostic, check)
                diagnostics[str(stage)] = record(diagnostic_path)
            del accounts, memory_candidates, batches, batch, history, teacher
        result = dict(status='TRAINING_COMPLETE_EVALUATION_CLOSED', **base, endpoints=endpoints,
                      updates=updates, workload=workload, diagnostics=diagnostics)
        if routed:
            result.update(history_bytes=history_sizes, current_stream=current_stream.hexdigest(),
                          history_stream=replay_stream.hexdigest(), survivors=survivors)
        write(job/'workload.json', workload)
        write(job/'completion.json', result)
        return result
    except BaseException as e:
        write(job/'failure.json', dict(error=repr(e), identity=identity, endpoints=endpoints,
              workload=workload, updates=updates, automatic_retry=False))
        raise


def restore(binding, task, endpoint, expected_identity, stage, factory=make_model):
    value = torch.load(endpoint, map_location='cpu', weights_only=False)
    if (value['identity'] != expected_identity or value['task'] != task or value['model_binding'] != binding
            or value['stage'] != stage):
        raise ValueError('Actual endpoint/data/config identity differs')
    model = factory(binding, task)
    if 'market_adapters' in task:
        from step35_market_adapters import restore_scoring_endpoint
        return restore_scoring_endpoint(model, value, expected_identity, stage), value['vendor_heads']
    if stage:
        model.encoder.load_state_dict(value['encoder'], strict=True)
    model.head.load_state_dict(value['head'], strict=True)
    model.eval()
    return model, value['vendor_heads']
