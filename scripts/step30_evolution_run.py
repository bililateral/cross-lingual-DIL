"""Actual Evolution updates and compact inference endpoints. No epoch saves."""
from __future__ import annotations

from dataclasses import asdict
import gc
import json
import random
from pathlib import Path

import torch
from torch.nn import functional as F

from step29_vendorlink_data import Group, seed_for
from step29_vendorlink_methods import Memory, Model, Settings, make_optimizer, update_h
from step29_vendorlink_run import set_rates
from step30_evolution_data import validate_group
from step30_evolution_policy import DOMAINS, SPLITS


def make_model(model_path: Path, task: dict, device: str = 'cuda:0', *, expected_hidden: int = 768) -> Model:
    from transformers import BertModel, BertTokenizer
    torch.manual_seed(seed_for(task['seed'], DOMAINS, 'initial'))
    encoder = BertModel.from_pretrained(model_path, local_files_only=True)
    if encoder.config.hidden_size != expected_hidden:
        raise ValueError('The approved pretrained backbone is BERT base with hidden size 768')
    tokenizer = BertTokenizer.from_pretrained(model_path, local_files_only=True, do_lower_case=False)
    if tokenizer.do_lower_case:
        raise ValueError('Evolution requires the shared case-preserving tokenizer')
    return Model(encoder, tokenizer, Settings(**task['settings'])).to(device)


class VendorMemory(Memory):
    """Same Algorithm R and draws as H; no H teacher computation or payload."""
    def retain(self, groups, stage, model=None, check=lambda: None, **unused):
        if stage not in (1, 2):
            raise ValueError('Only completed nonfinal stages may produce memory')
        selected = list(self.entries)
        for group in sorted(groups, key=lambda g: g.uid):
            check()
            self.seen += 1
            slot = len(selected) if len(selected) < self.settings.memory_groups else self.rng.randrange(self.seen)
            if slot < self.settings.memory_groups:
                row = dict(group=asdict(group), origin=stage, teacher=None)
                if slot == len(selected):
                    selected.append(row)
                else:
                    selected[slot] = row
        self.entries = selected
        if len(self.bytes()) > self.settings.memory_bytes:
            raise ValueError('Vendor retained text/mapping/sampler exceeds one MiB')

    def state(self):
        return {**super().state(), 'classes': self.classes()}

    def classes(self):
        return sorted({v['identity'] for row in self.entries for v in row['group']['views']})


def vendor_backward(model, records, identities, step_seed, weight, check):
    torch.manual_seed(step_seed)
    lookup = {v: i for i, v in enumerate(model.classes)}
    labels = torch.tensor([lookup[x] for x in identities], device=next(model.parameters()).device)
    check()
    output, _ = model.encode([r.title+' [SEP] '+r.description for r in records], model.settings.vendor_tokens)
    loss = F.cross_entropy(model.classifier(model.dropout(output.pooler_output)), labels)
    if not torch.isfinite(loss):
        raise ValueError('Nonfinite current/history classification loss')
    (weight*loss).backward()
    return float(loss.detach())


def update_vendor(model, optimizer, examples, history, step_seed, check):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    records, identities = zip(*examples)
    current = vendor_backward(model, records, identities, seed_for(step_seed, 'current'), 1., check)
    logs = dict(current_ce=current)
    if history is not None:
        old_records, old_ids = zip(*[(r, v.identity) for v in history.views for r in v.records])
        logs['history_ce'] = vendor_backward(model, old_records, old_ids, seed_for(step_seed, 'history'), .1, check)
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
    check()
    optimizer.step()
    return dict(**logs, gradient_norm=float(norm), optimizer_updates=1)


def atomic_tensor(path: Path, state, check):
    if path.exists() or path.with_suffix('.partial').exists():
        raise ValueError('No checkpoint overwrite/recovery is permitted')
    check()
    temporary = path.with_suffix('.partial')
    torch.save(state, temporary)
    temporary.replace(path)  # Same-volume rename; no second payload copy.
    check()


def execute(model: Model, load_fit, job: Path, task: dict, identity: dict, model_binding: dict,
            write, record, check=lambda: None, *, expected_groups: int = 20) -> dict:
    if job.exists() or task['stages'] not in (1, 3) or task['epochs'] < 1:
        raise ValueError('Fresh single/full trajectory required')
    if asdict(model.settings) != task['settings']:
        raise ValueError('Actual model recipe does not match task')
    job.mkdir(parents=True)
    s, seed, system = model.settings, task['seed'], task['system']
    vendor, replay = system.startswith('V_'), system in ('H_ER', 'H_full', 'V_ER')
    if vendor != (s.method == 'VendorLink_SEQ'):
        raise ValueError('Wrong actual model family')
    memory = (VendorMemory if vendor else Memory)(seed_for(seed, DOMAINS, 'reservoir'), s)
    optimizer, endpoints, updates = None, {}, 0
    workload = dict(current_updates=0, current_records=0, history_groups=0, history_records=0,
                    teacher_groups=0, teacher_records=0, stage_model_saves=0)
    initial = dict(identity=identity, task=task, model_binding=model_binding,
                   head={k: v.detach().cpu().clone() for k, v in model.head.state_dict().items()}, stage=0)
    atomic_tensor(job/'initial.pt', initial, check)
    endpoints['0'] = record(job/'initial.pt')
    try:
        for stage in range(task['stages']):
            check()
            # The loader is called once here for this CURRENT domain only.
            groups = load_fit(DOMAINS[stage])
            if len(groups) != expected_groups or any(g.domain != DOMAINS[stage] or g.split != 'fit' for g in groups):
                raise ValueError('Current fit quota/domain mismatch')
            for group in groups:
                validate_group(group, True)
            classes = sorted({v.identity for g in groups for v in g.views})
            if len(classes) != expected_groups*8:
                raise ValueError('Fit accounts must be disjoint across groups')
            record_ids = [r.uid for g in groups for v in g.views for r in v.records]
            if len(set(record_ids)) != expected_groups*32 or len({g.uid for g in groups}) != expected_groups:
                raise ValueError('Fit record/group identities overlap')
            del record_ids
            if vendor:
                old_classes = memory.classes() if replay else []
                if set(classes) & set(old_classes):
                    raise ValueError('Current fit and retained history identities overlap')
                model.start_vendor_stage(sorted(set(classes) | set(old_classes)), seed_for(seed, DOMAINS, stage, 'head'))
                optimizer = make_optimizer(model)
            elif optimizer is None:
                optimizer = make_optimizer(model)
            for epoch in range(task['epochs']):
                stream = random.Random(seed_for(seed, DOMAINS, stage, epoch, 'current'))
                if vendor:
                    examples = [(r, v.identity) for g in groups for v in g.views for r in v.records]
                    stream.shuffle(examples)
                    batches = [examples[k:k+32] for k in range(0, len(examples), 32)]
                    del examples
                else:
                    batches = list(groups)
                    stream.shuffle(batches)
                for batch_index, batch in enumerate(batches):
                    check()
                    update = epoch*expected_groups+batch_index+1
                    step_seed = seed_for(seed, DOMAINS, stage, update, 'update')
                    history, teacher = (None, None)
                    if stage and replay:
                        history, teacher = memory.draw(seed_for(seed, DOMAINS, stage, update, 'history'))
                    set_rates(optimizer, s, update, task['epochs']*expected_groups)
                    logs = (update_vendor(model, optimizer, batch, history, step_seed, check) if vendor else
                            update_h(model, optimizer, batch, history, teacher, step_seed, check))
                    updates += 1
                    workload['current_updates'] += 1
                    workload['current_records'] += 32
                    if history is not None:
                        workload['history_groups'] += 1
                        workload['history_records'] += 32
                    row = dict(stage=stage+1, epoch=epoch+1, update=update, total_updates=updates,
                               current_ids=[r.uid for r, _ in batch] if vendor else [batch.uid],
                               history_group=None if history is None else history.uid,
                               rates=[g['lr'] for g in optimizer.param_groups], **logs)
                    with (job/'updates.jsonl').open('a', encoding='utf-8') as output:
                        output.write(json.dumps(row, allow_nan=False)+'\n')
                write(job/'workload.json', workload)
                del batches, batch, history, teacher
            if replay and stage+1 < task['stages']:
                def teacher_done(group):
                    workload['teacher_groups'] += 1
                    workload['teacher_records'] += 32
                    write(job/'workload.json', workload)
                memory.retain(groups, stage+1, model, check, record_teacher=teacher_done)
                write(job/f'memory_stage{stage+1}.json', dict(bytes=len(memory.bytes()), seen=memory.seen,
                    groups=[dict(uid=r['group']['uid'], domain=r['group']['domain'], origin=r['origin']) for r in memory.entries]))
            endpoint = dict(identity=identity, task=task, model_binding=model_binding, stage=stage+1, updates=updates,
                            encoder=model.encoder.state_dict(), head=model.head.state_dict())
            # Classifier is training-only; all H/V scores restore encoder + H head.
            path = job/f'stage{stage+1}.pt'
            atomic_tensor(path, endpoint, check)
            endpoints[str(stage+1)] = record(path)
            workload['stage_model_saves'] += 1
            write(job/'workload.json', workload)
            del endpoint, groups, classes
        result = dict(status='TRAINING_COMPLETE_EVALUATION_CLOSED', identity=identity, task=task,
                      endpoints=endpoints, workload=workload, updates=updates)
        write(job/'completion.json', result)
        return result
    except BaseException as exc:
        write(job/'failure.json', dict(error=repr(exc), updates=updates, workload=workload,
                                      endpoints=endpoints, identity=identity, automatic_retry=False))
        raise


def restore(factory, task: dict, endpoint: Path, expected_identity: dict, binding: dict, stage: int):
    value = torch.load(endpoint, map_location='cpu', weights_only=False)
    if (value['identity'] != expected_identity or value['task'] != task or value['stage'] != stage
            or value['model_binding'] != binding):
        raise ValueError('Inference endpoint/config/pretrained identity differs')
    model = factory(task)
    if stage:
        model.encoder.load_state_dict(value['encoder'], strict=True)
    # For stage zero the factory restores the pinned pretrained encoder and the
    # ACTUAL small head below. V has an empty Identity head; no fit access.
    model.head.load_state_dict(value['head'], strict=True)
    model.eval()
    return model
