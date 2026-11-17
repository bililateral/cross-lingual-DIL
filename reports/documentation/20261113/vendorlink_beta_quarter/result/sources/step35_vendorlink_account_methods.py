"""Current-account CE on the original H representation and encoder gradient path."""
from contextlib import contextmanager
from dataclasses import asdict, replace
from pathlib import Path
import math

import torch
from torch import nn
from torch.nn import functional as F

from step29_vendorlink_data import seed_for
from step29_vendorlink_methods import Model, assignment, h_pool, h_objective, make_optimizer
from step29_vendorlink_run import rng_state, restore_rng, set_rates


@contextmanager
def deterministic_student(encoder):
    """Keep BERT training/checkpointing, with zero dropout through recomputation."""
    from transformers.models.bert.modeling_bert import BertSdpaSelfAttention
    changed = []
    device = next(encoder.parameters()).device
    devices = [device.index or 0] if device.type == 'cuda' else []
    with torch.random.fork_rng(devices=devices):
        try:
            for module in encoder.modules():
                attribute = ('p' if isinstance(module, nn.Dropout) else
                             'dropout_prob' if isinstance(module, BertSdpaSelfAttention) else None)
                if attribute is not None:
                    changed.append((module, attribute, getattr(module, attribute)))
                    setattr(module, attribute, 0.)
            yield
        finally:
            for module, attribute, value in changed:
                setattr(module, attribute, value)


class AccountModel(Model):
    def __init__(self, bert, tokenizer, settings):
        super().__init__(bert, tokenizer, settings)
        self.auxiliary_head = None
        self.auxiliary_classes = ()
        self.auxiliary_optimizer = None

    def ce_scale(self):
        return math.sqrt(2*self.encoder.config.hidden_size) if self.settings.ce_scale_mode == 'rms' else 1.

    def start_account_stage(self, classes, initialization_seed):
        if self.settings.method == 'VendorLink_SEQ' or len(classes) < 2 or len(set(classes)) != len(classes):
            raise ValueError('Unique current H account classes required')
        device = next(self.encoder.parameters()).device
        devices = [device.index or 0] if device.type == 'cuda' else []
        self.auxiliary_classes = tuple(sorted(classes))
        # Fork covers nn.Linear's default constructor initialization as well.
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(initialization_seed)
            head = nn.Linear(2*self.encoder.config.hidden_size, len(classes)).to(device)
            nn.init.normal_(head.weight, std=.02)
            nn.init.zeros_(head.bias)
        self.auxiliary_head = head
        self.auxiliary_optimizer = torch.optim.AdamW(head.parameters(), lr=self.settings.auxiliary_lr,
            betas=(.9, .999), eps=1e-8, weight_decay=.01, amsgrad=False)

    def backward_h(self, group, step_seed, teacher=None, history=False, check=lambda: None):
        torch.manual_seed(step_seed)
        z = self.records(group, check)  # Original normalized channels / sqrt(2).
        with torch.no_grad():
            values = self.table(z.detach())
        proxy = values.detach().requires_grad_(True)
        objective_settings = (replace(self.settings, method='H_ER', deterministic_history=False)
                              if history and self.settings.deterministic_history else self.settings)
        loss, logs = h_objective(proxy, group, seed_for(step_seed, 'regroup'), objective_settings, teacher, history)
        gradient, = torch.autograd.grad(loss, proxy)
        leaf = self.pair_backward(z, gradient, check)
        logs['record_gradient_H_norm'] = float(leaf.grad.detach().norm())
        if not history:
            if self.auxiliary_head is None or self.auxiliary_optimizer is None:
                raise ValueError('Current CE head and optimizer must be initialized')
            lookup = {identity: k for k, identity in enumerate(self.auxiliary_classes)}
            labels = torch.tensor([lookup[v.identity] for v in group.views for _ in v.records], device=z.device)
            logits = self.auxiliary_head(leaf * self.ce_scale())
            ce = F.cross_entropy(logits, labels, reduction='mean')
            ce_gradient, = torch.autograd.grad(ce, leaf, retain_graph=True)
            logs.update(ce=float(ce.detach()), gamma=self.settings.gamma, ce_records=len(labels),
                        record_gradient_CE_norm=float(ce_gradient.detach().norm()),
                        current_weighted=logs['weighted']+self.settings.gamma*float(ce.detach()))
            if self.settings.diagnostics:
                with torch.no_grad():
                    values = logits.detach()
                    wrong = values.clone()
                    wrong.scatter_(1, labels[:, None], -torch.inf)
                    margins = values.gather(1, labels[:, None]).squeeze(1)-wrong.amax(1)
                    norms = self.auxiliary_head.weight.detach().norm(dim=1)
                    logs.update(ce_scale=self.ce_scale(),
                        ce_correct=int((values.argmax(1) == labels).sum()),
                        ce_margin_sum=float(margins.sum()),
                        ce_logit_std_sum=float(values.std(dim=1, unbiased=False).sum()),
                        auxiliary_row_norm_mean=float(norms.mean()),
                        auxiliary_effective_row_norm_mean=float(norms.mean())*self.ce_scale())
            if self.settings.gamma:
                (self.settings.gamma*ce).backward()
        if leaf.grad is None or not torch.isfinite(leaf.grad).all():
            raise ValueError('Missing/nonfinite encoder-chain gradient')
        z.backward(leaf.grad)  # H + CE reach the same, single original encoder graph.
        return logs

    def pair_backward(self, z, gradient, check):
        leaf = z.detach().requires_grad_(True)
        i, j = torch.triu_indices(len(z), len(z), 1, device=z.device)
        for k in range(0, len(i), 512):
            check()
            u, v = leaf[i[k:k+512]], leaf[j[k:k+512]]
            self.head(torch.cat(((u-v).abs(), u*v), 1)).flatten().backward(gradient[k:k+512])
        return leaf

    def backward_birth(self, group, teacher, check=lambda: None):
        if not self.settings.deterministic_history or teacher is None:
            raise ValueError('Deterministic Full and retained birth teacher required')
        # The context covers z.backward: checkpoint recomputation must see p=0.
        with deterministic_student(self.encoder):
            z = self.records(group, check)
            with torch.no_grad():
                values = self.table(z.detach())
            proxy = values.detach().requires_grad_(True)
            slots = assignment(group)
            target = h_pool(teacher.to(proxy), slots).detach()
            mse = (h_pool(proxy, slots)-target).square().mean()
            loss = self.settings.beta*mse
            gradient, = torch.autograd.grad(loss, proxy)
            leaf = self.pair_backward(z, gradient, check)
            if leaf.grad is None or not torch.isfinite(leaf.grad).all():
                raise ValueError('Missing/nonfinite deterministic preservation gradient')
            z.backward(leaf.grad)
            return dict(distillation=float(mse.detach()), weighted=float(loss.detach()),
                        record_gradient_norm=float(leaf.grad.detach().norm()), deterministic=True)


def gradient_norm(parameters):
    grads = [p.grad.detach().float().norm() for p in parameters if p.grad is not None]
    return float(torch.stack(grads).norm()) if grads else 0.


def update_account(model, optimizer, current, history, teacher, step_seed, check=lambda: None):
    if model.settings.method not in ('H_SEQ', 'H_ER', 'H_full'):
        raise ValueError('H method required')
    auxiliary = model.auxiliary_optimizer
    if auxiliary is None:
        raise ValueError('Missing current auxiliary optimizer')
    original = [p for group in optimizer.param_groups for p in group['params']]
    extra = [p for group in auxiliary.param_groups for p in group['params']]
    if len({id(p) for p in original+extra}) != len(original)+len(extra):
        raise ValueError('Encoder/pair and auxiliary optimizers overlap')
    model.train()
    optimizer.zero_grad(set_to_none=True)
    auxiliary.zero_grad(set_to_none=True)
    logs = dict(current=model.backward_h(current, seed_for(step_seed, 'current'), check=check))
    logs['current_encoder_gradient_norm'] = gradient_norm(model.encoder.parameters())
    if history is not None:
        logs['history'] = model.backward_h(history, seed_for(step_seed, 'history'), teacher, True, check)
        if model.settings.deterministic_history:
            birth = model.backward_birth(history, teacher, check)
            logs['deterministic_birth'] = birth
            logs['history']['distillation'] = birth['distillation']
            logs['history']['weighted'] += birth['weighted']
    logs['total_encoder_gradient_norm'] = gradient_norm(model.encoder.parameters())
    norm = torch.nn.utils.clip_grad_norm_(original+extra, 1., error_if_nonfinite=True)
    check()
    optimizer.step()
    if model.settings.gamma:
        auxiliary.step()
    return dict(**logs, gradient_norm=float(norm), optimizer_updates=1,
                auxiliary_optimizer_updates=int(bool(model.settings.gamma)))


def set_account_rates(model, optimizer, update, total):
    set_rates(optimizer, model.settings, update, total)
    if model.settings.method != 'VendorLink_SEQ':
        # H1's pair-head and auxiliary head have the same fixed stage schedule.
        factor = optimizer.param_groups[1]['lr']/model.settings.head_lr
        model.auxiliary_optimizer.param_groups[0]['lr'] = model.settings.auxiliary_lr*factor


def save_account_state(path, model, optimizer, memory, cursor, identity):
    value = dict(identity=identity, settings=asdict(model.settings), model=model.state_dict(),
                 classes=model.classes, optimizer=optimizer.state_dict(), memory=memory.state(),
                 cursor=cursor, rng=rng_state(), modes=[m.training for m in model.modules()],
                 auxiliary_classes=model.auxiliary_classes,
                 auxiliary_optimizer=None if model.auxiliary_optimizer is None else model.auxiliary_optimizer.state_dict())
    temporary = Path(path).with_suffix('.partial')
    torch.save(value, temporary)
    temporary.replace(path)


def load_account_state(path, model, memory, identity):
    value = torch.load(path, map_location='cpu', weights_only=False)
    if value['identity'] != identity or value['settings'] != asdict(model.settings):
        raise ValueError('Recovery identity differs')
    if model.settings.method == 'VendorLink_SEQ':
        model.start_vendor_stage(value['classes'], 0)
    else:
        model.start_account_stage(value['auxiliary_classes'], 0)
    model.load_state_dict(value['model'], strict=True)
    optimizer = make_optimizer(model)
    optimizer.load_state_dict(value['optimizer'])
    if model.auxiliary_optimizer is not None:
        model.auxiliary_optimizer.load_state_dict(value['auxiliary_optimizer'])
    memory.restore(value['memory'])
    for module, mode in zip(model.modules(), value['modes'], strict=True):
        module.training = mode
    restore_rng(value['rng'])
    return optimizer, value['cursor']
