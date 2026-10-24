"""Fixed L2R/GCR implementations, separate from the active five-method campaign."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import random
from typing import Any

import numpy as np

import step28_continual_five as five
import step28_comparison_l2r as l2r
import step28_comparison_gcr as gcr

data, core, base, parent, diag = five.data, five.core, five.base, five.parent, five.diag
NEW_METHODS = ('L2R_text_fixed', 'ER_GCR_group_fixed')
METHODS = ('H_pos_tuned', 'S_tuned', *NEW_METHODS)
ORDERS, SEEDS = five.ORDERS, five.SEEDS


@dataclass(frozen=True)
class Spec:
    name: str
    order: str
    seed: int

    def __post_init__(self):
        if self.name not in METHODS or self.order not in ORDERS or self.seed not in SEEDS:
            raise ValueError('Outside the four fixed comparison systems')

    @property
    def roots(self) -> dict:
        return dict(initialization=2026101900+self.seed, schedule=2026102900+self.seed,
                    memory=2026103900+self.seed, partition=20260918)

    @property
    def run_id(self) -> str:
        return f'{self.order}_s{self.seed}_{self.name}'

    def identity(self) -> dict:
        return {**asdict(self), 'roots': self.roots}

    @classmethod
    def from_dict(cls, row: dict) -> Spec:
        result = cls(**{k: row[k] for k in ('name', 'order', 'seed')})
        if result.identity() != row:
            raise ValueError('Comparison spec differs')
        return result


def specifications(*, new_only: bool = False) -> list[Spec]:
    return [Spec(name, order, seed) for order in ORDERS for seed in SEEDS
            for name in (NEW_METHODS if new_only else METHODS)]


def config(spec: Spec) -> dict:
    if spec.name not in NEW_METHODS:
        return five.config(five.Spec(spec.name, spec.order, spec.seed))
    c = five.record.config()
    c.update(initialization_seed=spec.roots['initialization'], schedule_seed=spec.roots['schedule'],
             comparison_spec=spec.identity())
    return c


class Method:
    def __init__(self, spec: Spec):
        if spec.name not in NEW_METHODS:
            raise ValueError('H/S are inference-only existing systems, never retrained here')
        self.spec = spec
        self.algorithm = l2r if spec.name == 'L2R_text_fixed' else gcr

    def metadata(self) -> dict:
        return dict(comparison_spec=self.spec.identity(), method=self.spec.name,
                    architecture='raw_cls_account_dot' if self.algorithm is l2r else 'record_mean_mlp',
                    definition='20261024_final_design', outer_encoder_recompute=True)

    def bind(self, model, metadata: dict) -> None:
        if any(metadata.get(k) != v for k, v in self.metadata().items()):
            raise ValueError('Restored method identity differs')
        model.comparison_identity = self.metadata()

    def fresh(self, c: dict, device: str = 'cuda:0') -> tuple:
        import torch
        random.seed(self.spec.roots['initialization'])
        np.random.seed(self.spec.roots['initialization'])
        model = base.load_model(c, 'split_rank', device)
        if self.algorithm is l2r:
            del model.head
            # Actual raw CLS path calls the Transformer, never ST Normalize.
            five.checkpoint_encoder(model.encoder[0])
            p = c['optimizer']
            optimizer = torch.optim.AdamW(model.encoder.parameters(), lr=1e-5,
                weight_decay=p['encoder_weight_decay'], betas=tuple(p['betas']), eps=p['eps'], foreach=p['foreach'])
        else:
            five.checkpoint_encoder(model.encoder)
            optimizer = core.make_optimizer(model, c)
        self.bind(model, self.metadata())
        return model, optimizer

    def memory(self):
        return self.algorithm.Memory(self.spec.identity())

    def memory_from(self, payload: bytes):
        return self.algorithm.Memory.from_bytes(payload, self.spec.identity())

    def stage(self, number: int, current: list, memory, model):
        if len({g.uid for g in current}) != 48:
            raise ValueError('48 distinct current fitting groups required')
        return self.algorithm.Stage(self.spec.identity(), number, current, memory, model)

    def score(self, model, groups: list, c: dict, check=lambda: None) -> np.ndarray:
        import torch
        self.bind(model, getattr(model, 'comparison_identity', {}))
        if self.algorithm is gcr:
            return five.record.score(model, groups, c, check)
        model.eval()
        values = []
        with torch.inference_mode():
            for group in groups:
                z = l2r.encode(model, l2r.accounts(group, 'evaluation', supervised=False), c, check)
                i, j = torch.triu_indices(28, 28, 1, device=z.device)
                values.append((z[i]*z[j]).sum(1).cpu().numpy())
        result = np.stack(values).astype(np.float32)
        if result.shape != (len(groups), 378) or not np.isfinite(result).all():
            raise ValueError('Incomplete finite L2R all-pair scores')
        return result

    def update(self, model, optimizer, stage, group, c: dict, step: int, check=lambda: None) -> dict:
        import torch
        self.bind(model, getattr(model, 'comparison_identity', {}))
        number = stage.stage
        if (c.get('comparison_spec') != self.spec.identity() or not 1 <= step <= 288
                or parent.adam_step(optimizer) != (number-1)*288+step-1):
            raise ValueError('Current method or continuous Adam step differs')
        optimizer.zero_grad(set_to_none=True)
        peak = 3e-6 if self.algorithm is l2r and number > 1 else 1e-5
        optimizer.param_groups[0]['lr'] = parent.stage_lr(step)*(peak/1e-5)
        if self.algorithm is gcr:
            optimizer.param_groups[1]['lr'] = .001
        rows = stage.backward(model, group, c, step, check)
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
        check()
        optimizer.step()
        if self.algorithm is gcr:
            stage.after_update(group)  # Admission happens after the first real optimizer update.
        if parent.adam_step(optimizer) != (number-1)*288+step:
            raise ValueError('Actual optimizer update missing')
        # Full selections remain available to bounded handwritten probes, not huge per-step logs.
        for row in rows.values():
            if 'selections' in row:
                selections = row.pop('selections')
                row['selection_sha256'] = hashlib.sha256(data.json_bytes(selections)).hexdigest()
                row['denominator_min'] = min(len(v['denominator']) for v in selections)
                row['denominator_max'] = max(len(v['denominator']) for v in selections)
        return dict(**rows, gradient_norm=float(norm), total=sum(v['weighted'] for v in rows.values()),
                    learning_rates=[g['lr'] for g in optimizer.param_groups])

    def restore(self, path, receipt: dict, model, optimizer):
        metadata = core.restore_state(path, model, optimizer, receipt['state_sha256'])
        self.bind(model, metadata)
        for module, training in zip(model.modules(), metadata['modes'], strict=True):
            module.training = training
        diag.restore_rng(metadata['rng'])
        return metadata


def inference_method(spec: Spec):
    return Method(spec) if spec.name in NEW_METHODS else five.Method(five.Spec(spec.name, spec.order, spec.seed))
