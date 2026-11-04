"""Frozen five systems, three stages and own-run history; no corpus access."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import random
import struct
import types
from typing import Any

import numpy as np

import step28_optuna as tuned
import step28_er_weight as weighted

record, parent, core, data = tuned.record, tuned.parent, tuned.core, tuned.data
base, diag = record.base, tuned.diag
METHODS = ("SEQ_ref", "ER0.1_ref", "LOGIT0.1_ref", "S_tuned", "H_pos_tuned")
ORDERS, SEEDS = ("ABC", "BCA", "CAB"), (4, 5, 6)
PARAMETERS = {"S_tuned": (1.3572393683805832, 1.295036258870551),
              "H_pos_tuned": (1.3051794547562026, 1.2832655111666686)}


def checkpointed_encoder_forward(encoder, *args, **kwargs):
    """Keep only microbatch inputs across the whole-group graph, not every layer.

    Instance-bound original forward preserves parameter names/state dictionaries.
    Recomputations receive fresh feature dictionaries; dropout RNG is preserved.
    """
    import torch
    from torch.utils.checkpoint import checkpoint
    def forward(*inputs, **options):
        copied = tuple(dict(v) if isinstance(v, dict) else v for v in inputs)
        return encoder._five_original_forward(*copied, **options)
    if encoder._five_outer_checkpoint and torch.is_grad_enabled() and encoder.training:
        return checkpoint(forward, *args, use_reentrant=False, preserve_rng_state=True, **kwargs)
    return forward(*args, **kwargs)


def checkpoint_encoder(encoder) -> None:
    if hasattr(encoder, "_five_original_forward"):
        raise ValueError("Do not install the per-model checkpoint wrapper twice")
    encoder._five_original_forward = encoder.forward
    encoder._five_outer_checkpoint = True
    encoder.forward = types.MethodType(checkpointed_encoder_forward, encoder)


@dataclass(frozen=True)
class Spec:
    name: str
    order: str
    seed: int

    def __post_init__(self) -> None:
        if self.name not in METHODS or self.order not in ORDERS or self.seed not in SEEDS:
            raise ValueError("Outside the approved 45-run list")

    @property
    def arm(self) -> str:
        return {"S_tuned": "S", "H_pos_tuned": "H_pos"}.get(self.name, self.name)

    @property
    def is_record(self) -> bool:
        return self.name in PARAMETERS

    @property
    def lambda_rank(self) -> float:
        return PARAMETERS.get(self.name, (1., 1.))[0]

    @property
    def lr_multiplier(self) -> float:
        return PARAMETERS.get(self.name, (1., 1.))[1]

    @property
    def roots(self) -> dict:
        return dict(initialization=2026101900+self.seed, schedule=2026102900+self.seed,
                    memory=2026103900+self.seed, partition=20260918, diagnostic=20261008)

    @property
    def run_id(self) -> str:
        return f"{self.order}_s{self.seed}_{self.name}"

    def identity(self) -> dict:
        return {**asdict(self), "roots": self.roots, "lambda_rank": self.lambda_rank,
                "lr_multiplier": self.lr_multiplier}

    @classmethod
    def from_dict(cls, row: dict) -> Spec:
        spec = cls(**{k: row[k] for k in ("name", "order", "seed")})
        if spec.identity() != row:
            raise ValueError("Frozen run identity differs")
        return spec

    def rates(self, step: int) -> tuple[float, float]:
        return self.lr_multiplier*parent.stage_lr(step), self.lr_multiplier*.001


def specifications() -> list[Spec]:
    return [Spec(name, order, seed) for order in ORDERS for seed in SEEDS for name in METHODS]


def config(spec: Spec) -> dict:
    c = tuned.config(spec) if spec.is_record else parent.config(parent.contract())
    c.update(initialization_seed=spec.roots["initialization"], schedule_seed=spec.roots["schedule"],
             continual_spec=spec.identity())
    return c


class RecordMemory(record.Memory):
    def __init__(self, spec: Spec):
        super().__init__(spec.order)
        self.spec = spec
        self.reservoir = data.Memory(data.seed_for(spec.roots["memory"], spec.order, "retention"))

    def begin_stage(self, stage: int) -> None:
        super().begin_stage(stage)
        self.draw_rng = random.Random(data.seed_for(self.spec.roots["memory"], self.order, stage, "history_draws"))

    def to_bytes(self) -> bytes:
        header = data.json_bytes(self.spec.identity())
        payload = struct.pack("<I", len(header))+header+super().to_bytes()
        if len(payload) > 1048576:
            raise ValueError("Complete history exceeds 1 MiB")
        return payload

    @classmethod
    def from_bytes(cls, payload: bytes, spec: Spec) -> RecordMemory:
        size, = struct.unpack("<I", payload[:4])
        if json.loads(payload[4:4+size]) != spec.identity():
            raise ValueError("Foreign history identity")
        old = record.Memory.from_bytes(payload[4+size:])
        result = cls(spec)
        result.__dict__.update(old.__dict__)
        if result.to_bytes() != payload:
            raise ValueError("History roundtrip differs")
        return result


class Method:
    def __init__(self, spec: Spec):
        self.spec = spec
        self.tuned = tuned.Method(spec) if spec.is_record else None

    def metadata(self) -> dict:
        return {"continual_spec": self.spec.identity(), "method": self.spec.name, "outer_encoder_recompute": True,
                "objective": self.tuned.metadata(self.spec.arm) if self.tuned else {
                    "architecture": "account_separate_moments", "rank": "original_rank",
                    "history_weight": 0. if self.spec.name == "SEQ_ref" else .1,
                    "teacher_weight": .5 if self.spec.name == "LOGIT0.1_ref" else 0.,
                    "hard_weight": .5}}

    def bind(self, model: Any, meta: dict) -> None:
        if any(meta.get(k) != v for k, v in self.metadata().items()):
            raise ValueError("Wrong restored method/order/seed")
        model.continual_identity = self.metadata()
        if self.tuned:
            self.tuned.bind(model, meta["objective"])

    def check_model(self, model: Any) -> None:
        self.bind(model, getattr(model, "continual_identity", {}))

    def fresh(self, c: dict, device: str = "cuda:0") -> tuple:
        random.seed(self.spec.roots["initialization"])
        np.random.seed(self.spec.roots["initialization"])
        model = base.load_model(c, "split_rank", device)
        checkpoint_encoder(model.encoder)
        optimizer = core.make_optimizer(model, c)
        self.bind(model, self.metadata())
        return model, optimizer

    def score(self, model: Any, groups: list, c: dict, check=lambda: None) -> np.ndarray:
        self.check_model(model)
        if self.tuned:
            return self.tuned.score(model, groups, c, self.spec.arm, check)
        return base.score(model, groups, c, "split_rank", check)

    def update(self, model, optimizer, current, history, reference, c, stage, step,
               check=lambda: None, observe=None) -> dict:
        import torch
        self.check_model(model)
        spec = self.spec
        memory_expected = stage > 1 and spec.name != "SEQ_ref"
        target_expected = memory_expected and spec.name != "ER0.1_ref"
        if (stage not in (1, 2, 3) or c.get("continual_spec") != spec.identity()
                or parent.adam_step(optimizer) != (stage-1)*288+step-1
                or (history is not None) != memory_expected or (reference is not None) != target_expected):
            raise ValueError("Actual update stage/history/Adam identity differs")
        seeds = [data.seed_for(spec.roots["schedule"], spec.order, stage, step, role, "dropout")
                 for role in ("current", "history")]
        if not spec.is_record:
            if not memory_expected:
                return parent.update(model, optimizer, current, None, None, c, "seq", stage, step,
                                     *seeds, check=check)
            return weighted.update(model, optimizer, current, history, c, .1, stage, step, *seeds,
                                   reference=reference, logit_weight=.5 if target_expected else 0., check=check)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        rates = spec.rates(step)
        for group, rate in zip(optimizer.param_groups, rates, strict=True):
            group["lr"] = rate
        logs = {}
        for role, group, target, dropout in (("current", current, None, seeds[0]),
                                            ("history", history, reference, seeds[1])):
            if group is None:
                continue
            regroup = data.seed_for(spec.roots["schedule"], spec.order, stage, step, role, group.uid, "regroup")
            logs[role] = record.backward_group(model, group, c, spec.arm, dropout, regroup, target, check,
                                               objective_fn=self.tuned.objective)
            logs[role]["assignment_sha256"] = hashlib.sha256(record.assignment(group, regroup).tobytes()).hexdigest()
            if observe:
                observe(role, logs[role])
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
        check()
        optimizer.step()
        if parent.adam_step(optimizer) != (stage-1)*288+step:
            raise ValueError("Actual optimizer call missing")
        return {**logs, "gradient_norm": float(norm), "encoder_lr": rates[0], "head_lr": rates[1],
                "total": sum(v["weighted"] for v in logs.values())}

    def new_memory(self, first_map: dict):
        if self.spec.name == "SEQ_ref":
            return None
        if self.spec.is_record:
            return RecordMemory(self.spec)
        return parent.Memory(self.spec.order, self.spec.roots["memory"],
                             self.spec.name == "LOGIT0.1_ref", first_map)

    def memory_from(self, payload: bytes):
        if self.spec.is_record:
            return RecordMemory.from_bytes(payload, self.spec)
        memory = parent.Memory.from_bytes(payload)
        if (memory.order != self.spec.order or memory.seed != self.spec.roots["memory"]
                or memory.with_logits != (self.spec.name == "LOGIT0.1_ref")
                or memory.auxiliary.get("continual_spec") != self.spec.identity()):
            raise ValueError("Wrong restored baseline history")
        return memory

    def retain(self, memory, model, current, c, stage, maps, check=lambda: None) -> None:
        if memory is None:
            return
        aux = {**maps, **self.metadata(), "rng": diag.rng_state(), "completed_stage": stage}
        if self.spec.is_record:
            memory.maps = aux
            if stage < 3:
                memory.retain(current, stage, lambda g: record.reference(model, g, c, check))
        else:
            memory.auxiliary = aux
            if stage < 3:
                memory.retain(current, stage, lambda groups: self.score(model, groups, c, check))
        memory.to_bytes()

    def restore(self, path, receipt, model, optimizer):
        meta = core.restore_state(path, model, optimizer, receipt["state_sha256"])
        self.bind(model, meta)
        for module, mode in zip(model.modules(), meta["modes"], strict=True):
            module.training = mode
        diag.restore_rng(meta["rng"])
        return meta
