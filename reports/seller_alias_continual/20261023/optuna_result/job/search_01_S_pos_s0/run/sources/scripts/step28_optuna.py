"""Explicit three-arm trial specification and actual record-VJP training.

No data loading, Optuna study or model construction occurs on import.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
import random
import struct
from typing import Any

import numpy as np

import step28_positive_ranking as positive
import step28_replay_diagnostics as diag

record, data = positive.record, positive.data
parent, core = record.parent, record.core
ARMS = ("S", "S_pos", "H_pos")
RNG_ORDER = "ABC"


@dataclass(frozen=True)
class TrialSpec:
    arm: str
    phase: str
    number: int
    lambda_rank: float = 1.
    lr_multiplier: float = 1.
    seed: int = 0

    def __post_init__(self) -> None:
        if self.arm not in ARMS or self.phase not in ("search", "replication", "handwritten"):
            raise ValueError("Unknown trial role")
        if type(self.number) is not int or not 0 <= self.number < 12:
            raise ValueError("Trial number out of range")
        if type(self.seed) is not int or self.seed not in range(4):
            raise ValueError("Unknown seed")
        if self.phase == "search" and self.seed != 0:
            raise ValueError("All search trials use s0")
        if self.phase == "replication" and (self.seed == 0 or self.number != self.seed-1):
            raise ValueError("Replication must use its prescribed seed slot")
        if not math.isfinite(self.lambda_rank) or not .25 <= self.lambda_rank <= 2:
            raise ValueError("Rank weight outside frozen space")
        if not math.isfinite(self.lr_multiplier) or not .5 <= self.lr_multiplier <= 2:
            raise ValueError("LR outside frozen space")
        if self.phase == "search" and self.number == 0 and (self.lambda_rank, self.lr_multiplier) != (1., 1.):
            raise ValueError("Default anchor is fixed")

    @property
    def roots(self) -> dict:
        return dict(initialization=20260909 if not self.seed else 2026101900+self.seed,
                    schedule=20260918 if not self.seed else 2026102900+self.seed,
                    memory=20260930 if not self.seed else 2026103900+self.seed,
                    partition=20260918, diagnostic=20261008, statistics=20260930)

    @property
    def run_id(self) -> str:
        return f"{self.phase}_{self.number:02}_{self.arm}_s{self.seed}"

    def identity(self) -> dict:
        return {**asdict(self), "roots": self.roots, "rng_order": RNG_ORDER}

    @classmethod
    def from_dict(cls, value: dict) -> TrialSpec:
        result = cls(**{k: value[k] for k in cls.__dataclass_fields__})
        if value != result.identity():
            raise ValueError("Trial identity/roots differ")
        return result

    def rates(self, step: int) -> tuple[float, float]:
        return self.lr_multiplier*parent.stage_lr(step), self.lr_multiplier*.001


def config(spec: TrialSpec) -> dict:
    c = record.config()
    c.update(initialization_seed=spec.roots["initialization"], schedule_seed=spec.roots["schedule"],
             optuna_trial=spec.identity())
    # Partition remains fixed even when training schedule shares its former numeric root.
    if c["partition"]["seed"] != 20260918:
        raise ValueError("Fixed partition changed")
    c["optimizer"] = {**c["optimizer"], "encoder_lr": spec.lr_multiplier*c["optimizer"]["encoder_lr"],
                      "head_lr": spec.lr_multiplier*.001}
    return c


class Memory(record.Memory):
    """Own-run envelope; legacy payload and future draw root are both restored."""

    def __init__(self, spec: TrialSpec):
        super().__init__(RNG_ORDER)
        self.spec = spec
        self.reservoir = data.Memory(data.seed_for(spec.roots["memory"], RNG_ORDER, "retention"))

    def begin_stage(self, stage: int) -> None:
        super().begin_stage(stage)
        self.draw_rng = random.Random(data.seed_for(self.spec.roots["memory"], RNG_ORDER, stage, "history_draws"))

    def to_bytes(self) -> bytes:
        header = data.json_bytes({"format": "optuna_own_memory_v1", "trial": self.spec.identity()})
        payload = struct.pack("<I", len(header))+header+super().to_bytes()
        if len(payload) > 1048576:
            raise ValueError("Complete trial Memory exceeds 1 MiB")
        return payload

    @classmethod
    def from_bytes(cls, payload: bytes, spec: TrialSpec) -> Memory:
        size, = struct.unpack("<I", payload[:4])
        header = json.loads(payload[4:4+size])
        if header != {"format": "optuna_own_memory_v1", "trial": spec.identity()}:
            raise ValueError("Foreign run Memory")
        old = record.Memory.from_bytes(payload[4+size:])
        result = cls(spec)
        result.__dict__.update(old.__dict__)
        if result.to_bytes() != payload:
            raise ValueError("Trial Memory roundtrip differs")
        return result


class Method:
    """Per-run adapter; no global monkey patches or shared model/optimizer state."""

    RNG_ORDER = RNG_ORDER

    def __init__(self, spec: TrialSpec):
        self.spec = spec
        self.ARMS = (spec.arm,)

    def identity(self, arm: str) -> dict:
        self.check_arm(arm)
        return positive.identity("H_pos" if arm == "H_pos" else "S_pos")

    def check_arm(self, arm: str) -> None:
        if arm != self.spec.arm:
            raise ValueError("Foreign candidate")

    def metadata(self, arm: str) -> dict:
        return dict(pooling=self.identity(arm), trial=self.spec.identity(), training=dict(
            arm=arm, rank="original_rank" if arm == "S" else "positive_rank",
            lambda_rank=self.spec.lambda_rank, history_supervision=.1, teacher_A0=.5, hard_weight=.5))

    def bind(self, model: Any, meta: dict) -> str:
        expected = self.metadata(self.spec.arm)
        if any(meta.get(k) != v for k, v in expected.items()):
            raise ValueError("Restored trial specification differs")
        model.optuna_identity = expected
        return self.spec.arm

    def check_model(self, model: Any) -> None:
        self.bind(model, getattr(model, "optuna_identity", {}))

    def account_logits(self, table: Any, slots: np.ndarray, arm: str) -> Any:
        self.check_arm(arm)
        return positive.account_logits(table, slots, "H_pos" if arm == "H_pos" else "S_pos")

    def objective(self, table: Any, group: Any, seed: int, arm: str, reference=None):
        import torch
        self.check_arm(arm)
        if group.labels is None:
            raise ValueError("Authorized training supervision required")
        slots = (record.assignment(group), record.assignment(group, seed))
        predictions = [self.account_logits(table, a, arm) for a in slots]
        truth = torch.as_tensor(group.labels, dtype=table.dtype, device=table.device)
        pieces = []
        for predicted in predictions:
            original = (parent.ranking.objectives(predicted, truth, .5) if arm == "S"
                        else positive.objectives(predicted, truth))
            rank = original["rank"] if arm == "S" else original["rank_pos"]
            pieces.append(dict(bce=original["bce"], rank_used=rank, hard=original["hard"],
                               total=original["bce"]+self.spec.lambda_rank*rank+.5*original["hard"]))
        supervised = (pieces[0]["total"]+pieces[1]["total"])/2
        mse = table.new_zeros(())
        if reference is not None:
            target = torch.as_tensor(reference, dtype=table.dtype, device=table.device)
            mse = (predictions[0]-self.account_logits(target, slots[0], arm)).square().mean()
        weight = 1. if reference is None else .1
        result = weight*supervised+.5*mse
        if not torch.isfinite(result):
            raise ValueError("Nonfinite trial objective")
        return result, dict(supervised=float(supervised.detach()), mse0=float(mse.detach()), mse1=None,
            distillation=float(mse.detach()), supervised_contribution=float((weight*supervised).detach()),
            teacher_contribution=float((.5*mse).detach()), weighted=float(result.detach()),
            views=[{k: float(v.detach()) for k, v in p.items()} for p in pieces])

    def update(self, model, optimizer, current, history, reference, c, arm, stage, step,
               check=lambda: None, observe=None):
        import torch
        self.check_arm(arm)
        self.check_model(model)
        if c.get("optuna_trial") != self.spec.identity() or stage not in (1, 2):
            raise ValueError("Wrong actual update specification")
        if (history is None) != (reference is None) or (stage == 1 and history is not None) or (stage == 2 and history is None):
            raise ValueError("Current/history stage differs")
        model.train()
        optimizer.zero_grad(set_to_none=True)
        encoder_lr, head_lr = self.spec.rates(step)
        optimizer.param_groups[0]["lr"], optimizer.param_groups[1]["lr"] = encoder_lr, head_lr
        logs = {}
        for role, group, target in (("current", current, None), ("history", history, reference)):
            if group is None:
                continue
            root = self.spec.roots["schedule"]
            seed = data.seed_for(root, RNG_ORDER, stage, step, role, group.uid, "regroup")
            logs[role] = record.backward_group(model, group, c, arm,
                data.seed_for(root, RNG_ORDER, stage, step, role, "dropout"), seed, target, check,
                objective_fn=self.objective)
            logs[role]["assignment_sha256"] = hashlib.sha256(record.assignment(group, seed).tobytes()).hexdigest()
            if observe is not None:
                observe(role, logs[role])
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
        check()
        optimizer.step()
        return {**logs, "gradient_norm": float(norm), "encoder_lr": encoder_lr, "head_lr": head_lr,
                "total": sum(v["weighted"] for v in logs.values()), **self.metadata(arm)}

    def score(self, model, groups, c, arm, check=lambda: None):
        self.check_arm(arm)
        self.check_model(model)
        if not groups:
            return np.empty((0, 378), np.float32)
        if arm != "H_pos":
            return record.score(model, groups, c, check)
        return positive.h.score(model, groups, c, "G_half", check)

    def restored_score(self, model, groups, c, check=lambda: None):
        return self.score(model, groups, c, self.spec.arm, check)

    def capture(self, model, optimizer, current, memory, valid, c, arm, stage, epoch,
                root, point, check=lambda: None):
        self.check_model(model)
        result = diag.capture(model, optimizer, current, memory, valid, c, arm, RNG_ORDER, stage,
            epoch, root, point, check, scope="AB", score_fn=self.score,
            pooling_fn=lambda table, slots, group, view: self.account_logits(table, slots, arm))
        path = root/result["path"]
        row = data.read_json(path)
        row.update(seed=f"s{self.spec.seed}", **self.metadata(arm))
        data.write_json(path, row)
        return data.record(path, root)
