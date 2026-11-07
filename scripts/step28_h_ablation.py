"""Eight fixed interventions on the selected H; no data or execution entry."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import random
import struct

import numpy as np

import step28_history_optuna as history
import step28_g_half as pooling

five, data, record = history.five, history.data, history.five.record
base, core, parent, diag = history.base, history.core, history.parent, history.diag
NAMES = ("Full", "ER", "KD", "SEQ", "NoRegroup", "MeanPool", "StandardRank", "AccountPool")
ORDERS, SEEDS = five.ORDERS, five.SEEDS
ALPHA, BETA = .07049361507739538, .767438161798796
SELECTED = dict(history_weight=ALPHA, teacher_weight=BETA)


@dataclass(frozen=True)
class Spec:
    name: str
    order: str
    seed: int

    def __post_init__(self):
        if self.name not in NAMES or self.order not in ORDERS or type(self.seed) is not int or self.seed not in SEEDS:
            raise ValueError("Outside the fixed eight-arm ablation")

    @property
    def arm(self):
        return "H_pos"

    @property
    def is_record(self):
        return True

    @property
    def uses_history(self):
        return self.name != "SEQ"

    @property
    def alpha(self):
        return 0. if self.name in ("KD", "SEQ") else ALPHA

    @property
    def beta(self):
        return 0. if self.name in ("ER", "SEQ") else BETA

    @property
    def target_kind(self):
        return "none" if not self.beta else "account378" if self.name == "AccountPool" else "record_upper"

    @property
    def lambda_rank(self):
        return five.PARAMETERS["H_pos_tuned"][0]

    @property
    def lr_multiplier(self):
        return five.PARAMETERS["H_pos_tuned"][1]

    @property
    def roots(self):
        return dict(initialization=2026101900+self.seed, schedule=2026102900+self.seed,
                    memory=2026103900+self.seed, partition=20260918, diagnostic=20261008)

    @property
    def run_id(self):
        return f"{self.order}_s{self.seed}_{self.name}"

    def identity(self):
        return dict(name=self.name, order=self.order, seed=self.seed, roots=self.roots,
                    alpha=self.alpha, beta=self.beta, lambda_rank=self.lambda_rank,
                    lr_multiplier=self.lr_multiplier, target_kind=self.target_kind,
                    configuration="h_history_ablation_v1")

    @classmethod
    def from_dict(cls, row):
        spec = cls(**{k: row[k] for k in ("name", "order", "seed")})
        if row != spec.identity():
            raise ValueError("Foreign ablation parameters or roots")
        return spec

    def rates(self, step):
        return self.lr_multiplier*parent.stage_lr(step), self.lr_multiplier*.001


def specifications(*, include_full=False):
    return [Spec(name, order, seed) for order in ORDERS for seed in SEEDS
            for name in NAMES if include_full or name != "Full"]


config = five.config


def mean_scores(table, slots):
    """Full record blocks, FP64 mean, one cast back to the live table dtype."""
    import torch
    n = len(slots)
    if table.shape != (n*(n-1)//2,) or not np.array_equal(np.unique(slots), np.arange(28)):
        raise ValueError("Complete record table and 28 accounts required")
    members = [np.flatnonzero(slots == i) for i in range(28)]
    if any(not 2 <= len(v) <= 8 for v in members):
        raise ValueError("Complete 2..8 records required")
    i, j = torch.triu_indices(n, n, 1, device=table.device)
    matrix = table.new_zeros((n, n))
    matrix[i, j], matrix[j, i] = table, table
    return torch.stack([matrix[np.ix_(members[a], members[b])].double().mean()
                        for a, b in zip(*np.triu_indices(28, 1))]).to(table.dtype)


def account_scores(model, z, slots):
    """Mean record vectors before matching; no std or second normalization."""
    import torch
    if not np.array_equal(np.unique(slots), np.arange(28)):
        raise ValueError("All 28 accounts required")
    accounts = torch.stack([z[np.flatnonzero(slots == i)].mean(0) for i in range(28)])
    return record.pair_table(model, accounts)


class Memory:
    """One bounded history payload, explicitly typed own-run targets or none."""
    def __init__(self, spec):
        self.spec = spec
        self.reservoir = data.Memory(data.seed_for(spec.roots["memory"], spec.order, "retention"))
        self.tables, self.origins, self.maps = {}, {}, {}
        self.stage = self.count = 0
        self.draw_rng = None

    def retain(self, current, stage, teacher=None):
        if (stage not in (1, 2) or len(current) != 48 or self.reservoir.seen != 48*(stage-1)
                or (stage == 2 and (self.stage != 2 or self.count != 288))
                or (teacher is not None) != bool(self.spec.beta)):
            raise ValueError("History boundary or teacher role differs")
        old = {uid: t.tobytes() for uid, t in self.tables.items()}
        self.reservoir.add_stage(sorted(current, key=lambda g: g.uid))
        kept = {g.uid for g in self.reservoir.groups}
        self.tables = {uid: t for uid, t in self.tables.items() if uid in kept}
        self.origins = {uid: s for uid, s in self.origins.items() if uid in kept}
        for group in self.reservoir.groups:
            if group.uid not in self.origins:
                self.origins[group.uid] = stage
                if teacher is not None:
                    self.tables[group.uid] = np.asarray(teacher(group), np.float32).copy()
        if any(self.tables[uid].tobytes() != old[uid] for uid in kept & old.keys()):
            raise ValueError("Surviving birth target was refreshed")
        self.stage = self.count = 0
        self.draw_rng = None
        self.to_bytes()

    def begin_stage(self, stage):
        if stage not in (2, 3) or self.stage != 0 or self.reservoir.seen != 48*(stage-1):
            raise ValueError("History stage differs")
        self.stage, self.count = stage, 0
        self.draw_rng = random.Random(data.seed_for(self.spec.roots["memory"], self.spec.order, stage, "history_draws"))

    def draw(self):
        if self.draw_rng is None or self.count >= 288 or len(self.reservoir.groups) != 6:
            raise ValueError("History draw boundary differs")
        group = self.draw_rng.choice(self.reservoir.groups)
        self.count += 1
        return group, self.tables.get(group.uid)

    def to_bytes(self):
        kept = {g.uid for g in self.reservoir.groups}
        if set(self.origins) != kept or set(self.tables) != (kept if self.spec.beta else set()):
            raise ValueError("History membership/target mismatch")
        arrays, rows = [], []
        for g in self.reservoir.groups:
            if not self.spec.beta:
                continue
            n = sum(map(len, g.items))
            count = 378 if self.spec.target_kind == "account378" else n*(n-1)//2
            t = self.tables[g.uid]
            if t.shape != (count,) or t.dtype != np.float32 or not np.isfinite(t).all():
                raise ValueError("Wrong typed historical target")
            rows.append(dict(uid=g.uid, count=count))
            arrays.append(t.astype("<f4", copy=False).tobytes())
        header = data.json_bytes(dict(format="h_ablation_memory_v1", spec=self.spec.identity(),
            reservoir=json.loads(self.reservoir.to_bytes()), origins=self.origins, maps=self.maps,
            stage=self.stage, count=self.count, targets=rows,
            draw_rng=self.draw_rng.getstate() if self.draw_rng else None))
        payload = struct.pack("<I", len(header))+header+b"".join(arrays)
        if len(payload) > 1048576:
            raise ValueError("Complete history exceeds 1 MiB")
        return payload

    @classmethod
    def from_bytes(cls, payload, spec):
        size, = struct.unpack("<I", payload[:4])
        row = json.loads(payload[4:4+size])
        if row["format"] != "h_ablation_memory_v1" or row["spec"] != spec.identity():
            raise ValueError("Foreign history configuration")
        result = cls(spec)
        result.reservoir = data.Memory.from_bytes(data.json_bytes(row["reservoir"]))
        result.origins, result.maps = row["origins"], row["maps"]
        result.stage, result.count = row["stage"], row["count"]
        cursor = 4+size
        for item in row["targets"]:
            end = cursor+4*item["count"]
            result.tables[item["uid"]] = np.frombuffer(payload[cursor:end], dtype="<f4").copy()
            cursor = end
        if row["draw_rng"] is not None:
            result.draw_rng = random.Random()
            result.draw_rng.setstate(data.tuple_state(row["draw_rng"]))
        if cursor != len(payload) or result.to_bytes() != payload:
            raise ValueError("Historical state failed exact roundtrip")
        return result

    def summary(self):
        payload = self.to_bytes()
        return dict(members=[g.uid for g in self.reservoir.groups], seen=self.reservoir.seen,
            stage=self.stage, count=self.count, bytes=len(payload), origins=self.origins.copy(),
            target_kind=self.spec.target_kind, sha256=hashlib.sha256(payload).hexdigest(),
            tables={uid: hashlib.sha256(t.tobytes()).hexdigest() for uid, t in self.tables.items()})


class Method(five.Method):
    def __init__(self, spec):
        self.spec = spec

    def metadata(self):
        return dict(continual_spec=self.spec.identity(), method=self.spec.name,
                    outer_encoder_recompute=True, objective="h_ablation_explicit_role_v1")

    def bind(self, model, meta):
        if any(meta.get(k) != v for k, v in self.metadata().items()):
            raise ValueError("Wrong ablation checkpoint identity")
        model.continual_identity = self.metadata()

    def logits(self, table, slots):
        return (mean_scores(table, slots) if self.spec.name == "MeanPool"
                else pooling.account_logits(table, slots, "G_half"))

    def objective(self, values, group, seed, role, reference=None, model=None):
        import torch
        if role not in ("current", "history") or role == "history" and not self.spec.uses_history:
            raise ValueError("Explicit live current/history role required")
        supervised_enabled = role == "current" or self.spec.alpha > 0
        teacher_enabled = role == "history" and self.spec.beta > 0
        if (reference is not None) != teacher_enabled:
            raise ValueError("Teacher presence differs from the ablated branch")
        slots = [record.assignment(group)]
        if supervised_enabled and self.spec.name != "NoRegroup":
            slots.append(record.assignment(group, seed))
        predictions = [(account_scores(model, values, a) if self.spec.name == "AccountPool"
                        else self.logits(values, a)) for a in slots]
        supervised, mse = values.new_zeros(()), values.new_zeros(())
        pieces = []
        if supervised_enabled:
            if group.labels is None:
                raise ValueError("Authorized supervised labels required")
            truth = torch.as_tensor(group.labels, dtype=predictions[0].dtype, device=values.device)
            for predicted in predictions:
                terms = five.tuned.positive.objectives(predicted, truth, old_rank=self.spec.name == "StandardRank")
                rank = terms["rank_used"]
                total = terms["bce"]+self.spec.lambda_rank*rank+.5*terms["hard"]
                pieces.append(dict(bce=terms["bce"], rank_used=rank, hard=terms["hard"], total=total))
            supervised = sum(v["total"] for v in pieces)/len(pieces)
        if teacher_enabled:
            target = torch.as_tensor(reference, dtype=predictions[0].dtype, device=values.device)
            if self.spec.target_kind == "account378":
                if target.shape != (378,):
                    raise ValueError("Account teacher must contain 378 natural scores")
            else:
                target = self.logits(target, slots[0])
            mse = (predictions[0]-target).square().mean()
        weight = 1. if role == "current" else self.spec.alpha
        sup, teacher = weight*supervised, self.spec.beta*mse
        loss = sup+teacher
        if not torch.isfinite(loss):
            raise ValueError("Nonfinite ablation objective")
        return loss, dict(supervised=float(supervised.detach()), distillation=float(mse.detach()),
            supervised_contribution=float(sup.detach()), teacher_contribution=float(teacher.detach()),
            weighted=float(loss.detach()), supervised_enabled=supervised_enabled,
            teacher_enabled=teacher_enabled, views=[{k: float(v.detach()) for k, v in row.items()} for row in pieces],
            assignment_sha256=hashlib.sha256(slots[1].tobytes()).hexdigest() if len(slots) == 2 else None)

    def update(self, model, optimizer, current, history_group, reference, c, stage, step, check=lambda: None, observe=None):
        import torch
        self.check_model(model)
        expects = stage > 1 and self.spec.uses_history
        if (stage not in (1, 2, 3) or c.get("continual_spec") != self.spec.identity()
                or parent.adam_step(optimizer) != (stage-1)*288+step-1
                or (history_group is not None) != expects or (reference is not None) != (expects and self.spec.beta > 0)):
            raise ValueError("Actual history/optimizer/configuration differs")
        model.train()
        optimizer.zero_grad(set_to_none=True)
        rates = self.spec.rates(step)
        for param_group, rate in zip(optimizer.param_groups, rates, strict=True):
            param_group["lr"] = rate
        logs = {}
        for role, group, target in (("current", current, None), ("history", history_group, reference)):
            if group is None:
                continue
            dropout = data.seed_for(self.spec.roots["schedule"], self.spec.order, stage, step, role, "dropout")
            regroup = data.seed_for(self.spec.roots["schedule"], self.spec.order, stage, step, role, group.uid, "regroup")
            if self.spec.name == "AccountPool":
                torch.manual_seed(dropout)
                z = record.record_vectors(model, group, c, check)
                loss, log = self.objective(z, group, regroup, role, target, model)
                loss.backward()
            else:
                log = record.backward_group(model, group, c, self.spec.arm, dropout, regroup, target, check,
                    objective_fn=lambda t, g, seed, arm, ref: self.objective(t, g, seed, role, ref))
            logs[role] = log
            if observe is not None:
                observe(role, log)
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
        check()
        optimizer.step()
        if parent.adam_step(optimizer) != (stage-1)*288+step:
            raise ValueError("Optimizer update missing")
        return dict(**logs, gradient_norm=float(norm), encoder_lr=rates[0], head_lr=rates[1],
                    total=sum(v["weighted"] for v in logs.values()))

    def score(self, model, groups, c, check=lambda: None):
        import torch
        self.check_model(model)
        model.eval()
        rows = []
        with torch.inference_mode():
            for g in groups:
                z = record.record_vectors(model, g, c, check)
                slots = record.assignment(g)
                scores = (account_scores(model, z, slots) if self.spec.name == "AccountPool"
                          else self.logits(record.pair_table(model, z, check), slots))
                rows.append(scores.float().cpu().numpy())
        result = np.asarray(rows, np.float32)
        if result.shape != (len(groups), 378) or not np.isfinite(result).all():
            raise ValueError("Incomplete blind scores")
        return result

    def new_memory(self, first_map):
        return Memory(self.spec) if self.spec.uses_history else None

    def memory_from(self, payload):
        return Memory.from_bytes(payload, self.spec)

    def retain(self, memory, model, current, c, stage, maps, check=lambda: None):
        if memory is None:
            if self.spec.uses_history:
                raise ValueError("Required history is missing")
            return
        memory.maps = {**maps, **self.metadata(), "rng": diag.rng_state(), "completed_stage": stage}
        if stage < 3:
            teacher = None
            if self.spec.beta:
                teacher = (lambda g: self.score(model, [g], c, check)[0]) if self.spec.name == "AccountPool" else (
                    lambda g: record.reference(model, g, c, check))
            memory.retain(current, stage, teacher)
        memory.to_bytes()
