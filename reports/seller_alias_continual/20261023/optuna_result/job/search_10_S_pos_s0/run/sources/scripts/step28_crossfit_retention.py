"""Group-excluded sources and bounded, one-sided historical ranking floors."""
from __future__ import annotations

import copy
import hashlib
import json
import random
import struct
from typing import Any, Callable

import numpy as np

import step28_record_replay as record

data, parent, core = record.data, record.parent, record.core
ARMS = ("S", "C_cf", "C_perm")
RNG_ORDER = "ABC"  # Only stages 1/2: preserve the established S random streams.
COMPARISONS = 1016


def indices(group: data.Group) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[slice]]:
    if group.labels is None:
        raise ValueError("Authorized training labels required for historical comparisons")
    record.assignment(group, 0)  # Check the actual disjoint K2/K3 components.
    pair = np.full((28, 28), -1, np.int64)
    i, j = np.triu_indices(28, 1)
    pair[i, j] = pair[j, i] = np.arange(378)
    positive = np.zeros((28, 28), bool)
    positive[i, j] = positive[j, i] = np.asarray(group.labels, bool)
    left, right, weights, blocks = [], [], [], []
    for q in range(28):
        ps = np.flatnonzero(positive[q])
        ns = np.flatnonzero(~positive[q] & (np.arange(28) != q))
        for p in ps:
            start = len(left)
            for n in ns:
                left.append(pair[q, p])
                right.append(pair[q, n])
                weights.append(1 / (28 * len(ps) * len(ns)))
            blocks.append(slice(start, len(left)))
    if len(left) != COMPARISONS:
        raise ValueError("Comparison coverage differs")
    return np.asarray(left), np.asarray(right), np.asarray(weights, np.float32), blocks


def margins(scores: Any, group: data.Group) -> tuple[Any, Any]:
    import torch
    if scores.shape != (378,) or scores.dtype != torch.float32 or not torch.isfinite(scores).all():
        raise ValueError("Finite FP32 account scores required")
    left, right, weights, _ = indices(group)
    delta = scores[torch.as_tensor(left, device=scores.device)] - scores[torch.as_tensor(right, device=scores.device)]
    if not torch.isfinite(delta).all():
        raise ValueError("Nonfinite FP32 difference")
    return delta, torch.as_tensor(weights, device=scores.device)


def validate_floor(values: np.ndarray) -> None:
    if (values.shape != (COMPARISONS,) or values.dtype != np.float32
            or not np.isfinite(values).all() or np.any(values < 0)):
        raise ValueError("Invalid historical floor")


def birth(group: data.Group, full: Any, outside: Any) -> tuple[np.ndarray, np.ndarray, dict]:
    """Inputs are account tensors pooled on the same device as the live objective."""
    import torch
    if full.device != outside.device:
        raise ValueError("Birth sources must use the same numeric device")
    mf, _ = margins(full, group)
    mo, _ = margins(outside, group)
    values = torch.where((mf > 0) & (mo > 0), torch.minimum(mf, mo), torch.zeros_like(mf))
    b, capacity = values.detach().cpu().numpy().copy(), mf.detach().cpu().numpy().copy()
    validate_floor(b)
    loss, summary = floor_loss(full, group, b)
    if summary["violations"] != 0 or float(loss) != 0:
        raise ValueError("Birth source is not feasible under the live numeric semantics")
    return b, capacity, summary


def permute(group: data.Group, values: np.ndarray, capacity: np.ndarray,
            rng: np.random.Generator) -> np.ndarray:
    validate_floor(values)
    if capacity.shape != values.shape or capacity.dtype != np.float32 or not np.isfinite(capacity).all():
        raise ValueError("Finite precomputed FP32 capacities required")
    if np.any(values[values > 0] > capacity[values > 0]):
        raise ValueError("Original assignment is infeasible")
    result = np.zeros_like(values)
    for block in indices(group)[3]:
        b, cap = values[block], capacity[block]
        assigned = np.zeros(len(b), bool)
        for k in sorted(np.flatnonzero(b > 0), key=lambda k: (-float(b[k]), int(k))):
            allowed = np.flatnonzero(~assigned & (cap >= b[k]))
            if not len(allowed):
                raise ValueError("Descending feasible assignment failed")
            slot = int(allowed[int(rng.integers(len(allowed)))])
            result[block.start + slot] = b[k]
            assigned[slot] = True
        if not np.array_equal(np.sort(result[block]), np.sort(b)):
            raise ValueError("Permutation changed the floor multiset")
    if np.any(result[result > 0] > capacity[result > 0]):
        raise ValueError("Permutation is not birth-feasible")
    return result


def floor_loss(scores: Any, group: data.Group, values: np.ndarray) -> tuple[Any, dict]:
    import torch
    validate_floor(values)
    delta, weights = margins(scores, group)
    b = torch.as_tensor(values, device=scores.device)
    residual = torch.where(b > 0, torch.relu(b - delta), torch.zeros_like(delta))
    loss = (weights * residual.square()).sum()
    if not torch.isfinite(loss):
        raise ValueError("Nonfinite floor loss")
    positive = values[values > 0]
    return loss, {"active": int(len(positive)), "violations": int(((b > 0) & (delta < b)).sum()),
                  "minimum_positive": float(positive.min()) if len(positive) else None,
                  "maximum": float(values.max()), "distillation": float(loss.detach())}


def objective(table: Any, group: data.Group, regroup_seed: int, arm: str,
              reference: np.ndarray | None = None) -> tuple[Any, dict]:
    if arm not in ("C_cf", "C_perm"):
        raise ValueError("Unknown floor arm")
    supervised, _ = record.objective(table, group, regroup_seed, "S")
    if reference is None:
        return supervised, {"supervised": float(supervised.detach()), "weighted": float(supervised.detach()),
                            "distillation": 0., "active": 0, "violations": 0}
    loss, details = floor_loss(record.account_logits(table, record.assignment(group)), group, reference)
    total = .1 * supervised + .5 * loss
    return total, {**details, "supervised": float(supervised.detach()), "weighted": float(total.detach())}


def update(model: Any, optimizer: Any, current: data.Group, history: data.Group | None,
           target: np.ndarray | None, c: dict, arm: str, stage: int, step: int,
           check: Callable = lambda: None) -> dict:
    if arm not in ARMS or stage not in (1, 2) or not 1 <= step <= 288:
        raise ValueError("Unapproved update scope")
    if (history is None) != (stage == 1) or (target is None) != (history is None):
        raise ValueError("History supply differs")
    before = (stage - 1) * 288 + step - 1
    if parent.adam_step(optimizer) != before:
        raise ValueError("Actual main Adam counter differs")
    row = record.update(model, optimizer, current, history, target, c, arm, RNG_ORDER, stage, step,
                        check, objective_fn=None if arm == "S" else objective)
    if parent.adam_step(optimizer) != before + 1:
        raise ValueError("Main update missing")
    return row


def folds(groups: list[data.Group]) -> tuple[list[data.Group], list[data.Group]]:
    if len(groups) != 48 or len({g.uid for g in groups}) != 48:
        raise ValueError("Exactly 48 complete fitting groups required")
    ordered = sorted(groups, key=lambda g: (hashlib.sha256(("C_cf/fold/s0/" + g.uid).encode()).digest(), g.uid.encode()))
    return ordered[:24], ordered[24:]


def source_schedule(groups: list[data.Group], role: str) -> list[data.Group]:
    if role not in ("U0", "U1") or len(groups) != 24 or len({g.uid for g in groups}) != 24:
        raise ValueError("Auxiliary scope differs")
    return data.schedule(groups, 6, data.seed_for(20260918, "C_cf", "s0", "A", role, "current"))


def source_update(model: Any, optimizer: Any, group: data.Group, c: dict, role: str, step: int,
                  check: Callable = lambda: None) -> dict:
    import torch
    if role not in ("U0", "U1") or not 1 <= step <= 144 or parent.adam_step(optimizer) != step - 1:
        raise ValueError("Actual auxiliary Adam counter/scope differs")
    model.train()
    optimizer.zero_grad(set_to_none=True)
    lr = parent.stage_lr(2 * step)
    optimizer.param_groups[0]["lr"], optimizer.param_groups[1]["lr"] = lr, .001
    row = record.backward_group(model, group, c, "S",
        data.seed_for(20260918, "C_cf", "s0", "A", role, step, "dropout"),
        data.seed_for(20260918, "C_cf", "s0", "A", role, step, group.uid, "regroup"), None, check)
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
    check()
    optimizer.step()
    if parent.adam_step(optimizer) != step:
        raise ValueError("Auxiliary update missing")
    return {"step": step, "lr_step": 2 * step, "encoder_lr": lr, "head_lr": .001,
            "gradient_norm": float(norm), "group_uid": group.uid, **row}


class Memory(record.Memory):
    """Self-sufficient six-group target replacement; no F/O or source model."""

    def __init__(self, arm: str):
        if arm not in ("C_cf", "C_perm"):
            raise ValueError("Unknown floor memory arm")
        super().__init__(RNG_ORDER)
        self.arm = arm
        self.source_ids: dict = {}

    @classmethod
    def from_shared(cls, shared: record.Memory, arm: str, values: dict, source_ids: dict) -> Memory:
        result = cls(arm)
        result.reservoir = data.Memory.from_bytes(shared.reservoir.to_bytes())
        result.origins, result.maps = copy.deepcopy(shared.origins), copy.deepcopy(shared.maps)
        result.tables = {g.uid: values[g.uid].copy() for g in result.reservoir.groups}
        result.source_ids = copy.deepcopy(source_ids)
        result.to_bytes()
        return result

    def retain(self, *args, **kwargs):
        raise ValueError("AB target birth only; B is terminal, no later retain")

    def to_bytes(self) -> bytes:
        members = [g.uid for g in self.reservoir.groups]
        if set(members) != set(self.tables) or set(members) != set(self.origins):
            raise ValueError("Floor/group alignment differs")
        if len(members) != 6 or self.reservoir.seen != 48 or set(self.origins.values()) != {1}:
            raise ValueError("Only the six A-born groups are legal in this AB study")
        if set(self.source_ids) != set(members):
            raise ValueError("Every target needs its own fold and both source identities")
        for row in self.source_ids.values():
            if (row.get("fold") not in (0, 1) or row.get("role") != "U"+str(row["fold"])
                    or any(len(row.get(k, "")) != 64 for k in ("source_sha256", "full_source_sha256"))):
                raise ValueError("Incomplete target birth source identity")
        blocks = []
        for uid in members:
            validate_floor(self.tables[uid])
            blocks.append(self.tables[uid].astype("<f4", copy=False).tobytes())
        header = data.json_bytes({"format": "crossfit_floor_f32", "arm": self.arm, "order": RNG_ORDER,
            "reservoir": json.loads(self.reservoir.to_bytes()), "origins": self.origins,
            "maps": self.maps, "source_ids": self.source_ids, "stage": self.stage, "count": self.count,
            "draw_rng": self.draw_rng.getstate() if self.draw_rng else None})
        payload = struct.pack("<I", len(header)) + header + b"".join(blocks)
        if len(payload) > 1048576:
            raise ValueError("Entire historical state exceeds 1 MiB")
        return payload

    @classmethod
    def from_bytes(cls, payload: bytes) -> Memory:
        size, = struct.unpack("<I", payload[:4])
        obj = json.loads(payload[4:4 + size])
        if obj["format"] != "crossfit_floor_f32" or obj["order"] != RNG_ORDER:
            raise ValueError("Unknown floor memory format")
        result = cls(obj["arm"])
        result.reservoir = data.Memory.from_bytes(data.json_bytes(obj["reservoir"]))
        for key in ("origins", "maps", "source_ids", "stage", "count"):
            setattr(result, key, obj[key])
        cursor = size + 4
        for group in result.reservoir.groups:
            end = cursor + 4 * COMPARISONS
            result.tables[group.uid] = np.frombuffer(payload[cursor:end], dtype="<f4").copy()
            cursor = end
        if obj["draw_rng"] is not None:
            result.draw_rng = random.Random()
            result.draw_rng.setstate(data.tuple_state(obj["draw_rng"]))
        if cursor != len(payload) or result.to_bytes() != payload:
            raise ValueError("Floor memory did not exactly roundtrip")
        return result
