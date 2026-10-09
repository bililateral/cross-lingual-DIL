"""Complete-record regrouping and source-table replay; no formal data entry here."""
from __future__ import annotations

import contextlib
import copy
import hashlib
import json
import random
import struct
from typing import Any, Callable

import numpy as np

import step28_bge_continual as parent

base, data, core = parent.base, parent.data, parent.core
POLICY = data.ROOT / "schema/step28_record_replay_policy.json"


def policy() -> dict:
    p = data.read_json(POLICY)
    if (p["methods"] != ["C", "S"] or p["orders"] != list(parent.ORDERS)
            or p["physical_updates"] != 4320 or p["gradient_group_presentations"] != 7776
            or p["pair_chunk"] != 512 or p["record_dimension"] != 2048
            or p["history_supervision"] != .1 or p["teacher_weight"] != .5
            or p["memory_maximum_bytes"] != 1048576):
        raise ValueError("Record replay contract differs")
    return p


def config() -> dict:
    c = copy.deepcopy(parent.config(parent.contract()))
    c["runtime"] = policy()["runtime"]
    # Reuse the pinned native loader, with this experiment's actual record head.
    c["interventions"]["split_rank"]["account_dim"] = 2048
    return c


def load_model(c: dict, device: str = "cuda:0") -> Any:
    return base.load_model(c, "split_rank", device)


def record_vectors(model: Any, group: data.Group, c: dict,
                   check: Callable = lambda: None) -> Any:
    import torch
    texts = base.record_texts(group, "separate_moments")
    device = next(model.parameters()).device
    parts = []
    for i in range(0, len(texts), c["input"]["microbatch"]):
        check()
        features = model.encoder.tokenizer(texts[i:i+c["input"]["microbatch"]],
            padding=True, truncation=False, return_tensors="pt")
        if features["input_ids"].shape[1] > c["input"]["token_budget"]:
            raise ValueError("Record exceeds frozen token budget; truncation forbidden")
        features = {k: v.to(device) for k, v in features.items()}
        autocast = (torch.autocast("cuda", dtype=torch.bfloat16)
                    if device.type == "cuda" else contextlib.nullcontext())
        with autocast:
            output = model.encoder(features)["sentence_embedding"]
        parts.append(output.float())
    channels = torch.nn.functional.normalize(torch.cat(parts), dim=1)
    title, description = channels.chunk(2)
    return torch.cat((title, description), dim=1) / (2 ** .5)


def assignment(group: data.Group, seed: int | None = None) -> np.ndarray:
    """Record indices in frozen order; labels consulted only for training A1."""
    counts = list(map(len, group.items))
    slots = np.repeat(np.arange(28), counts)
    if seed is None:
        return slots
    if group.labels is None:
        raise ValueError("Regrouping requires authorized binary train labels")
    adjacency = np.eye(28, dtype=bool)
    left, right = np.triu_indices(28, 1)
    adjacency[left, right] = group.labels
    adjacency[right, left] = group.labels
    components, remaining = [], set(range(28))
    while remaining:
        component = np.flatnonzero(adjacency[min(remaining)]).tolist()
        if not set(component) <= remaining or not all(
                np.array_equal(adjacency[j], adjacency[component[0]]) for j in component):
            raise ValueError("Positive labels must be disjoint cliques")
        remaining.difference_update(component)
        components.append(component)
    if sorted(map(len, components)) != [2]*8 + [3]*4:
        raise ValueError("Expected eight K2 and four K3")
    rng = random.Random(seed)
    result = slots.copy()
    for component in components:
        records = np.flatnonzero(np.isin(slots, component)).tolist()
        rng.shuffle(records)
        start = 0
        for account in component:
            result[records[start:start+counts[account]]] = account
            start += counts[account]
    if not np.array_equal(np.bincount(result, minlength=28), counts):
        raise ValueError("Record count conservation failed")
    return result


def pair_table(model: Any, z: Any, check: Callable = lambda: None) -> Any:
    import torch
    left, right = torch.triu_indices(len(z), len(z), 1, device=z.device)
    parts = []
    for start in range(0, len(left), 512):
        check()
        u, v = z[left[start:start+512]], z[right[start:start+512]]
        parts.append(model.head(torch.cat(((u-v).abs(), u*v), dim=1)).flatten())
    return torch.cat(parts)


def account_logits(table: Any, slots: np.ndarray) -> Any:
    import torch
    n = len(slots)
    if table.shape != (n*(n-1)//2,):
        raise ValueError("Complete off-diagonal record table required")
    left, right = torch.triu_indices(n, n, 1, device=table.device)
    matrix = table.new_zeros((n, n))
    matrix[left, right] = table
    matrix[right, left] = table
    assigned = torch.as_tensor(slots, device=table.device)
    pooling = torch.nn.functional.one_hot(assigned, 28).T.to(table.dtype)
    pooling = pooling / pooling.sum(1, keepdim=True)
    scores = pooling @ matrix @ pooling.T
    i, j = torch.triu_indices(28, 28, 1, device=table.device)
    return scores[i, j]


def objective(table: Any, group: data.Group, regroup_seed: int, arm: str,
              reference: np.ndarray | None = None) -> tuple[Any, dict]:
    import torch
    if arm not in ("C", "S", "R0") or group.labels is None:
        raise ValueError("Unknown arm or missing train labels")
    slots = [assignment(group)]
    if arm != "R0":
        slots.append(assignment(group, regroup_seed))
    predicted = [account_logits(table, a) for a in slots]
    truth = torch.tensor(group.labels, dtype=table.dtype, device=table.device)
    losses = [parent.ranking.objectives(v, truth, .5) for v in predicted]
    supervised = (losses[0]["total"] if arm == "R0" else
                  (losses[0]["total"] + losses[1]["total"]) / 2)
    mse = table.new_zeros(())
    mse0 = mse1 = mse
    if reference is not None:
        target = torch.as_tensor(reference, device=table.device, dtype=table.dtype)
        errors = [((p-account_logits(target, a))**2).mean()
                  for p, a in zip(predicted, slots)]
        mse0 = errors[0]
        if arm != "R0":
            mse1 = errors[1]
        mse = (mse0+mse1)/2 if arm == "C" else mse0
    result = supervised if reference is None else .1*supervised + .5*mse
    return result, {"supervised": float(supervised.detach()), "mse0": float(mse0.detach()),
                    "mse1": None if arm == "R0" else float(mse1.detach()), "distillation": float(mse.detach()),
                    "weighted": float(result.detach())}


def backward_group(model: Any, group: data.Group, c: dict, arm: str,
                   dropout_seed: int, regroup_seed: int, reference: np.ndarray | None,
                   check: Callable = lambda: None) -> dict:
    """One live encoder graph; exact chain rule through chunked deterministic head."""
    import torch
    torch.manual_seed(dropout_seed)
    z = record_vectors(model, group, c, check)
    with torch.no_grad():
        values = pair_table(model, z.detach(), check)
    proxy = values.detach().requires_grad_(True)
    loss, log = objective(proxy, group, regroup_seed, arm, reference)
    gradient, = torch.autograd.grad(loss, proxy)
    leaf = z.detach().requires_grad_(True)
    left, right = torch.triu_indices(len(z), len(z), 1, device=z.device)
    for start in range(0, len(left), 512):
        check()
        u, v = leaf[left[start:start+512]], leaf[right[start:start+512]]
        block = model.head(torch.cat(((u-v).abs(), u*v), dim=1)).flatten()
        block.backward(gradient[start:start+512])
    if leaf.grad is None or not torch.isfinite(leaf.grad).all():
        raise ValueError("Missing or nonfinite explicit dJ/dZ")
    z.backward(leaf.grad)
    log["record_gradient_norm"] = float(leaf.grad.norm())
    return log


def update(model: Any, optimizer: Any, current: data.Group, history: data.Group | None,
           reference: np.ndarray | None, c: dict, arm: str, order: str, stage: int,
           step: int, check: Callable = lambda: None, observe: Callable | None = None) -> dict:
    import torch
    model.train()
    optimizer.zero_grad(set_to_none=True)
    optimizer.param_groups[0]["lr"] = parent.stage_lr(step)
    optimizer.param_groups[1]["lr"] = .001
    logs = {}
    for role, group, target in (("current", current, None), ("history", history, reference)):
        if group is None:
            continue
        logs[role] = backward_group(model, group, c, arm,
            data.seed_for(20260918, order, stage, step, role, "dropout"),
            data.seed_for(20260918, order, stage, step, role, group.uid, "regroup"), target, check)
        if observe is not None:
            observe(role, logs[role])
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
    check()
    optimizer.step()
    return {**logs, "gradient_norm": float(norm), "encoder_lr": parent.stage_lr(step),
            "head_lr": .001, "total": sum(v["weighted"] for v in logs.values())}


def reference(model: Any, group: data.Group, c: dict, check: Callable = lambda: None) -> np.ndarray:
    import torch
    model.eval()
    with torch.inference_mode():
        table = pair_table(model, record_vectors(model, group, c, check), check)
    result = table.float().cpu().numpy().copy()
    if not np.isfinite(result).all():
        raise ValueError("Nonfinite source table")
    return result


def score(model: Any, groups: list[data.Group], c: dict, check: Callable = lambda: None) -> np.ndarray:
    import torch
    model.eval()
    with torch.inference_mode():
        rows = [account_logits(pair_table(model, record_vectors(model, g, c, check), check),
                               assignment(g)).float().cpu().numpy() for g in groups]
    result = np.stack(rows).astype(np.float32)
    if result.shape != (len(groups), 378) or not np.isfinite(result).all():
        raise ValueError("Incomplete blind scores")
    return result


class Memory:
    """Single bounded payload: UTF-8 groups/metadata followed by binary FP32 tables."""

    def __init__(self, order: str):
        if order not in parent.ORDERS:
            raise ValueError("Unknown order")
        self.order = order
        self.reservoir = data.Memory(data.seed_for(20260930, order, "retention"))
        self.tables: dict[str, np.ndarray] = {}
        self.origins: dict[str, int] = {}
        self.maps: dict = {}
        self.stage = self.count = 0
        self.draw_rng: random.Random | None = None

    def retain(self, current: list[data.Group], stage: int, teacher: Callable) -> dict:
        if (stage not in (1, 2) or len(current) != 48 or self.reservoir.seen != 48*(stage-1)
                or (stage == 2 and (self.stage != 2 or self.count != 288))):
            raise ValueError("Retention boundary differs")
        old = {uid: hashlib.sha256(t.tobytes()).hexdigest() for uid, t in self.tables.items()}
        self.reservoir.add_stage(sorted(current, key=lambda g: g.uid))
        kept = {g.uid for g in self.reservoir.groups}
        self.tables = {uid: table for uid, table in self.tables.items() if uid in kept}
        self.origins = {uid: birth for uid, birth in self.origins.items() if uid in kept}
        new = []
        for group in self.reservoir.groups:
            if group.uid not in self.tables:
                self.tables[group.uid] = np.asarray(teacher(group), dtype=np.float32).copy()
                self.origins[group.uid] = stage
                new.append(group.uid)
        for uid in kept & old.keys():
            if hashlib.sha256(self.tables[uid].tobytes()).hexdigest() != old[uid]:
                raise ValueError("Survivor source was refreshed")
        self.stage = self.count = 0
        self.draw_rng = None
        self.to_bytes()
        return {"created": new, "evicted": sorted(old.keys()-kept), "survivors_unchanged": True}

    def begin_stage(self, stage: int) -> None:
        if stage not in (2, 3) or self.stage != 0 or self.reservoir.seen != 48*(stage-1):
            raise ValueError("History stage boundary differs")
        self.stage, self.count = stage, 0
        self.draw_rng = random.Random(data.seed_for(20260930, self.order, stage, "history_draws"))

    def draw(self) -> tuple[data.Group, np.ndarray]:
        if self.draw_rng is None or self.count >= 288 or len(self.reservoir.groups) != 6:
            raise ValueError("History draw boundary differs")
        group = self.draw_rng.choice(self.reservoir.groups)
        self.count += 1
        self.to_bytes()
        return group, self.tables[group.uid]

    def to_bytes(self) -> bytes:
        if set(self.tables) != {g.uid for g in self.reservoir.groups} or set(self.origins) != set(self.tables):
            raise ValueError("Historical groups/source alignment differs")
        blocks, records = [], []
        for group in self.reservoir.groups:
            n = sum(map(len, group.items))
            t = self.tables[group.uid]
            if t.shape != (n*(n-1)//2,) or t.dtype != np.float32 or not np.isfinite(t).all():
                raise ValueError("Historical table shape, type or finiteness differs")
            blocks.append(t.astype("<f4", copy=False).tobytes())
            records.append({"uid": group.uid, "count": len(t), "origin": self.origins[group.uid]})
        header = data.json_bytes({"format": "record_table_f32_v1", "order": self.order,
            "reservoir": json.loads(self.reservoir.to_bytes()), "tables": records, "maps": self.maps,
            "stage": self.stage, "count": self.count,
            "draw_rng": self.draw_rng.getstate() if self.draw_rng else None})
        payload = struct.pack("<I", len(header)) + header + b"".join(blocks)
        if len(payload) > 1048576:
            raise ValueError("Entire historical state exceeds 1 MiB")
        return payload

    @classmethod
    def from_bytes(cls, payload: bytes) -> Memory:
        size, = struct.unpack("<I", payload[:4])
        obj = json.loads(payload[4:4+size])
        if obj["format"] != "record_table_f32_v1":
            raise ValueError("Unknown memory format")
        result = cls(obj["order"])
        result.reservoir = data.Memory.from_bytes(data.json_bytes(obj["reservoir"]))
        result.maps, result.stage, result.count = obj["maps"], obj["stage"], obj["count"]
        cursor = 4+size
        for row in obj["tables"]:
            end = cursor+4*row["count"]
            result.tables[row["uid"]] = np.frombuffer(payload[cursor:end], dtype="<f4").copy()
            result.origins[row["uid"]] = row["origin"]
            cursor = end
        if obj["draw_rng"] is not None:
            result.draw_rng = random.Random()
            result.draw_rng.setstate(data.tuple_state(obj["draw_rng"]))
        if cursor != len(payload) or result.to_bytes() != payload:
            raise ValueError("Full history payload did not roundtrip")
        return result

    def summary(self) -> dict:
        payload = self.to_bytes()
        return {"members": [g.uid for g in self.reservoir.groups], "seen": self.reservoir.seen,
                "stage": self.stage, "count": self.count, "bytes": len(payload),
                "origins": self.origins.copy(), "sha256": hashlib.sha256(payload).hexdigest(),
                "tables": {uid: hashlib.sha256(t.tobytes()).hexdigest() for uid, t in self.tables.items()}}
