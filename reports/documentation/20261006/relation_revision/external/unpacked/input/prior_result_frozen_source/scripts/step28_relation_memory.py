"""Supervised relation-objective memory; no archive or formal-data entry point."""
from __future__ import annotations

import hashlib
import json
import random
import struct
from typing import Any, Callable

import numpy as np

import step28_bge_continual as parent

base, data, core = parent.base, parent.data, parent.core
DIMENSION = 32
EPSILON = .001
MAXIMUM_BYTES = 1048576
MAGIC = b"RELMEM01"


def config() -> dict:
    return parent.config(parent.contract())


def build_model(encoder: Any, account_dimension: int = 4096) -> Any:
    import torch

    class RelationModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.encoder = encoder
            self.head = torch.nn.Sequential(
                torch.nn.Linear(account_dimension * 2, 128), torch.nn.ReLU(),
                torch.nn.Linear(128, 31), torch.nn.Tanh())
            self.weight = torch.nn.Parameter(torch.empty(32))
            torch.nn.init.uniform_(self.weight, -1 / 32**.5, 1 / 32**.5)

        def relations(self, accounts: Any) -> Any:
            left, right = torch.triu_indices(28, 28, 1, device=accounts.device)
            u, v = accounts[left], accounts[right]
            with torch.autocast(accounts.device.type, enabled=False):
                learned = self.head(torch.cat(((u-v).abs(), u*v), 1).float())
                return torch.cat((learned, learned.new_ones((378, 1))), 1).T

        def pair_logits(self, accounts: Any) -> Any:
            return self.relations(accounts).T @ self.weight

    model = RelationModel()
    disable_dropout(model)
    return model


def disable_dropout(model: Any) -> None:
    import torch
    for module in model.modules():
        if isinstance(module, torch.nn.modules.dropout._DropoutNd):
            module.p = 0.
        # Transformers attention implementations may use a numeric probability.
        for key in ("attention_dropout", "hidden_dropout_prob", "attention_probs_dropout_prob", "dropout_prob"):
            if hasattr(module, key) and isinstance(getattr(module, key), (int, float)):
                setattr(module, key, 0.)
    for module in model.modules():
        if hasattr(module, "config"):
            for key in ("hidden_dropout_prob", "attention_probs_dropout_prob", "attention_dropout"):
                if hasattr(module.config, key):
                    setattr(module.config, key, 0.)


def load_model(c: dict, device: str = "cuda:0") -> Any:
    import torch
    # Reuse the checked native BGE loader, not an experimental checkpoint.
    previous = base.load_model(c, "split_rank", device)
    encoder = previous.encoder
    del previous
    torch.manual_seed(c["initialization_seed"])
    return build_model(encoder).to(device)


def make_optimizer(model: Any, c: dict) -> Any:
    import torch
    p = c["optimizer"]
    return torch.optim.AdamW([
        {"params": model.encoder.parameters(), "lr": 1e-5, "weight_decay": .01},
        {"params": [*model.head.parameters(), model.weight], "lr": .001, "weight_decay": 0.}
    ], betas=tuple(p["betas"]), eps=p["eps"], foreach=False)


def relation_features(model: Any, group: Any, c: dict,
                      check: Callable[[], None] = lambda: None) -> Any:
    import torch
    group.validate()
    # Inherited forward casts only the encoder to BF16; pooling/head remain FP32.
    with torch.autocast(next(model.parameters()).device.type, enabled=False):
        accounts = base.account_vectors(model, group, c, "split_rank", check)
        z = model.relations(accounts)
    if z.shape != (32, 378) or z.dtype != torch.float32 or not torch.isfinite(z).all():
        raise ValueError("Noncanonical relation features")
    return z


def query_indices(labels: Any, device: Any) -> list[tuple[Any, Any]]:
    import torch
    truth = torch.as_tensor(labels, device=device)
    if truth.shape != (378,) or not torch.isin(truth, truth.new_tensor([0, 1])).all():
        raise ValueError("Complete binary relation labels required")
    left, right = torch.triu_indices(28, 28, 1, device=device)
    positive = torch.zeros((28, 28), dtype=torch.bool, device=device)
    positive[left, right] = truth.bool()
    positive[right, left] = truth.bool()
    degrees = positive.sum(1)
    diagonal = torch.eye(28, dtype=torch.bool, device=device)
    if (int(truth.sum()) != 20 or int((degrees == 1).sum()) != 16
            or int((degrees == 2).sum()) != 12
            or (((positive.float() @ positive.float()) > 0) & ~positive & ~diagonal).any()):
        raise ValueError("Expected four triples and eight pairs")
    result = []
    for q in range(28):
        incident = (left == q) | (right == q)
        result.append((torch.where(incident & truth.bool())[0],
                       torch.where(incident & ~truth.bool())[0]))
    return result


def group_loss(scores: Any, labels: Any) -> Any:
    import torch
    if scores.shape != (378,) or not torch.isfinite(scores).all():
        raise ValueError("Invalid full-group scores")
    pairs = query_indices(labels, scores.device)
    target = 2 * torch.as_tensor(labels, dtype=scores.dtype, device=scores.device) - 1
    rank = torch.stack([(scores[p, None] - scores[n][None, :] - 2).square().mean()
                        for p, n in pairs]).mean()
    return (scores - target).square().mean() + rank


def statistics(z: Any, labels: Any) -> tuple[Any, Any, float]:
    import torch
    z = z.detach().double()
    pairs = query_indices(labels, z.device)
    h = z @ z.T / 378
    target = 2 * torch.as_tensor(labels, dtype=z.dtype, device=z.device) - 1
    b = z @ target / 378
    for p, n in pairs:
        diff = (z[:, p, None] - z[:, n][:, None, :]).flatten(1)
        h = h + diff @ diff.T / (28 * diff.shape[1])
        b = b + 2 * diff.mean(1) / 28
    return h, b, 5.


def transport(x: Any, y: Any) -> Any:
    import torch
    x, y = x.detach().double(), y.double()
    eye = torch.eye(x.shape[0], dtype=x.dtype, device=x.device)
    k = x @ x.T / x.shape[1] + EPSILON * eye
    cross = y @ x.T / x.shape[1] + EPSILON * eye
    return torch.linalg.solve(k, cross.T).T


def historical_loss(y: Any, weight: Any, x: Any, h: Any, b: Any,
                    constant: float, count: int) -> Any:
    import torch
    if count <= 0 or y.shape != x.shape:
        raise ValueError("Missing or misaligned history")
    y, weight = y.double(), weight.double()
    x = torch.as_tensor(x, dtype=y.dtype, device=y.device).detach()
    h = torch.as_tensor(h, dtype=y.dtype, device=y.device).detach()
    b = torch.as_tensor(b, dtype=y.dtype, device=y.device).detach()
    k = x @ x.T / x.shape[1] + EPSILON * torch.eye(x.shape[0], dtype=y.dtype, device=y.device)
    p = y.T @ weight
    v = torch.linalg.solve(k, x @ p / x.shape[1] + EPSILON * weight)
    loss = (v @ h @ v - 2 * b @ v + constant) / count
    if not torch.isfinite(loss):
        raise ValueError("Nonfinite historical objective")
    return loss


class Memory:
    """One joint statistic and six raw groups; all persistent fields are charged."""

    def __init__(self, order: str, seed: int = 20260930):
        if order not in parent.ORDERS:
            raise ValueError("Unknown order")
        self.order, self.seed = order, seed
        self.reservoir = data.Memory(data.seed_for(seed, order, "retention"))
        self.h = np.zeros((32, 32), dtype="<f8")
        self.b = np.zeros(32, dtype="<f8")
        self.constant, self.count = 0., 0
        self.references: dict[str, np.ndarray] = {}
        self.reference_keys: dict[str, str] = {}
        self.stage, self.draw_stage, self.draw_count = 0, 0, 0
        self.draw_rng: random.Random | None = None
        self.maps: dict = {}

    @staticmethod
    def group_key(group: Any) -> str:
        return hashlib.sha256(data.json_bytes(group.payload())).hexdigest()

    def begin_stage(self, stage: int) -> None:
        if (stage not in (2, 3) or self.stage != stage - 1 or self.draw_stage
                or self.count != (stage - 1) * 48 or self.reservoir.seen != self.count):
            raise ValueError("History phase does not match stage")
        self.draw_stage, self.draw_count = stage, 0
        self.draw_rng = random.Random(data.seed_for(self.seed, self.order, stage, "history_draws"))
        try:
            self.to_bytes()
        except BaseException:
            self.draw_stage, self.draw_count, self.draw_rng = 0, 0, None
            raise

    def draw(self) -> tuple[Any, np.ndarray]:
        if self.draw_rng is None or self.draw_count >= 288:
            raise ValueError("Missing or exhausted history schedule")
        previous = self.draw_rng.getstate(), self.draw_count
        try:
            group = self.reservoir.groups[self.draw_rng.randrange(6)]
            if self.reference_keys[group.uid] != self.group_key(group):
                raise ValueError("Reference group/edge correspondence differs")
            self.draw_count += 1
            self.to_bytes()
        except BaseException:
            self.draw_rng.setstate(previous[0])
            self.draw_count = previous[1]
            raise
        return group, self.references[group.uid]

    def consolidate(self, current: list, stage: int,
                    features: Callable[[Any], np.ndarray]) -> dict:
        import torch
        if (stage not in (1, 2) or stage != self.stage + 1
                or len(current) != 48 or len({g.uid for g in current}) != 48
                or self.count != (stage-1)*48 or self.reservoir.seen != self.count
                or (stage == 2 and (self.draw_stage != 2 or self.draw_count != 288))
                or {g.uid for g in current} & {g.uid for g in self.reservoir.groups}):
            raise ValueError("Stage consolidation/insertion must occur exactly once")
        for group in current:
            group.validate()
            query_indices(group.labels, "cpu")
        # Mutate only a small bounded copy; failures cannot leave half a new memory.
        candidate = Memory.from_bytes(self.to_bytes())
        evidence = {"old_count": self.count, "old_members": [g.uid for g in self.reservoir.groups]}
        if self.count:
            xs, ys = [], []
            for group in self.reservoir.groups:
                if self.group_key(group) != self.reference_keys[group.uid]:
                    raise ValueError("Old reference correspondence differs")
                xs.append(torch.from_numpy(self.references[group.uid]).double())
                ys.append(torch.from_numpy(canonical_reference(features(group))).double())
            t = transport(torch.cat(xs, 1), torch.cat(ys, 1)).numpy()
            candidate.h = t @ self.h @ t.T
            candidate.h = (candidate.h + candidate.h.T) / 2
            candidate.b = t @ self.b
            evidence["transport_singular_values"] = np.linalg.svd(t, compute_uv=False).tolist()
        for group in sorted(current, key=lambda g: g.uid):
            z = canonical_reference(features(group))
            h, b, constant = statistics(torch.from_numpy(z), group.labels)
            candidate.h += h.numpy()
            candidate.b += b.numpy()
            candidate.constant += constant
            candidate.count += 1
        # Reservoir has its own old seen count; never initialize it from new N.
        candidate.reservoir.add_stage(sorted(current, key=lambda g: g.uid))
        candidate.references = {g.uid: canonical_reference(features(g))
                                for g in candidate.reservoir.groups}
        candidate.reference_keys = {g.uid: self.group_key(g) for g in candidate.reservoir.groups}
        candidate.stage = stage
        candidate.draw_stage, candidate.draw_count, candidate.draw_rng = 0, 0, None
        spectrum = np.linalg.eigvalsh((candidate.h + candidate.h.T) / 2)
        tolerance = 1e-10 + 1e-10 * float(np.max(np.abs(spectrum)))
        if not np.isfinite(spectrum).all() or spectrum[0] < -tolerance:
            raise ValueError(f"Non-PSD accumulated H: minimum={spectrum[0]}, tolerance={tolerance}")
        evidence.update(h_min_eigenvalue=float(spectrum[0]), h_psd_tolerance=tolerance)
        payload = candidate.to_bytes()
        self.__dict__.update(candidate.__dict__)
        evidence.update(new_count=self.count, reservoir_seen=self.reservoir.seen,
                        serialized_bytes=len(payload), new_members=[g.uid for g in self.reservoir.groups])
        return evidence

    def to_bytes(self) -> bytes:
        if self.reservoir.seen != self.count or self.count != self.stage * 48:
            raise ValueError("Statistic and reservoir counts diverge")
        uids = sorted(g.uid for g in self.reservoir.groups)
        if set(uids) != set(self.references) or set(uids) != set(self.reference_keys):
            raise ValueError("Reference coverage differs")
        if (not np.isfinite(self.h).all() or not np.isfinite(self.b).all()
                or self.constant != 5*self.count):
            raise ValueError("Invalid accumulated statistic")
        header = data.json_bytes({"order": self.order, "seed": self.seed, "stage": self.stage,
                                  "reservoir": json.loads(self.reservoir.to_bytes()),
                                  "uids": uids, "reference_keys": self.reference_keys,
                                  "maps": self.maps, "draw_stage": self.draw_stage,
                                  "draw_count": self.draw_count,
                                  "draw_rng": self.draw_rng.getstate() if self.draw_rng else None})
        numeric = (self.h.astype("<f8").tobytes() + self.b.astype("<f8").tobytes()
                   + struct.pack("<dq", self.constant, self.count)
                   + b"".join(canonical_reference(self.references[uid]).tobytes() for uid in uids))
        payload = MAGIC + struct.pack("<I", len(header)) + header + numeric
        if len(payload) > MAXIMUM_BYTES:
            raise ValueError("Complete effective history exceeds 1MiB")
        return payload

    @classmethod
    def from_bytes(cls, payload: bytes) -> Memory:
        if len(payload) > MAXIMUM_BYTES or payload[:8] != MAGIC:
            raise ValueError("Invalid memory container")
        size = struct.unpack("<I", payload[8:12])[0]
        obj = json.loads(payload[12:12+size])
        result = cls(obj["order"], obj["seed"])
        result.reservoir = data.Memory.from_bytes(data.json_bytes(obj["reservoir"]))
        offset = 12+size
        result.h = np.frombuffer(payload, "<f8", 1024, offset).reshape(32, 32).copy()
        offset += 8192
        result.b = np.frombuffer(payload, "<f8", 32, offset).copy()
        offset += 256
        result.constant, result.count = struct.unpack("<dq", payload[offset:offset+16])
        offset += 16
        for uid in obj["uids"]:
            result.references[uid] = np.frombuffer(payload, "<f4", 32*378, offset).reshape(32, 378).copy()
            offset += 32*378*4
        result.stage, result.draw_stage, result.draw_count = obj["stage"], obj["draw_stage"], obj["draw_count"]
        result.maps, result.reference_keys = obj["maps"], obj["reference_keys"]
        if obj["draw_rng"] is not None:
            result.draw_rng = random.Random()
            result.draw_rng.setstate(data.tuple_state(obj["draw_rng"]))
        if result.to_bytes() != payload:
            raise ValueError("Nonexact memory restore")
        return result


def canonical_reference(z: np.ndarray) -> np.ndarray:
    z = np.asarray(z)
    if (z.shape != (32, 378) or z.dtype != np.float32 or not np.isfinite(z).all()
            or not np.array_equal(z[-1], np.ones(378, np.float32))
            or np.max(np.abs(z[:-1])) > 1):
        raise ValueError("Reference must be the canonical FP32 relation matrix")
    return np.ascontiguousarray(z, dtype="<f4")


def reference(model: Any, group: Any, c: dict,
              check: Callable[[], None] = lambda: None) -> np.ndarray:
    import torch
    # Keep identical train-mode branches with p=0; only the graph is disabled.
    model.train()
    with torch.no_grad():
        return canonical_reference(relation_features(model, group, c, check).cpu().numpy())


def optimization_step(model: Any, optimizer: Any, current: Any, history: Any,
                      x: Any, memory: Memory, c: dict, encoder_lr: float,
                      *, check: Callable[[], None] = lambda: None,
                      observe_history: Callable[[Any], None] | None = None,
                      observe_stage: Callable[[str], None] = lambda stage: None) -> dict:
    """The actual optimizer kernel, also used for isolated native admission."""
    import torch
    previous = parent.adam_step(optimizer)
    if (len(optimizer.param_groups) != 2 or (history is None) != (x is None)
            or (history is not None and (current.uid == history.uid or memory.count <= 0))):
        raise ValueError("Invalid optimizer/history kernel inputs")
    rates = (encoder_lr, .001)
    for group, rate in zip(optimizer.param_groups, rates, strict=True):
        group["lr"] = rate
    model.train()
    optimizer.zero_grad(set_to_none=True)
    observe_stage("current_forward_backward")
    z = relation_features(model, current, c, check)
    loss = group_loss(z.T @ model.weight, current.labels)
    loss.backward()
    result = {"current": float(loss.detach()), "history": 0., "history_uid": None}
    del z, loss
    if history is not None:
        observe_stage("history_forward_backward")
        z = relation_features(model, history, c, check)
        loss = historical_loss(z, model.weight, x, memory.h, memory.b, memory.constant, memory.count)
        if observe_history is not None:
            observe_history(loss)
        loss.backward()
        result.update(history=float(loss.detach()), history_uid=history.uid)
        del z, loss
    observe_stage("clip_and_adam")
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
    check()
    optimizer.step()
    check()
    if parent.adam_step(optimizer) != previous+1:
        raise ValueError("Expected exactly one AdamW update")
    return {**result, "total": result["current"]+result["history"], "gradient_norm": float(norm),
            "adam_step": previous+1, "encoder_lr": rates[0], "head_lr": rates[1]}


def update(model: Any, optimizer: Any, current: Any, memory: Memory, c: dict,
           stage: int, step: int, *, check: Callable[[], None] = lambda: None) -> dict:
    if (stage not in (1, 2, 3) or parent.adam_step(optimizer) != (stage-1)*288+step-1
            or memory.stage != stage-1
            or (stage > 1 and (memory.draw_stage != stage or memory.draw_count != step-1))
            or any(current.uid == g.uid for g in memory.reservoir.groups)):
        raise ValueError("Update phase or current/history separation differs")
    encoder_lr = parent.stage_lr(step)
    history, x = memory.draw() if stage > 1 else (None, None)
    result = optimization_step(model, optimizer, current, history, x, memory, c,
                               encoder_lr, check=check)
    return {**result, "stage": stage, "step": step}


def train_stage(model: Any, optimizer: Any, current: list, memory: Memory, c: dict,
                stage: int, check: Callable[[], None] = lambda: None) -> list[dict]:
    sequence, _ = parent.schedule(current, parent.contract(), memory.order, stage)
    if stage > 1:
        memory.begin_stage(stage)
    records = [update(model, optimizer, g, memory, c, stage, i+1, check=check)
               for i, g in enumerate(sequence)]
    if stage < 3:
        memory.consolidate(current, stage, lambda g: reference(model, g, c, check))
    return records
