"""Cumulative stage-function statistics transported by six canonical reference groups."""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
import random
import struct
from typing import Any, Callable

import numpy as np
import step28_bge_continual as parent
from step28_relation_memory import query_indices

base, data, core = parent.base, parent.data, parent.core
DIMENSION = 129
EPSILON = .001
MAXIMUM_BYTES = 1048576
MAGIC = b"FUNMEM01"


def config() -> dict:
    return parent.config(parent.contract())


def build_model(encoder: Any, account_dimension: int = 4096) -> Any:
    return core.build_model(encoder, account_dimension, 128)


def load_model(c: dict, device: str = "cuda:0") -> Any:
    return base.load_model(c, "split_rank", device)


def make_optimizer(model: Any, c: dict) -> Any:
    return core.make_optimizer(model, c)


def weight(model: Any) -> Any:
    import torch
    return torch.cat((model.head[2].weight.reshape(-1), model.head[2].bias))


@contextmanager
def deterministic_history(model: Any):
    """Keep this context active through backward and checkpoint recomputation."""
    import torch
    saved = {}
    keys = ("dropout_prob", "attention_dropout", "hidden_dropout_prob", "attention_probs_dropout_prob")
    def zero(obj, key):
        value = getattr(obj, key, None)
        if isinstance(value, (float, int)) and (id(obj), key) not in saved:
            saved[id(obj), key] = (obj, key, value)
            setattr(obj, key, 0.)
    try:
        for module in model.modules():
            if isinstance(module, torch.nn.modules.dropout._DropoutNd):
                zero(module, "p")
            for key in keys:
                zero(module, key)
                if hasattr(module, "config"):
                    zero(module.config, key)
        yield
    finally:
        for obj, key, value in saved.values():
            setattr(obj, key, value)


def canonical_features(h: Any) -> Any:
    import torch
    class RoundHalfStraightThrough(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x):
            rounded = x.to(torch.float16).to(torch.float32)
            if not torch.isfinite(rounded).all():
                raise ValueError("FP16 memory coordinate overflow/nonfinite")
            return rounded

        @staticmethod
        def backward(ctx, grad):
            return grad  # Deliberate FP32 straight-through estimator, not round's derivative.

    if h.shape != (378, 128) or h.dtype != torch.float32:
        raise ValueError("Expected full baseline hidden features")
    rounded = RoundHalfStraightThrough.apply(h)
    return torch.cat((rounded, rounded.new_ones((378, 1))), 1).T


def forward_group(model: Any, group: Any, c: dict,
                  check: Callable[[], None] = lambda: None) -> tuple[Any, Any]:
    import torch
    group.validate()
    with torch.autocast(next(model.parameters()).device.type, enabled=False):
        accounts = base.account_vectors(model, group, c, "split_rank", check)
        left, right = torch.triu_indices(28, 28, 1, device=accounts.device)
        a, b = accounts[left], accounts[right]
        h = model.head[1](model.head[0](torch.cat(((a-b).abs(), a*b), 1).float()))
        scores = model.head[2](h).flatten()
        z = canonical_features(h)
    if not torch.isfinite(scores).all():
        raise ValueError("Nonfinite baseline scores")
    return scores, z


def relation_features(model: Any, group: Any, c: dict,
                      check: Callable[[], None] = lambda: None) -> Any:
    return forward_group(model, group, c, check)[1]


def statistics(z: Any, labels: Any, teacher: Any) -> tuple[Any, Any, float]:
    import torch
    z = z.detach().double()
    teacher = torch.as_tensor(teacher, device=z.device).detach().double()
    if z.shape != (DIMENSION, 378) or teacher.shape != (DIMENSION,):
        raise ValueError("Function statistic dimensions differ")
    h = z @ z.T / 378
    for pos, neg in query_indices(labels, z.device):
        delta = (z[:, pos, None] - z[:, neg][:, None, :]).flatten(1)
        h = h + delta @ delta.T / (28 * delta.shape[1])
    b = h @ teacher
    return h, b, float(teacher @ b)


def solve(k: Any, rhs: Any) -> Any:
    import torch
    result = torch.linalg.solve(k, rhs)
    with torch.no_grad():
        residual = torch.linalg.vector_norm(k @ result-rhs)
        scale = torch.linalg.vector_norm(k)*torch.linalg.vector_norm(result)+torch.linalg.vector_norm(rhs)
        if not torch.isfinite(result).all() or residual/(scale+torch.finfo(k.dtype).eps) > 1e-10:
            raise ValueError("FP64 solve residual exceeds contract")
    return result


def transport(x: Any, y: Any) -> Any:
    import torch
    x, y = x.detach().double(), y.double()
    eye = torch.eye(x.shape[0], dtype=x.dtype, device=x.device)
    return solve(x @ x.T/x.shape[1]+EPSILON*eye,
                 (y @ x.T/x.shape[1]+EPSILON*eye).T).T


def historical_loss(y: Any, w: Any, x: Any, h: Any, b: Any,
                    constant: float, count: int) -> Any:
    import torch
    if count <= 0 or y.shape != x.shape:
        raise ValueError("Missing or misaligned function history")
    y, w = y.double(), w.double()
    x = torch.as_tensor(x, dtype=y.dtype, device=y.device).detach()
    h = torch.as_tensor(h, dtype=y.dtype, device=y.device).detach()
    b = torch.as_tensor(b, dtype=y.dtype, device=y.device).detach()
    k = x @ x.T/x.shape[1]+EPSILON*torch.eye(x.shape[0], dtype=y.dtype, device=y.device)
    v = solve(k, x @ (y.T @ w)/x.shape[1]+EPSILON*w)
    quadratic, linear = v @ h @ v, b @ v
    loss = (quadratic-2*linear+constant)/(3*count)
    tolerance = 1e-10*(1+quadratic.detach().abs()+2*linear.detach().abs()+abs(constant))/(3*count)
    if not torch.isfinite(loss) or loss.detach() < -tolerance:
        raise ValueError("Invalid function residual quadratic")
    return loss  # Preserve signed cancellation error; do not clamp.


class Memory:
    """One joint statistic and six raw groups; all persistent fields are charged."""

    def __init__(self, order: str, seed: int = 20260930):
        if order not in parent.ORDERS:
            raise ValueError("Unknown order")
        self.order, self.seed = order, seed
        self.reservoir = data.Memory(data.seed_for(seed, order, "retention"))
        self.h = np.zeros((DIMENSION, DIMENSION), dtype="<f8")
        self.b = np.zeros(DIMENSION, dtype="<f8")
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
                    features: Callable[[Any], np.ndarray], teacher: Any) -> dict:
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
            h, b, constant = statistics(torch.from_numpy(z), group.labels, teacher)
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
                or not np.isfinite(self.constant) or self.h.shape != (DIMENSION, DIMENSION)
                or self.b.shape != (DIMENSION,)):
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
        result.h = np.frombuffer(payload, "<f8", DIMENSION**2, offset).reshape(DIMENSION, DIMENSION).copy()
        offset += DIMENSION**2*8
        result.b = np.frombuffer(payload, "<f8", DIMENSION, offset).copy()
        offset += DIMENSION*8
        result.constant, result.count = struct.unpack("<dq", payload[offset:offset+16])
        offset += 16
        for uid in obj["uids"]:
            result.references[uid] = np.frombuffer(payload, "<f2", DIMENSION*378, offset).reshape(DIMENSION, 378).copy()
            offset += DIMENSION*378*2
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
    if (z.shape != (DIMENSION, 378) or z.dtype not in (np.float16, np.float32)
            or not np.isfinite(z).all() or not np.all(z[-1] == 1) or np.any(z[:-1] < 0)):
        raise ValueError("Invalid canonical function reference")
    rounded = z.astype("<f2")
    if not np.isfinite(rounded).all() or not np.array_equal(z, rounded.astype(z.dtype)):
        raise ValueError("Reference was not rounded to canonical FP16 coordinates")
    return np.ascontiguousarray(rounded)


def reference(model: Any, group: Any, c: dict,
              check: Callable[[], None] = lambda: None) -> np.ndarray:
    import torch
    model.train()
    with deterministic_history(model), torch.no_grad():
        return canonical_reference(relation_features(model, group, c, check).cpu().numpy())


def optimization_step(model: Any, optimizer: Any, current: Any, history: Any,
                      x: Any, memory: Memory, c: dict, encoder_lr: float,
                      *, current_seed: int, history_seed: int,
                      check: Callable[[], None] = lambda: None,
                      observe_history: Callable[[Any, Any], None] | None = None,
                      observe_stage: Callable[[str], None] = lambda stage: None) -> dict:
    import torch
    previous = parent.adam_step(optimizer)
    if (len(optimizer.param_groups) != 2 or history is None or x is None
            or current.uid == history.uid or memory.count <= 0):
        raise ValueError("Function update needs distinct current and history groups")
    for group, lr in zip(optimizer.param_groups, (encoder_lr, .001), strict=True):
        group["lr"] = lr
    model.train()
    optimizer.zero_grad(set_to_none=True)
    observe_stage("current_forward_backward")
    torch.manual_seed(current_seed)
    scores = base.logits(model, current, c, "split_rank", check)
    truth = torch.tensor(current.labels, dtype=torch.float32, device=scores.device)
    terms = parent.ranking.objectives(scores, truth, .5)
    terms["total"].backward()
    result = {"current_"+k: float(v.detach()) for k, v in terms.items()}
    del scores, truth, terms
    observe_stage("history_forward_backward")
    torch.manual_seed(history_seed)
    with deterministic_history(model):
        scores, z = forward_group(model, history, c, check)
        truth = torch.tensor(history.labels, dtype=torch.float32, device=scores.device)
        terms = parent.ranking.objectives(scores, truth, .5)
        penalty = historical_loss(z, weight(model), x, memory.h, memory.b, memory.constant, memory.count)
        if observe_history is not None:
            observe_history(terms["total"], penalty)
        (.1*terms["total"]+.5*penalty).backward()
        result.update({"history_"+k: float(v.detach()) for k, v in terms.items()})
        result["function"] = float(penalty.detach())
        del scores, z, truth, terms, penalty
    observe_stage("clip_and_adam")
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
    check()
    optimizer.step()
    check()
    if parent.adam_step(optimizer) != previous+1:
        raise ValueError("Expected exactly one AdamW update")
    return {**result, "total": result["current_total"]+.1*result["history_total"]+.5*result["function"],
            "gradient_norm": float(norm), "history_uid": history.uid,
            "adam_step": previous+1, "encoder_lr": encoder_lr, "head_lr": .001}


def update(model: Any, optimizer: Any, current: Any, memory: Memory, c: dict,
           stage: int, step: int, *, check: Callable[[], None] = lambda: None) -> dict:
    if (stage not in (2, 3) or not 1 <= step <= 288
            or parent.adam_step(optimizer) != (stage-1)*288+step-1
            or memory.stage != stage-1 or memory.draw_stage != stage or memory.draw_count != step-1
            or any(current.uid == g.uid for g in memory.reservoir.groups)):
        raise ValueError("Update phase or current/history separation differs")
    p = parent.contract()
    stream = data.seed_for(p["schedule_seed"], memory.order, stage, "current")
    history, x = memory.draw()
    result = optimization_step(model, optimizer, current, history, x, memory, c, parent.stage_lr(step),
        current_seed=data.seed_for(stream, step-1, "dropout"),
        history_seed=data.seed_for(p["memory_seed"], memory.order, stage, step-1, "history_dropout"), check=check)
    return {**result, "stage": stage, "step": step}


