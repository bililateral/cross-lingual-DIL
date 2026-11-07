"""Limited-history BGE pilot primitives; no formal data or model loading here."""
from __future__ import annotations

import copy
import hashlib
import json
import os
import random
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np

import step28_alias_ranking as ranking
import step28_alias_calibration as calibration

base, data, core, metrics = ranking.base, ranking.data, ranking.core, ranking.metrics
POLICY = data.ROOT / "schema/step28_bge_continual_policy.json"
POLICY_SHA256 = "28036c44903bb848d596c71ae987ce81beb2edc181fb846c010c4b70f6195534"
ORDERS = ("ABC", "BCA", "CAB")
METHODS = ("frozen", "seq", "er", "logit")
UPDATED = METHODS[1:]
ROLES = ("raw", "stage-cal", "first-cal")
STEP_COLUMNS = ("current_bce", "current_rank", "current_hard", "current_total",
                "history_bce", "history_rank", "history_hard", "history_total",
                "logit_mse", "total", "encoder_lr", "head_lr", "gradient_norm",
                "logit_term_host_seconds")


def configure_numerics() -> dict:
    """Apply the historical LOGIT runtime before model work; read back actual flags."""
    import torch
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
        raise RuntimeError("Start Python with CUBLAS_WORKSPACE_CONFIG=:4096:8")
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    return {
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "deterministic_warn_only": torch.is_deterministic_algorithms_warn_only_enabled(),
        "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cublas_workspace_config": os.environ["CUBLAS_WORKSPACE_CONFIG"],
    }


def contract() -> dict:
    if data.sha256(POLICY) != POLICY_SHA256:
        raise ValueError("Continual policy differs from confirmed specification")
    policy = data.read_json(POLICY)
    ranking.contract()  # Checks inherited BGE policy/source without data access.
    if (policy["orders"] != list(ORDERS) or policy["methods"] != list(METHODS)
            or policy["physical_updates"] != 6048 or policy["updates_per_stage"] != 288
            or policy["supervision"]["test_access"] or policy["supervision"]["owners_access"]):
        raise ValueError("Unexpected pilot scope")
    return policy


def config(policy: dict) -> dict:
    result = copy.deepcopy(base.contract())
    result["initialization_seed"] = policy["initialization_seed"]
    result["schedule_seed"] = policy["schedule_seed"]
    result["runtime"] = copy.deepcopy(policy["runtime"])
    return result


def stage_lr(step: int) -> float:
    if type(step) is not int or not 1 <= step <= 288:
        raise ValueError("Stage step must be one-based in 1..288")
    return 1e-5 * (step / 29 if step <= 29 else (288 - step) / 259)


def schedule(current: list, policy: dict, order: str, stage: int) -> tuple[list, int]:
    if order not in ORDERS or stage not in (1, 2, 3) or len(current) != 48:
        raise ValueError("Only the 48 current fitting groups enter a stage")
    if len({g.uid for g in current}) != 48 or any(g.labels is None for g in current):
        raise ValueError("Missing or repeated current supervision")
    stream = data.seed_for(policy["schedule_seed"], order, stage, "current")
    return data.schedule(current, 6, stream), stream


class Supply:
    """Offline dispatcher. Never pass this object or its full archive to a learner."""

    def __init__(self, groups: dict, partition: dict):
        self.groups, self.partition = groups, partition
        self.next_stage: dict[str, int] = {}
        self.orders: dict[str, str] = {}
        seen = set()
        for role, count in (("fit", 48), ("calibration", 12)):
            rows = partition[role]
            if len(rows) != 3 * count:
                raise ValueError("Incomplete dispatcher partition")
            for group, row in zip(groups[role], rows, strict=True):
                if (group.uid != row["group_uid"] or row["domain"] not in "ABC"
                        or group.uid in seen or group.labels is None):
                    raise ValueError("Dispatcher identity, supervision or role separation differs")
                seen.add(group.uid)
            if any(sum(row["domain"] == d for row in rows) != count for d in "ABC"):
                raise ValueError("Dispatcher domain sizes differ")

    def current(self, path: str, order: str, stage: int) -> tuple[list, list]:
        if (order not in ORDERS or stage != self.next_stage.get(path, 1)
                or self.orders.get(path, order) != order):
            raise ValueError("Future, old or repeated stage request")
        if stage not in (1, 2, 3):
            raise ValueError("No fourth domain")
        result = []
        for role, count in (("fit", 48), ("calibration", 12)):
            selected = [g for g, row in zip(self.groups[role], self.partition[role], strict=True)
                        if row["domain"] == order[stage - 1]]
            if len(selected) != count:
                raise ValueError("Current domain role count differs")
            result.append(selected)
        self.next_stage[path] = stage + 1
        self.orders[path] = order
        return result[0], result[1]

    def branch_after_first(self, path: str, shared_path: str) -> None:
        if path in self.next_stage or self.next_stage.get(shared_path) != 2:
            raise ValueError("Branches must start from one completed first stage")
        self.next_stage[path] = 2
        self.orders[path] = self.orders[shared_path]


class Memory:
    """Self-contained Algorithm R state, fixed first map and optional float32 targets.

    Receives only retained data and the current stage, never an archive callback.
    JSON sizes include every serialized field, not just numeric tensor payloads.
    """

    def __init__(self, order: str, seed: int, with_logits: bool, first_map: dict):
        if order not in ORDERS:
            raise ValueError("Unknown order")
        self.order, self.seed, self.with_logits = order, seed, with_logits
        self.first_map = dict(zip(("a", "b"), calibration.parameters(first_map)))
        self.reservoir = data.Memory(data.seed_for(seed, order, "retention"))
        self.references: dict[str, np.ndarray] = {}
        self.reference_origins: dict[str, int] = {}
        self.draw_rng: random.Random | None = None
        self.draw_stage, self.draw_count = 0, 0
        # Serialized branch/phase/RNG/map metadata is charged in addition to samples.
        # Current dropout uses stateless per-update seeds; no uncharged token cache.
        self.auxiliary: dict = {}

    def retain(self, current: list, stage: int, score: Callable[[list], np.ndarray] | None) -> None:
        if (stage not in (1, 2) or self.reservoir.seen != (stage - 1) * 48
                or len(current) != 48 or len({g.uid for g in current}) != 48
                or any(g.labels is None for g in current)
                or {g.uid for g in current} & {g.uid for g in self.reservoir.groups}
                or (stage == 2 and (self.draw_stage != 2 or self.draw_count != 288))
                or (self.with_logits and score is None)):
            raise ValueError("Only one unique insertion per originating stage")
        previous = set(self.references)
        self.reservoir.add_stage(sorted(current, key=lambda g: g.uid))
        retained = {g.uid for g in self.reservoir.groups}
        self.references = {uid: value for uid, value in self.references.items() if uid in retained}
        self.reference_origins = {uid: value for uid, value in self.reference_origins.items()
                                  if uid in retained}
        if self.with_logits:
            newcomers = [g for g in self.reservoir.groups if g.uid not in previous]
            if newcomers:
                values = np.asarray(score(newcomers))
                if values.shape != (len(newcomers), 378) or values.dtype != np.float32 or not np.isfinite(values).all():
                    raise ValueError("Origin-stage targets must be finite float32 raw logits")
                for group, row in zip(newcomers, values, strict=True):
                    self.references[group.uid] = row.copy()
                    self.reference_origins[group.uid] = stage
        self.draw_rng, self.draw_stage, self.draw_count = None, 0, 0
        self.to_bytes()

    def begin_stage(self, stage: int) -> None:
        if (stage not in (2, 3) or self.reservoir.seen != (stage - 1) * 48
                or self.draw_stage != 0 or len(self.reservoir.groups) != 6):
            raise ValueError("Historical memory does not match the arriving stage")
        self.draw_rng = random.Random(data.seed_for(self.seed, self.order, stage, "history_draws"))
        self.draw_stage, self.draw_count = stage, 0
        self.to_bytes()

    def draw(self) -> tuple[data.Group, np.ndarray | None]:
        if self.draw_rng is None or self.draw_count >= 288:
            raise ValueError("Missing history stage or extra draw")
        group = self.reservoir.groups[self.draw_rng.randrange(len(self.reservoir.groups))]
        self.draw_count += 1
        return group, self.references[group.uid].copy() if self.with_logits else None

    def to_bytes(self) -> bytes:
        retained = {g.uid for g in self.reservoir.groups}
        if (self.with_logits and (set(self.references) != retained or set(self.reference_origins) != retained)):
            raise ValueError("Reference coverage differs from retained samples")
        if not self.with_logits and (self.references or self.reference_origins):
            raise ValueError("ER cannot silently use historical logits")
        payload = data.json_bytes({"order": self.order, "seed": self.seed,
                                  "with_logits": self.with_logits, "first_map": self.first_map,
                                  "auxiliary": self.auxiliary,
                                  "reservoir": json.loads(self.reservoir.to_bytes()),
                                  "references": {uid: row.tolist() for uid, row in self.references.items()},
                                  "reference_origins": self.reference_origins,
                                  "draw_stage": self.draw_stage, "draw_count": self.draw_count,
                                  "draw_rng": self.draw_rng.getstate() if self.draw_rng is not None else None})
        if len(payload) > 1048576:
            raise ValueError("Complete serialized history and auxiliary state exceed 1MiB")
        return payload

    @classmethod
    def from_bytes(cls, payload: bytes) -> Memory:
        if len(payload) > 1048576:
            raise ValueError("History payload exceeds budget")
        obj = json.loads(payload)
        result = cls(obj["order"], obj["seed"], obj["with_logits"], obj["first_map"])
        result.reservoir = data.Memory.from_bytes(data.json_bytes(obj["reservoir"]))
        if result.reservoir.capacity != 6 or result.reservoir.maximum_bytes != 1048576:
            raise ValueError("Reservoir budget differs")
        result.references = {uid: np.asarray(row, dtype=np.float32) for uid, row in obj["references"].items()}
        if any(row.shape != (378,) or not np.isfinite(row).all() for row in result.references.values()):
            raise ValueError("Bad reference vector")
        result.reference_origins = obj["reference_origins"]
        result.auxiliary = obj["auxiliary"]
        result.draw_stage, result.draw_count = obj["draw_stage"], obj["draw_count"]
        if obj["draw_rng"] is not None:
            result.draw_rng = random.Random()
            result.draw_rng.setstate(data.tuple_state(obj["draw_rng"]))
        if result.to_bytes() != payload:
            raise ValueError("History restore is not exact")
        return result

    def summary(self) -> dict:
        payload = self.to_bytes()
        return {"members": [g.uid for g in self.reservoir.groups], "seen": self.reservoir.seen,
                "serialized_bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest(),
                "with_logits": self.with_logits, "draw_stage": self.draw_stage,
                "auxiliary_serialized_bytes": len(data.json_bytes(self.auxiliary)),
                "draw_count": self.draw_count, "reference_origins": dict(self.reference_origins),
                "references": {uid: hashlib.sha256(row.tobytes()).hexdigest()
                               for uid, row in self.references.items()}}


def adam_step(optimizer: Any) -> int:
    steps = {int(float(state["step"])) for state in optimizer.state.values()}
    if not steps:
        return 0
    if len(steps) != 1:
        raise ValueError("Adam parameter steps disagree")
    return steps.pop()


def update(model: Any, optimizer: Any, current: data.Group, history: data.Group | None,
           reference: np.ndarray | None, c: dict, method: str, stage: int,
           step: int, current_seed: int, history_seed: int, *, observe: bool = False,
           check: Callable[[], None] = lambda: None) -> dict:
    """Separate live current/history graphs, sum gradients, clip once and step once."""
    import torch

    if (method not in UPDATED or stage not in (1, 2, 3) or current.labels is None
            or adam_step(optimizer) != (stage - 1) * 288 + step - 1
            or (history is None) != (stage == 1 or method == "seq")
            or (reference is not None) != (method == "logit" and stage > 1)):
        raise ValueError("Update does not match current/history/Adam contract")
    if history is not None and (history.labels is None or history.uid == current.uid):
        raise ValueError("History must be a separate supervised old group")
    if len(optimizer.param_groups) != 2:
        raise ValueError("Only encoder/head optimizer groups expected")
    rates = (stage_lr(step), 1e-3)
    for group, rate in zip(optimizer.param_groups, rates, strict=True):
        group["lr"] = rate
    model.train()
    optimizer.zero_grad(set_to_none=True)
    before = {name: core.state_digest(module.state_dict()) for name, module in
              (("encoder", model.encoder), ("head", model.head))} if observe else {}
    result = {name: 0. for name in STEP_COLUMNS}
    # Per-forward seeds pair the current path independently of replay draws/dropout.
    torch.manual_seed(current_seed)
    predicted = base.logits(model, current, c, "split_rank", check)
    truth = torch.tensor(current.labels, dtype=torch.float32, device=predicted.device)
    terms = ranking.objectives(predicted, truth, .5)
    terms["total"].backward()
    result.update({"current_" + k: float(v.detach()) for k, v in terms.items()})
    del predicted, truth, terms
    if history is not None:
        torch.manual_seed(history_seed)
        predicted = base.logits(model, history, c, "split_rank", check)
        truth = torch.tensor(history.labels, dtype=torch.float32, device=predicted.device)
        terms = ranking.objectives(predicted, truth, .5)
        penalty = predicted.new_zeros(())
        if reference is not None:
            if reference.shape != (378,) or reference.dtype != np.float32 or not np.isfinite(reference).all():
                raise ValueError("Historical reference shape/dtype differs")
            tick = time.perf_counter()
            target = torch.tensor(reference, dtype=predicted.dtype, device=predicted.device)
            penalty = (predicted - target).square().mean()
            result["logit_term_host_seconds"] = time.perf_counter() - tick
        (terms["total"] + .5 * penalty).backward()
        result.update({"history_" + k: float(v.detach()) for k, v in terms.items()})
        result["logit_mse"] = float(penalty.detach())
    result["total"] = result["current_total"] + result["history_total"] + .5 * result["logit_mse"]
    evidence = {}
    if observe:
        for name, module in (("encoder", model.encoder), ("head", model.head)):
            norms = [float(p.grad.float().norm()) for p in module.parameters() if p.grad is not None]
            if not norms or not np.isfinite(norms).all() or max(norms) <= 0:
                raise ValueError("No finite task gradient in " + name)
            evidence[name] = {"finite_nonzero_combined_gradient": True}
    result["gradient_norm"] = float(torch.nn.utils.clip_grad_norm_(
        model.parameters(), c["optimizer"]["clip_norm"], error_if_nonfinite=True))
    check()
    optimizer.step()
    if adam_step(optimizer) != (stage - 1) * 288 + step:
        raise ValueError("Actual Adam continuity differs")
    if observe:
        for name, module in (("encoder", model.encoder), ("head", model.head)):
            changed = before[name] != core.state_digest(module.state_dict())
            if changed != (name == "head" or rates[0] > 0):
                raise ValueError("Unexpected actual parameter change: " + name)
            evidence[name]["parameters_changed"] = changed
    result.update(encoder_lr=rates[0], head_lr=rates[1], modules=evidence,
                  stage=stage, step=step, adam_step=adam_step(optimizer))
    return result
