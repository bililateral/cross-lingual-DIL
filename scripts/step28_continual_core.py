#!/usr/bin/env python3
"""Minimal fixed-feature continual pair-head primitives and real CPU verification.

The check command uses hand-constructed numerical fixtures only. It does not
read project labels, run the formal comparison, or use CUDA. The full research
runner and independent development evaluator are subsequent work.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import io
import json
import platform
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

FEATURE_COUNT = 57
HIDDEN = 32
BUDGET = 512 * 1024
MEMORY_DTYPE = np.dtype([("features", "<f4", (FEATURE_COUNT,)), ("label", "u1")])


@dataclass
class Preprocessor:
    medians: np.ndarray
    means: np.ndarray
    scales: np.ndarray

    @classmethod
    def fit(cls, first_stage: np.ndarray) -> "Preprocessor":
        values = np.asarray(first_stage, dtype=np.float64)
        if values.ndim != 2 or not len(values) or values.shape[1] != FEATURE_COUNT or np.isinf(values).any():
            raise ValueError("Invalid first-stage features")
        medians = np.asarray([np.median(column[np.isfinite(column)])
                              if np.isfinite(column).any() else 0. for column in values.T])
        filled = np.where(np.isnan(values), medians, values)
        means, scales = filled.mean(axis=0), filled.std(axis=0, ddof=0)
        scales[scales == 0] = 1.
        return cls(medians, means, scales)

    def transform(self, values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != FEATURE_COUNT or np.isinf(values).any():
            raise ValueError("Invalid features")
        result = ((np.where(np.isnan(values), self.medians, values) - self.means) / self.scales)
        with np.errstate(over="ignore"):
            result = np.ascontiguousarray(result, dtype="<f4")
        if not np.isfinite(result).all():
            raise ValueError("Nonfinite transformed features")
        return result


def validate_examples(features: np.ndarray, labels: np.ndarray) -> None:
    if (features.ndim != 2 or features.shape[1] != FEATURE_COUNT
            or labels.shape != (len(features),) or not len(features)
            or not np.isfinite(features).all() or not np.isin(labels, [0, 1]).all()):
        raise ValueError("Invalid feature/label examples")


class Reservoir:
    """Uniform unique-arrival ER; update once AFTER completing each stage.

    Storage is the actual uncompressed NPY record array plus UTF-8 JSON state.
    Keep up to 1024 bytes for NPY headers and both persistent RNG states. The
    reported budget is retained historical state, not total process memory.
    """

    def __init__(self, budget: int, seed: int):
        self.budget = int(budget)
        self.capacity = (self.budget - 1024) // MEMORY_DTYPE.itemsize
        if self.capacity < 1:
            raise ValueError("Memory budget too small")
        self.records = np.empty(0, dtype=MEMORY_DTYPE)
        self.seen = 0
        self.selection_rng = np.random.default_rng(np.random.SeedSequence([seed, 1]))
        self.replay_rng = np.random.default_rng(np.random.SeedSequence([seed, 2]))

    def update_after_stage(self, features: np.ndarray, labels: np.ndarray) -> None:
        validate_examples(features, labels)
        current = np.empty(len(features), dtype=MEMORY_DTYPE)
        current["features"], current["label"] = features, labels
        if not np.isfinite(current["features"]).all():
            raise ValueError("Nonfinite serialized memory features")
        # The caller owns unique stage arrivals; repeated optimizer epochs must
        # never call this function. No discarded historical edge is read here.
        room = min(self.capacity - len(self.records), len(current))
        if room:
            self.records = np.concatenate((self.records, current[:room]))
            self.seen += room
        for record in current[room:]:
            self.seen += 1
            index = int(self.selection_rng.integers(0, self.seen))
            if index < self.capacity:
                self.records[index] = record
        self.serialized_parts()  # Enforce the actual retained-state budget.

    def sample(self, maximum: int) -> tuple[np.ndarray, np.ndarray] | None:
        if maximum < 1:
            raise ValueError("Replay batch must be positive")
        if not len(self.records):
            return None
        chosen = self.replay_rng.choice(len(self.records), min(maximum, len(self.records)), replace=False)
        return (np.ascontiguousarray(self.records["features"][chosen]),
                np.ascontiguousarray(self.records["label"][chosen]))

    def serialized_parts(self) -> tuple[bytes, bytes]:
        array_bytes = io.BytesIO()
        np.save(array_bytes, self.records, allow_pickle=False)
        state = {"budget": self.budget, "capacity": self.capacity, "seen": self.seen,
                 "selection_rng": self.selection_rng.bit_generator.state,
                 "replay_rng": self.replay_rng.bit_generator.state}
        state_bytes = json.dumps(state, sort_keys=True, separators=(",", ":")).encode("utf-8")
        payload = array_bytes.getvalue()
        if len(payload) + len(state_bytes) > self.budget:
            raise ValueError("Historical state exceeds byte budget")
        return payload, state_bytes

    @classmethod
    def restore(cls, array_bytes: bytes, state_bytes: bytes) -> "Reservoir":
        state = json.loads(state_bytes)
        obj = cls(state["budget"], 0)
        records = np.load(io.BytesIO(array_bytes), allow_pickle=False)
        if (records.dtype != MEMORY_DTYPE or records.ndim != 1 or len(records) > obj.capacity
                or state["capacity"] != obj.capacity or state["seen"] < len(records)
                or (len(records) and (not np.isfinite(records["features"]).all()
                                     or not np.isin(records["label"], [0, 1]).all()))):
            raise ValueError("Saved memory schema/state differs")
        obj.records, obj.seen = records, int(state["seen"])
        obj.selection_rng.bit_generator.state = state["selection_rng"]
        obj.replay_rng.bit_generator.state = state["replay_rng"]
        obj.serialized_parts()
        return obj


def torch_module():
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("PyTorch is required for real training verification; not installed here") from exc
    return torch


def make_head(seed: int = 20260907):
    torch = torch_module()
    # Avoid touching CUDA and do not consume other sampling RNG streams.
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed)
        model = torch.nn.Sequential(torch.nn.Linear(FEATURE_COUNT, HIDDEN),
                                    torch.nn.ReLU(), torch.nn.Linear(HIDDEN, 1))
    return model.cpu()


def make_optimizer(model):
    torch = torch_module()
    return torch.optim.Adam(model.parameters(), lr=.001, betas=(.9, .999),
                            eps=1e-8, weight_decay=0., foreach=False)


def clone_training_state(model, optimizer):
    clone = copy.deepcopy(model)
    cloned_optimizer = make_optimizer(clone)
    cloned_optimizer.load_state_dict(copy.deepcopy(optimizer.state_dict()))
    return clone, cloned_optimizer


def current_permutation(size: int, order_seed: int, stage: int, epoch: int) -> np.ndarray:
    return np.random.default_rng(np.random.SeedSequence([order_seed, stage, epoch, 3])).permutation(size)


def train_stage(model, optimizer, features: np.ndarray, labels: np.ndarray, *,
                order_seed: int, stage: int, memory: Reservoir | None = None,
                epochs: int = 5, batch_size: int = 256) -> dict[str, Any]:
    """Current mean BCE plus independent historical mean BCE, weight one each.

    Does NOT insert current rows in memory: orchestration must call
    update_after_stage exactly once afterwards. Stage 1 has no replay term.
    """
    torch = torch_module()
    validate_examples(features, labels)
    if epochs < 1 or batch_size < 1 or stage < 1 or (stage == 1 and memory is not None):
        raise ValueError("Invalid stage schedule or first-stage replay")
    model.train()
    x = torch.from_numpy(np.ascontiguousarray(features, dtype=np.float32))
    y = torch.from_numpy(np.ascontiguousarray(labels, dtype=np.float32))
    updates, new_rows, replay_rows, total_loss = 0, 0, 0, 0.
    order_digest = hashlib.sha256()
    memory_seen_before = memory.seen if memory else 0
    for epoch in range(epochs):
        order = current_permutation(len(x), order_seed, stage, epoch)
        order_digest.update(np.asarray(order, dtype="<i8").tobytes())
        for start in range(0, len(order), batch_size):
            indices = torch.from_numpy(order[start:start + batch_size])
            optimizer.zero_grad(set_to_none=True)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(model(x[indices]).squeeze(1), y[indices])
            old = memory.sample(batch_size) if memory is not None else None
            if old is not None:
                old_x, old_y = old
                loss = loss + torch.nn.functional.binary_cross_entropy_with_logits(
                    model(torch.from_numpy(old_x)).squeeze(1),
                    torch.from_numpy(old_y.astype(np.float32)))
                replay_rows += len(old_y)
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite training loss")
            loss.backward()
            if any(p.grad is None or not torch.isfinite(p.grad).all() for p in model.parameters()):
                raise ValueError("Missing/nonfinite head gradient")
            optimizer.step()
            if any(not torch.isfinite(p).all() for p in model.parameters()):
                raise ValueError("Nonfinite head state")
            updates += 1
            new_rows += len(indices)
            total_loss += float(loss.detach())
    if memory and memory.seen != memory_seen_before:
        raise ValueError("Training epochs incorrectly inserted memory arrivals")
    return {"updates": updates, "current_presentations": new_rows,
            "replay_presentations": replay_rows, "mean_update_loss": total_loss / updates,
            "current_order_sha256": order_digest.hexdigest()}


def score(model, features: np.ndarray, batch_size: int = 4096) -> np.ndarray:
    torch = torch_module()
    if (features.ndim != 2 or features.shape[1] != FEATURE_COUNT or not len(features)
            or not np.isfinite(features).all() or batch_size < 1):
        raise ValueError("Invalid score inputs")
    model.eval()
    values = []
    with torch.no_grad():
        for start in range(0, len(features), batch_size):
            x = torch.from_numpy(np.ascontiguousarray(features[start:start + batch_size], dtype=np.float32))
            values.append(model(x).squeeze(1).numpy().copy())
    result = np.concatenate(values)
    if not np.isfinite(result).all():
        raise ValueError("Nonfinite score")
    return result


def independent_bce_gradient(parameters: list[np.ndarray], x: np.ndarray,
                              y: np.ndarray) -> tuple[float, list[np.ndarray]]:
    """Independent NumPy equations for Linear/ReLU/Linear mean BCE verification."""
    w1, b1, w2, b2 = parameters
    z1 = x @ w1.T + b1
    hidden = np.maximum(z1, 0.)
    logits = (hidden @ w2.T + b2).ravel()
    loss = float(np.mean(np.logaddexp(0., logits) - y * logits))
    # Numerical fixtures have bounded logits; this branch is for verification.
    dlogit = ((1. / (1. + np.exp(-logits)) - y) / len(y))[:, None]
    dz1 = (dlogit @ w2) * (z1 > 0)
    return loss, [dz1.T @ x, dz1.sum(axis=0), dlogit.T @ hidden, dlogit.sum(axis=0)]


def runtime_check() -> dict[str, Any]:
    """Actual CPU autograd/update/replay checks, without scientific data labels."""
    torch = torch_module()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    rng = np.random.default_rng(43)
    x = rng.normal(0, .25, (7, FEATURE_COUNT))
    y = np.asarray([0., 1., 0., 1., 0., 0., 1.])
    model = make_head().double()
    # Positive biases keep the fixture away from the ReLU kink for finite differences.
    with torch.no_grad():
        model[0].bias.fill_(.4)
    parameters = [p.detach().numpy().copy() for p in model.parameters()]
    expected_loss, expected_gradient = independent_bce_gradient(parameters, x, y)
    loss = torch.nn.functional.binary_cross_entropy_with_logits(
        model(torch.from_numpy(x)).squeeze(1), torch.from_numpy(y))
    loss.backward()
    gradient_error = max(float(np.max(np.abs(p.grad.numpy() - g)))
                         for p, g in zip(model.parameters(), expected_gradient, strict=True))
    if abs(float(loss.detach()) - expected_loss) > 1e-12 or gradient_error > 1e-12:
        raise ValueError("NumPy/autograd full-gradient mismatch")
    finite_difference_error = 0.
    for parameter_index, (value, derivative) in enumerate(zip(parameters, expected_gradient, strict=True)):
        for coordinate in (0, value.size - 1):
            plus, minus = [p.copy() for p in parameters], [p.copy() for p in parameters]
            plus[parameter_index].flat[coordinate] += 1e-6
            minus[parameter_index].flat[coordinate] -= 1e-6
            difference = (independent_bce_gradient(plus, x, y)[0]
                          - independent_bce_gradient(minus, x, y)[0]) / 2e-6
            finite_difference_error = max(finite_difference_error,
                                          abs(difference - derivative.flat[coordinate]))
    if finite_difference_error > 1e-8:
        raise ValueError("Finite difference mismatch")
    optimizer = make_optimizer(model)
    optimizer.step()
    adam_error = max(float(np.max(np.abs(p.detach().numpy() -
                                        (before - .001 * g / (np.abs(g) + 1e-8)))))
                     for p, before, g in zip(model.parameters(), parameters, expected_gradient, strict=True))
    if adam_error > 1e-12:
        raise ValueError("First Adam update differs from independent equation")
    actual_updates = [bool(np.any(p.detach().numpy() != before))
                      for p, before in zip(model.parameters(), parameters, strict=True)]
    if not all(actual_updates):
        raise ValueError("Fixture did not update every head parameter tensor")

    # Independently verify the production replay objective on UNEQUAL current
    # and old batch sizes; pooling 7+3 rows would give different weights.
    probe = make_head()
    probe_optimizer = make_optimizer(probe)
    probe_x, probe_y = x.astype(np.float32), y.astype(np.uint8)
    old_x, old_y = (-probe_x[:3]).copy(), (1 - probe_y[:3]).copy()
    probe_memory = Reservoir(BUDGET, 23)
    probe_memory.update_after_stage(old_x, old_y)
    before = [p.detach().numpy().astype(np.float64) for p in probe.parameters()]
    current_loss, current_gradient = independent_bce_gradient(before, probe_x.astype(float), probe_y)
    old_loss, old_gradient = independent_bce_gradient(before, old_x.astype(float), old_y)
    combined_gradient = [a + b for a, b in zip(current_gradient, old_gradient, strict=True)]
    probe_log = train_stage(probe, probe_optimizer, probe_x, probe_y, order_seed=23,
                            stage=2, memory=probe_memory, epochs=1, batch_size=8)
    replay_gradient_error = max(float(np.max(np.abs(p.grad.numpy() - expected)))
                                for p, expected in zip(probe.parameters(), combined_gradient, strict=True))
    replay_update_error = max(float(np.max(np.abs(p.detach().numpy() -
                               (value - .001 * grad / (np.abs(grad) + 1e-8)))))
                              for p, value, grad in zip(probe.parameters(), before, combined_gradient, strict=True))
    if (abs(probe_log["mean_update_loss"] - current_loss - old_loss) > 1e-6
            or replay_gradient_error > 5e-7 or replay_update_error > 2e-6
            or probe_log["current_presentations"] != 7 or probe_log["replay_presentations"] != 3):
        raise ValueError("Production ER objective/full gradient/Adam update differs from NumPy")

    # Exercise the production train_stage path with a partial final batch.
    raw = rng.normal(size=(34, FEATURE_COUNT))
    raw[0, 0] = np.nan
    preprocessing = Preprocessor.fit(raw[:17])
    first, second = preprocessing.transform(raw[:17]), preprocessing.transform(raw[17:])
    first_y = (np.arange(17) % 3 == 0).astype(np.uint8)
    second_y = (np.arange(17) % 4 == 0).astype(np.uint8)
    shared = make_head()
    shared_optimizer = make_optimizer(shared)
    initial_parameters = [p.detach().numpy().copy() for p in shared.parameters()]
    initial_scores = score(shared, first)
    first_log = train_stage(shared, shared_optimizer, first, first_y, order_seed=11,
                            stage=1, epochs=2, batch_size=8)
    frozen_scores = score(shared, first)
    if np.array_equal(initial_scores, frozen_scores) or any(
        np.array_equal(p.detach().numpy(), before)
        for p, before in zip(shared.parameters(), initial_parameters, strict=True)
    ):
        raise ValueError("Production first-stage path did not update all head tensors")
    sequential, sequential_optimizer = clone_training_state(shared, shared_optimizer)
    replay, replay_optimizer = clone_training_state(shared, shared_optimizer)
    paired, paired_optimizer = clone_training_state(shared, shared_optimizer)
    original_states = list(shared_optimizer.state.values())
    if len(original_states) != len(list(shared.parameters())) or any(
        int(state["step"].item()) != first_log["updates"] for state in original_states
    ):
        raise ValueError("First-stage Adam state is missing or has wrong step count")
    for cloned_model, cloned_optimizer in ((sequential, sequential_optimizer),
                                           (replay, replay_optimizer), (paired, paired_optimizer)):
        if cloned_optimizer.state_dict()["param_groups"] != shared_optimizer.state_dict()["param_groups"]:
            raise ValueError("Shared optimizer hyperparameters differ")
        if len(cloned_optimizer.state) != len(original_states):
            raise ValueError("Clone has missing or additional Adam state")
        for original_parameter, cloned_parameter in zip(shared.parameters(), cloned_model.parameters(), strict=True):
            if not torch.equal(original_parameter, cloned_parameter):
                raise ValueError("Clone differs from original first-stage weights")
            original = shared_optimizer.state[original_parameter]
            cloned = cloned_optimizer.state.get(cloned_parameter)
            if cloned is None or set(cloned) != set(original) or any(
                not torch.equal(original[key], cloned[key])
                   for key in ("step", "exp_avg", "exp_avg_sq")):
                raise ValueError("Clone differs from original first-stage Adam state")
    memory = Reservoir(BUDGET, 11)
    memory.update_after_stage(first, first_y)
    before_memory = memory.serialized_parts()
    sequential_log = train_stage(sequential, sequential_optimizer, second, second_y,
                                 order_seed=11, stage=2, epochs=2, batch_size=8)
    replay_log = train_stage(replay, replay_optimizer, second, second_y, order_seed=11,
                             stage=2, memory=memory, epochs=2, batch_size=8)
    restored_memory = Reservoir.restore(*before_memory)
    paired_log = train_stage(paired, paired_optimizer, second, second_y, order_seed=11,
                             stage=2, memory=restored_memory, epochs=2, batch_size=8)
    if first_log["updates"] != 6 or sequential_log["current_presentations"] != 34 or (
        sequential_log["replay_presentations"] != 0 or replay_log["replay_presentations"] != 48
        or sequential_log["current_order_sha256"] != replay_log["current_order_sha256"]
        or replay_log != paired_log or not np.array_equal(score(replay, first), score(paired, first))
        or not np.array_equal(frozen_scores, score(shared, first)) or memory.seen != 17
    ):
        raise ValueError("Paired stage/update/replay semantics differ")
    memory.update_after_stage(second, second_y)
    if memory.seen != 34:
        raise ValueError("Unique stage arrival count")
    # A checkpoint must replay scores and the NEXT Adam/replay update, not only weights.
    with tempfile.TemporaryDirectory(prefix="continual_check_") as temp:
        path = Path(temp) / "checkpoint.pt"
        torch.save({"model": replay.state_dict(), "optimizer": replay_optimizer.state_dict(),
                    "medians": torch.from_numpy(preprocessing.medians.copy()),
                    "means": torch.from_numpy(preprocessing.means.copy()),
                    "scales": torch.from_numpy(preprocessing.scales.copy())}, path)
        saved = torch.load(path, map_location="cpu", weights_only=True)
        reload_model = make_head()
        reload_model.load_state_dict(saved["model"])
        reload_optimizer = make_optimizer(reload_model)
        reload_optimizer.load_state_dict(saved["optimizer"])
        reloaded_preprocessing = Preprocessor(*(saved[k].numpy() for k in ("medians", "means", "scales")))
        if not np.array_equal(first, reloaded_preprocessing.transform(raw[:17])) or (
            not np.array_equal(score(replay, first), score(reload_model, first))
        ):
            raise ValueError("Checkpoint score/preprocessing replay differs")
        array_bytes, state_bytes = memory.serialized_parts()
        (Path(temp) / "memory.npy").write_bytes(array_bytes)
        (Path(temp) / "memory.json").write_bytes(state_bytes)
        disk_memory = Reservoir.restore((Path(temp) / "memory.npy").read_bytes(),
                                        (Path(temp) / "memory.json").read_bytes())
        train_stage(replay, replay_optimizer, first, first_y, order_seed=11, stage=3,
                    memory=memory, epochs=1, batch_size=8)
        train_stage(reload_model, reload_optimizer, first, first_y, order_seed=11, stage=3,
                    memory=disk_memory, epochs=1, batch_size=8)
        if not np.array_equal(score(replay, first), score(reload_model, first)) or (
            memory.serialized_parts() != disk_memory.serialized_parts()
            or any(not torch.equal(a, b) for a, b in zip(
                replay.parameters(), reload_model.parameters(), strict=True))
        ):
            raise ValueError("Checkpoint next-update replay differs")
    return {"status": "PASSED_REAL_CPU_CORE_CHECK_NO_PROJECT_TRAINING",
            "torch": torch.__version__, "numpy": np.__version__, "platform": platform.platform(),
            "device": "cpu", "torch_threads": 1, "project_label_reads": 0,
            "full_gradient_max_abs_error": gradient_error,
            "finite_difference_max_abs_error": finite_difference_error,
            "first_adam_max_abs_error": adam_error, "all_head_parameter_tensors_updated": actual_updates,
            "production_er_full_gradient_max_abs_error": replay_gradient_error,
            "production_er_adam_max_abs_error": replay_update_error,
            "first_stage": first_log, "sequential": sequential_log, "er": replay_log,
            "shared_first_stage_optimizer": True, "frozen_scores_unchanged": True,
            "checkpoint_scores_and_next_update_exact": True,
            "memory_capacity": memory.capacity, "memory_record_bytes": MEMORY_DTYPE.itemsize,
            "limits": ["Numerical fixtures only; no formal stage loader or evaluation verified",
                       "No measured forgetting or scientific method advantage"]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("check",))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Do not overwrite runtime evidence")
    result = runtime_check()
    result["script_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
