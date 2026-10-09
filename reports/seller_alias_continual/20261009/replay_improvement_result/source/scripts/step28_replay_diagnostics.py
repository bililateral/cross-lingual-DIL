"""Epoch diagnostics on current fit/cache and blind valid, without learner mutation."""
from __future__ import annotations

from contextlib import contextmanager
import random
from pathlib import Path
from typing import Any, Callable

import numpy as np

import step28_record_replay as record
import step28_bge_continual_run as previous

parent, data = record.parent, record.data
COLUMNS = ("map", "recall_at_5", "log_loss")


def rng_state() -> dict:
    value = np.random.get_state()
    return {"torch": previous.rng_state(), "python": random.getstate(),
            "numpy": [value[0], value[1].tolist(), int(value[2]), int(value[3]), float(value[4])]}


def restore_rng(state: dict) -> None:
    previous.restore_rng(state["torch"])
    random.setstate(data.tuple_state(state["python"]))
    n = state["numpy"]
    np.random.set_state((n[0], np.asarray(n[1], np.uint32), n[2], n[3], n[4]))


@contextmanager
def neutral(model: Any):
    """Restore individual modes and global RNGs even if a diagnostic fails."""
    import torch
    state, modes = rng_state(), [(m, m.training) for m in model.modules()]
    try:
        model.eval()
        with torch.inference_mode():
            yield
    finally:
        for module, mode in modes:
            module.training = mode
        restore_rng(state)


def score(model: Any, groups: list, c: dict, arm: str, check: Callable) -> np.ndarray:
    if not groups:
        return np.empty((0, 378), np.float32)
    return (parent.ranking.score if arm == "LOGIT0.1" else record.score)(model, groups, c, check)


def metrics(groups: list, values: np.ndarray) -> np.ndarray:
    if not groups:
        return np.empty((0, len(COLUMNS)), np.float64)
    if any(g.labels is None for g in groups):
        raise ValueError("Only current authorized labelled groups may be scored")
    matrix, _ = parent.metrics.group_metrics(np.asarray([g.labels for g in groups], np.uint8), values)
    return matrix[:, [parent.metrics.COLUMNS.index(k) for k in COLUMNS]]


def capture(model: Any, optimizer: Any, current: list, memory: Any, valid: list,
            c: dict, arm: str, order: str, stage: int, epoch: int, root: Path,
            point: str, check: Callable = lambda: None) -> dict:
    """Called by the actual training loop after every 48 updates; never loads labels."""
    import torch
    if epoch not in range(1, 7) or parent.adam_step(optimizer) != (stage-1)*288+epoch*48:
        raise ValueError("Diagnostic not at the registered epoch endpoint")
    if len(current) != 48 or len(valid) != 60 or any(g.labels is not None for g in valid):
        raise ValueError("Diagnostic fit/valid scope differs")
    cache = [] if memory is None else list(memory.reservoir.groups)
    if len(cache) != (0 if stage == 1 else 6):
        raise ValueError("Wrong surviving cache")
    before = memory.to_bytes() if memory is not None else None
    step_before = parent.adam_step(optimizer)
    origins = ({} if memory is None else memory.reference_origins if arm == "LOGIT0.1" else memory.origins)
    with neutral(model):
        fit_values = metrics(current, score(model, current, c, arm, check))
        cache_scores = score(model, cache, c, arm, check)
        cache_values = metrics(cache, cache_scores)
        blind = score(model, valid, c, arm, check)
        errors = []
        for i, group in enumerate(cache):
            check()
            if arm == "LOGIT0.1":
                e0 = cache_scores[i] - memory.references[group.uid]
                errors.append({"uid": group.uid, "D0": float(np.mean(e0.astype(np.float64)**2)),
                               "D1": None, "residual_cross_mean": None, "residual_difference_mse": None})
            else:
                live = record.reference(model, group, c, check)
                target = memory.tables[group.uid]
                seeds = (None, data.seed_for(20261008, order, stage, group.uid, "diagnostic_regroup"))
                residual = []
                for seed in seeds:
                    slots = record.assignment(group, seed)
                    # Eval-only CPU pooling of the same FP32 tables; no model gradient.
                    residual.append((record.account_logits(torch.from_numpy(live), slots)
                                    -record.account_logits(torch.from_numpy(target), slots)).numpy().astype(np.float64))
                e0, e1 = residual
                errors.append({"uid": group.uid, "D0": float(np.mean(e0**2)), "D1": float(np.mean(e1**2)),
                               "residual_cross_mean": float(np.mean(e0*e1)),
                               "residual_difference_mse": float(np.mean((e1-e0)**2))})
    if (parent.adam_step(optimizer) != step_before
            or (memory.to_bytes() if memory is not None else None) != before):
        raise ValueError("Diagnostics changed optimizer step or bounded history")
    tag = f"{point}_epoch{epoch}"
    result = {"point": point, "order": order, "arm": arm, "stage": stage, "epoch": epoch,
              "stage_step": epoch*48, "adam_step": step_before, "seed": "s0", "metric_columns": list(COLUMNS),
              "fit_ids": [g.uid for g in current], "fit_domains": [order[stage-1]]*48,
              "cache_ids": [g.uid for g in cache], "cache_domains": [order[origins[g.uid]-1] for g in cache],
              "valid_ids": [g.uid for g in valid], "teacher_errors": errors,
              "mode": "A0_eval_no_autograd", "valid_labels_used": False,
              "fit": previous.save_array(root/(tag+"_fit.npy"), fit_values, root),
              "cache": previous.save_array(root/(tag+"_cache.npy"), cache_values, root),
              "valid_blind": previous.save_array(root/(tag+"_valid.npy"), blind, root)}
    path = root/(tag+".json")
    data.write_json(path, result)
    return data.record(path, root)


def summary(values: np.ndarray, draws: np.ndarray) -> dict:
    estimates = values[draws].mean(axis=1)
    return {"mean": values.mean(axis=0).tolist(),
            "conditional_95pct_interval": np.quantile(estimates, [.025, .975], axis=0).T.tolist()}


def curve_comparison(train: np.ndarray, valid: np.ndarray, seed: int) -> dict:
    """Six epochs x actual groups x three metrics; independent train/valid groups."""
    if train.shape[0] != 6 or valid.shape[0] != 6 or train.shape[2] != 3 or valid.shape[2] != 3:
        raise ValueError("Complete six-point curves required")
    if train.shape[1] == 0:
        return {"status": "NOT_APPLICABLE_NO_SURVIVING_GROUPS"}
    rng = np.random.Generator(np.random.PCG64(seed))
    td = rng.integers(train.shape[1], size=(5000, train.shape[1]))
    vd = rng.integers(valid.shape[1], size=(5000, valid.shape[1]))
    # Positive benefit is up for MAP/R5 and down for unweighted BCE.
    signs = np.array([1., 1., -1.])
    train_change = (train[-1]-train[0])*signs
    valid_change = (valid[-1]-valid[0])*signs
    tdelta, vdelta = summary(train_change, td), summary(valid_change, vd)
    gap = (train.mean(1)-valid.mean(1))*signs
    boot_gap = (train[:, td].mean(2)-valid[:, vd].mean(2))*signs
    widening = boot_gap[-1]-boot_gap[0]
    bounds = np.quantile(widening, [.025, .975], axis=0).T
    decisions = []
    for k in range(3):
        point_signal = tdelta["mean"][k] > 0 and vdelta["mean"][k] < 0 and gap[-1,k] > gap[0,k]
        supported = tdelta["conditional_95pct_interval"][k][0] > 0 and vdelta["conditional_95pct_interval"][k][1] < 0 and bounds[k,0] > 0
        decisions.append("有相应迹象" if supported else "证据不足" if point_signal else "未见明确迹象")
    return {"status": "DESCRIPTIVE_CONDITIONAL_DEVELOPMENT_DIAGNOSIS", "columns": list(COLUMNS),
            "fit_or_cache_groups": train.shape[1], "valid_groups": valid.shape[1],
            "train_raw_curve": train.mean(1).tolist(), "valid_raw_curve": valid.mean(1).tolist(),
            "benefit_gap_curve": gap.tolist(), "benefit_gap_conditional_intervals": np.quantile(boot_gap, [.025,.975], axis=1).transpose(1,2,0).tolist(),
            "epoch1_to_6_train_benefit": tdelta, "epoch1_to_6_valid_benefit": vdelta,
            "epoch1_to_6_gap_widening": {"mean": (gap[-1]-gap[0]).tolist(), "conditional_95pct_interval": bounds.tolist()},
            "interpretation": dict(zip(COLUMNS, decisions)),
            "limits": "No absence proof, causal attribution, seed coverage, selection-bias correction, or early stopping."}


def evaluate(root: Path, manifest: dict, labelled: list, partition: dict, out: Path) -> dict:
    """Only the caller after complete blind gates may supply newly opened valid."""
    if [g.uid for g in labelled] != [r["group_uid"] for r in partition["development"]]:
        raise ValueError("Valid group order mismatch")
    valid_domains = [r["domain"] for r in partition["development"]]
    report = {"columns": list(COLUMNS), "points": {}, "group_metrics": {}}
    out.mkdir()
    for point, records in manifest["diagnostics"].items():
        entries = [data.read_json(data.verify(root/r["path"], r)) for r in records]
        if [e["epoch"] for e in entries] != list(range(1,7)):
            raise ValueError("Epoch coverage differs")
        curves = {role: [] for role in ("fit", "cache", "valid")}
        for entry in entries:
            for key in ("fit_ids", "cache_ids", "fit_domains", "cache_domains", "valid_ids"):
                if entry[key] != entries[0][key]:
                    raise ValueError("Curve population changed within phase")
            for role in ("fit", "cache"):
                curves[role].append(previous.array(root, entry[role], (len(entry[role+"_ids"]),3), np.float64))
            values = metrics(labelled, previous.array(root, entry["valid_blind"], (60,378), np.float32))
            curves["valid"].append(values)
            report["group_metrics"][f"{point}_epoch{entry['epoch']}"] = previous.save_array(out/f"{point}_epoch{entry['epoch']}_valid_metrics.npy", values, out)
        curves = {k: np.stack(v) for k,v in curves.items()}
        entry = entries[0]
        result = {}
        for role in ("fit", "cache"):
            result[role] = {}
            domains = [entry["order"][entry["stage"]-1]] if role == "fit" else list(entry["order"][:entry["stage"]-1])
            for domain in domains:
                selected = np.flatnonzero(np.asarray(entry[role+"_domains"]) == domain)
                vr = np.flatnonzero(np.asarray(valid_domains) == domain)
                # Exclude arm from seed: same populations across arms share draws.
                result[role][domain] = curve_comparison(curves[role][:,selected], curves["valid"][:,vr],
                    data.seed_for(20260930, entry["order"], entry["stage"], role, domain, "diagnostics"))
            if not domains:
                result[role] = {"status": "NOT_APPLICABLE_FIRST_STAGE"}
        report["points"][point] = result
    data.write_json(out/"overfitting.json", report)
    return report
