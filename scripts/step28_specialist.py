#!/usr/bin/env python3
"""Fixed-capacity full-supervision versus public no-strong specialist comparison."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
from collections import defaultdict
from itertools import zip_longest
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import sklearn
from sklearn.metrics import roc_curve

import step28_saved_model_audit as audit

ROOT = audit.ROOT
POLICY = ROOT / "schema/step28_specialist_policy.json"
ROW_FIELDS = ["split", "world_ordinal", "world_uid", "canonical_pair_uid",
              "seller_uid_left", "seller_uid_right"]


def read_rows(path: Path, split: str) -> list[dict[str, str]]:
    if split not in ("train", "development"):
        raise ValueError("Only train/development public rows are authorized")
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != ROW_FIELDS:
            raise ValueError("Row header mismatch")
        rows = list(reader)
    seen, owners = set(), {}
    for row in rows:
        left, right, world = row["seller_uid_left"], row["seller_uid_right"], row["world_uid"]
        key = row["canonical_pair_uid"]
        if row["split"] != split or not world or not left or left >= right or key != f"{left}||{right}" or key in seen:
            raise ValueError("Split/canonical pair/duplicate mismatch")
        seen.add(key)
        for uid in (left, right):
            if uid in owners and owners[uid] != world:
                raise ValueError("Account crosses worlds")
            owners[uid] = world
    if not rows:
        raise ValueError("Empty public split")
    return rows


def read_labels(path: Path, rows: list[dict[str, str]]) -> np.ndarray:
    values = []
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != ["canonical_pair_uid", "world_uid", "label"]:
            raise ValueError("Label header mismatch")
        for row, label in zip_longest(rows, reader):
            if row is None or label is None:
                raise ValueError("Label row count mismatch")
            if any(row[k] != label[k] for k in ("canonical_pair_uid", "world_uid")) or label["label"] not in ("0", "1"):
                raise ValueError("Label alignment/value mismatch")
            values.append(int(label["label"]))
    return np.asarray(values, dtype=np.int8)


def disjoint_splits(first: list[dict[str, str]], second: list[dict[str, str]]) -> None:
    for fields in (("world_uid",), ("seller_uid_left", "seller_uid_right")):
        a = {row[field] for row in first for field in fields}
        b = {row[field] for row in second for field in fields}
        if a & b:
            raise ValueError("Train/development world or account overlap")


def no_strong_mask(identity: np.ndarray, names: list[str]) -> np.ndarray:
    if identity.ndim != 2 or identity.shape[1] != len(names) or not np.isfinite(identity).all():
        raise ValueError("Invalid public identity matrix")
    direct = identity[:, names.index("verified_direct_token_count_log1p")]
    rotation = identity[:, names.index("strong_rotation_path_count_log1p")]
    return ~((direct > 0) | (rotation > 0))


def shared_imputation(train: np.ndarray, development: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if train.ndim != 2 or development.ndim != 2 or train.shape[1] != development.shape[1]:
        raise ValueError("Imputation shape mismatch")
    if np.isinf(train).any() or np.isinf(development).any() or np.all(np.isnan(train), axis=0).any():
        raise ValueError("Invalid imputation inputs")
    medians = np.asarray(np.nanmedian(train, axis=0), dtype="<f8")
    return (np.ascontiguousarray(np.where(np.isnan(train), medians, train)),
            np.ascontiguousarray(np.where(np.isnan(development), medians, development)), medians)


def fit_model(matrix: np.ndarray, labels: np.ndarray, selected: np.ndarray,
              parameters: dict[str, Any]) -> lgb.Booster:
    if (matrix.ndim != 2 or labels.shape != (len(matrix),) or selected.shape != labels.shape
            or selected.dtype != np.bool_ or not np.isfinite(matrix).all()
            or not np.isin(labels, [0, 1]).all() or len(np.unique(labels[selected])) != 2):
        raise ValueError("Invalid supervised fit boundary")
    model = lgb.LGBMClassifier(**parameters)
    model.fit(matrix[selected], labels[selected])
    return model.booster_


def predictions(model: lgb.Booster, matrix: np.ndarray) -> np.ndarray:
    raw = model.predict(matrix, raw_score=True, num_threads=1)
    probability = model.predict(matrix, raw_score=False, num_threads=1)
    out = np.asarray(np.column_stack([raw, probability]), dtype="<f8")
    if out.shape != (len(matrix), 2) or not np.isfinite(out).all() or np.any((out[:, 1] < 0) | (out[:, 1] > 1)):
        raise ValueError("Invalid model predictions")
    return out


def probability_loss(labels: np.ndarray, probability: np.ndarray) -> float:
    p = np.clip(probability, 1e-15, 1 - 1e-15)
    return float(-np.mean(labels * np.log(p) + (1 - labels) * np.log1p(-p)))


def confusion(labels: np.ndarray, positive: np.ndarray) -> dict[str, Any]:
    tp, fp = int(np.sum((labels == 1) & positive)), int(np.sum((labels == 0) & positive))
    fn, tn = int(np.sum((labels == 1) & ~positive)), int(np.sum((labels == 0) & ~positive))
    def ratio(a: float, b: float) -> float:
        return a / b if b else 0.0
    precision, recall = ratio(tp, tp + fp), ratio(tp, tp + fn)
    specificity = ratio(tn, tn + fp)
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "precision": precision, "recall": recall,
            "f1": ratio(2 * tp, 2 * tp + fp + fn), "specificity": specificity,
            "balanced_accuracy": (recall + specificity) / 2,
            "mcc": ratio(tp * tn - fp * fn, math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))}


def empirical_operating_point(labels: np.ndarray, raw: np.ndarray, maximum_fpr: float) -> dict[str, Any]:
    if len(np.unique(labels)) != 2 or not 0 <= maximum_fpr < 1:
        raise ValueError("Operating point needs two classes and a valid FPR")
    fpr, tpr, thresholds = roc_curve(labels, raw, drop_intermediate=False)
    feasible = np.flatnonzero(fpr <= maximum_fpr)
    chosen = feasible[int(np.argmax(tpr[feasible]))]
    threshold = thresholds[chosen]
    point = confusion(labels, raw >= threshold)
    return {**point, "fpr": float(fpr[chosen]),
            "threshold": float(threshold) if np.isfinite(threshold) else None,
            "always_negative": bool(np.isinf(threshold)), "evaluation_only": True}


def query_ranking(labels_in_rank_order: np.ndarray) -> dict[str, float]:
    y = np.asarray(labels_in_rank_order, dtype=float)
    positives = float(y.sum())
    if positives == 0:
        raise ValueError("Query has no relevant candidate")
    ranks = np.arange(1, len(y) + 1)
    out = {"map": float(np.sum(np.cumsum(y) / ranks * y) / positives),
           "mrr": float(1 / (np.flatnonzero(y)[0] + 1))}
    for k in (1, 3, 5, 10):
        out[f"recall_at_{k}"] = float(y[:k].sum() / positives)
        ideal = np.sum(1 / np.log2(np.arange(1, min(k, int(positives)) + 1) + 1))
        out[f"ndcg_at_{k}"] = float(np.sum(y[:k] / np.log2(ranks[:k] + 1)) / ideal)
    return out


def retrieval(rows: list[dict[str, str]], labels: np.ndarray, raw: np.ndarray,
              selected: np.ndarray) -> dict[str, Any]:
    queries: dict[tuple[str, str], list[tuple[float, bytes, int]]] = defaultdict(list)
    for index in np.flatnonzero(selected):
        row = rows[index]
        left, right, world = row["seller_uid_left"], row["seller_uid_right"], row["world_uid"]
        queries[(world, left)].append((-float(raw[index]), right.encode("utf-8"), int(labels[index])))
        queries[(world, right)].append((-float(raw[index]), left.encode("utf-8"), int(labels[index])))
    by_world: dict[str, list[dict[str, float]]] = defaultdict(list)
    for (world, _), candidates in queries.items():
        ordered = np.asarray([value[2] for value in sorted(candidates)])
        if ordered.sum():
            by_world[world].append(query_ranking(ordered))
    values = [point for world in sorted(by_world) for point in by_world[world]]
    if not values:
        return {"eligible_queries": 0, "queries_with_candidates": len(queries), "eligible_worlds": 0}
    return {"eligible_queries": len(values), "queries_with_candidates": len(queries),
            "eligible_worlds": len(by_world),
            "query_equal": {key: float(np.mean([p[key] for p in values])) for key in values[0]},
            "world_equal": {key: float(np.mean([np.mean([p[key] for p in by_world[w]])
                                               for w in sorted(by_world)])) for key in values[0]}}


def paired_summary(delta: np.ndarray, indices: np.ndarray) -> dict[str, Any]:
    delta, indices = np.asarray(delta), np.asarray(indices)
    if (delta.ndim != 1 or not len(delta) or not np.isfinite(delta).all() or indices.ndim != 2
            or indices.shape[1] != len(delta) or indices.dtype.kind not in "iu"
            or not indices.size or indices.min() < 0 or indices.max() >= len(delta)):
        raise ValueError("Invalid paired world bootstrap")
    draws = delta[indices].mean(axis=1)
    return {"difference": float(delta.mean()),
            "percentile_95_interval": np.quantile(draws, [0.025, 0.975], method="linear").tolist(),
            "worlds": len(delta), "replicates": len(indices),
            "draw_indices_sha256": hashlib.sha256(np.asarray(indices, dtype="<i8").tobytes()).hexdigest()}


def save_array(path: Path, value: np.ndarray) -> dict[str, Any]:
    with path.open("xb") as handle:
        np.save(handle, value, allow_pickle=False)
    if not np.array_equal(np.load(path, allow_pickle=False), value):
        raise ValueError("Array disk replay mismatch")
    return audit.file_record(path)


def run(policy: dict[str, Any], output: Path, reads: dict[str, int]) -> dict[str, Any]:
    if lgb.__version__ != "4.6.0":
        raise ValueError("This comparison pins LightGBM 4.6.0")
    records: list[dict[str, Any]] = []
    for relative, digest in policy["pins"].items():
        audit.checked_file(ROOT, {"path": relative, "sha256": digest}, relative, records)
    prior = audit.read_json(ROOT / policy["prior"])
    registered = {r["path"]: r for r in prior["inputs"]}
    def verified(relative: str) -> Path:
        return audit.checked_file(ROOT, registered[relative], relative, records)
    execution = audit.read_json(ROOT / "schema/step28_train_development_execution_policy.json")
    original = audit.read_json(ROOT / "schema/step28_model_experiment_policy.json")
    summary = audit.read_json(audit.TRAINING / "training_summary.json")
    if summary["m3_selected_grid"]["m3_joint"]["value"] != policy["selected_grid"]:
        raise ValueError("Training-only selected grid changed")
    expected_parameters = dict(original["m3"]["fixed_parameters"])
    expected_parameters.update(dict(zip(original["m3"]["grid_parameter_names"], policy["selected_grid"])))
    if policy["parameters"] != expected_parameters or policy["allowed_splits"] != ["train", "development"]:
        raise ValueError("Fixed parameters or allowed split boundary changed")
    rows, matrices, masks = {}, {}, {}
    for split in ("train", "development"):
        base_root = audit.PROJECTION / "base_v1" / split
        id_root = audit.PROJECTION / "identity_v1" / split
        row_path = verified((base_root / "row_keys.csv").relative_to(ROOT).as_posix())
        id_path = verified((id_root / "row_keys.csv").relative_to(ROOT).as_posix())
        if audit.file_record(row_path)["sha256"] != audit.file_record(id_path)["sha256"]:
            raise ValueError("Base/identity row order mismatch")
        rows[split] = read_rows(row_path, split)
        base = np.load(verified((base_root / "base24.npy").relative_to(ROOT).as_posix()), allow_pickle=False)
        identity = np.load(verified((id_root / "identity33.npy").relative_to(ROOT).as_posix()), allow_pickle=False)
        if (base.shape != (len(rows[split]), 24) or identity.shape != (len(base), 33)
                or base.dtype.str != "<f8" or identity.dtype.str != "<f8" or np.isinf(base).any()):
            raise ValueError("Matrix size/dtype/value mismatch")
        if len(base) != 189000:
            raise ValueError("Frozen split row count mismatch")
        matrices[split] = np.ascontiguousarray(np.column_stack([base, identity]))
        masks[split] = no_strong_mask(identity, original["feature_contract"]["identity33"])
    disjoint_splits(rows["train"], rows["development"])

    def supervision(split: str) -> np.ndarray:
        if split not in ("train", "development"):
            raise ValueError("Unauthorized supervision")
        spec = execution["authorized_private_inputs"][f"{split}_labels"]
        path = audit.checked_file(ROOT / execution["private_supervision_root"], spec,
                                  f"{split}/pair_labels.csv", records)
        reads[f"{split}_labels"] += 1
        labels = read_labels(path, rows[split])
        if hashlib.sha256(labels.tobytes()).hexdigest() != summary[f"{split}_label_vector_sha256"]:
            raise ValueError("Original label vector hash differs")
        return labels

    train_y = supervision("train")
    train, dev, medians = shared_imputation(matrices["train"], matrices["development"])
    old_medians = np.load(verified((audit.TRAINING / "models/m3_joint/medians.npy").relative_to(ROOT).as_posix()), allow_pickle=False)
    if not np.array_equal(medians, old_medians):
        raise ValueError("Shared all-training medians differ from old M3")
    artifacts = [save_array(output / "medians.npy", medians)]
    predicted, fit_records = {}, {}
    for arm in policy["arms"]:
        selected = np.ones(len(train_y), dtype=bool) if arm == "all" else masks["train"]
        print(json.dumps({"stage": "fitting", "arm": arm, "rows": int(selected.sum()),
                          "positives": int(train_y[selected].sum())}), flush=True)
        model = fit_model(train, train_y, selected, policy["parameters"])
        fitted = predictions(model, train[selected])
        baseline = np.full(int(selected.sum()), train_y[selected].mean())
        before, after = probability_loss(train_y[selected], baseline), probability_loss(train_y[selected], fitted[:, 1])
        if model.num_trees() != policy["parameters"]["n_estimators"] or not after < before:
            raise ValueError("Expected tree updates/training loss improvement missing")
        model_path = output / f"{arm}.txt"
        with model_path.open("x", encoding="utf-8", newline="\n") as handle:
            handle.write(model.model_to_string())
        replay = lgb.Booster(model_file=str(model_path))
        scores = predictions(model, dev)
        if not np.array_equal(predictions(replay, dev), scores):
            raise ValueError("Full development model disk replay differs")
        order = np.asarray([17, 2, 120, 0, len(dev) - 1])
        if not np.array_equal(predictions(replay, dev[order]), scores[order]):
            raise ValueError("Reordered prediction differs")
        if arm == "all":
            saved = np.load(verified((audit.TRAINING / "predictions/development/m3_joint.npy").relative_to(ROOT).as_posix()), allow_pickle=False)
            if not np.array_equal(scores[:, 1], saved):
                raise ValueError("Refitted all arm does not exactly reproduce original M3-joint")
        predicted[arm] = scores
        fit_records[arm] = {"rows": int(selected.sum()), "positives": int(train_y[selected].sum()),
                            "negatives": int((1 - train_y[selected]).sum()), "trees": model.num_trees(),
                            "initial_constant_log_loss": before, "fitted_log_loss": after,
                            "disk_replay_max_abs_error": 0.0, "five_reordered_rows_exact": True,
                            "selected_rows_sha256": hashlib.sha256(np.flatnonzero(selected).astype("<i8").tobytes()).hexdigest()}
        artifacts.extend([audit.file_record(model_path), save_array(output / f"{arm}.npy", scores)])
        print(json.dumps({"stage": "fit_saved", "arm": arm, "trees": model.num_trees(),
                          "development_labels_read": reads["development_labels"]}), flush=True)

    # Both complete models and all development predictions are fixed on disk first.
    dev_y = supervision("development")
    worlds = np.asarray([row["world_uid"] for row in rows["development"]])
    world_order = sorted(set(worlds))
    groups = [np.flatnonzero(worlds == world) for world in world_order]
    evaluation_masks = {"all": np.ones(len(dev_y), dtype=bool), "no_strong": masks["development"],
                        "zero_identity": np.all(matrices["development"][:, 24:] == 0, axis=1)}
    results, world_ap = {}, {}
    for arm, scores in predicted.items():
        result = {}
        for name, mask in evaluation_masks.items():
            y, raw, p = dev_y[mask], scores[mask, 0], scores[mask, 1]
            point = audit.summarize_subset(dev_y, scores[:, 0], groups, mask)
            reference = prior["models"]["m3_joint"][name]
            for key in ("rows", "positives", "negatives", "eligible_worlds", "total_worlds"):
                if point[key] != reference[key]:
                    raise ValueError("Evaluation population differs from prior diagnostic")
            point["brier"] = float(np.mean((p - y) ** 2))
            point["log_loss"] = probability_loss(y, p)
            point["at_probability_half"] = confusion(y, p >= policy["descriptive_probability_threshold"])
            point["at_empirical_fpr_1pct"] = empirical_operating_point(y, raw, policy["operating_point"]["maximum_empirical_fpr"])
            if name in ("all", "no_strong"):
                point["retrieval"] = retrieval(rows["development"], dev_y, scores[:, 0], mask)
            result[name] = point
        world_ap[arm] = np.asarray([audit.curve_metrics(dev_y[index[masks["development"][index]]],
                                      scores[index[masks["development"][index]], 0])["ap"] for index in groups], dtype="<f8")
        results[arm] = result
    if len(world_order) != 500:
        raise ValueError("Expected 500 aligned development worlds")
    cfg = policy["bootstrap"]
    indices = np.random.Generator(np.random.PCG64(cfg["seed"])).integers(
        0, len(world_order), size=(cfg["replicates"], len(world_order)), dtype=np.int64)
    comparison = paired_summary(world_ap["no_strong"] - world_ap["all"], indices)
    recall_delta = (results["no_strong"]["no_strong"]["at_empirical_fpr_1pct"]["recall"]
                    - results["all"]["no_strong"]["at_empirical_fpr_1pct"]["recall"])
    comparison["recall_at_empirical_fpr_1pct_difference"] = recall_delta
    comparison["further_investigation_supported"] = comparison["percentile_95_interval"][0] > 0 and recall_delta >= 0
    artifacts.append(save_array(output / "world_ap.npy", np.column_stack([world_ap[a] for a in policy["arms"]])))
    return {"status": "EXPLORATORY_SPECIALIST_COMPARISON_COMPLETE", "policy": audit.file_record(POLICY),
            "script": audit.file_record(Path(__file__).resolve()), "inputs": records, "parameters": policy["parameters"],
            "environment": {"python": platform.python_version(), "numpy": np.__version__,
                            "scikit_learn": sklearn.__version__, "lightgbm": lgb.__version__, "threads": 1},
            "fits": fit_records, "shared_imputation_uses_all_training_public_rows": True,
            "all_arm_reproduces_original_development_probabilities_exactly": True,
            "prediction_columns": ["raw_tree_margin_for_ranking", "probability_for_proper_losses_and_fixed_half_threshold"],
            "results": results, "primary_comparison": comparison,
            "world_order_sha256": hashlib.sha256("\n".join(world_order).encode("utf-8")).hexdigest(),
            "world_ap_columns": policy["arms"], "artifacts": artifacts, "truth_reads": dict(reads),
            "limits": policy["limits"] + [
                "Bootstrap conditions on these fixed fitted models and reused development worlds; it does not include training, selection or generator uncertainty.",
                "FPR operating points are descriptive label-consuming evaluation curves with grouped ties, no interpolation and no deployment threshold guarantee.",
                "Full-population specialist metrics evaluate extrapolation outside its selected fit population; no score stitching or new deployment is implemented.",
                "Train fitting loss and exact reload verify this actual tree fitting path, not all algorithms or the complete upstream synthetic mechanism."]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["run"])
    parser.parse_args()
    policy = audit.read_json(POLICY)
    output = ROOT / policy["output"]
    if output.exists():
        raise FileExistsError("Do not overwrite or repeat this frozen comparison")
    output.mkdir(parents=True)
    reads = {"train_labels": 0, "development_labels": 0, "qrels": 0, "audit_a": 0, "audit_b": 0}
    try:
        result = run(policy, output, reads)
        encoded = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
        with (output / "result.json").open("x", encoding="utf-8", newline="\n") as handle:
            handle.write(encoded)
        print(json.dumps({"status": result["status"], "primary": result["primary_comparison"]}), flush=True)
    except Exception as error:
        with (output / "failure.json").open("x", encoding="utf-8", newline="\n") as handle:
            json.dump({"status": "FAILED_NO_COMPARISON_CONCLUSION", "error_type": type(error).__name__,
                       "message": str(error), "truth_reads": reads,
                       "script": audit.file_record(Path(__file__).resolve()), "policy": audit.file_record(POLICY)},
                      handle, ensure_ascii=False, indent=2, allow_nan=False)
        raise


if __name__ == "__main__":
    main()
