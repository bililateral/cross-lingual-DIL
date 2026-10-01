"""Independently audit saved pooling results on Linux; no formal-label/model reads."""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import math
import platform
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SEEDS = ("s0", "s1", "s2")
RUNS = tuple(f"{s}_{kind}" for s in SEEDS for kind in ("d", "weighted"))
COLUMNS = (
    "average_precision", "trapezoidal_pr_auc", "roc_auc", "recall_at_fpr_1pct",
    "brier", "log_loss", "precision", "recall", "f1", "specificity",
    "balanced_accuracy", "mcc", "map", "mrr", "recall_at_1", "recall_at_3",
    "recall_at_5", "recall_at_10", "ndcg_at_1", "ndcg_at_3", "ndcg_at_5", "ndcg_at_10",
)
GUARDS = {"recall_at_5": 1, "average_precision": 1, "roc_auc": 1, "brier": -1, "log_loss": -1}


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class Audit:
    def __init__(self) -> None:
        self.numbers = 0
        self.max_difference = 0.0
        self.blind_count_rows = 0
        self.files: dict[str, dict] = {}

    def numbers_equal(self, actual: Any, expected: Any, name: str, atol: float = 1e-12) -> None:
        a, b = np.asarray(actual, dtype=np.float64), np.asarray(expected, dtype=np.float64)
        if a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
            raise ValueError(f"Invalid numerical shape/finiteness: {name}")
        difference = float(np.max(np.abs(a - b))) if a.size else 0.0
        if difference > atol:
            raise ValueError(f"Numerical mismatch: {name}: {difference}")
        self.numbers += a.size
        self.max_difference = max(self.max_difference, difference)

    def mapping(self, actual: dict, expected: dict, name: str) -> None:
        for key, value in expected.items():
            if isinstance(value, dict):
                self.mapping(actual[key], value, name + "/" + key)
            else:
                self.numbers_equal(actual[key], value, name + "/" + key)

    def file(self, base: Path, record: dict) -> Path:
        path = (base / record["path"]).resolve()
        if not path.is_relative_to(ROOT) or path.suffix == ".pt":
            raise ValueError("Only project small evidence may be read")
        if path.stat().st_size != record["bytes"] or digest(path) != record["sha256"]:
            raise ValueError(f"File identity differs: {path}")
        self.files[path.relative_to(ROOT).as_posix()] = {
            "bytes": record["bytes"], "sha256": record["sha256"]}
        return path

    def array(self, base: Path, record: dict, rows: int, columns: int, dtype: Any) -> np.ndarray:
        path = self.file(base, record)
        if path.suffix != ".npy":
            raise ValueError("Expected an array of saved scores or metrics")
        value = np.load(path, allow_pickle=False)
        if value.shape != (rows, columns) or value.dtype != dtype or not np.isfinite(value).all():
            raise ValueError(f"Invalid saved matrix: {path}")
        return value


def interval(values: np.ndarray) -> list:
    ordered = np.sort(values, axis=0)
    positions = np.asarray([.025, .975]) * (len(ordered) - 1)
    lo, hi = np.floor(positions).astype(int), np.ceil(positions).astype(int)
    fraction = (positions - lo).reshape((-1,) + (1,) * (ordered.ndim - 1))
    return (ordered[lo] * (1 - fraction) + ordered[hi] * fraction).tolist()


def confusion(values: np.ndarray) -> dict:
    tp, fp, fn, tn = (float(x) for x in np.asarray(values).sum(axis=0))
    precision = tp / (tp + fp) if tp + fp else 0.
    recall, specificity = tp / (tp + fn), tn / (fp + tn)
    denominator = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "precision": precision,
            "recall": recall, "specificity": specificity, "fpr": 1 - specificity,
            "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.,
            "balanced_accuracy": (recall + specificity) / 2,
            "mcc": (tp * tn - fp * fn) / denominator if denominator else 0.}


def ratios(values: np.ndarray) -> dict[str, np.ndarray]:
    out = {}
    for name, a, b in (("fpr", 1, 3), ("recall", 0, 2), ("precision", 0, 1)):
        denom = values[:, a] + values[:, b]
        out[name] = np.divide(values[:, a], denom, out=np.zeros(len(values)), where=denom > 0)
    return out


def counts_array(raw: Any) -> np.ndarray:
    value = np.asarray([[r[k] for k in ("tp", "fp", "fn", "tn")] for r in raw]
                       if isinstance(raw[0], dict) else raw)
    if (value.ndim != 2 or value.shape[1] != 4 or value.dtype.kind not in "iu"
            or np.any(value < 0) or not np.all(value[:, 0] + value[:, 2] == 20)
            or not np.all(value[:, 1] + value[:, 3] == 358)):
        raise ValueError("Confusion counts do not describe complete 378-pair groups")
    return value


def run(job: Path, out: Path) -> dict:
    audit = Audit()
    inventory = read(out.parent / "sync_inventory.json")
    for rec in inventory["files"] + inventory["frozen_sources_fresh_verified"]:
        audit.file(ROOT, rec)
    evaluation = read(job / "evaluation/evaluation.json")
    collected = read(job / "evaluation/collected.json")
    manifest = read(job / "run/manifest.json")
    complete = read(job / "completion.json")
    policy = read(ROOT / "schema/step28_alias_pooling_policy.json")
    if (list(COLUMNS) != evaluation["columns"] or evaluation["policy"] != policy
            or manifest["policy"] != policy or complete["formal_updates"] != 4320
            or complete["label_parses"] != {"train": 1, "development": 1, "heldout": 0, "owners": 0}
            or (job / "exit_status.txt").read_text().strip() != "0"):
        raise ValueError("Frozen scope or completion differs")
    if read(job / "run/completion.json")["manifest_sha256"] != digest(job / "run/manifest.json"):
        raise ValueError("Training completion binding differs")
    preparse = read(job / "evaluation/preparse_verification.json")
    if len(preparse["files"]) != 12 or preparse["sources"] != manifest["source_files"]:
        raise ValueError("Pre-valid twelve-model/source verification missing")
    audit.file(ROOT, evaluation["provenance"]["training_manifest"])
    audit.file(job / "evaluation", evaluation["provenance"]["preparse_verification"])
    for rec in policy["historical_d"]["files"]:
        audit.file(ROOT, rec)
    domains = np.asarray(evaluation["domains"])
    rows = {d: np.flatnonzero(domains == d) for d in "ABC"}
    if any(len(x) != 20 for x in rows.values()) or len(set(evaluation["group_ids"])) != 60:
        raise ValueError("Development group identities differ")
    # Independent frequency weights, rather than production's indexed mean loop.
    draws = np.random.default_rng(20260925).integers(0, 20, size=(5000, 3, 20))
    weights = np.zeros((5000, 60), dtype=np.int64)
    for index, d in enumerate("ABC"):
        np.add.at(weights, (np.arange(5000)[:, None], rows[d][draws[:, index]]), 1)
    matrices, auto_counts, trajectories, training, metric_rows = {}, {}, [], {}, []
    old_root = ROOT / policy["historical_d"]["run_root"] / "split_rank"
    for run_id in RUNS:
        arm_root = old_root if run_id == "s0_d" else job / "run" / run_id
        arm = read(arm_root / "manifest.json")
        if run_id != "s0_d":
            audit.file(job / "run", manifest["runs"][run_id]["manifest"])
        if arm["updates"] != 864 or arm["trainable_parameter_count"] != arm["parameter_count"]:
            raise ValueError("Training budget or parameter trainability differs")
        loss_key = "losses_by_epoch" if run_id == "s0_d" else "mean_losses_by_epoch"
        observations = ([{str(s["start"] + 1): s["first_update_modules"]} for s in arm["training"]]
                        if run_id == "s0_d" else [s["observations"] for s in arm["training"]])
        losses = {key: sum((s[loss_key][key] for s in arm["training"]), [])
                  for key in ("bce", "rank", "total")}
        audit.numbers_equal(np.asarray(losses["bce"]) + losses["rank"], losses["total"],
                            run_id + "/loss", atol=1e-6)
        for segment_observations in observations:
            for step, observation in segment_observations.items():
                for module, evidence in observation.items():
                    if not evidence["finite_nonzero_gradient"] or not evidence["parameters_changed"]:
                        raise ValueError(f"Missing actual training: {run_id}/{step}/{module}")
                if run_id.endswith("weighted"):
                    norms = observation["aggregation"]["parameter_gradient_norms_before_clip"]
                    if step == "1":
                        audit.numbers_equal([norms["hidden.weight"], norms["hidden.bias"]], [0, 0], run_id)
                    elif any(v is None or v <= 0 for v in norms.values()):
                        raise ValueError("Hidden/output aggregation task gradients absent after first update")
        training[run_id] = {"losses_by_epoch": losses, "updates": arm["updates"],
                            "resources": arm["resources"], "formal_training_seconds": arm["formal_training_seconds"],
                            "parameter_count": arm["parameter_count"], "observations": observations,
                            "common_initial_state": arm["preflight"].get("common_state_sha256", arm["preflight"].get("initial_model_state_sha256")),
                            "schedule_sha256": arm["group_schedule_sha256"], "dropout_stream": arm["dropout_stream"]}
        for epoch in ("3", "6"):
            point = evaluation["runs"][run_id]["points"][epoch]
            original = collected["runs"][run_id]["points"][epoch]
            if point != original:
                raise ValueError("Final report changed collected metrics/counts")
            matrix = audit.array(job / "evaluation", point["file"], 60, 22, np.float64)
            matrices[run_id, epoch] = matrix
            for scope, index in {"mean": np.arange(60), **rows}.items():
                expected = dict(zip(COLUMNS, matrix[index].mean(0).tolist()))
                audit.mapping(point["mean"] if scope == "mean" else point["by_domain"][scope], expected, run_id)
                metric_rows.extend({"run_id": run_id, "epoch": epoch, "scope": scope, "metric": k, "value": v}
                                   for k, v in expected.items())
            counts = counts_array(point["counts_at_logit_zero"])
            for i, count in enumerate(counts):
                rates = confusion(count[None, :])
                for name in ("precision", "recall", "f1", "specificity", "balanced_accuracy", "mcc"):
                    audit.numbers_equal(matrix[i, COLUMNS.index(name)], rates[name], run_id + "/group confusion")
            for scope, index in {"pooled": np.arange(60), **rows}.items():
                expected = {k: v for k, v in confusion(counts[index]).items()
                            if k in ("tp", "fp", "fn", "tn", "fpr", "recall", "precision", "f1")}
                audit.mapping(point["fixed_classification"]["pooled"] if scope == "pooled"
                              else point["fixed_classification"]["by_domain"][scope], expected, run_id)
            apoint = arm["points"][epoch]
            if not apoint["full_model_and_adam_reloaded"] or not apoint["model"]["actual_reload_verified"]:
                raise ValueError("Missing actual checkpoint replay")
            score = audit.array(arm_root, apoint["scores"]["development"], 60, 378, np.float32)
            audit.numbers_equal((score >= 0).sum(1), counts[:, 0] + counts[:, 1], run_id + "/blind positives")
            audit.blind_count_rows += 60
            for role, n in (("fit", 144), ("calibration", 36)):
                record = apoint["train_metrics"][role]
                train_matrix = audit.array(arm_root, record["file"], n, 22, np.float64)
                train_scores = audit.array(arm_root, apoint["scores"][role], n, 378, np.float32)
                train_counts = counts_array(record["counts"])
                audit.numbers_equal((train_scores >= 0).sum(1), train_counts[:, 0] + train_counts[:, 1], run_id)
                audit.blind_count_rows += n
                trajectories.append({"run_id": run_id, "epoch": epoch, "split": role,
                                     **dict(zip(COLUMNS, train_matrix.mean(0).tolist()))})
            trajectories.append({"run_id": run_id, "epoch": epoch, "split": "valid", **point["mean"]})
        automatic = evaluation["runs"][run_id]["automatic_classification"]
        counts = counts_array(automatic["counts_by_group"])
        auto_counts[run_id] = counts
        calibration = read(audit.file(arm_root, arm["calibration"]))
        audit.numbers_equal(automatic["threshold"], calibration["threshold"], run_id)
        score = audit.array(arm_root, arm["points"]["6"]["scores"]["development"], 60, 378, np.float32)
        audit.numbers_equal((score.astype(np.float64) >= automatic["threshold"]).sum(1),
                            counts[:, 0] + counts[:, 1], run_id + "/calibrated positives")
        audit.blind_count_rows += 60
        for scope, index in {"pooled": np.arange(60), **rows}.items():
            actual = automatic["pooled"] if scope == "pooled" else automatic["by_domain"][scope]
            expected = {k: v for k, v in confusion(counts[index]).items()
                        if k in ("tp", "fp", "fn", "tn", "fpr", "recall", "precision", "f1")}
            expected["conditional_95pct_intervals"] = {
                name: interval(value) for name, value in ratios(weights[:, index] @ counts[index]).items()}
            audit.mapping(actual, expected, run_id + "/automatic")
    for seed in SEEDS:
        a, b = training[seed + "_d"], training[seed + "_weighted"]
        if any(a[k] != b[k] for k in ("common_initial_state", "schedule_sha256", "dropout_stream")):
            raise ValueError("Paired common initialization or random schedule differs")
    old_collected = read(ROOT / policy["historical_d"]["evaluation_root"] / "collected.json")
    if old_collected["group_ids"] != evaluation["group_ids"] or old_collected["domains"] != evaluation["domains"]:
        raise ValueError("Historical D group alignment differs")
    for epoch in ("3", "6"):
        old = old_collected["arms"]["split_rank"]["points"][epoch]
        old_matrix = audit.array(ROOT / policy["historical_d"]["evaluation_root"], old["file"], 60, 22, np.float64)
        audit.numbers_equal(matrices["s0_d", epoch], old_matrix, "historical D")
    deltas = np.stack([matrices[s + "_weighted", "6"] - matrices[s + "_d", "6"] for s in SEEDS])
    mean_delta = deltas.mean(0)
    lower, upper = np.asarray(interval(weights @ mean_delta / 60))
    paired = {}
    for column, name in enumerate(COLUMNS):
        expected = {"mean": float(mean_delta[:, column].mean()),
                    "per_seed": deltas[:, :, column].mean(1).tolist(),
                    "by_domain": {d: float(mean_delta[index, column].mean()) for d, index in rows.items()},
                    "conditional_95pct_interval": [float(lower[column]), float(upper[column])]}
        audit.mapping(evaluation["paired_primary"]["metrics"][name], expected, "paired primary")
        baseline = float(np.stack([matrices[s + "_d", "6"] for s in SEEDS])[:, :, column].mean())
        candidate = float(np.stack([matrices[s + "_weighted", "6"] for s in SEEDS])[:, :, column].mean())
        paired[name] = {"d_mean": baseline, "weighted_mean": candidate, **expected}
        for i, seed in enumerate(SEEDS):
            bootstrap = weights @ deltas[i, :, column] / 60
            audit.mapping(evaluation["per_seed_comparisons"][seed]["metrics"][name],
                          {"mean": float(deltas[i, :, column].mean()), "conditional_95pct_interval": interval(bootstrap)}, seed)
    m = paired["map"]
    checks = {"map_minimum_observed_gain": m["mean"] >= .01,
              "map_interval_above_zero": m["conditional_95pct_interval"][0] > 0,
              "map_improves_each_paired_seed": all(v > 0 for v in m["per_seed"])}
    for name, sign in GUARDS.items():
        checks[name + "_mean_non_degradation"] = sign * paired[name]["mean"] >= 0
        checks[name + "_fixed_s0_non_degradation"] = sign * paired[name]["per_seed"][0] >= 0
    if checks != evaluation["acceptance"]["checks"] or all(checks.values()) != evaluation["acceptance"]["passed"]:
        raise ValueError("Prospective acceptance differs")
    started = dt.datetime.fromisoformat((job / "started.txt").read_text().strip())
    finished = dt.datetime.fromisoformat((job / "finished.txt").read_text().strip())
    result = {"status": "PASS_SAVED_RESULT_AUDIT_NEGATIVE_EXPERIMENT", "verified_at": dt.datetime.now().astimezone().isoformat(),
              "started": started.isoformat(), "finished": finished.isoformat(), "wall_seconds": (finished - started).total_seconds(),
              "remaining_seconds": 0, "original_estimate_hours": [8, 10], "budget": complete["budget"],
              "checks": {"numerical_values": audit.numbers, "maximum_absolute_difference_including_loss_roundoff": audit.max_difference,
                         "blind_count_rows": audit.blind_count_rows, "verified_small_files": len(audit.files),
                         "formal_label_reads": 0, "formal_text_reads": 0, "model_loads": 0,
                         "retraining_updates": 0, "bootstrap_frequency_weighted_reproduction": True},
              "verified_files": audit.files, "paired": paired, "acceptance": evaluation["acceptance"],
              "training": training, "trajectories": trajectories,
              "scope": "Saved matrices/counts/scores only; no reconstruction of labels. Conditional group intervals, not training-seed population uncertainty. No new attention behavior inference or causal mechanism diagnosis."}
    out.mkdir(exist_ok=False)
    (out / "analysis.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for filename, fieldnames, records in (
        ("metrics.csv", ["run_id", "epoch", "scope", "metric", "value"], metric_rows),
        ("trajectory.csv", ["run_id", "epoch", "split", *COLUMNS], trajectories),
    ):
        with (out / filename).open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(records)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if platform.system() != "Linux":
        parser.error("Run only in the existing Linux py310 environment")
    result = run(args.job.resolve(), args.out.resolve())
    print(json.dumps({"status": result["status"], "checks": result["checks"], "map": result["paired"]["map"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
