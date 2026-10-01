"""Fit six frozen-score calibrators, then collect valid once; no test entry.

Formal execution is Linux-only and requires source-bound CPU/review/readiness
evidence. `finalize` operates only on a complete saved collection, without labels.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
from datetime import datetime, timezone
import itertools
import os
from pathlib import Path
import platform
import time
from typing import Any, Callable

import numpy as np

import step28_alias_calibration as method

data, metrics = method.data, method.metrics
POLICY = data.ROOT / "schema/step28_alias_calibration_policy.json"
COLLECTED = "ALL_TWELVE_CALIBRATION_VALID_MATRICES_SAVED"
SOURCE_NAMES = ("scripts/step28_alias_calibration.py", "scripts/step28_alias_calibration_run.py",
                "scripts/step28_alias_calibration_audit.py", "tests/test_step28_alias_calibration_contracts.py",
                "scripts/run_step28_alias_calibration_linux_20260928.sh",
                "schema/step28_alias_calibration_policy.json", "docs/SELLER_ALIAS_CALIBRATION.zh.md",
                "reports/documentation/20260928/calibration_plan/decision.json")


def contract() -> dict:
    p = data.read_json(POLICY)
    if (p["study"] != "seller_alias_frozen_ranking_probability_calibration"
            or p["runs"] != list(method.RUNS) or p["epoch"] != 6
            or p["calibration_groups"] != 36 or p["development_groups"] != 60
            or p["fit"]["initial"] != [1., 0.]
            or p["fit"]["bounds"] != [list(v) for v in method.BOUNDS]
            or p["fit"]["options"] != method.OPTIONS
            or p["fit"]["objective"] != "unweighted_unclipped_bernoulli_nll"
            or p["fit"]["optimizer"] != "L-BFGS-B" or p["fit"]["encoder_updates"] != 0
            or p["fit"]["trainable_scalars"] != 12 or p["fit"]["projected_gradient_limit"] != 1e-6
            or p["primary_comparison"] != list(method.COMPARISONS[0])
            or p["raw_reference_probability_protection"] is not True
            or p["performance_guards"] != 17 or p["minimum_map_gain"] != .01
            or p["fixed_test_seed"] != "s0" or p["bootstrap_replicates"] != 5000
            or p["bootstrap_seed"] != 20260927
            or p["label_parses"] != {"train": 1, "development": 1, "heldout": 0, "owners": 0}
            or p["model_loading"] is not False or p["formal_text_access"] is not False
            or p["runtime"] != {"cpu_threads": 1, "gpu": False, "maximum_seconds": 3600,
                                "maximum_output_bytes": 268435456}):
        raise ValueError("Calibration plan differs from the confirmed method and budget")
    return p


def sources() -> list[dict]:
    # The old file list defines the dependency snapshot, not an authority to execute it.
    p = contract()
    old = data.read_json(data.verify(data.ROOT / p["historical_job"] / "run/manifest.json",
                                    p["historical_records"]["run/manifest.json"]))
    for row in old["source_files"]:
        data.verify(data.ROOT / row["path"], row)
    names = sorted({row["path"] for row in old["source_files"]} | set(SOURCE_NAMES))
    return [data.record(data.ROOT / name, data.ROOT) for name in names]


def array(path: Path, record: dict, shape: tuple[int, ...], dtype: Any) -> np.ndarray:
    value = np.load(data.verify(path, record), allow_pickle=False)
    if value.shape != shape or value.dtype != dtype or not np.isfinite(value).all():
        raise ValueError("Wrong score/matrix shape, dtype or finite values: " + path.name)
    value.setflags(write=False)
    return value


def historical(p: dict) -> tuple[dict, dict, dict]:
    root = data.ROOT / p["historical_job"]
    original = {name: data.read_json(data.verify(root / name, record))
                for name, record in p["historical_records"].items()}
    manifest = original["run/manifest.json"]
    if (manifest["status"] != "COMPLETE_RANKING_7776_VALID_BLIND"
            or original["completion.json"]["status"] != "COMPLETE_RANKING_TRAIN_AND_VALID"
            or original["completion.json"]["label_parses"] != p["label_parses"]):
        raise ValueError("The pinned prior formal run is incomplete")
    part_record = manifest["partition"]
    if part_record["path"] != "partition.json":
        raise ValueError("Unexpected original partition path")
    partition = data.read_json(data.verify(root / "run/partition.json", part_record))
    all_ids: set[str] = set()
    for role, count in (("fit", 144), ("calibration", 36), ("development", 60)):
        rows = partition[role]
        ids = [row["group_uid"] for row in rows]
        if (len(ids) != count or len(set(ids)) != count or all_ids.intersection(ids)
                or any(sum(r["domain"] == d for r in rows) != count // 3 for d in "ABC")):
            raise ValueError("Original role/domain partition differs")
        all_ids.update(ids)
    saved = {}
    for run_id in method.RUNS:
        rec = manifest["runs"][run_id]["manifest"]
        if rec["path"] != f"{run_id}/manifest.json":
            raise ValueError("Prior model identity differs")
        arm = data.read_json(data.verify(root / "run" / rec["path"], rec))
        point = arm["points"]["6"]
        if (arm["run_id"] != run_id or arm["updates"] != 864 or point["epoch"] != 6
                or point["run_id"] != run_id or not point["full_model_and_adam_reloaded"]
                or arm["calibration_group_ids"] != [r["group_uid"] for r in partition["calibration"]]):
            raise ValueError("Prior E6 model/calibration role binding differs")
        scores, old_metrics = {}, {}
        for role, count in (("calibration", 36), ("development", 60)):
            record = point["scores"][role]
            if record["path"] != f"scores/epoch6_{role}.npy":
                raise ValueError("Score epoch/role differs")
            scores[role] = array(root / "run" / run_id / record["path"], record, (count, 378), np.float32)
            if role == "calibration":
                metric_record = point["train_metrics"][role]["file"]
                metric_path = root / "run" / run_id / metric_record["path"]
            else:
                metric_record = original["evaluation/collected.json"]["runs"][run_id]["points"]["6"]["file"]
                metric_path = root / "evaluation" / metric_record["path"]
            old_metrics[role] = array(metric_path, metric_record, (count, 22), np.float64)
        cal_rec = arm["calibration"]
        if cal_rec["path"] != "calibration.json":
            raise ValueError("Unexpected old threshold record")
        calibrated = data.read_json(data.verify(root / "run" / run_id / cal_rec["path"], cal_rec))
        if (calibrated["model_state_sha256"] != point["model_state_sha256"]
                or calibrated["score_sha256"] != point["scores"]["calibration"]["sha256"]):
            raise ValueError("Old threshold belongs to another model")
        saved[run_id] = {"scores": scores, "old_metrics": old_metrics,
                         "threshold": calibrated["threshold"],
                         "origin": {"manifest": rec, "model_state_sha256": point["model_state_sha256"],
                                    "model": point["model"], "scores": point["scores"]}}
    config = method.ranking.reference_config(manifest["policy"], "s0")
    root_data = data.ROOT / config["data_root"]
    if (data.sha256(root_data / "manifest.json") != config["data_manifest_sha256"]
            or data.sha256(root_data / "validation.json") != config["data_validation_sha256"]):
        raise ValueError("Pinned synthetic archive metadata differs")
    # No items.jsonl, model payload, heldout CSV or owners are opened.
    return saved, partition, config


def read_labels(path: Path, record: dict, all_group_ids: set[str], selected_ids: list[str],
                *, split: str) -> tuple[np.ndarray, dict]:
    """One CSV parse; keep selected calibration groups only for train fitting."""
    if split not in ("train", "development") or not set(selected_ids) <= all_group_ids:
        raise ValueError("Forbidden label role or groups")
    grouped: dict[str, list] = defaultdict(list)
    with data.verify(path, record).open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != ["group_uid", "seller_uid_left", "seller_uid_right", "label"]:
            raise ValueError("Unexpected binary supervision columns")
        for row in reader:
            if row["group_uid"] not in all_group_ids:
                raise ValueError("Unexpected label group")
            grouped[row["group_uid"]].append(row)
    if set(grouped) != all_group_ids or len(selected_ids) != len(set(selected_ids)):
        raise ValueError("Missing/repeated selected group")
    labels, identities = {}, {}
    sellers_seen: set[str] = set()
    for uid in sorted(grouped):
        rows = grouped[uid]
        sellers = sorted({r[key] for r in rows for key in ("seller_uid_left", "seller_uid_right")})
        expected = list(itertools.combinations(sellers, 2))
        found = {(r["seller_uid_left"], r["seller_uid_right"]): r["label"] for r in rows}
        if (len(sellers) != 28 or sellers_seen.intersection(sellers) or len(rows) != 378
                or set(found) != set(expected) or len(found) != len(rows)
                or any(r["label"] not in ("0", "1") for r in rows)):
            raise ValueError("Complete lexicographic seller-pair alignment differs")
        sellers_seen.update(sellers)
        if uid in selected_ids:
            y = np.array([int(found[pair]) for pair in expected], dtype=np.uint8)
            left, right = np.triu_indices(28, 1)
            adjacency = np.zeros((28, 28), dtype=np.uint8)
            adjacency[left, right] = adjacency[right, left] = y
            degree = adjacency.sum(1)
            if (int(y.sum()) != 20 or np.sum(degree == 1) != 16 or np.sum(degree == 2) != 12
                    or np.any((adjacency @ adjacency > 0) & ~np.eye(28, dtype=bool) & (adjacency == 0))):
                raise ValueError("Synthetic one/two-positive clique structure differs")
            labels[uid] = y
            identities[uid] = sellers
    return np.stack([labels[uid] for uid in selected_ids]), {
        "split": split, "selected_groups": selected_ids,
        "parsed_groups": len(grouped), "fitted_groups": len(selected_ids) if split == "train" else 0,
        "pair_order": "lexicographic combinations of 28 sorted seller identifiers",
        "selected_sellers": identities}


def parse_once(out: Path, config: dict, partition: dict, split: str) -> tuple[np.ndarray, dict]:
    if split not in ("train", "development"):
        raise ValueError("Only stage-authorized train/development allowed")
    record_path = out / f"{split}_access.json"
    # Exclusive creation prevents an implicit retry, including after failed parsing.
    with record_path.open("x", encoding="utf-8") as stream:
        stream.write(data.json_bytes({"split": split, "parse_attempts": 1,
                                      "time_utc": datetime.now(timezone.utc).isoformat(),
                                      "heldout": 0, "owners": 0}).decode())
    root = data.ROOT / config["data_root"]
    manifest = data.read_json(root / "manifest.json")
    role = "calibration" if split == "train" else "development"
    selected = [r["group_uid"] for r in partition[role]]
    allowed = {r["group_uid"] for r in partition[role]}
    if split == "train":
        allowed.update(r["group_uid"] for r in partition["fit"])
    name = f"{split}/supervision/pairs.csv"
    return read_labels(root / name, manifest["files"][name], allowed, selected, split=split)


def fit_all(out: Path, saved: dict, truth: np.ndarray, partition: dict,
            check: Callable[[], None] = lambda: None) -> dict:
    if set(saved) != set(method.RUNS) or truth.shape != (36, 378):
        raise ValueError("All six frozen models and 36 calibration groups required")
    result = {"status": "SIX_CALIBRATORS_SAVED_BEFORE_VALID", "runs": {}}
    for run_id in method.RUNS:
        check()
        directory = out / run_id
        directory.mkdir()
        entry = method.fit(saved[run_id]["scores"]["calibration"], truth, role="calibration", check=check)
        entry["origin"] = saved[run_id]["origin"]
        entry["group_ids"] = [r["group_uid"] for r in partition["calibration"]]
        data.write_json(directory / "fit.json", entry)
        if entry["status"] != "PASS_CALIBRATION_FIT":
            raise ValueError("Scalar optimizer failed; actual record preserved; no retry")
        restored = data.read_json(directory / "fit.json")
        records = {"fit": data.record(directory / "fit.json", out), "scores": {}, "order_checks": {}}
        for role in ("calibration", "development"):
            original = saved[run_id]["scores"][role]
            expected = method.transform(original, entry)
            replayed = method.transform(original, restored)
            if not np.array_equal(expected, replayed):
                raise ValueError("JSON-restored map changed full scores")
            records["order_checks"][role] = method.preserve_order(original, replayed)
            path = directory / f"{role}_scores.npy"
            np.save(path, replayed, allow_pickle=False)
            record = data.record(path, out)
            if not np.array_equal(array(path, record, replayed.shape, np.float64), expected):
                raise ValueError("Saved transformed scores changed")
            records["scores"][role] = record
        for variant, values in (("raw", saved[run_id]["scores"]["calibration"]),
                                ("calibrated", method.transform(saved[run_id]["scores"]["calibration"], restored))):
            matrix, counts = metrics.group_metrics(truth, values)
            path = directory / f"calibration_{variant}_metrics.npy"
            np.save(path, matrix, allow_pickle=False)
            records[f"calibration_{variant}"] = {"file": data.record(path, out), "counts": counts}
        result["runs"][run_id] = records
    data.write_json(out / "fitted.json", result)
    return result


def restored_scores(out: Path, saved: dict, partition: dict) -> dict:
    result = data.read_json(out / "fitted.json")
    if result["status"] != "SIX_CALIBRATORS_SAVED_BEFORE_VALID" or set(result["runs"]) != set(method.RUNS):
        raise ValueError("Incomplete calibrators; no valid parse")
    arrays = {}
    for run_id in method.RUNS:
        record = result["runs"][run_id]
        if record["fit"]["path"] != f"{run_id}/fit.json":
            raise ValueError("Map/model alignment differs")
        fitted = data.read_json(data.verify(out / record["fit"]["path"], record["fit"]))
        if (fitted["status"] != "PASS_CALIBRATION_FIT" or fitted["origin"] != saved[run_id]["origin"]
                or fitted["group_ids"] != [r["group_uid"] for r in partition["calibration"]]):
            raise ValueError("Restored map origin or fitting groups differ")
        for role, count in (("calibration", 36), ("development", 60)):
            score_record = record["scores"][role]
            if score_record["path"] != f"{run_id}/{role}_scores.npy":
                raise ValueError("Calibrated score role differs")
            values = array(out / score_record["path"], score_record, (count, 378), np.float64)
            original = saved[run_id]["scores"][role]
            if not np.array_equal(values, method.transform(original, fitted)):
                raise ValueError("Saved map and saved scores differ")
            method.preserve_order(original, values)
            if role == "development":
                arrays[run_id + "_calibrated"] = values
                arrays[run_id + "_raw"] = original
    return arrays


def collect(out: Path, truth: np.ndarray, arrays: dict, saved: dict, partition: dict) -> dict:
    expected = {f"{seed}_{v}" for seed in method.SEEDS for v in method.VARIANTS}
    if set(arrays) != expected or truth.shape != (60, 378):
        raise ValueError("Twelve score matrices and valid relevance required")
    destination = out / "evaluation"
    destination.mkdir()
    result = {"status": COLLECTED, "columns": list(metrics.COLUMNS), "points": {},
              "group_ids": [r["group_uid"] for r in partition["development"]],
              "domains": [r["domain"] for r in partition["development"]], "old_raw_metrics": {}}
    for point in sorted(expected):
        values = arrays[point]
        matrix, counts = metrics.group_metrics(truth, values)
        path = destination / f"{point}_metrics.npy"
        np.save(path, matrix, allow_pickle=False)
        result["points"][point] = {"file": data.record(path, destination), "counts_at_logit_zero": counts}
    # Diagnostic threshold decisions stay in the original score space, exactly.
    result["automatic_diagnostics"] = {}
    for run_id in method.RUNS:
        matrix_path = destination / f"{run_id}_previous_metrics.npy"
        np.save(matrix_path, saved[run_id]["old_metrics"]["development"], allow_pickle=False)
        result["old_raw_metrics"][run_id] = data.record(matrix_path, destination)
        fitted = data.read_json(out / run_id / "fit.json")
        threshold = float(saved[run_id]["threshold"])
        count = method.ranking.base.binary_counts(truth, saved[run_id]["scores"]["development"], threshold)
        result["automatic_diagnostics"][run_id] = {
            "original_threshold": threshold, "mapped_threshold": fitted["a"] * threshold + fitted["b"],
            "decision": "compare_original_logits_with_original_threshold_no_refit", "counts": count.tolist()}
    data.write_json(destination / "collected.json", result)
    return result


def finalize(out: Path) -> dict:
    destination = out / "evaluation"
    if (destination / "evaluation.json").exists():
        raise FileExistsError("Published evaluation cannot be overwritten")
    collected = data.read_json(destination / "collected.json")
    if collected["status"] != COLLECTED or collected["columns"] != list(metrics.COLUMNS):
        raise ValueError("Complete collection required before statistics")
    matrices = {}
    for point, record in collected["points"].items():
        if record["file"]["path"] != f"{point}_metrics.npy":
            raise ValueError("Saved metric point/path mismatch")
        matrices[point] = array(destination / record["file"]["path"], record["file"], (60, 22), np.float64)
    expected = {f"{seed}_{v}" for seed in method.SEEDS for v in method.VARIANTS}
    if set(matrices) != expected:
        raise ValueError("Missing calibrated or raw metric point")
    max_raw_difference = 0.
    ranking_columns = [metrics.COLUMNS.index(n) for n in (*metrics.CURVE_KEYS, *metrics.RETRIEVAL_KEYS)]
    for run_id in method.RUNS:
        record = collected["old_raw_metrics"][run_id]
        old = array(destination / record["path"], record, (60, 22), np.float64)
        difference = float(np.max(np.abs(old - matrices[run_id + "_raw"])))
        max_raw_difference = max(max_raw_difference, difference)
        if difference > 1e-12:
            raise ValueError("Raw model metrics differ from historical results; collection preserved")
        if not np.array_equal(matrices[run_id + "_raw"][:, ranking_columns],
                              matrices[run_id + "_calibrated"][:, ranking_columns]):
            raise ValueError("Ranking metric changed under positive affine map")
    result = {**collected, **method.summarize(matrices, collected["domains"]),
              "status": "CALIBRATION_VALID_EVALUATED_TEST_UNAUTHORIZED",
              "maximum_raw_metric_difference": max_raw_difference,
              "policy_sha256": data.sha256(POLICY), "evaluation_scope": method.EVALUATION,
              "rank_gain_is_retained_prior_gain": True}
    domains = collected["domains"]
    draws = np.random.default_rng(20260927).integers(0, 20, size=(5000, 3, 20))
    for point, matrix in matrices.items():
        record = result["points"][point]
        record["mean"] = dict(zip(metrics.COLUMNS, matrix.mean(0).tolist(), strict=True))
        record["by_domain"] = {d: dict(zip(metrics.COLUMNS, matrix[np.asarray(domains) == d].mean(0).tolist(), strict=True))
                               for d in "ABC"}
        record["fixed_classification"] = method.ranking.base.fixed_classification(record["counts_at_logit_zero"], domains)
    result["per_seed_comparisons"] = {}
    for candidate, reference in method.COMPARISONS:
        name = candidate + "_minus_" + reference
        result["per_seed_comparisons"][name] = {
            seed: method.ranking.base.metric_comparison(matrices[f"{seed}_{candidate}"],
                  matrices[f"{seed}_{reference}"], domains, draws) for seed in method.SEEDS}
    for record in result["automatic_diagnostics"].values():
        record["report"] = method.ranking.base.automatic_report(np.asarray(record["counts"]), domains, draws)
    data.write_json(destination / "evaluation.json", result)
    return result


class Budget:
    def __init__(self, out: Path, runtime: dict):
        self.out, self.runtime, self.started = out, runtime, time.monotonic()
        self.maximum_observed_bytes = 0

    def check(self) -> None:
        size = sum(path.stat().st_size for path in self.out.rglob("*") if path.is_file())
        self.maximum_observed_bytes = max(self.maximum_observed_bytes, size)
        if time.monotonic() - self.started > self.runtime["maximum_seconds"] or size > self.runtime["maximum_output_bytes"]:
            raise RuntimeError("Calibration stage resource budget exhausted")


def execute(out: Path, authorization_path: Path) -> dict:
    if platform.system() != "Linux" or len(os.sched_getaffinity(0)) != 1:
        raise ValueError("Existing Linux environment and one CPU affinity required")
    p, current_sources = contract(), sources()
    authorization = data.read_json(authorization_path)
    if (authorization.get("status") != "READY_CALIBRATION_FORMAL_SCOPE"
            or authorization.get("policy_sha256") != data.sha256(POLICY)
            or authorization.get("source_files") != current_sources
            or authorization.get("job") != out.resolve().relative_to(data.ROOT).as_posix()
            or authorization.get("label_parses") != p["label_parses"]
            or authorization.get("readiness_reported_to_user") is not True):
        raise ValueError("Matching formal readiness and scoped authorization required")
    for name, status in (("cpu_evidence", "PASS_CALIBRATION_HANDMADE_AUDIT"),
                         ("review_disposition", "PASS_CALIBRATION_IMPLEMENTATION_REVIEW")):
        rec = authorization[name]
        evidence = data.read_json(data.verify(data.ROOT / rec["path"], rec))
        if evidence["status"] != status or evidence["source_files"] != current_sources:
            raise ValueError("Missing actual current CPU or review evidence")
    saved, partition, config = historical(p)
    out.mkdir(parents=True, exist_ok=False)
    budget = Budget(out, p["runtime"])
    data.write_json(out / "preparation.json", {"source_files": current_sources, "policy": p,
                    "authorization": data.record(authorization_path, data.ROOT), "partition": partition,
                    "environment": {"python": platform.python_version(), "numpy": np.__version__,
                                    "cpu_affinity": sorted(os.sched_getaffinity(0))}})
    try:
        truth, identities = parse_once(out, config, partition, "train")
        data.write_json(out / "train_alignment.json", identities)
        fit_all(out, saved, truth, partition, budget.check)
        del truth
        arrays = restored_scores(out, saved, partition)
        budget.check()
        if sources() != current_sources:
            raise ValueError("Source changed before valid")
        data.write_json(out / "prevalid.json", {"status": "ALL_SIX_MAPS_AND_TWELVE_SCORE_ARRAYS_RELOADED",
                        "fitted": data.record(out / "fitted.json", out), "source_files": current_sources})
        valid_truth, valid_identities = parse_once(out, config, partition, "development")
        data.write_json(out / "valid_alignment.json", valid_identities)
        collect(out, valid_truth, arrays, saved, partition)
        del valid_truth
        result = finalize(out)
        budget.check()
        data.write_json(out / "completion.json", {"status": result["status"], "source_files": current_sources,
                        "label_parses": p["label_parses"], "acceptance": result["acceptance"],
                        "seconds": time.monotonic() - budget.started,
                        "peak_observed_output_bytes": budget.maximum_observed_bytes,
                        "model_loads": 0, "encoder_updates": 0, "formal_text_reads": 0})
        return {"status": result["status"], "acceptance": result["acceptance"]}
    except Exception as error:
        data.write_json(out / "failure.json", {"status": "CALIBRATION_FAILED_NO_AUTOMATIC_RETRY",
                        "type": type(error).__name__, "message": str(error),
                        "train_attempts": int((out / "train_access.json").exists()),
                        "development_attempts": int((out / "development_access.json").exists())})
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("execute", "finalize"))
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--authorization", type=Path)
    args = parser.parse_args()
    if platform.system() != "Linux":
        raise ValueError("Project research scripts run on Linux")
    if args.mode == "execute" and args.authorization is None:
        parser.error("execute requires --authorization")
    result = execute(args.out, args.authorization) if args.mode == "execute" else finalize(args.out)
    print(data.json_bytes({"status": result["status"], "acceptance": result["acceptance"]}).decode())


if __name__ == "__main__":
    main()
