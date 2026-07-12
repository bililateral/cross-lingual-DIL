#!/usr/bin/env python3
"""Run the preregistered Step12-v6 fixed-boundary robustness audit."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
from collections import Counter
from pathlib import Path

import numpy as np

import step7_train_baseline_models as step7


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_POLICY = ROOT / "schema" / "step12_v6_statistical_robustness_policy.json"
STEP15_POLICY = ROOT / "schema" / "step15_v6_paper_hardening_policy.json"
PR_AUC_DEFINITION = "trapezoidal_area_under_tie_grouped_precision_recall_curve"
PRIMARY_ANALYSIS_MODE = "primary_paired_grouped_component_bootstrap"
SUPPLEMENTAL_ANALYSIS_MODE = "supplemental_two_level_seed_and_component_bootstrap"
PERMUTATION_P_VALUE_METHOD = "paired_split_component_score_swap_randomization_two_sided"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", default=str(DEFAULT_POLICY))
    parser.add_argument("--resamples", type=int, default=None)
    parser.add_argument("--validate-config-only", action="store_true")
    parser.add_argument(
        "--validate-only",
        "--validate-inputs-only",
        dest="validate_inputs_only",
        action="store_true",
        help="Validate configuration and frozen inputs without creating or replacing outputs.",
    )
    parser.add_argument("--print-boundary-hashes", action="store_true")
    return parser.parse_args()


def resolve(path_value: str) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else ROOT / path


def load_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    if path.exists() or temporary.exists():
        raise FileExistsError(f"Refusing to overwrite Step12-v6 output: {path}")
    if not rows:
        temporary.open("x", encoding="utf-8").close()
        temporary.replace(path)
        return
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with temporary.open("x", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    if path.exists() or temporary.exists():
        raise FileExistsError(f"Refusing to overwrite Step12-v6 output: {path}")
    with temporary.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    temporary.replace(path)


def output_paths(policy: dict) -> list[Path]:
    return [resolve(str(path_value)) for path_value in policy["outputs"].values()]


def assert_output_targets_absent(policy: dict) -> None:
    existing = [path for path in output_paths(policy) if path.exists()]
    if existing:
        relative = [str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path) for path in existing]
        raise FileExistsError(
            "Step12-v6 outputs are immutable and at least one target already exists; "
            "use --validate-only or preregister a new output directory/run id: "
            + ", ".join(relative)
        )


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def current_git_commit() -> str | None:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return completed.stdout.strip() or None


def canonical_rows_sha256(rows: list[dict], fields: list[str]) -> str:
    canonical_rows = sorted(
        ([str(row.get(field, "")) for field in fields] for row in rows),
        key=lambda values: tuple(values),
    )
    payload = "".join(
        json.dumps(values, ensure_ascii=False, separators=(",", ":")) + "\n"
        for values in canonical_rows
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def load_verified_active_manifest(
    path: Path,
    expected_run_id: str,
    expected_policy_version: str,
) -> tuple[dict, dict[str, dict]]:
    manifest = step7.load_json(path)
    if manifest.get("run_id") != expected_run_id:
        raise ValueError(
            f"Step15 active-manifest run_id mismatch: expected={expected_run_id!r} "
            f"observed={manifest.get('run_id')!r}"
        )
    if manifest.get("policy_version") != expected_policy_version:
        raise ValueError(
            "Step15 active-manifest policy version mismatch: "
            f"expected={expected_policy_version!r} observed={manifest.get('policy_version')!r}"
        )
    manifest_core = {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    canonical = json.dumps(
        manifest_core,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    observed_manifest_sha = hashlib.sha256(canonical).hexdigest()
    if observed_manifest_sha != manifest.get("manifest_sha256"):
        raise ValueError(
            "Step15 active-manifest self hash mismatch: "
            f"expected={manifest.get('manifest_sha256')} observed={observed_manifest_sha}"
        )
    index: dict[str, dict] = {}
    for record in manifest.get("files", []):
        relative = str(record.get("path", ""))
        if not relative or relative in index:
            raise ValueError(f"Invalid or duplicate path in Step15 active manifest: {relative!r}")
        artifact_path = resolve(relative)
        if not artifact_path.exists():
            raise FileNotFoundError(f"Step15 active-manifest artifact is missing: {artifact_path}")
        observed_sha = file_sha256(artifact_path)
        if observed_sha != record.get("sha256"):
            raise ValueError(
                f"Step15 active-manifest SHA-256 mismatch for {relative}: "
                f"expected={record.get('sha256')} observed={observed_sha}"
            )
        index[relative] = record
    if not index:
        raise ValueError("Step15 active manifest contains no files")
    return manifest, index


def validate_embedded_input_manifest(summary: dict, summary_name: str) -> None:
    manifest = summary.get("input_manifest") or {}
    records = manifest.get("inputs") or []
    if not records:
        raise ValueError(f"{summary_name} has no embedded input manifest")
    expected_sha = hashlib.sha256(
        json.dumps(
            {"inputs": records},
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    if expected_sha != manifest.get("manifest_sha256"):
        raise ValueError(f"{summary_name} embedded input-manifest self hash is invalid")
    for record in records:
        path = resolve(str(record["path"]))
        if not path.exists() or file_sha256(path) != record.get("sha256"):
            raise ValueError(f"{summary_name} embedded input changed after training: {path}")


def validate_step9_context_fingerprint(summary: dict) -> None:
    context = summary.get("summary_context_fingerprints") or {}
    records = context.get("files") or []
    if not records:
        raise ValueError("Step9 summary has no context fingerprint records")
    refreshed = []
    for record in records:
        path = resolve(str(record["path"]))
        if not path.exists():
            raise ValueError(f"Step9 context dependency is missing: {path}")
        current = {
            "path": str(record["path"]),
            "exists": True,
            "size": path.stat().st_size,
            "sha256": file_sha256(path),
        }
        if current != record:
            raise ValueError(f"Step9 context dependency changed after training: {path}")
        refreshed.append(current)
    combined = hashlib.sha256()
    for record in sorted(refreshed, key=lambda item: item["path"]):
        combined.update(json.dumps(record, sort_keys=True, ensure_ascii=False).encode("utf-8"))
    if combined.hexdigest() != context.get("fingerprint"):
        raise ValueError("Step9 summary context fingerprint is internally inconsistent")


def require_manifest_path(path: Path, manifest_index: dict[str, dict]) -> None:
    relative = str(path.relative_to(ROOT))
    if relative not in manifest_index:
        raise ValueError(f"Step15 prediction is not frozen by the active manifest: {relative}")


def validate_output_validation_binding(report: dict, active_manifest_index: dict[str, dict]) -> None:
    records = report.get("validated_inputs") or []
    if not records:
        raise ValueError("Step15-v6 output validation report has no validated input records")
    observed_sha = hashlib.sha256(
        json.dumps(
            {"validated_inputs": records},
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    if observed_sha != report.get("validated_inputs_manifest_sha256"):
        raise ValueError("Step15-v6 output validation report has an invalid self hash")
    for record in records:
        active_record = active_manifest_index.get(str(record["path"]))
        if active_record is None or active_record.get("sha256") != record.get("sha256"):
            raise ValueError(
                f"Step15-v6 output validation report is not bound to the active manifest: {record['path']}"
            )


def eligible_split_rows(rows: list[dict], split_name: str) -> list[dict]:
    selected = [
        row
        for row in rows
        if row.get("split_name") == split_name
        and row.get("review_label") in {"positive", "negative"}
        and row.get("usable_for_supervision") == "1"
        and row.get("usable_for_core_transfer") == "1"
    ]
    return sorted(selected, key=lambda row: row["pair_uid"])


def labels_from_rows(rows: list[dict]) -> np.ndarray:
    return np.asarray([1.0 if row["review_label"] == "positive" else 0.0 for row in rows], dtype=float)


def score_vector(
    path: Path,
    ordered_rows: list[dict],
    column: str = "prob_positive",
    expected_experiment_name: str | None = None,
) -> tuple[np.ndarray, float | None]:
    rows = load_csv(path)
    ordered_pair_uids = [row["pair_uid"] for row in ordered_rows]
    index = {row["pair_uid"]: row for row in rows}
    if len(index) != len(rows):
        raise ValueError(f"Prediction file contains duplicate pair_uid values: {path}")
    missing = [pair_uid for pair_uid in ordered_pair_uids if pair_uid not in index]
    extras = sorted(set(index) - set(ordered_pair_uids))
    if missing or extras:
        raise ValueError(
            f"Prediction boundary mismatch for {path}: missing={len(missing)} extras={len(extras)} "
            f"first_missing={missing[:1]} first_extra={extras[:1]}"
        )
    for expected_row in ordered_rows:
        pair_uid = expected_row["pair_uid"]
        prediction = index[pair_uid]
        expected_y = 1 if expected_row["review_label"] == "positive" else 0
        if str(prediction.get("y_true", "")).strip() and int(float(prediction["y_true"])) != expected_y:
            raise ValueError(f"Prediction y_true disagrees with frozen labels for {pair_uid}: {path}")
        if str(prediction.get("review_label", "")).strip() and prediction["review_label"] != expected_row["review_label"]:
            raise ValueError(f"Prediction review_label disagrees with frozen labels for {pair_uid}: {path}")
        if str(prediction.get("split_component_id", "")).strip() and (
            prediction["split_component_id"] != expected_row.get("split_component_id", "")
        ):
            raise ValueError(f"Prediction component disagrees with frozen labels for {pair_uid}: {path}")
    if expected_experiment_name is not None:
        observed_names = {str(row.get("experiment_name", "")) for row in rows}
        if observed_names != {expected_experiment_name}:
            raise ValueError(
                f"Prediction experiment token mismatch for {path}: "
                f"expected={expected_experiment_name!r} observed={sorted(observed_names)}"
            )
    scores = np.asarray([float(index[pair_uid][column]) for pair_uid in ordered_pair_uids], dtype=float)
    thresholds = {
        float(row["threshold"])
        for row in rows
        if str(row.get("threshold", "")).strip()
    }
    if len(thresholds) > 1:
        raise ValueError(f"Prediction file contains multiple thresholds: {path} -> {sorted(thresholds)}")
    threshold = next(iter(thresholds)) if thresholds else None
    return scores, threshold


def choose_valid_threshold(y_valid: np.ndarray, scores: np.ndarray, step15_policy: dict) -> float:
    return float(
        step7.choose_threshold(
            y_valid,
            scores,
            step15_policy["threshold_selection"]["metric"],
            step15_policy,
        )
    )


def indexed_step15_runs(summary: dict) -> dict[tuple[str, str, int], dict]:
    index: dict[tuple[str, str, int], dict] = {}
    for run in summary.get("runs", []):
        key = (str(run["experiment_name"]), str(run["phase_id"]), int(run["seed"]))
        if key in index:
            raise ValueError(f"Duplicate Step15 run in summary: {key}")
        index[key] = run
    return index


def indexed_source_only_runs(summary: dict) -> dict[int, dict]:
    index: dict[int, dict] = {}
    for run in summary.get("runs", []):
        seed = int(run["seed"])
        if seed in index:
            raise ValueError(f"Duplicate source-only Step15 seed in summary: {seed}")
        index[seed] = run
    return index


def validate_step9_summary_run(
    summary: dict,
    spec: dict,
    seed: int,
    valid_path: Path | None,
    test_path: Path,
) -> dict:
    experiment_name = str(spec["step9_experiment"])
    ratio_token = str(spec.get("ratio_token", "100pct"))
    experiment = (summary.get("experiments") or {}).get(experiment_name)
    if not experiment:
        raise ValueError(f"Step9 summary is missing experiment {experiment_name}")
    run_key = f"{ratio_token}_seed_{seed}"
    run = (experiment.get("runs") or {}).get(run_key)
    if not run:
        raise ValueError(f"Step9 summary is missing run {experiment_name}/{run_key}")
    if int(run.get("seed", -1)) != seed or str(run.get("ratio_token", "")) != ratio_token:
        raise ValueError(f"Step9 summary run metadata mismatch for {experiment_name}/{run_key}")
    artifacts = run.get("artifacts") or {}
    expected_test = str(test_path.relative_to(ROOT))
    if artifacts.get("zh_test_predictions") != expected_test:
        raise ValueError(
            f"Step9 summary test path mismatch for {experiment_name}/{run_key}: "
            f"expected={expected_test} summary={artifacts.get('zh_test_predictions')}"
        )
    if valid_path is not None:
        expected_valid = str(valid_path.relative_to(ROOT))
        if artifacts.get("zh_valid_predictions") != expected_valid:
            raise ValueError(
                f"Step9 summary valid path mismatch for {experiment_name}/{run_key}: "
                f"expected={expected_valid} summary={artifacts.get('zh_valid_predictions')}"
            )
    return run


def assert_ranking_metrics_match_summary(
    y_true: np.ndarray,
    scores: np.ndarray,
    expected_metrics: dict,
    label: str,
    tolerance: float = 2e-5,
) -> None:
    observed = step7.evaluate_probabilities(y_true, scores, 0.5)
    for metric in ("roc_auc", "average_precision", "pr_auc"):
        expected = expected_metrics.get(metric)
        actual = observed.get(metric)
        if expected is None or actual is None or abs(float(expected) - float(actual)) > tolerance:
            raise ValueError(
                f"Prediction scores do not reproduce {label} summary metric {metric}: "
                f"expected={expected} observed={actual} tolerance={tolerance}"
            )


def select_by_validation_metric(
    candidate_values: dict[str, float],
    tie_break_order: list[str],
    tolerance: float,
) -> tuple[str, dict]:
    """Select only from validation values, resolving numerical ties by preregistered simplicity."""
    if not candidate_values:
        raise ValueError("Validation-only selection received no candidates")
    if set(candidate_values) != set(tie_break_order) or len(tie_break_order) != len(
        set(tie_break_order)
    ):
        raise ValueError(
            "Validation selection tie-break order must contain every candidate exactly once"
        )
    if tolerance < 0.0:
        raise ValueError("Validation selection tie tolerance must be non-negative")
    best_value = max(float(value) for value in candidate_values.values())
    tied = {
        model_id
        for model_id, value in candidate_values.items()
        if best_value - float(value) <= tolerance
    }
    selected = next(model_id for model_id in tie_break_order if model_id in tied)
    return selected, {
        "candidate_valid_average_precision": {
            model_id: round(float(candidate_values[model_id]), 8)
            for model_id in tie_break_order
        },
        "best_valid_average_precision": round(best_value, 8),
        "tied_within_tolerance": [
            model_id for model_id in tie_break_order if model_id in tied
        ],
        "simplicity_tie_break_order": list(tie_break_order),
        "tie_tolerance": float(tolerance),
        "selected_model_id": selected,
        "selection_split": "zh_valid",
        "selection_metric": "average_precision",
        "test_metrics_used_for_selection": False,
    }


def load_step15_validation_candidate(
    spec: dict,
    seeds: list[int],
    templates: dict,
    valid_rows: list[dict],
    y_valid: np.ndarray,
    step15_run_index: dict[tuple[str, str, int], dict],
    active_manifest_index: dict[str, dict],
    input_paths: set[Path],
    step15_policy: dict,
) -> dict:
    valid_seed_scores: list[np.ndarray] = []
    source_paths: list[str] = []
    experiment = str(spec["experiment"])
    phase = str(spec["phase"])
    for seed in seeds:
        run_key = (experiment, phase, seed)
        run = step15_run_index.get(run_key)
        if run is None:
            raise ValueError(f"Step15 summary is missing validation-only run {run_key}")
        valid_path = resolve(
            templates["valid"].format(
                seed=seed,
                experiment=experiment,
                phase=phase,
            )
        )
        if not valid_path.exists():
            raise FileNotFoundError(
                f"Missing Step12-v6 validation-only M5 prediction: {valid_path}"
            )
        expected_relative = str(valid_path.relative_to(ROOT))
        if (run.get("output_paths") or {}).get("zh_valid_predictions") != expected_relative:
            raise ValueError(f"Step15 summary validation path mismatch for {run_key}")
        require_manifest_path(valid_path, active_manifest_index)
        expected_name = f"{experiment}_{phase}_seed_{seed}"
        valid_scores, _ = score_vector(
            valid_path,
            valid_rows,
            expected_experiment_name=expected_name,
        )
        summary_valid_metrics = run.get("zh_valid_metrics")
        if summary_valid_metrics is not None:
            assert_ranking_metrics_match_summary(
                y_valid,
                valid_scores,
                summary_valid_metrics,
                f"{spec['model_id']}/seed={seed}/valid",
            )
        valid_seed_scores.append(valid_scores)
        input_paths.add(valid_path)
        source_paths.append(expected_relative)
    valid_matrix = np.vstack(valid_seed_scores)
    valid_mean = valid_matrix.mean(axis=0)
    return {
        "model_id": str(spec["model_id"]),
        "role": str(spec["role"]),
        "seed_ids": list(seeds),
        "valid_seed_scores": valid_matrix,
        "valid_scores": valid_mean,
        "valid_average_precision": float(step7.average_precision_score(y_valid, valid_mean)),
        "threshold": choose_valid_threshold(y_valid, valid_mean, step15_policy),
        "threshold_source": "mean_zh_valid_scores",
        "source_paths": source_paths,
        "spec": dict(spec),
    }


def load_selected_step15_test_candidate(
    validation_candidate: dict,
    templates: dict,
    test_rows: list[dict],
    y_test: np.ndarray,
    step15_run_index: dict[tuple[str, str, int], dict],
    active_manifest_index: dict[str, dict],
    input_paths: set[Path],
) -> dict:
    """Load test predictions only after validation has frozen the selected M5 candidate."""
    spec = validation_candidate["spec"]
    experiment = str(spec["experiment"])
    phase = str(spec["phase"])
    test_seed_scores: list[np.ndarray] = []
    source_paths = list(validation_candidate["source_paths"])
    for seed in validation_candidate["seed_ids"]:
        run_key = (experiment, phase, int(seed))
        run = step15_run_index[run_key]
        test_path = resolve(
            templates["test"].format(
                seed=seed,
                experiment=experiment,
                phase=phase,
            )
        )
        if not test_path.exists():
            raise FileNotFoundError(
                f"Validation-selected M5 candidate has no test prediction: {test_path}"
            )
        expected_relative = str(test_path.relative_to(ROOT))
        if (run.get("output_paths") or {}).get("zh_test_predictions") != expected_relative:
            raise ValueError(f"Step15 summary selected-M5 test path mismatch for {run_key}")
        require_manifest_path(test_path, active_manifest_index)
        expected_name = f"{experiment}_{phase}_seed_{seed}"
        test_scores, _ = score_vector(
            test_path,
            test_rows,
            expected_experiment_name=expected_name,
        )
        summary_test_metrics = run.get("zh_test_metrics")
        if summary_test_metrics is not None:
            assert_ranking_metrics_match_summary(
                y_test,
                test_scores,
                summary_test_metrics,
                f"{spec['model_id']}/seed={seed}/test",
            )
        test_seed_scores.append(test_scores)
        input_paths.add(test_path)
        source_paths.append(expected_relative)
    test_matrix = np.vstack(test_seed_scores)
    return {
        "model_id": validation_candidate["model_id"],
        "role": validation_candidate["role"],
        "seed_ids": list(validation_candidate["seed_ids"]),
        "seed_scores": test_matrix,
        "scores": test_matrix.mean(axis=0),
        "valid_scores": validation_candidate["valid_scores"],
        "valid_average_precision": validation_candidate["valid_average_precision"],
        "threshold": validation_candidate["threshold"],
        "threshold_source": validation_candidate["threshold_source"],
        "source_paths": source_paths,
    }


def load_models(
    policy: dict,
    labels: list[dict],
    features: list[dict],
    step15_summary: dict,
    source_only_summary: dict,
    step9_summary: dict,
    step7_summary: dict,
    active_manifest_index: dict[str, dict],
    input_paths: set[Path],
) -> tuple[dict[str, dict], dict]:
    test_rows = eligible_split_rows(labels, policy["fixed_test"]["split_name"])
    valid_rows = eligible_split_rows(labels, "valid")
    test_uids = [row["pair_uid"] for row in test_rows]
    valid_uids = [row["pair_uid"] for row in valid_rows]
    y_valid = labels_from_rows(valid_rows)
    y_test = labels_from_rows(test_rows)
    feature_index = {row["pair_uid"]: row for row in features}
    step15_policy = step7.load_json(STEP15_POLICY)
    input_paths.add(STEP15_POLICY)
    step15_run_index = indexed_step15_runs(step15_summary)
    source_run_index = indexed_source_only_runs(source_only_summary)
    models: dict[str, dict] = {}
    validation_only_candidates: dict[str, dict] = {}
    templates = policy["step15_prediction_templates"]
    alias_cfg = policy["validation_selected_aliases"]
    m5_cfg = alias_cfg["step15_v6_m5_selected"]
    m5_candidate_ids = set(m5_cfg["candidate_model_ids"])

    for spec in policy["models"]:
        spec = dict(spec)
        model_id = spec["model_id"]
        kind = spec["kind"]
        test_seed_scores: list[np.ndarray] = []
        valid_seed_scores: list[np.ndarray] = []
        frozen_thresholds: list[float] = []
        source_paths: list[str] = []
        seeds = [int(value) for value in spec.get("seeds", templates["seeds"])]
        if len(seeds) != len(set(seeds)):
            raise ValueError(f"Step12-v6 model has duplicate seed IDs: {model_id} -> {seeds}")
        if model_id in m5_candidate_ids:
            validation_only_candidates[model_id] = load_step15_validation_candidate(
                spec,
                seeds,
                templates,
                valid_rows,
                y_valid,
                step15_run_index,
                active_manifest_index,
                input_paths,
                step15_policy,
            )
            continue
        if kind == "feature":
            column = spec["score_column"]
            test_seed_scores = [
                np.asarray([float(feature_index[pair_uid][column]) for pair_uid in test_uids], dtype=float)
            ]
            valid_seed_scores = [
                np.asarray([float(feature_index[pair_uid][column]) for pair_uid in valid_uids], dtype=float)
            ]
            seeds = [0]
        else:
            if spec.get("experiment"):
                valid_template = templates["valid"]
                test_template = templates["test"]
            else:
                valid_template = spec.get("valid_template")
                test_template = spec["test_template"]
            for seed in seeds:
                test_path = resolve(
                    test_template.format(
                        seed=seed,
                        experiment=spec.get("experiment", ""),
                        phase=spec.get("phase", ""),
                    )
                )
                if not test_path.exists():
                    raise FileNotFoundError(f"Missing Step12-v6 model prediction: {test_path}")
                expected_experiment_name = None
                expected_valid_path = None
                summary_run = None
                summary_test_metrics = None
                summary_valid_metrics = None
                if spec.get("experiment"):
                    run_key = (str(spec["experiment"]), str(spec["phase"]), seed)
                    run = step15_run_index.get(run_key)
                    if run is None:
                        raise ValueError(f"Step15 summary is missing run {run_key}")
                    expected_relative = str(test_path.relative_to(ROOT))
                    if (run.get("output_paths") or {}).get("zh_test_predictions") != expected_relative:
                        raise ValueError(f"Step15 summary test path mismatch for {run_key}")
                    require_manifest_path(test_path, active_manifest_index)
                    expected_experiment_name = f"{spec['experiment']}_{spec['phase']}_seed_{seed}"
                    summary_run = run
                    summary_test_metrics = run.get("zh_test_metrics")
                    summary_valid_metrics = run.get("zh_valid_metrics")
                elif kind == "test_prediction_ensemble_with_frozen_threshold":
                    run = source_run_index.get(seed)
                    if run is None:
                        raise ValueError(f"Source-only Step15 summary is missing seed {seed}")
                    expected_relative = str(test_path.relative_to(ROOT))
                    if (run.get("output_paths") or {}).get("target_predictions") != expected_relative:
                        raise ValueError(f"Source-only Step15 summary test path mismatch for seed {seed}")
                    require_manifest_path(test_path, active_manifest_index)
                    expected_experiment_name = str(run["model_id"])
                    summary_run = run
                    summary_test_metrics = run.get("target_test_metrics")
                elif kind == "step7_test_prediction_with_frozen_source_threshold":
                    experiment_name = str(spec["step7_experiment"])
                    run = (step7_summary.get("experiments") or {}).get(experiment_name)
                    if not run:
                        raise ValueError(f"Step7 summary is missing experiment {experiment_name}")
                    expected_experiment_name = experiment_name
                    summary_run = run
                    summary_test_metrics = run.get("zh_zero_shot_test_metrics")
                    selected_threshold = run.get("selected_threshold")
                    if selected_threshold is None:
                        raise ValueError(f"Step7 summary has no source-valid threshold for {experiment_name}")
                    frozen_thresholds.append(float(selected_threshold))
                    require_manifest_path(test_path, active_manifest_index)
                test_scores, frozen_threshold = score_vector(
                    test_path,
                    test_rows,
                    expected_experiment_name=expected_experiment_name,
                )
                test_seed_scores.append(test_scores)
                if frozen_threshold is not None:
                    frozen_thresholds.append(frozen_threshold)
                input_paths.add(test_path)
                source_paths.append(str(test_path.relative_to(ROOT)))
                if valid_template:
                    valid_path = resolve(
                        valid_template.format(
                            seed=seed,
                            experiment=spec.get("experiment", ""),
                            phase=spec.get("phase", ""),
                        )
                    )
                    if not valid_path.exists():
                        raise FileNotFoundError(f"Missing Step12-v6 validation prediction: {valid_path}")
                    if spec.get("experiment"):
                        run_key = (str(spec["experiment"]), str(spec["phase"]), seed)
                        run = step15_run_index[run_key]
                        expected_relative = str(valid_path.relative_to(ROOT))
                        if (run.get("output_paths") or {}).get("zh_valid_predictions") != expected_relative:
                            raise ValueError(f"Step15 summary valid path mismatch for {run_key}")
                        require_manifest_path(valid_path, active_manifest_index)
                    expected_valid_path = valid_path
                    valid_scores, _ = score_vector(
                        valid_path,
                        valid_rows,
                        expected_experiment_name=expected_experiment_name,
                    )
                    valid_seed_scores.append(valid_scores)
                    input_paths.add(valid_path)
                    source_paths.append(str(valid_path.relative_to(ROOT)))
                if spec.get("step9_experiment"):
                    summary_run = validate_step9_summary_run(
                        step9_summary,
                        spec,
                        seed,
                        expected_valid_path,
                        test_path,
                    )
                    summary_test_metrics = summary_run.get("zh_test_metrics")
                    summary_valid_metrics = summary_run.get("zh_valid_metrics")
                    expected_step9_token = f"{spec['step9_experiment']}_{spec.get('ratio_token', '100pct')}_seed_{seed}"
                    require_manifest_path(test_path, active_manifest_index)
                    score_vector(test_path, test_rows, expected_experiment_name=expected_step9_token)
                    if expected_valid_path is not None:
                        require_manifest_path(expected_valid_path, active_manifest_index)
                        score_vector(
                            expected_valid_path,
                            valid_rows,
                            expected_experiment_name=expected_step9_token,
                        )
                if summary_test_metrics is not None:
                    assert_ranking_metrics_match_summary(
                        y_test,
                        test_scores,
                        summary_test_metrics,
                        f"{model_id}/seed={seed}/test",
                    )
                if summary_valid_metrics is not None and valid_seed_scores:
                    assert_ranking_metrics_match_summary(
                        y_valid,
                        valid_seed_scores[-1],
                        summary_valid_metrics,
                        f"{model_id}/seed={seed}/valid",
                    )
        test_matrix = np.vstack(test_seed_scores)
        valid_matrix = np.vstack(valid_seed_scores) if valid_seed_scores else None
        test_mean = test_matrix.mean(axis=0)
        valid_mean = valid_matrix.mean(axis=0) if valid_matrix is not None else None
        if valid_mean is not None:
            threshold = choose_valid_threshold(y_valid, valid_mean, step15_policy)
            valid_ap = step7.average_precision_score(y_valid, valid_mean)
            threshold_source = "mean_zh_valid_scores"
        elif frozen_thresholds:
            threshold = float(np.mean(frozen_thresholds))
            valid_ap = None
            threshold_source = "source_valid_frozen_threshold_mean"
        else:
            raise ValueError(f"No validation scores or frozen threshold for {model_id}")
        models[model_id] = {
            "model_id": model_id,
            "role": spec["role"],
            "seed_ids": seeds,
            "seed_scores": test_matrix,
            "scores": test_mean,
            "valid_scores": valid_mean,
            "valid_average_precision": None if valid_ap is None else float(valid_ap),
            "threshold": threshold,
            "threshold_source": threshold_source,
            "source_paths": source_paths,
        }

    selection_record = step15_summary.get("validation_only_model_selection", {}).get(
        "m5_auxiliary_loss_weight", {}
    )
    if selection_record.get("selection_scope") != "fixed_zh_valid_only_never_zh_test":
        raise ValueError("Step15 M5 summary does not certify validation-only selection")
    if selection_record.get("metric") != "average_precision":
        raise ValueError("Step15 M5 summary was not selected by zh_valid average precision")
    if selection_record.get("status") != "selected":
        raise ValueError("Step15 M5 summary has no frozen validation-selected candidate")
    expected_m5_experiment_tie_order = [
        str(validation_only_candidates[model_id]["spec"]["experiment"])
        for model_id in m5_cfg["simplicity_tie_break_order"]
    ]
    if selection_record.get("tie_break_order") != expected_m5_experiment_tie_order:
        raise ValueError("Step15 and Step12 M5 simplicity tie-break orders disagree")
    selected_experiment = selection_record.get("selected_experiment")
    experiment_to_model = {
        "step15_v6_m5_aux_evidence_lambda_0p1": "step15_v6_m5_lambda_0p1",
        "step15_v6_m5_aux_evidence_lambda_0p3": "step15_v6_m5_lambda_0p3",
    }
    selected_m5_id, m5_selection = select_by_validation_metric(
        {
            model_id: float(validation_only_candidates[model_id]["valid_average_precision"])
            for model_id in m5_cfg["candidate_model_ids"]
        },
        [str(model_id) for model_id in m5_cfg["simplicity_tie_break_order"]],
        float(m5_cfg["tie_tolerance"]),
    )
    recomputed_selected_experiment = str(
        validation_only_candidates[selected_m5_id]["spec"]["experiment"]
    )
    if selected_experiment != recomputed_selected_experiment:
        raise ValueError(
            "Step15 M5 selection does not match the recomputed ten-seed ensemble zh_valid AP: "
            f"summary={selected_experiment} recomputed={recomputed_selected_experiment}"
        )
    if experiment_to_model.get(selected_experiment) != selected_m5_id:
        raise ValueError(f"Step15 summary did not select a valid M5 candidate: {selected_experiment}")
    for unselected_m5_id in sorted(m5_candidate_ids - {selected_m5_id}):
        unselected = validation_only_candidates[unselected_m5_id]
        unselected_spec = unselected["spec"]
        for seed in unselected["seed_ids"]:
            run = step15_run_index[
                (
                    str(unselected_spec["experiment"]),
                    str(unselected_spec["phase"]),
                    int(seed),
                )
            ]
            if (run.get("output_paths") or {}).get("zh_test_predictions") or run.get(
                "zh_test_metrics"
            ) is not None:
                raise ValueError(
                    "Non-selected M5 lambda must remain validation-only and must not expose "
                    f"zh_test output: model={unselected_m5_id}, seed={seed}"
                )
    selected_m5_model = load_selected_step15_test_candidate(
        validation_only_candidates[selected_m5_id],
        templates,
        test_rows,
        y_test,
        step15_run_index,
        active_manifest_index,
        input_paths,
    )
    models[selected_m5_id] = selected_m5_model
    models["step15_v6_m5_selected"] = {
        **selected_m5_model,
        "model_id": "step15_v6_m5_selected",
        "role": "validation_selected_auxiliary_evidence_head",
        "alias_of": selected_m5_id,
    }
    final_cfg = alias_cfg["step15_v6_final_selected"]
    final_candidates = [str(model_id) for model_id in final_cfg["candidate_model_ids"]]
    final_selected_id, final_selection = select_by_validation_metric(
        {
            model_id: float(models[model_id]["valid_average_precision"])
            for model_id in final_candidates
        },
        [str(model_id) for model_id in final_cfg["simplicity_tie_break_order"]],
        float(final_cfg["tie_tolerance"]),
    )
    models["step15_v6_final_selected"] = {
        **models[final_selected_id],
        "model_id": "step15_v6_final_selected",
        "role": "validation_only_selected_final_clean_v6_candidate",
        "alias_of": final_selected_id,
    }
    step9_cfg = alias_cfg["step9_strongest_clean_validation_selected"]
    step9_candidate_ids = [str(model_id) for model_id in step9_cfg["candidate_model_ids"]]
    selected_step9_id, step9_selection = select_by_validation_metric(
        {
            model_id: float(models[model_id]["valid_average_precision"])
            for model_id in step9_candidate_ids
        },
        [str(model_id) for model_id in step9_cfg["simplicity_tie_break_order"]],
        float(step9_cfg["tie_tolerance"]),
    )
    models["step9_strongest_clean_validation_selected"] = {
        **models[selected_step9_id],
        "model_id": "step9_strongest_clean_validation_selected",
        "role": "strongest_clean_step9_selected_on_zh_valid_only",
        "alias_of": selected_step9_id,
    }
    selection = {
        "m5": m5_selection,
        "m5_selected": selected_m5_id,
        "m5_selected_experiment_recomputed_from_predictions": recomputed_selected_experiment,
        "m5_non_selected_candidates_are_validation_only": sorted(
            m5_candidate_ids - {selected_m5_id}
        ),
        "final": final_selection,
        "final_selected": final_selected_id,
        "strongest_clean_step9": step9_selection,
        "strongest_clean_step9_selected": selected_step9_id,
        "selection_scope": (
            "average_precision_of_ten_seed_mean_zh_valid_scores_only_never_zh_test"
        ),
        "test_metrics_used_for_any_selection": False,
    }
    return models, selection


def metric_value(metric: str, y: np.ndarray, scores: np.ndarray, threshold: float) -> float | None:
    values = step7.evaluate_probabilities(y, scores, threshold)
    value = (values.get("confusion") or {}).get(metric) if metric in {"tp", "tn", "fp", "fn"} else values.get(metric)
    return None if value is None else float(value)


def component_groups(rows: list[dict]) -> list[np.ndarray]:
    grouped: dict[str, list[int]] = {}
    for idx, row in enumerate(rows):
        component_id = str(row.get("split_component_id", ""))
        if not component_id:
            raise ValueError(f"Missing split_component_id for {row['pair_uid']}")
        grouped.setdefault(component_id, []).append(idx)
    return [np.asarray(indices, dtype=int) for _, indices in sorted(grouped.items())]


def sampled_component_indices(groups: list[np.ndarray], rng: np.random.Generator) -> np.ndarray:
    chosen = rng.integers(0, len(groups), size=len(groups))
    return np.concatenate([groups[int(idx)] for idx in chosen])


def component_swapped_scores(
    candidate_scores: np.ndarray,
    baseline_scores: np.ndarray,
    groups: list[np.ndarray],
    swap_flags: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Swap complete component score blocks; never split a component across model assignments."""
    if candidate_scores.shape != baseline_scores.shape:
        raise ValueError("Paired randomization requires score vectors with identical shapes")
    if len(groups) != len(swap_flags):
        raise ValueError("Paired randomization received one swap flag per component")
    seen = np.zeros(candidate_scores.shape[0], dtype=int)
    swapped_candidate = np.array(candidate_scores, copy=True)
    swapped_baseline = np.array(baseline_scores, copy=True)
    for group, should_swap in zip(groups, swap_flags, strict=True):
        seen[group] += 1
        if bool(should_swap):
            swapped_candidate[group] = baseline_scores[group]
            swapped_baseline[group] = candidate_scores[group]
    if not np.all(seen == 1):
        raise ValueError("Paired randomization groups must partition every evaluation row exactly once")
    return swapped_candidate, swapped_baseline


def paired_component_randomization_test(
    candidate: dict,
    baseline: dict,
    y_true: np.ndarray,
    groups: list[np.ndarray],
    metric: str,
    num_permutations: int,
    seed: int,
) -> dict:
    """Two-sided paired score-swap test under the component-level exchangeability null."""
    if num_permutations <= 0:
        raise ValueError("Paired randomization requires a positive permutation count")
    candidate_scores = np.asarray(candidate["scores"], dtype=float)
    baseline_scores = np.asarray(baseline["scores"], dtype=float)
    observed_candidate = metric_value(
        metric, y_true, candidate_scores, candidate["threshold"]
    )
    observed_baseline = metric_value(
        metric, y_true, baseline_scores, baseline["threshold"]
    )
    if observed_candidate is None or observed_baseline is None:
        return {
            "observed_difference": None,
            "p_value": None,
            "extreme_count": 0,
            "valid_permutations": 0,
        }
    observed_difference = float(observed_candidate - observed_baseline)
    rng = np.random.default_rng(seed)
    extreme_count = 0
    valid_permutations = 0
    for _ in range(num_permutations):
        swap_flags = rng.integers(0, 2, size=len(groups), dtype=np.int8).astype(bool)
        null_candidate, null_baseline = component_swapped_scores(
            candidate_scores,
            baseline_scores,
            groups,
            swap_flags,
        )
        null_candidate_value = metric_value(
            metric, y_true, null_candidate, candidate["threshold"]
        )
        null_baseline_value = metric_value(
            metric, y_true, null_baseline, baseline["threshold"]
        )
        if null_candidate_value is None or null_baseline_value is None:
            continue
        null_difference = float(null_candidate_value - null_baseline_value)
        valid_permutations += 1
        if abs(null_difference) >= abs(observed_difference) - 1e-15:
            extreme_count += 1
    if valid_permutations != num_permutations:
        raise ValueError(
            "Paired component randomization produced undefined metric values: "
            f"valid={valid_permutations} requested={num_permutations}"
        )
    return {
        "observed_difference": observed_difference,
        "p_value": (1.0 + extreme_count) / (valid_permutations + 1.0),
        "extreme_count": extreme_count,
        "valid_permutations": valid_permutations,
    }


def percentile_interval(values: list[float], confidence: float) -> tuple[float | None, float | None]:
    if not values:
        return None, None
    alpha = (1.0 - confidence) / 2.0
    return float(np.quantile(values, alpha)), float(np.quantile(values, 1.0 - alpha))


def build_positive_slice_masks(
    test_rows: list[dict],
    reaudit_rows: list[dict],
    policy: dict,
) -> tuple[dict[str, np.ndarray], dict[str, int]]:
    tier_index = {
        row["pair_uid"]: row["paper_evidence_tier"]
        for row in reaudit_rows
        if row.get("split_name") == "test"
    }
    strict_tiers = set(policy["positive_slices"]["strict_direct_or_component"])
    soft_tiers = set(policy["positive_slices"]["soft_primary"])
    buckets: list[str] = []
    for row in test_rows:
        if row["review_label"] == "negative":
            buckets.append("negative")
            continue
        tier = tier_index.get(row["pair_uid"])
        if tier is None:
            raise ValueError(f"Step16F is missing positive test pair: {row['pair_uid']}")
        if tier in strict_tiers:
            buckets.append("strict_direct_or_component")
        elif tier in soft_tiers:
            buckets.append("soft_primary")
        else:
            buckets.append("secondary_or_sensitivity_only")
    bucket_array = np.asarray(buckets, dtype=object)
    counts = dict(Counter(value for value in buckets if value != "negative"))
    expected = policy["expected_test_positive_slice_counts"]
    if counts != expected:
        raise ValueError(f"Step16F test positive slice mismatch: expected={expected}, actual={counts}")
    negative = bucket_array == "negative"
    strict = bucket_array == "strict_direct_or_component"
    soft = bucket_array == "soft_primary"
    secondary = bucket_array == "secondary_or_sensitivity_only"
    return {
        "all_test": np.ones(len(test_rows), dtype=bool),
        "strict_direct_or_component_positive_vs_all_negative": negative | strict,
        "strict_plus_soft_primary_positive_vs_all_negative": negative | strict | soft,
        "soft_primary_positive_vs_all_negative": negative | soft,
        "secondary_positive_vs_all_negative": negative | secondary,
    }, counts


def masked_model(model: dict, mask: np.ndarray) -> dict:
    return {
        **model,
        "scores": np.asarray(model["scores"])[mask],
        "seed_scores": np.asarray(model["seed_scores"])[:, mask],
    }


def model_metric_rows(
    models: dict[str, dict],
    test_rows: list[dict],
    y_test: np.ndarray,
    groups: list[np.ndarray],
    policy: dict,
    resamples: int,
) -> list[dict]:
    bootstrap_seed = int(policy["bootstrap"]["random_seed"])
    component_rng = np.random.default_rng(bootstrap_seed)
    confidence = float(policy["bootstrap"]["confidence_level"])
    metrics = [
        policy["metrics"]["primary"],
        *policy["metrics"]["secondary"],
        *policy["metrics"]["threshold"],
        *policy["metrics"].get("confusion", []),
    ]
    component_draws = [sampled_component_indices(groups, component_rng) for _ in range(resamples)]
    seed_draws_by_count: dict[int, list[np.ndarray]] = {}
    for seed_count in sorted({int(model["seed_scores"].shape[0]) for model in models.values()}):
        seed_rng = np.random.default_rng(bootstrap_seed + 1000003 * seed_count)
        seed_draws_by_count[seed_count] = [
            seed_rng.integers(0, seed_count, size=seed_count) for _ in range(resamples)
        ]
    rows = []
    for model_id, model in models.items():
        distributions = {metric: [] for metric in metrics}
        component_only_distributions = {metric: [] for metric in metrics}
        seed_count = int(model["seed_scores"].shape[0])
        for indices, seed_indices in zip(
            component_draws,
            seed_draws_by_count[seed_count],
            strict=True,
        ):
            y_sample = y_test[indices]
            if len(set(y_sample.tolist())) < 2:
                continue
            fixed_seed_mean_scores = model["scores"][indices]
            for metric in metrics:
                fixed_value = metric_value(
                    metric, y_sample, fixed_seed_mean_scores, model["threshold"]
                )
                if fixed_value is not None:
                    component_only_distributions[metric].append(fixed_value)
            scores_sample = model["seed_scores"][seed_indices].mean(axis=0)[indices]
            for metric in metrics:
                value = metric_value(metric, y_sample, scores_sample, model["threshold"])
                if value is not None:
                    distributions[metric].append(value)
        point = step7.evaluate_probabilities(y_test, model["scores"], model["threshold"])
        record = {
            "model_id": model_id,
            "role": model["role"],
            "alias_of": model.get("alias_of"),
            "seed_count": int(model["seed_scores"].shape[0]),
            "bootstrap_mode": (
                "primary_split_component_fixed_seed_mean_with_"
                "supplemental_two_level_seed_and_component"
            ),
            "threshold": round(float(model["threshold"]), 8),
            "threshold_source": model["threshold_source"],
            "valid_average_precision": None
            if model["valid_average_precision"] is None
            else round(float(model["valid_average_precision"]), 8),
            "seed_ids": ";".join(str(seed_id) for seed_id in model["seed_ids"]),
            "metric_semantics_version": policy["metrics"]["metric_semantics_version"],
            "pr_auc_definition": PR_AUC_DEFINITION,
        }
        for ranking_metric in [policy["metrics"]["primary"], *policy["metrics"]["secondary"]]:
            per_seed_values = [
                metric_value(ranking_metric, y_test, seed_scores, model["threshold"])
                for seed_scores in model["seed_scores"]
            ]
            finite_values = [float(value) for value in per_seed_values if value is not None]
            record[f"{ranking_metric}_per_seed_values"] = ";".join(
                "" if value is None else f"{float(value):.8f}" for value in per_seed_values
            )
            record[f"{ranking_metric}_seed_mean"] = (
                round(float(np.mean(finite_values)), 8) if finite_values else None
            )
            record[f"{ranking_metric}_seed_std"] = (
                round(float(np.std(finite_values, ddof=1)), 8) if len(finite_values) > 1 else 0.0
            )
            record[f"{ranking_metric}_seed_min"] = (
                round(float(np.min(finite_values)), 8) if finite_values else None
            )
            record[f"{ranking_metric}_seed_max"] = (
                round(float(np.max(finite_values)), 8) if finite_values else None
            )
        for metric in metrics:
            low, high = percentile_interval(component_only_distributions[metric], confidence)
            two_level_low, two_level_high = percentile_interval(
                distributions[metric], confidence
            )
            record[metric] = (
                (point.get("confusion") or {}).get(metric)
                if metric in {"tp", "tn", "fp", "fn"}
                else point.get(metric)
            )
            record[f"{metric}_ci_low"] = None if low is None else round(low, 8)
            record[f"{metric}_ci_high"] = None if high is None else round(high, 8)
            record[f"{metric}_two_level_ci_low"] = (
                None if two_level_low is None else round(two_level_low, 8)
            )
            record[f"{metric}_two_level_ci_high"] = (
                None if two_level_high is None else round(two_level_high, 8)
            )
        rows.append(record)
    return rows


def evidence_slice_rows(
    models: dict[str, dict],
    test_rows: list[dict],
    y_test: np.ndarray,
    masks: dict[str, np.ndarray],
    policy: dict,
    resamples: int,
) -> list[dict]:
    rows = []
    confidence = float(policy["bootstrap"]["confidence_level"])
    base_seed = int(policy["bootstrap"]["random_seed"])
    for slice_offset, (slice_name, mask) in enumerate(masks.items()):
        y_slice = y_test[mask]
        slice_source_rows = [row for row, keep in zip(test_rows, mask, strict=True) if keep]
        slice_groups = component_groups(slice_source_rows)
        component_rng = np.random.default_rng(base_seed + 200003 * (slice_offset + 1))
        component_draws = [
            sampled_component_indices(slice_groups, component_rng) for _ in range(resamples)
        ]
        seed_draws_by_count: dict[int, list[np.ndarray]] = {}
        for seed_count in sorted({int(model["seed_scores"].shape[0]) for model in models.values()}):
            seed_rng = np.random.default_rng(
                base_seed + 200003 * (slice_offset + 1) + 1000003 * seed_count
            )
            seed_draws_by_count[seed_count] = [
                seed_rng.integers(0, seed_count, size=seed_count) for _ in range(resamples)
            ]
        prevalence = float(y_slice.mean()) if len(y_slice) else 0.0
        for model_id, model in models.items():
            metrics = step7.evaluate_probabilities(y_slice, model["scores"][mask], model["threshold"])
            component_only_distributions: dict[str, list[float]] = {
                "roc_auc": [],
                "average_precision": [],
                "pr_auc": [],
                "average_precision_prevalence_lift": [],
            }
            two_level_distributions: dict[str, list[float]] = {
                metric_name: [] for metric_name in component_only_distributions
            }
            seed_count = int(model["seed_scores"].shape[0])
            slice_seed_scores = model["seed_scores"][:, mask]
            slice_seed_mean_scores = model["scores"][mask]
            for indices, seed_indices in zip(
                component_draws,
                seed_draws_by_count[seed_count],
                strict=True,
            ):
                y_sample = y_slice[indices]
                if len(set(y_sample.tolist())) < 2:
                    continue
                fixed_score_sample = slice_seed_mean_scores[indices]
                fixed_metrics = step7.evaluate_probabilities(
                    y_sample, fixed_score_sample, model["threshold"]
                )
                two_level_score_sample = slice_seed_scores[seed_indices].mean(axis=0)[indices]
                two_level_metrics = step7.evaluate_probabilities(
                    y_sample, two_level_score_sample, model["threshold"]
                )
                sampled_prevalence = float(y_sample.mean())
                for metric_name in ("roc_auc", "average_precision", "pr_auc"):
                    fixed_value = fixed_metrics.get(metric_name)
                    if fixed_value is not None:
                        component_only_distributions[metric_name].append(float(fixed_value))
                    two_level_value = two_level_metrics.get(metric_name)
                    if two_level_value is not None:
                        two_level_distributions[metric_name].append(float(two_level_value))
                fixed_ap = fixed_metrics.get("average_precision")
                if fixed_ap is not None and sampled_prevalence > 0.0:
                    component_only_distributions[
                        "average_precision_prevalence_lift"
                    ].append(float(fixed_ap) / sampled_prevalence)
                two_level_ap = two_level_metrics.get("average_precision")
                if two_level_ap is not None and sampled_prevalence > 0.0:
                    two_level_distributions["average_precision_prevalence_lift"].append(
                        float(two_level_ap) / sampled_prevalence
                    )
            interval_fields = {}
            for metric_name, values in component_only_distributions.items():
                low, high = percentile_interval(values, confidence)
                interval_fields[f"{metric_name}_ci_low"] = None if low is None else round(low, 8)
                interval_fields[f"{metric_name}_ci_high"] = None if high is None else round(high, 8)
                two_level_low, two_level_high = percentile_interval(
                    two_level_distributions[metric_name], confidence
                )
                interval_fields[f"{metric_name}_two_level_ci_low"] = (
                    None if two_level_low is None else round(two_level_low, 8)
                )
                interval_fields[f"{metric_name}_two_level_ci_high"] = (
                    None if two_level_high is None else round(two_level_high, 8)
                )
            rows.append(
                {
                    "slice_name": slice_name,
                    "model_id": model_id,
                    "row_count": len(y_slice),
                    "positive_count": int(y_slice.sum()),
                    "negative_count": int(len(y_slice) - y_slice.sum()),
                    "prevalence": round(prevalence, 8),
                    "unstable_slice": int(y_slice.sum()) < 10,
                    "bootstrap_mode": (
                        "primary_split_component_fixed_seed_mean_with_"
                        "supplemental_two_level_seed_and_component"
                    ),
                    "valid_bootstrap_resamples": len(
                        component_only_distributions["average_precision"]
                    ),
                    "valid_two_level_bootstrap_resamples": len(
                        two_level_distributions["average_precision"]
                    ),
                    "metric_semantics_version": policy["metrics"]["metric_semantics_version"],
                    "pr_auc_definition": PR_AUC_DEFINITION,
                    "roc_auc": metrics["roc_auc"],
                    "average_precision": metrics["average_precision"],
                    "average_precision_prevalence_lift": None
                    if metrics["average_precision"] is None or prevalence <= 0.0
                    else round(float(metrics["average_precision"]) / prevalence, 8),
                    "pr_auc": metrics["pr_auc"],
                    "accuracy": metrics["accuracy"],
                    "balanced_accuracy": metrics["balanced_accuracy"],
                    "f1": metrics["f1"],
                    "precision": metrics["precision"],
                    "recall": metrics["recall"],
                    **interval_fields,
                }
            )
    return rows


def two_level_comparison(
    candidate: dict,
    baseline: dict,
    y_test: np.ndarray,
    groups: list[np.ndarray],
    metrics: list[str],
    resamples: int,
    seed: int,
    confidence: float,
) -> list[dict]:
    rng = np.random.default_rng(seed)
    distributions = {metric: [] for metric in metrics}
    candidate_seed_ids = [int(seed_id) for seed_id in candidate["seed_ids"]]
    baseline_seed_ids = [int(seed_id) for seed_id in baseline["seed_ids"]]
    if len(candidate_seed_ids) != len(set(candidate_seed_ids)) or len(baseline_seed_ids) != len(
        set(baseline_seed_ids)
    ):
        raise ValueError("Paired comparison received duplicate seed IDs")
    paired_by_seed_id = set(candidate_seed_ids) == set(baseline_seed_ids)
    if paired_by_seed_id:
        paired_seed_ids = sorted(candidate_seed_ids)
        candidate_lookup = {seed_id: idx for idx, seed_id in enumerate(candidate_seed_ids)}
        baseline_lookup = {seed_id: idx for idx, seed_id in enumerate(baseline_seed_ids)}
        candidate_seed_scores = candidate["seed_scores"][
            [candidate_lookup[seed_id] for seed_id in paired_seed_ids]
        ]
        baseline_seed_scores = baseline["seed_scores"][
            [baseline_lookup[seed_id] for seed_id in paired_seed_ids]
        ]
    else:
        paired_seed_ids = []
        candidate_seed_scores = candidate["seed_scores"]
        baseline_seed_scores = baseline["seed_scores"]
    for _ in range(resamples):
        indices = sampled_component_indices(groups, rng)
        y_sample = y_test[indices]
        if len(set(y_sample.tolist())) < 2:
            continue
        if paired_by_seed_id:
            shared_seed_indices = rng.integers(
                0,
                candidate_seed_scores.shape[0],
                size=candidate_seed_scores.shape[0],
            )
            candidate_seed_indices = shared_seed_indices
            baseline_seed_indices = shared_seed_indices
        else:
            candidate_seed_indices = rng.integers(
                0, candidate_seed_scores.shape[0], size=candidate_seed_scores.shape[0]
            )
            baseline_seed_indices = rng.integers(
                0, baseline_seed_scores.shape[0], size=baseline_seed_scores.shape[0]
            )
        candidate_scores = candidate_seed_scores[candidate_seed_indices].mean(axis=0)[indices]
        baseline_scores = baseline_seed_scores[baseline_seed_indices].mean(axis=0)[indices]
        for metric in metrics:
            cand_value = metric_value(metric, y_sample, candidate_scores, candidate["threshold"])
            base_value = metric_value(metric, y_sample, baseline_scores, baseline["threshold"])
            if cand_value is not None and base_value is not None:
                distributions[metric].append(cand_value - base_value)
    rows = []
    for metric in metrics:
        candidate_point = metric_value(metric, y_test, candidate["scores"], candidate["threshold"])
        baseline_point = metric_value(metric, y_test, baseline["scores"], baseline["threshold"])
        values = distributions[metric]
        low, high = percentile_interval(values, confidence)
        positive_seed_count = 0
        if metric == "average_precision":
            if paired_by_seed_id:
                for candidate_scores, baseline_scores in zip(
                    candidate_seed_scores, baseline_seed_scores, strict=True
                ):
                    candidate_ap = step7.average_precision_score(y_test, candidate_scores)
                    baseline_ap = step7.average_precision_score(y_test, baseline_scores)
                    if candidate_ap is not None and baseline_ap is not None and candidate_ap > baseline_ap:
                        positive_seed_count += 1
            else:
                baseline_ap = step7.average_precision_score(y_test, baseline["scores"])
                if baseline_ap is not None:
                    positive_seed_count = sum(
                        1
                        for scores in candidate["seed_scores"]
                        if (
                            (candidate_ap := step7.average_precision_score(y_test, scores))
                            is not None
                            and candidate_ap > baseline_ap
                        )
                    )
        rows.append(
            {
                "metric": metric,
                "candidate_value": candidate_point,
                "baseline_value": baseline_point,
                "difference": None
                if candidate_point is None or baseline_point is None
                else round(candidate_point - baseline_point, 8),
                "ci_low": None if low is None else round(low, 8),
                "ci_high": None if high is None else round(high, 8),
                "p_value_raw": None,
                "permutation_p_value_raw": None,
                "p_value_method": "not_computed_for_supplemental_bootstrap",
                "valid_permutations": 0,
                "permutation_extreme_count": None,
                "valid_bootstrap_resamples": len(values),
                "candidate_positive_seed_count": positive_seed_count if metric == "average_precision" else None,
                "candidate_seed_count": int(candidate["seed_scores"].shape[0]),
                "baseline_seed_count": int(baseline["seed_scores"].shape[0]),
                "paired_by_seed_id": paired_by_seed_id,
                "paired_seed_ids": ";".join(str(seed_id) for seed_id in paired_seed_ids)
                if paired_by_seed_id
                else "",
                "analysis_mode": SUPPLEMENTAL_ANALYSIS_MODE,
                "bootstrap_mode": "two_level_seed_and_split_component_paired_when_seed_ids_match",
            }
        )
    return rows


def paired_grouped_comparison(
    candidate: dict,
    baseline: dict,
    y_test: np.ndarray,
    groups: list[np.ndarray],
    metrics: list[str],
    resamples: int,
    seed: int,
    confidence: float,
    num_permutations: int | None = None,
    randomization_seed: int | None = None,
) -> list[dict]:
    """Primary paired comparison: freeze each seed-mean scorer, resample components only."""
    rng = np.random.default_rng(seed)
    distributions = {metric: [] for metric in metrics}
    for _ in range(resamples):
        indices = sampled_component_indices(groups, rng)
        y_sample = y_test[indices]
        if len(set(y_sample.tolist())) < 2:
            continue
        candidate_scores = candidate["scores"][indices]
        baseline_scores = baseline["scores"][indices]
        for metric in metrics:
            candidate_value = metric_value(
                metric, y_sample, candidate_scores, candidate["threshold"]
            )
            baseline_value = metric_value(
                metric, y_sample, baseline_scores, baseline["threshold"]
            )
            if candidate_value is not None and baseline_value is not None:
                distributions[metric].append(candidate_value - baseline_value)

    candidate_seed_ids = [int(value) for value in candidate["seed_ids"]]
    baseline_seed_ids = [int(value) for value in baseline["seed_ids"]]
    paired_by_seed_id = set(candidate_seed_ids) == set(baseline_seed_ids)
    positive_seed_count = 0
    paired_seed_ids: list[int] = []
    if paired_by_seed_id:
        paired_seed_ids = sorted(candidate_seed_ids)
        candidate_lookup = {seed_id: idx for idx, seed_id in enumerate(candidate_seed_ids)}
        baseline_lookup = {seed_id: idx for idx, seed_id in enumerate(baseline_seed_ids)}
        for seed_id in paired_seed_ids:
            candidate_ap = step7.average_precision_score(
                y_test, candidate["seed_scores"][candidate_lookup[seed_id]]
            )
            baseline_ap = step7.average_precision_score(
                y_test, baseline["seed_scores"][baseline_lookup[seed_id]]
            )
            if candidate_ap is not None and baseline_ap is not None and candidate_ap > baseline_ap:
                positive_seed_count += 1
    else:
        baseline_ap = step7.average_precision_score(y_test, baseline["scores"])
        if baseline_ap is not None:
            positive_seed_count = sum(
                1
                for scores in candidate["seed_scores"]
                if (
                    (candidate_ap := step7.average_precision_score(y_test, scores)) is not None
                    and candidate_ap > baseline_ap
                )
            )

    rows = []
    permutation_count = resamples if num_permutations is None else int(num_permutations)
    permutation_seed = seed + 7000003 if randomization_seed is None else int(randomization_seed)
    for metric_offset, metric in enumerate(metrics):
        candidate_point = metric_value(metric, y_test, candidate["scores"], candidate["threshold"])
        baseline_point = metric_value(metric, y_test, baseline["scores"], baseline["threshold"])
        values = distributions[metric]
        low, high = percentile_interval(values, confidence)
        randomization = paired_component_randomization_test(
            candidate,
            baseline,
            y_test,
            groups,
            metric,
            permutation_count,
            permutation_seed + metric_offset * 104729,
        )
        rows.append(
            {
                "metric": metric,
                "candidate_value": candidate_point,
                "baseline_value": baseline_point,
                "difference": None
                if candidate_point is None or baseline_point is None
                else round(candidate_point - baseline_point, 8),
                "ci_low": None if low is None else round(low, 8),
                "ci_high": None if high is None else round(high, 8),
                "p_value_raw": None
                if randomization["p_value"] is None
                else round(float(randomization["p_value"]), 8),
                "permutation_p_value_raw": None
                if randomization["p_value"] is None
                else round(float(randomization["p_value"]), 8),
                "p_value_method": PERMUTATION_P_VALUE_METHOD,
                "valid_permutations": int(randomization["valid_permutations"]),
                "permutation_extreme_count": int(randomization["extreme_count"]),
                "valid_bootstrap_resamples": len(values),
                "candidate_positive_seed_count": (
                    positive_seed_count if metric == "average_precision" else None
                ),
                "candidate_seed_count": int(candidate["seed_scores"].shape[0]),
                "baseline_seed_count": int(baseline["seed_scores"].shape[0]),
                "paired_by_seed_id": paired_by_seed_id,
                "paired_seed_ids": ";".join(str(seed_id) for seed_id in paired_seed_ids)
                if paired_by_seed_id
                else "",
                "analysis_mode": PRIMARY_ANALYSIS_MODE,
                "bootstrap_mode": "paired_split_component_fixed_seed_mean",
            }
        )
    return rows


def holm_adjust(rows: list[dict]) -> None:
    """Holm-adjust only valid paired component-randomization p-values."""
    families = sorted(
        {
            (row.get("analysis_mode"), row["metric"])
            for row in rows
            if row.get("p_value_method") == PERMUTATION_P_VALUE_METHOD
        }
    )
    for analysis_mode, metric in families:
        eligible = [
            row
            for row in rows
            if row["analysis_mode"] == analysis_mode
            and row["metric"] == metric
            and row.get("p_value_method") == PERMUTATION_P_VALUE_METHOD
            and row.get("permutation_p_value_raw") is not None
        ]
        eligible.sort(key=lambda row: float(row["permutation_p_value_raw"]))
        running = 0.0
        count = len(eligible)
        for rank, row in enumerate(eligible):
            adjusted = min(
                1.0,
                (count - rank) * float(row["permutation_p_value_raw"]),
            )
            running = max(running, adjusted)
            row["p_value_holm"] = round(running, 8)
            row["p_value_holm_source"] = "permutation_p_value_raw"
            row["holm_family"] = f"{analysis_mode}|{metric}"
        for row in rows:
            row.setdefault("p_value_holm", None)
            row.setdefault("p_value_holm_source", None)
            row.setdefault("holm_family", None)


def evaluate_promotion(comparison_rows: list[dict], policy: dict) -> dict:
    promotion_pairs = {
        (row["baseline_model_id"], row.get("evaluation_scope", "all_test")): row
        for row in comparison_rows
        if row["candidate_model_id"] == "step15_v6_final_selected"
        and row["metric"] == policy["promotion_rule"]["primary_metric"]
        and row["analysis_mode"] == PRIMARY_ANALYSIS_MODE
    }
    required_baselines = policy["promotion_rule"]["required_positive_ci_against"]
    required_ci_scopes = policy["promotion_rule"]["required_positive_ci_evaluation_scopes"]
    required_holm_scopes = policy["promotion_rule"]["required_holm_evaluation_scopes"]
    ci_pass = all(
        (baseline, scope) in promotion_pairs
        and promotion_pairs[(baseline, scope)]["ci_low"] is not None
        and float(promotion_pairs[(baseline, scope)]["ci_low"]) > 0.0
        for baseline in required_baselines
        for scope in required_ci_scopes
    )
    holm_alpha = float(policy["promotion_rule"]["holm_adjusted_alpha"])
    holm_pass = all(
        (baseline, scope) in promotion_pairs
        and promotion_pairs[(baseline, scope)].get("p_value_method")
        == PERMUTATION_P_VALUE_METHOD
        and promotion_pairs[(baseline, scope)].get("p_value_holm_source")
        == "permutation_p_value_raw"
        and promotion_pairs[(baseline, scope)].get("p_value_holm") is not None
        and float(promotion_pairs[(baseline, scope)]["p_value_holm"]) <= holm_alpha
        for baseline in required_baselines
        for scope in required_holm_scopes
    )
    required_seed_count = int(policy["promotion_rule"]["total_seed_count"])
    paired_seed_pass = all(
        bool(promotion_pairs.get((baseline, "all_test"), {}).get("paired_by_seed_id"))
        and int(promotion_pairs[(baseline, "all_test")].get("candidate_seed_count") or 0)
        == required_seed_count
        and int(promotion_pairs[(baseline, "all_test")].get("baseline_seed_count") or 0)
        == required_seed_count
        for baseline in required_baselines
    )
    positive_seed_counts = {
        baseline: promotion_pairs.get((baseline, "all_test"), {}).get(
            "candidate_positive_seed_count"
        )
        for baseline in required_baselines
    }
    seed_pass = all(
        int(positive_seed_counts.get(baseline) or 0)
        >= int(policy["promotion_rule"]["minimum_positive_seed_count"])
        for baseline in required_baselines
    )
    return {
        "eligible": bool(ci_pass and holm_pass and paired_seed_pass and seed_pass),
        "claim_scope": policy["promotion_rule"]["claim_scope"],
        "positive_ci_against_required_baselines": ci_pass,
        "required_positive_ci_evaluation_scopes": required_ci_scopes,
        "holm_adjusted_significance_against_required_baselines": holm_pass,
        "required_holm_evaluation_scopes": required_holm_scopes,
        "holm_p_value_source": "paired_component_randomization_test_only",
        "holm_adjusted_alpha": holm_alpha,
        "paired_ten_seed_comparisons_against_required_baselines": paired_seed_pass,
        "positive_seed_counts_against_required_baselines": positive_seed_counts,
        "minimum_positive_seed_count": policy["promotion_rule"]["minimum_positive_seed_count"],
        "rule": policy["promotion_rule"],
    }


def evaluate_method_claims(comparison_rows: list[dict]) -> dict:
    primary_ap = {
        row["comparison_id"]: row
        for row in comparison_rows
        if row["metric"] == "average_precision"
        and row["analysis_mode"] == PRIMARY_ANALYSIS_MODE
        and row.get("evaluation_scope", "all_test") == "all_test"
    }

    def passes(comparison_ids: list[str]) -> bool:
        return all(
            comparison_id in primary_ap
            and primary_ap[comparison_id].get("ci_low") is not None
            and float(primary_ap[comparison_id]["ci_low"]) > 0.0
            for comparison_id in comparison_ids
        )

    return {
        "hard_negative_weighting": {
            "required_comparisons": ["m1_vs_m0"],
            "claim_allowed": passes(["m1_vs_m0"]),
        },
        "domain_balance": {
            "required_comparisons": ["m2_vs_m1"],
            "claim_allowed": passes(["m2_vs_m1"]),
        },
        "curriculum": {
            "required_comparisons": ["m3_vs_m2", "m3_curriculum_vs_m2b_matched_budget"],
            "claim_allowed": passes(
                ["m3_vs_m2", "m3_curriculum_vs_m2b_matched_budget"]
            ),
        },
        "positive_pair_mixup": {
            "required_comparisons": ["m4_vs_m3", "m4_mixup_vs_m4c_matched_continuation"],
            "claim_allowed": passes(
                ["m4_vs_m3", "m4_mixup_vs_m4c_matched_continuation"]
            ),
        },
        "auxiliary_evidence_head": {
            "required_comparisons": ["m5_vs_m3"],
            "claim_allowed": passes(["m5_vs_m3"]),
        },
        "rule": "A contribution is claimable only when every preregistered and matched-control AP CI lower bound is above zero.",
    }


def main() -> None:
    args = parse_args()
    policy_path = resolve(args.policy)
    policy = step7.load_json(policy_path)
    if args.print_boundary_hashes:
        labels = load_csv(resolve(policy["inputs"]["labels"]))
        reaudit = load_csv(resolve(policy["inputs"]["step16f_positive_reaudit"]))
        expected = policy["fixed_test"]
        test_rows = eligible_split_rows(labels, expected["split_name"])
        positive_uids = {
            row["pair_uid"] for row in test_rows if row["review_label"] == "positive"
        }
        tier_rows = [
            row
            for row in reaudit
            if row.get("split_name") == "test" and row.get("pair_uid") in positive_uids
        ]
        print(
            json.dumps(
                {
                    "canonical_sha256": canonical_rows_sha256(
                        test_rows,
                        [str(field) for field in expected["canonical_fields"]],
                    ),
                    "step16f_tier_canonical_sha256": canonical_rows_sha256(
                        tier_rows,
                        [str(field) for field in expected["step16f_tier_canonical_fields"]],
                    ),
                    "test_row_count": len(test_rows),
                    "positive_tier_row_count": len(tier_rows),
                },
                indent=2,
            )
        )
        return
    if args.validate_config_only:
        model_ids = [spec["model_id"] for spec in policy["models"]]
        if len(model_ids) != len(set(model_ids)):
            raise SystemExit("Step12-v6 model_id values must be unique")
        known_ids = set(model_ids) | set(policy.get("validation_selected_aliases", {}))
        unknown_comparison_ids = sorted(
            {
                comparison[key]
                for comparison in policy["paired_comparisons"]
                for key in ("candidate", "baseline")
                if comparison[key] not in known_ids
            }
        )
        if unknown_comparison_ids:
            raise SystemExit(f"Step12-v6 comparisons reference unknown models: {unknown_comparison_ids}")
        if policy["metrics"]["metric_semantics_version"] != step7.METRIC_SEMANTICS_VERSION:
            raise SystemExit("Step12-v6 metric semantics do not match Step7 metric implementation")
        for field in ("canonical_sha256", "step16f_tier_canonical_sha256"):
            value = str(policy["fixed_test"].get(field, ""))
            if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
                raise SystemExit(f"Step12-v6 fixed boundary has no valid preregistered {field}")
        if int(policy["bootstrap"]["num_resamples"]) <= 0:
            raise SystemExit("Step12-v6 bootstrap num_resamples must be positive")
        randomization = policy.get("randomization_test") or {}
        if randomization.get("method") != "paired_split_component_score_swap":
            raise SystemExit("Step12-v6 must use paired split-component score-swap randomization")
        if randomization.get("grouping_unit") != policy["fixed_test"]["grouping_unit"]:
            raise SystemExit("Step12-v6 randomization and fixed-test grouping units disagree")
        if int(randomization.get("num_permutations", 0)) <= 0:
            raise SystemExit("Step12-v6 randomization permutation count must be positive")
        expected_seeds = list(range(20260320, 20260330))
        for spec in policy["models"]:
            if spec.get("step9_experiment") and list(spec.get("seeds", [])) != expected_seeds:
                raise SystemExit(f"Step12-v6 Step9 control does not use the ten matched seeds: {spec['model_id']}")
        if int(policy["promotion_rule"].get("total_seed_count", 0)) != 10:
            raise SystemExit("Step12-v6 promotion requires exactly ten paired seeds")
        if not bool(policy["promotion_rule"].get("require_holm_adjusted_significance", False)):
            raise SystemExit("Step12-v6 promotion must require Holm-adjusted significance")
        if not bool(policy["promotion_rule"].get("require_paired_seed_ids_against_strongest_step9", False)):
            raise SystemExit("Step12-v6 promotion must require paired seed IDs")
        aliases = policy.get("validation_selected_aliases") or {}
        required_aliases = {
            "step9_strongest_clean_validation_selected",
            "step15_v6_m5_selected",
            "step15_v6_final_selected",
        }
        if not required_aliases.issubset(aliases):
            raise SystemExit("Step12-v6 is missing a required validation-selected alias")
        for alias_name in sorted(required_aliases):
            alias = aliases[alias_name]
            candidates = [str(value) for value in alias["candidate_model_ids"]]
            tie_order = [str(value) for value in alias["simplicity_tie_break_order"]]
            if set(candidates) != set(tie_order) or len(tie_order) != len(set(tie_order)):
                raise SystemExit(
                    f"Step12-v6 alias has an invalid simplicity tie-break order: {alias_name}"
                )
            if alias.get("selection_split") != "zh_valid":
                raise SystemExit(f"Step12-v6 alias is not selected on zh_valid: {alias_name}")
            if bool(alias.get("test_metrics_used_for_selection", True)):
                raise SystemExit(f"Step12-v6 alias permits test-informed selection: {alias_name}")
        expected_clean_step9_candidates = {
            "step9_e5_lr_l2_100pct_seed_mean",
            "step9_bge_m3_residual_lr_100pct_seed_mean",
            "step9_labse_lr_l2_100pct_seed_mean",
        }
        observed_clean_step9_candidates = set(
            aliases["step9_strongest_clean_validation_selected"]["candidate_model_ids"]
        )
        if observed_clean_step9_candidates != expected_clean_step9_candidates:
            raise SystemExit(
                "Step12-v6 strongest-clean Step9 selection must contain exactly E5 LR/L2, "
                "BGE residual, and LaBSE LR/L2"
            )
        if aliases["step15_v6_m5_selected"]["simplicity_tie_break_order"] != [
            "step15_v6_m5_lambda_0p1",
            "step15_v6_m5_lambda_0p3",
        ]:
            raise SystemExit("Step12-v6 M5 ties must prefer lambda 0.1 over lambda 0.3")
        if aliases["step15_v6_final_selected"]["simplicity_tie_break_order"] != [
            "step15_v6_m3",
            "step15_v6_m4",
            "step15_v6_m5_selected",
        ]:
            raise SystemExit("Step12-v6 final-model ties must prefer M3, then M4, then M5")
        promotion_scopes = set(
            policy["promotion_rule"].get("required_positive_ci_evaluation_scopes", [])
        )
        if "strict_plus_soft_primary_positive_vs_all_negative" not in promotion_scopes:
            raise SystemExit("Step12-v6 promotion must gate on the strict+soft positive slice")
        required_baselines = set(policy["promotion_rule"]["required_positive_ci_against"])
        scoped_comparisons = {
            comparison["baseline"]: set(comparison.get("evaluation_scopes", ["all_test"]))
            for comparison in policy["paired_comparisons"]
            if comparison["candidate"] == "step15_v6_final_selected"
        }
        for baseline in required_baselines:
            if not promotion_scopes.issubset(scoped_comparisons.get(baseline, set())):
                raise SystemExit(
                    f"Step12-v6 promotion scopes are not generated for required baseline: {baseline}"
                )
        if policy["fixed_test"].get("role") != (
            "internal_development_test_not_prospective_final_holdout"
        ):
            raise SystemExit("Step12-v6 fixed boundary must remain explicitly internal-development")
        if policy["inputs"].get("features") != (
            "reports/step15_v6/features/"
            "step7_pair_features.zh_target_strict.inductive_train_reference.csv"
        ):
            raise SystemExit("Step12-v6 must consume the isolated inductive ZH feature file")
        if policy["inputs"].get("expected_step15_v6_policy_version") != (
            "2026-07-12-step15-v6-paper-hardening-v4-inductive-endpoint"
        ):
            raise SystemExit("Step12-v6 is not bound to the Step15-v6 v4 inductive endpoint")
        if policy["inputs"].get("step15_v6_active_manifest") != (
            "reports/step15_v6/manifests/step15_v6_internal_dev_v4_20260712.json"
        ) or policy["inputs"].get("step15_v6_active_manifest_run_id") != (
            "step15-v6-method-audit-v4-inductive-internal-dev-20260712"
        ):
            raise SystemExit("Step12-v6 is not bound to the v4 inductive active manifest")
        target_outputs = output_paths(policy)
        if len(target_outputs) != len(set(target_outputs)):
            raise SystemExit("Step12-v6 output targets must be unique")
        if "pr_auc" not in policy["metrics"]["secondary"]:
            raise SystemExit("Step12-v6 paired secondary metrics must include PR-AUC")
        print(
            json.dumps(
                {
                    "status": "pass",
                    "policy_version": policy["version"],
                    "model_count": len(model_ids),
                    "comparison_count": len(policy["paired_comparisons"]),
                },
                indent=2,
            )
        )
        return
    if not args.validate_inputs_only:
        assert_output_targets_absent(policy)
    input_cfg = policy["inputs"]
    labels_path = resolve(input_cfg["labels"])
    features_path = resolve(input_cfg["features"])
    reaudit_path = resolve(input_cfg["step16f_positive_reaudit"])
    step15_summary_path = resolve(input_cfg["step15_v6_summary"])
    source_only_summary_path = resolve(input_cfg["step15_v6_source_only_summary"])
    active_manifest_path = resolve(input_cfg["step15_v6_active_manifest"])
    output_validation_path = resolve(input_cfg["step15_v6_output_validation"])
    step9_summary_path = resolve(input_cfg["step9_summary"])
    step7_summary_path = resolve(input_cfg["step7_summary"])
    required = [
        policy_path,
        labels_path,
        features_path,
        reaudit_path,
        step15_summary_path,
        source_only_summary_path,
        active_manifest_path,
        output_validation_path,
        step9_summary_path,
        step7_summary_path,
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise SystemExit(f"Missing Step12-v6 inputs: {missing}")
    labels = load_csv(labels_path)
    features = load_csv(features_path)
    reaudit = load_csv(reaudit_path)
    step15_summary = step7.load_json(step15_summary_path)
    source_only_summary = step7.load_json(source_only_summary_path)
    step9_summary = step7.load_json(step9_summary_path)
    step7_summary = step7.load_json(step7_summary_path)
    output_validation = step7.load_json(output_validation_path)
    if output_validation.get("status") != "pass":
        raise ValueError("Step15-v6 output validation did not pass")
    step15_policy_payload = step7.load_json(STEP15_POLICY)
    if step15_policy_payload.get("version") != input_cfg.get(
        "expected_step15_v6_policy_version"
    ):
        raise ValueError(
            "Step12-v6 expected Step15 policy version mismatch: "
            f"expected={input_cfg.get('expected_step15_v6_policy_version')} "
            f"observed={step15_policy_payload.get('version')}"
        )
    if step15_summary.get("policy_version") != step15_policy_payload.get("version"):
        raise ValueError("Step15 v6 summary and current policy version disagree")
    if source_only_summary.get("policy_version") != step15_policy_payload.get("version"):
        raise ValueError("Step15 source-only summary and current policy version disagree")
    validate_embedded_input_manifest(step15_summary, "Step15 v6 summary")
    validate_embedded_input_manifest(source_only_summary, "Step15 source-only summary")
    validate_step9_context_fingerprint(step9_summary)
    active_manifest, active_manifest_index = load_verified_active_manifest(
        active_manifest_path,
        str(input_cfg["step15_v6_active_manifest_run_id"]),
        str(step15_policy_payload["version"]),
    )
    require_manifest_path(step15_summary_path, active_manifest_index)
    require_manifest_path(source_only_summary_path, active_manifest_index)
    require_manifest_path(step9_summary_path, active_manifest_index)
    require_manifest_path(step7_summary_path, active_manifest_index)
    require_manifest_path(output_validation_path, active_manifest_index)
    validate_output_validation_binding(output_validation, active_manifest_index)
    require_manifest_path(STEP15_POLICY, active_manifest_index)
    expected_step9_experiments = sorted(
        str(spec["step9_experiment"])
        for spec in policy["models"]
        if spec.get("step9_experiment")
    )
    expected_step9_seeds = sorted(
        {
            int(seed)
            for spec in policy["models"]
            if spec.get("step9_experiment")
            for seed in spec["seeds"]
        }
    )
    manifest_step9 = active_manifest.get("step9_selection") or {}
    if sorted(manifest_step9.get("experiments") or []) != expected_step9_experiments:
        raise ValueError("Step15 active manifest does not freeze the exact Step9 experiment allow-list")
    if sorted(int(seed) for seed in manifest_step9.get("seeds") or []) != expected_step9_seeds:
        raise ValueError("Step15 active manifest does not freeze the exact Step9 seed allow-list")
    if manifest_step9.get("ratio_token") != "100pct":
        raise ValueError("Step15 active manifest does not freeze the required Step9 100pct controls")
    test_rows = eligible_split_rows(labels, policy["fixed_test"]["split_name"])
    y_test = labels_from_rows(test_rows)
    expected = policy["fixed_test"]
    observed = (len(test_rows), int(y_test.sum()), int(len(y_test) - y_test.sum()))
    expected_tuple = (
        int(expected["expected_row_count"]),
        int(expected["expected_positive_count"]),
        int(expected["expected_negative_count"]),
    )
    if observed != expected_tuple:
        raise ValueError(f"Fixed test boundary mismatch: expected={expected_tuple}, observed={observed}")
    canonical_fields = [str(field) for field in expected["canonical_fields"]]
    observed_boundary_sha = canonical_rows_sha256(test_rows, canonical_fields)
    if observed_boundary_sha != expected["canonical_sha256"]:
        raise ValueError(
            "Fixed test boundary SHA-256 mismatch: "
            f"expected={expected['canonical_sha256']} observed={observed_boundary_sha}"
        )
    positive_test_uids = {
        row["pair_uid"] for row in test_rows if row["review_label"] == "positive"
    }
    test_tier_rows = [
        row
        for row in reaudit
        if row.get("split_name") == "test" and row.get("pair_uid") in positive_test_uids
    ]
    tier_fields = [str(field) for field in expected["step16f_tier_canonical_fields"]]
    observed_tier_sha = canonical_rows_sha256(test_tier_rows, tier_fields)
    if observed_tier_sha != expected["step16f_tier_canonical_sha256"]:
        raise ValueError(
            "Step16F fixed-test tier mapping SHA-256 mismatch: "
            f"expected={expected['step16f_tier_canonical_sha256']} observed={observed_tier_sha}"
        )
    masks, slice_counts = build_positive_slice_masks(test_rows, reaudit, policy)
    input_paths = {
        *required,
        Path(__file__).resolve(),
        Path(step7.__file__).resolve(),
        STEP15_POLICY,
    }
    models, selection = load_models(
        policy,
        labels,
        features,
        step15_summary,
        source_only_summary,
        step9_summary,
        step7_summary,
        active_manifest_index,
        input_paths,
    )
    if args.validate_inputs_only:
        print(
            json.dumps(
                {
                    "status": "pass",
                    "test_counts": observed,
                    "positive_slice_counts": slice_counts,
                    "models": sorted(models),
                    "selection": selection,
                },
                indent=2,
            )
        )
        return

    canonical_resamples = int(policy["bootstrap"]["num_resamples"])
    if args.resamples is not None and int(args.resamples) != canonical_resamples:
        raise ValueError(
            "Step12-v6 canonical output paths require the preregistered bootstrap count; "
            f"expected={canonical_resamples} requested={args.resamples}"
        )
    if canonical_resamples <= 0:
        raise ValueError("Step12-v6 bootstrap resamples must be positive")
    resamples = canonical_resamples
    groups = component_groups(test_rows)
    model_rows = model_metric_rows(models, test_rows, y_test, groups, policy, resamples)
    slice_rows = evidence_slice_rows(models, test_rows, y_test, masks, policy, resamples)
    comparison_rows = []
    comparison_metrics = list(
        dict.fromkeys([policy["metrics"]["primary"], *policy["metrics"]["secondary"]])
    )
    randomization_cfg = policy["randomization_test"]
    for offset, comparison in enumerate(policy["paired_comparisons"]):
        candidate_id = comparison["candidate"]
        baseline_id = comparison["baseline"]
        evaluation_scopes = comparison.get("evaluation_scopes", ["all_test"])
        for scope_offset, evaluation_scope in enumerate(evaluation_scopes):
            if evaluation_scope not in masks:
                raise ValueError(
                    f"Unknown Step12-v6 comparison evaluation scope: {evaluation_scope}"
                )
            mask = masks[evaluation_scope]
            scope_rows = [
                row for row, keep in zip(test_rows, mask, strict=True) if keep
            ]
            scope_y = y_test[mask]
            scope_groups = component_groups(scope_rows)
            candidate = masked_model(models[candidate_id], mask)
            baseline = masked_model(models[baseline_id], mask)
            comparison_seed = (
                int(policy["bootstrap"]["random_seed"])
                + offset * 1009
                + scope_offset * 100003
            )
            primary_rows = paired_grouped_comparison(
                candidate,
                baseline,
                scope_y,
                scope_groups,
                comparison_metrics,
                resamples,
                comparison_seed,
                float(policy["bootstrap"]["confidence_level"]),
                num_permutations=int(randomization_cfg["num_permutations"]),
                randomization_seed=(
                    int(randomization_cfg["random_seed"])
                    + offset * 1009
                    + scope_offset * 100003
                ),
            )
            supplemental_rows = two_level_comparison(
                candidate,
                baseline,
                scope_y,
                scope_groups,
                comparison_metrics,
                resamples,
                comparison_seed + 500003,
                float(policy["bootstrap"]["confidence_level"]),
            )
            rows = [*primary_rows, *supplemental_rows]
            for row in rows:
                row.update(
                    {
                        "comparison_id": comparison["comparison_id"],
                        "candidate_model_id": candidate_id,
                        "baseline_model_id": baseline_id,
                        "evaluation_scope": evaluation_scope,
                        "evaluation_row_count": int(mask.sum()),
                        "evaluation_positive_count": int(scope_y.sum()),
                        "evaluation_negative_count": int(len(scope_y) - scope_y.sum()),
                    }
                )
            comparison_rows.extend(rows)
    holm_adjust(comparison_rows)

    promotion = evaluate_promotion(comparison_rows, policy)
    method_claims = evaluate_method_claims(comparison_rows)

    outputs = policy["outputs"]
    metrics_path = resolve(outputs["model_metrics_csv"])
    slices_path = resolve(outputs["slice_metrics_csv"])
    comparisons_path = resolve(outputs["paired_comparisons_csv"])
    summary_path = resolve(outputs["summary_json"])
    completion_manifest_path = resolve(outputs["completion_manifest_json"])
    write_csv(metrics_path, model_rows)
    write_csv(slices_path, slice_rows)
    write_csv(comparisons_path, comparison_rows)
    input_manifest = [
        {
            "path": str(path.relative_to(ROOT)),
            "sha256": file_sha256(path),
            "size_bytes": path.stat().st_size,
        }
        for path in sorted(input_paths, key=lambda value: str(value))
    ]
    summary = {
        "step": "step12_v6_statistical_robustness_audit",
        "policy": str(policy_path.relative_to(ROOT)),
        "policy_version": policy["version"],
        "metric_semantics_version": policy["metrics"]["metric_semantics_version"],
        "pr_auc_definition": PR_AUC_DEFINITION,
        "fixed_test_role": policy["fixed_test"]["role"],
        "fixed_test_counts": {
            "rows": observed[0],
            "positive": observed[1],
            "negative": observed[2],
            "component_count": len(groups),
        },
        "fixed_test_canonical_sha256": observed_boundary_sha,
        "step16f_tier_canonical_sha256": observed_tier_sha,
        "positive_slice_counts": slice_counts,
        "selection": selection,
        "promotion": promotion,
        "claim_scope": "internal_development_only_not_prospective_final_holdout",
        "method_claims": method_claims,
        "bootstrap_resamples": resamples,
        "randomization_test": policy["randomization_test"],
        "input_manifest": input_manifest,
        "outputs": {
            "summary_json": str(summary_path.relative_to(ROOT)),
            "model_metrics_csv": str(metrics_path.relative_to(ROOT)),
            "slice_metrics_csv": str(slices_path.relative_to(ROOT)),
            "paired_comparisons_csv": str(comparisons_path.relative_to(ROOT)),
            "completion_manifest_json": str(completion_manifest_path.relative_to(ROOT)),
        },
        "producer_context": {
            "git_commit": current_git_commit(),
            "step12_script": {
                "path": str(Path(__file__).resolve().relative_to(ROOT)),
                "sha256": file_sha256(Path(__file__).resolve()),
            },
            "step7_metric_implementation": {
                "path": str(Path(step7.__file__).resolve().relative_to(ROOT)),
                "sha256": file_sha256(Path(step7.__file__).resolve()),
            },
        },
    }
    write_json(summary_path, summary)
    completion_files = [
        *input_manifest,
        *[
            {
                "path": str(path.relative_to(ROOT)),
                "sha256": file_sha256(path),
                "size_bytes": path.stat().st_size,
            }
            for path in (summary_path, metrics_path, slices_path, comparisons_path)
        ],
    ]
    completion_core = {
        "step": "step12_v6_completion_manifest",
        "policy_version": policy["version"],
        "execution_git_commit": current_git_commit(),
        "files": sorted(completion_files, key=lambda item: item["path"]),
    }
    completion_core["manifest_sha256"] = hashlib.sha256(
        json.dumps(
            completion_core,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    write_json(completion_manifest_path, completion_core)
    print(
        json.dumps(
            {
                "summary": str(summary_path.relative_to(ROOT)),
                "completion_manifest": str(completion_manifest_path.relative_to(ROOT)),
                "promotion": promotion,
                "method_claims": method_claims,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
