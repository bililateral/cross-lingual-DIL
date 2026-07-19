#!/usr/bin/env python3
"""Audit Step27 synthetic rows before residual-model training."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

import step27_train_residual_models as step27


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_POLICY = ROOT / "schema" / "step27_english_pretrained_synthetic_adaptation_policy.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", default=str(DEFAULT_POLICY))
    parser.add_argument("--validate-config-only", action="store_true")
    parser.add_argument("--validate-inputs-only", action="store_true")
    return parser.parse_args()


def synthetic_track(row: dict) -> str:
    return str(row.get("synthetic_track") or row.get("track") or "primary")


def generation_seed(row: dict) -> int:
    return int(row.get("generation_seed", row.get("seed", -1)))


def recipe_name(row: dict) -> str:
    value = str(row.get("recipe") or row.get("transform_recipe") or row.get("transform") or "")
    if not value:
        raise ValueError(f"Step27 synthetic row has no recipe: {step27.row_uid(row)}")
    return value


def one_hot_recipe(rows: list[dict]) -> tuple[np.ndarray, list[str]]:
    recipes = sorted({recipe_name(row) for row in rows})
    feature_names = [f"recipe={value}" for value in recipes]
    matrix = np.zeros((len(rows), len(recipes) + 3), dtype=float)
    for index, row in enumerate(rows):
        matrix[index, recipes.index(recipe_name(row))] = 1.0
        matrix[index, len(recipes)] = float(row.get("changed_side_count", 0) or 0)
        matrix[index, len(recipes) + 1] = float(row.get("synthetic_segment_count", 0) or 0)
        matrix[index, len(recipes) + 2] = float(row.get("synthetic_text_length", 0) or 0)
    return matrix, feature_names + ["changed_side_count", "synthetic_segment_count", "synthetic_text_length"]


def grouped_oof_distinguishability(
    x: np.ndarray,
    y: np.ndarray,
    groups: list[str],
    fold_by_component: dict[str, int],
    fold_count: int,
) -> dict:
    if len(x) == 0 or set(y.tolist()) != {0, 1}:
        return {"status": "not_estimable", "roc_auc": None, "row_count": len(x)}
    scores = np.full(len(y), np.nan, dtype=float)
    for held_fold in range(fold_count):
        held = np.asarray([fold_by_component[group] == held_fold for group in groups], dtype=bool)
        train = ~held
        if not held.any() or set(y[train].tolist()) != {0, 1}:
            return {
                "status": "not_estimable",
                "reason": f"fold_{held_fold}_empty_or_single_class",
                "row_count": len(x),
            }
        artifact = step27.fit_offset_logistic(
            x[train],
            y[train].astype(float),
            np.ones(int(train.sum()), dtype=float),
            np.zeros(int(train.sum()), dtype=float),
            l2_penalty=10.0,
            max_iter=400,
            tolerance=1e-8,
        )
        if not artifact.get("solver_converged", False):
            return {
                "status": "not_estimable",
                "reason": f"fold_{held_fold}_solver_not_converged",
                "row_count": len(x),
            }
        scores[held] = step27.predict_offset_logistic(
            artifact,
            x[held],
            np.zeros(int(held.sum()), dtype=float),
        )
    if not np.isfinite(scores).all():
        return {"status": "not_estimable", "reason": "incomplete_oof_scores", "row_count": len(x)}
    return {
        "status": "estimated",
        "row_count": len(x),
        "component_count": len(set(groups)),
        "roc_auc": step27.roc_auc(y, scores),
        "average_precision": step27.average_precision(y, scores),
    }


def assert_nonzero_synthetic_feature_displacement(
    parents_by_uid: dict[str, dict],
    synthetic_rows: list[dict],
    feature_names: list[str],
    *,
    atol: float = 1e-12,
) -> dict:
    changed_value_count = 0
    changed_feature_names: set[str] = set()
    maximum_absolute_displacement = 0.0
    compared_value_count = 0
    for row in synthetic_rows:
        parent_uid = str(row.get("parent_pair_uid") or row.get("source_pair_uid") or "")
        parent = parents_by_uid.get(parent_uid)
        if parent is None:
            raise ValueError(f"Step27 feature displacement parent is missing: {parent_uid}")
        for name in feature_names:
            try:
                delta = abs(float(row[name]) - float(parent[name]))
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(
                    f"Step27 feature displacement value is invalid: {parent_uid}:{name}"
                ) from exc
            compared_value_count += 1
            maximum_absolute_displacement = max(maximum_absolute_displacement, delta)
            if delta > atol:
                changed_value_count += 1
                changed_feature_names.add(name)
    if changed_value_count == 0:
        raise ValueError("Step27 synthetic feature displacement is zero for every input feature")
    return {
        "row_count": len(synthetic_rows),
        "feature_count": len(feature_names),
        "compared_value_count": compared_value_count,
        "changed_value_count": changed_value_count,
        "changed_feature_count": len(changed_feature_names),
        "changed_feature_names": sorted(changed_feature_names),
        "maximum_absolute_displacement": maximum_absolute_displacement,
        "atol": atol,
    }


def main() -> None:
    args = parse_args()
    policy_path = step27.resolve(args.policy)
    policy = step27.load_json(policy_path)
    cfg = step27.validate_policy(policy, policy_path)
    if args.validate_config_only:
        print(json.dumps({"status": "pass", "run_id": cfg["run_id"]}, indent=2))
        return

    root = step27.outputs_root(policy)
    real_rows, primary_rows, feature_paths = step27.materialize_feature_tables(
        policy, cfg, real_split="train"
    )
    sensitivity_rows, sensitivity_feature_paths = step27.load_sensitivity_feature_tables(
        policy, cfg
    )
    synthetic_rows = primary_rows + sensitivity_rows
    all_feature_paths = feature_paths + sensitivity_feature_paths
    duplication_feature_paths = [
        step27.common.track_root(policy, seed, track)
        / "pair_features"
        / "equal_weight_duplication_pair_features.csv"
        for seed in cfg["seeds"]
        for track in ("primary", "silver_sensitivity")
    ]
    synthetic_clean_profile_paths = [
        step27.common.track_root(policy, seed, track)
        / "embeddings"
        / "clean_profiles.jsonl"
        for seed in cfg["seeds"]
        for track in ("primary", "silver_sensitivity")
    ]
    seller_profiles_path = step27.common.policy_input(
        policy, "seller_profiles", "zh_seller_profiles"
    )
    identity_signals_path = step27.common.policy_input(
        policy, "item_identity_signals", "zh_item_identity_signals"
    )
    manifest = step27.input_manifest(
        policy_path,
        [
            policy_path,
            Path(__file__).resolve(),
            Path(step27.__file__).resolve(),
            Path(step27.common.__file__).resolve(),
            step27.common.parent_root(policy) / "manifest.json",
            *(step27.common.seed_root(policy, seed) / "generation_manifest.json" for seed in cfg["seeds"]),
            *all_feature_paths,
            *duplication_feature_paths,
            *synthetic_clean_profile_paths,
            seller_profiles_path,
            identity_signals_path,
        ],
        cfg["run_id"],
    )
    names = step27.feature_names(policy, real_rows)
    source_name = step27.source_feature_name(policy)
    step27.validate_rows(
        real_rows,
        primary_rows,
        names,
        cfg["fold_count"],
        cfg["seeds"],
        required_real_splits=("train",),
    )
    step27.validate_rows(
        real_rows,
        sensitivity_rows,
        names,
        cfg["fold_count"],
        cfg["seeds"],
        required_real_splits=("train",),
    )

    audit_cfg = dict(policy.get("synthetic_audit") or policy.get("audit") or policy.get("shortcut_and_leakage_audits") or {})
    weighting_cfg = dict(policy.get("weighting") or {})
    parent_cap = float(
        audit_cfg.get(
            "maximum_total_child_weight_relative_to_parent",
            weighting_cfg.get("maximum_child_total_weight_relative_to_parent", 0.5),
        )
    )
    total_cap = float(
        audit_cfg.get(
            "maximum_total_synthetic_weight_relative_to_real_train",
            weighting_cfg.get("maximum_total_primary_synthetic_effective_weight_fraction_of_real_chinese_train", 0.25),
        )
    )
    recipe_auc_cap = float(audit_cfg.get("maximum_recipe_label_roc_auc", audit_cfg.get("recipe_label_predictability_auc_maximum", 0.60)))
    synthetic_real_auc_cap = float(audit_cfg.get("maximum_synthetic_real_roc_auc", audit_cfg.get("synthetic_vs_real_predictability_auc_maximum", 0.70)))
    max_rows = {
        "primary": int(audit_cfg.get("maximum_primary_rows_per_seed", 64)),
        "silver_sensitivity": int(audit_cfg.get("maximum_sensitivity_rows_per_seed", 112)),
    }
    if args.validate_inputs_only:
        print(
            json.dumps(
                {
                    "status": "pass",
                    "input_manifest_sha256": manifest["manifest_sha256"],
                    "real_rows": len(real_rows),
                    "synthetic_rows": len(synthetic_rows),
                },
                indent=2,
            )
        )
        return

    real_train = [row for row in real_rows if row.get("split_name") == "train"]
    real_index = {step27.row_uid(row): row for row in real_train}
    real_train_weight = sum(step27.row_weight(row) for row in real_train)
    fold_by_component = {step27.row_component(row): step27.row_fold(row) for row in real_train}
    violations: list[dict] = []
    group_reports: list[dict] = []
    source_profiles = step27.common.load_profiles_index(seller_profiles_path)
    signal_literals, _ = step27.common.redaction.signal_literals_by_seller(identity_signals_path)
    text_fields = step27.common.text_fields(policy)
    redaction_profile_count = 0
    for clean_path in synthetic_clean_profile_paths:
        for synthetic_profile in step27.common.load_jsonl(clean_path):
            redaction_profile_count += 1
            lineage = synthetic_profile.get("synthetic_lineage") or {}
            parent_uid = str(lineage.get("parent_seller_uid") or "")
            parent_profile = source_profiles.get(parent_uid)
            if parent_profile is None:
                violations.append(
                    {
                        "audit": "synthetic_profile_parent_exists",
                        "severity": "fatal",
                        "synthetic_seller_uid": synthetic_profile.get("seller_uid", ""),
                        "parent_seller_uid": parent_uid,
                    }
                )
                continue
            parent_clean, _ = step27.common.clean_profile_fields(
                parent_profile, text_fields, signal_literals
            )
            literals = step27.common.profile_literals(parent_profile, signal_literals)
            for field in text_fields:
                synthetic_value = str(synthetic_profile.get(field, "") or "")
                try:
                    step27.common.redaction.assert_no_known_identifier_residue(
                        synthetic_value, literals, str(synthetic_profile.get("seller_uid", ""))
                    )
                except ValueError as exc:
                    violations.append(
                        {
                            "audit": "synthetic_identifier_residue",
                            "severity": "fatal",
                            "synthetic_seller_uid": synthetic_profile.get("seller_uid", ""),
                            "parent_seller_uid": parent_uid,
                            "field": field,
                            "detail": str(exc),
                        }
                    )
                expected_segments = Counter(
                    step27.common.normalize_layout(value)
                    for value in step27.common.split_segments(parent_clean.get(field, ""))
                )
                observed_segments = Counter(
                    step27.common.normalize_layout(value)
                    for value in step27.common.split_segments(synthetic_value)
                )
                if observed_segments != expected_segments:
                    violations.append(
                        {
                            "audit": "cross_parent_content_splicing_or_content_loss",
                            "severity": "fatal",
                            "synthetic_seller_uid": synthetic_profile.get("seller_uid", ""),
                            "parent_seller_uid": parent_uid,
                            "field": field,
                        }
                    )

    grouped: dict[tuple[int, str], list[dict]] = defaultdict(list)
    for row in synthetic_rows:
        grouped[(generation_seed(row), synthetic_track(row))].append(row)
    duplication_grouped: dict[tuple[int, str], list[dict]] = {}
    for seed in cfg["seeds"]:
        for track in ("primary", "silver_sensitivity"):
            path = step27.common.track_root(policy, seed, track) / "pair_features" / "equal_weight_duplication_pair_features.csv"
            if not path.is_file():
                raise FileNotFoundError(f"Step27 duplication feature table is missing: {path}")
            duplication_grouped[(seed, track)] = step27.load_csv(path)

    expected_groups = {
        (seed, track)
        for seed in cfg["seeds"]
        for track in sorted({synthetic_track(row) for row in synthetic_rows})
    }
    if set(grouped) != expected_groups:
        violations.append(
            {
                "audit": "seed_track_coverage",
                "severity": "fatal",
                "detail": f"observed={sorted(grouped)} expected={sorted(expected_groups)}",
            }
        )

    for (seed, track), rows in sorted(grouped.items()):
        duplication_rows = duplication_grouped.get((seed, track), [])
        labels = np.asarray([step27.row_label(row) for row in rows], dtype=int)
        components = [step27.row_component(row) for row in rows]
        max_count = max_rows.get(track)
        if max_count is None:
            violations.append(
                {"audit": "unknown_track", "severity": "fatal", "seed": seed, "track": track}
            )
        elif len(rows) > max_count:
            violations.append(
                {
                    "audit": "row_budget",
                    "severity": "fatal",
                    "seed": seed,
                    "track": track,
                    "observed": len(rows),
                    "maximum": max_count,
                }
            )
        elif (
            policy.get("generation", {})
            .get("recipe_contract", {})
            .get("fail_closed_on_no_op", False)
            and len(rows) != max_count
        ):
            violations.append(
                {
                    "audit": "complete_fail_closed_child_budget",
                    "severity": "fatal",
                    "seed": seed,
                    "track": track,
                    "observed": len(rows),
                    "expected": max_count,
                }
            )

        parent_weights: dict[str, float] = defaultdict(float)
        recipe_by_label: dict[int, Counter[str]] = {0: Counter(), 1: Counter()}
        for row in rows:
            parent_uid = str(row.get("parent_pair_uid") or row.get("source_pair_uid") or "")
            parent = real_index.get(parent_uid)
            if parent is None:
                violations.append(
                    {
                        "audit": "lineage",
                        "severity": "fatal",
                        "seed": seed,
                        "track": track,
                        "pair_uid": step27.row_uid(row),
                        "detail": "missing_real_train_parent",
                    }
                )
                continue
            if step27.row_fold(row) != step27.row_fold(parent) or step27.row_component(row) != step27.row_component(parent):
                violations.append(
                    {
                        "audit": "lineage",
                        "severity": "fatal",
                        "seed": seed,
                        "track": track,
                        "pair_uid": step27.row_uid(row),
                        "detail": "child_parent_component_or_fold_mismatch",
                    }
                )
            parent_weights[parent_uid] += step27.row_weight(row)
            recipe_by_label[step27.row_label(row)][recipe_name(row)] += 1
        for parent_uid, child_weight in parent_weights.items():
            allowed = parent_cap * step27.row_weight(real_index[parent_uid])
            if child_weight > allowed + 1e-10:
                violations.append(
                    {
                        "audit": "per_parent_weight_cap",
                        "severity": "fatal",
                        "seed": seed,
                        "track": track,
                        "parent_pair_uid": parent_uid,
                        "observed": child_weight,
                        "maximum": allowed,
                    }
                )
        total_weight = sum(step27.row_weight(row) for row in rows)
        total_ratio = total_weight / real_train_weight
        if total_ratio > total_cap + 1e-10:
            violations.append(
                {
                    "audit": "total_weight_cap",
                    "severity": "fatal",
                    "seed": seed,
                    "track": track,
                    "observed_ratio": total_ratio,
                    "maximum_ratio": total_cap,
                }
            )
        if recipe_by_label[0] != recipe_by_label[1]:
            violations.append(
                {
                    "audit": "matched_recipe_counts",
                    "severity": "fatal",
                    "seed": seed,
                    "track": track,
                    "negative": dict(recipe_by_label[0]),
                    "positive": dict(recipe_by_label[1]),
                }
            )
        synthetic_budget = Counter(
            (
                str(row.get("parent_pair_uid")),
                step27.row_label(row),
                step27.row_component(row),
                step27.row_fold(row),
                round(step27.row_weight(row), 12),
            )
            for row in rows
        )
        duplication_budget = Counter(
            (
                str(row.get("parent_pair_uid")),
                step27.row_label(row),
                step27.row_component(row),
                step27.row_fold(row),
                round(step27.row_weight(row), 12),
            )
            for row in duplication_rows
        )
        if synthetic_budget != duplication_budget:
            violations.append(
                {
                    "audit": "M1_M2_parent_component_fold_label_weight_parity",
                    "severity": "fatal",
                    "seed": seed,
                    "track": track,
                    "synthetic_row_count": len(rows),
                    "duplication_row_count": len(duplication_rows),
                    "detail": "transformed and duplication budget multisets differ",
                }
            )

        displacement_names = [source_name, *names]
        try:
            displacement_report = assert_nonzero_synthetic_feature_displacement(
                real_index,
                rows,
                displacement_names,
                atol=float(audit_cfg.get("minimum_feature_displacement_tolerance", 1e-12)),
            )
        except ValueError as exc:
            displacement_report = {
                "status": "fail",
                "detail": str(exc),
                "row_count": len(rows),
            }
            violations.append(
                {
                    "audit": "synthetic_feature_displacement",
                    "severity": "fatal",
                    "seed": seed,
                    "track": track,
                    "detail": str(exc),
                }
            )
        else:
            displacement_report["status"] = "pass"

        recipe_x, recipe_fields = one_hot_recipe(rows)
        recipe_report = grouped_oof_distinguishability(
            recipe_x, labels, components, fold_by_component, cfg["fold_count"]
        )
        if recipe_report.get("status") != "estimated":
            violations.append(
                {
                    "audit": "recipe_label_distinguishability_not_estimable",
                    "severity": "fatal",
                    "seed": seed,
                    "track": track,
                    "detail": recipe_report.get("reason", "unknown"),
                }
            )
        elif recipe_report["roc_auc"] > recipe_auc_cap:
            violations.append(
                {
                    "audit": "recipe_label_distinguishability",
                    "severity": "fatal",
                    "seed": seed,
                    "track": track,
                    "observed_roc_auc": recipe_report["roc_auc"],
                    "maximum_roc_auc": recipe_auc_cap,
                }
            )

        # Match each synthetic row to one real parent row. The classifier is evaluated
        # by parent component so descendants can never leak into the opposite fold.
        distinguish_rows: list[dict] = []
        distinguish_y: list[int] = []
        distinguish_groups: list[str] = []
        for row in rows:
            parent_uid = str(row.get("parent_pair_uid") or row.get("source_pair_uid") or "")
            parent = real_index.get(parent_uid)
            if parent is None:
                continue
            distinguish_rows.extend([parent, row])
            distinguish_y.extend([0, 1])
            distinguish_groups.extend([step27.row_component(parent), step27.row_component(parent)])
        distinguish_names = [source_name, *names]
        distinguish_report = grouped_oof_distinguishability(
            step27.matrix(distinguish_rows, distinguish_names),
            np.asarray(distinguish_y, dtype=int),
            distinguish_groups,
            fold_by_component,
            cfg["fold_count"],
        )
        if distinguish_report.get("status") != "estimated":
            violations.append(
                {
                    "audit": "synthetic_real_distinguishability_not_estimable",
                    "severity": "fatal",
                    "seed": seed,
                    "track": track,
                    "detail": distinguish_report.get("reason", "unknown"),
                }
            )
        elif distinguish_report["roc_auc"] > synthetic_real_auc_cap:
            violations.append(
                {
                    "audit": "synthetic_real_distinguishability",
                    "severity": "fatal",
                    "seed": seed,
                    "track": track,
                    "observed_roc_auc": distinguish_report["roc_auc"],
                    "maximum_roc_auc": synthetic_real_auc_cap,
                }
            )
        group_reports.append(
            {
                "seed": seed,
                "track": track,
                "row_count": len(rows),
                "positive_count": int(labels.sum()),
                "negative_count": int(len(labels) - labels.sum()),
                "parent_count": len(parent_weights),
                "duplication_row_count": len(duplication_rows),
                "M1_M2_budget_multiset_equal": synthetic_budget == duplication_budget,
                "effective_weight": total_weight,
                "effective_weight_relative_to_real_train": total_ratio,
                "recipe_fields": recipe_fields,
                "recipe_label_distinguishability": recipe_report,
                "synthetic_real_distinguishability": distinguish_report,
                "synthetic_feature_displacement": displacement_report,
            }
        )

    status = "pass" if not violations else "fail"
    summary = {
        "status": status,
        "run_id": cfg["run_id"],
        "input_manifest_sha256": manifest["manifest_sha256"],
        "pair_feature_bundle_sha256": step27.file_bundle_sha256(
            all_feature_paths + duplication_feature_paths
        ),
        "scientific_contract": {
            "synthetic_rows_are_train_only": True,
            "children_inherit_parent_component_and_fold": True,
            "recipe_label_roc_auc_maximum": recipe_auc_cap,
            "synthetic_real_roc_auc_maximum": synthetic_real_auc_cap,
            "per_parent_effective_weight_cap": parent_cap,
            "total_effective_weight_cap": total_cap,
            "distinguishability_is_component_grouped_oof": True,
            "nonzero_synthetic_feature_displacement_required": True,
        },
        "real_train_effective_weight": real_train_weight,
        "independently_redaction_checked_synthetic_profile_count": redaction_profile_count,
        "group_reports": group_reports,
        "fatal_violation_count": len(violations),
        "violations": violations,
    }
    output_dir = root / "synthetic_audit"
    summary_path = output_dir / "step27_synthetic_data_audit.json"
    violations_path = output_dir / "step27_synthetic_data_audit_violations.csv"
    old_manifest_path = output_dir / "step27_synthetic_audit_input_manifest.json"
    if output_dir.exists() and old_manifest_path.is_file():
        old = step27.load_json(old_manifest_path)
        if old.get("manifest_sha256") != manifest["manifest_sha256"]:
            raise ValueError("Refusing to overwrite Step27 synthetic audit across a different manifest")
    step27.write_json_immutable(old_manifest_path, manifest)
    step27.write_json_immutable(summary_path, summary)
    step27.write_csv_immutable(
        violations_path,
        violations
        or [{"audit": "none", "severity": "none", "detail": "all_preregistered_checks_passed"}],
    )
    print(json.dumps({"status": status, "summary": str(summary_path), "violations": len(violations)}, indent=2))
    if violations:
        raise SystemExit("Step27 synthetic-data audit failed; model training is blocked")


if __name__ == "__main__":
    main()
