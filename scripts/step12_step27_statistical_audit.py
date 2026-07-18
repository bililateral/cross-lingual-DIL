#!/usr/bin/env python3
"""Run grouped Step12 statistics for the preregistered Step27 M0/M1/M2 study."""

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
PRIMARY = ("step27_m2_synthetic", "step27_m1_equal_effective_weight_duplication")
SECONDARY = ("step27_m2_synthetic", "step27_m0_real_only")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", default=str(DEFAULT_POLICY))
    parser.add_argument("--resamples", type=int, default=None)
    parser.add_argument("--validate-config-only", action="store_true")
    parser.add_argument("--validate-inputs-only", action="store_true")
    parser.add_argument(
        "--mode",
        choices=("oof_gate", "valid_gate", "final_diagnostic"),
        default="oof_gate",
    )
    return parser.parse_args()


def statistical_config(policy: dict, requested_resamples: int | None) -> dict:
    stat_cfg = dict(
        policy.get("statistical_audit")
        or policy.get("statistics")
        or policy.get("evaluation")
        or {}
    )
    required = (
        "grouped_bootstrap_resamples",
        "paired_permutation_resamples",
        "grouped_bootstrap_seed",
        "paired_permutation_seed",
    )
    missing = [key for key in required if key not in stat_cfg]
    if missing:
        raise ValueError(f"Step27 preregistered statistical configuration is missing: {missing}")
    bootstrap_resamples = int(stat_cfg["grouped_bootstrap_resamples"])
    permutation_resamples = int(stat_cfg["paired_permutation_resamples"])
    if requested_resamples is not None and requested_resamples != bootstrap_resamples:
        raise ValueError(
            "--resamples may not override the preregistered Step27 grouped-bootstrap "
            f"count: requested={requested_resamples} policy={bootstrap_resamples}"
        )
    if bootstrap_resamples < 1000 or permutation_resamples < 1000:
        raise ValueError("Step27 publication audit requires at least 1000 bootstrap/permutation resamples")
    return {
        "bootstrap_resamples": bootstrap_resamples,
        "permutation_resamples": permutation_resamples,
        "bootstrap_seed": int(stat_cfg["grouped_bootstrap_seed"]),
        "permutation_seed": int(stat_cfg["paired_permutation_seed"]),
    }


def promotion_gate_config(policy: dict) -> dict:
    development = dict(policy.get("development_promotion_gates") or {})
    oof = dict(development.get("train_oof") or {})
    valid = dict(development.get("single_open_valid") or {})
    required = {
        "train_oof": (
            "M2_minus_M1_average_precision_minimum",
            "positive_seed_delta_count_minimum",
            "seller_component_grouped_bootstrap_lower_bound_minimum",
            "direct_or_component_positive_recall_drop_maximum",
            "template_clone_negative_fpr_increase_maximum",
            "public_contact_or_url_negative_fpr_increase_maximum",
        ),
        "single_open_valid": (
            "M2_minus_M1_average_precision_minimum",
            "M2_minus_M0_average_precision_minimum",
            "direct_or_component_positive_recall_drop_maximum",
            "template_clone_negative_fpr_increase_maximum",
            "public_contact_or_url_negative_fpr_increase_maximum",
        ),
    }
    for section_name, keys in required.items():
        section = oof if section_name == "train_oof" else valid
        missing = [key for key in keys if key not in section]
        if missing:
            raise ValueError(f"Step27 {section_name} gate configuration is missing: {missing}")
    return {
        "minimum_oof_primary_ap_delta": float(oof["M2_minus_M1_average_precision_minimum"]),
        "minimum_positive_seed_count": int(oof["positive_seed_delta_count_minimum"]),
        "minimum_oof_primary_bootstrap_lower": float(
            oof["seller_component_grouped_bootstrap_lower_bound_minimum"]
        ),
        "maximum_oof_direct_component_recall_drop": float(
            oof["direct_or_component_positive_recall_drop_maximum"]
        ),
        "maximum_oof_template_fpr_increase": float(
            oof["template_clone_negative_fpr_increase_maximum"]
        ),
        "maximum_oof_public_noise_fpr_increase": float(
            oof["public_contact_or_url_negative_fpr_increase_maximum"]
        ),
        "minimum_valid_primary_ap_delta": float(
            valid["M2_minus_M1_average_precision_minimum"]
        ),
        "minimum_valid_secondary_ap_delta": float(
            valid["M2_minus_M0_average_precision_minimum"]
        ),
        "maximum_valid_direct_component_recall_drop": float(
            valid["direct_or_component_positive_recall_drop_maximum"]
        ),
        "maximum_valid_template_fpr_increase": float(
            valid["template_clone_negative_fpr_increase_maximum"]
        ),
        "maximum_valid_public_noise_fpr_increase": float(
            valid["public_contact_or_url_negative_fpr_increase_maximum"]
        ),
    }


def paired_rows(rows: list[dict], left_model: str, right_model: str, split: str) -> list[dict]:
    selected = [row for row in rows if row["split_name"] == split and row["model_id"] in {left_model, right_model}]
    index: dict[tuple[str, str], dict] = {}
    for row in selected:
        key = (row["model_id"], row["pair_uid"])
        if key in index:
            raise ValueError(f"Duplicate Step27 seed-mean prediction: {key}")
        index[key] = row
    left_uids = {uid for model, uid in index if model == left_model}
    right_uids = {uid for model, uid in index if model == right_model}
    if left_uids != right_uids or not left_uids:
        raise ValueError(f"Unpaired Step27 comparison: {left_model} vs {right_model}/{split}")
    output = []
    for uid in sorted(left_uids):
        left = index[(left_model, uid)]
        right = index[(right_model, uid)]
        for field in ("label", "component_id", "evidence_type"):
            if left[field] != right[field]:
                raise ValueError(f"Step27 comparison metadata differs for {uid}: {field}")
        output.append(
            {
                "pair_uid": uid,
                "label": int(left["label"]),
                "component_id": left["component_id"],
                "evidence_type": left.get("evidence_type", ""),
                "left_score": float(left["prob_positive"]),
                "right_score": float(right["prob_positive"]),
            }
        )
    return output


def component_index(rows: list[dict]) -> dict[str, list[int]]:
    groups: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        groups[row["component_id"]].append(index)
    return dict(groups)


def ap_delta(rows: list[dict], indices: list[int]) -> float:
    y = np.asarray([rows[index]["label"] for index in indices], dtype=int)
    if len(set(y.tolist())) < 2:
        return math.nan
    left = np.asarray([rows[index]["left_score"] for index in indices], dtype=float)
    right = np.asarray([rows[index]["right_score"] for index in indices], dtype=float)
    return step27.average_precision(y, left) - step27.average_precision(y, right)


def grouped_bootstrap(rows: list[dict], resamples: int, seed: int) -> dict:
    groups = component_index(rows)
    component_ids = sorted(groups)
    rng = np.random.default_rng(seed)
    observed = ap_delta(rows, list(range(len(rows))))
    values = []
    for _ in range(resamples):
        sampled = rng.choice(component_ids, size=len(component_ids), replace=True)
        indices = [index for component in sampled for index in groups[str(component)]]
        delta = ap_delta(rows, indices)
        if math.isfinite(delta):
            values.append(delta)
    if len(values) < max(100, int(resamples * 0.5)):
        raise ValueError("Too few estimable Step27 grouped-bootstrap replicates")
    array = np.asarray(values, dtype=float)
    return {
        "unit": "seller_component",
        "random_seed": seed,
        "component_count": len(component_ids),
        "requested_resamples": resamples,
        "estimable_resamples": len(values),
        "observed_ap_delta": observed,
        "ci_95_lower": float(np.quantile(array, 0.025)),
        "ci_95_upper": float(np.quantile(array, 0.975)),
        "probability_delta_positive": float(np.mean(array > 0.0)),
    }


def paired_component_permutation(rows: list[dict], resamples: int, seed: int) -> dict:
    groups = component_index(rows)
    component_ids = sorted(groups)
    observed = abs(ap_delta(rows, list(range(len(rows)))))
    rng = np.random.default_rng(seed)
    y = np.asarray([row["label"] for row in rows], dtype=int)
    left = np.asarray([row["left_score"] for row in rows], dtype=float)
    right = np.asarray([row["right_score"] for row in rows], dtype=float)
    exceed = 0
    for _ in range(resamples):
        swap_components = {component for component in component_ids if rng.integers(0, 2)}
        swap = np.asarray([row["component_id"] in swap_components for row in rows], dtype=bool)
        perm_left = np.where(swap, right, left)
        perm_right = np.where(swap, left, right)
        delta = abs(step27.average_precision(y, perm_left) - step27.average_precision(y, perm_right))
        if delta >= observed - 1e-15:
            exceed += 1
    return {
        "unit": "seller_component",
        "method": "paired_component_score_swap_two_sided",
        "random_seed": seed,
        "component_count": len(component_ids),
        "resamples": resamples,
        "observed_absolute_ap_delta": observed,
        "p_value": float((exceed + 1) / (resamples + 1)),
    }


def seed_direction_count(seed_rows: list[dict], left_model: str, right_model: str, split: str) -> dict:
    by_seed: dict[int, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for row in seed_rows:
        if row["split_name"] == split and row["model_id"] in {left_model, right_model}:
            by_seed[int(row["seed"])][row["model_id"]].append(row)
    deltas = {}
    for seed in step27.DEFAULT_SEEDS:
        left = sorted(by_seed[seed][left_model], key=lambda row: row["pair_uid"])
        right = sorted(by_seed[seed][right_model], key=lambda row: row["pair_uid"])
        if [row["pair_uid"] for row in left] != [row["pair_uid"] for row in right] or not left:
            raise ValueError(f"Incomplete per-seed Step27 comparison: seed={seed}")
        y = np.asarray([int(row["label"]) for row in left], dtype=int)
        delta = step27.average_precision(
            y, np.asarray([float(row["prob_positive"]) for row in left], dtype=float)
        ) - step27.average_precision(
            y, np.asarray([float(row["prob_positive"]) for row in right], dtype=float)
        )
        deltas[str(seed)] = delta
    return {
        "descriptive_only": True,
        "seeds_are_not_independent_inferential_units": True,
        "ap_deltas": deltas,
        "positive_seed_count": sum(value > 0.0 for value in deltas.values()),
        "zero_seed_count": sum(value == 0.0 for value in deltas.values()),
        "negative_seed_count": sum(value < 0.0 for value in deltas.values()),
    }


def slice_rates(rows: list[dict], threshold: float) -> dict:
    direct_types = {"same_controller_direct_identifier", "same_controller_component_anchor"}
    noise_types = {
        "template_clone_not_controller",
        "semantic_topic_not_controller",
        "public_contact_or_url_noise",
    }
    output = {}
    for name, predicate in (
        ("direct_or_component_positive", lambda row: row["label"] == 1 and row["evidence_type"] in direct_types),
        ("template_negative", lambda row: row["label"] == 0 and row["evidence_type"] == "template_clone_not_controller"),
        ("public_noise_negative", lambda row: row["label"] == 0 and row["evidence_type"] == "public_contact_or_url_noise"),
        ("all_preregistered_noise_negative", lambda row: row["label"] == 0 and row["evidence_type"] in noise_types),
    ):
        selected = [row for row in rows if predicate(row)]
        if not selected:
            output[name] = {"row_count": 0, "rate": None}
            continue
        predicted_positive = np.asarray([row["left_score"] >= threshold for row in selected], dtype=bool)
        output[name] = {
            "row_count": len(selected),
            "rate": float(predicted_positive.mean()),
            "rate_definition": "recall" if selected[0]["label"] == 1 else "false_positive_rate",
        }
    return output


def model_metric_rows(predictions: list[dict], splits: tuple[str, ...]) -> list[dict]:
    output = []
    for model_id in step27.REPORTING_MODEL_IDS:
        for split in splits:
            selected = [row for row in predictions if row["model_id"] == model_id and row["split_name"] == split]
            if not selected:
                raise ValueError(f"Step27 seed-mean predictions are missing: {model_id}/{split}")
            threshold_values = {float(row["frozen_oof_threshold"]) for row in selected}
            if len(threshold_values) != 1:
                raise ValueError(f"Step27 frozen threshold is inconsistent: {model_id}/{split}")
            y = np.asarray([int(row["label"]) for row in selected], dtype=int)
            scores = np.asarray([float(row["prob_positive"]) for row in selected], dtype=float)
            result = step27.metrics(y, scores, threshold_values.pop())
            output.append({"model_id": model_id, "split_name": split, **result})
    return output


def paired_slice_audit(
    mean_rows: list[dict], metric_index: dict[tuple[str, str], dict], split: str
) -> dict:
    rows = paired_rows(mean_rows, PRIMARY[0], PRIMARY[1], split)
    m2_threshold = float(metric_index[(PRIMARY[0], split)]["threshold"])
    m1_threshold = float(metric_index[(PRIMARY[1], split)]["threshold"])
    m2 = slice_rates(rows, m2_threshold)
    m1 = slice_rates([{**row, "left_score": row["right_score"]} for row in rows], m1_threshold)
    direct_m2 = m2["direct_or_component_positive"]["rate"]
    direct_m1 = m1["direct_or_component_positive"]["rate"]
    noise = {}
    for name in ("template_negative", "public_noise_negative", "all_preregistered_noise_negative"):
        m2_rate = m2[name]["rate"]
        m1_rate = m1[name]["rate"]
        noise[name] = {
            "m2_fpr": m2_rate,
            "m1_fpr": m1_rate,
            "increase": None if m2_rate is None or m1_rate is None else m2_rate - m1_rate,
            "estimable": m2_rate is not None and m1_rate is not None,
        }
    return {
        "m2": m2,
        "m1_duplication": m1,
        "direct_recall_drop": (
            None if direct_m2 is None or direct_m1 is None else direct_m1 - direct_m2
        ),
        "direct_recall_estimable": direct_m2 is not None and direct_m1 is not None,
        "noise_checks": noise,
    }


def validate_prediction_tables(
    seed_rows: list[dict],
    mean_rows: list[dict],
    canonical_rows: list[dict],
    splits: tuple[str, ...],
) -> None:
    allowed_models = set(step27.REPORTING_MODEL_IDS)
    allowed_splits = set(splits)
    for table_name, rows in (("seed", seed_rows), ("mean", mean_rows)):
        unexpected = [
            (row.get("model_id"), row.get("split_name"))
            for row in rows
            if row.get("model_id") not in allowed_models or row.get("split_name") not in allowed_splits
        ]
        if unexpected:
            raise ValueError(f"Step27 {table_name} predictions contain unexpected model/split: {unexpected[0]}")
    expected_by_split = {
        split: {
            row["pair_uid"]: row
            for row in canonical_rows
            if row.get("split_name") == ("train" if split == "train_oof" else split)
        }
        for split in splits
    }
    seed_keys = set()
    for row in seed_rows:
        key = (row["model_id"], row["split_name"], row["pair_uid"], int(row["seed"]))
        if key in seed_keys:
            raise ValueError(f"Duplicate Step27 per-seed prediction: {key}")
        seed_keys.add(key)
        score = float(row["prob_positive"])
        threshold = float(row["frozen_oof_threshold"])
        if not math.isfinite(score) or not 0.0 <= score <= 1.0:
            raise ValueError(f"Step27 per-seed probability is invalid: {key}={score}")
        if not math.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
            raise ValueError(f"Step27 frozen threshold is invalid: {key}={threshold}")
        if int(row["predicted_label"]) != int(score >= threshold):
            raise ValueError(f"Step27 per-seed predicted label disagrees with its threshold: {key}")
    mean_keys = set()
    for row in mean_rows:
        key = (row["model_id"], row["split_name"], row["pair_uid"])
        if key in mean_keys:
            raise ValueError(f"Duplicate Step27 seed-mean prediction: {key}")
        mean_keys.add(key)
    for model_id in step27.REPORTING_MODEL_IDS:
        for split in splits:
            expected = expected_by_split[split]
            observed_seed = {
                row["pair_uid"]
                for row in seed_rows
                if row["model_id"] == model_id and row["split_name"] == split
            }
            observed_mean = {
                row["pair_uid"]
                for row in mean_rows
                if row["model_id"] == model_id and row["split_name"] == split
            }
            if observed_seed != set(expected) or observed_mean != set(expected):
                raise ValueError(
                    f"Step27 prediction universe differs for {model_id}/{split}: "
                    f"expected={len(expected)} seed={len(observed_seed)} mean={len(observed_mean)}"
                )
            for uid, expected_row in expected.items():
                selected = [
                    row
                    for row in seed_rows
                    if row["model_id"] == model_id
                    and row["split_name"] == split
                    and row["pair_uid"] == uid
                ]
                if len(selected) != len(step27.DEFAULT_SEEDS):
                    raise ValueError(f"Step27 seed coverage is incomplete: {model_id}/{split}/{uid}")
                for row in selected:
                    if (
                        int(row["label"]) != step27.row_label(expected_row)
                        or row["component_id"] != step27.row_component(expected_row)
                    ):
                        raise ValueError(
                            f"Step27 prediction metadata differs from canonical: {model_id}/{split}/{uid}"
                        )
    replay = step27.aggregate_seed_predictions(seed_rows)
    replay_index = {(row["model_id"], row["split_name"], row["pair_uid"]): row for row in replay}
    for row in mean_rows:
        key = (row["model_id"], row["split_name"], row["pair_uid"])
        expected = replay_index.get(key)
        if expected is None:
            raise ValueError(f"Step27 seed-mean row has no per-seed source: {key}")
        if (
            abs(float(row["prob_positive"]) - float(expected["prob_positive"])) > 1e-12
            or abs(float(row["frozen_oof_threshold"]) - float(expected["frozen_oof_threshold"]))
            > 1e-12
            or int(row["predicted_label"]) != int(expected["predicted_label"])
            or int(row.get("seed_count", 0)) != len(step27.DEFAULT_SEEDS)
        ):
            raise ValueError(f"Step27 persisted seed mean does not replay: {key}")


def validate_gate_binding(
    *,
    binding: dict,
    split_name: str,
    cfg: dict,
    policy_path: Path,
    training_summary_path: Path,
    artifacts_path: Path,
    training_manifest: dict,
    training_summary: dict,
    evaluation_input_manifest: dict,
    prerequisite_path: Path,
) -> None:
    prerequisite_relative = str(prerequisite_path.relative_to(ROOT)).replace("\\", "/")
    expected = {
        "run_id": cfg["run_id"],
        "split_name": split_name,
        "policy_sha256": step27.sha256_file(policy_path),
        "producer_sha256": step27.sha256_file(Path(step27.__file__).resolve()),
        "common_sha256": step27.sha256_file(Path(step27.common.__file__).resolve()),
        "frozen_source_artifact_sha256": step27.sha256_file(cfg["source_artifact_path"]),
        "training_summary_sha256": step27.sha256_file(training_summary_path),
        "model_artifacts_sha256": step27.sha256_file(artifacts_path),
        "pair_feature_bundle_sha256": training_summary.get("pair_feature_bundle_sha256"),
        "training_input_manifest_sha256": training_manifest.get("manifest_sha256"),
        "evaluation_input_manifest_sha256": evaluation_input_manifest.get("manifest_sha256"),
        "prerequisite_gate_summary_path": prerequisite_relative,
        "prerequisite_gate_summary_sha256": step27.sha256_file(prerequisite_path),
        "test_metrics_used_for_gate": False,
    }
    missing = [key for key in expected if key not in binding]
    if missing:
        raise ValueError(f"Step27 {split_name} gate binding is incomplete: {missing}")
    mismatched = [key for key, value in expected.items() if binding.get(key) != value]
    if mismatched:
        raise ValueError(
            f"Step27 {split_name} gate binding differs from the frozen inputs: {mismatched}"
        )


def rebuild_evaluation_input_manifest(
    policy: dict, cfg: dict, policy_path: Path, split_name: str
) -> dict:
    _, _, feature_paths = step27.materialize_feature_tables(
        policy, cfg, real_split=split_name
    )
    _, sensitivity_feature_paths = step27.load_sensitivity_feature_tables(policy, cfg)
    duplication_feature_paths = [
        step27.common.track_root(policy, seed, track)
        / "pair_features"
        / "equal_weight_duplication_pair_features.csv"
        for seed in cfg["seeds"]
        for track in ("primary", "silver_sensitivity")
    ]
    manifest_paths = [
        policy_path,
        Path(step27.__file__).resolve(),
        Path(step27.common.__file__).resolve(),
        cfg["source_artifact_path"],
        step27.common.parent_root(policy) / "manifest.json",
        *(step27.common.seed_root(policy, seed) / "generation_manifest.json" for seed in cfg["seeds"]),
        *feature_paths,
        *sensitivity_feature_paths,
        *duplication_feature_paths,
    ]
    return step27.input_manifest(policy_path, manifest_paths, cfg["run_id"])


def validate_delayed_completion(
    *,
    completion_path: Path,
    payload_paths: list[Path],
    cfg: dict,
    binding: dict,
    delayed_summary: dict,
) -> None:
    evaluation_sha256 = binding.get("evaluation_input_manifest_sha256")
    if (
        not evaluation_sha256
        or delayed_summary.get("evaluation_input_manifest_sha256") != evaluation_sha256
        or delayed_summary.get("training_input_manifest_sha256")
        != binding.get("training_input_manifest_sha256")
    ):
        raise ValueError("Step27 delayed summary and binding disagree on frozen inputs")
    expected = step27.completion_manifest(cfg["run_id"], evaluation_sha256, payload_paths)
    if step27.load_json(completion_path) != expected:
        raise ValueError(f"Step27 delayed completion manifest no longer matches: {completion_path}")


def exploratory_source_diagnostics(
    artifact_bundle: dict,
    metric_index: dict[tuple[str, str], dict],
    splits: tuple[str, ...],
    policy: dict,
) -> dict:
    learned_model = step27.EXPLORATORY_MODEL_IDS[0]
    alpha_zero_model = step27.EXPLORATORY_MODEL_IDS[1]
    exploratory_cfg = policy["models"]["exploratory_controls"]
    near_zero_limit = float(
        exploratory_cfg["learned_source_logit_coefficient"][
            "descriptive_near_zero_absolute_alpha_maximum"
        ]
    )
    equivalence_margin = float(
        exploratory_cfg["target_only_alpha_zero"]["descriptive_ap_equivalence_margin"]
    )
    model_artifacts = artifact_bundle.get("artifacts", {}).get(learned_model, {})
    alpha_zero_artifacts = artifact_bundle.get("artifacts", {}).get(alpha_zero_model, {})
    expected_seed_keys = {str(seed) for seed in step27.DEFAULT_SEEDS}
    if set(model_artifacts) != expected_seed_keys:
        raise ValueError("Step27 learned-alpha artifacts do not cover the ten preregistered seeds")
    if set(alpha_zero_artifacts) != expected_seed_keys:
        raise ValueError("Step27 alpha-zero artifacts do not cover the ten preregistered seeds")
    alpha_by_seed = {}
    for seed in step27.DEFAULT_SEEDS:
        artifact = model_artifacts[str(seed)]
        if (
            artifact.get("model_id") != learned_model
            or int(artifact.get("seed", -1)) != seed
            or artifact.get("source_mode") != "learned_source_alpha"
        ):
            raise ValueError(f"Step27 learned-alpha artifact identity differs for seed={seed}")
        alpha = float(artifact.get("learned_source_alpha", math.nan))
        if not math.isfinite(alpha):
            raise ValueError(f"Step27 learned source alpha is not finite for seed={seed}")
        alpha_by_seed[str(seed)] = alpha
        alpha_zero_artifact = alpha_zero_artifacts[str(seed)]
        if (
            alpha_zero_artifact.get("model_id") != alpha_zero_model
            or int(alpha_zero_artifact.get("seed", -1)) != seed
            or alpha_zero_artifact.get("source_mode") != "target_only_alpha_zero"
        ):
            raise ValueError(f"Step27 alpha-zero artifact identity differs for seed={seed}")
    alpha_values = np.asarray(list(alpha_by_seed.values()), dtype=float)
    ap_deltas = {}
    descriptive_equivalence = {}
    for split in splits:
        m2 = metric_index[(step27.MODEL_IDS[2], split)]
        alpha_zero = metric_index[(alpha_zero_model, split)]
        ap_deltas[split] = float(m2["average_precision"] - alpha_zero["average_precision"])
        descriptive_equivalence[split] = abs(ap_deltas[split]) <= equivalence_margin
    return {
        "diagnostic_only": True,
        "can_affect_promotion": False,
        "learned_source_alpha": {
            "model_id": learned_model,
            "seed_count": len(alpha_values),
            "alpha_by_seed": alpha_by_seed,
            "mean": float(alpha_values.mean()),
            "minimum": float(alpha_values.min()),
            "maximum": float(alpha_values.max()),
            "descriptive_near_zero_absolute_maximum": near_zero_limit,
            "near_zero_seed_count": int(np.sum(np.abs(alpha_values) <= near_zero_limit)),
            "all_seeds_near_zero": bool(np.all(np.abs(alpha_values) <= near_zero_limit)),
        },
        "m2_vs_target_only_alpha_zero": {
            "left_model": step27.MODEL_IDS[2],
            "right_model": alpha_zero_model,
            "metric": "average_precision",
            "ap_delta_by_split": ap_deltas,
            "descriptive_equivalence_margin": equivalence_margin,
            "descriptively_equivalent_by_split": descriptive_equivalence,
            "test_delta_is_diagnostic_only": "test" in ap_deltas,
        },
    }


def main() -> None:
    args = parse_args()
    policy_path = step27.resolve(args.policy)
    policy = step27.load_json(policy_path)
    cfg = step27.validate_policy(policy, policy_path)
    statistics = statistical_config(policy, args.resamples)
    gates = promotion_gate_config(policy)
    resamples = statistics["bootstrap_resamples"]
    permutation_resamples = statistics["permutation_resamples"]
    bootstrap_seed = statistics["bootstrap_seed"]
    permutation_seed = statistics["permutation_seed"]
    if args.validate_config_only:
        print(
            json.dumps(
                {
                    "status": "pass",
                    "primary": f"{PRIMARY[0]}_vs_{PRIMARY[1]}",
                    "secondary": f"{SECONDARY[0]}_vs_{SECONDARY[1]}",
                    "bootstrap_resamples": resamples,
                    "permutation_resamples": permutation_resamples,
                    "bootstrap_seed": bootstrap_seed,
                    "permutation_seed": permutation_seed,
                    "gates": gates,
                },
                indent=2,
            )
        )
        return

    root = step27.outputs_root(policy)
    training_dir = root / "training"
    summary_path = training_dir / "step27_training_summary.json"
    artifacts_path = training_dir / "step27_model_artifacts.json"
    seed_path = training_dir / "step27_seed_predictions.csv"
    mean_path = training_dir / "step27_seed_mean_predictions.csv"
    training_manifest_path = training_dir / "step27_training_input_manifest.json"
    synthetic_audit_path = root / "synthetic_audit" / "step27_synthetic_data_audit.json"
    canonical_path = root / "parent_manifest" / "canonical_pairs.csv"
    valid_dir = root / "valid_diagnostic"
    valid_seed_path = valid_dir / "step27_valid_seed_predictions.csv"
    valid_mean_path = valid_dir / "step27_valid_seed_mean_predictions.csv"
    valid_summary_path = valid_dir / "step27_valid_summary.json"
    valid_binding_path = valid_dir / "step27_valid_gate_binding.json"
    valid_completion_path = valid_dir / "step27_valid_completion_manifest.json"
    oof_audit_path = root / "statistical_audit" / "oof_gate" / "step12_step27_statistical_audit.json"
    valid_audit_path = root / "statistical_audit" / "valid_gate" / "step12_step27_statistical_audit.json"
    internal_dir = root / "internal_test_diagnostic"
    internal_seed_path = internal_dir / "step27_test_seed_predictions.csv"
    internal_mean_path = internal_dir / "step27_test_seed_mean_predictions.csv"
    internal_summary_path = internal_dir / "step27_test_summary.json"
    internal_binding_path = internal_dir / "step27_test_gate_binding.json"
    internal_completion_path = internal_dir / "step27_test_completion_manifest.json"
    required = [
        policy_path,
        Path(__file__).resolve(),
        Path(step27.__file__).resolve(),
        Path(step27.common.__file__).resolve(),
        cfg["source_artifact_path"],
        summary_path,
        artifacts_path,
        seed_path,
        mean_path,
        training_manifest_path,
        synthetic_audit_path,
        canonical_path,
    ]
    if args.mode in {"valid_gate", "final_diagnostic"}:
        required.extend(
            [
                valid_seed_path,
                valid_mean_path,
                valid_summary_path,
                valid_binding_path,
                valid_completion_path,
                oof_audit_path,
            ]
        )
    if args.mode == "final_diagnostic":
        required.extend(
            [
                internal_seed_path,
                internal_mean_path,
                internal_summary_path,
                internal_binding_path,
                internal_completion_path,
                valid_audit_path,
            ]
        )
    manifest = step27.input_manifest(policy_path, required, cfg["run_id"])
    training_summary = step27.load_json(summary_path)
    artifact_bundle = step27.load_json(artifacts_path)
    training_manifest = step27.load_json(training_manifest_path)
    synthetic_audit = step27.load_json(synthetic_audit_path)
    training_manifest_core = dict(training_manifest)
    training_manifest_sha256 = training_manifest_core.pop("manifest_sha256", None)
    if not training_manifest_sha256 or training_manifest_sha256 != step27.canonical_hash(
        training_manifest_core
    ):
        raise ValueError("Step27 frozen training input manifest failed its self-hash check")
    for record in training_manifest.get("inputs", []):
        path = step27.resolve(record["path"])
        if not path.is_file() or step27.sha256_file(path) != record.get("sha256"):
            raise ValueError(f"Step27 frozen training input changed before statistics: {path}")
    if training_summary.get("run_id") != cfg["run_id"]:
        raise ValueError("Step27 training summary run_id differs from the policy")
    if training_summary.get("input_manifest_sha256") != training_manifest_sha256:
        raise ValueError("Step27 training summary is not bound to its input manifest")
    if (
        artifact_bundle.get("run_id") != cfg["run_id"]
        or artifact_bundle.get("input_manifest_sha256") != training_manifest_sha256
    ):
        raise ValueError("Step27 model artifacts are not bound to the frozen training manifest")
    artifacts = artifact_bundle.get("artifacts") or {}
    if set(artifacts) != set(step27.REPORTING_MODEL_IDS):
        raise ValueError("Step27 model artifact set differs from the preregistered reporting models")
    residual_names = list(training_summary.get("residual_feature_names") or [])
    expected_seeds = {str(seed) for seed in step27.DEFAULT_SEEDS}
    for model_id in step27.REPORTING_MODEL_IDS:
        if set(artifacts[model_id]) != expected_seeds:
            raise ValueError(f"Step27 artifact seed set changed: {model_id}")
        for seed in step27.DEFAULT_SEEDS:
            step27.validate_persisted_artifact_contract(
                artifacts[model_id][str(seed)],
                model_id,
                seed,
                residual_names,
                training_manifest_sha256,
            )
    if synthetic_audit.get("status") != "pass":
        raise ValueError("Step27 synthetic data audit did not pass")
    if synthetic_audit.get("pair_feature_bundle_sha256") != training_summary.get("pair_feature_bundle_sha256"):
        raise ValueError("Step27 synthetic audit and training do not share the same frozen pair features")
    contract = training_summary.get("scientific_contract") or {}
    if contract.get("source_offset_coefficient") != 1.0:
        raise ValueError("Step27 statistics refuse a non-unit source offset coefficient")
    if contract.get("test_metrics_used_for_configuration_selection", False) or contract.get(
        "valid_or_test_used_for_configuration_selection", False
    ):
        raise ValueError("Step27 test-informed model selection is forbidden")
    seed_rows = step27.load_csv(seed_path)
    mean_rows = step27.load_csv(mean_path)
    available_splits = ("train_oof",)
    prerequisite_oof = None
    prerequisite_valid = None
    if args.mode in {"valid_gate", "final_diagnostic"}:
        prerequisite_oof = step27.load_json(oof_audit_path)
        valid_summary = step27.load_json(valid_summary_path)
        valid_binding = step27.load_json(valid_binding_path)
        if (
            prerequisite_oof.get("run_id") != cfg["run_id"]
            or prerequisite_oof.get("analysis_contract", {}).get("audit_mode") != "oof_gate"
        ):
            raise ValueError("Step27 valid access is not bound to an OOF-gate audit")
        if not prerequisite_oof.get("promotion", {}).get("eligible_for_valid", False):
            raise ValueError("Step27 valid remains blocked because the OOF gate failed")
        if prerequisite_oof.get("promotion", {}).get("test_metrics_used_for_gate", False):
            raise ValueError("Step27 OOF gate illegally used test metrics")
        prerequisite_sha256 = step27.sha256_file(oof_audit_path)
        if (
            valid_summary.get("status") != "complete"
            or valid_summary.get("split_name") != "valid"
            or valid_summary.get("run_id") != cfg["run_id"]
            or valid_summary.get("prerequisite_gate_summary_sha256") != prerequisite_sha256
        ):
            raise ValueError("Step27 valid predictions are not bound to the passing OOF gate")
        valid_evaluation_manifest = rebuild_evaluation_input_manifest(
            policy, cfg, policy_path, "valid"
        )
        validate_gate_binding(
            binding=valid_binding,
            split_name="valid",
            cfg=cfg,
            policy_path=policy_path,
            training_summary_path=summary_path,
            artifacts_path=artifacts_path,
            training_manifest=training_manifest,
            training_summary=training_summary,
            evaluation_input_manifest=valid_evaluation_manifest,
            prerequisite_path=oof_audit_path,
        )
        validate_delayed_completion(
            completion_path=valid_completion_path,
            payload_paths=[
                valid_binding_path,
                valid_seed_path,
                valid_mean_path,
                valid_summary_path,
            ],
            cfg=cfg,
            binding=valid_binding,
            delayed_summary=valid_summary,
        )
        seed_rows.extend(step27.load_csv(valid_seed_path))
        mean_rows.extend(step27.load_csv(valid_mean_path))
        available_splits = ("train_oof", "valid")
    if args.mode == "final_diagnostic":
        prerequisite_valid = step27.load_json(valid_audit_path)
        internal_summary = step27.load_json(internal_summary_path)
        internal_binding = step27.load_json(internal_binding_path)
        if (
            prerequisite_valid.get("run_id") != cfg["run_id"]
            or prerequisite_valid.get("analysis_contract", {}).get("audit_mode") != "valid_gate"
        ):
            raise ValueError("Step27 internal test is not bound to a valid-gate audit")
        if not prerequisite_valid.get("promotion", {}).get("eligible_for_internal_test", False):
            raise ValueError("Step27 internal test remains blocked because the valid gate failed")
        if prerequisite_valid.get("promotion", {}).get("test_metrics_used_for_gate", False):
            raise ValueError("Step27 valid gate illegally used test metrics")
        prerequisite_sha256 = step27.sha256_file(valid_audit_path)
        if (
            internal_summary.get("status") != "complete"
            or internal_summary.get("split_name") != "test"
            or internal_summary.get("run_id") != cfg["run_id"]
            or internal_summary.get("prerequisite_gate_summary_sha256") != prerequisite_sha256
        ):
            raise ValueError("Step27 test predictions are not bound to the passing valid gate")
        test_evaluation_manifest = rebuild_evaluation_input_manifest(
            policy, cfg, policy_path, "test"
        )
        validate_gate_binding(
            binding=internal_binding,
            split_name="test",
            cfg=cfg,
            policy_path=policy_path,
            training_summary_path=summary_path,
            artifacts_path=artifacts_path,
            training_manifest=training_manifest,
            training_summary=training_summary,
            evaluation_input_manifest=test_evaluation_manifest,
            prerequisite_path=valid_audit_path,
        )
        validate_delayed_completion(
            completion_path=internal_completion_path,
            payload_paths=[
                internal_binding_path,
                internal_seed_path,
                internal_mean_path,
                internal_summary_path,
            ],
            cfg=cfg,
            binding=internal_binding,
            delayed_summary=internal_summary,
        )
        seed_rows.extend(step27.load_csv(internal_seed_path))
        mean_rows.extend(step27.load_csv(internal_mean_path))
        available_splits = ("train_oof", "valid", "test")
    if any(int(row.get("seed_count", 0)) != 10 for row in mean_rows):
        raise ValueError("Step27 seed-mean table is not based on all ten preregistered seeds")
    if any(step27.bool_value(row.get("seeds_are_independent_inferential_units")) for row in mean_rows):
        raise ValueError("Step27 seed means incorrectly mark seeds as inferential units")
    validate_prediction_tables(
        seed_rows, mean_rows, step27.load_csv(canonical_path), available_splits
    )
    if args.validate_inputs_only:
        print(
            json.dumps(
                {
                    "status": "pass",
                    "mode": args.mode,
                    "input_manifest_sha256": manifest["manifest_sha256"],
                    "validated_splits": list(available_splits),
                },
                indent=2,
            )
        )
        return

    metric_rows = model_metric_rows(mean_rows, available_splits)
    metric_index = {(row["model_id"], row["split_name"]): row for row in metric_rows}
    exploratory_diagnostics = exploratory_source_diagnostics(
        artifact_bundle, metric_index, available_splits, policy
    )
    comparisons = []
    comparison_details = {}
    for comparison_name, (left, right) in (("primary", PRIMARY), ("required_secondary", SECONDARY)):
        comparison_details[comparison_name] = {}
        for split in available_splits:
            rows = paired_rows(mean_rows, left, right, split)
            bootstrap = grouped_bootstrap(
                rows, resamples, bootstrap_seed + len(comparisons) * 17
            )
            permutation = paired_component_permutation(
                rows, permutation_resamples, permutation_seed + len(comparisons) * 31
            )
            record = {
                "comparison_role": comparison_name,
                "left_model": left,
                "right_model": right,
                "split_name": split,
                "split_role": "internal_diagnostic_only" if split == "test" else "development",
                "ap_delta": bootstrap["observed_ap_delta"],
                "bootstrap_ci_95_lower": bootstrap["ci_95_lower"],
                "bootstrap_ci_95_upper": bootstrap["ci_95_upper"],
                "paired_component_permutation_p_value": permutation["p_value"],
                "component_count": bootstrap["component_count"],
            }
            comparisons.append(record)
            comparison_details[comparison_name][split] = {
                "bootstrap": bootstrap,
                "permutation": permutation,
            }
        comparison_details[comparison_name]["seed_direction_train_oof"] = seed_direction_count(
            seed_rows, left, right, "train_oof"
        )

    primary_oof = comparison_details["primary"]["train_oof"]["bootstrap"]
    secondary_oof = comparison_details["required_secondary"]["train_oof"]["bootstrap"]
    seed_directions = comparison_details["primary"]["seed_direction_train_oof"]
    oof_slices = paired_slice_audit(mean_rows, metric_index, "train_oof")
    oof_gate_results = {
        "oof_primary_ap_delta": primary_oof["observed_ap_delta"] >= gates["minimum_oof_primary_ap_delta"],
        "oof_secondary_ap_delta_positive": secondary_oof["observed_ap_delta"] > 0.0,
        "positive_seed_count": seed_directions["positive_seed_count"] >= gates["minimum_positive_seed_count"],
        "oof_primary_bootstrap_non_degradation": primary_oof["ci_95_lower"] >= gates["minimum_oof_primary_bootstrap_lower"],
        "oof_direct_component_recall_non_degradation": bool(
            oof_slices["direct_recall_estimable"]
            and oof_slices["direct_recall_drop"]
            <= gates["maximum_oof_direct_component_recall_drop"]
        ),
        "oof_template_fpr_non_degradation": bool(
            oof_slices["noise_checks"]["template_negative"]["estimable"]
            and oof_slices["noise_checks"]["template_negative"]["increase"]
            <= gates["maximum_oof_template_fpr_increase"]
        ),
        "oof_public_noise_fpr_non_degradation": bool(
            oof_slices["noise_checks"]["public_noise_negative"]["estimable"]
            and oof_slices["noise_checks"]["public_noise_negative"]["increase"]
            <= gates["maximum_oof_public_noise_fpr_increase"]
        ),
        "synthetic_data_and_shortcut_audit_pass": synthetic_audit.get("status") == "pass",
    }
    eligible_for_valid = all(oof_gate_results.values())
    valid_slices = None
    valid_gate_results = None
    eligible_for_internal_test = False
    if args.mode in {"valid_gate", "final_diagnostic"}:
        primary_valid = comparison_details["primary"]["valid"]["bootstrap"]
        secondary_valid = comparison_details["required_secondary"]["valid"]["bootstrap"]
        valid_slices = paired_slice_audit(mean_rows, metric_index, "valid")
        valid_gate_results = {
            "oof_gate_was_passed_before_valid_access": bool(
                prerequisite_oof.get("promotion", {}).get("eligible_for_valid", False)
            ),
            "valid_primary_ap_delta": primary_valid["observed_ap_delta"]
            >= gates["minimum_valid_primary_ap_delta"],
            "valid_secondary_ap_delta": secondary_valid["observed_ap_delta"]
            >= gates["minimum_valid_secondary_ap_delta"],
            "valid_direct_component_recall_non_degradation": bool(
                valid_slices["direct_recall_estimable"]
                and valid_slices["direct_recall_drop"]
                <= gates["maximum_valid_direct_component_recall_drop"]
            ),
            "valid_template_fpr_non_degradation": bool(
                valid_slices["noise_checks"]["template_negative"]["estimable"]
                and valid_slices["noise_checks"]["template_negative"]["increase"]
                <= gates["maximum_valid_template_fpr_increase"]
            ),
            "valid_public_noise_fpr_non_degradation": bool(
                valid_slices["noise_checks"]["public_noise_negative"]["estimable"]
                and valid_slices["noise_checks"]["public_noise_negative"]["increase"]
                <= gates["maximum_valid_public_noise_fpr_increase"]
            ),
        }
        eligible_for_internal_test = all(valid_gate_results.values())
    gate_results = (
        oof_gate_results if args.mode == "oof_gate" else {**oof_gate_results, **valid_gate_results}
    )
    summary = {
        "status": "complete",
        "run_id": cfg["run_id"],
        "input_manifest_sha256": manifest["manifest_sha256"],
        "analysis_contract": {
            "primary_comparison": f"{PRIMARY[0]}_vs_{PRIMARY[1]}",
            "required_secondary_comparison": f"{SECONDARY[0]}_vs_{SECONDARY[1]}",
            "score_aggregation": "per_real_pair_mean_across_ten_preregistered_seeds",
            "seeds_are_independent_inferential_units": False,
            "bootstrap_unit": "seller_component",
            "permutation_unit": "seller_component",
            "bootstrap_resamples": resamples,
            "permutation_resamples": permutation_resamples,
            "bootstrap_seed": bootstrap_seed,
            "permutation_seed": permutation_seed,
            "test_role": "retrospective_internal_diagnostic_only",
            "test_metrics_used_for_selection_or_promotion": False,
            "audit_mode": args.mode,
            "available_splits": list(available_splits),
            "confirmatory_evaluation": (
                "Step20 remains blocked until a separate Step27-specific prospective freeze"
            ),
            "single_primary_comparison_no_Holm_adjustment_required": True,
        },
        "model_metrics": metric_rows,
        "comparisons": comparison_details,
        "exploratory_source_diagnostics": exploratory_diagnostics,
        "slice_audit_train_oof": oof_slices,
        "slice_audit_valid": valid_slices,
        "promotion": {
            "eligible_for_valid": eligible_for_valid,
            "eligible_for_internal_test": eligible_for_internal_test,
            "eligible_for_step20_freeze_preparation": eligible_for_internal_test,
            "eligible_for_step20_prospective_evaluation": False,
            "step27_specific_prospective_freeze_exists": False,
            "test_metrics_used_for_gate": False,
            "internal_test_scored": args.mode == "final_diagnostic",
            "not_a_publication_claim": True,
            "gates": gates,
            "gate_results": gate_results,
            "oof_gate_results": oof_gate_results,
            "valid_gate_results": valid_gate_results,
            "failed_gates": [name for name, passed in gate_results.items() if not passed],
        },
    }

    output_dir = root / "statistical_audit" / args.mode
    audit_manifest_path = output_dir / "step12_step27_input_manifest.json"
    if output_dir.exists() and audit_manifest_path.is_file():
        old = step27.load_json(audit_manifest_path)
        if old.get("manifest_sha256") != manifest["manifest_sha256"]:
            raise ValueError("Refusing to overwrite Step27 statistics across a different manifest")
    step27.write_json_immutable(audit_manifest_path, manifest)
    step27.write_csv_immutable(output_dir / "step12_step27_model_metrics.csv", metric_rows)
    step27.write_csv_immutable(output_dir / "step12_step27_paired_comparisons.csv", comparisons)
    step27.write_json_immutable(output_dir / "step12_step27_statistical_audit.json", summary)
    print(
        json.dumps(
            {
                "status": "complete",
                "mode": args.mode,
                "eligible_for_valid": eligible_for_valid,
                "eligible_for_internal_test": eligible_for_internal_test,
                "eligible_for_step20_freeze_preparation": eligible_for_internal_test,
                "eligible_for_step20_prospective_evaluation": False,
                "failed_gates": summary["promotion"]["failed_gates"],
                "summary": str(output_dir / "step12_step27_statistical_audit.json"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
