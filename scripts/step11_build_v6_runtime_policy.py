#!/usr/bin/env python3
"""Build a Step11 runtime policy from the validation-selected Step15-v6 scorer."""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
from pathlib import Path

from immutable_artifact_io import json_bytes, write_immutable_bytes


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_BASE_POLICY = ROOT / "schema" / "step11_clustering_policy.json"
DEFAULT_STEP12_SUMMARY = (
    ROOT
    / "reports"
    / "step12_v6"
    / "method_audit_v4_inductive_20260712"
    / "step12_v6_statistical_robustness.json"
)
DEFAULT_OUTPUT = ROOT / "reports" / "step15_v6" / "manifests" / "step11_v6_runtime_policy.json"
STEP15_SUMMARY = "reports/step15_v6/step15_v6_training_summary.json"
STEP15_POLICY = "schema/step15_v6_paper_hardening_policy.json"
STEP9_SUMMARY = "reports/step15_v6/baselines/step9/step9_few_shot_summary.json"
STEP12_SCRIPT = "scripts/step12_v6_statistical_robustness_audit.py"
STEP5_ZH_LABELS = "reports/step5_zh_target_strict_frozen_silver_labels.csv"
EXPECTED_ACTIVE_MANIFEST = (
    "reports/step15_v6/manifests/step15_v6_internal_dev_v4_20260712.json"
)
EXPECTED_ACTIVE_RUN_ID = "step15-v6-method-audit-v4-inductive-internal-dev-20260712"
SEEDS = list(range(20260320, 20260330))


MODEL_MAPPING = {
    "step15_v6_m3": ("step15_v6_m3_warm_start_curriculum", "phase3_add_contact_url_noise"),
    "step15_v6_m4": ("step15_v6_m4_trusted_positive_mixup", "phase4_add_trusted_positive_mixup"),
    "step15_v6_m5_lambda_0p1": (
        "step15_v6_m5_aux_evidence_lambda_0p1",
        "phase3_add_contact_url_noise",
    ),
    "step15_v6_m5_lambda_0p3": (
        "step15_v6_m5_aux_evidence_lambda_0p3",
        "phase3_add_contact_url_noise",
    ),
}

STEP9_MODEL_MAPPING = {
    "step9_e5_lr_l2_100pct_seed_mean": "core_few_shot_multilingual_e5_large_lr_l2",
    "step9_bge_m3_residual_lr_100pct_seed_mean": "core_few_shot_bge_m3_residual_lr",
    "step9_labse_lr_l2_100pct_seed_mean": "core_few_shot_labse_lr_l2",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-policy", default=str(DEFAULT_BASE_POLICY))
    parser.add_argument("--step12-summary", default=str(DEFAULT_STEP12_SUMMARY))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--allow-nonpromoted-diagnostic", action="store_true")
    parser.add_argument(
        "--validation-mode",
        choices=("clean_topology", "identifier_assisted_operational"),
        default="clean_topology",
    )
    return parser.parse_args()


def resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalized_relative_path(value: str | Path) -> str:
    path = Path(value)
    if path.is_absolute():
        try:
            path = path.relative_to(ROOT)
        except ValueError as exc:
            raise ValueError(f"Frozen publication input must live under the project root: {path}") from exc
    return path.as_posix()


def verify_self_hashed_manifest(path: Path, manifest: dict, label: str) -> None:
    expected = str(manifest.get("manifest_sha256", ""))
    core = {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    observed = hashlib.sha256(
        json.dumps(core, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
    ).hexdigest()
    if not expected or expected != observed:
        raise ValueError(f"{label} self-hash mismatch: {path}")


def load_verified_active_manifest(step12: dict) -> tuple[Path, dict, dict[str, str]]:
    step12_policy_value = str(step12.get("policy", "")).strip()
    if not step12_policy_value:
        raise ValueError("Step12-v6 summary does not record its policy path")
    step12_policy_path = resolve(step12_policy_value)
    if not step12_policy_path.exists():
        raise FileNotFoundError(step12_policy_path)
    step12_policy = load_json(step12_policy_path)
    input_cfg = step12_policy.get("inputs", {}) or {}
    active_value = str(input_cfg.get("step15_v6_active_manifest", "")).strip()
    expected_run_id = str(input_cfg.get("step15_v6_active_manifest_run_id", "")).strip()
    if not active_value or not expected_run_id:
        raise ValueError(
            "Step12-v6 policy must bind step15_v6_active_manifest and its run_id"
        )
    if normalized_relative_path(active_value) != EXPECTED_ACTIVE_MANIFEST:
        raise ValueError(
            "Step11-v6 requires the inductive v4 active manifest referenced by Step12: "
            f"expected={EXPECTED_ACTIVE_MANIFEST!r} observed={active_value!r}"
        )
    if expected_run_id != EXPECTED_ACTIVE_RUN_ID:
        raise ValueError(
            "Step11-v6 requires the inductive v4 active-manifest run id: "
            f"expected={EXPECTED_ACTIVE_RUN_ID!r} observed={expected_run_id!r}"
        )
    active_path = resolve(active_value)
    if not active_path.exists():
        raise FileNotFoundError(active_path)
    active = load_json(active_path)
    verify_self_hashed_manifest(active_path, active, "Step15-v6 active manifest")
    if str(active.get("run_id", "")) != expected_run_id:
        raise ValueError(
            "Step15-v6 active-manifest run_id mismatch: "
            f"expected={expected_run_id!r} observed={active.get('run_id')!r}"
        )
    step15_policy = load_json(resolve(STEP15_POLICY))
    expected_policy_version = str(step15_policy.get("version", "")).strip()
    if not expected_policy_version or str(active.get("policy_version", "")) != expected_policy_version:
        raise ValueError(
            "Step15-v6 active-manifest policy version mismatch: "
            f"expected={expected_policy_version!r} observed={active.get('policy_version')!r}"
        )

    frozen_hashes: dict[str, str] = {}
    records = active.get("files", []) or []
    if not records:
        raise ValueError("Step15-v6 active manifest contains no file records")
    for record in records:
        relative = normalized_relative_path(str(record.get("path", "")))
        expected_hash = str(record.get("sha256", "")).strip()
        if not relative or not expected_hash or relative in frozen_hashes:
            raise ValueError(
                f"Invalid or duplicate Step15-v6 active-manifest file record: {relative!r}"
            )
        path = resolve(relative)
        if not path.exists():
            raise FileNotFoundError(f"Frozen Step15-v6 input is missing: {path}")
        observed_hash = sha256(path)
        if observed_hash != expected_hash:
            raise ValueError(
                f"Frozen Step15-v6 input hash mismatch for {relative}: "
                f"expected={expected_hash} observed={observed_hash}"
            )
        frozen_hashes[relative] = expected_hash
    return active_path, active, frozen_hashes


def require_frozen_paths(frozen_hashes: dict[str, str], paths: list[str | Path]) -> None:
    missing = [
        normalized_relative_path(path)
        for path in paths
        if normalized_relative_path(path) not in frozen_hashes
    ]
    if missing:
        raise ValueError(
            "Step15-v6 active manifest does not bind required Step11 publication inputs: "
            f"{sorted(set(missing))}"
        )


def verify_inductive_feature_lineage(
    manifest_path: Path,
    reference_path: Path,
    zh_pair_features_path: Path,
) -> dict:
    manifest = load_json(manifest_path)
    verify_self_hashed_manifest(manifest_path, manifest, "Step15-v6 inductive feature manifest")
    if bool(manifest.get("transductive_valid_or_test_covariates_used_for_reference", True)):
        raise ValueError("Step15-v6 inductive feature manifest reports transductive reference use")
    if bool(manifest.get("candidate_pair_universe_changed", True)):
        raise ValueError("Step15-v6 inductive feature manifest reports a changed pair universe")
    recorded_reference = normalized_relative_path(str(manifest.get("reference_bundle", "")))
    if recorded_reference != normalized_relative_path(reference_path):
        raise ValueError("Step15-v6 inductive manifest references a different train-only bundle")
    if sha256(reference_path) != str(manifest.get("reference_bundle_sha256", "")):
        raise ValueError("Step15-v6 train-only reference bundle hash mismatch")
    domain_records = manifest.get("domains", {}) or {}
    if not isinstance(domain_records, dict) or set(domain_records) != {
        "en_content_train_pool",
        "zh_target_strict",
    }:
        raise ValueError("Step15-v6 inductive manifest must bind exactly the EN and ZH domains")
    zh_record = domain_records["zh_target_strict"]
    if normalized_relative_path(str(zh_record.get("output_path", ""))) != normalized_relative_path(
        zh_pair_features_path
    ):
        raise ValueError("Step15-v6 inductive manifest points to a different ZH pair-feature file")
    if sha256(zh_pair_features_path) != str(zh_record.get("output_sha256", "")):
        raise ValueError("Step15-v6 inductive ZH pair-feature hash mismatch")
    if not bool(zh_record.get("canonical_semantic_values_preserved", False)):
        raise ValueError("Step15-v6 inductive manifest does not preserve semantic values")
    return {
        "manifest_path": normalized_relative_path(manifest_path),
        "manifest_sha256": manifest["manifest_sha256"],
        "reference_scope": manifest.get("reference_scope"),
        "reference_bundle_path": normalized_relative_path(reference_path),
        "reference_bundle_sha256": sha256(reference_path),
        "zh_pair_features_path": normalized_relative_path(zh_pair_features_path),
        "zh_pair_features_sha256": sha256(zh_pair_features_path),
        "transductive_valid_or_test_covariates_used_for_reference": False,
    }


def verify_step12_completion(
    step12_path: Path,
    step12: dict,
) -> tuple[Path, dict, dict[str, str]]:
    completion_value = str((step12.get("outputs") or {}).get("completion_manifest_json", ""))
    if not completion_value:
        raise ValueError("Step12-v6 summary does not reference a completion manifest")
    completion_path = resolve(completion_value)
    completion = load_json(completion_path)
    expected_self_hash = str(completion.get("manifest_sha256", ""))
    core = {key: value for key, value in completion.items() if key != "manifest_sha256"}
    observed_self_hash = hashlib.sha256(
        json.dumps(core, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
    ).hexdigest()
    if not expected_self_hash or expected_self_hash != observed_self_hash:
        raise ValueError(f"Step12-v6 completion manifest self-hash mismatch: {completion_path}")
    file_index: dict[str, dict] = {}
    completion_hashes: dict[str, str] = {}
    for record in completion.get("files", []) or []:
        relative = normalized_relative_path(str(record.get("path", "")))
        expected_hash = str(record.get("sha256", "")).strip()
        if not relative or not expected_hash or relative in file_index:
            raise ValueError(
                f"Invalid or duplicate Step12-v6 completion file record: {relative!r}"
            )
        path = resolve(relative)
        if not path.exists() or sha256(path) != expected_hash:
            raise ValueError(
                f"Step12-v6 completion manifest file hash mismatch: {relative}"
            )
        file_index[relative] = record
        completion_hashes[relative] = expected_hash
    required_paths = {
        normalized_relative_path(step12_path),
        STEP12_SCRIPT,
        STEP15_SUMMARY,
        STEP15_POLICY,
    }
    step12_policy_value = str(step12.get("policy", "")).strip()
    metric_script_value = str(
        ((step12.get("producer_context") or {}).get("step7_metric_implementation") or {}).get(
            "path", ""
        )
    ).strip()
    if not step12_policy_value or not metric_script_value:
        raise ValueError("Step12-v6 summary does not bind its policy and metric implementation")
    required_paths.update(
        {
            normalized_relative_path(step12_policy_value),
            normalized_relative_path(metric_script_value),
        }
    )
    for relative in required_paths:
        record = file_index.get(relative)
        path = resolve(relative)
        if record is None or not path.exists() or sha256(path) != record.get("sha256"):
            raise ValueError(f"Step12-v6 completion manifest does not bind current file: {relative}")
    completion_hashes[normalized_relative_path(completion_path)] = sha256(completion_path)
    return completion_path, completion, completion_hashes


def merge_frozen_hashes(*sources: dict[str, str]) -> dict[str, str]:
    merged: dict[str, str] = {}
    for source in sources:
        for path, digest in source.items():
            relative = normalized_relative_path(path)
            previous = merged.get(relative)
            if previous is not None and previous != digest:
                raise ValueError(
                    "Frozen publication manifests disagree on a file hash: "
                    f"path={relative!r} first={previous} second={digest}"
                )
            merged[relative] = str(digest)
    return merged


def load_raw_bge_control(step12: dict, completion_hashes: dict[str, str]) -> dict:
    metrics_value = str((step12.get("outputs") or {}).get("model_metrics_csv", "")).strip()
    if not metrics_value:
        raise ValueError("Step12-v6 summary does not record model_metrics_csv")
    metrics_relative = normalized_relative_path(metrics_value)
    if metrics_relative not in completion_hashes:
        raise ValueError(
            "Step12-v6 completion manifest does not bind the model metrics used by raw BGE"
        )
    metrics_path = resolve(metrics_relative)
    with metrics_path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = [
            row
            for row in csv.DictReader(handle)
            if str(row.get("model_id", "")).strip() == "raw_bge_m3_cosine"
        ]
    if len(rows) != 1:
        raise ValueError(
            "Step12-v6 model metrics must contain exactly one raw_bge_m3_cosine row"
        )
    row = rows[0]
    threshold_source = str(row.get("threshold_source", "")).strip()
    if threshold_source != "mean_zh_valid_scores":
        raise ValueError(
            "Raw BGE graph threshold must be frozen from zh_valid scores only; "
            f"observed threshold_source={threshold_source!r}"
        )
    threshold = float(row["threshold"])
    if not 0.0 <= threshold <= 1.0:
        raise ValueError(f"Raw BGE validation threshold is outside [0, 1]: {threshold}")
    summary_value = str((step12.get("outputs") or {}).get("summary_json", "")).strip()
    if not summary_value:
        raise ValueError("Step12-v6 summary does not self-declare outputs.summary_json")
    return {
        "feature_name": "embedding_cosine_bge_m3",
        "primary_threshold": threshold,
        "threshold_source": "step12_v6_model_metrics::raw_bge_m3_cosine::mean_zh_valid_scores",
        "threshold_metrics_path": metrics_relative,
        "step12_summary_path": normalized_relative_path(summary_value),
        "test_metrics_used_for_threshold_selection": False,
    }


def configure_validation_mode(
    policy: dict,
    validation_mode: str,
    strict_clean_features: list[str] | None = None,
) -> str:
    reliability = policy["graph_policy"]["graph_edge_filters"]["relation_reliability_filter"]
    if validation_mode == "clean_topology":
        if strict_clean_features is None:
            step15_policy = load_json(resolve(STEP15_POLICY))
            strict_clean_features = [
                str(value)
                for value in (step15_policy.get("feature_sets", {}) or {}).get(
                    "strict_clean_30d", []
                )
            ]
        if len(strict_clean_features) != 30 or len(set(strict_clean_features)) != 30:
            raise ValueError(
                "Step11-v6 clean reliability requires the exact 30-feature strict_clean_30d allow-list"
            )
        reliability["hard_keep_direct_identity"] = False
        reliability["use_direct_identity_context_for_reliability_rules"] = False
        reliability["weights"]["shared_pgp_fingerprint"] = 0.0
        reliability["weights"]["shared_seller_contact"] = 0.0
        reliability["feature_allowlist"] = list(strict_clean_features)
        reliability["feature_allowlist_source"] = (
            "schema/step15_v6_paper_hardening_policy.json::feature_sets.strict_clean_30d"
        )
        reliability["scoring_contract"] = (
            "identifier_free_scoring_and_graph_filter_conditional_on_fixed_candidate_universe"
        )
        policy["step15_v6_validation_gate"]["direct_proof_edge_filter_contract"] = (
            "audit_retention_only_no_identifier_weight_no_direct_hard_keep"
        )
        policy["step15_v6_validation_gate"]["scientific_scope"] = (
            "identifier_free_scoring_and_graph_filter_conditional_on_fixed_candidate_universe"
        )
        return "clean_topology"
    if validation_mode == "identifier_assisted_operational":
        reliability["hard_keep_direct_identity"] = True
        reliability["use_direct_identity_context_for_reliability_rules"] = True
        reliability.pop("feature_allowlist", None)
        reliability.pop("feature_allowlist_source", None)
        reliability["scoring_contract"] = "identifier_assisted_operational_control"
        policy["step15_v6_validation_gate"]["direct_proof_edge_filter_contract"] = (
            "hard_keep_and_count_at_reliability_reciprocal_shared_neighbor_triangle"
        )
        policy["step15_v6_validation_gate"]["scientific_scope"] = (
            "identifier_assisted_operational_control_not_clean_method_claim"
        )
        return "identifier_operational"
    raise ValueError(f"Unsupported Step11-v6 validation mode: {validation_mode}")


def main() -> None:
    args = parse_args()
    base_path = resolve(args.base_policy)
    step12_path = resolve(args.step12_summary)
    output_path = resolve(args.output)
    base = load_json(base_path)
    step12 = load_json(step12_path)
    completion_path, completion, completion_hashes = verify_step12_completion(
        step12_path, step12
    )
    active_manifest_path, active_manifest, active_hashes = load_verified_active_manifest(step12)
    frozen_hashes = merge_frozen_hashes(active_hashes, completion_hashes)
    raw_bge_control = load_raw_bge_control(step12, completion_hashes)
    if raw_bge_control["step12_summary_path"] != normalized_relative_path(step12_path):
        raise ValueError(
            "Step12-v6 summary outputs.summary_json is not self-consistent with the supplied path"
        )
    promoted = bool(step12.get("promotion", {}).get("eligible", False))
    if not promoted and not args.allow_nonpromoted_diagnostic:
        raise SystemExit(
            "Step15 v6 did not pass the preregistered Step12 promotion rule. "
            "Step11 publication validation is blocked. Use --allow-nonpromoted-diagnostic only for an explicitly labelled diagnostic graph."
        )
    selected_model = str(step12.get("selection", {}).get("final_selected", ""))
    if selected_model == "step15_v6_m5_selected":
        selected_model = str(step12.get("selection", {}).get("m5_selected", ""))
    if selected_model not in MODEL_MAPPING:
        raise SystemExit(f"Unsupported Step15-v6 final model selection: {selected_model!r}")
    experiment, phase = MODEL_MAPPING[selected_model]
    selected_step9_model = str(
        (step12.get("selection", {}) or {}).get("strongest_clean_step9_selected", "")
    )
    if selected_step9_model not in STEP9_MODEL_MAPPING:
        raise SystemExit(
            "Unsupported or missing validation-selected strongest clean Step9 model: "
            f"{selected_step9_model!r}"
        )
    selected_step9_experiment = STEP9_MODEL_MAPPING[selected_step9_model]

    step15_policy_payload = load_json(resolve(STEP15_POLICY))
    strict_clean_features = [
        str(value)
        for value in (step15_policy_payload.get("feature_sets", {}) or {}).get(
            "strict_clean_30d", []
        )
    ]
    inductive_lineage = step15_policy_payload.get("inductive_feature_lineage", {}) or {}
    zh_inductive_cfg = (inductive_lineage.get("domains", {}) or {}).get(
        "zh_target_strict", {}
    ) or {}
    zh_inductive_features = str(zh_inductive_cfg.get("output_pair_features", "")).strip()
    reference_bundle = str(inductive_lineage.get("reference_bundle_output", "")).strip()
    inductive_manifest = str(inductive_lineage.get("manifest_output", "")).strip()
    if not zh_inductive_features or not reference_bundle or not inductive_manifest:
        raise ValueError(
            "Step15-v6 policy does not fully declare inductive Step11 feature lineage"
        )
    step9_payload = load_json(resolve(STEP9_SUMMARY))
    step9_policy_value = str(step9_payload.get("step9_policy_path", "")).strip()
    if not step9_policy_value:
        raise ValueError("Isolated Step9 summary does not record step9_policy_path")

    policy = copy.deepcopy(base)
    policy["version"] = (
        f"2026-07-12-step11-v6-explicit-runtime-v5-hash-closed::{args.validation_mode}"
    )
    policy["objective"] = (
        "Explicit Step11 validation for the validation-selected Step15-v6 scorer after Step12. "
        "This generated policy never enables auto model selection. Clean topology mode audits "
        "direct-proof retention without identifier weighting or hard-keep; identifier-assisted "
        "operational mode preserves direct proof edges through every graph filter stage."
    )
    policy.setdefault("baseline_reference", {})["step15_summary"] = STEP15_SUMMARY
    policy["baseline_reference"]["step15_policy"] = STEP15_POLICY
    policy["baseline_reference"]["step9_summary"] = STEP9_SUMMARY
    policy.setdefault("input_paths", {})["step15_summary"] = STEP15_SUMMARY
    policy["input_paths"]["step15_policy"] = STEP15_POLICY
    policy["input_paths"]["step9_summary"] = STEP9_SUMMARY
    policy["input_paths"]["pair_features"] = normalized_relative_path(
        zh_inductive_features
    )
    policy["input_paths"]["inductive_feature_manifest"] = normalized_relative_path(
        inductive_manifest
    )
    policy["input_paths"]["train_only_feature_reference"] = normalized_relative_path(
        reference_bundle
    )
    policy["input_paths"]["step5_frozen_labels"] = STEP5_ZH_LABELS
    policy["input_paths"]["step9_policy"] = normalized_relative_path(step9_policy_value)
    policy["input_paths"]["step12_v6_summary"] = normalized_relative_path(step12_path)
    policy["input_paths"]["step12_v6_policy"] = normalized_relative_path(
        str(step12.get("policy", ""))
    )
    policy["input_paths"]["step12_v6_model_metrics"] = raw_bge_control[
        "threshold_metrics_path"
    ]
    policy["input_dependencies"] = list(
        dict.fromkeys(
            [
                policy["input_paths"]["pair_features"],
                policy["input_paths"]["seller_profiles"],
                STEP5_ZH_LABELS,
                STEP15_SUMMARY,
                STEP15_POLICY,
                STEP9_SUMMARY,
                normalized_relative_path(step9_policy_value),
                normalized_relative_path(zh_inductive_features),
                normalized_relative_path(inductive_manifest),
                normalized_relative_path(reference_bundle),
                normalized_relative_path(step12_path),
                policy["input_paths"]["step12_v6_policy"],
                raw_bge_control["threshold_metrics_path"],
                normalized_relative_path(completion_path),
            ]
        )
    )
    summary_resolution = policy.setdefault("summary_resolution", {})
    allowed_current = summary_resolution.setdefault("allowed_current_main_summary_paths", {})
    allowed_current["step15_summary"] = [STEP15_SUMMARY]
    allowed_current["step9_summary"] = [STEP9_SUMMARY]
    selection = policy["scorer_selection"]
    selection["default_scorer_family"] = "step15"
    selection["default_step15_experiment_name"] = experiment
    selection["default_step15_phase"] = phase
    selection["default_step15_seeds"] = SEEDS
    selection["default_step9_experiment_name"] = selected_step9_experiment
    selection["default_step9_ratio"] = 1.0
    selection["default_step9_seeds"] = SEEDS
    selection["default_step9_seed"] = None
    selection["default_raw_feature_control"] = "raw_bge_m3_cosine"
    selection["raw_feature_controls"] = {
        "raw_bge_m3_cosine": raw_bge_control,
    }
    alias_key = f"{experiment}::{phase}::seed_mean"
    selection.setdefault("step15_scorer_token_aliases", {})[alias_key] = "step15_v6_final_selected_seed_mean"
    selection["step15_scorer_token_aliases"][
        "step15_v6_m0_all_at_once_binary::phase3_add_contact_url_noise::seed_mean"
    ] = "step15_v6_m0_seed_mean"
    selection.setdefault("step9_scorer_token_aliases", {})[
        f"{selected_step9_experiment}::100pct::seed_mean"
    ] = "step9_v6_strongest_clean_selected_seed_mean"
    selection["step9_scorer_token_aliases"][
        "identifier_augmented_few_shot_default_lr_l2::100pct::seed_mean"
    ] = "step9_v6_identifier_operational_seed_mean"
    selection["dynamic_mainline_candidates"]["enabled"] = False
    selection["publication_validation"] = {
        "selection_mode": "explicit_allowlist_only",
        "auto_selector_allowed": False,
        "status": "eligible_for_explicit_validation" if promoted else "diagnostic_only_not_promoted",
        "step12_summary": str(step12_path.relative_to(ROOT)),
        "selected_model_id": selected_model,
        "selected_experiment": experiment,
        "selected_phase": phase,
        "selected_seeds": SEEDS,
        "selected_step9_model_id": selected_step9_model,
        "selected_step9_experiment": selected_step9_experiment,
        "selected_step9_ratio": 1.0,
        "selected_step9_seeds": SEEDS,
        "graph_validation_mode": args.validation_mode,
        "allowed_scorer_families": (
            ["raw_feature", "step9", "step15"]
            if args.validation_mode == "clean_topology"
            else ["step9"]
        ),
        "allowed_scorer_tokens": (
            [
                "step15_v6_final_selected_seed_mean",
                "step15_v6_m0_seed_mean",
                "step9_v6_strongest_clean_selected_seed_mean",
                "raw_bge_m3_cosine",
            ]
            if args.validation_mode == "clean_topology"
            else ["step9_v6_identifier_operational_seed_mean"]
        ),
        "current_validation_summaries": [],
        "rule": "Run exactly the configured scorer, then pass its generated summary explicitly to step11_cluster_level_audit.py. Never glob reports/.",
    }
    policy["step15_v6_validation_gate"] = {
        "step12_promotion_eligible": promoted,
        "diagnostic_override_used": bool(args.allow_nonpromoted_diagnostic and not promoted),
        "current_test_role": "internal_development_test_not_prospective_final_holdout",
        "graph_validation_mode": args.validation_mode,
        "step12_summary_path": str(step12_path.relative_to(ROOT)),
        "step12_summary_sha256": sha256(step12_path),
        "step12_completion_manifest_path": str(completion_path.relative_to(ROOT)),
        "step12_completion_manifest_sha256": completion["manifest_sha256"],
        "active_manifest_path": normalized_relative_path(active_manifest_path),
        "active_manifest_sha256": active_manifest["manifest_sha256"],
        "frozen_input_hash_source": "step12_referenced_step15_v6_active_manifest",
        "frozen_input_hash_sources": [
            "step12_referenced_step15_v6_active_manifest",
            "step12_v6_completion_manifest",
        ],
        "posthoc_label_source": STEP5_ZH_LABELS,
        "posthoc_label_sha256": frozen_hashes.get(STEP5_ZH_LABELS),
    }
    required_paths = [
        policy["input_paths"]["pair_features"],
        policy["input_paths"]["seller_profiles"],
        STEP5_ZH_LABELS,
        STEP15_SUMMARY,
        STEP15_POLICY,
        STEP9_SUMMARY,
        normalized_relative_path(step9_policy_value),
        normalized_relative_path(inductive_manifest),
        normalized_relative_path(reference_bundle),
        normalized_relative_path(step12_path),
        policy["input_paths"]["step12_v6_policy"],
        raw_bge_control["threshold_metrics_path"],
        normalized_relative_path(completion_path),
    ]
    require_frozen_paths(frozen_hashes, required_paths)
    inductive_lineage_verification = verify_inductive_feature_lineage(
        resolve(inductive_manifest),
        resolve(reference_bundle),
        resolve(zh_inductive_features),
    )
    policy["step15_v6_validation_gate"]["frozen_input_file_sha256"] = dict(
        sorted(frozen_hashes.items())
    )
    policy["step15_v6_validation_gate"]["frozen_input_file_count"] = len(frozen_hashes)
    policy["step15_v6_validation_gate"][
        "inductive_feature_lineage_verification"
    ] = inductive_lineage_verification
    output_namespace = configure_validation_mode(
        policy, args.validation_mode, strict_clean_features
    )
    if not promoted:
        output_namespace = f"diagnostic_nonpromoted/{output_namespace}"
    policy["output_templates"] = {
        "scored_pairs": f"reports/step11_v6/{output_namespace}/step11_{{experiment_name}}_zh_target_strict_scored_pairs.csv",
        "clusters": f"reports/step11_v6/{output_namespace}/step11_{{experiment_name}}_zh_target_strict_clusters.threshold_{{threshold_token}}.csv",
        "summary": f"reports/step11_v6/{output_namespace}/step11_{{experiment_name}}_clustering_summary.json",
    }
    write_immutable_bytes(
        output_path,
        json_bytes(policy, ensure_ascii=False, indent=2),
    )
    print(
        json.dumps(
            {
                "output": str(output_path.relative_to(ROOT)),
                "promoted": promoted,
                "selected_model": selected_model,
                "experiment": experiment,
                "phase": phase,
                "seeds": SEEDS,
                "validation_mode": args.validation_mode,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
