#!/usr/bin/env python3
"""Load and validate the frozen V9.3 method-qualification policy."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping

import step28_v13_common as common
import step28_v13_v1_13_audit_design_v9_3 as audit_design
import step28_v13_v1_13_quality_text_probe_views_v9 as text_views
import step28_v13_v1_13_structure_matrix_v9_3 as structure_matrix


POLICY_PATH = "schema/step28_v13_v1_13_method_qualification_policy_v9_3.json"
EXPECTED_VERSION = (
    "2026-08-27-step28-v13-v1-13-method-qualification-policy-"
    "v9-3-r2-user-accepted-residual-22"
)
EXPECTED_REGISTERED_NEGATIVE_PLAN_ROOT = (
    "reports/step28_v13_v1_13_balanced_schedule_v9_3/"
    "registered_negative_bounded_preflight_r2_20260827"
)
EXPECTED_PREBUILD_STRUCTURE_GATE_CONTRACT = {
    "version": (
        "2026-08-27-step28-v13-v1-13-prebuild-structure-gate-"
        "v9-3-r2-user-accepted-residual-22"
    ),
    "result_path": (
        "reports/step28_v13_v1_13_balanced_schedule_v9_3/"
        "registered_negative_structure_gate_r2_20260827/"
        "structure_gate_result.json"
    ),
    "required_pass_status": (
        "PASS_PREBUILD_STRUCTURE_GATE_NOT_METHOD_ROOT_OR_TRAINING_QUALIFIED"
    ),
    "finite_preregistered_projection_map_sha256": (
        "8b49db61cc9195bb0d1aaaa11c041d66e007a04ee7d21c0e56671e151268c261"
    ),
    "coverage_kind": (
        "finite_preregistered_projection_not_cellwise_or_"
        "cross_view_interaction_complete"
    ),
    "matrix_views": ["seller_slot", "noise_visible"],
    "models_per_view": ["logistic_l2", "hist_gradient_boosting_depth2"],
    "matrix_concatenation_forbidden": True,
    "must_pass_before_method_root": True,
    "failure_closes_version": True,
    "m0_m1_m2_m3_training_authorized": False,
}
WORLD_COUNTS = {"train": 500, "development": 500, "audit_a": 2, "audit_b": 2}
PROBE_MODELS = ("logistic_l2", "hist_gradient_boosting_depth2")


class MethodPolicyV93Error(common.ContractError):
    """Raised when the method policy or runtime contract has drifted."""


def _validate_frozen_file_pins(rows: Any) -> None:
    if not isinstance(rows, list) or len(rows) != 49:
        raise MethodPolicyV93Error("Frozen file pin registry cardinality drift")
    paths = [row.get("path") if isinstance(row, Mapping) else None for row in rows]
    if (
        common.canonical_sha256(paths)
        != "efd87c51567d3cde6f5d3b86edeeb97da8897447362c962d97ea1307101bccce"
        or len(paths) != len(set(paths))
    ):
        raise MethodPolicyV93Error("Frozen file pin path registry drift")
    for row in rows:
        if set(row) != {"path", "size_bytes", "sha256", "canonical_self_sha256"}:
            raise MethodPolicyV93Error("Frozen file pin schema drift")
        path = common.repo_path(str(row["path"]))
        if (
            not path.is_file()
            or type(row["size_bytes"]) is not int
            or path.stat().st_size != row["size_bytes"]
            or common.sha256_file(path) != row["sha256"]
        ):
            raise MethodPolicyV93Error(f"Frozen file pin drift: {row['path']}")
        canonical = row["canonical_self_sha256"]
        if canonical is not None:
            value = common.load_json(path)
            if value.get("canonical_self_sha256") != canonical:
                raise MethodPolicyV93Error(
                    f"Frozen JSON canonical pin drift: {row['path']}"
                )


def expected_observation_registry() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    def add(
        observation_id: str,
        *,
        object_name: str,
        split: str,
        surface: str,
        view: str,
        model: str,
        metric: str,
        role: str,
    ) -> None:
        rows.append(
            {
                "observation_id": observation_id,
                "object": object_name,
                "split": split,
                "surface": surface,
                "view": view,
                "model": model,
                "metric": metric,
                "role": role,
                "expected_occurrences": 1,
            }
        )

    for name in (
        "public_and_private_schema_closure",
        "artificial_code_zero_occurrence",
        "private_coordinate_nonintervention",
        "document_collision_closure",
        "split_isolation",
        "identity_asset_and_shared_relation_positive_control",
        "style_counterfactual_positive_control",
        "noise_counterfactual_identity_invariance",
    ):
        add(
            f"structural::{name}::boolean_pass",
            object_name="structural_and_mechanism_contract",
            split="all_1004_label_free",
            surface="not_applicable",
            view=name,
            model="deterministic_validator",
            metric="boolean_pass",
            role="qualification_hard_gate",
        )

    for view in ("seller_slot", "noise_visible"):
        add(
            f"structure::{view}::univariate::maximum_symmetric_roc_auc",
            object_name="structure_coordinate_family",
            split="development",
            surface="label_free_structure_frozen_before_truth",
            view=view,
            model="univariate_raw_column_scan",
            metric="maximum_symmetric_roc_auc",
            role="qualification_hard_gate",
        )
        for model in PROBE_MODELS:
            for metric in ("symmetric_roc_auc", "average_precision"):
                add(
                    f"structure::{view}::{model}::{metric}",
                    object_name="structure_coordinate_family",
                    split="development",
                    surface="label_free_structure_frozen_before_truth",
                    view=view,
                    model=model,
                    metric=metric,
                    role="qualification_hard_gate",
                )
    for metric in (
        "maximum_symmetric_roc_auc",
        "maximum_average_precision_uplift",
        "bootstrap_95_upper_symmetric_roc_auc",
        "bootstrap_95_upper_average_precision_uplift",
    ):
        add(
            f"structure::family::{metric}",
            object_name="structure_coordinate_family",
            split="development_world_bootstrap" if metric.startswith("bootstrap") else "development",
            surface="label_free_structure_frozen_before_truth",
            view="seller_slot_and_noise_visible_separate_models_maximized",
            model="family_maximum_without_matrix_concatenation",
            metric=metric,
            role="qualification_hard_gate",
        )

    for surface, role in (
        ("style_deranged", "qualification_hard_gate"),
        ("original_author", "descriptive_only"),
    ):
        for view in text_views.VIEW_ORDER:
            add(
                f"text::{surface}::{view}::univariate::maximum_symmetric_roc_auc",
                object_name="text_shortcut_family",
                split="development",
                surface=surface,
                view=view,
                model="univariate_raw_column_scan",
                metric="maximum_symmetric_roc_auc",
                role=role,
            )
            for model in PROBE_MODELS:
                for metric in ("symmetric_roc_auc", "average_precision"):
                    add(
                        f"text::{surface}::{view}::{model}::{metric}",
                        object_name="text_shortcut_family",
                        split="development",
                        surface=surface,
                        view=view,
                        model=model,
                        metric=metric,
                        role=role,
                    )
    for metric in (
        "maximum_symmetric_roc_auc",
        "maximum_average_precision_uplift",
        "bootstrap_95_upper_symmetric_roc_auc",
        "bootstrap_95_upper_average_precision_uplift",
    ):
        add(
            f"text::style_deranged::family::{metric}",
            object_name="text_shortcut_family",
            split="development_world_bootstrap" if metric.startswith("bootstrap") else "development",
            surface="style_deranged",
            view="all_seven_views_maximized",
            model="family_maximum",
            metric=metric,
            role="qualification_hard_gate",
        )
    identifiers = [row["observation_id"] for row in rows]
    if len(rows) != 96 or len(identifiers) != len(set(identifiers)):
        raise MethodPolicyV93Error("Internal observation registry drift")
    return rows


def expected_probe_model_contract() -> dict[str, Any]:
    return {
        "logistic_l2": {
            "preprocessing": "StandardScaler_fit_on_train_only_then_transform_train_and_development",
            "C": 1,
            "class_weight": None,
            "dual": False,
            "fit_intercept": True,
            "intercept_scaling": 1,
            "l1_ratio": None,
            "max_iter": 10000,
            "n_jobs": None,
            "penalty": "l2",
            "random_state": 793820367,
            "solver": "lbfgs",
            "tol": 1e-10,
            "verbose": 0,
            "warm_start": False,
        },
        "hist_gradient_boosting_depth2": {
            "class": "sklearn.ensemble.HistGradientBoostingClassifier",
            "preprocessing": "raw_unstandardized_float64",
            "categorical_features": "from_dtype",
            "class_weight": None,
            "early_stopping": False,
            "interaction_cst": None,
            "l2_regularization": 1,
            "learning_rate": 0.03,
            "loss": "log_loss",
            "max_bins": 255,
            "max_depth": 2,
            "max_features": 1.0,
            "max_iter": 200,
            "max_leaf_nodes": 31,
            "min_samples_leaf": 20,
            "monotonic_cst": None,
            "n_iter_no_change": 10,
            "random_state": 793820367,
            "scoring": "loss",
            "tol": 1e-7,
            "validation_fraction": 0.1,
            "verbose": 0,
            "warm_start": False,
        },
        "runtime": {
            "float_dtype": "float64",
            "threads": 1,
            "positive_label": 1,
            "negative_label": 0,
            "score_column": "predict_proba[:,1]",
            "sample_weight_forbidden": True,
            "class_reweighting_forbidden": True,
            "development_fit_forbidden": True,
        },
    }


def _validate_self_hash(policy: Mapping[str, Any]) -> None:
    expected = policy.get("canonical_self_sha256")
    if not isinstance(expected, str) or len(expected) != 64:
        raise MethodPolicyV93Error("Method policy self hash is absent")
    canonical = deepcopy(dict(policy))
    canonical["canonical_self_sha256"] = None
    if common.canonical_sha256(canonical) != expected:
        raise MethodPolicyV93Error("Method policy canonical self hash drift")


def validate_policy(policy: Mapping[str, Any]) -> dict[str, Any]:
    _validate_self_hash(policy)
    status = policy.get("status")
    if status == "PREEXECUTION_IMPLEMENTATION_NOT_TRAINING_QUALIFIED":
        if "frozen_file_pins" in policy:
            raise MethodPolicyV93Error("Preexecution policy already contains formal pins")
    elif status == "FROZEN_METHOD_QUALIFICATION_INPUTS_NOT_TRAINING_QUALIFIED":
        _validate_frozen_file_pins(policy.get("frozen_file_pins"))
    else:
        raise MethodPolicyV93Error("Method policy status drift")
    scope = policy.get("scientific_scope")
    gates = policy.get("quality_gates")
    text = policy.get("text_probe_contract")
    bootstrap = policy.get("bootstrap")
    frozen = policy.get("frozen_inputs")
    prebuild_gate = policy.get("prebuild_structure_gate_contract")
    registry = policy.get("observation_registry")
    if (
        policy.get("version") != EXPECTED_VERSION
        or not isinstance(scope, Mapping)
        or scope.get("world_counts") != WORLD_COUNTS
        or scope.get("m0_m1_m2_m3_training_authorized") is not False
        or scope.get("pair_count_per_world") != 378
        or scope.get("positive_pair_count_per_world") != 20
        or scope.get("negative_pair_count_per_world") != 358
        or not isinstance(frozen, Mapping)
        or frozen.get("registered_negative_plan_root")
        != EXPECTED_REGISTERED_NEGATIVE_PLAN_ROOT
        or prebuild_gate != EXPECTED_PREBUILD_STRUCTURE_GATE_CONTRACT
        or policy.get("structure_matrix_contract")
        != structure_matrix.contract_payload()
        or not isinstance(gates, Mapping)
        or gates.get("maximum_single_feature_symmetric_roc_auc") != 0.52
        or gates.get("maximum_family_symmetric_roc_auc") != 0.53
        or gates.get("maximum_family_average_precision_uplift") != 0.01
        or gates.get("bootstrap_95_upper_symmetric_roc_auc") != 0.53
        or gates.get("bootstrap_95_upper_average_precision_uplift") != 0.015
        or gates.get("audit_a_b_truth_read_count") != 0
        or not isinstance(text, Mapping)
        or text.get("rows_per_world") != 372
        or text.get("positives_per_world") != 20
        or text.get("views") != list(text_views.VIEW_ORDER)
        or text.get("feature_counts") != {
            name: width
            for name, width in zip(
                text_views.VIEW_ORDER, text_views.EXPECTED_WIDTHS, strict=True
            )
        }
        or text.get("feature_name_sha256s") != text_views.EXPECTED_NAME_HASHES
        or text.get("models")
        != ["logistic_l2", "hist_gradient_boosting_depth2"]
        or policy.get("probe_models") != expected_probe_model_contract()
        or type(
            policy["probe_models"]["hist_gradient_boosting_depth2"][
                "max_features"
            ]
        )
        is not float
        or not isinstance(bootstrap, Mapping)
        or bootstrap.get("replicates") != 9999
        or bootstrap.get("seed") != 281320260810
        or bootstrap.get("development_world_count") != 500
        or bootstrap.get("refit_models_inside_bootstrap") is not False
        or bootstrap.get("draws_raw_i8_c_sha256")
        != "111b1338cc607c6bd78bad88efe47606ffa2230e9cc764eec940e84f86e56661"
        or registry != expected_observation_registry()
    ):
        raise MethodPolicyV93Error("Method policy frozen scientific contract drift")
    parent_policy = common.load_json(
        common.repo_path(str(frozen["parent_policy"]))
    )
    if frozen.get("text_template") != parent_policy["template_library"]["path"]:
        raise MethodPolicyV93Error("Method policy text-template lineage drift")
    template = common.load_json(
        common.repo_path(str(frozen["text_template"]))
    )
    libraries = template.get("split_libraries", {})
    if (
        set(libraries) != set(WORLD_COUNTS)
        or max(len(row["title_skeletons"]) for row in libraries.values())
        != structure_matrix.TITLE_TEMPLATE_COUNT
        or max(len(row["description_skeletons"]) for row in libraries.values())
        != structure_matrix.DESCRIPTION_TEMPLATE_COUNT
        or len(template["generic_lexicon"]["categories"])
        != structure_matrix.CATEGORY_COUNT
        or len(template["generic_lexicon"]["service"])
        != structure_matrix.SERVICE_COUNT
        or len(template["generic_lexicon"]["delivery"])
        != structure_matrix.DELIVERY_COUNT
    ):
        raise MethodPolicyV93Error("Structure-matrix template domain drift")
    blind_pin = frozen.get("blind_audit_design")
    if not isinstance(blind_pin, Mapping):
        raise MethodPolicyV93Error("Blind-audit design pin is absent")
    blind_path = common.repo_path(str(blind_pin.get("path", "")))
    signature_path = common.repo_path(str(frozen["joint_noise_signature"]))
    if (
        not blind_path.is_file()
        or not signature_path.is_file()
        or common.sha256_file(blind_path) != blind_pin.get("file_sha256")
    ):
        raise MethodPolicyV93Error("Blind-audit design file pin drift")
    blind_payload = common.load_json(blind_path)
    blind_audit = audit_design.validate_payload(
        blind_payload, common.load_json(signature_path)
    )
    if (
        blind_audit["canonical_self_sha256"]
        != blind_pin.get("canonical_self_sha256")
    ):
        raise MethodPolicyV93Error("Blind-audit design canonical pin drift")
    return dict(policy)


def load_policy(path: Path | None = None) -> dict[str, Any]:
    resolved = common.repo_path(POLICY_PATH) if path is None else path.resolve()
    policy = common.load_json(resolved)
    validate_policy(policy)
    return policy
