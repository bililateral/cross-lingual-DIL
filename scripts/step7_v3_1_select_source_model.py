#!/usr/bin/env python3
"""Select Step7-v3.1 full-text encoders and attributable English pipelines."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np

import step7_v3_1_source_data as source
import step7_v3_1_build_sync_manifest as sync_builder
import step7_v3_1_common as common
import step7_v3_1_encode_chunked_models as encoder
import step7_v3_1_selection_core as core


SELECTOR_SCRIPT = Path(__file__).resolve()


def candidate_specs(policy: dict) -> list[dict]:
    shortcut_features = list(
        policy["pair_feature_roles"]["shortcut_audit_only_features"]
    )
    shortcut = set(shortcut_features)
    transfer = list(policy["pair_feature_roles"]["model_eligible_transfer_features"])
    matched = policy["selection_rule"]["pipeline_attribution"][
        "matched_no_encoder_control_by_tier"
    ]
    output = []
    for model_key, cfg in policy["embedding_models"].items():
        aggregate_features = common.aggregate_feature_names(cfg)
        for tier in policy["candidate_tiers"]:
            features = list(aggregate_features)
            if tier == "encoder_aggregates_plus_transfer":
                features.extend(transfer)
            if shortcut & set(features):
                raise ValueError("Step7-v3.1 selectable encoder pipeline uses a shortcut")
            output.append(
                {
                    "candidate_id": f"{model_key}__{tier}",
                    "model_key": model_key,
                    "tier": tier,
                    "feature_names": features,
                    "candidate_role": "encoder_pipeline",
                    "encoder_comparison_eligible": tier == "encoder_aggregates_only",
                    "m0_pipeline_eligible": True,
                    "attribution_control_only": False,
                    "shortcut_audit_only": False,
                    "matched_no_encoder_control": matched[tier],
                }
            )
    for control_name in policy["no_encoder_controls"]:
        features = [] if control_name == "intercept_only" else list(transfer)
        output.append(
            {
                "candidate_id": f"control__{control_name}",
                "model_key": None,
                "tier": control_name,
                "feature_names": features,
                "candidate_role": "no_encoder_control",
                "encoder_comparison_eligible": False,
                "m0_pipeline_eligible": False,
                "attribution_control_only": True,
                "shortcut_audit_only": False,
                "matched_no_encoder_control": None,
            }
        )
    output.append(
        {
            "candidate_id": "audit__shortcut_features_only",
            "model_key": None,
            "tier": "shortcut_features_only",
            "feature_names": shortcut_features,
            "candidate_role": "shortcut_audit_control",
            "encoder_comparison_eligible": False,
            "m0_pipeline_eligible": False,
            "attribution_control_only": False,
            "shortcut_audit_only": True,
            "matched_no_encoder_control": None,
        }
    )
    ids = [row["candidate_id"] for row in output]
    if len(output) != 13 or len(ids) != len(set(ids)):
        raise ValueError("Step7-v3.1 candidate universe drift")
    return output


def index_unique(rows: list[dict], role: str) -> dict[str, dict]:
    output = {row["pair_uid"]: row for row in rows}
    if len(output) != len(rows):
        raise ValueError(f"Step7-v3.1 duplicate pair UID in {role}")
    return output


def expected_gpu_paths(policy: dict) -> set[str]:
    return {common.relative(common.resolve(path)) for path in encoder.expected_payload_paths(policy)}


def verify_runtime_provenance(policy: dict) -> dict:
    public_paths = common.validate_source_public_artifacts(policy)
    development_paths = common.validate_source_development_artifacts(policy)
    source_manifest, field_rows = encoder.verify_source_preparation(policy)
    outputs = policy["outputs"]
    sync_path = common.resolve(outputs["gpu_sync_manifest"])
    bundle_path = common.resolve(outputs["gpu_output_manifest"])
    if not sync_path.is_file() or not bundle_path.is_file():
        raise FileNotFoundError("Step7-v3.1 GPU outputs have not been synchronized")
    sync = common.load_json(sync_path)
    expected_sync = sync_builder.build_payload(policy, common.DEFAULT_POLICY)
    if sync != expected_sync:
        raise ValueError("Step7-v3.1 GPU sync manifest is stale")
    bundle = common.load_json(bundle_path)
    if (
        bundle.get("step") != "step7_v3_1_label_free_gpu_output_bundle"
        or bundle.get("version") != policy["version"]
        or bundle.get("policy_sha256") != common.sha256_file(common.DEFAULT_POLICY)
        or bundle.get("policy_contract_sha256") != common.canonical_hash(policy)
        or bundle.get("generator_script_path")
        != "scripts/step7_v3_1_encode_chunked_models.py"
        or bundle.get("generator_script_sha256")
        != common.sha256_file(common.resolve("scripts/step7_v3_1_encode_chunked_models.py"))
        or bundle.get("gpu_sync_manifest_sha256") != common.sha256_file(sync_path)
        or bundle.get("source_preparation_manifest_sha256")
        != common.sha256_file(common.resolve(outputs["preparation_manifest"]))
        or bundle.get("label_or_raw_source_files_present_in_gpu_workspace") is not False
    ):
        raise ValueError("Step7-v3.1 GPU output provenance drift")
    records = bundle.get("files", [])
    expected_paths = expected_gpu_paths(policy)
    if (
        {record.get("path") for record in records} != expected_paths
        or len(records) != len(expected_paths)
        or int(bundle.get("file_count", -1)) != len(records)
        or int(bundle.get("total_file_bytes", -1))
        != sum(int(record["size_bytes"]) for record in records)
    ):
        raise ValueError("Step7-v3.1 GPU output bundle universe drift")
    for record in records:
        common.verify_file_record(record, "GPU output")
    return {
        "source_public_paths": {key: common.relative(path) for key, path in public_paths.items()},
        "source_development_paths": {
            key: common.relative(path) for key, path in development_paths.items()
        },
        "source_preparation": source_manifest,
        "field_rows": field_rows,
        "gpu_sync": sync,
        "gpu_output": bundle,
    }


def replay_model_scores(
    policy: dict,
    model_key: str,
    cfg: dict,
    pair_rows: list[dict],
    chunk_rows: list[dict],
    score_rows: list[dict],
    manifest: dict,
) -> tuple[dict, np.ndarray]:
    matrix_path = common.resolve(
        policy["outputs"]["embedding_matrix_template"].format(model_key=model_key)
    )
    try:
        matrix = np.load(matrix_path, allow_pickle=False)
    except (OSError, ValueError) as exc:
        raise ValueError(f"Step7-v3.1 cannot safely load {model_key} matrix") from exc
    if matrix.dtype != np.float32 or list(matrix.shape) != manifest.get("shape"):
        raise ValueError(f"Step7-v3.1 embedding matrix dtype/shape drift: {model_key}")
    replayed = common.compute_pair_score_rows(policy, cfg, matrix, chunk_rows, pair_rows)
    if len(replayed) != len(score_rows):
        raise ValueError(f"Step7-v3.1 pair-score replay count drift: {model_key}")
    tolerance = float(policy["aggregation"]["score_replay_absolute_tolerance"])
    names = common.aggregate_feature_names(cfg)
    maximum = 0.0
    for expected, observed in zip(replayed, score_rows, strict=True):
        if expected["pair_uid"] != observed["pair_uid"]:
            raise ValueError(f"Step7-v3.1 pair-score replay order drift: {model_key}")
        for name in names:
            difference = abs(float(expected[name]) - float(observed[name]))
            maximum = max(maximum, difference)
            if not math.isfinite(difference) or difference > tolerance:
                raise ValueError(
                    f"Step7-v3.1 aggregate does not replay: {model_key}/{name}"
                )
    return {
        "status": "pass",
        "matrix_dtype": str(matrix.dtype),
        "matrix_shape": list(matrix.shape),
        "pair_count_replayed": len(pair_rows),
        "feature_count_replayed": len(names),
        "absolute_tolerance": tolerance,
        "maximum_absolute_difference": maximum,
    }, matrix


def load_feature_bundle(policy: dict) -> tuple[list[dict], dict[str, dict], dict]:
    provenance = verify_runtime_provenance(policy)
    source_policy = common.source_policy(policy)
    pair_path = common.resolve(policy["outputs"]["pair_manifest"])
    safe_path = common.resolve(policy["outputs"]["safe_pair_features"])
    pair_rows = source.load_csv(pair_path)
    source.validate_public_pair_rows(source_policy, pair_rows)
    pair_uids = [row["pair_uid"] for row in pair_rows]
    safe_rows = source.load_csv(safe_path)
    source.validate_safe_pair_feature_rows(safe_rows)
    safe_index = index_unique(safe_rows, "safe pair features")
    if set(safe_index) != set(pair_uids):
        raise ValueError("Step7-v3.1 safe feature pair universe drift")
    features = {
        pair_uid: {
            name: float(safe_index[pair_uid][name]) for name in source.SAFE_FEATURE_NAMES
        }
        for pair_uid in pair_uids
    }
    outputs = policy["outputs"]
    chunk_rows = common.load_jsonl(common.resolve(outputs["shared_chunks"]))
    chunk_audit = common.validate_shared_chunk_rows(
        policy, provenance["field_rows"], chunk_rows
    )
    chunk_manifest = common.load_json(common.resolve(outputs["shared_chunks_manifest"]))
    registered_fingerprints = {
        model_key: {
            key: value
            for key, value in provenance["gpu_sync"]["model_directories"][
                model_key
            ].items()
            if key != "path"
        }
        for model_key in policy["embedding_models"]
    }
    if (
        chunk_manifest.get("step") != "step7_v3_1_build_complete_shared_chunks"
        or chunk_manifest.get("version") != policy["version"]
        or chunk_manifest.get("policy_sha256")
        != common.sha256_file(common.DEFAULT_POLICY)
        or chunk_manifest.get("policy_contract_sha256")
        != common.canonical_hash(policy)
        or chunk_manifest.get("generator_script_path")
        != "scripts/step7_v3_1_encode_chunked_models.py"
        or chunk_manifest.get("generator_script_sha256")
        != common.sha256_file(common.resolve("scripts/step7_v3_1_encode_chunked_models.py"))
        or chunk_manifest.get("gpu_sync_manifest_sha256")
        != common.sha256_file(common.resolve(outputs["gpu_sync_manifest"]))
        or chunk_manifest.get("source_preparation_manifest_sha256")
        != common.sha256_file(common.resolve(outputs["preparation_manifest"]))
        or chunk_manifest.get("labels_or_evidence_types_read") is not False
        or chunk_manifest.get("chunking_contract") != policy["shared_chunking"]
        or chunk_manifest.get("field_group_contract")
        != policy["clean_text_contract"]["field_groups"]
        or chunk_manifest.get("chunk_audit", {}).get("exact_character_reconstruction")
        is not True
        or chunk_manifest.get("shared_chunks", {}).get("sha256")
        != common.sha256_file(common.resolve(outputs["shared_chunks"]))
        or chunk_manifest.get("chunk_audit", {}).get("chunk_count") != len(chunk_rows)
        or chunk_audit["chunk_count"] != len(chunk_rows)
        or chunk_manifest.get("field_corpus_sha256")
        != common.sha256_file(common.resolve(outputs["field_corpus"]))
        or chunk_manifest.get("model_fingerprints") != registered_fingerprints
        or not str(chunk_manifest.get("transformers_version", "")).strip()
    ):
        raise ValueError("Step7-v3.1 shared chunk manifest drift")
    for key, value in chunk_audit.items():
        if chunk_manifest.get("chunk_audit", {}).get(key) != value:
            raise ValueError(f"Step7-v3.1 shared chunk audit replay drift: {key}")
    runtime = {"shared_chunks": chunk_manifest, "embedding_models": {}}
    expected_schema = ["pair_uid"]
    for model_key, cfg in policy["embedding_models"].items():
        score_path = common.resolve(
            outputs["embedding_pair_scores_template"].format(model_key=model_key)
        )
        manifest_path = common.resolve(
            outputs["embedding_manifest_template"].format(model_key=model_key)
        )
        matrix_path = common.resolve(
            outputs["embedding_matrix_template"].format(model_key=model_key)
        )
        manifest = common.load_json(manifest_path)
        names = common.aggregate_feature_names(cfg)
        required = {
            "step": "step7_v3_1_encode_full_text_shared_chunks",
            "version": policy["version"],
            "model_key": model_key,
            "repo_id": cfg["repo_id"],
            "local_path": cfg["local_path"],
            "aggregate_feature_names": names,
            "primary_raw_encoder_feature_name": common.primary_feature_name(cfg),
            "policy_sha256": common.sha256_file(common.DEFAULT_POLICY),
            "policy_contract_sha256": common.canonical_hash(policy),
            "generator_script_sha256": common.sha256_file(
                common.resolve("scripts/step7_v3_1_encode_chunked_models.py")
            ),
            "gpu_sync_manifest_sha256": common.sha256_file(
                common.resolve(outputs["gpu_sync_manifest"])
            ),
            "source_preparation_manifest_sha256": common.sha256_file(
                common.resolve(outputs["preparation_manifest"])
            ),
            "feature_generation_reads_label_values": False,
            "label_or_raw_source_files_present_in_gpu_workspace": False,
            "same_shared_chunks_for_all_models": True,
            "text_prefix": cfg["text_prefix"],
            "max_length": int(cfg["max_length"]),
            "shared_chunk_token_budget": int(
                policy["shared_chunking"][
                    "token_budget_including_model_prefix_and_special_tokens"
                ]
            ),
            "batch_size": int(cfg["batch_size"]),
            "runtime_token_lengths_replay_shared_manifest": True,
            "shared_chunks_sha256": common.sha256_file(
                common.resolve(outputs["shared_chunks"])
            ),
            "shared_chunks_manifest_sha256": common.sha256_file(
                common.resolve(outputs["shared_chunks_manifest"])
            ),
            "embedding_matrix_sha256": common.sha256_file(matrix_path),
            "pair_scores_sha256": common.sha256_file(score_path),
            "pair_count": len(pair_rows),
            "device": "cuda",
        }
        for key, value in required.items():
            if manifest.get(key) != value:
                raise ValueError(f"Step7-v3.1 encoder manifest drift: {model_key}/{key}")
        if manifest.get("chunk_uids") != [row["chunk_uid"] for row in chunk_rows]:
            raise ValueError(f"Step7-v3.1 chunk order drift: {model_key}")
        if manifest.get("layout_validation") != source.validate_sentence_transformer_layout(
            model_key, cfg
        ):
            raise ValueError(f"Step7-v3.1 model layout manifest drift: {model_key}")
        if manifest.get("model_fingerprint") != registered_fingerprints[model_key]:
            raise ValueError(f"Step7-v3.1 model/sync fingerprint drift: {model_key}")
        if manifest.get("shared_tokenizer_digest") != manifest.get(
            "runtime_sentence_transformer_tokenizer_digest"
        ):
            raise ValueError(f"Step7-v3.1 tokenizer replay drift: {model_key}")
        if (
            list(manifest.get("shape", []))[:1] != [len(chunk_rows)]
            or not isinstance(manifest.get("maximum_unit_norm_error"), (int, float))
            or float(manifest["maximum_unit_norm_error"]) > 1e-3
            or any(
                not str(manifest.get(field, "")).strip()
                for field in (
                    "gpu_name",
                    "torch_version",
                    "transformers_version",
                    "sentence_transformers_version",
                )
            )
        ):
            raise ValueError(f"Step7-v3.1 runtime embedding diagnostics drift: {model_key}")
        core.verify_model_fingerprint(model_key, manifest.get("model_fingerprint", {}), cfg)
        score_rows = source.load_csv(score_path)
        if not score_rows or list(score_rows[0]) != expected_schema + names:
            raise ValueError(f"Step7-v3.1 score schema drift: {model_key}")
        score_index = index_unique(score_rows, f"{model_key} aggregate scores")
        if set(score_index) != set(pair_uids):
            raise ValueError(f"Step7-v3.1 score pair universe drift: {model_key}")
        replay, _matrix = replay_model_scores(
            policy, model_key, cfg, pair_rows, chunk_rows, score_rows, manifest
        )
        for pair_uid in pair_uids:
            for name in names:
                value = float(score_index[pair_uid][name])
                if not math.isfinite(value):
                    raise ValueError(f"Step7-v3.1 non-finite aggregate: {model_key}/{name}")
                features[pair_uid][name] = value
        runtime["embedding_models"][model_key] = {**manifest, "numeric_replay": replay}
    return pair_rows, features, runtime


def load_split_rows(policy: dict, split: str, pair_rows: list[dict]) -> list[dict]:
    if split not in {"train", "valid"}:
        raise ValueError("Step7-v3.1 implementation keeps historical test sealed")
    key = f"{split}_labels"
    label_path = common.resolve(policy["outputs"][key])
    labels = source.load_csv(label_path)
    pair_index = {row["pair_uid"]: row for row in pair_rows}
    joined = []
    for label in labels:
        pair = pair_index.get(label["pair_uid"])
        if pair is None or pair["split_name"] != split or label["component_id"] != pair[
            "component_id"
        ]:
            raise ValueError(f"Step7-v3.1 private label/pair mismatch: {split}")
        joined.append({**pair, **label})
    expected = policy["supervision_boundary"]["expected_counts"][split]
    observed = Counter(row["review_label"] for row in joined)
    if len(joined) != int(expected["total"]) or observed != Counter(
        {"positive": int(expected["positive"]), "negative": int(expected["negative"])}
    ):
        raise ValueError(f"Step7-v3.1 private label boundary drift: {split}")
    return joined


def raw_encoder_results(
    policy: dict,
    train_rows: list[dict],
    valid_rows: list[dict],
    features: dict[str, dict],
) -> tuple[dict, dict]:
    train_labels = core.labels_array(train_rows)
    valid_labels = core.labels_array(valid_rows)
    train_weights = core.component_weights(
        train_rows, "component_equal_normalized_to_row_count"
    )
    public, internal = {}, {}
    for model_key, cfg in policy["embedding_models"].items():
        candidate_id = f"{model_key}__encoder_aggregates_only"
        name = common.primary_feature_name(cfg)
        train_scores = np.asarray(
            [features[row["pair_uid"]][name] for row in train_rows], dtype=np.float64
        )
        valid_scores = np.asarray(
            [features[row["pair_uid"]][name] for row in valid_rows], dtype=np.float64
        )
        threshold, threshold_selection = core.choose_threshold(
            train_labels, train_scores, train_weights
        )
        conditioned = core.shortcut_conditioned_component_equal_average_precision(
            valid_rows, valid_labels, valid_scores, policy, require_expected_strata=True
        )
        metrics = {
            "score_role": "raw_full_text_shared_chunk_field_equal_cosine_not_probability",
            "roc_auc": core.roc_auc(valid_labels, valid_scores),
            "average_precision": core.average_precision(valid_labels, valid_scores),
            "threshold_selected_on_raw_train_score": threshold,
            "threshold_metrics": core.threshold_metrics(
                valid_labels, valid_scores, threshold
            ),
            "labelled_pair_ranking": core.labelled_pair_ranking_metrics(
                valid_rows, valid_labels, valid_scores
            ),
        }
        public[candidate_id] = {
            "candidate_id": candidate_id,
            "model_key": model_key,
            "feature_name": name,
            "score_source": policy["selection_rule"]["encoder_selection"][
                "score_source"
            ],
            "fitted_head_used_for_encoder_ranking": False,
            "train_threshold_selection": threshold_selection,
            "primary_valid_metrics": metrics,
            "primary_valid_shortcut_conditioned_macro_component_equal_average_precision": conditioned[
                "macro_average_precision"
            ],
            "primary_valid_shortcut_conditioned_details": conditioned,
            "primary_valid_component_equal_average_precision": core.component_equal_average_precision(
                valid_rows, valid_labels, valid_scores
            ),
            "valid_evidence_slices": core.evidence_slice_rates(
                valid_rows, valid_scores, threshold, policy
            ),
        }
        internal[candidate_id] = {
            "primary_valid_scores": valid_scores,
            "sensitivity_valid_scores": valid_scores,
        }
    return public, internal


def runtime_input_fingerprints(policy: dict) -> dict[str, dict]:
    paths = {
        "policy": common.DEFAULT_POLICY,
        "source_data_policy": common.resolve(
            policy["source_data_policy"]["path"]
        ),
        "selector_script": SELECTOR_SCRIPT,
        "common_script": common.COMMON_SCRIPT,
        "source_data_script": common.resolve("scripts/step7_v3_1_source_data.py"),
        "selection_core": common.resolve("scripts/step7_v3_1_selection_core.py"),
        "field_corpus": common.resolve(policy["outputs"]["field_corpus"]),
        "source_preparation_manifest": common.resolve(
            policy["outputs"]["preparation_manifest"]
        ),
        "gpu_sync_manifest": common.resolve(policy["outputs"]["gpu_sync_manifest"]),
        "gpu_output_manifest": common.resolve(policy["outputs"]["gpu_output_manifest"]),
        "shared_chunks": common.resolve(policy["outputs"]["shared_chunks"]),
        "shared_chunks_manifest": common.resolve(
            policy["outputs"]["shared_chunks_manifest"]
        ),
    }
    for role in (
        "pair_manifest",
        "train_feature_reference",
        "safe_pair_features",
        "train_labels",
        "valid_labels",
        "development_labels_manifest",
    ):
        paths[f"source:{role}"] = common.resolve(policy["outputs"][role])
    for model_key in policy["embedding_models"]:
        paths[f"embedding_matrix:{model_key}"] = common.resolve(
            policy["outputs"]["embedding_matrix_template"].format(model_key=model_key)
        )
        paths[f"embedding_manifest:{model_key}"] = common.resolve(
            policy["outputs"]["embedding_manifest_template"].format(model_key=model_key)
        )
        paths[f"embedding_scores:{model_key}"] = common.resolve(
            policy["outputs"]["embedding_pair_scores_template"].format(
                model_key=model_key
            )
        )
    return {
        key: {
            "path": common.relative(path),
            "size_bytes": path.stat().st_size,
            "sha256": common.sha256_file(path),
        }
        for key, path in paths.items()
    }


def run_selection(policy: dict) -> tuple[dict, list[dict], list[dict], dict]:
    pair_rows, features, runtime_manifests = load_feature_bundle(policy)
    train_rows = load_split_rows(policy, "train", pair_rows)
    valid_rows = load_split_rows(policy, "valid", pair_rows)
    core.attach_shortcut_control_strata(train_rows, features)
    core.attach_shortcut_control_strata(valid_rows, features)
    train_labels = core.labels_array(train_rows)
    valid_labels = core.labels_array(valid_rows)
    valid_component_weights = core.component_weights(
        valid_rows, "component_equal_normalized_to_row_count"
    )
    specs = candidate_specs(policy)
    primary_mode = policy["training"]["primary_sample_weight"]
    sensitivity_mode = policy["training"]["sensitivity_sample_weight"]
    internal, public = {}, {}
    train_predictions, valid_predictions = [], []
    for spec in specs:
        train_matrix = core.matrix_for_rows(train_rows, features, spec["feature_names"])
        valid_matrix = core.matrix_for_rows(valid_rows, features, spec["feature_names"])
        primary = core.tune_and_fit(
            train_rows, train_matrix, spec["feature_names"], policy, primary_mode
        )
        sensitivity = core.tune_and_fit(
            train_rows, train_matrix, spec["feature_names"], policy, sensitivity_mode
        )
        primary_valid = core.apply_logistic(valid_matrix, primary["final_train_artifact"])
        sensitivity_valid = core.apply_logistic(
            valid_matrix, sensitivity["final_train_artifact"]
        )
        primary_conditioned = core.shortcut_conditioned_component_equal_average_precision(
            valid_rows, valid_labels, primary_valid, policy, require_expected_strata=True
        )
        sensitivity_conditioned = core.shortcut_conditioned_component_equal_average_precision(
            valid_rows,
            valid_labels,
            sensitivity_valid,
            policy,
            require_expected_strata=True,
        )
        slices = core.evidence_slice_rates(
            valid_rows, primary_valid, primary["selected_threshold"], policy
        )
        internal[spec["candidate_id"]] = {
            "spec": spec,
            "primary": primary,
            "sensitivity": sensitivity,
            "primary_valid_scores": primary_valid,
            "sensitivity_valid_scores": sensitivity_valid,
            "valid_evidence_slices": slices,
        }
        public[spec["candidate_id"]] = {
            **spec,
            "primary_training": core.without_score_arrays(primary),
            "primary_valid_metrics": core.full_metrics(
                valid_rows, valid_labels, primary_valid, primary["selected_threshold"]
            ),
            "primary_valid_shortcut_conditioned_macro_component_equal_average_precision": primary_conditioned[
                "macro_average_precision"
            ],
            "primary_valid_shortcut_conditioned_details": primary_conditioned,
            "primary_valid_component_equal_average_precision": core.weighted_average_precision(
                valid_labels, primary_valid, valid_component_weights
            ),
            "uniform_weight_sensitivity_training": core.without_score_arrays(
                sensitivity
            ),
            "uniform_weight_sensitivity_valid_metrics": core.full_metrics(
                valid_rows,
                valid_labels,
                sensitivity_valid,
                sensitivity["selected_threshold"],
            ),
            "uniform_weight_sensitivity_valid_shortcut_conditioned_macro_component_equal_average_precision": sensitivity_conditioned[
                "macro_average_precision"
            ],
            "uniform_weight_sensitivity_valid_shortcut_conditioned_details": sensitivity_conditioned,
            "uniform_weight_sensitivity_valid_component_equal_average_precision": core.weighted_average_precision(
                valid_labels, sensitivity_valid, valid_component_weights
            ),
            "valid_evidence_slices": slices,
        }
        for row, score in zip(train_rows, primary["train_oof_scores"], strict=True):
            train_predictions.append(
                {
                    "candidate_id": spec["candidate_id"],
                    "pair_uid": row["pair_uid"],
                    "review_label": row["review_label"],
                    "evidence_type": row["evidence_type"],
                    "component_id": row["component_id"],
                    "oof_prob_positive": f"{float(score):.12f}",
                    "threshold_from_train_oof": f"{primary['selected_threshold']:.12f}",
                    "predicted_positive": int(score >= primary["selected_threshold"]),
                }
            )
        for row, score in zip(valid_rows, primary_valid, strict=True):
            valid_predictions.append(
                {
                    "candidate_id": spec["candidate_id"],
                    "pair_uid": row["pair_uid"],
                    "review_label": row["review_label"],
                    "evidence_type": row["evidence_type"],
                    "component_id": row["component_id"],
                    "prob_positive": f"{float(score):.12f}",
                    "threshold_from_train_oof": f"{primary['selected_threshold']:.12f}",
                    "predicted_positive": int(score >= primary["selected_threshold"]),
                }
            )

    encoder_ids = [row["candidate_id"] for row in specs if row["encoder_comparison_eligible"]]
    m0_ids = [row["candidate_id"] for row in specs if row["m0_pipeline_eligible"]]
    control_ids = [row["candidate_id"] for row in specs if row["attribution_control_only"]]
    if len(encoder_ids) != 5 or len(m0_ids) != 10 or len(control_ids) != 2:
        raise ValueError("Step7-v3.1 candidate role counts drift")
    raw_public, raw_internal = raw_encoder_results(
        policy, train_rows, valid_rows, features
    )
    encoder_selection = core.assess_candidate_group(
        encoder_ids,
        raw_public,
        raw_internal,
        valid_rows,
        policy,
        training_weight_sensitivity_applicable=False,
    )
    encoder_selection["score_source"] = policy["selection_rule"]["encoder_selection"][
        "score_source"
    ]
    encoder_selection["fitted_head_used_for_ranking"] = False
    if encoder_selection["unique_winner"]:
        encoder_selection["selection_outcome"] = "unique_full_text_encoder_winner"
        encoder_selection["carry_forward_encoder_candidates"] = [
            encoder_selection["top_candidate"]
        ]
    else:
        encoder_selection["selection_outcome"] = (
            "no_unique_full_text_encoder_winner_carry_top_two_without_e5_privilege"
        )
        encoder_selection["carry_forward_encoder_candidates"] = encoder_selection[
            "candidate_ranking"
        ][:2]

    pipeline_selection = core.assess_candidate_group(
        m0_ids,
        public,
        internal,
        valid_rows,
        policy,
        training_weight_sensitivity_applicable=True,
    )
    spec_by_id = {row["candidate_id"]: row for row in specs}
    top_id = pipeline_selection["top_candidate"]
    runner_id = pipeline_selection["runner_up_candidate"]
    top_spec = spec_by_id[top_id]
    matched_id = top_spec["matched_no_encoder_control"]
    attribution_cfg = policy["selection_rule"]["pipeline_attribution"]
    encoder_increment = float(
        public[top_id][
            "primary_valid_shortcut_conditioned_macro_component_equal_average_precision"
        ]
        - public[matched_id][
            "primary_valid_shortcut_conditioned_macro_component_equal_average_precision"
        ]
    )
    matched_bootstrap = core.grouped_bootstrap_ap_delta(
        valid_rows,
        internal[top_id]["primary_valid_scores"],
        internal[matched_id]["primary_valid_scores"],
        policy,
    )
    attribution_checks = {
        "top_pipeline_contains_encoder": {
            "observed_candidate_role": top_spec["candidate_role"],
            "required_candidate_role": "encoder_pipeline",
            "pass": top_spec["candidate_role"] == "encoder_pipeline",
        },
        "encoder_increment_shortcut_conditioned_ap": {
            "matched_control": matched_id,
            "observed": encoder_increment,
            "required_minimum": float(
                attribution_cfg["encoder_increment_shortcut_conditioned_ap_at_least"]
            ),
            "pass": encoder_increment
            >= float(
                attribution_cfg["encoder_increment_shortcut_conditioned_ap_at_least"]
            ),
        },
        "encoder_increment_grouped_bootstrap_ci_lower": {
            "matched_control": matched_id,
            "observed": matched_bootstrap["ci95_lower"],
            "required_above": float(
                attribution_cfg["encoder_increment_grouped_bootstrap_ci95_lower_above"]
            ),
            "pass": matched_bootstrap["ci95_lower"]
            > float(
                attribution_cfg["encoder_increment_grouped_bootstrap_ci95_lower_above"]
            ),
        },
    }
    attribution_pass = all(value["pass"] for value in attribution_checks.values())
    pipeline_selection["encoder_attribution"] = {
        "matched_no_encoder_control": matched_id,
        "matched_control_grouped_bootstrap": matched_bootstrap,
        "checks": attribution_checks,
        "pass": attribution_pass,
    }
    if pipeline_selection["unique_winner"] and attribution_pass:
        outcome = "unique_attributable_full_text_encoder_pipeline_winner"
        carry = [top_id]
    elif not attribution_pass:
        outcome = "no_attributable_full_text_encoder_pipeline_winner_carry_top_two"
        carry = pipeline_selection["candidate_ranking"][:2]
    else:
        outcome = "no_unique_full_text_encoder_pipeline_winner_carry_top_two"
        carry = pipeline_selection["candidate_ranking"][:2]
    pipeline_selection["selection_outcome"] = outcome
    pipeline_selection["carry_forward_to_step28"] = carry

    e5_id = "multilingual_e5_large__encoder_aggregates_only"
    shortcut_audit_id = "audit__shortcut_features_only"
    identity_scores = np.asarray(
        [float(row["identity_rule_control_score"]) for row in valid_rows], dtype=np.float64
    )
    summary = {
        "step": "step7_v3_1_select_full_text_source_model",
        "version": policy["version"],
        "scope": policy["result_scope"],
        "train_counts": dict(Counter(row["review_label"] for row in train_rows)),
        "valid_counts": dict(Counter(row["review_label"] for row in valid_rows)),
        "candidate_count": len(specs),
        "encoder_comparison_candidate_count": len(encoder_ids),
        "m0_pipeline_candidate_count": len(m0_ids),
        "attribution_control_candidate_count": len(control_ids),
        "shortcut_audit_candidate_count": 1,
        "candidate_ranking": pipeline_selection["candidate_ranking"],
        "top_candidate": top_id,
        "runner_up_candidate": runner_id,
        "selection_outcome": outcome,
        "carry_forward_to_step28": carry,
        "encoder_only_selection": encoder_selection,
        "raw_encoder_comparison_results": raw_public,
        "m0_pipeline_selection": pipeline_selection,
        "e5_continuity_control": {
            "candidate_id": e5_id,
            "encoder_only_rank": encoder_selection["candidate_ranking"].index(e5_id)
            + 1,
            "changes_ranking_or_carry_forward": False,
        },
        "shortcut_feature_audit": {
            "train": core.shortcut_feature_label_association_audit(
                train_rows, features, policy
            ),
            "valid": core.shortcut_feature_label_association_audit(
                valid_rows, features, policy
            ),
            "audit_control_candidate": shortcut_audit_id,
            "eligible_for_selection": False,
        },
        "identity_rule_control": {
            "eligible_for_m0": False,
            "valid_metrics": core.full_metrics(
                valid_rows, valid_labels, identity_scores, threshold=0.5
            ),
        },
        "candidates": public,
        "runtime_model_manifests": runtime_manifests,
        "historical_test_label_values_parsed_during_selection": False,
        "historical_test_label_file_touched_during_selection": False,
        "test_metrics_used_for_selection": False,
        "prospective_claim_allowed": False,
    }
    freeze = {
        "step": "step7_v3_1_frozen_full_text_source_selection",
        "version": policy["version"],
        "policy_contract_sha256": common.canonical_hash(policy),
        "candidate_specs_sha256": common.canonical_hash(specs),
        "selection_outcome": outcome,
        "carry_forward_to_step28": carry,
        "top_candidate": top_id,
        "runner_up_candidate": runner_id,
        "encoder_selection_outcome": encoder_selection["selection_outcome"],
        "encoder_top_candidate": encoder_selection["top_candidate"],
        "encoder_runner_up_candidate": encoder_selection["runner_up_candidate"],
        "encoder_selection_sha256": common.canonical_hash(encoder_selection),
        "raw_encoder_comparison_results_sha256": common.canonical_hash(raw_public),
        "m0_pipeline_selection_sha256": common.canonical_hash(pipeline_selection),
        "e5_continuity_changes_selection": False,
        "runtime_inputs": runtime_input_fingerprints(policy),
        "historical_test_label_values_parsed_during_selection": False,
        "historical_test_label_file_hashed_during_selection": False,
        "historical_test_execution_implemented": False,
    }
    return summary, valid_predictions, train_predictions, freeze


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", default=str(common.DEFAULT_POLICY))
    parser.add_argument("--stage", choices=("select",), default="select")
    parser.add_argument("--validate-config-only", action="store_true")
    args = parser.parse_args()
    policy = common.load_json(common.resolve(args.policy))
    common.validate_policy(policy)
    specs = candidate_specs(policy)
    common.validate_source_development_artifacts(policy)
    if args.validate_config_only:
        print(
            json.dumps(
                {
                    "status": "pass",
                    "candidate_count": len(specs),
                    "encoder_comparison_candidate_count": 5,
                    "m0_pipeline_candidate_count": 10,
                    "attribution_control_candidate_count": 2,
                    "shortcut_audit_candidate_count": 1,
                    "candidate_ids": [row["candidate_id"] for row in specs],
                    "train_rows": 401,
                    "valid_rows": 152,
                    "historical_test_supported_in_this_round": False,
                    "gpu_required_after_scores_exist": False,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return
    summary, valid_predictions, train_predictions, freeze = run_selection(policy)
    outputs = policy["outputs"]
    summary_path = common.resolve(outputs["selection_summary"])
    valid_path = common.resolve(outputs["valid_predictions"])
    train_path = common.resolve(outputs["train_oof_predictions"])
    freeze_path = common.resolve(outputs["frozen_selection_manifest"])
    common.write_json_immutable(summary_path, summary)
    common.write_csv_immutable(valid_path, valid_predictions)
    common.write_csv_immutable(train_path, train_predictions)
    freeze["selection_summary"] = {
        "path": common.relative(summary_path),
        "size_bytes": summary_path.stat().st_size,
        "sha256": common.sha256_file(summary_path),
    }
    freeze["valid_predictions"] = {
        "path": common.relative(valid_path),
        "size_bytes": valid_path.stat().st_size,
        "sha256": common.sha256_file(valid_path),
    }
    freeze["train_oof_predictions"] = {
        "path": common.relative(train_path),
        "size_bytes": train_path.stat().st_size,
        "sha256": common.sha256_file(train_path),
    }
    common.write_json_immutable(freeze_path, freeze)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
