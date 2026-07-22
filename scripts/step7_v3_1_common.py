#!/usr/bin/env python3
"""Shared contracts for Step7-v3.1 full-text, shared-chunk selection."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

import step7_v3_1_source_data as source


ROOT = source.ROOT
DEFAULT_POLICY = ROOT / "schema" / "step7_v3_1_full_text_chunked_selection_policy.json"
COMMON_SCRIPT = Path(__file__).resolve()

EXPECTED_VERSION = "2026-07-22-step7-v3.1-full-text-shared-chunks-v1"
EXPECTED_FIELDS = [
    "category_concat_top",
    "signature_title_concat",
    "title_concat_top",
    "signature_description_concat",
    "description_concat_top",
]
EXPECTED_GROUPS = {
    "category": ["category_concat_top"],
    "title": ["signature_title_concat", "title_concat_top"],
    "description": ["signature_description_concat", "description_concat_top"],
}
EXPECTED_AGGREGATES = [
    "all_chunk_mean_cosine",
    "field_equal_mean_cosine",
    "symmetric_top3_block_cosine",
    "category_mean_cosine",
    "title_mean_cosine",
    "description_mean_cosine",
]
EXPECTED_CHUNK_ALGORITHM = (
    "greedy_longest_nonempty_prefix_with_preferred_whitespace_or_punctuation_"
    "boundary_and_character_fallback"
)
EXPECTED_AGGREGATION_DEFINITIONS = {
    "all_chunk_mean_cosine": (
        "cosine_between_l2_normalized_means_of_all_unit_chunk_vectors"
    ),
    "field_equal_mean_cosine": (
        "arithmetic_mean_of_per_field_mean_vector_cosines_over_fields_present_"
        "on_both_sellers_error_if_none"
    ),
    "symmetric_top3_block_cosine": (
        "for_each_left_chunk_mean_its_top_three_right_chunk_cosines_then_average_"
        "left_chunks_repeat_right_to_left_and_average_both_directions"
    ),
    "category_mean_cosine": (
        "cosine_between_l2_normalized_means_of_category_group_chunks_use_field_"
        "equal_score_if_either_group_is_absent"
    ),
    "title_mean_cosine": (
        "cosine_between_l2_normalized_means_of_title_group_chunks_use_field_"
        "equal_score_if_either_group_is_absent"
    ),
    "description_mean_cosine": (
        "cosine_between_l2_normalized_means_of_description_group_chunks_use_"
        "field_equal_score_if_either_group_is_absent"
    ),
}
EXPECTED_MODEL_PREFIXES = {
    "gte_multilingual_base": "chunk_gte_multilingual_base",
    "bge_m3": "chunk_bge_m3",
    "multilingual_e5_large": "chunk_multilingual_e5_large",
    "labse": "chunk_labse",
    "paraphrase_multilingual_mpnet_base_v2": (
        "chunk_paraphrase_multilingual_mpnet"
    ),
}
EXPECTED_CHUNK_PREFLIGHT = {
    "reference_transformers_version": "4.46.3",
    "field_corpus_sha256": (
        "7e9d0948f9936d0f4b70987412345096d4f837fc7962e7e4fba1d501e6749655"
    ),
    "shared_chunk_rows_canonical_sha256": (
        "da2083494163e966f66180d201cb512f654d3e98f34f4d369f3046ac94aeca53"
    ),
    "seller_count": 855,
    "chunk_count": 5378,
    "nonempty_field_count": 4198,
    "chunk_count_min": 1,
    "chunk_count_median": 5.0,
    "chunk_count_p90": 9.0,
    "chunk_count_p95": 10.0,
    "chunk_count_max": 19,
    "missing_group_seller_counts_audit_only": {
        "category": 0,
        "title": 1,
        "description": 2,
    },
    "token_length_maximum_by_model": {
        "gte_multilingual_base": 477,
        "bge_m3": 478,
        "multilingual_e5_large": 480,
        "labse": 480,
        "paraphrase_multilingual_mpnet_base_v2": 477,
    },
    "over_budget_count_by_model": {
        "gte_multilingual_base": 0,
        "bge_m3": 0,
        "multilingual_e5_large": 0,
        "labse": 0,
        "paraphrase_multilingual_mpnet_base_v2": 0,
    },
}


resolve = source.resolve
load_json = source.load_json
load_csv = source.load_csv
load_jsonl = source.load_jsonl
sha256_file = source.sha256_file
sha256_text = source.sha256_text
canonical_hash = source.canonical_hash
write_json_immutable = source.write_json_immutable
write_json_atomic = source.write_json_atomic
write_jsonl_immutable = source.write_jsonl_immutable
write_csv_immutable = source.write_csv_immutable
write_npy_immutable = source.write_npy_immutable
write_bytes_immutable = source.write_bytes_immutable


def relative(path: Path) -> str:
    return str(path.resolve().relative_to(ROOT.resolve())).replace("\\", "/")


def verify_file_record(record: dict, role: str) -> Path:
    expected_keys = {"path", "size_bytes", "sha256"}
    if set(record) != expected_keys:
        raise ValueError(f"Step7-v3.1 {role} record schema drift")
    path = resolve(record["path"])
    if not path.is_file():
        raise FileNotFoundError(f"Step7-v3.1 {role} is missing: {path}")
    if path.stat().st_size != int(record["size_bytes"]):
        raise ValueError(f"Step7-v3.1 {role} size drift: {record['path']}")
    if sha256_file(path) != record["sha256"]:
        raise ValueError(f"Step7-v3.1 {role} hash drift: {record['path']}")
    return path


def source_policy(policy: dict) -> dict:
    path = verify_file_record(policy["source_data_policy"], "source-data policy")
    source_policy_payload = load_json(path)
    source.validate_policy(source_policy_payload)
    return source_policy_payload


def _verify_frozen_source_artifact(
    source_policy_payload: dict, role: str, path: Path
) -> None:
    expected = source_policy_payload["expected_artifacts"][role]
    if (
        path.stat().st_size != int(expected["size_bytes"])
        or sha256_file(path) != expected["sha256"]
    ):
        raise ValueError(f"Step7-v3.1 frozen source artifact drift: {role}")


def _verify_source_implementation_files(
    source_policy_payload: dict, roles: tuple[str, ...]
) -> dict[str, Path]:
    return {
        role: verify_file_record(
            source_policy_payload["implementation"][role],
            f"source implementation {role}",
        )
        for role in roles
    }


def _validate_source_input_manifest(
    source_policy_payload: dict, manifest: dict, input_names: tuple[str, ...]
) -> None:
    observed = manifest.get("input_manifest", {})
    if set(observed) != set(input_names):
        raise ValueError("Step7-v3.1 source input-manifest universe drift")
    for name in input_names:
        expected = source_policy_payload["inputs"][name]
        record = observed[name]
        if (
            record.get("path") != expected["path"]
            or record.get("sha256") != expected["sha256"]
            or int(record.get("size_bytes", -1)) <= 0
        ):
            raise ValueError(f"Step7-v3.1 source input-manifest drift: {name}")


def _validate_public_preparation_manifest_core(
    policy: dict, source_policy_payload: dict, manifest: dict
) -> None:
    implementation = source_policy_payload["implementation"]
    source_policy_path = resolve(policy["source_data_policy"]["path"])
    if (
        manifest.get("step")
        != "step7_v3_1_prepare_standalone_label_free_source_data"
        or manifest.get("version") != source_policy_payload["version"]
        or manifest.get("policy_path") != relative(source_policy_path)
        or manifest.get("policy_sha256") != sha256_file(source_policy_path)
        or manifest.get("generator_script_path")
        != implementation["preparation_script"]["path"]
        or manifest.get("generator_script_sha256")
        != implementation["preparation_script"]["sha256"]
        or manifest.get("common_script_sha256")
        != implementation["source_data_module"]["sha256"]
        or manifest.get("redaction_dependency_script_path")
        != implementation["redaction_dependency"]["path"]
        or manifest.get("redaction_dependency_script_sha256")
        != implementation["redaction_dependency"]["sha256"]
        or manifest.get("feature_generation_uses_review_label_values") is not False
        or manifest.get("feature_generation_uses_evidence_type_values") is not False
        or manifest.get("complete_field_text_replay") is not True
    ):
        raise ValueError("Step7-v3.1 source preparation provenance drift")
    _validate_source_input_manifest(
        source_policy_payload,
        manifest,
        ("seller_profiles", "item_identity_signals", "component_assignments"),
    )


def validate_source_public_artifacts(policy: dict) -> dict[str, Path]:
    source_payload = source_policy(policy)
    _verify_source_implementation_files(
        source_payload,
        ("source_data_module", "preparation_script", "redaction_dependency"),
    )
    outputs = policy["outputs"]
    roles = {
        "preparation_manifest",
        "pair_manifest",
        "field_corpus",
        "train_feature_reference",
        "safe_pair_features",
    }
    paths = {role: resolve(outputs[role]) for role in roles}
    if any(not path.is_file() for path in paths.values()):
        missing = sorted(role for role, path in paths.items() if not path.is_file())
        raise FileNotFoundError(f"Step7-v3.1 source artifacts are missing: {missing}")
    for role in roles - {"preparation_manifest"}:
        expected_path = source_payload["outputs"][role]
        if relative(paths[role]) != expected_path:
            raise ValueError(f"Step7-v3.1 source artifact path drift: {role}")
        _verify_frozen_source_artifact(source_payload, role, paths[role])
    manifest = load_json(paths["preparation_manifest"])
    _validate_public_preparation_manifest_core(policy, source_payload, manifest)
    for role in roles - {"preparation_manifest"}:
        recorded = manifest.get("output_files", {}).get(role)
        if (
            recorded is None
            or recorded.get("path") != relative(paths[role])
            or recorded.get("sha256") != sha256_file(paths[role])
            or int(recorded.get("size_bytes", -1)) != paths[role].stat().st_size
        ):
            raise ValueError(f"Step7-v3.1 source preparation output drift: {role}")
    return paths


def validate_source_encoding_artifacts(policy: dict) -> dict[str, Path]:
    """Validate the label-free subset that is permitted in the GPU workspace."""
    source_payload = source_policy(policy)
    _verify_source_implementation_files(
        source_payload, ("source_data_module", "redaction_dependency")
    )
    outputs = policy["outputs"]
    roles = {"preparation_manifest", "pair_manifest", "field_corpus"}
    paths = {role: resolve(outputs[role]) for role in roles}
    if any(not path.is_file() for path in paths.values()):
        missing = sorted(role for role, path in paths.items() if not path.is_file())
        raise FileNotFoundError(f"Step7-v3.1 encoding inputs are missing: {missing}")
    for role in ("pair_manifest", "field_corpus"):
        if relative(paths[role]) != source_payload["outputs"][role]:
            raise ValueError(f"Step7-v3.1 encoding input path drift: {role}")
        _verify_frozen_source_artifact(source_payload, role, paths[role])
    manifest = load_json(paths["preparation_manifest"])
    _validate_public_preparation_manifest_core(policy, source_payload, manifest)
    for role in ("pair_manifest", "field_corpus"):
        record = manifest.get("output_files", {}).get(role, {})
        if (
            record.get("path") != relative(paths[role])
            or record.get("sha256") != sha256_file(paths[role])
            or int(record.get("size_bytes", -1)) != paths[role].stat().st_size
        ):
            raise ValueError(f"Step7-v3.1 encoding source record drift: {role}")
    return paths


def validate_source_development_artifacts(policy: dict) -> dict[str, Path]:
    source_payload = source_policy(policy)
    _verify_source_implementation_files(
        source_payload,
        ("source_data_module", "preparation_script", "redaction_dependency"),
    )
    outputs = policy["outputs"]
    roles = {"development_labels_manifest", "train_labels", "valid_labels"}
    paths = {role: resolve(outputs[role]) for role in roles}
    if any(not path.is_file() for path in paths.values()):
        missing = sorted(role for role, path in paths.items() if not path.is_file())
        raise FileNotFoundError(f"Step7-v3.1 development artifacts are missing: {missing}")
    for role in ("train_labels", "valid_labels"):
        if relative(paths[role]) != source_payload["outputs"][role]:
            raise ValueError(f"Step7-v3.1 development artifact path drift: {role}")
        _verify_frozen_source_artifact(source_payload, role, paths[role])
    manifest = load_json(paths["development_labels_manifest"])
    implementation = source_payload["implementation"]
    preparation_manifest_path = resolve(outputs["preparation_manifest"])
    if (
        manifest.get("step") != "step7_v3_1_prepare_development_labels"
        or manifest.get("version") != source_payload["version"]
        or manifest.get("policy_sha256")
        != sha256_file(resolve(policy["source_data_policy"]["path"]))
        or manifest.get("generator_script_path")
        != implementation["preparation_script"]["path"]
        or manifest.get("generator_script_sha256")
        != implementation["preparation_script"]["sha256"]
        or manifest.get("common_script_sha256")
        != implementation["source_data_module"]["sha256"]
        or manifest.get("redaction_dependency_script_path")
        != implementation["redaction_dependency"]["path"]
        or manifest.get("redaction_dependency_script_sha256")
        != implementation["redaction_dependency"]["sha256"]
        or manifest.get("public_preparation_manifest_sha256")
        != sha256_file(preparation_manifest_path)
        or manifest.get("splits_written") != ["train", "valid"]
        or manifest.get("other_split_label_values_used_during_materialization") is not False
        or manifest.get("split_projection_applied_before_label_or_evidence_access")
        is not True
    ):
        raise ValueError("Step7-v3.1 development-label boundary drift")
    _validate_source_input_manifest(
        source_payload, manifest, ("frozen_labels", "evidence_labels")
    )
    expected_output_keys = {"private_labels_train", "private_labels_valid"}
    if set(manifest.get("output_files", {})) != expected_output_keys:
        raise ValueError("Step7-v3.1 development output-record universe drift")
    for split in ("train", "valid"):
        role = f"{split}_labels"
        record = manifest["output_files"][f"private_labels_{split}"]
        if (
            record.get("path") != relative(paths[role])
            or record.get("sha256") != sha256_file(paths[role])
            or int(record.get("size_bytes", -1)) != paths[role].stat().st_size
        ):
            raise ValueError(f"Step7-v3.1 development output record drift: {split}")
    return paths


def aggregate_feature_names(model_cfg: dict) -> list[str]:
    prefix = str(model_cfg["feature_prefix"])
    return [f"{prefix}__{name}" for name in EXPECTED_AGGREGATES]


def primary_feature_name(model_cfg: dict) -> str:
    return f"{model_cfg['feature_prefix']}__field_equal_mean_cosine"


def validate_policy(policy: dict) -> None:
    if policy.get("version") != EXPECTED_VERSION:
        raise ValueError("Step7-v3.1 policy version drift")
    source_payload = source_policy(policy)
    for key in (
        "expected_counts",
        "expected_component_count_by_split",
        "expected_seller_count_by_split",
    ):
        if policy["supervision_boundary"][key] != source_payload[
            "supervision_boundary"
        ][key]:
            raise ValueError(
                f"Step7-v3.1 supervision boundary drift from source contract: {key}"
            )
    if policy["clean_text_contract"]["fields_in_order"] != EXPECTED_FIELDS:
        raise ValueError("Step7-v3.1 clean field order drift")
    if policy["clean_text_contract"]["field_groups"] != EXPECTED_GROUPS:
        raise ValueError("Step7-v3.1 field grouping drift")
    chunk = policy["shared_chunking"]
    if (
        chunk["algorithm"] != EXPECTED_CHUNK_ALGORITHM
        or int(chunk["token_budget_including_model_prefix_and_special_tokens"]) != 480
        or chunk["all_five_tokenizers_must_fit_every_shared_chunk"] is not True
        or chunk["same_exact_chunk_text_and_order_for_every_encoder"] is not True
        or chunk["require_exact_character_reconstruction_per_field"] is not True
        or chunk["require_complete_nonoverlapping_field_coverage"] is not True
        or int(chunk["chunk_overlap_characters"]) != 0
        or chunk["maximum_chunks_per_seller"] is not None
        or chunk["long_history_sampling_or_dropping_allowed"] is not False
        or chunk["labels_or_evidence_types_used"] is not False
        or chunk["chunk_count_length_or_missingness_allowed_as_model_features"] is not False
        or chunk.get("expected_label_free_tokenizer_preflight")
        != EXPECTED_CHUNK_PREFLIGHT
    ):
        raise ValueError("Step7-v3.1 complete shared-chunk contract drift")
    aggregation = policy["aggregation"]
    if (
        aggregation["aggregate_order"] != EXPECTED_AGGREGATES
        or aggregation["primary_raw_encoder_aggregate"] != "field_equal_mean_cosine"
        or int(aggregation["top_k_matches_per_chunk"]) != 3
        or int(aggregation["serialized_score_decimal_places"]) != 12
        or not math.isclose(float(aggregation["score_replay_absolute_tolerance"]), 2e-6)
    ):
        raise ValueError("Step7-v3.1 aggregate contract drift")
    for name, expected in EXPECTED_AGGREGATION_DEFINITIONS.items():
        if aggregation.get(name) != expected:
            raise ValueError(f"Step7-v3.1 aggregate definition drift: {name}")
    if list(policy["embedding_models"]) != list(EXPECTED_MODEL_PREFIXES):
        raise ValueError("Step7-v3.1 encoder universe or order drift")
    prefixes = []
    all_feature_names = []
    for model_key, cfg in policy["embedding_models"].items():
        if int(cfg["max_length"]) < 480:
            raise ValueError(f"Step7-v3.1 model cannot support shared chunks: {model_key}")
        if cfg.get("feature_prefix") != EXPECTED_MODEL_PREFIXES[model_key]:
            raise ValueError(f"Step7-v3.1 aggregate prefix drift: {model_key}")
        source.validate_expected_model_pin(model_key, cfg)
        source.validate_sentence_transformer_layout(model_key, cfg)
        prefixes.append(cfg["feature_prefix"])
        all_feature_names.extend(aggregate_feature_names(cfg))
    if len(prefixes) != len(set(prefixes)) or len(all_feature_names) != len(set(all_feature_names)):
        raise ValueError("Step7-v3.1 aggregate feature names are not unique")
    if policy["pair_feature_roles"] != source_payload["pair_feature_roles"]:
        raise ValueError("Step7-v3.1 safe/shortcut pair feature roles drift")
    if policy["candidate_tiers"] != {
        "encoder_aggregates_only": ["{embedding_aggregate_features}"],
        "encoder_aggregates_plus_transfer": [
            "{embedding_aggregate_features}",
            "{model_eligible_transfer_features}",
        ],
    }:
        raise ValueError("Step7-v3.1 candidate tiers drift")
    if policy["no_encoder_controls"] != {
        "intercept_only": [],
        "transfer_features_only": ["{model_eligible_transfer_features}"],
    }:
        raise ValueError("Step7-v3.1 no-encoder controls drift")
    matched = policy["selection_rule"]["pipeline_attribution"][
        "matched_no_encoder_control_by_tier"
    ]
    if matched != {
        "encoder_aggregates_only": "control__intercept_only",
        "encoder_aggregates_plus_transfer": "control__transfer_features_only",
    }:
        raise ValueError("Step7-v3.1 matched attribution control drift")
    training = policy["training"]
    if (
        training["fit_split"] != "train"
        or training["selection_split"] != "valid"
        or training["model_family"] != "standardized_logistic_regression_l2"
        or training["solver"] != "newton_with_armijo_backtracking"
        or training["l2_grid"] != [0.1, 1.0, 10.0, 100.0]
        or int(training["fold_count"]) != 5
        or int(training["fold_seed"]) != 20260721
        or training["primary_sample_weight"]
        != "component_equal_normalized_to_row_count"
        or training["sensitivity_sample_weight"] != "uniform"
        or training["threshold_reference"] != "train_oof_predictions_only"
    ):
        raise ValueError("Step7-v3.1 training contract drift")
    outputs = policy["outputs"]
    root = outputs["root"].rstrip("/")
    if not root.startswith("reports/step7_v3_1_full_text_chunked_selection/"):
        raise ValueError("Step7-v3.1 output root drift")
    for name, value in outputs.items():
        if name != "root" and not str(value).startswith(root + "/"):
            raise ValueError(f"Step7-v3.1 output escapes versioned root: {name}")
    if outputs["root"] != source_payload["outputs"]["root"]:
        raise ValueError("Step7-v3.1 source/model output roots diverged")
    for role in source_payload["outputs"]:
        if role != "root" and outputs.get(role) != source_payload["outputs"][role]:
            raise ValueError(f"Step7-v3.1 source output path drift: {role}")


def validate_field_corpus_rows(policy: dict, rows: list[dict]) -> None:
    source.validate_field_corpus_rows(source_policy(policy), rows)
    # The checks below also bind the corpus to the model-side field contract.
    expected_sellers = 855
    if len(rows) != expected_sellers:
        raise ValueError(f"Step7-v3.1 field corpus row count drift: {len(rows)}")
    seller_uids = [str(row.get("seller_uid", "")) for row in rows]
    if seller_uids != sorted(seller_uids) or len(seller_uids) != len(set(seller_uids)):
        raise ValueError("Step7-v3.1 field corpus seller order/uniqueness drift")
    fields = policy["clean_text_contract"]["fields_in_order"]
    fallback = policy["clean_text_contract"]["empty_text_fallback"]
    for row in rows:
        if set(row) != {
            "seller_uid",
            "split_name",
            "field_texts",
            "field_text_sha256",
            "model_text",
            "model_text_sha256",
        }:
            raise ValueError("Step7-v3.1 field corpus schema drift")
        if row["split_name"] not in {"train", "valid", "test"}:
            raise ValueError("Step7-v3.1 field corpus split drift")
        if list(row["field_texts"]) != fields or list(row["field_text_sha256"]) != fields:
            raise ValueError("Step7-v3.1 field corpus field order drift")
        for field in fields:
            value = row["field_texts"][field]
            if not isinstance(value, str) or row["field_text_sha256"][field] != sha256_text(value):
                raise ValueError("Step7-v3.1 field text hash drift")
        reconstructed = "\n".join(
            row["field_texts"][field] for field in fields if row["field_texts"][field]
        ).strip()
        if not reconstructed:
            reconstructed = fallback
        if reconstructed != row["model_text"] or row["model_text_sha256"] != sha256_text(
            row["model_text"]
        ):
            raise ValueError("Step7-v3.1 field corpus cannot replay parent model_text")


def validate_shared_chunk_rows(
    policy: dict, field_rows: list[dict], chunk_rows: list[dict]
) -> dict:
    fields = policy["clean_text_contract"]["fields_in_order"]
    groups = policy["clean_text_contract"]["field_groups"]
    group_by_field = {
        field: group for group, group_fields in groups.items() for field in group_fields
    }
    expected_sellers = {row["seller_uid"]: row for row in field_rows}
    seen_uids = set()
    chunks_by_seller_field: dict[tuple[str, str], list[dict]] = {}
    previous_order = None
    token_budget = int(
        policy["shared_chunking"][
            "token_budget_including_model_prefix_and_special_tokens"
        ]
    )
    model_keys = list(policy["embedding_models"])
    for row in chunk_rows:
        if set(row) != {
            "chunk_uid",
            "seller_uid",
            "split_name",
            "field_name",
            "field_group",
            "chunk_index",
            "char_start",
            "char_end",
            "text",
            "text_sha256",
            "token_lengths",
        }:
            raise ValueError("Step7-v3.1 shared chunk schema drift")
        chunk_uid = str(row["chunk_uid"])
        if not chunk_uid or chunk_uid in seen_uids:
            raise ValueError("Step7-v3.1 shared chunk UID is empty or duplicated")
        seen_uids.add(chunk_uid)
        seller_uid = str(row["seller_uid"])
        field = str(row["field_name"])
        if seller_uid not in expected_sellers or field not in fields:
            raise ValueError("Step7-v3.1 shared chunk seller/field universe drift")
        if row["split_name"] != expected_sellers[seller_uid]["split_name"]:
            raise ValueError("Step7-v3.1 shared chunk split drift")
        if row["field_group"] != group_by_field[field]:
            raise ValueError("Step7-v3.1 shared chunk field-group drift")
        text = row["text"]
        if not isinstance(text, str) or not text or not text.strip():
            raise ValueError("Step7-v3.1 shared chunk text is empty")
        if row["text_sha256"] != sha256_text(text):
            raise ValueError("Step7-v3.1 shared chunk text hash drift")
        expected_chunk_uid = canonical_hash(
            {
                "seller_uid": seller_uid,
                "field_name": field,
                "chunk_index": int(row["chunk_index"]),
                "char_start": int(row["char_start"]),
                "char_end": int(row["char_end"]),
                "text_sha256": row["text_sha256"],
            }
        )
        if chunk_uid != expected_chunk_uid:
            raise ValueError("Step7-v3.1 shared chunk UID/content drift")
        if list(row["token_lengths"]) != model_keys:
            raise ValueError("Step7-v3.1 shared chunk tokenizer order drift")
        if any(
            not isinstance(row["token_lengths"][key], int)
            or not 0 < row["token_lengths"][key] <= token_budget
            for key in model_keys
        ):
            raise ValueError("Step7-v3.1 shared chunk exceeds common token budget")
        order = (seller_uid, fields.index(field), int(row["chunk_index"]))
        if previous_order is not None and order <= previous_order:
            raise ValueError("Step7-v3.1 shared chunk global order drift")
        previous_order = order
        chunks_by_seller_field.setdefault((seller_uid, field), []).append(row)

    chunks_per_seller = []
    missing_group_seller_counts = {
        group: 0 for group in policy["clean_text_contract"]["field_groups"]
    }
    nonempty_fields = 0
    for seller_uid, field_row in expected_sellers.items():
        seller_chunk_count = 0
        for field in fields:
            expected_text = field_row["field_texts"][field]
            chunks = chunks_by_seller_field.get((seller_uid, field), [])
            if not expected_text:
                if chunks:
                    raise ValueError("Step7-v3.1 empty field unexpectedly has chunks")
                continue
            nonempty_fields += 1
            if not chunks:
                raise ValueError("Step7-v3.1 nonempty field has no chunk")
            position = 0
            pieces = []
            for expected_index, chunk in enumerate(chunks):
                if (
                    int(chunk["chunk_index"]) != expected_index
                    or int(chunk["char_start"]) != position
                    or int(chunk["char_end"]) <= position
                ):
                    raise ValueError("Step7-v3.1 chunk coverage is overlapping or discontinuous")
                pieces.append(chunk["text"])
                position = int(chunk["char_end"])
            if position != len(expected_text) or "".join(pieces) != expected_text:
                raise ValueError("Step7-v3.1 chunk character reconstruction failed")
            seller_chunk_count += len(chunks)
        if seller_chunk_count <= 0:
            raise ValueError("Step7-v3.1 seller has no shared chunks")
        for group, group_fields in policy["clean_text_contract"]["field_groups"].items():
            if not any(field_row["field_texts"][field] for field in group_fields):
                missing_group_seller_counts[group] += 1
        chunks_per_seller.append(seller_chunk_count)
    if not chunk_rows:
        raise ValueError("Step7-v3.1 shared chunk corpus is empty")
    values = np.asarray(chunks_per_seller, dtype=np.int64)
    return {
        "status": "pass",
        "seller_count": len(expected_sellers),
        "chunk_count": len(chunk_rows),
        "nonempty_field_count": nonempty_fields,
        "exact_character_reconstruction": True,
        "chunk_count_min": int(np.min(values)),
        "chunk_count_median": float(np.median(values)),
        "chunk_count_p90": float(np.quantile(values, 0.90)),
        "chunk_count_p95": float(np.quantile(values, 0.95)),
        "chunk_count_max": int(np.max(values)),
        "missing_group_seller_counts_audit_only": missing_group_seller_counts,
    }


def chunk_layout(policy: dict, chunk_rows: list[dict]) -> dict[str, dict]:
    fields = policy["clean_text_contract"]["fields_in_order"]
    groups = policy["clean_text_contract"]["field_groups"]
    layout: dict[str, dict] = {}
    for matrix_index, row in enumerate(chunk_rows):
        seller = layout.setdefault(
            row["seller_uid"],
            {
                "all": [],
                "fields": {field: [] for field in fields},
                "groups": {group: [] for group in groups},
            },
        )
        seller["all"].append(matrix_index)
        seller["fields"][row["field_name"]].append(matrix_index)
        seller["groups"][row["field_group"]].append(matrix_index)
    return layout


def _unit_mean(matrix: np.ndarray, indices: list[int]) -> np.ndarray | None:
    if not indices:
        return None
    value = np.mean(np.asarray(matrix[indices], dtype=np.float64), axis=0)
    norm = float(np.linalg.norm(value))
    if not math.isfinite(norm) or norm <= 1e-12:
        raise ValueError("Step7-v3.1 chunk mean vector has zero/non-finite norm")
    return value / norm


def seller_embedding_summaries(
    policy: dict, matrix: np.ndarray, chunk_rows: list[dict]
) -> dict[str, dict]:
    if matrix.dtype != np.float32 or matrix.ndim != 2 or matrix.shape[0] != len(chunk_rows):
        raise ValueError("Step7-v3.1 chunk embedding matrix dtype/shape drift")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("Step7-v3.1 chunk embedding matrix is non-finite")
    norms = np.linalg.norm(matrix, axis=1)
    if float(np.max(np.abs(norms - 1.0))) > 1e-3:
        raise ValueError("Step7-v3.1 chunk embedding matrix is not unit normalized")
    fields = policy["clean_text_contract"]["fields_in_order"]
    groups = policy["clean_text_contract"]["field_groups"]
    summaries = {}
    for seller_uid, indices in chunk_layout(policy, chunk_rows).items():
        summaries[seller_uid] = {
            "all_vector": _unit_mean(matrix, indices["all"]),
            "field_vectors": {
                field: _unit_mean(matrix, indices["fields"][field]) for field in fields
            },
            "group_vectors": {
                group: _unit_mean(matrix, indices["groups"][group]) for group in groups
            },
            "chunk_matrix": np.asarray(matrix[indices["all"]], dtype=np.float64),
        }
    return summaries


def aggregate_pair(
    policy: dict, left: dict, right: dict
) -> dict[str, float]:
    def cosine(first: np.ndarray, second: np.ndarray) -> float:
        value = float(np.dot(first, second))
        if not math.isfinite(value):
            raise ValueError("Step7-v3.1 aggregate cosine is non-finite")
        return max(-1.0, min(1.0, value))

    all_score = cosine(left["all_vector"], right["all_vector"])
    field_scores = []
    for field in policy["clean_text_contract"]["fields_in_order"]:
        first = left["field_vectors"][field]
        second = right["field_vectors"][field]
        if first is not None and second is not None:
            field_scores.append(cosine(first, second))
    if not field_scores:
        raise ValueError("Step7-v3.1 seller pair has no jointly present clean field")
    cross = left["chunk_matrix"] @ right["chunk_matrix"].T
    if cross.size <= 0 or not np.all(np.isfinite(cross)):
        raise ValueError("Step7-v3.1 cross-chunk cosine matrix is invalid")
    requested_top_k = int(policy["aggregation"]["top_k_matches_per_chunk"])

    def directional_top_k(values: np.ndarray) -> float:
        top_k = min(requested_top_k, values.shape[1])
        selected = np.partition(values, values.shape[1] - top_k, axis=1)[
            :, -top_k:
        ]
        return float(np.mean(np.mean(selected, axis=1)))

    symmetric_top_k = 0.5 * (
        directional_top_k(cross) + directional_top_k(cross.T)
    )

    field_equal_score = float(np.mean(field_scores))

    def group_score(group: str) -> float:
        first = left["group_vectors"][group]
        second = right["group_vectors"][group]
        # A numeric sentinel (for example 0 or -1) would make field missingness
        # an implicit selectable shortcut.  Reuse the already available
        # field-equal evidence instead; missingness remains audit-only.
        return (
            field_equal_score
            if first is None or second is None
            else cosine(first, second)
        )

    output = {
        "all_chunk_mean_cosine": all_score,
        "field_equal_mean_cosine": field_equal_score,
        "symmetric_top3_block_cosine": symmetric_top_k,
        "category_mean_cosine": group_score("category"),
        "title_mean_cosine": group_score("title"),
        "description_mean_cosine": group_score("description"),
    }
    if list(output) != EXPECTED_AGGREGATES or not all(
        math.isfinite(value) and -1.000001 <= value <= 1.000001
        for value in output.values()
    ):
        raise ValueError("Step7-v3.1 aggregate feature output drift")
    return output


def compute_pair_score_rows(
    policy: dict,
    model_cfg: dict,
    matrix: np.ndarray,
    chunk_rows: list[dict],
    pair_rows: list[dict],
) -> list[dict]:
    summaries = seller_embedding_summaries(policy, matrix, chunk_rows)
    names = aggregate_feature_names(model_cfg)
    decimals = int(policy["aggregation"]["serialized_score_decimal_places"])
    output = []
    for pair in pair_rows:
        try:
            values = aggregate_pair(
                policy,
                summaries[pair["seller_uid_left"]],
                summaries[pair["seller_uid_right"]],
            )
        except KeyError as exc:
            raise ValueError("Step7-v3.1 pair endpoint lacks chunk embeddings") from exc
        row = {"pair_uid": pair["pair_uid"]}
        row.update(
            {
                name: f"{values[aggregate]:.{decimals}f}"
                for aggregate, name in zip(EXPECTED_AGGREGATES, names, strict=True)
            }
        )
        output.append(row)
    return output
