#!/usr/bin/env python3
"""Select a provisional English M0 with train-only nested component CV."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np

import step7_v3_1_selection_core as solver
import step7_v3_1_source_data as source
import step7_v4_build_sync_manifest as sync_builder
import step7_v4_common as common
import step7_v4_encode_item_models as encoder
import step7_v4_prepare_source_data as preparation


SELECTOR_SCRIPT = Path(__file__).resolve()


def index_unique(rows: list[dict], key: str, role: str) -> dict[str, dict]:
    output = {str(row[key]): row for row in rows}
    if len(output) != len(rows):
        raise ValueError(f"Step7-v4 {role} contains duplicate {key}")
    return output


def verify_gpu_outputs(
    policy: dict, preparation_manifest: dict, preparation_bundle: dict
) -> tuple[dict, dict[str, dict]]:
    outputs = policy["outputs"]
    sync_path = common.resolve(outputs["gpu_sync_manifest"])
    bundle_path = common.resolve(outputs["gpu_output_manifest"])
    if not sync_path.is_file() or not bundle_path.is_file():
        raise FileNotFoundError("Step7-v4 compact GPU outputs have not been synchronized")
    sync_manifest = common.load_json(sync_path)
    expected_sync = sync_builder.build_payload(policy, common.DEFAULT_POLICY)
    if sync_manifest != expected_sync:
        raise ValueError("Step7-v4 GPU sync manifest is stale")
    bundle = common.load_json(bundle_path)
    common.verify_canonical_self_hash(
        bundle, "bundle_content_sha256", "GPU output bundle"
    )
    expected_paths = set(sync_manifest["expected_gpu_outputs_to_sync_back"])
    payload_paths = expected_paths - {outputs["gpu_output_manifest"]}
    records = bundle.get("files", [])
    if (
        bundle.get("policy_sha256") != common.sha256_file(common.DEFAULT_POLICY)
        or bundle.get("generator_script_sha256")
        != common.sha256_file(
            common.resolve(policy["implementation"]["encoder"]["path"])
        )
        or bundle.get("step") != "step7_v4_label_free_gpu_output_bundle"
        or bundle.get("version") != policy["version"]
        or bundle.get("label_or_raw_source_files_present_in_gpu_workspace") is not False
        or bundle.get("embedding_matrices_published") is not False
        or bundle.get("gpu_sync_manifest_sha256") != common.sha256_file(sync_path)
        or {record.get("path") for record in records} != payload_paths
        or len(records) != len(payload_paths)
        or int(bundle.get("file_count", -1)) != len(records)
        or int(bundle.get("total_file_bytes", -1))
        != sum(int(record["size_bytes"]) for record in records)
    ):
        raise ValueError("Step7-v4 GPU output bundle contract drift")
    for record in records:
        common.verify_file_record(record, "GPU output")
    unexpected_matrices = sorted(common.resolve(outputs["root"]).rglob("*.npy"))
    if unexpected_matrices:
        raise ValueError(
            "Step7-v4 compact result root unexpectedly contains an embedding matrix: "
            + str(unexpected_matrices[0])
        )

    chunk_rows = common.load_jsonl(common.resolve(outputs["shared_chunks"]))
    chunk_audit = encoder.validate_shared_chunks(
        policy, preparation_bundle["unique_text_rows"], chunk_rows
    )
    chunk_manifest = common.load_json(common.resolve(outputs["shared_chunks_manifest"]))
    common.verify_canonical_self_hash(
        chunk_manifest, "manifest_content_sha256", "shared chunk manifest"
    )
    if (
        chunk_manifest.get("step") != "step7_v4_build_complete_shared_item_chunks"
        or chunk_manifest.get("version") != policy["version"]
        or chunk_manifest.get("labels_or_evidence_types_read") is not False
        or chunk_manifest.get("raw_source_workbooks_present") is not False
        or chunk_manifest.get("chunking_contract") != policy["shared_chunking"]
        or chunk_manifest.get("chunk_audit") != chunk_audit
        or chunk_manifest.get("shared_chunks")
        != common.file_record(common.resolve(outputs["shared_chunks"]))
        or chunk_manifest.get("unique_text_corpus_sha256")
        != common.sha256_file(common.resolve(outputs["unique_text_corpus"]))
        or chunk_manifest.get("source_preparation_manifest_file_sha256")
        != common.sha256_file(common.resolve(outputs["preparation_manifest"]))
        or chunk_manifest.get("source_preparation_manifest_content_sha256")
        != preparation_manifest["manifest_content_sha256"]
        or chunk_manifest.get("gpu_sync_manifest_sha256")
        != common.sha256_file(sync_path)
        or chunk_manifest.get("policy_sha256") != common.sha256_file(common.DEFAULT_POLICY)
        or chunk_manifest.get("generator_script_sha256")
        != common.sha256_file(common.resolve(policy["implementation"]["encoder"]["path"]))
        or not isinstance(chunk_manifest.get("transformers_version"), str)
        or not chunk_manifest.get("transformers_version")
    ):
        raise ValueError("Step7-v4 shared chunk manifest drift")

    pair_rows = preparation_bundle["pair_rows"]
    pair_uids = [row["pair_uid"] for row in pair_rows]
    opaque_pair_uids = [
        row["pair_uid"] for row in preparation_bundle["gpu_pair_rows"]
    ]
    features: dict[str, dict[str, float | None]] = {pair_uid: {} for pair_uid in pair_uids}
    for rows, names, role in (
        (
            preparation_bundle["pair_stylometry_rows"],
            common.stylometry_feature_names(),
            "stylometry",
        ),
        (
            preparation_bundle["legacy_rows"],
            list(source.MODEL_ELIGIBLE_TRANSFER_FEATURE_NAMES),
            "legacy18",
        ),
    ):
        if [row["pair_uid"] for row in rows] != pair_uids:
            raise ValueError(f"Step7-v4 {role} pair order drift")
        for row in rows:
            for name in names:
                value = str(row[name]).strip()
                features[row["pair_uid"]][name] = None if value == "" else float(value)

    runtime = {
        "gpu_sync": sync_manifest,
        "gpu_bundle": bundle,
        "chunks": chunk_manifest,
        "models": {},
        "multiplicity_sensitivity": {},
    }
    expected_model_fingerprints = chunk_manifest.get("model_fingerprints", {})
    if set(expected_model_fingerprints) != set(common.MODEL_KEYS):
        raise ValueError("Step7-v4 chunk manifest model universe drift")
    expected_input_contracts = {
        model_key: {
            "text_prefix": cfg["text_prefix"],
            "sentence_transformer_prompt": cfg["sentence_transformer_prompt"],
            "native_max_seq_length": int(cfg["native_max_seq_length"]),
        }
        for model_key, cfg in policy["embedding_models"].items()
    }
    if chunk_manifest.get("model_input_contracts") != expected_input_contracts:
        raise ValueError("Step7-v4 chunk manifest model-input contract drift")
    runtime_environment_signature = None
    for model_key, cfg in policy["embedding_models"].items():
        score_path = common.resolve(
            outputs["pair_scores_template"].format(model_key=model_key)
        )
        runtime_path = common.resolve(
            outputs["model_runtime_manifest_template"].format(model_key=model_key)
        )
        score_rows = common.load_csv(score_path)
        names = common.encoder_feature_names(cfg)
        audit_names = common.frequency_audit_feature_names(cfg)
        expected_schema = ["pair_uid", *names, *audit_names]
        if (
            not score_rows
            or any(list(row) != expected_schema for row in score_rows)
            or [row["pair_uid"] for row in score_rows] != opaque_pair_uids
        ):
            raise ValueError(f"Step7-v4 pair-score schema/order drift: {model_key}")
        sensitivity_values = {name: [] for name in names}
        for full_pair_uid, row in zip(pair_uids, score_rows, strict=True):
            for name in (*names, *audit_names):
                value = str(row[name]).strip()
                if value:
                    number = float(value)
                    if not math.isfinite(number) or number < -1.000001 or number > 1.000001:
                        raise ValueError(f"Step7-v4 invalid aggregate: {model_key}/{name}")
                if name in names:
                    features[full_pair_uid][name] = None if not value else float(value)
            for name, audit_name in zip(names, audit_names, strict=True):
                primary_value = str(row[name]).strip()
                weighted_value = str(row[audit_name]).strip()
                if bool(primary_value) != bool(weighted_value):
                    raise ValueError(
                        f"Step7-v4 multiplicity audit missingness drift: {model_key}/{name}"
                    )
                if primary_value:
                    sensitivity_values[name].append(
                        abs(float(primary_value) - float(weighted_value))
                    )
        manifest = common.load_json(runtime_path)
        common.verify_canonical_self_hash(
            manifest, "runtime_content_sha256", f"{model_key} runtime manifest"
        )
        required = {
            "step": "step7_v4_encode_complete_item_shared_chunks",
            "version": policy["version"],
            "model_key": model_key,
            "role": cfg["role"],
            "repo_id": cfg["repo_id"],
            "revision": cfg["revision"],
            "local_path": cfg["local_path"],
            "model_fingerprint": expected_model_fingerprints[model_key],
            "encoder_parameters_updated": False,
            "feature_generation_reads_label_values": False,
            "label_or_raw_source_files_present_in_gpu_workspace": False,
            "same_exact_shared_chunks_for_all_models": True,
            "text_prefix": cfg["text_prefix"],
            "sentence_transformer_prompt": cfg["sentence_transformer_prompt"],
            "explicit_sentence_transformer_prompt_argument_used": True,
            "default_prompt_name_cleared_before_encoding": True,
            "native_max_seq_length": int(cfg["native_max_seq_length"]),
            "batch_size": int(cfg["batch_size"]),
            "sentence_transformers_version": policy["gpu_execution"][
                "required_sentence_transformers_version"
            ],
            "embedding_matrix_published": False,
            "embedding_matrix_ephemeral": True,
            "pair_count": len(pair_rows),
            "aggregate_feature_names": names,
            "multiplicity_audit_feature_names": audit_names,
            "pair_scores": common.file_record(score_path),
            "shared_chunks_sha256": common.sha256_file(
                common.resolve(outputs["shared_chunks"])
            ),
            "gpu_sync_manifest_sha256": common.sha256_file(sync_path),
            "source_preparation_manifest_file_sha256": common.sha256_file(
                common.resolve(outputs["preparation_manifest"])
            ),
            "source_preparation_manifest_content_sha256": preparation_manifest[
                "manifest_content_sha256"
            ],
            "sync_manifest_content_sha256": sync_manifest[
                "manifest_content_sha256"
            ],
            "policy_sha256": common.sha256_file(common.DEFAULT_POLICY),
            "policy_contract_sha256": common.canonical_hash(policy),
            "generator_script_sha256": common.sha256_file(
                common.resolve(policy["implementation"]["encoder"]["path"])
            ),
            "device": "cuda",
            "deterministic_gpu_runtime": policy["gpu_execution"][
                "expected_runtime"
            ],
        }
        for key, value in required.items():
            if manifest.get(key) != value:
                raise ValueError(f"Step7-v4 runtime manifest drift: {model_key}/{key}")
        if (
            manifest.get("loaded_default_prompt_name") is not None
            and not isinstance(manifest.get("loaded_default_prompt_name"), str)
        ) or not isinstance(manifest.get("loaded_prompts"), dict) or any(
            not isinstance(key, str) or not isinstance(value, str)
            for key, value in manifest.get("loaded_prompts", {}).items()
        ):
            raise ValueError(
                f"Step7-v4 loaded prompt provenance drift: {model_key}"
            )
        if (
            manifest.get("loaded_native_max_seq_length")
            != int(cfg["native_max_seq_length"])
            or manifest.get("shared_tokenizer_digest")
            != chunk_manifest["tokenizer_digests"][model_key]
            or manifest.get("runtime_sentence_transformer_tokenizer_digest")
            != manifest.get("shared_tokenizer_digest")
            or manifest.get("runtime_token_lengths_replay_shared_manifest") is not True
            or manifest.get("embedding_matrix_shape")
            != [len(chunk_rows), int(cfg["expected_dimension"])]
            or manifest.get("embedding_matrix_dtype") != "float32"
            or not isinstance(manifest.get("model_parameter_dtypes"), list)
            or not manifest.get("model_parameter_dtypes")
            or any(
                not isinstance(value, str) or not value
                for value in manifest.get("model_parameter_dtypes", [])
            )
            or not isinstance(manifest.get("torch_cuda_runtime_version"), str)
            or not manifest.get("torch_cuda_runtime_version")
            or not isinstance(manifest.get("cudnn_runtime_version"), int)
            or not isinstance(manifest.get("gpu_compute_capability"), list)
            or len(manifest.get("gpu_compute_capability", [])) != 2
            or any(
                not isinstance(value, int) or value < 0
                for value in manifest.get("gpu_compute_capability", [])
            )
            or not isinstance(manifest.get("embedding_matrix_content_sha256"), str)
            or len(manifest["embedding_matrix_content_sha256"]) != 64
            or float(manifest.get("maximum_unit_norm_error", math.inf)) > 1e-3
        ):
            raise ValueError(f"Step7-v4 runtime numerical provenance drift: {model_key}")
        string_environment_fields = (
            "gpu_name",
            "python_version",
            "numpy_version",
            "torch_version",
            "transformers_version",
            "sentence_transformers_version",
        )
        if any(
            not isinstance(manifest.get(field), str) or not manifest.get(field)
            for field in string_environment_fields
        ) or manifest.get("transformers_version") != chunk_manifest.get(
            "transformers_version"
        ):
            raise ValueError(
                f"Step7-v4 runtime library/device provenance drift: {model_key}"
            )
        observed_environment_signature = {
            field: manifest[field]
            for field in (
                *string_environment_fields,
                "torch_cuda_runtime_version",
                "cudnn_runtime_version",
                "gpu_compute_capability",
            )
        }
        if runtime_environment_signature is None:
            runtime_environment_signature = observed_environment_signature
        elif observed_environment_signature != runtime_environment_signature:
            raise ValueError(
                f"Step7-v4 runtime environment changed between models: {model_key}"
            )
        runtime["models"][model_key] = manifest
        runtime["multiplicity_sensitivity"][model_key] = {
            "status": "audit_only_forbidden_from_candidate_features",
            "features": {
                name: (
                    {
                        "status": "estimable",
                        "comparable_pair_count": len(values),
                        "nonzero_difference_count": int(
                            np.sum(np.asarray(values, dtype=np.float64) > 0.0)
                        ),
                        "mean_absolute_difference": float(np.mean(values)),
                        "p95_absolute_difference": float(
                            np.quantile(values, 0.95)
                        ),
                        "maximum_absolute_difference": float(np.max(values)),
                    }
                    if values
                    else {
                        "status": "not_estimable_all_pairs_missing",
                        "comparable_pair_count": 0,
                    }
                )
                for name, values in sensitivity_values.items()
            },
        }

    expected_names = {
        name
        for names in common.feature_blocks(policy).values()
        for name in names
    }
    for pair_uid, row in features.items():
        if set(row) != expected_names:
            raise ValueError(f"Step7-v4 selectable feature universe drift: {pair_uid}")
    return runtime, features


def load_label_split(policy: dict, pair_rows: list[dict], split: str) -> list[dict]:
    if split not in {"train", "valid"}:
        raise ValueError("Step7-v4 historical test labels remain sealed")
    outputs = policy["outputs"]
    manifest = common.load_json(common.resolve(outputs["development_labels_manifest"]))
    common.verify_canonical_self_hash(
        manifest, "manifest_content_sha256", "private development manifest"
    )
    role = f"{split}_labels"
    if (
        manifest.get("step") != "step7_v4_prepare_source_data_private_labels"
        or manifest.get("version") != policy["version"]
        or manifest.get("historical_test_labels_materialized") is not False
        or manifest.get("selection_label_columns")
        != ["pair_uid", "review_label", "component_id"]
        or manifest.get("evidence_is_physically_separate_from_selection_labels")
        is not True
        or manifest.get("identity_rule_control_score_materialized") is not False
        or manifest.get("policy_sha256") != common.sha256_file(common.DEFAULT_POLICY)
        or manifest.get("producer_sha256")
        != common.sha256_file(
            common.resolve(policy["implementation"]["preparation"]["path"])
        )
        or manifest.get("outputs", {}).get(role)
        != common.file_record(common.resolve(outputs[role]))
    ):
        raise ValueError(f"Step7-v4 private label manifest drift: {split}")
    labels = common.load_csv(common.resolve(outputs[role]))
    schema = ["pair_uid", "review_label", "component_id"]
    if not labels or any(list(row) != schema for row in labels):
        raise ValueError(f"Step7-v4 private label schema drift: {split}")
    pairs = [row for row in pair_rows if row["split_name"] == split]
    if [row["pair_uid"] for row in labels] != [row["pair_uid"] for row in pairs]:
        raise ValueError(f"Step7-v4 private label order drift: {split}")
    joined = []
    for pair, label in zip(pairs, labels, strict=True):
        if label["component_id"] != pair["component_id"]:
            raise ValueError(f"Step7-v4 label component drift: {split}")
        joined.append({**pair, **{key: label[key] for key in schema[1:]}})
    expected = policy["supervision_boundary"]["expected_counts"][split]
    counts = Counter(row["review_label"] for row in joined)
    if counts != Counter({"positive": expected["positive"], "negative": expected["negative"]}):
        raise ValueError(f"Step7-v4 private label counts drift: {split}")
    return joined


def load_evidence_split(
    policy: dict, labelled_rows: list[dict], split: str
) -> list[dict]:
    """Open diagnostic evidence only after selection and scoring are locked."""

    if split not in {"train", "valid"}:
        raise ValueError("Step7-v4 historical test evidence remains sealed")
    outputs = policy["outputs"]
    manifest = common.load_json(
        common.resolve(outputs["development_labels_manifest"])
    )
    common.verify_canonical_self_hash(
        manifest, "manifest_content_sha256", "private development manifest"
    )
    role = f"{split}_evidence"
    if (
        manifest.get("step") != "step7_v4_prepare_source_data_private_labels"
        or manifest.get("version") != policy["version"]
        or manifest.get("historical_test_labels_materialized") is not False
        or manifest.get("evidence_is_physically_separate_from_selection_labels")
        is not True
        or manifest.get("policy_sha256") != common.sha256_file(common.DEFAULT_POLICY)
        or manifest.get("producer_sha256")
        != common.sha256_file(
            common.resolve(policy["implementation"]["preparation"]["path"])
        )
        or manifest.get("outputs", {}).get(role)
        != common.file_record(common.resolve(outputs[role]))
    ):
        raise ValueError(f"Step7-v4 private evidence manifest drift: {split}")
    evidence = common.load_csv(common.resolve(outputs[role]))
    schema = ["pair_uid", "evidence_type"]
    if not evidence or any(list(row) != schema for row in evidence):
        raise ValueError(f"Step7-v4 private evidence schema drift: {split}")
    if [row["pair_uid"] for row in evidence] != [
        row["pair_uid"] for row in labelled_rows
    ]:
        raise ValueError(f"Step7-v4 private evidence order drift: {split}")
    if any(not str(row["evidence_type"]).strip() for row in evidence):
        raise ValueError(f"Step7-v4 private evidence contains an empty value: {split}")
    return [
        {**row, "evidence_type": evidence_row["evidence_type"]}
        for row, evidence_row in zip(labelled_rows, evidence, strict=True)
    ]


def replay_legacy_context(
    policy: dict,
    pair_rows: list[dict],
    frozen_features: dict[str, dict[str, float | None]],
) -> tuple[dict[str, dict], dict[str, str], dict]:
    parent_policy, public, replayed_pairs, _safe = preparation.replay_parent_public(policy)
    if replayed_pairs != pair_rows:
        raise ValueError("Step7-v4 legacy replay pair manifest drift")
    source.validate_input_hashes(
        parent_policy,
        ("seller_profiles", "item_identity_signals"),
    )
    profiles = public["profiles"]
    seller_uids = {
        row[endpoint]
        for row in pair_rows
        for endpoint in ("seller_uid_left", "seller_uid_right")
    }
    if not seller_uids.issubset(profiles):
        raise ValueError("Step7-v4 legacy replay lacks a pair-universe seller")
    seller_records = {
        seller_uid: public["seller_records"][seller_uid]
        for seller_uid in sorted(seller_uids)
    }
    seller_markets = {}
    for seller_uid in sorted(seller_uids):
        market = str(profiles[seller_uid].get("source_market_raw", "")).strip().casefold()
        if not market:
            raise ValueError("Step7-v4 market stress lacks a seller market")
        seller_markets[seller_uid] = market
    train_sellers = {
        row[endpoint]
        for row in pair_rows
        if row["split_name"] == "train"
        for endpoint in ("seller_uid_left", "seller_uid_right")
    }
    reference = source.train_reference(seller_records, train_sellers)
    replayed = source.build_safe_pair_rows(pair_rows, seller_records, reference)
    maximum_difference = 0.0
    for row in replayed:
        for name in source.MODEL_ELIGIBLE_TRANSFER_FEATURE_NAMES:
            frozen = frozen_features[row["pair_uid"]][name]
            if frozen is None:
                raise ValueError("Step7-v4 legacy18 unexpectedly contains a missing value")
            difference = abs(float(row[name]) - float(frozen))
            maximum_difference = max(maximum_difference, difference)
            if not math.isfinite(difference) or difference > 1e-15:
                raise ValueError(f"Step7-v4 full-train legacy feature replay drift: {name}")
    return seller_records, seller_markets, {
        "status": "pass",
        "seller_count": len(seller_records),
        "train_reference_seller_count": len(train_sellers),
        "full_train_reference_sha256": common.canonical_hash(reference),
        "maximum_absolute_replay_difference": maximum_difference,
        "labels_or_evidence_types_read": False,
    }


class FeatureFactory:
    def __init__(
        self,
        policy: dict,
        pair_rows: list[dict],
        fixed_features: dict[str, dict[str, float | None]],
        seller_records: dict[str, dict],
    ) -> None:
        self.policy = policy
        self.pair_by_uid = index_unique(pair_rows, "pair_uid", "pair manifest")
        self.fixed_features = fixed_features
        self.seller_records = seller_records
        self._view_cache: dict[tuple, tuple[dict[str, dict[str, float | None]], dict]] = {}

    def view(self, fit_rows: list[dict], target_rows: list[dict]) -> tuple[dict, dict]:
        fit_ids = tuple(sorted(row["pair_uid"] for row in fit_rows))
        target_ids = tuple(row["pair_uid"] for row in target_rows)
        key = (fit_ids, target_ids)
        if key in self._view_cache:
            return self._view_cache[key]
        fit_sellers = {
            row[endpoint]
            for row in fit_rows
            for endpoint in ("seller_uid_left", "seller_uid_right")
        }
        reference = source.train_reference(self.seller_records, fit_sellers)
        pair_inputs = [self.pair_by_uid[row["pair_uid"]] for row in target_rows]
        local_legacy = source.build_safe_pair_rows(
            pair_inputs, self.seller_records, reference
        )
        legacy_index = index_unique(local_legacy, "pair_uid", "fold-local legacy features")
        output = {}
        for row in target_rows:
            pair_uid = row["pair_uid"]
            values = dict(self.fixed_features[pair_uid])
            for name in source.MODEL_ELIGIBLE_TRANSFER_FEATURE_NAMES:
                values[name] = float(legacy_index[pair_uid][name])
            output[pair_uid] = values
        audit = {
            "fit_pair_count": len(fit_rows),
            "target_pair_count": len(target_rows),
            "fit_component_count": len({row["component_id"] for row in fit_rows}),
            "fit_seller_count": len(fit_sellers),
            "fit_seller_uid_sha256": common.canonical_hash(sorted(fit_sellers)),
            "feature_reference_sha256": common.canonical_hash(reference),
        }
        self._view_cache[key] = (output, audit)
        return output, audit

    @staticmethod
    def raw_matrix(rows: list[dict], names: list[str], view: dict) -> np.ndarray:
        matrix = np.empty((len(rows), len(names)), dtype=np.float64)
        for row_index, row in enumerate(rows):
            values = view[row["pair_uid"]]
            for column, name in enumerate(names):
                value = values[name]
                matrix[row_index, column] = np.nan if value is None else float(value)
        finite_or_missing = np.isfinite(matrix) | np.isnan(matrix)
        if not np.all(finite_or_missing):
            raise ValueError("Step7-v4 raw feature matrix contains infinity")
        return matrix

    def design(
        self, fit_rows: list[dict], hold_rows: list[dict], names: list[str]
    ) -> tuple[np.ndarray, np.ndarray, list[float], dict]:
        target_rows = [*fit_rows, *hold_rows]
        view, audit = self.view(fit_rows, target_rows)
        fit = self.raw_matrix(fit_rows, names, view)
        hold = self.raw_matrix(hold_rows, names, view)
        if not names:
            return fit, hold, [], audit
        medians = np.empty(len(names), dtype=np.float64)
        for column, name in enumerate(names):
            values = fit[:, column]
            observed = values[np.isfinite(values)]
            if len(observed) == 0:
                raise ValueError(f"Step7-v4 fold has an all-missing feature: {name}")
            medians[column] = float(np.median(observed))
        fit = np.where(np.isnan(fit), medians[None, :], fit)
        hold = np.where(np.isnan(hold), medians[None, :], hold)
        if not np.all(np.isfinite(fit)) or not np.all(np.isfinite(hold)):
            raise ValueError("Step7-v4 fold-local imputation did not produce finite values")
        return fit, hold, [float(value) for value in medians], audit


def labels_array(rows: list[dict]) -> np.ndarray:
    labels = np.asarray(
        [1 if row["review_label"] == "positive" else 0 for row in rows],
        dtype=np.int8,
    )
    if set(labels.tolist()) != {0, 1}:
        raise ValueError("Step7-v4 metric/training rows require both classes")
    return labels


def weighted_roc_auc(labels: np.ndarray, scores: np.ndarray, weights: np.ndarray) -> float:
    y = np.asarray(labels, dtype=np.int8)
    s = np.asarray(scores, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    positive_mass = float(np.sum(w[y == 1]))
    negative_mass = float(np.sum(w[y == 0]))
    if positive_mass <= 0.0 or negative_mass <= 0.0:
        raise ValueError("Step7-v4 weighted ROC AUC requires both classes")
    order = np.argsort(s, kind="mergesort")
    y, s, w = y[order], s[order], w[order]
    preceding_negative = 0.0
    concordance = 0.0
    start = 0
    while start < len(y):
        stop = start + 1
        while stop < len(y) and s[stop] == s[start]:
            stop += 1
        group_positive = float(np.sum(w[start:stop][y[start:stop] == 1]))
        group_negative = float(np.sum(w[start:stop][y[start:stop] == 0]))
        concordance += group_positive * (preceding_negative + 0.5 * group_negative)
        preceding_negative += group_negative
        start = stop
    return concordance / (positive_mass * negative_mass)


def trapezoidal_pr_auc(
    labels: np.ndarray, scores: np.ndarray, weights: np.ndarray | None = None
) -> float:
    y = np.asarray(labels, dtype=np.int8)
    s = np.asarray(scores, dtype=np.float64)
    w = np.ones(len(y), dtype=np.float64) if weights is None else np.asarray(weights, dtype=np.float64)
    positive_mass = float(np.sum(w[y == 1]))
    if positive_mass <= 0.0:
        raise ValueError("Step7-v4 PR AUC requires a positive label")
    order = np.argsort(-s, kind="mergesort")
    y, s, w = y[order], s[order], w[order]
    recalls = [0.0]
    precisions = [1.0]
    cumulative_positive = 0.0
    cumulative_total = 0.0
    start = 0
    while start < len(y):
        stop = start + 1
        while stop < len(y) and s[stop] == s[start]:
            stop += 1
        group = w[start:stop]
        cumulative_positive += float(np.sum(group[y[start:stop] == 1]))
        cumulative_total += float(np.sum(group))
        recalls.append(cumulative_positive / positive_mass)
        precisions.append(cumulative_positive / cumulative_total)
        start = stop
    precision_values = np.asarray(precisions)
    recall_values = np.asarray(recalls)
    if hasattr(np, "trapezoid"):
        return float(np.trapezoid(precision_values, recall_values))
    return float(np.trapz(precision_values, recall_values))


def strict_ranking_metrics(rows: list[dict], labels: np.ndarray, scores: np.ndarray) -> dict:
    queries: dict[str, list[tuple[str, int, float]]] = defaultdict(list)
    for row, label, score in zip(rows, labels, scores, strict=True):
        left, right = row["seller_uid_left"], row["seller_uid_right"]
        queries[left].append((right, int(label), float(score)))
        queries[right].append((left, int(label), float(score)))
    reciprocal, hits, recalls = [], [], {1: [], 3: [], 5: []}
    excluded_no_positive = excluded_no_negative = 0
    for _query, candidates in sorted(queries.items()):
        positives = sum(label == 1 for _candidate, label, _score in candidates)
        negatives = sum(label == 0 for _candidate, label, _score in candidates)
        if positives == 0:
            excluded_no_positive += 1
            continue
        if negatives == 0:
            excluded_no_negative += 1
            continue
        ranked = sorted(candidates, key=lambda item: (-item[2], item[0]))
        first = next(index for index, item in enumerate(ranked, start=1) if item[1] == 1)
        reciprocal.append(1.0 / first)
        hits.append(float(first == 1))
        for k in recalls:
            recalls[k].append(sum(item[1] == 1 for item in ranked[:k]) / positives)
    if not reciprocal:
        return {
            "status": "not_estimable_no_query_with_both_label_classes",
            "eligible_query_count": 0,
            "excluded_no_positive_count": excluded_no_positive,
            "excluded_no_negative_count": excluded_no_negative,
        }
    return {
        "status": "diagnostic_incomplete_labelled_candidate_graph",
        "eligible_query_count": len(reciprocal),
        "excluded_no_positive_count": excluded_no_positive,
        "excluded_no_negative_count": excluded_no_negative,
        "mrr": float(np.mean(reciprocal)),
        "hits_at_1": float(np.mean(hits)),
        "recall_at_1": float(np.mean(recalls[1])),
        "recall_at_3": float(np.mean(recalls[3])),
        "recall_at_5": float(np.mean(recalls[5])),
    }


def full_metrics(rows: list[dict], scores: np.ndarray, threshold: float) -> dict:
    labels = labels_array(rows)
    probabilities = np.asarray(scores, dtype=np.float64)
    if probabilities.shape != labels.shape or not np.all(np.isfinite(probabilities)):
        raise ValueError("Step7-v4 metric score shape/value drift")
    clipped = np.clip(probabilities, 1e-12, 1.0 - 1e-12)
    weights = solver.component_weights(rows, "component_equal_normalized_to_row_count")
    row_threshold = solver.threshold_metrics(labels, probabilities, threshold)
    weighted_threshold = solver.weighted_threshold_selection_metrics(
        labels, probabilities, threshold, weights
    )
    return {
        "row_count": len(rows),
        "positive_count": int(np.sum(labels == 1)),
        "negative_count": int(np.sum(labels == 0)),
        "row": {
            "roc_auc": solver.roc_auc(labels, probabilities),
            "average_precision": solver.average_precision(labels, probabilities),
            "trapezoidal_pr_auc": trapezoidal_pr_auc(labels, probabilities),
            "brier": float(np.mean((probabilities - labels) ** 2)),
            "logloss": float(
                -np.mean(labels * np.log(clipped) + (1 - labels) * np.log(1 - clipped))
            ),
            **row_threshold,
        },
        "component_equal": {
            "roc_auc": weighted_roc_auc(labels, probabilities, weights),
            "average_precision": solver.weighted_average_precision(
                labels, probabilities, weights
            ),
            "trapezoidal_pr_auc": trapezoidal_pr_auc(labels, probabilities, weights),
            "brier": float(np.average((probabilities - labels) ** 2, weights=weights)),
            "logloss": float(
                np.average(
                    -(labels * np.log(clipped) + (1 - labels) * np.log(1 - clipped)),
                    weights=weights,
                )
            ),
            **weighted_threshold,
        },
        "ranking": strict_ranking_metrics(rows, labels, probabilities),
    }


def derived_seed(*parts: object) -> int:
    digest = hashlib.sha256("|".join(str(part) for part in parts).encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big") & 0x7FFFFFFF


def balanced_component_folds(
    rows: list[dict], fold_count: int, seed: int
) -> dict[str, int]:
    try:
        return solver.balanced_component_folds(rows, int(fold_count), int(seed))
    except StopIteration as error:
        raise ValueError(
            "Step7-v4 cannot construct class-supported component folds from this subset"
        ) from error


def fit_logistic(
    matrix: np.ndarray, rows: list[dict], l2_penalty: float, policy: dict
) -> dict:
    cfg = policy["training"]
    sample_weights = solver.component_weights(rows, cfg["sample_weight"])
    weight_total = float(np.sum(sample_weights))
    if cfg.get("l2_parameterization") != (
        "weighted_mean_logloss_plus_half_l2_squared_coefficient_norm"
    ):
        raise ValueError("Step7-v4 L2 parameterization drift")
    # The inherited numerical solver minimizes a weighted *sum* of log losses.
    # Scaling its penalty by the total weight is exactly equivalent to minimizing
    # weighted mean log loss plus 0.5 * l2 * ||beta||^2.  This keeps one L2 value
    # comparable across inner, outer, full-train, and market-stress fit sizes.
    solver_sum_loss_l2_penalty = float(l2_penalty) * weight_total
    artifact = solver.fit_logistic(
        matrix,
        labels_array(rows),
        sample_weights,
        solver_sum_loss_l2_penalty,
        int(cfg["max_iter"]),
        float(cfg["tolerance"]),
        float(cfg["armijo_c1"]),
        float(cfg["minimum_line_search_step"]),
    )
    if artifact.get("solver_converged") is not True:
        raise ValueError("Step7-v4 accepted a non-converged logistic fit")
    if not math.isclose(
        float(artifact["sample_weight_total"]),
        weight_total,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError("Step7-v4 solver sample-weight total drift")
    artifact["solver_sum_loss_l2_penalty"] = float(
        artifact.pop("l2_penalty")
    )
    artifact["l2_penalty"] = float(l2_penalty)
    artifact["l2_parameterization"] = cfg["l2_parameterization"]
    return artifact


def tune_l2(
    policy: dict,
    factory: FeatureFactory,
    rows: list[dict],
    feature_names: list[str],
    *,
    fold_count: int,
    fold_seed: int,
) -> dict:
    cfg = policy["training"]
    assignments = balanced_component_folds(rows, int(fold_count), int(fold_seed))
    designs = []
    for fold in range(int(fold_count)):
        fit_rows = [row for row in rows if assignments[row["component_id"]] != fold]
        hold_rows = [row for row in rows if assignments[row["component_id"]] == fold]
        fit_matrix, hold_matrix, medians, audit = factory.design(
            fit_rows, hold_rows, feature_names
        )
        designs.append(
            {
                "fold": fold,
                "fit_rows": fit_rows,
                "hold_rows": hold_rows,
                "fit_matrix": fit_matrix,
                "hold_matrix": hold_matrix,
                "imputation_medians": medians,
                "reference_audit": audit,
            }
        )
    labels = labels_array(rows)
    row_index = {row["pair_uid"]: index for index, row in enumerate(rows)}
    weights = solver.component_weights(rows, cfg["sample_weight"])

    def evaluate(penalty: float) -> dict:
        oof = np.full(len(rows), np.nan, dtype=np.float64)
        fold_records = []
        for design in designs:
            artifact = fit_logistic(
                design["fit_matrix"], design["fit_rows"], penalty, policy
            )
            probabilities = solver.apply_logistic(design["hold_matrix"], artifact)
            indices = [row_index[row["pair_uid"]] for row in design["hold_rows"]]
            oof[np.asarray(indices, dtype=int)] = probabilities
            fold_records.append(
                {
                    "fold": design["fold"],
                    "fit_row_count": len(design["fit_rows"]),
                    "holdout_row_count": len(design["hold_rows"]),
                    "solver_iterations": artifact["solver_iterations"],
                    "solver_final_normalized_gradient_inf_norm": artifact[
                        "solver_final_normalized_gradient_inf_norm"
                    ],
                }
            )
        if not np.all(np.isfinite(oof)):
            raise ValueError("Step7-v4 inner OOF predictions are incomplete")
        return {
            "l2_penalty": float(penalty),
            "component_equal_average_precision": solver.weighted_average_precision(
                labels, oof, weights
            ),
            "row_average_precision": solver.average_precision(labels, oof),
            "component_equal_roc_auc": weighted_roc_auc(
                labels, oof, weights
            ),
            "roc_auc": solver.roc_auc(labels, oof),
            "folds": fold_records,
            "oof_scores": oof,
        }

    if feature_names:
        penalties = [float(value) for value in cfg["l2_initial_grid"]]
    else:
        penalties = [1.0]
    results = [evaluate(penalty) for penalty in penalties]

    def metric_key(result: dict) -> tuple[float, float, float, float]:
        return (
            float(result["component_equal_average_precision"]),
            float(result["row_average_precision"]),
            float(result["component_equal_roc_auc"]),
            float(result["roc_auc"]),
        )

    def selection_key(result: dict) -> tuple[float, float, float, float, float]:
        return (*metric_key(result), float(result["l2_penalty"]))

    stop_reason = "not_applicable_no_penalized_features"
    extension_count = 0
    if feature_names:
        stop_reason = "selected_interior"
        while True:
            results.sort(key=lambda item: float(item["l2_penalty"]))
            selected = max(results, key=selection_key)
            position = results.index(selected)
            if position not in {0, len(results) - 1}:
                stop_reason = "selected_interior"
                break
            alternatives = [item for item in results if item is not selected]
            strict_improvement = not alternatives or metric_key(selected) > max(
                metric_key(item) for item in alternatives
            )
            if not strict_improvement:
                stop_reason = "boundary_without_strict_metric_improvement"
                break
            if position == 0:
                proposed = float(selected["l2_penalty"]) / float(
                    cfg["l2_boundary_extension_factor"]
                )
                if proposed < float(cfg["l2_minimum"]) * (1.0 - 1e-12):
                    stop_reason = "lower_search_limit_reached"
                    break
            else:
                proposed = float(selected["l2_penalty"]) * float(
                    cfg["l2_boundary_extension_factor"]
                )
                if proposed > float(cfg["l2_maximum"]) * (1.0 + 1e-12):
                    stop_reason = "upper_search_limit_reached"
                    break
            if any(math.isclose(proposed, item["l2_penalty"]) for item in results):
                raise ValueError("Step7-v4 adaptive L2 search repeated a grid value")
            results.append(evaluate(proposed))
            extension_count += 1
    results.sort(key=lambda item: float(item["l2_penalty"]))
    selected = max(results, key=selection_key)
    threshold, threshold_audit = solver.choose_threshold(labels, selected["oof_scores"], weights)
    return {
        "selected_l2_penalty": float(selected["l2_penalty"]),
        "selected_l2_at_grid_boundary": bool(
            feature_names
            and (selected is results[0] or selected is results[-1])
        ),
        "l2_search_stop_reason": stop_reason,
        "l2_extension_count": extension_count,
        "l2_grid": [
            {key: value for key, value in item.items() if key != "oof_scores"}
            for item in results
        ],
        "oof_scores": selected["oof_scores"],
        "selected_threshold": threshold,
        "threshold_selection": threshold_audit,
        "fold_assignment_sha256": common.canonical_hash(assignments),
        "fold_diagnostics": solver.component_fold_diagnostics(
            rows, assignments, int(fold_count)
        ),
        "fold_reference_audit": [design["reference_audit"] for design in designs],
        "formal_fit_count": len(results) * int(fold_count),
    }


def candidate_rank(candidate_results: dict[str, dict]) -> list[str]:
    return sorted(
        candidate_results,
        key=lambda candidate_id: (
            -float(
                candidate_results[candidate_id]["metrics"]["component_equal"][
                    "average_precision"
                ]
            ),
            -float(
                candidate_results[candidate_id]["metrics"]["row"][
                    "average_precision"
                ]
            ),
            -float(
                candidate_results[candidate_id]["metrics"]["component_equal"][
                    "roc_auc"
                ]
            ),
            -float(candidate_results[candidate_id]["metrics"]["row"]["roc_auc"]),
            int(candidate_results[candidate_id]["feature_count"]),
            candidate_id,
        ),
    )


def run_nested_selection(
    policy: dict,
    factory: FeatureFactory,
    train_rows: list[dict],
) -> tuple[dict[str, dict], list[str], dict]:
    specs = common.candidate_specs(policy)
    cfg = policy["training"]
    row_index = {row["pair_uid"]: index for index, row in enumerate(train_rows)}
    state = {
        spec["id"]: {
            "candidate_id": spec["id"],
            "role": spec["role"],
            "blocks": spec["blocks"],
            "feature_names": spec["feature_names"],
            "feature_count": len(spec["feature_names"]),
            "selected_l2_values": [],
            "outer_fold_records": [],
            "seed_scores": [],
            "seed_metrics": [],
            "formal_fit_count": 0,
        }
        for spec in specs
    }
    outer_audit = []
    for outer_seed in cfg["outer_seeds"]:
        print(f"[Step7-v4] outer repeat seed={outer_seed}", flush=True)
        assignments = balanced_component_folds(
            train_rows, int(cfg["outer_fold_count"]), int(outer_seed)
        )
        seed_predictions = {
            spec["id"]: np.full(len(train_rows), np.nan, dtype=np.float64)
            for spec in specs
        }
        for outer_fold in range(int(cfg["outer_fold_count"])):
            print(
                f"[Step7-v4] outer seed={outer_seed} fold={outer_fold + 1}/"
                f"{int(cfg['outer_fold_count'])}",
                flush=True,
            )
            outer_fit = [
                row
                for row in train_rows
                if assignments[row["component_id"]] != outer_fold
            ]
            outer_hold = [
                row
                for row in train_rows
                if assignments[row["component_id"]] == outer_fold
            ]
            fit_sellers = {
                row[endpoint]
                for row in outer_fit
                for endpoint in ("seller_uid_left", "seller_uid_right")
            }
            hold_sellers = {
                row[endpoint]
                for row in outer_hold
                for endpoint in ("seller_uid_left", "seller_uid_right")
            }
            if fit_sellers & hold_sellers:
                raise ValueError("Step7-v4 outer component fold leaks a seller")
            inner_seed = derived_seed(outer_seed, outer_fold, "inner_component_folds")
            for candidate_index, spec in enumerate(specs, start=1):
                candidate_id = spec["id"]
                print(
                    f"[Step7-v4] fitting {candidate_id} "
                    f"({candidate_index}/{len(specs)})",
                    flush=True,
                )
                tuned = tune_l2(
                    policy,
                    factory,
                    outer_fit,
                    spec["feature_names"],
                    fold_count=int(cfg["inner_fold_count"]),
                    fold_seed=inner_seed,
                )
                fit_matrix, hold_matrix, medians, reference_audit = factory.design(
                    outer_fit, outer_hold, spec["feature_names"]
                )
                artifact = fit_logistic(
                    fit_matrix,
                    outer_fit,
                    tuned["selected_l2_penalty"],
                    policy,
                )
                probabilities = solver.apply_logistic(hold_matrix, artifact)
                indices = np.asarray(
                    [row_index[row["pair_uid"]] for row in outer_hold], dtype=int
                )
                seed_predictions[candidate_id][indices] = probabilities
                state[candidate_id]["selected_l2_values"].append(
                    float(tuned["selected_l2_penalty"])
                )
                state[candidate_id]["formal_fit_count"] += int(
                    tuned["formal_fit_count"]
                ) + 1
                state[candidate_id]["outer_fold_records"].append(
                    {
                        "outer_seed": int(outer_seed),
                        "outer_fold": outer_fold,
                        "fit_row_count": len(outer_fit),
                        "holdout_row_count": len(outer_hold),
                        "fit_holdout_seller_overlap_count": 0,
                        "inner_seed": inner_seed,
                        "inner_tuning": {
                            key: value
                            for key, value in tuned.items()
                            if key != "oof_scores"
                        },
                        "outer_imputation_medians": medians,
                        "outer_reference_audit": reference_audit,
                        "outer_solver_iterations": artifact["solver_iterations"],
                        "outer_solver_final_normalized_gradient_inf_norm": artifact[
                            "solver_final_normalized_gradient_inf_norm"
                        ],
                    }
                )
        seed_result_view = {}
        for spec in specs:
            candidate_id = spec["id"]
            scores = seed_predictions[candidate_id]
            if not np.all(np.isfinite(scores)):
                raise ValueError("Step7-v4 outer OOF predictions are incomplete")
            weights = solver.component_weights(train_rows, cfg["sample_weight"])
            threshold, _audit = solver.choose_threshold(
                labels_array(train_rows), scores, weights
            )
            metrics = full_metrics(train_rows, scores, threshold)
            state[candidate_id]["seed_scores"].append(scores)
            state[candidate_id]["seed_metrics"].append(
                {
                    "outer_seed": int(outer_seed),
                    "selected_threshold_diagnostic": threshold,
                    "metrics": metrics,
                }
            )
            seed_result_view[candidate_id] = {
                "feature_count": len(spec["feature_names"]),
                "metrics": metrics,
            }
        outer_audit.append(
            {
                "outer_seed": int(outer_seed),
                "component_fold_assignment_sha256": common.canonical_hash(assignments),
                "fold_diagnostics": solver.component_fold_diagnostics(
                    train_rows, assignments, int(cfg["outer_fold_count"])
                ),
                "seed_winner": candidate_rank(seed_result_view)[0],
            }
        )

    results = {}
    weights = solver.component_weights(train_rows, cfg["sample_weight"])
    labels = labels_array(train_rows)
    for spec in specs:
        candidate_id = spec["id"]
        item = state[candidate_id]
        scores_by_seed = np.vstack(item.pop("seed_scores"))
        if scores_by_seed.shape != (len(cfg["outer_seeds"]), len(train_rows)):
            raise AssertionError("Step7-v4 repeated outer OOF shape drift")
        mean_scores = np.mean(scores_by_seed, axis=0)
        threshold, threshold_audit = solver.choose_threshold(labels, mean_scores, weights)
        selected_values = item["selected_l2_values"]
        expected_values = len(cfg["outer_seeds"]) * int(cfg["outer_fold_count"])
        if len(selected_values) != expected_values:
            raise AssertionError("Step7-v4 final L2 aggregation count drift")
        final_l2 = float(np.median(np.asarray(selected_values, dtype=np.float64)))
        results[candidate_id] = {
            **item,
            "final_l2_penalty": final_l2,
            "selected_threshold": threshold,
            "threshold_selection": threshold_audit,
            "mean_repeated_nested_oof_scores": mean_scores,
            "outer_seed_oof_scores": scores_by_seed,
            "metrics": full_metrics(train_rows, mean_scores, threshold),
            "all_formal_fits_converged": True,
        }
    ranking = candidate_rank(results)
    audit = {
        "outer_fold_count": int(cfg["outer_fold_count"]),
        "outer_seeds": list(cfg["outer_seeds"]),
        "inner_fold_count": int(cfg["inner_fold_count"]),
        "outer_seed_audit": outer_audit,
        "all_candidate_formal_fit_count": sum(
            int(result["formal_fit_count"]) for result in results.values()
        ),
        "all_formal_fits_converged": True,
    }
    return results, ranking, audit


def grouped_bootstrap_delta(
    rows: list[dict],
    left_scores: np.ndarray,
    right_scores: np.ndarray,
    *,
    resamples: int,
    seed: int,
    confidence: float,
) -> dict:
    labels = labels_array(rows)
    left = np.asarray(left_scores, dtype=np.float64)
    right = np.asarray(right_scores, dtype=np.float64)
    if left.shape != labels.shape or right.shape != labels.shape:
        raise ValueError("Step7-v4 bootstrap score shape drift")
    grouped: dict[str, np.ndarray] = {}
    for component_id in sorted({row["component_id"] for row in rows}):
        grouped[component_id] = np.asarray(
            [index for index, row in enumerate(rows) if row["component_id"] == component_id],
            dtype=int,
        )
    components = sorted(grouped)
    rng = np.random.default_rng(int(seed))
    deltas = np.empty(int(resamples), dtype=np.float64)
    iteration = 0
    rejected_single_class = 0
    maximum_attempts = max(int(resamples) * 100, int(resamples) + 1000)
    attempts = 0
    while iteration < int(resamples):
        attempts += 1
        if attempts > maximum_attempts:
            raise ValueError("Step7-v4 bootstrap cannot obtain both-class resamples")
        draws = rng.integers(0, len(components), size=len(components))
        index_parts = [grouped[components[int(draw)]] for draw in draws]
        indices = np.concatenate(index_parts)
        if set(labels[indices].tolist()) != {0, 1}:
            rejected_single_class += 1
            continue
        weights = np.concatenate(
            [np.full(len(part), 1.0 / len(part), dtype=np.float64) for part in index_parts]
        )
        deltas[iteration] = solver.weighted_average_precision(
            labels[indices], left[indices], weights
        ) - solver.weighted_average_precision(labels[indices], right[indices], weights)
        iteration += 1
    alpha = (1.0 - float(confidence)) / 2.0
    return {
        "group": "component_id",
        "resamples": int(resamples),
        "seed": int(seed),
        "confidence": float(confidence),
        "bootstrap_attempt_count": attempts,
        "rejected_single_class_resample_count": rejected_single_class,
        "observed_delta": float(
            solver.weighted_average_precision(
                labels,
                left,
                solver.component_weights(rows, "component_equal_normalized_to_row_count"),
            )
            - solver.weighted_average_precision(
                labels,
                right,
                solver.component_weights(rows, "component_equal_normalized_to_row_count"),
            )
        ),
        "ci_lower": float(np.quantile(deltas, alpha)),
        "ci_upper": float(np.quantile(deltas, 1.0 - alpha)),
        "probability_delta_above_zero": float(np.mean(deltas > 0.0)),
        "zero_delta_probability": float(np.mean(deltas == 0.0)),
    }


def grouped_bootstrap_winner_above_all(
    rows: list[dict],
    candidate_scores: dict[str, np.ndarray],
    winner: str,
    *,
    resamples: int,
    seed: int,
    confidence: float,
) -> dict:
    """Selection-aware component bootstrap against the complete candidate set."""

    if winner not in candidate_scores or len(candidate_scores) < 2:
        raise ValueError("Step7-v4 simultaneous bootstrap candidate universe is invalid")
    labels = labels_array(rows)
    scores = {
        candidate_id: np.asarray(values, dtype=np.float64)
        for candidate_id, values in candidate_scores.items()
    }
    if any(
        values.shape != labels.shape or not np.all(np.isfinite(values))
        for values in scores.values()
    ):
        raise ValueError("Step7-v4 simultaneous bootstrap score shape/value drift")
    grouped = {
        component_id: np.asarray(
            [
                index
                for index, row in enumerate(rows)
                if row["component_id"] == component_id
            ],
            dtype=int,
        )
        for component_id in sorted({row["component_id"] for row in rows})
    }
    components = sorted(grouped)
    other_ids = sorted(set(scores) - {winner})
    rng = np.random.default_rng(int(seed))
    minimum_deltas = np.empty(int(resamples), dtype=np.float64)
    pairwise_wins = Counter()
    iteration = 0
    rejected_single_class = 0
    maximum_attempts = max(int(resamples) * 100, int(resamples) + 1000)
    attempts = 0
    while iteration < int(resamples):
        attempts += 1
        if attempts > maximum_attempts:
            raise ValueError(
                "Step7-v4 simultaneous bootstrap cannot obtain both-class resamples"
            )
        draws = rng.integers(0, len(components), size=len(components))
        parts = [grouped[components[int(draw)]] for draw in draws]
        indices = np.concatenate(parts)
        if set(labels[indices].tolist()) != {0, 1}:
            rejected_single_class += 1
            continue
        weights = np.concatenate(
            [
                np.full(len(part), 1.0 / len(part), dtype=np.float64)
                for part in parts
            ]
        )
        winner_ap = solver.weighted_average_precision(
            labels[indices], scores[winner][indices], weights
        )
        deltas = {}
        for candidate_id in other_ids:
            delta = winner_ap - solver.weighted_average_precision(
                labels[indices], scores[candidate_id][indices], weights
            )
            deltas[candidate_id] = delta
            pairwise_wins[candidate_id] += int(delta > 0.0)
        minimum_deltas[iteration] = min(deltas.values())
        iteration += 1
    full_weights = solver.component_weights(
        rows, "component_equal_normalized_to_row_count"
    )
    observed_winner_ap = solver.weighted_average_precision(
        labels, scores[winner], full_weights
    )
    observed_deltas = {
        candidate_id: observed_winner_ap
        - solver.weighted_average_precision(
            labels, scores[candidate_id], full_weights
        )
        for candidate_id in other_ids
    }
    alpha = (1.0 - float(confidence)) / 2.0
    return {
        "group": "component_id",
        "resamples": int(resamples),
        "seed": int(seed),
        "confidence": float(confidence),
        "bootstrap_attempt_count": attempts,
        "rejected_single_class_resample_count": rejected_single_class,
        "winner": winner,
        "candidate_count": len(scores),
        "observed_minimum_delta_above_any_competitor": float(
            min(observed_deltas.values())
        ),
        "observed_closest_competitor": min(
            observed_deltas, key=lambda candidate_id: observed_deltas[candidate_id]
        ),
        "minimum_delta_ci_lower": float(np.quantile(minimum_deltas, alpha)),
        "minimum_delta_ci_upper": float(
            np.quantile(minimum_deltas, 1.0 - alpha)
        ),
        "probability_winner_strictly_above_all_candidates": float(
            np.mean(minimum_deltas > 0.0)
        ),
        "probability_by_competitor": {
            candidate_id: pairwise_wins[candidate_id] / int(resamples)
            for candidate_id in other_ids
        },
    }


def assess_selection(
    policy: dict,
    train_rows: list[dict],
    results: dict[str, dict],
    ranking: list[str],
    nested_audit: dict,
    no_clone_rows: list[dict],
    no_clone_results: dict[str, dict],
    no_clone_ranking: list[str],
    no_clone_nested_audit: dict,
) -> dict:
    winner, runner_up = ranking[:2]
    bootstrap_cfg = policy["evaluation"]["bootstrap"]
    winner_delta = grouped_bootstrap_delta(
        train_rows,
        results[winner]["mean_repeated_nested_oof_scores"],
        results[runner_up]["mean_repeated_nested_oof_scores"],
        resamples=int(bootstrap_cfg["resamples"]),
        seed=int(bootstrap_cfg["seed"]),
        confidence=float(bootstrap_cfg["confidence"]),
    )
    simultaneous = grouped_bootstrap_winner_above_all(
        train_rows,
        {
            candidate_id: result["mean_repeated_nested_oof_scores"]
            for candidate_id, result in results.items()
        },
        winner,
        resamples=int(bootstrap_cfg["resamples"]),
        seed=derived_seed(bootstrap_cfg["seed"], "winner_above_all"),
        confidence=float(bootstrap_cfg["confidence"]),
    )
    if not no_clone_rows or {
        row["review_label"] for row in no_clone_rows
    } != {"positive", "negative"}:
        raise ValueError("Step7-v4 no-exact-clone robustness subset is not estimable")
    no_clone_simultaneous = grouped_bootstrap_winner_above_all(
        no_clone_rows,
        {
            candidate_id: result["mean_repeated_nested_oof_scores"]
            for candidate_id, result in no_clone_results.items()
        },
        winner,
        resamples=int(bootstrap_cfg["resamples"]),
        seed=derived_seed(bootstrap_cfg["seed"], "winner_above_all_no_clone"),
        confidence=float(bootstrap_cfg["confidence"]),
    )
    seed_winners = [item["seed_winner"] for item in nested_audit["outer_seed_audit"]]
    winner_rate = seed_winners.count(winner) / len(seed_winners)
    no_clone_seed_winners = [
        item["seed_winner"]
        for item in no_clone_nested_audit["outer_seed_audit"]
    ]
    no_clone_winner_rate = (
        no_clone_seed_winners.count(winner) / len(no_clone_seed_winners)
    )
    rule = policy["selection_rule"]["unique_provisional_m0_requires"]
    unique = bool(
        winner_rate >= float(rule["winner_rate_across_outer_repeats_at_least"])
        and winner_delta["probability_delta_above_zero"]
        >= float(rule["component_bootstrap_probability_delta_above_runner_up_at_least"])
        and simultaneous[
            "probability_winner_strictly_above_all_candidates"
        ]
        >= float(
            rule[
                "simultaneous_component_bootstrap_probability_winner_above_all_candidates_at_least"
            ]
        )
        and no_clone_winner_rate
        >= float(
            rule[
                "no_exact_clone_nested_winner_rate_across_outer_repeats_at_least"
            ]
        )
        and no_clone_simultaneous[
            "probability_winner_strictly_above_all_candidates"
        ]
        >= float(
            rule[
                "no_exact_clone_component_bootstrap_probability_winner_above_all_candidates_at_least"
            ]
        )
        and nested_audit["all_formal_fits_converged"] is True
    )

    matched_encoder_ids = [
        "matched__e5_stylometry",
        "matched__labse_stylometry",
        "style__pcm_stylometry",
        "style__mstyle_stylometry",
    ]
    matched_encoder_ranking = [
        candidate_id
        for candidate_id in ranking
        if candidate_id in matched_encoder_ids
    ]
    if len(matched_encoder_ranking) != len(matched_encoder_ids):
        raise AssertionError("Step7-v4 matched single-encoder comparison drift")
    matched_encoder_winner = matched_encoder_ranking[0]
    matched_encoder_stability = grouped_bootstrap_winner_above_all(
        train_rows,
        {
            candidate_id: results[candidate_id][
                "mean_repeated_nested_oof_scores"
            ]
            for candidate_id in matched_encoder_ids
        },
        matched_encoder_winner,
        resamples=int(bootstrap_cfg["resamples"]),
        seed=derived_seed(bootstrap_cfg["seed"], "matched_single_encoder"),
        confidence=float(bootstrap_cfg["confidence"]),
    )

    style_ids = [candidate_id for candidate_id in ranking if candidate_id.startswith("style__")]
    if not style_ids:
        raise AssertionError("Step7-v4 policy has no pure style candidate")
    best_style = style_ids[0]
    primary_style_ids = [
        candidate_id
        for candidate_id, result in results.items()
        if result["role"] == "primary_style_candidate"
    ]
    if primary_style_ids != ["style__pcm_mstyle_stylometry"]:
        raise AssertionError("Step7-v4 preregistered primary style candidate drift")
    primary_style = primary_style_ids[0]
    semantic_ids = [
        "encoder__e5",
        "encoder__labse",
        "matched__e5_stylometry",
        "matched__labse_stylometry",
    ]
    best_semantic = next(candidate_id for candidate_id in ranking if candidate_id in semantic_ids)
    style_vs_simple = grouped_bootstrap_delta(
        train_rows,
        results[primary_style]["mean_repeated_nested_oof_scores"],
        results["control__stylometry"]["mean_repeated_nested_oof_scores"],
        resamples=int(bootstrap_cfg["resamples"]),
        seed=derived_seed(bootstrap_cfg["seed"], "style_vs_simple"),
        confidence=float(bootstrap_cfg["confidence"]),
    )
    style_vs_semantic = grouped_bootstrap_delta(
        train_rows,
        results[primary_style]["mean_repeated_nested_oof_scores"],
        results[best_semantic]["mean_repeated_nested_oof_scores"],
        resamples=int(bootstrap_cfg["resamples"]),
        seed=derived_seed(bootstrap_cfg["seed"], "style_vs_semantic"),
        confidence=float(bootstrap_cfg["confidence"]),
    )
    increment_rule = policy["selection_rule"]["style_increment_claim_requires"]
    style_increment = bool(
        style_vs_simple["ci_lower"]
        > float(
            increment_rule[
                "preregistered_primary_style_candidate_component_bootstrap_ci_lower_above_simple_style_control"
            ]
        )
        and style_vs_semantic["ci_lower"]
        > float(
            increment_rule[
                "preregistered_primary_style_candidate_component_bootstrap_ci_lower_above_best_semantic_control"
            ]
        )
    )
    winner_blocks = set(results[winner]["blocks"])
    encoder_blocks = {"pcm6", "mstyle6", "e5_6", "labse6"}
    encoder_type = bool(winner_blocks & encoder_blocks)
    return {
        "winner": winner,
        "runner_up": runner_up,
        "seed_winners": seed_winners,
        "winner_rate_across_outer_seeds": winner_rate,
        "winner_vs_runner_up_component_bootstrap": winner_delta,
        "winner_vs_all_candidates_simultaneous_component_bootstrap": simultaneous,
        "winner_no_exact_clone_robustness": {
            "training_contract": (
                "complete_repeated_nested_cv_retrained_after_removing_all_pairs_"
                "with_exact_clean_title_or_description_overlap"
            ),
            "row_count": len(no_clone_rows),
            "positive_count": sum(
                row["review_label"] == "positive" for row in no_clone_rows
            ),
            "negative_count": sum(
                row["review_label"] == "negative" for row in no_clone_rows
            ),
            "candidate_ranking": no_clone_ranking,
            "outer_seed_winners": no_clone_seed_winners,
            "original_winner_rate_across_no_clone_outer_seeds": (
                no_clone_winner_rate
            ),
            "simultaneous_component_bootstrap": no_clone_simultaneous,
        },
        "unique_provisional_m0_gate_passed": unique,
        "selection_status": (
            "provisional_m0_selected_requires_new_real_english_confirmation"
            if unique
            else "no_stable_unique_provisional_m0"
        ),
        "winner_is_encoder_type_pipeline": encoder_type,
        "encoder_type_status": (
            "provisional_encoder_type_m0_requires_new_real_english_confirmation"
            if unique and encoder_type
            else "no_unique_encoder_type_m0"
        ),
        "best_pure_style_candidate": best_style,
        "matched_single_encoder_comparison": {
            "shared_added_block": "stylometry22",
            "candidate_ids": matched_encoder_ids,
            "ranking": matched_encoder_ranking,
            "winner": matched_encoder_winner,
            "winner_vs_other_encoders_component_bootstrap": matched_encoder_stability,
            "stable_unique_encoder_at_0_95": matched_encoder_stability[
                "probability_winner_strictly_above_all_candidates"
            ]
            >= 0.95,
        },
        "preregistered_primary_style_candidate_for_increment_claim": primary_style,
        "best_semantic_control": best_semantic,
        "style_vs_stylometry_bootstrap": style_vs_simple,
        "style_vs_best_semantic_bootstrap": style_vs_semantic,
        "style_increment_claim_passed": style_increment,
        "style_increment_status": (
            "supported_on_repeated_nested_train_oof_requires_new_confirmation"
            if style_increment
            else "not_supported"
        ),
        "validation_metrics_used_for_selection": False,
        "historical_test_labels_read": False,
        "new_real_english_confirmation_required": True,
    }


def single_class_slice_metrics(
    rows: list[dict], scores: np.ndarray, threshold: float
) -> dict:
    labels = np.asarray(
        [1 if row["review_label"] == "positive" else 0 for row in rows], dtype=np.int8
    )
    probabilities = np.asarray(scores, dtype=np.float64)
    predicted = probabilities >= float(threshold)
    result = {
        "status": "single_class_threshold_diagnostic",
        "row_count": len(rows),
        "positive_count": int(np.sum(labels == 1)),
        "negative_count": int(np.sum(labels == 0)),
        "mean_probability": float(np.mean(probabilities)),
        "predicted_positive_rate": float(np.mean(predicted)),
    }
    if np.all(labels == 1):
        result["recall"] = float(np.mean(predicted))
    elif np.all(labels == 0):
        result["false_positive_rate"] = float(np.mean(predicted))
    else:
        raise ValueError("Step7-v4 single-class slice helper received both classes")
    return result


def subset_metrics(
    rows: list[dict], scores: np.ndarray, mask: np.ndarray, threshold: float
) -> dict:
    indices = np.flatnonzero(mask)
    if len(indices) == 0:
        return {"status": "not_estimable_empty_slice", "row_count": 0}
    selected_rows = [rows[int(index)] for index in indices]
    selected_scores = np.asarray(scores, dtype=np.float64)[indices]
    labels = {row["review_label"] for row in selected_rows}
    if labels == {"positive", "negative"}:
        return {"status": "estimable_both_classes", **full_metrics(selected_rows, selected_scores, threshold)}
    return single_class_slice_metrics(selected_rows, selected_scores, threshold)


def evaluation_slices(
    policy: dict,
    rows: list[dict],
    scores: np.ndarray,
    threshold: float,
    overlap_by_pair: dict[str, dict[str, bool]],
    seller_markets: dict[str, str],
) -> dict:
    hard_negative_types = set(policy["evaluation"]["hard_negative_evidence_types"])
    direct_positive_types = set(policy["evaluation"]["direct_positive_evidence_types"])
    masks = {
        "hard_negatives": np.asarray(
            [
                row["review_label"] == "negative"
                and row["evidence_type"] in hard_negative_types
                for row in rows
            ],
            dtype=bool,
        ),
        "direct_positives": np.asarray(
            [
                row["review_label"] == "positive"
                and row["evidence_type"] in direct_positive_types
                for row in rows
            ],
            dtype=bool,
        ),
        "no_exact_clean_title_or_description_overlap": np.asarray(
            [not overlap_by_pair[row["pair_uid"]]["any_exact_clean_text_overlap"] for row in rows],
            dtype=bool,
        ),
        "same_market": np.asarray(
            [
                seller_markets[row["seller_uid_left"]]
                == seller_markets[row["seller_uid_right"]]
                for row in rows
            ],
            dtype=bool,
        ),
    }
    masks["cross_market"] = ~masks["same_market"]
    return {
        name: subset_metrics(rows, scores, mask, threshold)
        for name, mask in masks.items()
    }


def leave_one_market_stress(
    policy: dict,
    factory: FeatureFactory,
    train_rows: list[dict],
    winner_spec: dict,
    seller_markets: dict[str, str],
) -> dict:
    component_markets: dict[str, set[str]] = defaultdict(set)
    for row in train_rows:
        component_markets[row["component_id"]].update(
            (
                seller_markets[row["seller_uid_left"]],
                seller_markets[row["seller_uid_right"]],
            )
        )
    markets = sorted({market for values in component_markets.values() for market in values})
    records = []
    for market in markets:
        hold_components = {
            component_id
            for component_id, values in component_markets.items()
            if market in values
        }
        fit_rows = [row for row in train_rows if row["component_id"] not in hold_components]
        hold_rows = [row for row in train_rows if row["component_id"] in hold_components]
        fit_labels = Counter(row["review_label"] for row in fit_rows)
        hold_labels = Counter(row["review_label"] for row in hold_rows)
        positive_components = {
            row["component_id"] for row in fit_rows if row["review_label"] == "positive"
        }
        negative_components = {
            row["component_id"] for row in fit_rows if row["review_label"] == "negative"
        }
        base = {
            "market": market,
            "fit_row_count": len(fit_rows),
            "holdout_row_count": len(hold_rows),
            "fit_component_count": len({row["component_id"] for row in fit_rows}),
            "holdout_component_count": len(hold_components),
            "fit_label_counts": dict(sorted(fit_labels.items())),
            "holdout_label_counts": dict(sorted(hold_labels.items())),
        }
        if (
            set(fit_labels) != {"positive", "negative"}
            or set(hold_labels) != {"positive", "negative"}
            or len(positive_components) < int(policy["training"]["inner_fold_count"])
            or len(negative_components) < int(policy["training"]["inner_fold_count"])
        ):
            records.append({**base, "status": "not_estimable_class_or_component_support"})
            continue
        fold_seed = derived_seed(policy["evaluation"]["bootstrap"]["seed"], "leave_market", market)
        try:
            balanced_component_folds(
                fit_rows, int(policy["training"]["inner_fold_count"]), fold_seed
            )
        except ValueError:
            records.append({**base, "status": "not_estimable_fold_class_support"})
            continue
        tuned = tune_l2(
            policy,
            factory,
            fit_rows,
            winner_spec["feature_names"],
            fold_count=int(policy["training"]["inner_fold_count"]),
            fold_seed=fold_seed,
        )
        fit_matrix, hold_matrix, medians, reference_audit = factory.design(
            fit_rows, hold_rows, winner_spec["feature_names"]
        )
        artifact = fit_logistic(
            fit_matrix, fit_rows, tuned["selected_l2_penalty"], policy
        )
        scores = solver.apply_logistic(hold_matrix, artifact)
        records.append(
            {
                **base,
                "status": "estimable_train_only_market_stress",
                "fold_seed": fold_seed,
                "selected_l2_penalty": tuned["selected_l2_penalty"],
                "selected_threshold": tuned["selected_threshold"],
                "imputation_medians": medians,
                "reference_audit": reference_audit,
                "metrics": full_metrics(
                    hold_rows, scores, tuned["selected_threshold"]
                ),
            }
        )
    return {
        "status": "diagnostic_does_not_control_selection",
        "winner_candidate_id": winner_spec["id"],
        "market_count": len(markets),
        "estimable_market_count": sum(
            record["status"] == "estimable_train_only_market_stress"
            for record in records
        ),
        "records": records,
    }


def fit_final_candidates_before_valid_labels(
    policy: dict,
    factory: FeatureFactory,
    train_rows: list[dict],
    valid_pair_rows: list[dict],
    nested_results: dict[str, dict],
) -> tuple[dict[str, dict], dict[str, np.ndarray]]:
    artifacts = {}
    valid_scores = {}
    for spec in common.candidate_specs(policy):
        candidate_id = spec["id"]
        fit_matrix, valid_matrix, medians, reference_audit = factory.design(
            train_rows, valid_pair_rows, spec["feature_names"]
        )
        l2_penalty = float(nested_results[candidate_id]["final_l2_penalty"])
        logistic = fit_logistic(fit_matrix, train_rows, l2_penalty, policy)
        scores = solver.apply_logistic(valid_matrix, logistic)
        if not np.all(np.isfinite(scores)):
            raise ValueError("Step7-v4 final model produced non-finite valid scores")
        artifacts[candidate_id] = {
            "candidate_id": candidate_id,
            "role": spec["role"],
            "blocks": spec["blocks"],
            "feature_names": spec["feature_names"],
            "feature_count": len(spec["feature_names"]),
            "l2_penalty": l2_penalty,
            "selected_threshold": float(
                nested_results[candidate_id]["selected_threshold"]
            ),
            "imputation_medians": medians,
            "standardization_and_logistic": logistic,
            "train_reference_audit": reference_audit,
            "fit_row_count": len(train_rows),
            "unlabelled_valid_score_count": len(valid_pair_rows),
            "valid_label_values_read_for_fit_or_scoring": False,
            "diagnostic_evidence_values_read_for_fit_or_scoring": False,
            "historical_test_label_values_read": False,
        }
        valid_scores[candidate_id] = scores
    return artifacts, valid_scores


def train_prediction_rows(
    policy: dict,
    train_rows: list[dict],
    ranking: list[str],
    nested_results: dict[str, dict],
) -> list[dict]:
    seeds = list(policy["training"]["outer_seeds"])
    output = []
    for candidate_id in ranking:
        result = nested_results[candidate_id]
        seed_scores = np.asarray(result["outer_seed_oof_scores"], dtype=np.float64)
        mean_scores = np.asarray(
            result["mean_repeated_nested_oof_scores"], dtype=np.float64
        )
        for index, row in enumerate(train_rows):
            record: dict[str, object] = {
                "pair_uid": row["pair_uid"],
                "component_id": row["component_id"],
                "review_label": row["review_label"],
                "candidate_id": candidate_id,
                "mean_repeated_nested_oof_probability": serialize_probability(
                    mean_scores[index]
                ),
            }
            for seed_index, seed in enumerate(seeds):
                record[f"outer_seed_{seed}_oof_probability"] = (
                    serialize_probability(seed_scores[seed_index, index])
                )
            output.append(record)
    return output


def blind_valid_prediction_rows(
    valid_pair_rows: list[dict],
    ranking: list[str],
    valid_scores: dict[str, np.ndarray],
) -> list[dict]:
    """Serialize validation scores without labels, components, or evidence."""

    output = []
    for candidate_id in ranking:
        scores = np.asarray(valid_scores[candidate_id], dtype=np.float64)
        if scores.shape != (len(valid_pair_rows),) or not np.all(
            np.isfinite(scores)
        ):
            raise ValueError(
                "Step7-v4 blind validation score shape/value drift"
            )
        for row, score in zip(valid_pair_rows, scores, strict=True):
            output.append(
                {
                    "pair_uid": row["pair_uid"],
                    "candidate_id": candidate_id,
                    "probability": serialize_probability(score),
                }
            )
    return output


def serialize_probability(value: float) -> str:
    number = float(value)
    if not math.isfinite(number) or number < 0.0 or number > 1.0:
        raise ValueError("Step7-v4 probability is non-finite or outside [0, 1]")
    rendered = repr(number)
    if float(rendered) != number:
        raise AssertionError("Step7-v4 probability serialization is not exact")
    return rendered


def replay_blind_valid_scores(
    valid_pair_rows: list[dict],
    ranking: list[str],
    rows: list[dict],
) -> dict[str, np.ndarray]:
    """Rebuild the only validation score arrays allowed after the blind lock."""

    expected_count = len(valid_pair_rows) * len(ranking)
    if len(rows) != expected_count:
        raise ValueError("Step7-v4 blind prediction row-count drift")
    output = {}
    position = 0
    seen = set()
    for candidate_id in ranking:
        values = []
        for pair_row in valid_pair_rows:
            row = rows[position]
            position += 1
            key = (row.get("pair_uid"), row.get("candidate_id"))
            if (
                list(row) != ["pair_uid", "candidate_id", "probability"]
                or row.get("pair_uid") != pair_row["pair_uid"]
                or row.get("candidate_id") != candidate_id
                or key in seen
            ):
                raise ValueError("Step7-v4 blind prediction order/key drift")
            seen.add(key)
            try:
                number = float(row["probability"])
            except (TypeError, ValueError) as error:
                raise ValueError(
                    "Step7-v4 blind prediction probability parse drift"
                ) from error
            if serialize_probability(number) != row["probability"]:
                raise ValueError(
                    "Step7-v4 blind prediction is not canonical exact-round-trip"
                )
            values.append(number)
        output[candidate_id] = np.asarray(values, dtype=np.float64)
    if len(seen) != expected_count or position != expected_count:
        raise AssertionError("Step7-v4 blind prediction replay accounting drift")
    return output


def valid_prediction_rows(
    valid_rows: list[dict],
    ranking: list[str],
    valid_scores: dict[str, np.ndarray],
) -> list[dict]:
    output = []
    for candidate_id in ranking:
        scores = valid_scores[candidate_id]
        for row, score in zip(valid_rows, scores, strict=True):
            output.append(
                {
                    "pair_uid": row["pair_uid"],
                    "component_id": row["component_id"],
                    "review_label": row["review_label"],
                    "candidate_id": candidate_id,
                    "probability": serialize_probability(score),
                }
            )
    return output


def json_ready(value):
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return [json_ready(item) for item in value.tolist()]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def compact_candidate_result(result: dict) -> dict:
    return {
        key: json_ready(value)
        for key, value in result.items()
        if key not in {
            "mean_repeated_nested_oof_scores",
            "outer_seed_oof_scores",
        }
    }


def run_selection(policy: dict) -> dict:
    preparation_manifest, preparation_bundle = common.validate_preparation_artifacts(
        policy
    )
    pair_rows = preparation_bundle["pair_rows"]
    runtime_provenance, fixed_features = verify_gpu_outputs(
        policy, preparation_manifest, preparation_bundle
    )
    seller_records, seller_markets, legacy_replay = replay_legacy_context(
        policy, pair_rows, fixed_features
    )
    factory = FeatureFactory(policy, pair_rows, fixed_features, seller_records)

    # Only the train label file is opened during model/hyperparameter/threshold
    # selection.  The valid file is deliberately opened much later below.
    train_selection_rows = load_label_split(policy, pair_rows, "train")
    nested_results, ranking, nested_audit = run_nested_selection(
        policy, factory, train_selection_rows
    )
    overlap_by_pair = common.exact_overlap_audit_by_pair(
        pair_rows, preparation_bundle["seller_text_rows"]
    )
    no_clone_train_rows = [
        row
        for row in train_selection_rows
        if not overlap_by_pair[row["pair_uid"]][
            "any_exact_clean_text_overlap"
        ]
    ]
    if (
        not no_clone_train_rows
        or {row["review_label"] for row in no_clone_train_rows}
        != {"positive", "negative"}
    ):
        raise ValueError(
            "Step7-v4 no-exact-clone nested retraining lacks both classes"
        )
    print(
        "[Step7-v4] starting complete no-exact-clone nested-CV retraining",
        flush=True,
    )
    (
        no_clone_nested_results,
        no_clone_ranking,
        no_clone_nested_audit,
    ) = run_nested_selection(policy, factory, no_clone_train_rows)
    decision = assess_selection(
        policy,
        train_selection_rows,
        nested_results,
        ranking,
        nested_audit,
        no_clone_train_rows,
        no_clone_nested_results,
        no_clone_ranking,
        no_clone_nested_audit,
    )
    winner = decision["winner"]
    specs = {spec["id"]: spec for spec in common.candidate_specs(policy)}
    outputs = policy["outputs"]
    lock_payload = {
        "step": "step7_v4_train_only_selection_lock",
        "version": policy["version"],
        "ranking": ranking,
        "decision": decision,
        "candidate_final_l2_and_threshold": {
            candidate_id: {
                "final_l2_penalty": nested_results[candidate_id]["final_l2_penalty"],
                "selected_threshold": nested_results[candidate_id]["selected_threshold"],
                "feature_names": nested_results[candidate_id]["feature_names"],
            }
            for candidate_id in ranking
        },
        "policy_sha256": common.sha256_file(common.DEFAULT_POLICY),
        "selector_sha256": common.sha256_file(SELECTOR_SCRIPT),
        "diagnostic_evidence_values_read": False,
        "valid_label_values_read": False,
        "historical_test_label_values_read": False,
    }
    lock_payload = json_ready(lock_payload)
    lock_payload["lock_content_sha256"] = common.canonical_hash(lock_payload)
    selection_lock_path = common.resolve(outputs["train_selection_lock"])
    common.write_json_immutable(selection_lock_path, lock_payload)
    observed_selection_lock = common.load_json(selection_lock_path)
    common.verify_canonical_self_hash(
        observed_selection_lock,
        "lock_content_sha256",
        "train-only selection lock",
    )
    if observed_selection_lock != lock_payload:
        raise ValueError("Step7-v4 train-only selection lock replay drift")
    print(
        f"[Step7-v4] immutable train-only selection lock written: "
        f"winner={winner}",
        flush=True,
    )

    market_stress = leave_one_market_stress(
        policy, factory, train_selection_rows, specs[winner], seller_markets
    )

    valid_pair_rows = [row for row in pair_rows if row["split_name"] == "valid"]
    model_artifacts, unlabelled_valid_scores = fit_final_candidates_before_valid_labels(
        policy,
        factory,
        train_selection_rows,
        valid_pair_rows,
        nested_results,
    )

    artifact_payload = {
        "step": "step7_v4_final_train_logistic_artifacts",
        "version": policy["version"],
        "train_selection_lock": common.file_record(selection_lock_path),
        "train_selection_lock_content_sha256": lock_payload[
            "lock_content_sha256"
        ],
        "winner_candidate_id": winner,
        "candidate_ranking": ranking,
        "valid_label_values_read_for_fit_or_scoring": False,
        "diagnostic_evidence_values_read_for_fit_or_scoring": False,
        "historical_test_label_values_read": False,
        "candidates": model_artifacts,
        "policy_sha256": common.sha256_file(common.DEFAULT_POLICY),
        "producer_sha256": common.sha256_file(SELECTOR_SCRIPT),
    }
    artifact_payload = json_ready(artifact_payload)
    artifact_payload["artifact_content_sha256"] = common.canonical_hash(
        artifact_payload
    )
    artifact_path = common.resolve(outputs["model_artifacts"])
    common.write_json_immutable(artifact_path, artifact_payload)
    observed_artifacts = common.load_json(artifact_path)
    common.verify_canonical_self_hash(
        observed_artifacts,
        "artifact_content_sha256",
        "final train logistic artifacts",
    )
    if observed_artifacts != artifact_payload:
        raise ValueError("Step7-v4 final model artifact replay drift")

    blind_predictions = blind_valid_prediction_rows(
        valid_pair_rows, ranking, unlabelled_valid_scores
    )
    blind_prediction_path = common.resolve(
        outputs["blind_valid_predictions"]
    )
    common.write_csv_immutable(blind_prediction_path, blind_predictions)
    observed_blind_predictions = common.load_csv(blind_prediction_path)
    if (
        observed_blind_predictions != blind_predictions
        or any(
            list(row) != ["pair_uid", "candidate_id", "probability"]
            for row in observed_blind_predictions
        )
    ):
        raise ValueError("Step7-v4 blind validation prediction replay drift")
    locked_valid_scores = replay_blind_valid_scores(
        valid_pair_rows, ranking, observed_blind_predictions
    )
    for candidate_id in ranking:
        original = np.asarray(
            unlabelled_valid_scores[candidate_id], dtype=np.float64
        )
        if not np.array_equal(locked_valid_scores[candidate_id], original):
            raise ValueError(
                "Step7-v4 blind prediction exact numeric round-trip drift: "
                f"{candidate_id}"
            )

    blind_lock_payload = {
        "step": "step7_v4_blind_validation_scoring_lock",
        "version": policy["version"],
        "train_selection_lock": common.file_record(selection_lock_path),
        "model_artifacts": common.file_record(artifact_path),
        "blind_valid_predictions": common.file_record(
            blind_prediction_path
        ),
        "candidate_ranking": ranking,
        "candidate_count": len(ranking),
        "valid_pair_count": len(valid_pair_rows),
        "blind_prediction_row_count": len(blind_predictions),
        "blind_prediction_columns": [
            "pair_uid",
            "candidate_id",
            "probability",
        ],
        "probability_serialization": "python_float_repr_exact_round_trip",
        "locked_scores_replayed_before_validation_labels": True,
        "valid_label_values_read": False,
        "diagnostic_evidence_values_read": False,
        "historical_test_label_values_read": False,
        "policy_sha256": common.sha256_file(common.DEFAULT_POLICY),
        "producer_sha256": common.sha256_file(SELECTOR_SCRIPT),
    }
    blind_lock_payload = json_ready(blind_lock_payload)
    blind_lock_payload["lock_content_sha256"] = common.canonical_hash(
        blind_lock_payload
    )
    blind_lock_path = common.resolve(outputs["blind_scoring_lock"])
    common.write_json_immutable(blind_lock_path, blind_lock_payload)
    observed_blind_lock = common.load_json(blind_lock_path)
    common.verify_canonical_self_hash(
        observed_blind_lock,
        "lock_content_sha256",
        "blind validation scoring lock",
    )
    if observed_blind_lock != blind_lock_payload:
        raise ValueError("Step7-v4 blind validation scoring lock replay drift")
    del original
    del unlabelled_valid_scores

    train_predictions = train_prediction_rows(
        policy, train_selection_rows, ranking, nested_results
    )
    train_prediction_path = common.resolve(outputs["train_oof_predictions"])
    common.write_csv_immutable(train_prediction_path, train_predictions)
    print(
        "[Step7-v4] final train fits, model artifacts, blind validation "
        "predictions, and blind-scoring lock are immutable; opening "
        "diagnostic evidence/valid labels",
        flush=True,
    )

    # The two lock files, all final coefficients, and label-free validation
    # scores are now immutable and replay-verified.  Only at this point are
    # train evidence and valid labels/evidence opened for diagnostics.
    train_rows = load_evidence_split(
        policy, train_selection_rows, "train"
    )
    train_slices = evaluation_slices(
        policy,
        train_rows,
        nested_results[winner]["mean_repeated_nested_oof_scores"],
        nested_results[winner]["selected_threshold"],
        overlap_by_pair,
        seller_markets,
    )
    valid_label_rows = load_label_split(policy, pair_rows, "valid")
    if [row["pair_uid"] for row in valid_label_rows] != [
        row["pair_uid"] for row in valid_pair_rows
    ]:
        raise ValueError("Step7-v4 valid label/prescore order drift")
    valid_rows = load_evidence_split(policy, valid_label_rows, "valid")
    valid_metrics = {
        candidate_id: full_metrics(
            valid_rows,
            locked_valid_scores[candidate_id],
            nested_results[candidate_id]["selected_threshold"],
        )
        for candidate_id in ranking
    }
    valid_slices = evaluation_slices(
        policy,
        valid_rows,
        locked_valid_scores[winner],
        nested_results[winner]["selected_threshold"],
        overlap_by_pair,
        seller_markets,
    )

    valid_predictions = valid_prediction_rows(
        valid_rows, ranking, locked_valid_scores
    )
    valid_prediction_path = common.resolve(outputs["valid_predictions"])
    common.write_csv_immutable(valid_prediction_path, valid_predictions)

    candidate_summaries = {
        candidate_id: compact_candidate_result(nested_results[candidate_id])
        for candidate_id in ranking
    }
    summary = {
        "step": "step7_v4_raw_item_authorship_source_selection",
        "version": policy["version"],
        "objective": policy["objective"],
        "train_selection_lock": lock_payload,
        "train_selection_lock_file": common.file_record(
            selection_lock_path
        ),
        "blind_scoring_lock": blind_lock_payload,
        "blind_scoring_lock_file": common.file_record(blind_lock_path),
        "train_only_candidate_ranking": ranking,
        "selection_decision": decision,
        "nested_training_audit": nested_audit,
        "candidate_train_results": candidate_summaries,
        "no_exact_clone_nested_training_audit": no_clone_nested_audit,
        "no_exact_clone_candidate_train_results": {
            candidate_id: compact_candidate_result(
                no_clone_nested_results[candidate_id]
            )
            for candidate_id in no_clone_ranking
        },
        "winner_train_slices": train_slices,
        "winner_leave_one_market_stress": market_stress,
        "valid_loaded_after_selection_lock": True,
        "valid_loaded_after_blind_scoring_lock": True,
        "valid_scores_replayed_from_locked_blind_predictions": True,
        "diagnostic_evidence_loaded_after_all_model_fits_and_scoring": True,
        "valid_metrics_may_change_selection": False,
        "candidate_valid_development_metrics": valid_metrics,
        "winner_valid_development_slices": valid_slices,
        "historical_test_labels_read": False,
        "new_real_english_confirmation_required": True,
        "legacy18_fold_reference_replay": legacy_replay,
        "development_supervision_provenance": {
            "manifest_sha256": common.sha256_file(
                common.resolve(outputs["development_labels_manifest"])
            ),
            "train_selection_labels_sha256": common.sha256_file(
                common.resolve(outputs["train_labels"])
            ),
            "valid_diagnostic_labels_sha256": common.sha256_file(
                common.resolve(outputs["valid_labels"])
            ),
            "train_diagnostic_evidence_sha256": common.sha256_file(
                common.resolve(outputs["train_evidence"])
            ),
            "valid_diagnostic_evidence_sha256": common.sha256_file(
                common.resolve(outputs["valid_evidence"])
            ),
            "evidence_loaded_after_all_model_fits_and_scoring": True,
            "historical_test_supervision_materialized": False,
        },
        "source_preparation": {
            "manifest_content_sha256": preparation_manifest[
                "manifest_content_sha256"
            ],
            "raw_item_lineage_count": preparation_manifest["counts"][
                "raw_item_lineage_count"
            ],
            "global_unique_clean_text_count": preparation_manifest["counts"][
                "global_unique_clean_text_count"
            ],
        },
        "gpu_provenance": {
            "gpu_sync_manifest_sha256": common.sha256_file(
                common.resolve(outputs["gpu_sync_manifest"])
            ),
            "gpu_output_manifest_sha256": common.sha256_file(
                common.resolve(outputs["gpu_output_manifest"])
            ),
            "shared_chunks_manifest_sha256": common.sha256_file(
                common.resolve(outputs["shared_chunks_manifest"])
            ),
            "model_runtime_manifest_sha256": {
                model_key: common.sha256_file(
                    common.resolve(
                        outputs["model_runtime_manifest_template"].format(
                            model_key=model_key
                        )
                    )
                )
                for model_key in common.MODEL_KEYS
            },
            "embedding_matrices_published": False,
            "runtime_model_count": len(runtime_provenance["models"]),
            "multiplicity_sensitivity_audit": runtime_provenance[
                "multiplicity_sensitivity"
            ],
        },
        "selection_execution_environment": {
            "python_version": platform.python_version(),
            "numpy_version": np.__version__,
            "platform": platform.platform(),
            "numerical_solver": policy["training"]["solver"],
        },
        "outputs": {
            "train_selection_lock": common.file_record(selection_lock_path),
            "model_artifacts": common.file_record(artifact_path),
            "train_oof_predictions": common.file_record(train_prediction_path),
            "blind_valid_predictions": common.file_record(
                blind_prediction_path
            ),
            "blind_scoring_lock": common.file_record(blind_lock_path),
            "valid_predictions": common.file_record(valid_prediction_path),
        },
        "policy_sha256": common.sha256_file(common.DEFAULT_POLICY),
        "producer_sha256": common.sha256_file(SELECTOR_SCRIPT),
    }
    summary = json_ready(summary)
    summary["summary_content_sha256"] = common.canonical_hash(summary)
    summary_path = common.resolve(outputs["selection_summary"])
    common.write_json_immutable(summary_path, summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--validate-config-only",
        action="store_true",
        help="Validate the frozen policy and implementation pins without numerical work.",
    )
    args = parser.parse_args()
    policy = common.load_policy()
    common.verify_implementation_files(policy)
    if args.validate_config_only:
        print(
            json.dumps(
                {
                    "status": "pass",
                    "candidate_count": len(common.candidate_specs(policy)),
                    "historical_test_labels_may_be_materialized": False,
                    "numerical_execution_performed": False,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return
    summary = run_selection(policy)
    print(
        json.dumps(
            {
                "status": summary["selection_decision"]["selection_status"],
                "winner": summary["selection_decision"]["winner"],
                "style_increment_status": summary["selection_decision"][
                    "style_increment_status"
                ],
                "valid_loaded_after_selection_lock": True,
                "valid_loaded_after_blind_scoring_lock": True,
                "historical_test_labels_read": False,
                "summary": common.relative(
                    common.resolve(policy["outputs"]["selection_summary"])
                ),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
