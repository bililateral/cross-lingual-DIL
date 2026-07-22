from __future__ import annotations

import copy
import hashlib
import inspect
import json
import math
import os
import re
import subprocess
import sys
import tempfile
import unittest
from collections import Counter
from pathlib import Path
from unittest import mock

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import step7_v3_common as common  # noqa: E402
import step7_v3_encode_clean_models as encode  # noqa: E402
import step7_v3_prepare_clean_data as prepare  # noqa: E402
import step7_v3_select_source_model as select  # noqa: E402


class Step7V3CleanSourceSelectionContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.policy = common.load_json(common.DEFAULT_POLICY)
        common.validate_policy(cls.policy)
        cls._model_fingerprints = None

    @classmethod
    def model_fingerprints(cls) -> dict:
        if cls._model_fingerprints is None:
            cfgs = {
                **cls.policy["embedding_models"],
                cls.policy["shared_reranker"]["model_key"]: cls.policy[
                    "shared_reranker"
                ],
            }
            cls._model_fingerprints = {
                key: common.validate_model_content_pin(key, cfg)
                for key, cfg in cfgs.items()
            }
        return cls._model_fingerprints

    @staticmethod
    def file_record(path: Path) -> dict:
        return {
            "path": str(path.relative_to(ROOT)).replace("\\", "/"),
            "size_bytes": path.stat().st_size,
            "sha256": common.sha256_file(path),
        }

    @staticmethod
    def uid_score(pair_uid: str, salt: str, lower: float, upper: float) -> float:
        digest = int.from_bytes(
            __import__("hashlib").sha256(f"{salt}|{pair_uid}".encode()).digest()[:8],
            "big",
        )
        unit = digest / float((1 << 64) - 1)
        return lower + (upper - lower) * unit

    @staticmethod
    def deterministic_unit_matrix(seller_uids: list[str], salt: str) -> np.ndarray:
        vectors = []
        for seller_uid in seller_uids:
            raw = np.frombuffer(
                hashlib.sha256(f"{salt}|{seller_uid}".encode()).digest(),
                dtype=np.uint8,
            )[:16].astype(np.float32)
            vector = raw - np.float32(127.5)
            vector /= np.linalg.norm(vector)
            vectors.append(vector)
        return np.asarray(vectors, dtype=np.float32)

    def materialize_label_free_mock_gpu_bundle(self, policy: dict, temp_root: Path) -> None:
        outputs = policy["outputs"]
        outputs["embedding_matrix_template"] = str(temp_root / "embeddings" / "{model_key}.npy")
        outputs["embedding_manifest_template"] = str(temp_root / "embeddings" / "{model_key}.json")
        outputs["embedding_pair_scores_template"] = str(temp_root / "scores" / "{model_key}.csv")
        outputs["reranker_pair_scores"] = str(temp_root / "scores" / "reranker.csv")
        outputs["reranker_manifest"] = str(temp_root / "scores" / "reranker.json")
        outputs["gpu_sync_manifest"] = str(temp_root / "gpu_sync.json")
        outputs["gpu_output_manifest"] = str(temp_root / "gpu_output.json")
        policy_contract = common.canonical_hash(policy)
        public_path = common.resolve(outputs["preparation_manifest"])
        encoder_path = ROOT / "scripts" / "step7_v3_encode_clean_models.py"
        sync_files = [
            self.file_record(common.DEFAULT_POLICY),
            self.file_record(ROOT / "scripts" / "step3_build_seller_profiles.py"),
            self.file_record(ROOT / "scripts" / "step7_v3_common.py"),
            self.file_record(ROOT / "scripts" / "step7_v3_build_sync_manifest.py"),
            self.file_record(encoder_path),
            self.file_record(
                ROOT / "scripts" / "run_step7_v3_clean_source_linux_20260722.sh"
            ),
            self.file_record(common.resolve(outputs["pair_manifest"])),
            self.file_record(common.resolve(outputs["clean_corpus"])),
            self.file_record(public_path),
        ]
        fingerprints = self.model_fingerprints()
        model_directories = {
            key: {"path": cfg["local_path"], **fingerprints[key]}
            for key, cfg in {
                **policy["embedding_models"],
                policy["shared_reranker"]["model_key"]: policy["shared_reranker"],
            }.items()
        }
        forbidden_paths = sorted(
            {spec["path"] for spec in policy["inputs"].values()}
            | {
                outputs["train_labels"],
                outputs["valid_labels"],
                outputs["historical_test_labels"],
                outputs["development_labels_manifest"],
                outputs["historical_test_labels_manifest"],
            }
        )
        sync = {
            "step": "step7_v3_label_free_windows_to_linux_gpu_sync",
            "version": policy["version"],
            "generator_script_path": "scripts/step7_v3_build_sync_manifest.py",
            "generator_script_sha256": common.sha256_file(
                ROOT / "scripts" / "step7_v3_build_sync_manifest.py"
            ),
            "policy_sha256": common.sha256_file(common.DEFAULT_POLICY),
            "policy_contract_sha256": policy_contract,
            "public_preparation_manifest_sha256": common.sha256_file(public_path),
            "file_count": len(sync_files),
            "total_file_bytes": sum(record["size_bytes"] for record in sync_files),
            "files": sync_files,
            "model_directories": model_directories,
            "label_files_included": False,
            "raw_source_files_included": False,
            "forbidden_workspace_paths": forbidden_paths,
        }
        sync_path = Path(outputs["gpu_sync_manifest"])
        common.write_json_immutable(sync_path, sync)
        corpus_rows = common.load_jsonl(common.resolve(outputs["clean_corpus"]))
        seller_uids = [row["seller_uid"] for row in corpus_rows]
        pair_rows = common.load_csv(common.resolve(outputs["pair_manifest"]))
        provenance = {
            "policy_sha256": sync["policy_sha256"],
            "policy_contract_sha256": policy_contract,
            "generator_script_path": "scripts/step7_v3_encode_clean_models.py",
            "generator_script_sha256": common.sha256_file(encoder_path),
            "gpu_sync_manifest_sha256": common.sha256_file(sync_path),
            "public_preparation_manifest_sha256": common.sha256_file(public_path),
        }
        for model_key, cfg in policy["embedding_models"].items():
            matrix_path = Path(
                outputs["embedding_matrix_template"].format(model_key=model_key)
            )
            score_path = Path(
                outputs["embedding_pair_scores_template"].format(model_key=model_key)
            )
            manifest_path = Path(
                outputs["embedding_manifest_template"].format(model_key=model_key)
            )
            matrix = self.deterministic_unit_matrix(seller_uids, model_key)
            seller_index = {
                seller_uid: index for index, seller_uid in enumerate(seller_uids)
            }
            common.write_npy_immutable(matrix_path, matrix)
            common.write_csv_immutable(
                score_path,
                [
                    {
                        "pair_uid": row["pair_uid"],
                        cfg["feature_name"]: f"{float(np.dot(matrix[seller_index[row['seller_uid_left']]], matrix[seller_index[row['seller_uid_right']]])):.12f}",
                    }
                    for row in pair_rows
                ],
            )
            layout = common.validate_sentence_transformer_layout(model_key, cfg)
            common.write_json_immutable(
                manifest_path,
                {
                    "step": "step7_v3_encode_clean_embedding",
                    "version": policy["version"],
                    "model_key": model_key,
                    "repo_id": cfg["repo_id"],
                    "local_path": cfg["local_path"],
                    "feature_name": cfg["feature_name"],
                    "pooling_contract": cfg["pooling_contract"],
                    "layout_validation": layout,
                    "model_fingerprint": fingerprints[model_key],
                    **provenance,
                    "feature_generation_reads_label_values": False,
                    "label_or_raw_source_files_present_in_gpu_workspace": False,
                    "text_prefix": cfg["text_prefix"],
                    "max_length": int(cfg["max_length"]),
                    "seller_uids": seller_uids,
                    "shape": list(matrix.shape),
                    "pair_count": len(pair_rows),
                    "maximum_unit_norm_error": float(
                        np.max(np.abs(np.linalg.norm(matrix, axis=1) - 1.0))
                    ),
                    "token_length_diagnostics": {
                        "row_count": len(seller_uids),
                        "max_length_contract": int(cfg["max_length"]),
                        "truncated_row_count": 0,
                        "truncated_row_fraction": 0.0,
                        "token_length_min": 1,
                        "token_length_median": 1.0,
                        "token_length_p90": 1.0,
                        "token_length_p95": 1.0,
                        "token_length_max": 1,
                    },
                    "clean_corpus_sha256": common.sha256_file(
                        common.resolve(outputs["clean_corpus"])
                    ),
                    "pair_manifest_sha256": common.sha256_file(
                        common.resolve(outputs["pair_manifest"])
                    ),
                    "embedding_matrix_sha256": common.sha256_file(matrix_path),
                    "pair_scores_sha256": common.sha256_file(score_path),
                    "device": "cuda",
                    "torch_version": "mock",
                    "transformers_version": "mock",
                    "sentence_transformers_version": "mock",
                },
            )
        reranker_cfg = policy["shared_reranker"]
        reranker_path = Path(outputs["reranker_pair_scores"])
        common.write_csv_immutable(
            reranker_path,
            [
                {
                    "pair_uid": row["pair_uid"],
                    reranker_cfg["feature_name"]: f"{self.uid_score(row['pair_uid'], 'reranker', 0.01, 0.99):.12f}",
                }
                for row in pair_rows
            ],
        )
        common.write_json_immutable(
            Path(outputs["reranker_manifest"]),
            {
                "step": "step7_v3_encode_clean_reranker",
                "version": policy["version"],
                "model_key": reranker_cfg["model_key"],
                "repo_id": reranker_cfg["repo_id"],
                "local_path": reranker_cfg["local_path"],
                "feature_name": reranker_cfg["feature_name"],
                "layout_validation": common.validate_reranker_layout(
                    reranker_cfg["model_key"], reranker_cfg
                ),
                "model_fingerprint": fingerprints[reranker_cfg["model_key"]],
                **provenance,
                "feature_generation_reads_label_values": False,
                "label_or_raw_source_files_present_in_gpu_workspace": False,
                "pair_symmetrization": reranker_cfg["pair_symmetrization"],
                "single_logit_transform": reranker_cfg["single_logit_transform"],
                "max_length": int(reranker_cfg["max_length"]),
                "clean_corpus_sha256": common.sha256_file(
                    common.resolve(outputs["clean_corpus"])
                ),
                "pair_manifest_sha256": common.sha256_file(
                    common.resolve(outputs["pair_manifest"])
                ),
                "pair_scores_sha256": common.sha256_file(reranker_path),
                "pair_count": len(pair_rows),
                "forward_reverse_mean_absolute_gap": 0.0,
                "device": "cuda",
                "token_length_diagnostics": {
                    "row_count": len(pair_rows),
                    "max_length_contract": int(reranker_cfg["max_length"]),
                    "truncated_row_count": 0,
                    "truncated_row_fraction": 0.0,
                    "token_length_min": 1,
                    "token_length_median": 1.0,
                    "token_length_p90": 1.0,
                    "token_length_p95": 1.0,
                    "token_length_max": 1,
                },
                "torch_version": "mock",
                "transformers_version": "mock",
            },
        )
        gpu_records = [
            self.file_record(common.resolve(path))
            for path in sorted(select.expected_gpu_output_paths(policy))
        ]
        common.write_json_immutable(
            Path(outputs["gpu_output_manifest"]),
            {
                "step": "step7_v3_label_free_gpu_output_bundle",
                "version": policy["version"],
                **provenance,
                "label_or_raw_source_files_present_in_gpu_workspace": False,
                "file_count": len(gpu_records),
                "total_file_bytes": sum(
                    record["size_bytes"] for record in gpu_records
                ),
                "files": gpu_records,
            },
        )

    def test_policy_fixes_current_734_pair_boundary(self) -> None:
        labels = prepare.eligible_label_rows(
            self.policy,
            common.load_csv(common.resolve(self.policy["inputs"]["frozen_labels"]["path"])),
        )
        counts = prepare.validate_supervision_counts(self.policy, labels)
        self.assertEqual(counts["train"], {"positive": 116, "negative": 285, "total": 401})
        self.assertEqual(counts["valid"], {"positive": 42, "negative": 110, "total": 152})
        self.assertEqual(counts["test"], {"positive": 51, "negative": 130, "total": 181})
        self.assertEqual(len(labels), 734)

    def test_prepared_feature_artifacts_have_no_labels_or_identity_columns(self) -> None:
        outputs = self.policy["outputs"]
        pair_rows = common.load_csv(common.resolve(outputs["pair_manifest"]))
        safe_rows = common.load_csv(common.resolve(outputs["safe_pair_features"]))
        corpus_rows = common.load_jsonl(common.resolve(outputs["clean_corpus"]))
        self.assertEqual(len(pair_rows), 734)
        self.assertEqual(len(safe_rows), 734)
        self.assertEqual(list(pair_rows[0]), [
            "pair_uid",
            "split_name",
            "component_id",
            "seller_uid_left",
            "seller_uid_right",
        ])
        self.assertEqual(list(safe_rows[0]), ["pair_uid", *common.SAFE_FEATURE_NAMES])
        self.assertEqual(
            list(corpus_rows[0]),
            ["seller_uid", "split_name", "model_text", "model_text_sha256"],
        )
        forbidden = set(self.policy["forbidden_m0_features"])
        self.assertFalse(forbidden & set(safe_rows[0]))
        self.assertNotIn("review_label", pair_rows[0])
        self.assertNotIn("review_label", corpus_rows[0])
        common.validate_public_pair_rows(self.policy, pair_rows)
        common.validate_clean_corpus_rows(corpus_rows)
        common.validate_safe_pair_feature_rows(safe_rows)
        invalid_safe = copy.deepcopy(safe_rows)
        invalid_safe[0]["same_market_bool"] = "2"
        with self.assertRaises(ValueError):
            common.validate_safe_pair_feature_rows(invalid_safe)

    def test_preparation_manifest_replays_current_outputs(self) -> None:
        outputs = self.policy["outputs"]
        manifest = common.load_json(common.resolve(outputs["preparation_manifest"]))
        self.assertFalse(manifest["feature_generation_uses_review_label_values"])
        self.assertFalse(manifest["feature_generation_uses_evidence_type_values"])
        self.assertEqual(manifest["pair_feature_roles"], self.policy["pair_feature_roles"])
        self.assertFalse(
            manifest["shortcut_features_eligible_for_model_training_or_selection"]
        )
        self.assertTrue(manifest["boundary_source_file_contains_review_label_column"])
        self.assertEqual(
            manifest["pair_universe_source"],
            "component_assignments_public_column_projection",
        )
        self.assertEqual(manifest["identity_residue_scan"]["status"], "pass")
        self.assertEqual(manifest["identity_residue_scan"]["total_residue_count"], 0)
        self.assertEqual(
            manifest["identity_residue_scan"]["claim_scope"],
            self.policy["clean_text_contract"]["identity_residue_claim_scope"],
        )
        self.assertFalse(
            manifest["identity_residue_scan"]["unknown_identifier_absence_proven"]
        )
        self.assertTrue(manifest["content_fidelity"]["quality_gates_passed"])
        self.assertGreaterEqual(
            manifest["content_fidelity"]["aggregate_character_retention"],
            self.policy["clean_text_contract"]["quality_gates"][
                "minimum_aggregate_character_retention"
            ],
        )
        self.assertEqual(manifest["content_fidelity"]["empty_text_fallback_count"], 0)
        self.assertEqual(
            manifest["identity_residue_scan"]["scan_scope"],
            "serialized_final_model_text_each_seller_row_independently",
        )
        self.assertEqual(manifest["split_isolation"]["cross_split_component_count"], 0)
        self.assertEqual(manifest["split_isolation"]["cross_split_seller_count"], 0)
        self.assertEqual(manifest["seller_count"], 855)
        self.assertEqual(
            manifest["signal_summary"]["contextual_alias_profile_scope"],
            "all_pinned_english_seller_profiles_plus_all_safe_pinned_identity_signal_literals_label_free_context_gated",
        )
        self.assertEqual(
            manifest["signal_summary"]["contextual_alias_profile_count"], 7522
        )
        self.assertEqual(
            manifest["signal_summary"]["global_identity_profile_scope"],
            "fixed_snapshot_union_of_all_pinned_english_profile_mixed_aliases_and_identity_signal_mixed_values_minus_preregistered_content_collisions_label_free",
        )
        self.assertEqual(
            manifest["signal_summary"]["global_identity_profile_count"], 7522
        )
        common.validate_global_identity_audit_manifest(self.policy, manifest)
        global_audit = manifest["signal_summary"][
            "global_identity_fixed_snapshot_audit"
        ]
        expected_global_audit = self.policy["clean_text_contract"][
            "global_mixed_alias_expected_audit"
        ]
        self.assertEqual(
            manifest["signal_summary"]["global_identity_token_count"],
            expected_global_audit["registry_token_count_after_denylist"],
        )
        self.assertEqual(
            global_audit["removed_distinct_token_count"],
            expected_global_audit["removed_distinct_token_count"],
        )
        self.assertEqual(
            global_audit["removed_occurrence_count"],
            expected_global_audit["removed_occurrence_count"],
        )
        self.assertEqual(
            common.canonical_hash(global_audit["removed_token_sha256_counts"]),
            expected_global_audit[
                "removed_token_sha256_counts_canonical_sha256"
            ],
        )
        self.assertEqual(
            global_audit["removed_occurrence_count"],
            manifest["redaction_summary"]["global_identifier_token_match_count"],
        )
        self.assertEqual(
            set(global_audit["content_collision_denylist"]),
            set(common.GLOBAL_IDENTITY_CONTENT_COLLISION_DENYLIST),
        )
        self.assertTrue(global_audit["input_hash_change_requires_full_reaudit"])
        phrase_audit = manifest["signal_summary"][
            "audited_global_identity_phrase_fixed_snapshot_audit"
        ]
        expected_phrase_audit = self.policy["clean_text_contract"][
            "audited_global_identity_phrase_expected_audit"
        ]
        self.assertEqual(
            manifest["signal_summary"][
                "audited_global_identity_phrase_token_count"
            ],
            expected_phrase_audit["registry_token_count"],
        )
        self.assertEqual(
            phrase_audit["removed_distinct_surface_count"],
            expected_phrase_audit["removed_distinct_surface_count"],
        )
        self.assertEqual(
            phrase_audit["matched_registry_token_count"],
            expected_phrase_audit["matched_registry_token_count"],
        )
        self.assertEqual(
            phrase_audit["registry_tokens_canonical_sha256"],
            expected_phrase_audit["registry_tokens_canonical_sha256"],
        )
        self.assertEqual(
            phrase_audit["removed_occurrence_count"],
            expected_phrase_audit["removed_occurrence_count"],
        )
        self.assertEqual(
            common.canonical_hash(phrase_audit["removed_surface_sha256_counts"]),
            expected_phrase_audit[
                "removed_surface_sha256_counts_canonical_sha256"
            ],
        )
        self.assertEqual(
            phrase_audit["removed_occurrence_count"],
            manifest["redaction_summary"][
                "audited_global_identity_phrase_match_count"
            ],
        )
        self.assertTrue(
            phrase_audit[
                "registry_is_disjoint_from_protected_content_collisions"
            ]
        )
        self.assertEqual(
            manifest["identity_residue_scan"][
                "audited_global_identity_phrase_residue_count"
            ],
            0,
        )
        full_census = manifest["signal_summary"][
            "full_known_alias_residual_fixed_snapshot_census"
        ]
        expected_full_census = self.policy["clean_text_contract"][
            "full_known_alias_residual_expected_audit"
        ]
        for key in (
            "registry_token_count",
            "scanned_seller_row_count",
            "scanned_segment_count",
            "matched_registry_token_count",
            "matched_occurrence_count",
            "matched_surface_count",
            "confirmed_identity_residual_anchor_count",
            "retained_ambiguous_or_content_collision_anchor_count",
            "retained_ambiguous_or_content_collision_occurrence_count",
            "match_kind_counts",
        ):
            self.assertEqual(full_census[key], expected_full_census[key], key)
        for map_key, hash_key in (
            (
                "matched_anchor_sha256_counts",
                "matched_anchor_sha256_counts_canonical_sha256",
            ),
            (
                "matched_surface_sha256_counts",
                "matched_surface_sha256_counts_canonical_sha256",
            ),
            (
                "matched_anchor_sha256_seller_counts",
                "matched_anchor_sha256_seller_counts_canonical_sha256",
            ),
        ):
            self.assertEqual(
                common.canonical_hash(full_census[map_key]),
                expected_full_census[hash_key],
                map_key,
            )
        embedded_census = manifest["signal_summary"][
            "audited_identity_embedded_residual_fixed_snapshot_census"
        ]
        expected_embedded_census = self.policy["clean_text_contract"][
            "audited_identity_embedded_residual_expected_audit"
        ]
        for key in (
            "registry_token_count",
            "scanned_seller_row_count",
            "matched_registry_token_count",
            "matched_occurrence_count",
            "matched_alias_surface_pair_count",
            "matched_surface_count",
            "confirmed_identity_residual_count",
            "retained_content_collision_occurrence_count",
        ):
            self.assertEqual(
                embedded_census[key], expected_embedded_census[key], key
            )
        for map_key, hash_key in (
            (
                "matched_anchor_sha256_counts",
                "matched_anchor_sha256_counts_canonical_sha256",
            ),
            (
                "matched_surface_sha256_counts",
                "matched_surface_sha256_counts_canonical_sha256",
            ),
            (
                "matched_alias_surface_pair_sha256_counts",
                "matched_alias_surface_pair_sha256_counts_canonical_sha256",
            ),
        ):
            self.assertEqual(
                common.canonical_hash(embedded_census[map_key]),
                expected_embedded_census[hash_key],
                map_key,
            )
        self.assertEqual(embedded_census["confirmed_identity_residual_count"], 0)
        tampered_manifest = copy.deepcopy(manifest)
        tampered_manifest["signal_summary"][
            "full_known_alias_residual_fixed_snapshot_census"
        ]["confirmed_identity_residual_anchor_count"] = 1
        with self.assertRaises(ValueError):
            common.validate_global_identity_audit_manifest(
                self.policy, tampered_manifest
            )
        tampered_manifest = copy.deepcopy(manifest)
        tampered_manifest["identity_residue_scan"]["claim_scope"] = (
            "stale_identity_claim"
        )
        with self.assertRaises(ValueError):
            common.validate_global_identity_audit_manifest(
                self.policy, tampered_manifest
            )
        self.assertGreater(
            manifest["signal_summary"][
                "contextual_alias_one_character_omission_count"
            ],
            0,
        )
        for term, record in manifest["content_fidelity"][
            "protected_identity_collision_term_retention"
        ].items():
            self.assertEqual(
                record["raw_count"],
                self.policy["clean_text_contract"]["quality_gates"][
                    "expected_protected_identity_collision_raw_counts"
                ][term],
                term,
            )
            self.assertGreaterEqual(
                record["retention"],
                self.policy["clean_text_contract"]["quality_gates"][
                    "minimum_protected_identity_collision_term_retention"
                ],
                term,
            )
        self.assertEqual(
            manifest["signal_summary"]["eligible_pair_seller_count"], 855
        )
        self.assertEqual(
            set(
                manifest["signal_summary"][
                    "contextual_alias_content_word_denylist"
                ]
            ),
            set(common.CONTEXTUAL_ALIAS_CONTENT_WORD_DENYLIST),
        )
        for record in manifest["output_files"].values():
            self.assertEqual(common.sha256_file(common.resolve(record["path"])), record["sha256"])

    def test_public_pair_and_feature_generation_cannot_consult_labels(self) -> None:
        assignment_rows = [
            {
                "pair_uid": "p1",
                "split_name": "train",
                "seller_uid_left": "a",
                "seller_uid_right": "b",
                "recomputed_component_id": "c1",
            },
            {
                "pair_uid": "p2",
                "split_name": "train",
                "seller_uid_left": "c",
                "seller_uid_right": "d",
                "recomputed_component_id": "c2",
            },
        ]
        original_pairs = prepare.build_pair_manifest(
            assignment_rows, "recomputed_component_id"
        )
        projected_positive = prepare.eligible_assignment_rows(
            {
                **self.policy,
                "supervision_boundary": {
                    **self.policy["supervision_boundary"],
                    "expected_counts": {
                        **self.policy["supervision_boundary"]["expected_counts"],
                        "train": {"positive": 1, "negative": 1, "total": 2},
                        "valid": {"positive": 0, "negative": 0, "total": 0},
                        "test": {"positive": 0, "negative": 0, "total": 0},
                        "total": 2,
                    },
                },
            },
            [
                {**row, "dataset": "en_content_train_pool", "review_label": "positive"}
                for row in assignment_rows
            ],
        )
        projected_negative = prepare.eligible_assignment_rows(
            {
                **self.policy,
                "supervision_boundary": {
                    **self.policy["supervision_boundary"],
                    "expected_counts": {
                        **self.policy["supervision_boundary"]["expected_counts"],
                        "train": {"positive": 1, "negative": 1, "total": 2},
                        "valid": {"positive": 0, "negative": 0, "total": 0},
                        "test": {"positive": 0, "negative": 0, "total": 0},
                        "total": 2,
                    },
                },
            },
            [
                {**row, "dataset": "en_content_train_pool", "review_label": "negative"}
                for row in assignment_rows
            ],
        )
        self.assertEqual(projected_positive, projected_negative)
        self.assertNotIn("review_label", projected_positive[0])
        records = {
            uid: {
                "seller_uid": uid,
                "clean_categories": ["cat"],
                "clean_titles": [f"title {uid}"],
                "clean_descriptions": ["shared" if uid in {"a", "b"} else uid],
                "source_dataset": "dataset",
                "source_market_raw": "market",
                "numeric_profile": {
                    name: float(index + 1)
                    for index, name in enumerate(common.NUMERIC_PROFILE_FIELDS)
                },
            }
            for uid in ("a", "b", "c", "d")
        }
        reference = common.train_reference(records, set(records))
        self.assertEqual(
            common.build_safe_pair_rows(original_pairs, records, reference),
            common.build_safe_pair_rows(copy.deepcopy(original_pairs), records, reference),
        )
        source = inspect.getsource(prepare.prepare_public)
        self.assertNotIn("frozen_labels", source)
        self.assertNotIn("evidence_labels", source)
        self.assertNotIn("review_label", source)

    def test_obfuscated_contacts_and_alias_suffixes_are_removed_from_model_text(self) -> None:
        examples = (
            "w.i.c.k.r drugstores1",
            "TELEGRAM:...................@limestone420",
            "K.I.K:.............blazeupp5",
            "WATSAPP/TEXT +1 646 535 9092",
            "W*ckr...newusa",
            "W*ckr...Supperfast001",
            "W*ckr...Drugsking001",
            "contact us on at express74",
            "Wicke: doli2",
            "Snapchat: psych_trip247",
            "snapchat::: vibzv2019",
            "Snapchat : m_valote2019",
            "WELCOME TO truu00shop",
            "WELCOME TO CITYSHOP92",
        )
        for value in examples:
            cleaned, diagnostics = common.redact_identifiers(value, [])
            self.assertEqual(cleaned, "")
            self.assertGreater(diagnostics["generic_identifier_match_count"], 0)
        for value, forbidden in (
            ("/ID>>>>>>> usaplug33", "usaplug33"),
            ("contact us at nevadakay3gmailc", "nevadakay3gmailc"),
            ("id=1234567890abcdef1234567890", "1234567890abcdef1234567890"),
        ):
            cleaned, diagnostics = common.redact_identifiers(value, [])
            self.assertNotIn(forbidden, cleaned.casefold())
            self.assertGreater(diagnostics["generic_identifier_match_count"], 0)
        for ordinary in (
            "HACK FB, SNAPCHAT, IG accounts",
            "Snapchat porn archive",
            "gmail accounts tutorial",
            "WELCOME BONUS 30 capsules",
            "contact support within 24 hours",
        ):
            cleaned, _ = common.redact_identifiers(ordinary, [])
            self.assertEqual(cleaned, ordinary)
        for value in (
            "ADD ME ON WICKRl",
            "Wick ID: newsale",
            "WiWickr...",
            "-----BEGIN PGP PUBLIC KEY BLOCK----- Version: GnuPG truncated",
            "text.... Email.....fastbusiness453@gmail.c",
            "0.5mg pills 500EUR goblinking00000gmail.c",
            "*PROTONMAIL.... @p",
            "*Protonmail // @pro",
            "For more info text or call 3238635014",
            "Mobile: 8049869967",
            "call/text.......+1 669 228 0192",
            "Texts or call me 323..407..75..67",
            "SSN: 230517723",
            "7 FOR FAST COMMUNICATION XXX+1.504.407.1267XXX",
            "contact us via text....+153049",
            "whtspp... , .. cyrillebrono Telegr..jamescoo.",
            "TOTAL-SATISFACTION 100% wick id..... 100 Tabs LSD",
            "WEARY BUYERS INBOX SUPPORT STAFF DIMITRI (SUPPORT STAFF)",
            "SHIPPING WORLDWIDE James || next listing",
        ):
            cleaned, diagnostics = common.redact_identifiers(value, [])
            self.assertNotRegex(
                cleaned.casefold(),
                r"wickr|pgp public key|gmail\.c|@p(?:ro)?\b|3238635014|"
                r"8049869967|669 228 0192|323\.\.407|230517723",
            )
            self.assertGreater(diagnostics["generic_identifier_match_count"], 0)
        for protected_content in (
            "DMT is also called Dimitri or the Spirit Molecule",
            "Delivery format: John Wick 72 Garden Close",
            "James Bone - Cognitive Hack",
            "price +12345",
            "Mexican cartel crystal meth",
            "Key Lime Pie and lime green buds",
            "We are building an empire",
            "Blue Dream and Grey Hat",
        ):
            cleaned_content, _ = common.redact_identifiers(protected_content, [])
            self.assertEqual(cleaned_content, protected_content)
        for market_context in (
            "Best selling Xanax on Empire and Grey",
            "We were on Dream and had many reviews",
            "feedback sur dream et wallstreet",
            "grey, samsara, verified vendor",
        ):
            cleaned_market, diagnostics = common.redact_identifiers(
                market_context,
                [],
                audited_global_phrase_tokens=(
                    common.AUDITED_GLOBAL_IDENTITY_PHRASE_TOKENS
                ),
            )
            self.assertNotRegex(
                cleaned_market.casefold(),
                r"\b(?:empire|grey|dream|wallstreet|samsara)\b",
                market_context,
            )
            self.assertGreater(
                diagnostics["generic_identifier_match_count"]
                + diagnostics["audited_global_identity_phrase_match_count"],
                0,
            )
        decimal_text = "price 0.1905617980645162 BTC"
        cleaned_decimal, _ = common.redact_identifiers(decimal_text, [])
        self.assertEqual(cleaned_decimal, decimal_text)
        cleaned_literal, _ = common.redact_identifiers(
            "mastercard accepts master", ["master"]
        )
        self.assertEqual(cleaned_literal, "mastercard accepts")
        common.assert_no_identifier_residue(
            "mastercard accepts", ["master"], "boundary-test"
        )
        self.assertIn("LILpump305", common.seller_alias_variants("LILpump305 (100%)"))
        corpus = common.load_jsonl(common.resolve(self.policy["outputs"]["clean_corpus"]))
        joined = "\n".join(row["model_text"].casefold() for row in corpus)
        for residue in (
            "limestone420",
            "drugstores1",
            "legitmed007",
            "lilpump305",
            "psych_trip247",
            "vibzv2019",
            "m_valote2019",
            "nevadakay3gmailc",
            "truu00shop",
            "chrisstore",
            "carlosjuan",
            "grandmastershop",
            "kaiserplug",
            "drfraud",
            "medicalfamer",
            "everreadyshop",
            "kaiserplug",
            "davidesales",
            "lizardpro",
            "benzoneil",
            "newsales",
            "maker24-sales",
            "usapitcher",
        ):
            self.assertNotIn(residue, joined)
        self.assertNotRegex(
            joined,
            r"(?<![a-z0-9])(?<!\d\.)\d{7,19}(?![a-z0-9])(?!\.\d)",
        )
        self.assertNotRegex(
            joined,
            r"(?i)(?:call|text|phone|mobile|contact)[^|\n]{0,48}?"
            r"\+?\d(?:[\s().-]*\d){5,15}(?![a-z0-9])",
        )
        with self.assertRaises(ValueError):
            common.scan_final_corpus_identity_residues(
                [
                    {
                        "seller_uid": "audit-only",
                        "model_text": "please call 323 407 7567",
                    }
                ],
                {"audit-only": []},
                set(),
            )
        for residual in (
            "W*ckr...newusa",
            "Wick ID: newsale",
            "Snapchat: psych_trip247",
            "WELCOME TO CITYSHOP92",
            "/ID>>>>>>> usaplug33",
            "contact us at nevadakay3gmailc",
            "id=1234567890abcdef1234567890",
        ):
            with self.subTest(final_audit_residual=residual), self.assertRaises(
                ValueError
            ):
                common.scan_final_corpus_identity_residues(
                    [{"seller_uid": "audit-only", "model_text": residual}],
                    {"audit-only": []},
                    set(),
                )
        for separated_fields in (
            "REACH US VIA ---- || 200mg",
            "message me. || 2fdck",
            "contact us Via || 2fdck",
            "CONTACT US ON || 100x",
        ):
            audit = common.scan_final_corpus_identity_residues(
                [{"seller_uid": "audit-only", "model_text": separated_fields}],
                {"audit-only": []},
                set(),
            )
            self.assertEqual(audit["status"], "pass", separated_fields)

    def test_global_handle_registry_does_not_delete_ordinary_content_words(self) -> None:
        ordinary_words = (
            "premium",
            "private",
            "master",
            "pfizer",
            "however",
            "opportunity",
            "fedex",
        )
        profiles = [
            {"source_seller_raw": word, "alias_normalized": word}
            for word in ordinary_words
        ]
        registry = common.global_identity_tokens({}, profiles)
        self.assertFalse(set(ordinary_words) & registry)
        sentence = " ".join(ordinary_words)
        cleaned, diagnostics = common.redact_identifiers(sentence, [], registry)
        self.assertEqual(cleaned, sentence)
        self.assertEqual(diagnostics["global_identifier_token_match_count"], 0)

        collision_profiles = [
            {"source_seller_raw": value, "alias_normalized": value}
            for value in ("250mg", "25I-NBOMe", "BET365")
        ]
        collision_signals = {"seller": ["250mg", "25I-NBOMe", "BET365"]}
        contextual_collisions = common.contextual_global_alias_tokens(
            collision_profiles, collision_signals
        )
        collision_fuzzy = common.contextual_alias_deletion_tokens(
            contextual_collisions
        )
        local_aliases = common.seller_identity_literals(
            {
                "source_seller_raw": "actualseller",
                "alias_normalized": "actualseller",
            }
        )
        self.assertEqual(local_aliases, ["actualseller"])
        self.assertNotIn("250mg", local_aliases)
        for ordinary in ("250mg tablets", "25I-NBOMe blotters", "BET365 account"):
            self.assertEqual(
                common.redact_identifiers(
                    ordinary,
                    [],
                    set(),
                    contextual_collisions,
                    collision_fuzzy,
                )[0],
                ordinary,
            )
        for contact_occurrence in (
            "Wickr: 250mg",
            "Wickr: 25I-NBOMe",
            "contact us at BET365",
        ):
            self.assertEqual(
                common.redact_identifiers(
                    contact_occurrence,
                    [],
                    set(),
                    contextual_collisions,
                    collision_fuzzy,
                )[0],
                "",
            )

        audited_registry = set(common.AUDITED_GLOBAL_IDENTITY_PHRASE_TOKENS) | {
            "maker24"
        }
        audited_identity_examples = (
            "Contact us today on kaiserplugs",
            "davidesales..treat anxiety",
            "Let your dreams come true with Lizardpro Hackers",
            "Buy online from benzoneil Drug Store",
            "FOR FAST RESPONSE newsales",
            "MAKER24-SALES IPHONE 12 PRO MAX",
            "Wickr:USAPITCHER",
        )
        for identity_text in audited_identity_examples:
            cleaned, diagnostics = common.redact_identifiers(
                identity_text,
                [],
                audited_global_phrase_tokens=audited_registry,
            )
            self.assertNotRegex(
                cleaned.casefold(),
                r"kaiserplug|davidesales|lizardpro|benzoneil|newsales?|maker24|usapitcher",
                identity_text,
            )
            self.assertGreater(
                diagnostics["audited_global_identity_phrase_match_count"]
                + diagnostics["generic_identifier_match_count"],
                0,
                identity_text,
            )
            with self.assertRaises(ValueError):
                common.scan_final_corpus_identity_residues(
                    [{"seller_uid": "audit-only", "model_text": identity_text}],
                    {"audit-only": []},
                    set(),
                    audited_global_phrase_tokens=audited_registry,
                )

        compact_local_aliases = common.seller_identity_literals(
            {
                "source_seller_raw": "USA_PITCHER",
                "alias_normalized": "usa_pitcher",
            }
        )
        self.assertIn("usapitcher", compact_local_aliases)
        self.assertEqual(
            common.redact_identifiers(
                "USAPITCHER offers products", compact_local_aliases
            )[0],
            "offers products",
        )

        suffixed, suffixed_diagnostics = common.redact_identifiers(
            "truu00SHOP brings products", [], {"truu00"}
        )
        self.assertEqual(suffixed, "brings products")
        self.assertEqual(
            suffixed_diagnostics["global_identifier_token_match_count"], 1
        )
        self.assertEqual(
            common.redact_identifiers(
                "vitaminshop brings products", [], {"truu00"}
            )[0],
            "vitaminshop brings products",
        )
        with self.assertRaises(ValueError):
            common.scan_final_corpus_identity_residues(
                [{"seller_uid": "other", "model_text": "truu00SHOP offer"}],
                {"other": []},
                {"truu00"},
            )

        outputs = self.policy["outputs"]
        corpus = common.load_jsonl(common.resolve(outputs["clean_corpus"]))
        corpus_by_uid = {row["seller_uid"]: row["model_text"] for row in corpus}
        profiles_by_uid = {
            row["seller_uid"]: row
            for row in common.load_jsonl(
                common.resolve(self.policy["inputs"]["seller_profiles"]["path"])
            )
            if row["seller_uid"] in corpus_by_uid
        }
        fields = self.policy["clean_text_contract"]["fields_in_order"]
        raw_text = "\n".join(
            str(profile.get(field, ""))
            for profile in profiles_by_uid.values()
            for field in fields
        )
        clean_text = "\n".join(corpus_by_uid.values())
        for word in ordinary_words:
            pattern = re.compile(rf"(?i)(?<![a-z0-9]){re.escape(word)}(?![a-z0-9])")
            raw_count = len(pattern.findall(raw_text))
            clean_count = len(pattern.findall(clean_text))
            self.assertGreater(raw_count, 0, word)
            self.assertGreaterEqual(clean_count / raw_count, 0.80, word)

    def test_audited_fixed_snapshot_seller_and_market_identities_are_absent(
        self,
    ) -> None:
        registry = common.AUDITED_GLOBAL_IDENTITY_PHRASE_TOKENS
        challenging_examples = (
            "Telegram me at @ahemweedshop",
            "Buy online from benzoneil Drug Store",
            "Email: biohaz",
            "Email: biohazar",
            "Skype: bio.teamz",
            "WELCOME TO blue heavens",
            "Terms and conditions of charlieUK",
            "Darkkings have successful delivery",
            "davidesales offers products",
            "We the Drugskingdom welcome you",
            "WICKR ME APP drugusa7",
            "Welcome to FLEX COMPANY SHOP",
            "My name is flexowned",
            "Contact Foreigshop before ordering",
            "My name Full plez",
            "Telegram: Dark Hackman",
            "ICQ: johnston",
            "Contact us on kaiserplugs",
            "I LEGAL DRUG SHOP sell products",
            "Order from LEGIT CONNECT",
            "Lizard pro Hackers",
            "Welcome to maxprisc",
            "Mr.CodeZ Support",
            "Mr.Dank Support",
            "Wick ID: newsales",
            "We are the genuine Original Ger Meds",
            "Perfect sales is committed",
            "Available with Prestige Vendor",
            "Order from Solution Center",
            "TOP MONEY MAKER services",
            "Buy from Universal Vendor",
            "Welcome to USA DRUG LORD",
            "Wickr: USA_PITCHER",
            "Wickr: vonadolf",
            "Email: xenite",
            "My name is Youngmoney",
            "Welcome to your_service shop",
            "WESERUNIONSTORE Vendor AND GETS EXTRA BONUS",
        )
        self.assertEqual(
            self.policy["clean_text_contract"][
                "audited_global_identity_phrase_tokens"
            ],
            sorted(registry),
        )
        for term in common.PROTECTED_IDENTITY_COLLISION_TERMS:
            self.assertIsNone(
                common.anchored_alias_registry_token(term, registry), term
            )
        self.assertEqual(
            self.policy["clean_text_contract"][
                "audited_global_identity_phrase_dot_separator_tokens"
            ],
            sorted(common.AUDITED_GLOBAL_IDENTITY_DOT_SEPARATOR_TOKENS),
        )
        for expected_alias in sorted(registry):
            identity_text = f"Welcome to {expected_alias}"
            with self.subTest(
                expected_alias=expected_alias, identity_text=identity_text
            ):
                raw_spans = common.unconditional_alias_spans(
                    identity_text, registry
                )
                observed_anchors = {
                    common.anchored_alias_registry_token(
                        identity_text[start:end], registry
                    )
                    for start, end in raw_spans
                }
                self.assertIn(expected_alias, observed_anchors)
                cleaned, diagnostics = common.redact_identifiers(
                    identity_text,
                    [],
                    audited_global_phrase_tokens=registry,
                )
                self.assertGreater(
                    diagnostics[
                        "audited_global_identity_phrase_match_count"
                    ],
                    0,
                )
                self.assertFalse(
                    common.unconditional_alias_spans(cleaned, registry)
                )
                with self.assertRaises(ValueError):
                    common.scan_final_corpus_identity_residues(
                        [
                            {
                                "seller_uid": "audit-only",
                                "model_text": identity_text,
                            }
                        ],
                        {"audit-only": []},
                        audited_global_phrase_tokens=registry,
                    )
        for identity_text in challenging_examples:
            cleaned, diagnostics = common.redact_identifiers(
                identity_text,
                [],
                audited_global_phrase_tokens=registry,
            )
            self.assertGreater(
                diagnostics["audited_global_identity_phrase_match_count"]
                + diagnostics["generic_identifier_match_count"],
                0,
                identity_text,
            )
            self.assertFalse(
                common.unconditional_alias_spans(cleaned, registry),
                identity_text,
            )
        dotted_non_handle = "Available with Prestige. Vendor"
        self.assertEqual(
            common.redact_identifiers(
                dotted_non_handle,
                [],
                audited_global_phrase_tokens=registry,
            )[0],
            dotted_non_handle,
        )

    def test_collision_prone_identity_spellings_use_exact_boundaries(self) -> None:
        clean_cfg = self.policy["clean_text_contract"]
        registry = common.AUDITED_GLOBAL_IDENTITY_PHRASE_TOKENS
        exact_tokens = common.EXACT_CONTIGUOUS_IDENTITY_TOKENS
        collision_denylist = (
            common.SEPARATOR_INVARIANT_IDENTITY_CONTENT_COLLISION_DENYLIST
        )
        self.assertEqual(exact_tokens, frozenset({"darkmarket", "unionstore"}))
        self.assertEqual(
            set(clean_cfg["exact_contiguous_identity_tokens"]),
            set(exact_tokens),
        )
        self.assertEqual(
            set(
                clean_cfg[
                    "separator_invariant_identity_content_collision_denylist"
                ]
            ),
            set(collision_denylist),
        )
        self.assertFalse(
            exact_tokens & common.AUDITED_GLOBAL_IDENTITY_PHRASE_TOKENS
        )

        collision_profiles = [
            {
                "source_seller_raw": token,
                "alias_normalized": token,
            }
            for token in sorted(exact_tokens)
        ]
        contextual_registry = common.contextual_global_alias_tokens(
            collision_profiles
        )
        self.assertFalse(exact_tokens & contextual_registry)
        for profile in collision_profiles:
            self.assertFalse(
                exact_tokens & common.seller_identity_phrase_tokens(profile)
            )

        local_literals = sorted(exact_tokens)
        ordinary_content = (
            "highest quality ketamine available on the dark market",
            "keywords include cashapp fraud dark market cc",
            "the best dark-market store online",
            "security tools sold on darkmarkets",
            "Pick Up Funds From Any Western union Store or Bank Agent",
        )
        for value in ordinary_content:
            cleaned, _ = common.redact_identifiers(
                value,
                local_literals,
                contextual_alias_tokens=contextual_registry,
                seller_local_phrase_tokens=set(),
                audited_global_phrase_tokens=(
                    common.AUDITED_GLOBAL_IDENTITY_PHRASE_TOKENS
                ),
            )
            self.assertEqual(cleaned, value)

        for identity_text, forbidden in (
            ("DARKmarket special", "darkmarket"),
            ("Account unionstore offers products", "unionstore"),
            ("Dark Market: Versus: named vendor", "dark market"),
        ):
            cleaned, diagnostics = common.redact_identifiers(
                identity_text,
                [],
                audited_global_phrase_tokens=(
                    common.AUDITED_GLOBAL_IDENTITY_PHRASE_TOKENS
                ),
            )
            self.assertNotIn(forbidden, cleaned.casefold())
            self.assertGreater(diagnostics["generic_identifier_match_count"], 0)

        with self.assertRaises(ValueError):
            common.scan_final_corpus_identity_residues(
                [{"seller_uid": "audit-only", "model_text": "unionstore"}],
                {"audit-only": []},
            )

        # A first-pass local-alias removal must not expose an audited identity
        # that starts after a hyphen and survives later redaction passes.
        exposed_identity, exposed_diagnostics = common.redact_identifiers(
            "WITH HI-TECH PROGRAMMERS HACKERS DECIDE YOUR FUTURE",
            ["PROGRAMMERS"],
            audited_global_phrase_tokens=registry,
        )
        self.assertFalse(common.unconditional_alias_spans(exposed_identity, registry))
        self.assertGreater(
            exposed_diagnostics["seller_local_alias_match_count"], 0
        )
        self.assertGreater(
            exposed_diagnostics["audited_global_identity_phrase_match_count"],
            0,
        )
        self.assertGreaterEqual(exposed_diagnostics["redaction_pass_count"], 2)

        for profile_alias, display_alias in (
            ("PrestigeVendor", "Prestige Vendor offers products"),
            ("USA_PITCHER", "USAPITCHER offers products"),
            ("fullplez", "My name Full plez"),
        ):
            local_registry = common.seller_identity_phrase_tokens(
                {
                    "source_seller_raw": profile_alias,
                    "alias_normalized": profile_alias,
                }
            )
            cleaned, diagnostics = common.redact_identifiers(
                display_alias,
                [],
                seller_local_phrase_tokens=local_registry,
            )
            self.assertFalse(
                common.unconditional_alias_spans(cleaned, local_registry)
            )
            self.assertGreater(
                diagnostics["seller_local_alias_phrase_match_count"], 0
            )

        # Deliberately independent of unconditional_alias_spans: NFKC/casefold,
        # character-by-character separator tolerance, explicit boundaries, and
        # registry-anchored plural/identity suffixes.  This prevents the
        # redactor and the publication audit from validating each other through
        # the same implementation bug.
        dot_confirmed_handles = (
            common.AUDITED_GLOBAL_IDENTITY_DOT_SEPARATOR_TOKENS
        )

        def independent_pattern(alias: str) -> re.Pattern[str]:
            separator = (
                r"[ \t._-]*"
                if alias in dot_confirmed_handles
                else r"[ \t_-]*"
            )
            body = separator.join(re.escape(character) for character in alias)
            suffixes = ("s", *common.IDENTITY_HANDLE_SUFFIXES)
            suffix_body = "|".join(
                separator.join(re.escape(character) for character in suffix)
                for suffix in suffixes
            )
            return re.compile(
                rf"(?<![a-z0-9]){body}(?:(?:{suffix_body}))?(?![a-z0-9])",
                flags=re.IGNORECASE,
            )

        patterns = {
            alias: independent_pattern(alias) for alias in sorted(registry)
        }
        corpus = common.load_jsonl(
            common.resolve(self.policy["outputs"]["clean_corpus"])
        )
        residues = []
        for row in corpus:
            normalized = __import__("unicodedata").normalize(
                "NFKC", row["model_text"]
            ).casefold()
            for alias, pattern in patterns.items():
                if pattern.search(normalized):
                    residues.append(
                        (common.sha256_text(row["seller_uid"])[:16], alias)
                    )
        self.assertEqual(residues, [])

    def test_pure_alpha_seller_aliases_are_removed_only_in_identity_context(self) -> None:
        profiles = [
            {"source_seller_raw": "CITYSHOP", "alias_normalized": "cityshop"},
            {"source_seller_raw": "CARLOSJUAN", "alias_normalized": "carlosjuan"},
            {"source_seller_raw": "premium", "alias_normalized": "premium"},
            {"source_seller_raw": "quality", "alias_normalized": "quality"},
            {"source_seller_raw": "support", "alias_normalized": "support"},
            {"source_seller_raw": "microsoft", "alias_normalized": "microsoft"},
            {"source_seller_raw": "applestore", "alias_normalized": "applestore"},
            {"source_seller_raw": "chrisstore20", "alias_normalized": "chrisstore20"},
            {"source_seller_raw": "CARLOSJUAN", "alias_normalized": "carlosjuan"},
            {"source_seller_raw": "grandmaster", "alias_normalized": "grandmaster"},
            {"source_seller_raw": "kaiserplug", "alias_normalized": "kaiserplug"},
            {"source_seller_raw": "DrFRAUD51", "alias_normalized": "drfraud51"},
            {"source_seller_raw": "Daha2020", "alias_normalized": "daha2020"},
            {"source_seller_raw": "medicalfarmer", "alias_normalized": "medicalfarmer"},
            {"source_seller_raw": "everready", "alias_normalized": "everready"},
            {"source_seller_raw": "vendor_men", "alias_normalized": "vendor_men"},
            {"source_seller_raw": "DHLExpress (100%)", "alias_normalized": "dhlexpress"},
            {"source_seller_raw": "davidesales", "alias_normalized": "davidesales"},
            {"source_seller_raw": "FOREIGNER", "alias_normalized": "foreigner"},
            {"source_seller_raw": "Lizardpro", "alias_normalized": "lizardpro"},
            {"source_seller_raw": "USA_PITCHER", "alias_normalized": "usa_pitcher"},
            {"source_seller_raw": "ExpressScripts", "alias_normalized": "expressscripts"},
            {"source_seller_raw": "Rodrigomendez", "alias_normalized": "rodrigomendez"},
            {"source_seller_raw": "drugcenter", "alias_normalized": "drugcenter"},
            {"source_seller_raw": "walgreens", "alias_normalized": "walgreens"},
            {"source_seller_raw": "benzoneil", "alias_normalized": "benzoneil"},
            {"source_seller_raw": "smackers", "alias_normalized": "smackers"},
            {"source_seller_raw": "amazonprimestore (98%)", "alias_normalized": "amazonprimestore"},
        ]
        registry = common.contextual_global_alias_tokens(profiles)
        deletion_registry = common.contextual_alias_deletion_tokens(registry)
        self.assertIn("cityshop", registry)
        self.assertIn("carlosjuan", registry)
        self.assertNotIn("premium", registry)
        self.assertNotIn("private", registry)
        self.assertNotIn("quality", registry)
        self.assertNotIn("support", registry)
        self.assertNotIn("microsoft", registry)
        self.assertNotIn("applestore", registry)
        for derived in ("chrisstore", "drfraud", "daha"):
            self.assertIn(derived, registry)
        self.assertIn("medicalfamer", deletion_registry)

        cleaned, diagnostics = common.redact_identifiers(
            "WELCOME TO CITYSHOP. Welcome to CARLOSJUAN. premium products",
            [],
            set(),
            registry,
        )
        self.assertNotRegex(cleaned.casefold(), r"cityshop|carlosjuan")
        self.assertIn("premium products", cleaned)
        self.assertEqual(diagnostics["contextual_alias_match_count"], 2)

        identity_examples = (
            "WELCOME VALUED CLIENTS TO chrisstore WHERE",
            "Welcome to CARLOSJUAN.We sell",
            "WELCOME VALUED CLIENTS TO GRANDMASTERSHOP WHERE",
            "Welcome to kaiserplug.We sell",
            "Welcome to DrFRAUD shop. The store",
            "WELCOME TO DAHA DRUG STORE",
            "WELCOME TO medicalfamer, WHERE",
            "WELCOME VALUED CLIENTS TO EVERREADYSHOP WHERE",
            "HERE AT VENDOR MEN OUR CUSTOMER SAFETY",
            "WELCOME TO DHL EXPRESS - NEW VENDOR",
            "davidesales offer quality",
            "FOREIGNER OFFERS goods",
            "Lizardpro, offer goods",
            "USA_PITCHER OFFER goods",
            "ExpressScripts bring goods",
            "Rodrigomendez offer goods",
            "drugcenter brings goods",
            "walgreens offer goods",
            "benzoneil offers goods",
        )
        for identity_text in identity_examples:
            cleaned_identity, identity_diagnostics = common.redact_identifiers(
                identity_text,
                [],
                set(),
                registry,
                deletion_registry,
            )
            self.assertGreater(
                identity_diagnostics["contextual_alias_match_count"],
                0,
                identity_text,
            )
            self.assertFalse(
                common.contextual_alias_spans(
                    cleaned_identity, registry, deletion_registry
                ),
                identity_text,
            )
            with self.assertRaises(ValueError):
                common.scan_final_corpus_identity_residues(
                    [{"seller_uid": "audit-only", "model_text": identity_text}],
                    {"audit-only": []},
                    set(),
                    registry,
                    deletion_registry,
                )

        for ordinary_text in (
            "All orders are sent via DHL express delivery",
            "smackers - ships free",
            "Walgreens pharmacy products",
            "WELCOME BONUS 30 capsules",
            "Amazon Prime Store Card",
        ):
            self.assertEqual(
                common.redact_identifiers(
                    ordinary_text, [], set(), registry, deletion_registry
                )[0],
                ordinary_text,
            )

        boundary_cleaned, boundary_diagnostics = common.redact_identifiers(
            "Amnesia Coffeeshop Quality. AppleStore Yahoo.",
            [],
            set(),
            {"quality", "yahoo"},
        )
        self.assertEqual(
            boundary_cleaned,
            "Amnesia Coffeeshop Quality. AppleStore Yahoo.",
        )
        self.assertEqual(boundary_diagnostics["contextual_alias_match_count"], 0)
        self.assertEqual(
            [
                match.group(0)
                for match in common.IDENTIFIER_TOKEN_RE.finditer(
                    "CITYSHOP92...FOR"
                )
            ],
            ["CITYSHOP92", "FOR"],
        )

        with self.assertRaises(ValueError):
            common.scan_final_corpus_identity_residues(
                [{"seller_uid": "audit-only", "model_text": "WELCOME TO CITYSHOP"}],
                {"audit-only": []},
                set(),
                registry,
                deletion_registry,
            )

    def test_all_safe_pure_signal_aliases_enter_only_the_contextual_registry(self) -> None:
        profiles = [
            {
                "source_seller_raw": "ordinaryseller",
                "alias_normalized": "ordinaryseller",
            }
        ]
        signals = {"seller": ["OCTAPUSTICKETS"]}
        contextual = common.contextual_global_alias_tokens(profiles, signals)
        unconditional = common.global_identity_tokens(signals, profiles)
        self.assertIn("octapustickets", contextual)
        self.assertNotIn("octapustickets", unconditional)
        self.assertEqual(
            common.redact_identifiers(
                "Welcome to OCTAPUSTICKETS",
                [],
                unconditional,
                contextual,
            )[0],
            "",
        )
        ordinary = "octapustickets data export"
        self.assertEqual(
            common.redact_identifiers(
                ordinary,
                [],
                unconditional,
                contextual,
            )[0],
            ordinary,
        )

    def test_full_known_alias_census_is_boundary_safe_and_longest_first(self) -> None:
        census = common.full_known_alias_residual_census(
            [
                {
                    "seller_uid": "seller-a",
                    "model_text": (
                        "alpha beta || alpha-beta\n"
                        "alpha.beta makers makerpharma || alpha. beta"
                    ),
                },
                {
                    "seller_uid": "seller-b",
                    "model_text": "alpha || beta\nalpha.\nbeta",
                },
            ],
            {"alphabeta", "maker"},
        )
        self.assertEqual(census["scanned_seller_row_count"], 2)
        self.assertEqual(census["matched_registry_token_count"], 2)
        self.assertEqual(census["matched_occurrence_count"], 5)
        self.assertEqual(
            census["match_kind_counts"],
            {
                "exact": 3,
                "known_alias_plural": 1,
                "known_alias_plus_pharma": 1,
            },
        )
        self.assertEqual(
            census["matched_anchor_sha256_counts"][
                common.sha256_text("alphabeta")
            ],
            3,
        )
        self.assertEqual(
            census["matched_anchor_sha256_counts"][common.sha256_text("maker")],
            2,
        )

    def test_embedded_identity_census_excludes_anchored_forms_and_hashes_pairs(
        self,
    ) -> None:
        census = common.audited_identity_embedded_residual_census(
            [
                {
                    "seller_uid": "seller-a",
                    "model_text": (
                        "BioHazard wallstreetbet wallstreetbet || "
                        "wallstreet maker makerpharma"
                    ),
                }
            ],
            {"biohaz", "biohazar", "wallstreet", "maker"},
        )
        self.assertEqual(census["scanned_seller_row_count"], 1)
        self.assertEqual(census["matched_registry_token_count"], 3)
        self.assertEqual(census["matched_occurrence_count"], 4)
        self.assertEqual(census["matched_alias_surface_pair_count"], 3)
        self.assertEqual(census["matched_surface_count"], 2)
        self.assertEqual(
            census["matched_anchor_sha256_counts"][
                common.sha256_text("wallstreet")
            ],
            2,
        )
        self.assertNotIn(
            common.sha256_text("maker"),
            census["matched_anchor_sha256_counts"],
        )
        self.assertEqual(
            census["matched_alias_surface_pair_sha256_counts"][
                common.sha256_text("biohaz" + "\0" + "biohazard")
            ],
            1,
        )

    def test_model_native_pooling_contracts_match_local_models(self) -> None:
        observed = {
            key: common.validate_sentence_transformer_layout(key, cfg)
            for key, cfg in self.policy["embedding_models"].items()
        }
        self.assertEqual(observed["gte_multilingual_base"]["pooling"], "cls")
        self.assertEqual(observed["bge_m3"]["pooling"], "cls")
        self.assertEqual(observed["multilingual_e5_large"]["pooling"], "mean")
        self.assertEqual(
            self.policy["embedding_models"]["multilingual_e5_large"]["text_prefix"],
            "query: ",
        )
        self.assertEqual(observed["labse"]["pooling"], "cls")
        self.assertTrue(observed["labse"]["has_dense_module"])
        self.assertEqual(
            observed["paraphrase_multilingual_mpnet_base_v2"]["pooling"], "mean"
        )
        for model_key, cfg in {
            **self.policy["embedding_models"],
            self.policy["shared_reranker"]["model_key"]: self.policy["shared_reranker"],
        }.items():
            common.validate_expected_model_pin(model_key, cfg)

    def test_gpu_encoder_source_has_no_label_reader(self) -> None:
        source = "\n".join(
            inspect.getsource(function)
            for function in (
                encode.corpus_and_pairs,
                encode.encode_embedding_model,
                encode.encode_reranker,
            )
        )
        for forbidden in (
            "review_label",
            "train_labels",
            "valid_labels",
            "historical_test_labels",
            "evidence_type",
        ):
            self.assertNotIn(forbidden, source)

    def test_candidate_matrix_separates_encoder_pipeline_and_shortcut_roles(self) -> None:
        specs = select.candidate_specs(self.policy)
        self.assertEqual(len(specs), 25)
        forbidden = set(self.policy["forbidden_m0_features"])
        shortcuts = set(
            self.policy["pair_feature_roles"]["shortcut_audit_only_features"]
        )
        for spec in specs:
            self.assertFalse(forbidden & set(spec["feature_names"]))
            if spec["m0_pipeline_eligible"] or spec["encoder_comparison_eligible"]:
                self.assertFalse(shortcuts & set(spec["feature_names"]))
        encoder_only = [spec for spec in specs if spec["encoder_comparison_eligible"]]
        self.assertEqual(len(encoder_only), 5)
        for spec in encoder_only:
            self.assertEqual(
                spec["feature_names"],
                [self.policy["embedding_models"][spec["model_key"]]["feature_name"]],
            )
        self.assertEqual(sum(spec["m0_pipeline_eligible"] for spec in specs), 20)
        controls = [spec for spec in specs if spec["candidate_role"] == "no_encoder_control"]
        self.assertEqual(len(controls), 4)
        self.assertTrue(all(spec["attribution_control_only"] for spec in controls))
        self.assertTrue(all(not spec["m0_pipeline_eligible"] for spec in controls))
        audit = [spec for spec in specs if spec["shortcut_audit_only"]]
        self.assertEqual([spec["candidate_id"] for spec in audit], ["audit__shortcut_features_only"])
        self.assertEqual(set(audit[0]["feature_names"]), shortcuts)
        full = [
            spec
            for spec in specs
            if spec["tier"] == "encoder_plus_transfer_plus_shared_reranker"
        ]
        self.assertEqual(len(full), 5)
        for spec in full:
            self.assertIn(self.policy["shared_reranker"]["feature_name"], spec["feature_names"])
        for spec in specs:
            if spec["candidate_role"] == "encoder_pipeline":
                self.assertIn(
                    spec["matched_no_encoder_control"],
                    {control["candidate_id"] for control in controls},
                )
        training = self.policy["training"]
        self.assertFalse(training["evidence_type_used_as_training_feature"])
        self.assertFalse(training["evidence_type_used_as_training_weight"])
        self.assertTrue(training["evidence_type_used_for_validation_safety_guards"])
        self.assertTrue(training["evidence_type_used_for_reporting_slices"])
        self.assertEqual(
            training["l2_selection"],
            "five_fold_component_grouped_train_oof_shortcut_conditioned_macro_component_equal_average_precision",
        )
        self.assertEqual(
            training["threshold_metric"],
            "weighted_balanced_accuracy_using_weighting_mode",
        )
        self.assertEqual(training["solver"], "newton_with_armijo_backtracking")
        self.assertEqual(
            training["solver_convergence_criterion"],
            "normalized_gradient_inf_norm_at_most_tolerance",
        )
        self.assertGreater(
            self.policy["evaluation"]["embedding_score_replay_absolute_tolerance"],
            0.0,
        )

    def test_shared_idf_is_byte_stable_across_python_hash_seeds(self) -> None:
        child = "\n".join(
            (
                "import json, sys",
                f"sys.path.insert(0, {str(SCRIPTS)!r})",
                "import step7_v3_common as common",
                "shared = {f'token_{index:03d}' for index in range(300)}",
                "frequency = {f'token_{index:03d}': (index * index * 37) % 581 for index in range(300)}",
                "print(json.dumps(common.shared_idf(shared, frequency, 582), separators=(',', ':')))",
            )
        )
        outputs = []
        for seed in ("0", "1", "7", "20260721"):
            environment = dict(os.environ)
            environment["PYTHONHASHSEED"] = seed
            outputs.append(
                subprocess.check_output(
                    [sys.executable, "-c", child],
                    cwd=ROOT,
                    env=environment,
                    text=True,
                    encoding="utf-8",
                ).strip()
            )
        self.assertEqual(len(set(outputs)), 1, outputs)

    def test_metrics_include_auc_ap_f1_recall_and_mrr(self) -> None:
        rows = [
            {"seller_uid_left": "q", "seller_uid_right": f"c{i}"}
            for i in range(4)
        ]
        labels = np.asarray([1, 0, 1, 0], dtype=np.int8)
        scores = np.asarray([0.9, 0.8, 0.7, 0.1], dtype=np.float64)
        metrics = select.full_metrics(rows, labels, scores, threshold=0.75)
        self.assertAlmostEqual(metrics["roc_auc"], 0.75)
        self.assertAlmostEqual(metrics["average_precision"], 5.0 / 6.0)
        self.assertIn("f1", metrics)
        self.assertIn("recall", metrics)
        self.assertIn("balanced_accuracy", metrics)
        self.assertEqual(metrics["labelled_candidate_ranking"]["status"], "diagnostic_labelled_pair_set_only")
        self.assertIn("mrr", metrics["labelled_candidate_ranking"])
        self.assertIn("recall_at_3", metrics["labelled_candidate_ranking"])
        uniform = np.ones(len(labels), dtype=np.float64)
        self.assertAlmostEqual(
            select.weighted_average_precision(labels, scores, uniform),
            metrics["average_precision"],
        )

    def test_shortcut_conditioned_ap_ignores_between_stratum_score_offsets(self) -> None:
        strata = (
            "same_market=0|same_source=1",
            "same_market=0|same_source=1",
            "same_market=1|same_source=1",
            "same_market=1|same_source=1",
        )
        rows = [
            {
                "pair_uid": f"p{index}",
                "split_name": "valid",
                "component_id": f"c{index}",
                "review_label": "positive" if index % 2 == 0 else "negative",
                select.SHORTCUT_CONTROL_STRATUM_FIELD: stratum,
            }
            for index, stratum in enumerate(strata)
        ]
        labels = np.asarray([1, 0, 1, 0], dtype=np.int8)
        original = np.asarray([0.90, 0.80, 0.40, 0.30], dtype=np.float64)
        shifted = np.asarray([0.90, 0.80, 0.85, 0.75], dtype=np.float64)
        original_result = select.shortcut_conditioned_component_equal_average_precision(
            rows, labels, original, self.policy, require_expected_strata=True
        )
        shifted_result = select.shortcut_conditioned_component_equal_average_precision(
            rows, labels, shifted, self.policy, require_expected_strata=True
        )
        self.assertEqual(original_result["macro_average_precision"], 1.0)
        self.assertEqual(shifted_result["macro_average_precision"], 1.0)
        self.assertNotEqual(
            select.average_precision(labels, original),
            select.average_precision(labels, shifted),
        )

    def test_grouped_folds_never_split_components_and_hold_both_classes(self) -> None:
        rows = []
        for component_index in range(15):
            rows.append(
                {
                    "component_id": f"c{component_index:02d}",
                    "review_label": "positive" if component_index % 3 == 0 else "negative",
                }
            )
            rows.append(
                {
                    "component_id": f"c{component_index:02d}",
                    "review_label": "negative" if component_index % 3 == 0 else "positive",
                }
            )
        folds = select.balanced_component_folds(rows, 5, 20260721)
        self.assertEqual(set(folds), {row["component_id"] for row in rows})
        for fold in range(5):
            labels = {
                row["review_label"]
                for row in rows
                if folds[row["component_id"]] == fold
            }
            self.assertEqual(labels, {"positive", "negative"})

    def test_logistic_fit_and_train_oof_threshold_are_finite(self) -> None:
        policy = copy.deepcopy(self.policy)
        policy["training"]["l2_grid"] = [0.1, 1.0]
        policy["training"]["fold_count"] = 5
        rows = []
        values = []
        for component_index in range(20):
            for label_index, label in enumerate(("negative", "positive")):
                rows.append(
                    {
                        "pair_uid": f"p{component_index}_{label_index}",
                        "split_name": "train",
                        "component_id": f"c{component_index}",
                        "review_label": label,
                        "seller_uid_left": f"l{component_index}_{label_index}",
                        "seller_uid_right": f"r{component_index}_{label_index}",
                        select.SHORTCUT_CONTROL_STRATUM_FIELD: (
                            "same_market=0|same_source=1"
                            if component_index % 2 == 0
                            else "same_market=1|same_source=1"
                        ),
                    }
                )
                values.append([float(label_index), float(component_index % 4)])
        matrix = np.asarray(values, dtype=np.float64)
        result = select.tune_and_fit(
            rows,
            matrix,
            ["signal", "nuisance"],
            policy,
            "component_equal_normalized_to_row_count",
        )
        self.assertTrue(result["final_train_artifact"]["solver_converged"])
        self.assertLessEqual(
            result["final_train_artifact"][
                "solver_final_normalized_gradient_inf_norm"
            ],
            policy["training"]["tolerance"],
        )
        self.assertEqual(
            result["final_train_artifact"]["solver_line_search"],
            "armijo_backtracking",
        )
        self.assertTrue(math.isfinite(result["selected_threshold"]))
        self.assertEqual(len(result["train_oof_scores"]), len(rows))
        self.assertGreater(result["train_oof_metrics"]["average_precision"], 0.95)
        self.assertEqual(len(result["component_fold_diagnostics"]), 5)
        self.assertTrue(
            all(item["component_count"] > 0 for item in result["component_fold_diagnostics"])
        )
        self.assertIn(
            "weighted_balanced_accuracy", result["threshold_selection"]
        )
        intercept_only = select.tune_and_fit(
            rows,
            np.empty((len(rows), 0), dtype=np.float64),
            [],
            policy,
            "component_equal_normalized_to_row_count",
        )
        self.assertEqual(
            intercept_only["final_train_artifact"]["coefficients"], []
        )
        self.assertLessEqual(
            intercept_only["final_train_artifact"][
                "solver_final_normalized_gradient_inf_norm"
            ],
            policy["training"]["tolerance"],
        )
        self.assertTrue(
            np.all(np.isfinite(intercept_only["train_oof_scores"]))
        )

    def test_selection_code_does_not_parse_test_labels(self) -> None:
        selection_source = inspect.getsource(select.run_selection)
        test_source = inspect.getsource(select.run_historical_test)
        self.assertNotIn('load_split_rows(policy, "test"', selection_source)
        self.assertIn('load_split_rows(policy, "test"', test_source)
        self.assertTrue(
            self.policy["selection_rule"]["identity_rule_control"]["eligible_for_m0"]
            is False
        )

    def test_development_label_projection_never_accesses_test_label_values(self) -> None:
        pair_rows = common.load_csv(
            common.resolve(self.policy["outputs"]["pair_manifest"])
        )
        labels_path = common.resolve(self.policy["inputs"]["frozen_labels"]["path"])
        evidence_path = common.resolve(self.policy["inputs"]["evidence_labels"]["path"])
        original_load_csv = common.load_csv

        class GuardedOtherSplitRow(dict):
            guarded = {
                "review_label",
                "usable_for_supervision",
                "evidence_type",
                "shared_contact_count",
                "shared_pgp_fingerprint_count",
            }

            def get(self, key, default=None):
                if key in self.guarded:
                    raise AssertionError(f"other-split value was accessed: {key}")
                return super().get(key, default)

            def __getitem__(self, key):
                if key in self.guarded:
                    raise AssertionError(f"other-split value was accessed: {key}")
                return super().__getitem__(key)

        def guarded_load(path):
            rows = original_load_csv(path)
            if Path(path).resolve() in {labels_path.resolve(), evidence_path.resolve()}:
                return [
                    row
                    if row.get("split_name") in {"train", "valid"}
                    else GuardedOtherSplitRow(row)
                    for row in rows
                ]
            return rows

        with mock.patch.object(common, "load_csv", side_effect=guarded_load):
            prepared = prepare.prepare_private_labels(
                self.policy, pair_rows, ("train", "valid")
            )
        self.assertEqual(prepared["label_counts"]["train"]["total"], 401)
        self.assertEqual(prepared["label_counts"]["valid"]["total"], 152)

    def test_label_coupled_minimal_gpu_manifest_is_rejected(self) -> None:
        policy = copy.deepcopy(self.policy)
        outputs = policy["outputs"]
        pair_rows = common.load_csv(common.resolve(outputs["pair_manifest"]))
        labels = common.load_csv(common.resolve(outputs["train_labels"])) + common.load_csv(
            common.resolve(outputs["valid_labels"])
        )
        label_by_pair = {row["pair_uid"]: row["review_label"] for row in labels}
        with tempfile.TemporaryDirectory(
            prefix=".step7_v3_reject_", dir=ROOT / "reports"
        ) as temporary:
            temp_root = Path(temporary)
            model_key, cfg = next(iter(policy["embedding_models"].items()))
            score_path = temp_root / "label_coupled.csv"
            common.write_csv_immutable(
                score_path,
                [
                    {
                        "pair_uid": pair["pair_uid"],
                        cfg["feature_name"]: (
                            "0.900000000000"
                            if label_by_pair.get(pair["pair_uid"]) == "positive"
                            else "0.100000000000"
                        ),
                    }
                    for pair in pair_rows
                ],
            )
            outputs["embedding_pair_scores_template"] = str(score_path)
            outputs["embedding_manifest_template"] = str(temp_root / "minimal.json")
            common.write_json_immutable(
                temp_root / "minimal.json",
                {
                    "version": policy["version"],
                    "model_key": model_key,
                    "pair_manifest_sha256": common.sha256_file(
                        common.resolve(outputs["pair_manifest"])
                    ),
                    "pair_scores_sha256": common.sha256_file(score_path),
                },
            )
            with self.assertRaises((FileNotFoundError, ValueError)):
                select.run_selection(policy)

    def test_provenance_valid_but_numerically_inconsistent_gpu_score_is_rejected(self) -> None:
        policy = copy.deepcopy(self.policy)
        with tempfile.TemporaryDirectory(
            prefix=".step7_v3_numeric_replay_", dir=ROOT / "reports"
        ) as temporary:
            self.materialize_label_free_mock_gpu_bundle(policy, Path(temporary))
            model_key, cfg = next(iter(policy["embedding_models"].items()))
            outputs = policy["outputs"]
            score_path = common.resolve(
                outputs["embedding_pair_scores_template"].format(model_key=model_key)
            )
            manifest_path = common.resolve(
                outputs["embedding_manifest_template"].format(model_key=model_key)
            )
            score_rows = common.load_csv(score_path)
            old_value = float(score_rows[0][cfg["feature_name"]])
            score_rows[0][cfg["feature_name"]] = (
                "0.750000000000" if old_value < 0.5 else "-0.750000000000"
            )
            score_path.write_bytes(common.render_csv(score_rows))
            embedding_manifest = common.load_json(manifest_path)
            embedding_manifest["pair_scores_sha256"] = common.sha256_file(score_path)
            manifest_path.write_bytes(
                (json.dumps(embedding_manifest, ensure_ascii=False, indent=2) + "\n").encode(
                    "utf-8"
                )
            )
            bundle_path = common.resolve(outputs["gpu_output_manifest"])
            bundle = common.load_json(bundle_path)
            refreshed = {
                str(path.relative_to(ROOT)).replace("\\", "/"): path
                for path in (score_path, manifest_path)
            }
            for record in bundle["files"]:
                if record["path"] in refreshed:
                    path = refreshed[record["path"]]
                    record["sha256"] = common.sha256_file(path)
                    record["size_bytes"] = path.stat().st_size
            bundle["total_file_bytes"] = sum(
                int(record["size_bytes"]) for record in bundle["files"]
            )
            bundle_path.write_bytes(
                (json.dumps(bundle, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
            )
            with self.assertRaisesRegex(ValueError, "does not replay from matrix"):
                select.load_feature_bundle(policy)

    def test_historical_label_file_and_manifest_cannot_be_pair_tampered(self) -> None:
        policy = copy.deepcopy(self.policy)
        outputs = policy["outputs"]
        source_label_path = common.resolve(outputs["historical_test_labels"])
        source_manifest = common.load_json(
            common.resolve(outputs["historical_test_labels_manifest"])
        )
        with tempfile.TemporaryDirectory(
            prefix=".step7_v3_test_label_replay_", dir=ROOT / "reports"
        ) as temporary:
            temp_root = Path(temporary)
            tampered_label_path = temp_root / "private_labels.test.csv"
            tampered_manifest_path = temp_root / "historical_manifest.json"
            rows = common.load_csv(source_label_path)
            rows[0]["review_label"] = (
                "negative" if rows[0]["review_label"] == "positive" else "positive"
            )
            tampered_label_path.write_bytes(common.render_csv(rows))
            record = source_manifest["output_files"]["private_labels_test"]
            record["path"] = str(tampered_label_path.relative_to(ROOT)).replace("\\", "/")
            record["sha256"] = common.sha256_file(tampered_label_path)
            record["size_bytes"] = tampered_label_path.stat().st_size
            tampered_manifest_path.write_bytes(
                (json.dumps(source_manifest, ensure_ascii=False, indent=2) + "\n").encode(
                    "utf-8"
                )
            )
            outputs["historical_test_labels"] = str(tampered_label_path)
            outputs["historical_test_labels_manifest"] = str(tampered_manifest_path)
            with self.assertRaisesRegex(ValueError, "do not byte-replay"):
                select.verify_historical_test_labels(policy)

    def test_full_selection_then_delayed_test_with_uid_only_gpu_scores(self) -> None:
        policy = copy.deepcopy(self.policy)
        policy["training"]["l2_grid"] = [1.0]
        policy["evaluation"]["bootstrap"]["resamples"] = 200
        with tempfile.TemporaryDirectory(
            prefix=".step7_v3_contract_", dir=ROOT / "reports"
        ) as temporary:
            self.materialize_label_free_mock_gpu_bundle(policy, Path(temporary))
            historical_path = common.resolve(policy["outputs"]["historical_test_labels"])
            all_split_label_source = common.resolve(
                policy["inputs"]["frozen_labels"]["path"]
            )
            all_split_evidence_source = common.resolve(
                policy["inputs"]["evidence_labels"]["path"]
            )
            original_load_csv = common.load_csv
            original_sha256 = common.sha256_file

            def guarded_load_csv(path):
                if Path(path).resolve() == historical_path.resolve():
                    raise AssertionError("selection touched historical-test label values")
                return original_load_csv(path)

            def guarded_sha256(path):
                if Path(path).resolve() in {
                    historical_path.resolve(),
                    all_split_label_source.resolve(),
                    all_split_evidence_source.resolve(),
                }:
                    raise AssertionError(
                        "selection hashed historical-test or all-split label source"
                    )
                return original_sha256(path)

            with mock.patch.object(common, "load_csv", side_effect=guarded_load_csv), mock.patch.object(
                common, "sha256_file", side_effect=guarded_sha256
            ):
                summary, valid_predictions, train_oof_predictions, freeze = (
                    select.run_selection(policy)
                )
            self.assertEqual(summary["candidate_count"], 25)
            self.assertEqual(summary["encoder_comparison_candidate_count"], 5)
            self.assertEqual(summary["m0_pipeline_candidate_count"], 20)
            self.assertEqual(summary["attribution_control_candidate_count"], 4)
            self.assertEqual(summary["shortcut_audit_candidate_count"], 1)
            self.assertFalse(
                summary["historical_test_label_values_parsed_during_selection"]
            )
            self.assertFalse(
                summary["historical_test_label_file_touched_during_selection"]
            )
            self.assertEqual(len(valid_predictions), 25 * 152)
            self.assertEqual(len(train_oof_predictions), 25 * 401)
            self.assertIn("selection_script", freeze["runtime_inputs"])
            self.assertIn("common_script", freeze["runtime_inputs"])
            self.assertFalse(
                summary["encoder_only_selection"]["fitted_head_used_for_ranking"]
            )
            self.assertFalse(
                summary["encoder_only_selection"][
                    "training_weight_sensitivity_applicable"
                ]
            )
            self.assertIsNone(
                summary["encoder_only_selection"][
                    "uniform_training_shortcut_conditioned_ap_ranking"
                ]
            )
            self.assertFalse(
                any(
                    key.startswith("uniform_training")
                    for key in summary["encoder_only_selection"][
                        "unique_winner_checks"
                    ]
                )
            )
            self.assertEqual(
                summary["encoder_only_selection"]["score_source"],
                "raw_frozen_encoder_cosine_without_fitted_head",
            )
            self.assertEqual(len(summary["raw_encoder_comparison_results"]), 5)
            self.assertFalse(
                summary["e5_continuity_control"][
                    "changes_ranking_or_carry_forward"
                ]
            )
            self.assertEqual(
                freeze["carry_forward_to_step28"],
                summary["m0_pipeline_selection"]["candidate_ranking"][: len(freeze["carry_forward_to_step28"])],
            )
            json.dumps(summary)
            tampered_freeze = copy.deepcopy(freeze)
            tampered_freeze["carry_forward_to_step28"] = [
                next(
                    candidate_id
                    for candidate_id in summary["candidate_ranking"]
                    if candidate_id not in freeze["carry_forward_to_step28"]
                )
            ]
            with self.assertRaises(ValueError):
                select.verify_frozen_selection_consistency(
                    policy, tampered_freeze, summary
                )
            test_summary, test_predictions = select.run_historical_test(
                policy, freeze, summary
            )
            self.assertTrue(
                test_summary[
                    "historical_test_metrics_computed_after_selection_freeze"
                ]
            )
            self.assertFalse(test_summary["prospective_claim_allowed"])
            self.assertFalse(test_summary["raw_encoder_test_metrics_used_for_selection"])
            self.assertEqual(
                list(test_summary["raw_encoder_candidates_selected_on_valid_only"]),
                summary["encoder_only_selection"][
                    "carry_forward_encoder_candidates"
                ],
            )
            self.assertEqual(
                len(test_predictions),
                len(freeze["carry_forward_to_step28"]) * 181,
            )

    def test_current_prepared_split_and_component_counts(self) -> None:
        pairs = common.load_csv(common.resolve(self.policy["outputs"]["pair_manifest"]))
        split_counts = Counter(row["split_name"] for row in pairs)
        component_counts = {
            split: len({row["component_id"] for row in pairs if row["split_name"] == split})
            for split in ("train", "valid", "test")
        }
        self.assertEqual(split_counts, Counter({"train": 401, "test": 181, "valid": 152}))
        self.assertEqual(component_counts, {"train": 229, "valid": 28, "test": 38})


if __name__ == "__main__":
    unittest.main()
