from __future__ import annotations

import copy
import hashlib
import json
import sys
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import step11_cluster_chinese_graph as step11  # noqa: E402
import step11_build_v6_runtime_policy as runtime_policy  # noqa: E402
import step11_build_explicit_summary_manifest as explicit_manifest  # noqa: E402
import step11_cluster_level_audit as cluster_audit  # noqa: E402
import immutable_artifact_io as immutable_io  # noqa: E402


class Step11V6ContractsTests(unittest.TestCase):
    @staticmethod
    def _sha256(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()

    def test_step15_domain_standardizer_is_applied_before_global_scaler(self) -> None:
        artifact = {
            "feature_names": ["x"],
            "feature_means": [0.0],
            "feature_stds": [1.0],
            "standardizer_bundle": {
                "feature_names": ["x"],
                "domain_standardized_features": ["x"],
                "domain_stats": {"zh_target_strict": {"means": [2.0], "stds": [2.0]}},
                "global_means": [0.0],
                "global_stds": [1.0],
            },
            "params": {
                "w1": [[1.0]],
                "b1": [0.0],
                "wi": [1.0],
                "bi": [0.0],
            },
        }
        _, probabilities = step11.apply_step15_mlp_artifact([{"x": "4.0"}], artifact)
        expected = 1.0 / (1.0 + np.exp(-np.tanh(1.0)))
        self.assertAlmostEqual(float(probabilities[0]), float(expected), places=12)

    def test_direct_identity_edge_survives_every_graph_filter_stage(self) -> None:
        edge = {
            "pair_uid": "direct",
            "seller_uid_left": "a",
            "seller_uid_right": "b",
            "review_label": "positive",
            "relation_reliability_score": 0.0,
            "relation_reliability_direct_identity_support": 1,
        }
        policy = {
            "graph_policy": {
                "graph_edge_filters": {
                    "enabled": True,
                    "reciprocal_top_k": 1,
                    "require_shared_neighbor_count_min": 1,
                    "require_triangle_participation": True,
                    "relation_reliability_filter": {
                        "enabled": True,
                        "minimum_score": 1.0,
                        "hard_keep_direct_identity": True,
                    },
                }
            }
        }
        distractors = [
            {
                "pair_uid": "a-c",
                "seller_uid_left": "a",
                "seller_uid_right": "c",
                "review_label": "negative",
                "relation_reliability_score": 2.0,
                "relation_reliability_direct_identity_support": 0,
            },
            {
                "pair_uid": "b-d",
                "seller_uid_left": "b",
                "seller_uid_right": "d",
                "review_label": "negative",
                "relation_reliability_score": 2.0,
                "relation_reliability_direct_identity_support": 0,
            },
        ]
        retained, diagnostics = step11.apply_graph_edge_filters(
            [edge, *distractors], {"direct": 0.1, "a-c": 0.9, "b-d": 0.9}, "v6", policy
        )
        self.assertEqual([row["pair_uid"] for row in retained], ["direct"])
        self.assertEqual(
            diagnostics["relation_reliability_filter"]["direct_identity_hard_kept_count"], 1
        )
        self.assertEqual(diagnostics["direct_identity_hard_kept_by_reciprocal_top_k"], 1)
        self.assertEqual(diagnostics["direct_identity_hard_kept_by_shared_neighbor"], 1)
        self.assertEqual(diagnostics["direct_identity_hard_kept_by_triangle"], 1)
        reference_audit = diagnostics["frozen_label_reference_audit"]
        self.assertEqual(reference_audit["reviewed_direct_proof_positive_count"], 1)
        self.assertEqual(reference_audit["reviewed_negative_count"], 2)
        self.assertEqual(reference_audit["isolated_direct_proof_pair_count"], 1)
        final_stage = reference_audit["stages"][-1]
        self.assertEqual(final_stage["stage"], "post_filter")
        self.assertEqual(final_stage["reviewed_direct_proof_positive_retention_rate"], 1.0)
        self.assertEqual(final_stage["isolated_direct_proof_pair_retention_rate"], 1.0)
        self.assertEqual(final_stage["reviewed_negative_removal_rate_after_threshold"], 1.0)

    def test_clean_and_operational_graph_modes_are_identifier_separated(self) -> None:
        base = json.loads(
            (ROOT / "schema" / "step11_clustering_policy.json").read_text(encoding="utf-8")
        )
        base["step15_v6_validation_gate"] = {}
        clean = copy.deepcopy(base)
        operational = copy.deepcopy(base)
        self.assertEqual(runtime_policy.configure_validation_mode(clean, "clean_topology"), "clean_topology")
        self.assertEqual(
            runtime_policy.configure_validation_mode(
                operational, "identifier_assisted_operational"
            ),
            "identifier_operational",
        )
        clean_reliability = clean["graph_policy"]["graph_edge_filters"]["relation_reliability_filter"]
        operational_reliability = operational["graph_policy"]["graph_edge_filters"][
            "relation_reliability_filter"
        ]
        self.assertFalse(clean_reliability["hard_keep_direct_identity"])
        self.assertFalse(clean_reliability["use_direct_identity_context_for_reliability_rules"])
        self.assertEqual(clean_reliability["weights"]["shared_pgp_fingerprint"], 0.0)
        self.assertEqual(clean_reliability["weights"]["shared_seller_contact"], 0.0)
        self.assertTrue(operational_reliability["hard_keep_direct_identity"])
        self.assertTrue(
            operational_reliability["use_direct_identity_context_for_reliability_rules"]
        )
        self.assertGreater(operational_reliability["weights"]["shared_seller_contact"], 0.0)

        semantic_row = {
            "embedding_cosine_multilingual_e5_large": 0.9,
            "has_shared_contact_exact": 0,
            "shared_contact_count_capped": 0,
        }
        contact_row = {**semantic_row, "has_shared_contact_exact": 1}
        clean_without_contact = step11.compute_relation_reliability(
            semantic_row, clean_reliability
        )
        clean_with_contact = step11.compute_relation_reliability(contact_row, clean_reliability)
        self.assertEqual(clean_without_contact["score"], clean_with_contact["score"])
        self.assertTrue(clean_without_contact["semantic_topic_only"])
        self.assertTrue(clean_with_contact["semantic_topic_only"])

    def test_clean_reliability_ignores_identifier_uppercase_and_raw_retrieval_fields(self) -> None:
        base = json.loads(
            (ROOT / "schema" / "step11_clustering_policy.json").read_text(encoding="utf-8")
        )
        base["step15_v6_validation_gate"] = {}
        runtime_policy.configure_validation_mode(base, "clean_topology")
        cfg = base["graph_policy"]["graph_edge_filters"]["relation_reliability_filter"]
        self.assertEqual(len(cfg["feature_allowlist"]), 30)
        plain = {
            "embedding_cosine_multilingual_e5_large": 0.9,
            "profile_category_jaccard": 0.0,
            "uppercase_ratio_mean_percentile_gap_abs": 0.0,
            "structural_support_score_raw": 0.0,
            "candidate_rule_count_raw": 0.0,
            "has_shared_contact_exact": 0,
        }
        contaminated = {
            **plain,
            "uppercase_ratio_mean_percentile_gap_abs": 1.0,
            "structural_support_score_raw": 1.0,
            "candidate_rule_count_raw": 99.0,
            "has_shared_contact_exact": 1,
        }
        score_plain = step11.compute_relation_reliability(plain, cfg)
        score_contaminated = step11.compute_relation_reliability(contaminated, cfg)
        self.assertEqual(score_plain["score"], score_contaminated["score"])
        self.assertEqual(
            score_plain["style_consistency_score"],
            score_contaminated["style_consistency_score"],
        )
        self.assertNotEqual(
            score_plain["has_direct_identity_support"],
            score_contaminated["has_direct_identity_support"],
        )

    def test_posthoc_labels_do_not_change_filters_and_make_reference_audit_nonempty(self) -> None:
        records = [
            {
                "pair_uid": "positive",
                "seller_uid_left": "a",
                "seller_uid_right": "b",
                "relation_reliability_score": 0.8,
                "relation_reliability_direct_identity_support": 1,
            },
            {
                "pair_uid": "negative",
                "seller_uid_left": "c",
                "seller_uid_right": "d",
                "relation_reliability_score": 0.8,
                "relation_reliability_direct_identity_support": 0,
            },
        ]
        audit_records, join_diagnostics = step11.attach_posthoc_review_labels(
            records,
            {
                "positive": {"review_label": "positive"},
                "negative": {"review_label": "negative"},
            },
        )
        policy = {"graph_policy": {"graph_edge_filters": {"enabled": False}}}
        lookup = {"positive": 0.9, "negative": 0.8}
        retained_without, _ = step11.apply_graph_edge_filters(
            records, lookup, "clean", policy, reference_universe=records
        )
        retained_with, diagnostics = step11.apply_graph_edge_filters(
            records, lookup, "clean", policy, reference_universe=audit_records
        )
        self.assertEqual(
            [row["pair_uid"] for row in retained_without],
            [row["pair_uid"] for row in retained_with],
        )
        self.assertEqual(join_diagnostics["matched_pair_uid_count"], 2)
        self.assertEqual(join_diagnostics["auditable_reviewed_label_count"], 2)
        audit = diagnostics["frozen_label_reference_audit"]
        self.assertEqual(audit["reviewed_direct_proof_positive_count"], 1)
        self.assertEqual(audit["reviewed_negative_count"], 1)

    def test_step9_seed_mean_averages_per_seed_artifact_probabilities(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = []
            for index, coefficient in enumerate((1.0, -1.0)):
                path = Path(tmpdir) / f"artifact_{index}.json"
                path.write_text(
                    json.dumps(
                        {
                            "artifact_type": "logistic_regression_l2",
                            "feature_names": ["x"],
                            "standardization": {"mean": [0.0], "scale": [1.0]},
                            "parameter_coefficients": [coefficient],
                            "parameter_intercept": 0.0,
                        }
                    ),
                    encoding="utf-8",
                )
                paths.append(path)
            features, probabilities, metadata = step11.apply_step9_seed_mean_ensemble(
                [{"pair_uid": "p", "x": "1.0"}],
                [
                    {
                        "run_key": f"run_{index}",
                        "seed": 20260320 + index,
                        "scoring_backend": "logistic_regression_l2",
                        "model_path": None,
                        "scorer_artifact_path": path,
                    }
                    for index, path in enumerate(paths)
                ],
                None,
            )
        self.assertEqual(features, ["x"])
        self.assertAlmostEqual(float(probabilities[0]), 0.5, places=12)
        self.assertEqual(metadata["seed_count"], 2)

    def test_raw_feature_scorer_reads_only_the_declared_semantic_column(self) -> None:
        features, scores = step11.apply_raw_feature_scorer(
            [
                {"pair_uid": "a", "embedding_cosine_bge_m3": "0.25", "review_label": "positive"},
                {"pair_uid": "b", "embedding_cosine_bge_m3": "0.75", "review_label": "negative"},
            ],
            "embedding_cosine_bge_m3",
        )
        self.assertEqual(features, ["embedding_cosine_bge_m3"])
        np.testing.assert_allclose(scores, np.asarray([0.25, 0.75]))

    def test_publication_runtime_rejects_scorers_outside_frozen_roster(self) -> None:
        policy = {
            "scorer_selection": {
                "publication_validation": {
                    "selection_mode": "explicit_allowlist_only",
                    "allowed_scorer_families": ["raw_feature"],
                    "allowed_scorer_tokens": ["raw_bge_m3_cosine"],
                }
            }
        }
        step11.validate_publication_scorer_allowlist(
            policy,
            {"scorer_family": "raw_feature", "scorer_token": "raw_bge_m3_cosine"},
        )
        with self.assertRaises(SystemExit):
            step11.validate_publication_scorer_allowlist(
                policy,
                {"scorer_family": "step7", "scorer_token": "core_zero_shot_bge_m3"},
            )

    def test_step9_seed_mean_threshold_uses_nested_training_policy(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = []
            rows_by_seed = (
                (("p1", 1, 0.9), ("p2", 1, 0.7), ("p3", 0, 0.3), ("p4", 0, 0.1)),
                (("p1", 1, 0.8), ("p2", 1, 0.6), ("p3", 0, 0.4), ("p4", 0, 0.2)),
            )
            for index, rows in enumerate(rows_by_seed):
                path = Path(tmpdir) / f"valid_{index}.csv"
                path.write_text(
                    "pair_uid,y_true,prob_positive\n"
                    + "".join(f"{uid},{label},{score}\n" for uid, label, score in rows),
                    encoding="utf-8",
                )
                paths.append(path)
            nested = {
                "mode": "bootstrap_valid",
                "split": "zh_valid",
                "metric": "balanced_accuracy",
                "candidate_mode": "midpoints",
                "tie_break_order": ["f1", "higher_threshold"],
                "degenerate_prediction_guard": {"enabled": False},
                "bootstrap": {
                    "num_resamples": 5,
                    "aggregation": "median",
                    "minimum_unique_thresholds_for_non_fallback": 1,
                },
            }
            threshold, diagnostics = step11.step9_ensemble_threshold_from_valid_predictions(
                paths,
                {
                    "threshold_selection": {"metric": "invalid_top_level_value"},
                    "training": {"threshold_selection": nested},
                },
                [20260320, 20260321],
            )
        self.assertGreater(threshold, 0.0)
        self.assertLess(threshold, 1.0)
        self.assertEqual(diagnostics["source"], "ten_seed_mean_zh_valid_predictions_only")
        self.assertFalse(diagnostics["test_metrics_used_for_threshold_selection"])

    def test_runtime_builder_loads_raw_bge_threshold_from_step12_valid_metrics(self) -> None:
        original_root = runtime_policy.ROOT
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                root = Path(tmpdir)
                runtime_policy.ROOT = root
                metrics = root / "reports/step12/model_metrics.csv"
                metrics.parent.mkdir(parents=True, exist_ok=True)
                metrics.write_text(
                    "model_id,threshold,threshold_source\n"
                    "raw_bge_m3_cosine,0.625,mean_zh_valid_scores\n",
                    encoding="utf-8",
                )
                step12 = {
                    "outputs": {
                        "summary_json": "reports/step12/summary.json",
                        "model_metrics_csv": "reports/step12/model_metrics.csv",
                    }
                }
                control = runtime_policy.load_raw_bge_control(
                    step12,
                    {"reports/step12/model_metrics.csv": self._sha256(metrics)},
                )
                self.assertEqual(control["feature_name"], "embedding_cosine_bge_m3")
                self.assertEqual(control["primary_threshold"], 0.625)
                self.assertFalse(control["test_metrics_used_for_threshold_selection"])
                metrics.write_text(
                    "model_id,threshold,threshold_source\n"
                    "raw_bge_m3_cosine,0.625,test_selected\n",
                    encoding="utf-8",
                )
                with self.assertRaises(ValueError):
                    runtime_policy.load_raw_bge_control(
                        step12,
                        {"reports/step12/model_metrics.csv": self._sha256(metrics)},
                    )
        finally:
            runtime_policy.ROOT = original_root

    def test_runtime_frozen_binding_fails_closed_after_artifact_drift(self) -> None:
        original_root = step11.ROOT
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                root = Path(tmpdir)
                step11.ROOT = root
                files = {
                    "reports/pairs.csv": "pair_uid\n",
                    "reports/profiles.jsonl": "{}\n",
                    "reports/labels.csv": "pair_uid,review_label\n",
                    "reports/summary.json": "{}\n",
                    "schema/policy.json": "{}\n",
                    "reports/artifact.json": "{}\n",
                }
                for relative, contents in files.items():
                    path = root / relative
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_text(contents, encoding="utf-8")
                hashes = {
                    relative: self._sha256(root / relative) for relative in files
                }
                policy = {
                    "input_paths": {"step5_frozen_labels": "reports/labels.csv"},
                    "step15_v6_validation_gate": {
                        "frozen_input_file_sha256": hashes,
                        "frozen_input_hash_source": "unit_test",
                        "inductive_feature_lineage_verification": {
                            "transductive_valid_or_test_covariates_used_for_reference": False
                        },
                    },
                }
                scorer = {
                    "summary_path": root / "reports/summary.json",
                    "policy_path": root / "schema/policy.json",
                    "scorer_artifact_path": root / "reports/artifact.json",
                }
                result = step11.verify_runtime_frozen_inputs(
                    policy,
                    scorer,
                    root / "reports/pairs.csv",
                    root / "reports/profiles.jsonl",
                )
                self.assertEqual(result["verified_file_count"], len(files))
                (root / "reports/artifact.json").write_text('{"drift": true}\n', encoding="utf-8")
                with self.assertRaises(SystemExit):
                    step11.verify_runtime_frozen_inputs(
                        policy,
                        scorer,
                        root / "reports/pairs.csv",
                        root / "reports/profiles.jsonl",
                    )
        finally:
            step11.ROOT = original_root

    def test_runtime_builder_verifies_v4_active_manifest_self_hash_and_every_file(self) -> None:
        original_root = runtime_policy.ROOT
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                root = Path(tmpdir)
                runtime_policy.ROOT = root
                frozen_path = root / "reports/frozen.txt"
                frozen_path.parent.mkdir(parents=True, exist_ok=True)
                frozen_path.write_text("frozen\n", encoding="utf-8")
                active_path = root / runtime_policy.EXPECTED_ACTIVE_MANIFEST
                active_path.parent.mkdir(parents=True, exist_ok=True)
                active = {
                    "run_id": runtime_policy.EXPECTED_ACTIVE_RUN_ID,
                    "policy_version": "unit-test-step15-v6",
                    "files": [
                        {
                            "path": "reports/frozen.txt",
                            "sha256": self._sha256(frozen_path),
                            "size_bytes": frozen_path.stat().st_size,
                        }
                    ],
                }
                core = json.dumps(
                    active,
                    ensure_ascii=True,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
                active["manifest_sha256"] = hashlib.sha256(core).hexdigest()
                active_path.write_text(json.dumps(active), encoding="utf-8")
                step15_policy_path = root / runtime_policy.STEP15_POLICY
                step15_policy_path.parent.mkdir(parents=True, exist_ok=True)
                step15_policy_path.write_text(
                    json.dumps({"version": "unit-test-step15-v6"}), encoding="utf-8"
                )
                policy_path = root / "schema/step12.json"
                policy_path.parent.mkdir(parents=True, exist_ok=True)
                policy_path.write_text(
                    json.dumps(
                        {
                            "inputs": {
                                "step15_v6_active_manifest": runtime_policy.EXPECTED_ACTIVE_MANIFEST,
                                "step15_v6_active_manifest_run_id": runtime_policy.EXPECTED_ACTIVE_RUN_ID,
                            }
                        }
                    ),
                    encoding="utf-8",
                )
                _, _, hashes = runtime_policy.load_verified_active_manifest(
                    {"policy": "schema/step12.json"}
                )
                self.assertEqual(hashes["reports/frozen.txt"], self._sha256(frozen_path))
                frozen_path.write_text("drift\n", encoding="utf-8")
                with self.assertRaises(ValueError):
                    runtime_policy.load_verified_active_manifest(
                        {"policy": "schema/step12.json"}
                    )
        finally:
            runtime_policy.ROOT = original_root

    def test_v6_runners_defer_step13_and_use_raw_bge_control(self) -> None:
        core_runner = (ROOT / "scripts" / "run_step15_v6_linux_20260711.sh").read_text(
            encoding="utf-8"
        )
        graph_runner = (
            ROOT / "scripts" / "run_step11_v6_after_promotion_20260711.sh"
        ).read_text(encoding="utf-8")
        self.assertNotIn('"$PYTHON_BIN" scripts/step13_concept_drift_audit.py', core_runner)
        self.assertIn("--scorer-family raw_feature", graph_runner)
        self.assertIn("--raw-feature-control raw_bge_m3_cosine", graph_runner)
        self.assertNotIn("--step7-experiment core_zero_shot_bge_m3", graph_runner)
        self.assertIn('"$PYTHON_BIN" scripts/step13_concept_drift_audit.py', graph_runner)

    def test_runtime_builder_verifies_inductive_feature_lineage(self) -> None:
        original_root = runtime_policy.ROOT
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                root = Path(tmpdir)
                runtime_policy.ROOT = root
                reference = root / "reports/reference.json"
                features = root / "reports/features.csv"
                manifest_path = root / "reports/lineage.json"
                reference.parent.mkdir(parents=True, exist_ok=True)
                reference.write_text("{}\n", encoding="utf-8")
                features.write_text("pair_uid\np\n", encoding="utf-8")
                manifest = {
                    "reference_scope": "frozen_train_sellers_only",
                    "transductive_valid_or_test_covariates_used_for_reference": False,
                    "candidate_pair_universe_changed": False,
                    "reference_bundle": "reports/reference.json",
                    "reference_bundle_sha256": self._sha256(reference),
                    "domains": {
                        "en_content_train_pool": {
                            "output_path": "reports/features.csv",
                            "output_sha256": self._sha256(features),
                            "canonical_semantic_values_preserved": True,
                        },
                        "zh_target_strict": {
                            "output_path": "reports/features.csv",
                            "output_sha256": self._sha256(features),
                            "canonical_semantic_values_preserved": True,
                        }
                    },
                }
                manifest["manifest_sha256"] = hashlib.sha256(
                    json.dumps(
                        manifest,
                        ensure_ascii=True,
                        sort_keys=True,
                        separators=(",", ":"),
                    ).encode("utf-8")
                ).hexdigest()
                manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
                result = runtime_policy.verify_inductive_feature_lineage(
                    manifest_path, reference, features
                )
                self.assertFalse(
                    result["transductive_valid_or_test_covariates_used_for_reference"]
                )
                features.write_text("pair_uid\ndrift\n", encoding="utf-8")
                with self.assertRaises(ValueError):
                    runtime_policy.verify_inductive_feature_lineage(
                        manifest_path, reference, features
                    )
        finally:
            runtime_policy.ROOT = original_root

    def test_publication_manifest_and_audit_reject_incomplete_or_summary_mode(self) -> None:
        with self.assertRaises(SystemExit):
            explicit_manifest.validate_publication_args(
                Namespace(
                    publication_v6=True,
                    validation_mode=None,
                    expected_scorer_token=["a"],
                )
            )
        with self.assertRaises(SystemExit):
            explicit_manifest.validate_publication_args(
                Namespace(
                    publication_v6=True,
                    validation_mode="clean_topology",
                    expected_scorer_token=["a", "a"],
                )
            )
        with self.assertRaises(SystemExit):
            cluster_audit.validate_input_mode(
                Namespace(publication_v6=True, manifest=None, summary_paths=[Path("x")])
            )

    def test_publication_manifest_rejects_empty_or_unreachable_primary_graph(self) -> None:
        runtime = {
            "scorer_selection": {
                "publication_validation": {
                    "selection_mode": "explicit_allowlist_only",
                    "auto_selector_allowed": False,
                }
            }
        }
        summary = {
            "frozen_input_verification": {
                "enabled": True,
                "verified_file_count": 3,
            },
            "acceptance_checks": {
                check: True
                for check in explicit_manifest.PUBLICATION_REQUIRED_ACCEPTANCE_CHECKS
            },
            "posthoc_frozen_label_audit": {
                "used_by_model_features": False,
                "used_by_graph_filter_decisions": False,
                "auditable_reviewed_label_count": 1,
            },
        }
        explicit_manifest.validate_publication_summary(
            summary,
            runtime,
            summary_path=Path("summary.json"),
            runtime_policy_path=Path("runtime.json"),
        )
        for failed_check in (
            "graph_primary_threshold_not_above_score_ceiling",
            "graph_primary_threshold_has_candidate_edges",
            "graph_primary_threshold_has_post_filter_edges",
        ):
            invalid = copy.deepcopy(summary)
            invalid["acceptance_checks"][failed_check] = False
            with self.assertRaises(ValueError):
                explicit_manifest.validate_publication_summary(
                    invalid,
                    runtime,
                    summary_path=Path("summary.json"),
                    runtime_policy_path=Path("runtime.json"),
                )

    def test_publication_artifacts_are_immutable_but_exact_replay_is_allowed(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "artifact.json"
            first = b'{"run":"fixed"}\n'
            self.assertEqual(immutable_io.write_immutable_bytes(path, first), "new")
            self.assertEqual(
                immutable_io.write_immutable_bytes(path, first), "identical_replay"
            )
            with self.assertRaises(FileExistsError):
                immutable_io.write_immutable_bytes(path, b'{"run":"changed"}\n')
            self.assertEqual(path.read_bytes(), first)

    def test_cluster_audit_decisions_are_independent_per_scorer(self) -> None:
        def appearance(token: str, family: str, proof: int, clone: int) -> dict:
            return {
                "seller_set_id": "a||b",
                "members": ["a", "b"],
                "seller_raw_members": ["a", "b"],
                "top_categories": ["x"],
                "contact_member_preview_count": 0,
                "summary_path": f"{token}.json",
                "cluster_path": f"{token}.csv",
                "scorer_token": token,
                "family": family,
                "cluster_id": "c1",
                "cluster_rank": 1,
                "graph_threshold": 0.5,
                "cluster_score_mean": 0.8,
                "edge_count": 1,
                "contact_edges": proof,
                "pgp_edges": 0,
                "identifier_edges": proof,
                "proof_positive_edges": proof,
                "nonproof_positive_edges": 0,
                "negative_edges": 0,
                "uncertain_edges": 0,
                "missing_label_edges": 1 - proof,
                "title_clone_edges": clone,
                "description_clone_edges": 0,
                "stratum_counts": cluster_audit.Counter(
                    {"text_clone_primary": clone}
                ),
                "label_counts": cluster_audit.Counter(
                    {"positive": proof, "missing": 1 - proof}
                ),
            }

        rows = cluster_audit.aggregate_audit_rows(
            [
                appearance("step15", "step15", 1, 0),
                appearance("step9", "step9", 0, 1),
            ]
        )
        self.assertEqual(len(rows), 2)
        decisions = {row["scorer_token"]: row["decision"] for row in rows}
        self.assertEqual(decisions["step15"], "same_controller_high_confidence")
        self.assertEqual(decisions["step9"], "template_clone_not_controller")
        self.assertTrue(
            all(row["comparison_scope"] == "per_scorer_no_cross_model_max" for row in rows)
        )

    def test_cluster_audit_rechecks_posthoc_label_binding(self) -> None:
        original_root = cluster_audit.ROOT
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                root = Path(tmpdir)
                cluster_audit.ROOT = root
                label_path = root / "reports/labels.csv"
                label_path.parent.mkdir(parents=True, exist_ok=True)
                label_path.write_text("pair_uid,review_label\np,positive\n", encoding="utf-8")
                digest = self._sha256(label_path)
                policy = {
                    "input_paths": {"step5_frozen_labels": "reports/labels.csv"},
                    "step15_v6_validation_gate": {
                        "frozen_input_file_sha256": {"reports/labels.csv": digest}
                    },
                }
                summary = {
                    "posthoc_frozen_label_audit": {
                        "sha256": digest,
                        "used_by_model_features": False,
                        "used_by_graph_filter_decisions": False,
                    }
                }
                cluster_audit.validate_runtime_posthoc_label_binding(
                    summary, policy, label_path
                )
                label_path.write_text("pair_uid,review_label\np,negative\n", encoding="utf-8")
                with self.assertRaises(ValueError):
                    cluster_audit.validate_runtime_posthoc_label_binding(
                        summary, policy, label_path
                    )
        finally:
            cluster_audit.ROOT = original_root


if __name__ == "__main__":
    unittest.main()
