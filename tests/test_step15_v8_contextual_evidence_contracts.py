from __future__ import annotations

import csv
import json
import sys
import tempfile
import unittest
from collections import Counter
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import step15_v8_common as common  # noqa: E402
import step15_v8_downstream_gate as downstream_gate  # noqa: E402
import step15_v8_preflight as preflight  # noqa: E402
import step16_build_v8_context_review_queues as review_queues  # noqa: E402
import step16_build_v8_identity_control_queues as identity_control_queues  # noqa: E402
import step16_apply_v8_context_reviews as review_apply  # noqa: E402
import step20_build_representative_validation as representative_validation  # noqa: E402


class Step15V8ContextualEvidenceContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.policy = json.loads(
            (ROOT / "schema" / "step15_v8_contextual_evidence_policy.json").read_text(
                encoding="utf-8"
            )
        )
        cls.v7_policy = json.loads(
            (ROOT / "schema" / "step15_v7_two_stage_policy.json").read_text(
                encoding="utf-8"
            )
        )
        cls.readiness_root = (
            ROOT
            / "reports"
            / "step16_v8_validation_refreeze"
            / "readiness_expansion_v2_20260715"
        )

    @staticmethod
    def _csv_rows(path: Path) -> list[dict]:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            return list(csv.DictReader(handle))

    def test_preregistered_B0_B3_contract_and_forbidden_features(self) -> None:
        validation = common.validate_policy_contract(self.policy, self.v7_policy)
        self.assertEqual(validation["status"], "pass")
        self.assertEqual(validation["seed_count"], 10)
        self.assertEqual(validation["group_folds"], 5)
        self.assertEqual(
            self.policy["clean_semantics"]["reranker_pair_symmetrization"],
            "mean_forward_reverse",
        )
        forbidden = set(self.policy["bridge_audit"]["forbidden_features"])
        for feature_set_id in self.policy["bridge_audit"]["feature_sets"]:
            names = common.feature_names(feature_set_id, self.policy, self.v7_policy)
            self.assertFalse(set(names) & forbidden)
        self.assertEqual(
            len(common.feature_names("B0_v7_20d_plus_e5_latent64", self.policy, self.v7_policy)),
            84,
        )
        self.assertEqual(
            len(common.feature_names("B1_v7_20d_e5_cosine_only", self.policy, self.v7_policy)),
            20,
        )

    def _preflight_feature_rows(self) -> list[dict]:
        feature_names = common.feature_names(
            "B1_v7_20d_e5_cosine_only", self.policy, self.v7_policy
        )
        rows = []
        for index, domain in enumerate(("en", "zh")):
            row = {
                "pair_uid": f"preflight-{domain}",
                "domain": domain,
                "sparse_lexical_similarity_raw": str(0.2 + index),
                "structural_support_score_raw": str(0.4 + index),
            }
            row.update({name: str(index + 1.0) for name in feature_names})
            rows.append(row)
        return rows

    def test_preflight_allows_cells_covered_by_train_median_imputation(self) -> None:
        rows = self._preflight_feature_rows()
        rows[1]["price_median_percentile_gap_abs"] = ""
        diagnostics = preflight.validate_v7_feature_availability(
            rows, self.policy, self.v7_policy
        )
        self.assertEqual(
            diagnostics["nonfinite_cell_counts"]["price_median_percentile_gap_abs"],
            1,
        )
        self.assertEqual(diagnostics["imputation_mode"], "train_median_per_feature")

    def test_preflight_rejects_a_v7_column_absent_from_any_row(self) -> None:
        rows = self._preflight_feature_rows()
        del rows[1]["price_median_percentile_gap_abs"]
        with self.assertRaisesRegex(ValueError, "required v7 columns are absent"):
            preflight.validate_v7_feature_availability(
                rows, self.policy, self.v7_policy
            )

    def test_preflight_rejects_a_v7_feature_missing_on_all_train_rows(self) -> None:
        rows = self._preflight_feature_rows()
        for row in rows:
            row["price_median_percentile_gap_abs"] = ""
        with self.assertRaisesRegex(ValueError, "entirely missing"):
            preflight.validate_v7_feature_availability(
                rows, self.policy, self.v7_policy
            )

    def test_nonidentifier_candidate_rules_are_allowlist_not_subtraction(self) -> None:
        self.assertEqual(
            set(self.policy["bridge_audit"]["nonidentifier_candidate_rule_allowlist"]),
            {
                "profile_lexical_neighbor",
                "shared_title_clone",
                "shared_description_clone",
                "structural_support",
            },
        )
        self.assertNotIn(
            "shared_contact_exact",
            self.policy["bridge_audit"]["nonidentifier_candidate_rule_allowlist"],
        )

    def test_factorized_weighting_forbids_global_multiplier(self) -> None:
        weighting = self.v7_policy["factorized_evidence_weighting"]
        self.assertIs(weighting["forbid_global_eight_x_multiplier"], True)
        self.assertIn("evidence_type_factor", weighting)
        self.assertIn("confidence_factor", weighting)

    def test_seeded_group_folds_are_component_disjoint_and_exhaustive(self) -> None:
        rows = []
        for group_index in range(20):
            domain = "en" if group_index % 2 == 0 else "zh"
            for label in ("negative", "positive"):
                rows.append(
                    {
                        "pair_uid": f"{group_index}:{label}",
                        "domain": domain,
                        "v7_component_id": f"component-{group_index}",
                        "review_label": label,
                    }
                )
        folds = common.seeded_component_group_folds(rows, 5, 20260320)
        covered = []
        for train, held in folds:
            train_groups = {common.component_group_key(rows[index]) for index in train}
            held_groups = {common.component_group_key(rows[index]) for index in held}
            self.assertFalse(train_groups & held_groups)
            covered.extend(held.tolist())
        self.assertEqual(sorted(covered), list(range(len(rows))))

    def test_review_queue_split_assignment_is_component_level(self) -> None:
        pairs = {
            ("seller_a", "seller_b"),
            ("seller_b", "seller_c"),
            ("seller_x", "seller_y"),
        }
        components, component_ids = review_queues.candidate_component_index(pairs)
        self.assertEqual(components["seller_a"], ("seller_a", "seller_b", "seller_c"))
        self.assertEqual(components["seller_a"], components["seller_c"])
        self.assertEqual(component_ids["seller_a"], component_ids["seller_c"])
        self.assertNotEqual(component_ids["seller_a"], component_ids["seller_x"])
        self.assertEqual(
            review_queues.deterministic_unseen_component_split(components["seller_a"]),
            review_queues.deterministic_unseen_component_split(components["seller_c"]),
        )

    def test_unreviewed_public_candidates_do_not_form_a_global_component(self) -> None:
        left_pair, left_id = review_queues.pair_review_component("seller_a", "seller_b")
        right_pair, right_id = review_queues.pair_review_component("seller_b", "seller_c")
        self.assertEqual(left_pair, ("seller_a", "seller_b"))
        self.assertEqual(right_pair, ("seller_b", "seller_c"))
        self.assertNotEqual(left_id, right_id)

    def test_pair_split_eligibility_only_blocks_internal_test(self) -> None:
        seller_splits = {
            "train_seller": {"train"},
            "valid_seller": {"valid"},
            "test_seller": {"internal_development_test"},
        }
        state, membership = review_queues.pair_split_eligibility(
            "train_seller", "valid_seller", seller_splits
        )
        self.assertEqual(state, "train_valid_refreeze_candidate")
        self.assertEqual(membership, ("train", "valid"))
        state, _ = review_queues.pair_split_eligibility(
            "train_seller", "test_seller", seller_splits
        )
        self.assertEqual(state, "diagnostic_test_only")

    def test_representative_validation_excludes_silver_and_nonbenchmark_rows(self) -> None:
        base = {
            "review_label": "positive",
            "usable_for_supervision": "1",
            "usable_for_core_transfer": "1",
            "benchmark_eligible": "1",
            "silver_train_only": "0",
        }
        self.assertTrue(representative_validation.eligible(base))
        silver = dict(base, silver_train_only="1")
        self.assertFalse(representative_validation.eligible(silver))
        nonbenchmark = dict(base, benchmark_eligible="0")
        self.assertFalse(representative_validation.eligible(nonbenchmark))

    def test_v7_e5_corpus_hash_is_nested_under_redaction_diagnostics(self) -> None:
        metadata = json.loads(
            (
                ROOT
                / "reports"
                / "step15_v7"
                / "v2_identifier_redacted_20260714"
                / "clean_embeddings"
                / "multilingual_e5_large_identifier_redacted.zh_target_strict.json"
            ).read_text(encoding="utf-8")
        )
        self.assertNotIn("clean_text_corpus_sha256", metadata)
        self.assertEqual(
            len(metadata["redaction_diagnostics"]["clean_text_corpus_sha256"]),
            64,
        )

    def test_review_queue_keeps_candidate_evidence_immutable_and_decisions_separate(self) -> None:
        self.assertNotIn("reviewer_a_identity_label", review_queues.QUEUE_FIELDS)
        self.assertIn("existing_v7_pair_feature_ready", review_queues.QUEUE_FIELDS)
        self.assertIn("identity_label", review_queues.BLIND_REVIEW_PACKET_FIELDS)
        self.assertNotIn("queue_kind", review_queues.BLIND_REVIEW_PACKET_FIELDS)
        self.assertNotIn("evidence_state", review_queues.BLIND_REVIEW_PACKET_FIELDS)
        self.assertEqual(
            set(review_queues.BLIND_REVIEW_PACKET_FIELDS),
            review_apply.BLIND_PACKET_FIELDS,
        )
        self.assertFalse(
            any("model_score" in field for field in review_queues.BLIND_REVIEW_PACKET_FIELDS)
        )

    def test_identity_control_packet_hides_split_and_model_fields(self) -> None:
        fields = set(identity_control_queues.REVIEW_PACKET_FIELDS)
        self.assertIn("platform_vendor_id", fields)
        self.assertIn("same_vendor_path_evidence", fields)
        self.assertFalse(
            fields
            & {
                "candidate_kind",
                "candidate_rule",
                "assigned_split",
                "existing_split_membership",
                "model_score",
                "old_label",
                "test_membership",
            }
        )
        self.assertIn(
            "primary_alias_benchmark_eligible",
            identity_control_queues.MASTER_FIELDS,
        )

    def test_blind_review_preview_hides_parser_state_flags(self) -> None:
        occurrence = {
            "contact_type": "telegram",
            "context": "Telegram @seller_support",
            "direct_identity_eligible": "1",
            "seller_facing_context": "1",
            "product_data_risk_context": "0",
            "support_only": "0",
        }
        blind = review_queues.blind_occurrence_preview([occurrence])
        diagnostic = review_queues.diagnostic_occurrence_preview([occurrence])
        self.assertIn("Telegram @seller_support", blind)
        for forbidden in ("direct=", "seller_facing=", "risky=", "support="):
            self.assertNotIn(forbidden, blind)
        self.assertIn("direct=1", diagnostic)
        self.assertIn("seller_facing=1", diagnostic)

    def test_dual_review_requires_independence_and_high_confidence(self) -> None:
        cfg = json.loads(
            (ROOT / "schema" / "step16_v8_validation_refreeze_policy.json").read_text(
                encoding="utf-8"
            )
        )["review_protocol"]
        row = {
            "review_candidate_uid": "candidate-1",
            "queue_kind": "risky_only_public_noise",
            "reviewer_a_id": "reviewer-a",
            "reviewer_a_identity_label": "negative",
            "reviewer_a_evidence_type": "public_contact_or_url_noise",
            "reviewer_a_confidence": "high",
            "reviewer_b_id": "reviewer-b",
            "reviewer_b_identity_label": "negative",
            "reviewer_b_evidence_type": "public_contact_or_url_noise",
            "reviewer_b_confidence": "high",
        }
        resolved = review_apply.resolve_review_decision(row, cfg)
        self.assertEqual(resolved["status"], "resolved_high_confidence")
        self.assertEqual(resolved["decision_source"], "matching_independent_reviews")
        row["reviewer_b_id"] = "reviewer-a"
        with self.assertRaises(ValueError):
            review_apply.resolve_review_decision(row, cfg)

    def test_dual_review_disagreement_requires_distinct_adjudication(self) -> None:
        cfg = json.loads(
            (ROOT / "schema" / "step16_v8_validation_refreeze_policy.json").read_text(
                encoding="utf-8"
            )
        )["review_protocol"]
        row = {
            "review_candidate_uid": "candidate-2",
            "queue_kind": "mixed_context_identifier",
            "reviewer_a_id": "reviewer-a",
            "reviewer_a_identity_label": "positive",
            "reviewer_a_evidence_type": "same_controller_direct_identifier",
            "reviewer_a_confidence": "high",
            "reviewer_b_id": "reviewer-b",
            "reviewer_b_identity_label": "negative",
            "reviewer_b_evidence_type": "public_contact_or_url_noise",
            "reviewer_b_confidence": "high",
        }
        self.assertEqual(
            review_apply.resolve_review_decision(row, cfg)["status"],
            "requires_adjudication",
        )
        row.update(
            {
                "adjudicator_id": "reviewer-c",
                "adjudicated_identity_label": "positive",
                "adjudicated_evidence_type": "same_controller_direct_identifier",
                "adjudication_confidence": "high",
            }
        )
        resolved = review_apply.resolve_review_decision(row, cfg)
        self.assertEqual(resolved["status"], "resolved_high_confidence")
        self.assertEqual(resolved["decision_source"], "independent_adjudication")

    def test_completed_blind_packet_cannot_change_candidate_evidence(self) -> None:
        candidate = {
            "review_candidate_uid": "candidate-3",
            "seller_uid_left": "seller-left",
            "seller_uid_right": "seller-right",
            "shared_identifier_types": "telegram",
            "shared_identifier_values": "telegram:shared",
            "left_context_preview": "left evidence",
            "right_context_preview": "right evidence",
        }
        row = {
            **candidate,
            "reviewer_id": "reviewer-a",
            "identity_label": "negative",
            "evidence_type": "public_contact_or_url_noise",
            "confidence": "high",
            "notes": "public support account",
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "review.csv"
            path.write_bytes(
                common.render_csv([row], review_queues.BLIND_REVIEW_PACKET_FIELDS)
            )
            loaded = review_apply.load_completed_blind_packet(
                path, {"candidate-3": candidate}, "a", True
            )
            self.assertEqual(loaded["candidate-3"]["identity_label"], "negative")
            tampered = dict(row)
            tampered["left_context_preview"] = "changed evidence"
            path.write_bytes(
                common.render_csv([tampered], review_queues.BLIND_REVIEW_PACKET_FIELDS)
            )
            with self.assertRaises(ValueError):
                review_apply.load_completed_blind_packet(
                    path, {"candidate-3": candidate}, "a", True
                )

    def test_effective_v7_policy_uses_reviewed_overlay_bindings(self) -> None:
        policy = json.loads(json.dumps(self.policy))
        policy["pools"]["zh_target_strict"]["frozen_labels"] = "reports/new-labels.csv"
        policy["pools"]["zh_target_strict"]["evidence_labels"] = "reports/new-evidence.csv"
        policy["frozen_dependencies"][
            "representative_validation_assignments"
        ] = "reports/new-assignments.csv"
        effective = common.materialize_effective_v7_policy(policy, self.v7_policy)
        self.assertEqual(
            effective["pools"]["zh_target_strict"]["frozen_labels"],
            "reports/new-labels.csv",
        )
        self.assertEqual(
            effective["representative_validation"]["split_assignment_output"],
            "reports/new-assignments.csv",
        )

    def test_frozen_representative_manifest_uses_documented_legacy_hash_order(self) -> None:
        path = (
            ROOT
            / "reports"
            / "step15_v7"
            / "v2_identifier_redacted_20260714"
            / "splits"
            / "representative_validation_manifest.json"
        )
        manifest = json.loads(path.read_text(encoding="utf-8"))
        expected = manifest.pop("manifest_sha256")
        assignment_hash = manifest.pop("assignment_csv_sha256")
        self.assertEqual(expected, common.canonical_hash(manifest))
        self.assertEqual(len(assignment_hash), 64)

    def test_step20_lock_is_bound_to_exact_v8_freeze(self) -> None:
        lock = {
            "step15_v8_run_id": "run-a",
            "step15_v8_model_freeze_manifest_sha256": "freeze-a",
            "evaluation_completed_once": True,
            "evaluation_count": 1,
            "model_configuration_frozen_before_holdout_unseal": True,
            "threshold_frozen_before_holdout_unseal": True,
            "prospective_holdout_used_for_model_selection": False,
        }
        self.assertEqual(
            downstream_gate.validate_step20_lock(lock, "run-a", "freeze-a"), []
        )
        self.assertIn(
            "step15_v8_model_freeze_manifest_sha256",
            downstream_gate.validate_step20_lock(lock, "run-a", "stale-freeze"),
        )

    def test_fold_train_domain_normalization_never_reads_eval_rows(self) -> None:
        feature_set = "B3_nonidentifier_retrieval_bridge"
        names = common.feature_names(feature_set, self.policy, self.v7_policy)
        rows = []
        for index, domain in enumerate(("en", "en", "zh", "zh")):
            row = {name: str(index + 1) for name in names if not name.endswith("_train_domain_z")}
            row.update(
                {
                    "pair_uid": str(index),
                    "domain": domain,
                    "sparse_lexical_similarity_raw": str(index + 1),
                    "structural_support_score_raw": str((index + 1) * 2),
                }
            )
            rows.append(row)
        matrix, artifact = common.fit_feature_transform(
            rows, feature_set, self.policy, self.v7_policy, None
        )
        self.assertEqual(matrix.shape[1], len(names))
        self.assertEqual(artifact["domain_normalization"]["sparse_lexical_similarity_train_domain_z"]["en"]["mean"], 1.5)
        eval_row = dict(rows[-1])
        eval_row["sparse_lexical_similarity_raw"] = "1000"
        common.apply_feature_transform(
            [eval_row], self.policy, self.v7_policy, artifact, None
        )
        self.assertEqual(artifact["domain_normalization"]["sparse_lexical_similarity_train_domain_z"]["zh"]["mean"], 3.5)

    def test_occurrence_state_distinguishes_pure_direct_from_mixed(self) -> None:
        token = ("telegram", "shared")
        direct = {
            "direct_identity_eligible": "1",
            "seller_facing_context": "1",
            "product_data_risk_context": "0",
            "support_only": "0",
            "source_dataset": "fixture",
            "source_row_number": "1",
            "source_market_raw": "m",
        }
        risky = {
            "direct_identity_eligible": "0",
            "seller_facing_context": "0",
            "product_data_risk_context": "1",
            "support_only": "0",
            "source_dataset": "fixture",
            "source_row_number": "2",
            "source_market_raw": "m",
        }
        row = {"seller_uid_left": "a", "seller_uid_right": "b", "domain": "zh"}
        pure = common.occurrence_evidence(
            row, {"a": {token: [direct]}, "b": {token: [direct]}}, Counter({token: 2}), 3
        )
        mixed = common.occurrence_evidence(
            row,
            {"a": {token: [direct, risky]}, "b": {token: [direct]}},
            Counter({token: 2}),
            3,
        )
        self.assertEqual(pure["evidence_state"], "verified_direct_both_sides")
        self.assertEqual(mixed["evidence_state"], "direct_with_mixed_context")

    def test_contextual_correction_is_direction_constrained(self) -> None:
        clean = np.asarray([0.4, 0.8, 0.6, 0.3])
        evidence = [
            {"evidence_state": "verified_direct_both_sides"},
            {"evidence_state": "risky_only_shared"},
            {"evidence_state": "direct_with_mixed_context"},
            {"evidence_state": "no_shared_identifier"},
        ]
        fused, decisions = common.apply_constrained_expert(
            clean, evidence, np.asarray([-2.0, 2.0, -3.0, 4.0])
        )
        np.testing.assert_allclose(fused, clean, rtol=0.0, atol=1e-12)
        self.assertEqual(decisions[0]["applied_logit_correction"], 0.0)
        self.assertEqual(decisions[1]["applied_logit_correction"], 0.0)
        uplifted, _ = common.apply_constrained_expert(
            clean[:2], evidence[:2], np.asarray([1.0, -1.0])
        )
        self.assertGreater(uplifted[0], clean[0])
        self.assertLess(uplifted[1], clean[1])

    def test_chinese_interactions_receive_stronger_l2(self) -> None:
        names = self.policy["occurrence_evidence_expert"]["feature_names"]
        interactions = self.policy["occurrence_evidence_expert"][
            "domain_interaction_features"
        ]
        self.assertTrue(set(interactions).issubset(names))
        self.assertGreater(
            self.policy["occurrence_evidence_expert"]["chinese_interaction_l2_multiplier"],
            1.0,
        )

    def test_promotion_cannot_use_internal_test(self) -> None:
        self.assertIs(self.policy["evaluation"]["selection_reads_internal_test"], False)
        self.assertIs(
            self.policy["promotion_gates"]["internal_test_may_satisfy_no_gate"], True
        )
        self.assertTrue(
            self.policy["promotion_gates"]["step20_prospective_holdout_required_for_publication"]
        )
        self.assertIn(
            "grouped_bootstrap_clean_vs_B0_noninferiority_margin",
            self.policy["promotion_gates"],
        )
        self.assertIn(
            "grouped_bootstrap_fusion_vs_clean_noninferiority_margin",
            self.policy["promotion_gates"],
        )

    def test_validation_readiness_uses_occurrence_states_not_legacy_type_alone(self) -> None:
        rows = [
            {
                "review_label": "negative",
                "evidence_type": "public_contact_or_url_noise",
            },
            {"review_label": "negative", "evidence_type": "ordinary_negative"},
            {
                "review_label": "positive",
                "evidence_type": "same_controller_direct_identifier",
            },
            {
                "review_label": "positive",
                "evidence_type": "same_controller_component_anchor",
            },
        ]
        states = [
            "no_shared_identifier",
            "risky_only_shared",
            "verified_direct_both_sides",
            "no_shared_identifier",
        ]
        masks = common.validation_slice_masks(rows, states)
        self.assertEqual(
            masks["state_backed_public_noise_negative"].tolist(),
            [False, True, False, False],
        )
        self.assertEqual(int(np.sum(masks["state_backed_verified_direct_positive"])), 1)
        self.assertEqual(int(np.sum(masks["same_controller_component_anchor_positive"])), 1)
        self.assertEqual(
            self.policy["promotion_gates"]["minimum_valid_slice_counts"],
            {
                "state_backed_public_noise_negative": 20,
                "state_backed_verified_direct_positive": 20,
                "same_controller_component_anchor_positive": 15,
            },
        )

    def test_evidence_controls_count_for_readiness_but_not_primary_benchmark(self) -> None:
        gates = self.policy["promotion_gates"]
        self.assertEqual(
            gates["readiness_slice_scope"],
            "primary_representative_validation_plus_isolated_evidence_expert_validation_controls",
        )
        self.assertFalse(gates["evidence_expert_controls_used_for_primary_model_selection"])
        self.assertIn(
            "evidence_expert_validation_eligible=1",
            gates["readiness_eligibility"],
        )
        for definition in gates["validation_slice_definition"].values():
            self.assertIn("readiness_eligibility", definition)

        rows = [
            {
                "review_label": "positive",
                "evidence_type": "same_controller_direct_identifier",
                "benchmark_eligible": "0",
                "primary_identity_model_eligible": "0",
                "evidence_expert_eligible": "1",
                "evidence_expert_validation_eligible": "1",
            },
            {
                "review_label": "positive",
                "evidence_type": "same_controller_component_anchor",
                "benchmark_eligible": "0",
                "primary_identity_model_eligible": "0",
                "evidence_expert_eligible": "1",
                "evidence_expert_validation_eligible": "1",
            },
            {
                "review_label": "positive",
                "evidence_type": "same_controller_direct_identifier",
                "benchmark_eligible": "0",
                "primary_identity_model_eligible": "1",
                "evidence_expert_eligible": "1",
                "evidence_expert_validation_eligible": "1",
            },
        ]
        states = [
            "verified_direct_both_sides",
            "no_shared_identifier",
            "verified_direct_both_sides",
        ]
        masks = common.validation_slice_masks(rows, states)
        self.assertEqual(
            masks["state_backed_verified_direct_positive"].tolist(),
            [True, False, False],
        )
        self.assertEqual(
            masks["same_controller_component_anchor_positive"].tolist(),
            [False, True, False],
        )
        stacked = np.vstack(
            [
                masks["state_backed_public_noise_negative"],
                masks["state_backed_verified_direct_positive"],
                masks["same_controller_component_anchor_positive"],
            ]
        ).astype(int)
        self.assertTrue(np.all(np.sum(stacked, axis=0) <= 1))

    def test_primary_split_excludes_evidence_expert_only_controls(self) -> None:
        def row(uid: str, split: str, label: str, *, control: bool = False) -> dict:
            return {
                "pair_uid": uid,
                "v7_split_name": split,
                "review_label": label,
                "primary_identity_model_eligible": "0" if control else "1",
                "evidence_expert_eligible": "1",
            }

        rows_by_pool = {
            "en_content_train_pool": [
                row("en-pos", "train", "positive"),
                row("en-neg", "train", "negative"),
            ],
            "zh_target_strict": [
                row("zh-train-pos", "train", "positive"),
                row("zh-train-neg", "train", "negative"),
                row("zh-valid-pos", "valid", "positive"),
                row("zh-valid-neg", "valid", "negative"),
                row("zh-test-pos", "internal_development_test", "positive"),
                row("zh-test-neg", "internal_development_test", "negative"),
                row("expert-train", "train", "positive", control=True),
                row("expert-valid", "valid", "positive", control=True),
            ],
        }
        splits = common.split_rows(rows_by_pool)
        self.assertNotIn("expert-train", {item["pair_uid"] for item in splits["train"]})
        self.assertNotIn("expert-valid", {item["pair_uid"] for item in splits["valid"]})
        self.assertEqual(
            [item["pair_uid"] for item in splits["evidence_expert_train_controls"]],
            ["expert-train"],
        )
        self.assertEqual(
            [item["pair_uid"] for item in splits["evidence_expert_valid_controls"]],
            ["expert-valid"],
        )

    def test_materialized_readiness_freeze_hash_chain_is_closed(self) -> None:
        root = self.readiness_root
        self.assertTrue(root.is_dir())
        specifications = [
            ("step16_v8_readiness_freeze_manifest.json", "manifest_sha256"),
            ("representative_validation_manifest.v8_readiness.json", "manifest_sha256"),
            ("step16_v8_readiness_expansion_summary.json", "summary_sha256"),
        ]
        for filename, hash_field in specifications:
            payload = json.loads((root / filename).read_text(encoding="utf-8"))
            expected = payload.pop(hash_field)
            self.assertEqual(expected, common.canonical_hash(payload), filename)

        manifest = json.loads(
            (root / "step16_v8_readiness_freeze_manifest.json").read_text(
                encoding="utf-8"
            )
        )
        for artifact_name, record in manifest["outputs"].items():
            self.assertEqual(
                record["sha256"],
                common.sha256(common.resolve(record["path"])),
                artifact_name,
            )

    def test_generated_readiness_policy_has_atomic_linux_output_layout(self) -> None:
        policy = json.loads(
            (self.readiness_root / "step15_v7_readiness_policy.json").read_text(
                encoding="utf-8"
            )
        )
        execution = policy["readiness_execution_contract"]
        self.assertEqual(
            execution["clean_embedding_rebuild_pools"], ["zh_target_strict"]
        )
        self.assertEqual(
            execution["inductive_feature_rebuild_pools"],
            ["en_content_train_pool", "zh_target_strict"],
        )
        feature_parent = common.resolve(
            policy["inductive_features"]["manifest_output"]
        ).parent
        self.assertTrue(
            all(
                common.resolve(pool["v7_pair_features"]).parent == feature_parent
                for pool in policy["pools"].values()
            )
        )
        v8_policy = json.loads(
            (self.readiness_root / "step15_v8_readiness_policy.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(v8_policy["default_run_id"], "bridge_v8_readiness_20260715")
        self.assertEqual(
            common.resolve(v8_policy["validation_context_refreeze"]["freeze_manifest"]),
            self.readiness_root / "step16_v8_readiness_freeze_manifest.json",
        )

    def test_materialized_readiness_freeze_preserves_upstream_pair_rows(self) -> None:
        comparisons = [
            (
                ROOT / "reports" / "step4_zh_target_strict_silver_candidate_pairs.csv",
                self.readiness_root
                / "step4_zh_target_strict_candidates.v8_readiness.csv",
            ),
            (
                ROOT / "reports" / "step7_pair_features.zh_target_strict.csv",
                self.readiness_root
                / "step7_pair_features.zh_target_strict.canonical.v8_readiness.csv",
            ),
        ]
        for original_path, frozen_path in comparisons:
            original = self._csv_rows(original_path)
            frozen = {row["pair_uid"]: row for row in self._csv_rows(frozen_path)}
            self.assertTrue(set(row["pair_uid"] for row in original).issubset(frozen))
            for row in original:
                observed = frozen[row["pair_uid"]]
                self.assertEqual(
                    row,
                    {field: observed.get(field, "") for field in row},
                    row["pair_uid"],
                )

    def test_materialized_readiness_refreeze_only_supersedes_non_supervision(self) -> None:
        original = {
            row["pair_uid"]: row
            for row in self._csv_rows(
                ROOT / "reports" / "step5_zh_target_strict_frozen_silver_labels.csv"
            )
        }
        frozen = {
            row["pair_uid"]: row
            for row in self._csv_rows(
                self.readiness_root / "step5_zh_target_strict_labels.v8_readiness.csv"
            )
        }
        allowed_component_fields = {"split_component_id", "split_component_size"}
        superseded = []
        for pair_uid, row in original.items():
            self.assertIn(pair_uid, frozen)
            observed = frozen[pair_uid]
            changed = {
                field for field, value in row.items() if observed.get(field, "") != value
            }
            if not changed or changed <= allowed_component_fields:
                continue
            self.assertEqual(row.get("review_label"), "uncertain", pair_uid)
            self.assertEqual(row.get("usable_for_supervision"), "0", pair_uid)
            self.assertEqual(row.get("usable_for_core_transfer"), "0", pair_uid)
            superseded.append(pair_uid)
        self.assertEqual(len(superseded), 6)

    def test_materialized_readiness_isolated_controls_and_fixed_test(self) -> None:
        root = self.readiness_root
        labels = self._csv_rows(root / "step5_zh_target_strict_labels.v8_readiness.csv")
        assignments = self._csv_rows(
            root / "representative_validation_assignments.v8_readiness.csv"
        )
        controls = [
            row for row in labels if row.get("primary_identity_model_eligible") == "0"
        ]
        counts = Counter(
            (row["split_name"], row["identity_control_role"], row["review_label"])
            for row in controls
        )
        self.assertEqual(
            counts,
            Counter(
                {
                    ("train", "public_noise_control", "negative"): 20,
                    ("valid", "public_noise_control", "negative"): 20,
                    ("train", "direct_control", "positive"): 30,
                    ("valid", "direct_control", "positive"): 20,
                    ("train", "component_control", "positive"): 10,
                    ("valid", "component_control", "positive"): 15,
                }
            ),
        )
        self.assertTrue(
            all(
                row.get("benchmark_eligible") == "0"
                and row.get("usable_for_supervision") == "0"
                and row.get("usable_for_core_transfer") == "0"
                and row.get("evidence_expert_eligible") == "1"
                and row.get("evidence_expert_validation_eligible") == "1"
                and row.get("split_name") in {"train", "valid"}
                for row in controls
            )
        )
        public_controls = [
            row for row in controls if row["identity_control_role"] == "public_noise_control"
        ]
        self.assertEqual(len(public_controls), 40)
        self.assertTrue(
            all(
                row.get("label_tier")
                == "high_confidence_silver_agent_reviewed_public_noise_control"
                for row in public_controls
            )
        )

        old_test = {
            row["pair_uid"]
            for row in self._csv_rows(
                ROOT
                / "reports"
                / "step15_v7"
                / "v2_identifier_redacted_20260714"
                / "splits"
                / "representative_validation_assignments.csv"
            )
            if row["v7_split_name"] == "internal_development_test"
        }
        new_test = {
            row["pair_uid"]
            for row in assignments
            if row["v7_split_name"] == "internal_development_test"
        }
        self.assertEqual(len(new_test), 200)
        self.assertEqual(new_test, old_test)

        seller_splits: dict[str, set[str]] = {}
        for row in assignments:
            for seller_uid in (row["seller_uid_left"], row["seller_uid_right"]):
                seller_splits.setdefault(seller_uid, set()).add(row["v7_split_name"])
        self.assertFalse(
            {seller_uid: splits for seller_uid, splits in seller_splits.items() if len(splits) > 1}
        )

    def test_readiness_manifest_train_count_uses_only_english_train_split(self) -> None:
        root = self.readiness_root
        manifest = json.loads(
            (root / "representative_validation_manifest.v8_readiness.json").read_text(
                encoding="utf-8"
            )
        )
        transitive_paths = set(manifest["inputs"])
        for expected in (
            "products_data.csv",
            "reports/step3_seller_profiles.zh_target_aux.jsonl",
            "reports/step5_zh_target_aux_frozen_silver_labels.csv",
            "schema/step4_silver_candidate_schema.json",
            "schema/step7_transfer_safe_pair_feature_schema.json",
        ):
            self.assertIn(expected, transitive_paths)
        en_rows = self._csv_rows(ROOT / "reports" / "step5_en_frozen_silver_labels.csv")
        eligible_en = [
            row
            for row in en_rows
            if row.get("review_label") in {"positive", "negative"}
            and row.get("usable_for_supervision") == "1"
            and row.get("usable_for_core_transfer") == "1"
        ]
        en_train_count = sum(row.get("split_name") == "train" for row in eligible_en)
        labels = {
            row["pair_uid"]: row
            for row in self._csv_rows(
                root / "step5_zh_target_strict_labels.v8_readiness.csv"
            )
        }
        assignments = self._csv_rows(
            root / "representative_validation_assignments.v8_readiness.csv"
        )
        zh_primary_train_count = sum(
            row["v7_split_name"] == "train"
            and labels[row["pair_uid"]].get("primary_identity_model_eligible", "1")
            != "0"
            for row in assignments
        )
        expected = en_train_count + zh_primary_train_count
        self.assertEqual(en_train_count, 401)
        self.assertEqual(zh_primary_train_count, 523)
        self.assertEqual(manifest["row_counts"]["train"], expected)
        self.assertEqual(expected, 924)
        self.assertLess(expected, len(eligible_en) + zh_primary_train_count)

    def test_materialized_identity_controls_have_occurrence_level_backing(self) -> None:
        root = self.readiness_root
        labels = self._csv_rows(root / "step5_zh_target_strict_labels.v8_readiness.csv")
        signals = self._csv_rows(
            root / "step3_item_identity_signals.zh_target_strict.v8_readiness.csv"
        )
        platform_signals: dict[str, set[str]] = {}
        for row in signals:
            if row.get("contact_type") != "platform_vendor_id":
                continue
            self.assertEqual(row.get("direct_identity_eligible"), "1")
            self.assertEqual(row.get("seller_facing_context"), "1")
            self.assertEqual(row.get("product_data_risk_context"), "0")
            platform_signals.setdefault(row["seller_uid"], set()).add(
                row["normalized_value"]
            )

        controls = [
            row for row in labels if row.get("primary_identity_model_eligible") == "0"
        ]
        for row in controls:
            left = platform_signals.get(row["seller_uid_left"], set())
            right = platform_signals.get(row["seller_uid_right"], set())
            if row["identity_control_role"] == "direct_control":
                self.assertTrue(left & right, row["pair_uid"])
            elif row["identity_control_role"] == "component_control":
                self.assertFalse(left | right, row["pair_uid"])
            else:
                self.assertEqual(row["identity_control_role"], "public_noise_control")
                self.assertEqual(row["review_label"], "negative")


if __name__ == "__main__":
    unittest.main()
