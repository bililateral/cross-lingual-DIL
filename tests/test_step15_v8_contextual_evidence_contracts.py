from __future__ import annotations

import json
import sys
import unittest
from collections import Counter
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import step15_v8_common as common  # noqa: E402
import step15_v8_downstream_gate as downstream_gate  # noqa: E402
import step16_build_v8_context_review_queues as review_queues  # noqa: E402


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


if __name__ == "__main__":
    unittest.main()
