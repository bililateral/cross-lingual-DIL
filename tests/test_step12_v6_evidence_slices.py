from __future__ import annotations

import json
import hashlib
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import step12_v6_statistical_robustness_audit as step12  # noqa: E402


class Step12V6EvidenceSliceTests(unittest.TestCase):
    def test_current_step16f_test_slice_contract(self) -> None:
        policy_path = ROOT / "schema" / "step12_v6_statistical_robustness_policy.json"
        with policy_path.open("r", encoding="utf-8") as handle:
            policy = json.load(handle)
        labels = step12.load_csv(ROOT / policy["inputs"]["labels"])
        reaudit = step12.load_csv(ROOT / policy["inputs"]["step16f_positive_reaudit"])
        test_rows = step12.eligible_split_rows(labels, policy["fixed_test"]["split_name"])
        masks, counts = step12.build_positive_slice_masks(test_rows, reaudit, policy)
        self.assertEqual(
            counts,
            {
                "strict_direct_or_component": 18,
                "soft_primary": 6,
                "secondary_or_sensitivity_only": 26,
            },
        )
        self.assertEqual(int(masks["all_test"].sum()), 200)
        self.assertEqual(
            int(masks["strict_direct_or_component_positive_vs_all_negative"].sum()), 168
        )
        self.assertEqual(
            int(masks["strict_plus_soft_primary_positive_vs_all_negative"].sum()), 174
        )
        self.assertEqual(int(masks["soft_primary_positive_vs_all_negative"].sum()), 156)
        self.assertEqual(int(masks["secondary_positive_vs_all_negative"].sum()), 176)

    def test_fixed_boundary_hashes_match_policy(self) -> None:
        policy_path = ROOT / "schema" / "step12_v6_statistical_robustness_policy.json"
        with policy_path.open("r", encoding="utf-8") as handle:
            policy = json.load(handle)
        labels = step12.load_csv(ROOT / policy["inputs"]["labels"])
        reaudit = step12.load_csv(ROOT / policy["inputs"]["step16f_positive_reaudit"])
        test_rows = step12.eligible_split_rows(labels, policy["fixed_test"]["split_name"])
        self.assertEqual(
            step12.canonical_rows_sha256(test_rows, policy["fixed_test"]["canonical_fields"]),
            policy["fixed_test"]["canonical_sha256"],
        )
        positive_uids = {row["pair_uid"] for row in test_rows if row["review_label"] == "positive"}
        tier_rows = [
            row
            for row in reaudit
            if row.get("split_name") == "test" and row.get("pair_uid") in positive_uids
        ]
        self.assertEqual(
            step12.canonical_rows_sha256(
                tier_rows, policy["fixed_test"]["step16f_tier_canonical_fields"]
            ),
            policy["fixed_test"]["step16f_tier_canonical_sha256"],
        )

    def test_two_level_comparison_pairs_matching_seed_ids(self) -> None:
        seed_scores = np.asarray(
            [
                [0.9, 0.8, 0.2, 0.1],
                [0.6, 0.4, 0.3, 0.2],
            ],
            dtype=float,
        )
        candidate = {
            "seed_ids": [1, 2],
            "seed_scores": seed_scores,
            "scores": seed_scores.mean(axis=0),
            "threshold": 0.5,
        }
        baseline = {
            "seed_ids": [2, 1],
            "seed_scores": seed_scores[[1, 0]],
            "scores": seed_scores.mean(axis=0),
            "threshold": 0.5,
        }
        rows = step12.two_level_comparison(
            candidate,
            baseline,
            np.asarray([1.0, 1.0, 0.0, 0.0]),
            [np.asarray([0]), np.asarray([1]), np.asarray([2]), np.asarray([3])],
            ["average_precision"],
            200,
            7,
            0.95,
        )
        self.assertTrue(rows[0]["paired_by_seed_id"])
        self.assertEqual(rows[0]["difference"], 0.0)
        self.assertEqual(rows[0]["ci_low"], 0.0)
        self.assertEqual(rows[0]["ci_high"], 0.0)

    def test_promotion_requires_holm_and_ten_paired_seeds(self) -> None:
        policy = json.loads(
            (ROOT / "schema" / "step12_v6_statistical_robustness_policy.json").read_text(
                encoding="utf-8"
            )
        )
        rows = []
        baselines = [
            "step15_v6_m0",
            "step9_strongest_clean_validation_selected",
        ]
        for baseline in baselines:
            for scope in (
                "all_test",
                "strict_plus_soft_primary_positive_vs_all_negative",
            ):
                rows.append(
                    {
                        "candidate_model_id": "step15_v6_final_selected",
                        "baseline_model_id": baseline,
                        "evaluation_scope": scope,
                        "metric": "average_precision",
                        "analysis_mode": step12.PRIMARY_ANALYSIS_MODE,
                        "ci_low": 0.01,
                        "p_value_method": step12.PERMUTATION_P_VALUE_METHOD,
                        "p_value_holm_source": "permutation_p_value_raw",
                        "p_value_holm": 0.01,
                        "paired_by_seed_id": True,
                        "candidate_seed_count": 10,
                        "baseline_seed_count": 10,
                        "candidate_positive_seed_count": 9,
                    }
                )
        rows[2]["p_value_holm"] = 0.2
        self.assertFalse(step12.evaluate_promotion(rows, policy)["eligible"])
        rows[2]["p_value_holm"] = 0.04
        self.assertTrue(step12.evaluate_promotion(rows, policy)["eligible"])

    def test_promotion_requires_strict_plus_soft_ci_gate(self) -> None:
        policy = json.loads(
            (ROOT / "schema" / "step12_v6_statistical_robustness_policy.json").read_text(
                encoding="utf-8"
            )
        )
        rows = []
        for baseline in policy["promotion_rule"]["required_positive_ci_against"]:
            for scope in policy["promotion_rule"]["required_positive_ci_evaluation_scopes"]:
                rows.append(
                    {
                        "candidate_model_id": "step15_v6_final_selected",
                        "baseline_model_id": baseline,
                        "evaluation_scope": scope,
                        "metric": "average_precision",
                        "analysis_mode": step12.PRIMARY_ANALYSIS_MODE,
                        "ci_low": 0.01,
                        "p_value_method": step12.PERMUTATION_P_VALUE_METHOD,
                        "p_value_holm_source": "permutation_p_value_raw",
                        "p_value_holm": 0.01,
                        "paired_by_seed_id": True,
                        "candidate_seed_count": 10,
                        "baseline_seed_count": 10,
                        "candidate_positive_seed_count": 9,
                    }
                )
        strict_soft = next(
            row
            for row in rows
            if row["baseline_model_id"] == "step15_v6_m0"
            and row["evaluation_scope"]
            == "strict_plus_soft_primary_positive_vs_all_negative"
        )
        strict_soft["ci_low"] = -0.001
        result = step12.evaluate_promotion(rows, policy)
        self.assertFalse(result["eligible"])
        self.assertFalse(result["positive_ci_against_required_baselines"])
        strict_soft["ci_low"] = 0.001
        self.assertTrue(step12.evaluate_promotion(rows, policy)["eligible"])

    def test_aliases_receive_identical_model_bootstrap_intervals(self) -> None:
        policy = json.loads(
            (ROOT / "schema" / "step12_v6_statistical_robustness_policy.json").read_text(
                encoding="utf-8"
            )
        )
        scores = np.asarray([[0.9, 0.7, 0.3, 0.1], [0.8, 0.6, 0.4, 0.2]])
        base = {
            "role": "test",
            "seed_ids": [1, 2],
            "seed_scores": scores,
            "scores": scores.mean(axis=0),
            "threshold": 0.5,
            "threshold_source": "test",
            "valid_average_precision": 1.0,
        }
        rows = step12.model_metric_rows(
            {"base": {**base}, "alias": {**base, "alias_of": "base"}},
            [{"pair_uid": str(idx)} for idx in range(4)],
            np.asarray([1.0, 1.0, 0.0, 0.0]),
            [np.asarray([0]), np.asarray([1]), np.asarray([2]), np.asarray([3])],
            policy,
            100,
        )
        by_id = {row["model_id"]: row for row in rows}
        for key in by_id["base"]:
            if key.endswith("_ci_low") or key.endswith("_ci_high"):
                self.assertEqual(by_id["base"][key], by_id["alias"][key])

    def test_active_manifest_self_hash_is_verified(self) -> None:
        with tempfile.TemporaryDirectory(dir=ROOT) as temporary_dir:
            temporary = Path(temporary_dir)
            artifact = temporary / "artifact.txt"
            artifact.write_text("frozen", encoding="utf-8")
            record = {
                "path": str(artifact.relative_to(ROOT)),
                "sha256": step12.file_sha256(artifact),
                "size_bytes": artifact.stat().st_size,
            }
            core = {
                "run_id": "run",
                "policy_version": "policy",
                "files": [record],
            }
            digest = hashlib.sha256(
                json.dumps(core, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode(
                    "utf-8"
                )
            ).hexdigest()
            manifest_path = temporary / "manifest.json"
            manifest_path.write_text(
                json.dumps({**core, "manifest_sha256": digest}), encoding="utf-8"
            )
            _, index = step12.load_verified_active_manifest(manifest_path, "run", "policy")
            self.assertIn(str(artifact.relative_to(ROOT)), index)
            manifest_path.write_text(
                json.dumps({**core, "files": [], "manifest_sha256": digest}), encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "self hash mismatch"):
                step12.load_verified_active_manifest(manifest_path, "run", "policy")

    def test_prediction_ranking_metrics_must_reproduce_summary(self) -> None:
        y_true = np.asarray([1.0, 1.0, 0.0, 0.0])
        scores = np.asarray([0.9, 0.8, 0.2, 0.1])
        metrics = step12.step7.evaluate_probabilities(y_true, scores, 0.5)
        step12.assert_ranking_metrics_match_summary(y_true, scores, metrics, "test")
        tampered = dict(metrics)
        tampered["average_precision"] = 0.25
        with self.assertRaisesRegex(ValueError, "do not reproduce"):
            step12.assert_ranking_metrics_match_summary(y_true, scores, tampered, "test")

    def test_slice_report_includes_prevalence_lift_and_grouped_ci(self) -> None:
        policy = json.loads(
            (ROOT / "schema" / "step12_v6_statistical_robustness_policy.json").read_text(
                encoding="utf-8"
            )
        )
        y_true = np.asarray([1.0, 1.0, 0.0, 0.0])
        seed_scores = np.asarray([[0.9, 0.8, 0.2, 0.1], [0.8, 0.7, 0.3, 0.2]])
        models = {
            "model": {
                "seed_scores": seed_scores,
                "scores": seed_scores.mean(axis=0),
                "threshold": 0.5,
            }
        }
        test_rows = [
            {"pair_uid": str(idx), "split_component_id": f"component-{idx}"}
            for idx in range(4)
        ]
        rows = step12.evidence_slice_rows(
            models,
            test_rows,
            y_true,
            {"slice": np.ones(4, dtype=bool)},
            policy,
            100,
        )
        self.assertEqual(rows[0]["prevalence"], 0.5)
        self.assertEqual(rows[0]["average_precision_prevalence_lift"], 2.0)
        self.assertTrue(rows[0]["unstable_slice"])
        self.assertIsNotNone(rows[0]["average_precision_ci_low"])
        self.assertIsNotNone(rows[0]["average_precision_two_level_ci_low"])
        self.assertEqual(
            rows[0]["bootstrap_mode"],
            "primary_split_component_fixed_seed_mean_with_supplemental_two_level_seed_and_component",
        )


if __name__ == "__main__":
    unittest.main()
