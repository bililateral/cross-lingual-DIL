from __future__ import annotations

import csv
import json
import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import step28_common as base  # noqa: E402
import step28_history_common as history  # noqa: E402
import step28_score_real_identity_candidates as scorer  # noqa: E402


TRAINING_POLICY_PATH = ROOT / "schema" / "step28_transferable_identity_history_v12_policy.json"
APPLICATION_POLICY_PATH = ROOT / "schema" / "step28_transferable_identity_history_v12_1_guarded_application_policy.json"


class Step28V12ApplicationContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.training_policy = history.load_policy(TRAINING_POLICY_PATH)
        cls.application_policy = history.load_policy(APPLICATION_POLICY_PATH)
        cls.training_root = base.output_root(cls.training_policy)
        cls.application_root = base.output_root(cls.application_policy)
        cls.train_outputs = cls.training_policy["outputs"]
        cls.app_outputs = cls.application_policy["outputs"]
        cls.model_rows = base.load_csv(
            cls.training_root / cls.train_outputs["model_inputs"]
        )
        cls.artifacts = base.load_json(
            cls.training_root / cls.train_outputs["model_artifacts"]
        )

    def test_reviewed_registry_is_uid_only_and_excluded_everywhere(self) -> None:
        registry_path = base.resolve(
            self.application_policy["inputs"]["known_reviewed_pair_uid_exclusions"]
        )
        with registry_path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            self.assertEqual(reader.fieldnames, ["pair_uid"])
            reviewed = {row["pair_uid"] for row in reader}
        self.assertEqual(len(reviewed), 1259)
        scores = base.load_csv(
            self.application_root / self.app_outputs["real_candidate_scores"]
        )
        queue = base.load_csv(
            self.application_root / self.app_outputs["prospective_review_queue"]
        )
        self.assertTrue(reviewed.isdisjoint(row["pair_uid"] for row in scores))
        self.assertTrue(reviewed.isdisjoint(row["pair_uid"] for row in queue))

    def test_observable_state_support_recomputes(self) -> None:
        observed = scorer.recompute_observable_state_support(
            self.model_rows, self.training_policy["model"]["feature_names"]
        )
        self.assertEqual(
            observed,
            self.artifacts["primary_model"]["observable_state_support"],
        )

    def test_threshold_uses_all_nonambiguous_development_rows_with_state_weights(self) -> None:
        names = self.training_policy["model"]["feature_names"]
        support = scorer.recompute_observable_state_support(self.model_rows, names)
        rows = [
            row
            for row in self.model_rows
            if row["synthetic_split"] == "synthetic_development"
        ]
        matrix = np.asarray(
            [[float(row[name]) for name in names] for row in rows], dtype=float
        )
        labels = np.asarray(
            [float(row["review_label"] == "positive") for row in rows], dtype=float
        )
        selected = np.asarray(
            [
                index
                for index, values in enumerate(matrix)
                if support[history.observable_state_hash(values)]["status"]
                != "ambiguous"
            ],
            dtype=int,
        )
        matrix = matrix[selected]
        labels = labels[selected]
        hashes = [history.observable_state_hash(values) for values in matrix]
        counts = {value: hashes.count(value) for value in set(hashes)}
        weights = np.asarray([1.0 / counts[value] for value in hashes])
        scores = base.sigmoid(
            history.identity_correction(matrix, self.artifacts["primary_model"])
        )
        threshold, _ = history.choose_threshold_weighted(labels, scores, weights)
        self.assertAlmostEqual(
            threshold, self.artifacts["frozen_threshold"], places=12
        )

    def test_primary_audit_retains_every_row_and_equalizes_states(self) -> None:
        summary = base.load_json(
            self.training_root / self.train_outputs["training_summary"]
        )
        primary = summary["metrics_by_split"]["synthetic_audit"][
            "all_rows_equal_observable_state_weight"
        ]["m2_full_history"]
        generated = base.load_json(
            self.training_root / self.train_outputs["generation_summary"]
        )
        self.assertEqual(primary["raw_row_count"], 1280)
        self.assertEqual(
            primary["state_equal_weight_total"],
            generated["feature_state_diagnostics"]["synthetic_audit"][
                "unique_feature_state_count"
            ],
        )
        self.assertTrue(summary["checks"]["primary_audit_does_not_filter_on_audit_labels"])

    def test_recipe_layers_use_unique_states_and_production_contract(self) -> None:
        summary = base.load_json(
            self.training_root / self.train_outputs["training_summary"]
        )
        rules = self.training_policy["audit_gates"]["audit_recipe_rules"]
        diagnostics = summary["audit_recipe_diagnostics"]
        self.assertEqual(set(rules), set(diagnostics))
        for recipe, rule in rules.items():
            self.assertLessEqual(
                diagnostics[recipe]["unique_observable_state_count"],
                diagnostics[recipe]["raw_row_count"],
            )
            if rule["layer"] == "identity_model":
                self.assertEqual(rule["metric"], "identity_model_positive_state_rate")
            else:
                self.assertEqual(
                    rule["metric"], "production_review_eligible_state_rate"
                )

    def test_empty_queue_has_full_schema_and_exact_eligibility(self) -> None:
        score_path = self.application_root / self.app_outputs["real_candidate_scores"]
        queue_path = self.application_root / self.app_outputs["prospective_review_queue"]
        with score_path.open("r", encoding="utf-8-sig", newline="") as handle:
            score_header = next(csv.reader(handle))
        with queue_path.open("r", encoding="utf-8-sig", newline="") as handle:
            queue_header = next(csv.reader(handle))
        expected = [
            "queue_rank", "blind_id",
            *[name for name in score_header if name != "rank"],
            "review_status", "review_label", "review_notes",
        ]
        self.assertEqual(queue_header, expected)
        scores = base.load_csv(score_path)
        self.assertEqual(sum(int(row["production_review_eligible"]) for row in scores), 0)

    def test_blind_occurrence_schema_removes_model_context_flags(self) -> None:
        fake = {name: f"value-{name}" for name in scorer.BLIND_OCCURRENCE_FIELDS}
        fake.update(
            {
                "evidence_level": "derived",
                "seller_facing_context": "1",
                "product_data_risk_context": "0",
                "direct_identity_eligible": "1",
                "support_only": "0",
                "context": "derived",
            }
        )
        observed = scorer.occurrence_evidence(fake)
        self.assertEqual(list(observed), list(scorer.BLIND_OCCURRENCE_FIELDS))
        self.assertTrue(set(observed).isdisjoint(set(fake) - set(observed)))

    def test_current_application_is_separate_empty_abstention(self) -> None:
        training = base.load_json(
            self.training_root / self.train_outputs["training_summary"]
        )
        real = base.load_json(
            self.application_root / self.app_outputs["real_scoring_summary"]
        )
        self.assertEqual(training["decision"], "GO")
        self.assertEqual(real["positive_identity_correction_count"], 0)
        self.assertEqual(real["prospective_review_queue_count"], 0)
        self.assertEqual(real["real_candidate_rows_used_for_model_fitting_selection_or_gating"], 0)

    def test_sync_manifest_is_closed_and_hashes_match(self) -> None:
        manifest_path = self.application_root / self.app_outputs["sync_manifest"]
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.assertTrue(manifest["complete"])
        self.assertEqual(manifest["artifact_count"], len(manifest["artifacts"]))
        for record in manifest["artifacts"]:
            path = base.resolve(record["path"])
            self.assertTrue(path.is_file(), path)
            self.assertEqual(path.stat().st_size, int(record["size_bytes"]), path)
            self.assertEqual(base.sha256_file(path), record["sha256"], path)


if __name__ == "__main__":
    unittest.main()
