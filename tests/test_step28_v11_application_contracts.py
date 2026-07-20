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


TRAINING_POLICY_PATH = ROOT / "schema" / "step28_transferable_identity_history_v11_policy.json"
APPLICATION_POLICY_PATH = ROOT / "schema" / "step28_transferable_identity_history_v11_1_guarded_application_policy.json"


@unittest.skip("v11 was withdrawn after final audit; v12 owns the current contracts")
class Step28V11ApplicationContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.training_policy = history.load_policy(TRAINING_POLICY_PATH)
        cls.application_policy = history.load_policy(APPLICATION_POLICY_PATH)
        cls.training_root = base.output_root(cls.training_policy)
        cls.application_root = base.output_root(cls.application_policy)
        cls.train_outputs = cls.training_policy["outputs"]
        cls.app_outputs = cls.application_policy["outputs"]

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

    def test_observable_state_support_recomputes_from_train_and_development(self) -> None:
        rows = base.load_csv(
            self.training_root / self.train_outputs["model_inputs"]
        )
        artifacts = base.load_json(
            self.training_root / self.train_outputs["model_artifacts"]
        )
        observed = scorer.recompute_observable_state_support(
            rows, self.training_policy["model"]["feature_names"]
        )
        self.assertEqual(
            observed, artifacts["primary_model"]["observable_state_support"]
        )

    def test_threshold_selection_excludes_train_ambiguous_states(self) -> None:
        rows = base.load_csv(
            self.training_root / self.train_outputs["model_inputs"]
        )
        names = self.training_policy["model"]["feature_names"]
        labels_by_split_state: dict[str, dict[str, set[str]]] = {}
        row_by_split_state: dict[str, dict[str, dict]] = {}
        for row in rows:
            values = np.asarray([float(row[name]) for name in names], dtype=float)
            state_hash = history.observable_state_hash(values)
            split = row["synthetic_split"]
            labels_by_split_state.setdefault(split, {}).setdefault(
                state_hash, set()
            ).add(row["review_label"])
            row_by_split_state.setdefault(split, {}).setdefault(state_hash, row)
        train_ambiguous = {
            state_hash
            for state_hash, labels in labels_by_split_state["synthetic_train"].items()
            if len(labels) > 1
        }
        selected = [
            state_hash
            for state_hash, labels in labels_by_split_state[
                "synthetic_development"
            ].items()
            if len(labels) == 1 and state_hash not in train_ambiguous
        ]
        artifacts = base.load_json(
            self.training_root / self.train_outputs["model_artifacts"]
        )
        model = artifacts["primary_model"]
        matrix = np.asarray(
            [
                [
                    float(row_by_split_state["synthetic_development"][state][name])
                    for name in names
                ]
                for state in selected
            ],
            dtype=float,
        )
        labels = np.asarray(
            [
                float(
                    next(
                        iter(
                            labels_by_split_state["synthetic_development"][state]
                        )
                    )
                    == "positive"
                )
                for state in selected
            ]
        )
        scores = base.sigmoid(history.identity_correction(matrix, model))
        threshold, _metrics = history.choose_threshold(labels, scores)
        self.assertAlmostEqual(threshold, artifacts["frozen_threshold"], places=12)
        training = base.load_json(
            self.training_root / self.train_outputs["training_summary"]
        )
        self.assertEqual(
            len(selected),
            training["observable_state_accounting"][
                "identifiable_development_states_used_for_threshold"
            ],
        )

    def test_blind_packet_has_no_model_outputs_or_pair_uid(self) -> None:
        blind_path = self.application_root / self.app_outputs["blind_evidence_packet"]
        with blind_path.open("r", encoding="utf-8-sig", newline="") as handle:
            header = next(csv.reader(handle))
        expected = [
            "blind_id",
            "seller_a_uid",
            "seller_b_uid",
            "seller_a_market",
            "seller_b_market",
            "seller_a_alias",
            "seller_b_alias",
            "shared_identity_evidence_json",
            "rotation_identity_evidence_json",
        ]
        self.assertEqual(header, expected)
        forbidden = (
            "score", "rank", "feature", "support", "correction", "probability", "pair_uid"
        )
        self.assertFalse(
            any(fragment in column.lower() for column in header for fragment in forbidden)
        )

    def test_current_outcome_is_zero_queue_abstention(self) -> None:
        training = base.load_json(
            self.training_root / self.train_outputs["training_summary"]
        )
        summary = base.load_json(
            self.application_root / self.app_outputs["real_scoring_summary"]
        )
        self.assertEqual(training["decision"], "GO")
        self.assertTrue(all(training["checks"].values()))
        self.assertEqual(summary["known_reviewed_pair_uid_remaining_in_universe_count"], 0)
        self.assertEqual(summary["positive_identity_correction_count"], 0)
        self.assertEqual(summary["prospective_review_queue_count"], 0)
        self.assertTrue(summary["review_queue_empty_is_valid_abstention"])

    def test_sync_manifest_is_closed_and_hashes_match(self) -> None:
        manifest_path = self.application_root / self.app_outputs["sync_manifest"]
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.assertTrue(manifest["complete"])
        self.assertEqual(manifest["artifact_count"], len(manifest["artifacts"]))
        paths = {record["path"] for record in manifest["artifacts"]}
        required = {
            "scripts/step28_common.py",
            "scripts/step28_history_common.py",
            "scripts/step28_generate_transferable_identity_histories.py",
            "scripts/step28_train_transferable_identity_model.py",
            "scripts/step28_score_real_identity_candidates.py",
            "scripts/step28_self_audit_v10.py",
            "tests/test_step28_v4_transferable_identity_history.py",
            "tests/test_step28_v11_application_contracts.py",
        }
        self.assertTrue(required <= paths)
        for record in manifest["artifacts"]:
            path = base.resolve(record["path"])
            self.assertTrue(path.is_file(), path)
            self.assertEqual(path.stat().st_size, int(record["size_bytes"]), path)
            self.assertEqual(base.sha256_file(path), record["sha256"], path)


if __name__ == "__main__":
    unittest.main()
