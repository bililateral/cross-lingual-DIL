from __future__ import annotations

import csv
import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent


def load_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


class Step16HBlindReviewContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.policy = json.loads(
            (ROOT / "schema" / "step16h_blind_positive_reaudit_policy.json").read_text(
                encoding="utf-8"
            )
        )
        outputs = cls.policy["outputs"]
        cls.rows_a = load_csv(ROOT / outputs["reviewer_a_queue"])
        cls.rows_b = load_csv(ROOT / outputs["reviewer_b_queue"])

    def test_queues_are_label_concealed_and_contain_raw_evidence(self) -> None:
        forbidden = {
            "review_label",
            "review_stratum",
            "evidence_type",
            "paper_evidence_tier",
            "recommended_use",
            "confidence",
            "rationale",
            "reviewer_id",
            "review_notes",
            "original_review_notes",
            "prob_positive",
        }
        fields = set(self.rows_a[0])
        self.assertFalse(fields & forbidden)
        self.assertIn("left_preview", fields)
        self.assertIn("right_preview", fields)
        self.assertIn("shared_contact_values", fields)

    def test_queues_share_the_same_160_pair_universe_in_different_orders(self) -> None:
        self.assertEqual(len(self.rows_a), 160)
        self.assertEqual(len(self.rows_b), 160)
        uids_a = [row["pair_uid"] for row in self.rows_a]
        uids_b = [row["pair_uid"] for row in self.rows_b]
        self.assertEqual(len(set(uids_a)), 160)
        self.assertEqual(set(uids_a), set(uids_b))
        self.assertNotEqual(uids_a, uids_b)

    def test_concealed_reference_balance_is_80_positive_and_80_negative(self) -> None:
        label_rows = load_csv(ROOT / self.policy["inputs"]["frozen_labels"])
        labels = {row["pair_uid"]: row["review_label"] for row in label_rows}
        counts = {"positive": 0, "negative": 0}
        for row in self.rows_a:
            counts[labels[row["pair_uid"]]] += 1
        self.assertEqual(counts, {"positive": 80, "negative": 80})

    def test_v3_uses_opaque_keys_and_keeps_mapping_out_of_reviewer_queues(self) -> None:
        policy = json.loads(
            (ROOT / "schema" / "step16h_blind_positive_reaudit_v3_policy.json").read_text(
                encoding="utf-8"
            )
        )
        rows_a = load_csv(ROOT / policy["outputs"]["reviewer_a_queue"])
        rows_b = load_csv(ROOT / policy["outputs"]["reviewer_b_queue"])
        mapping = load_csv(ROOT / policy["outputs"]["blind_mapping"])
        reviewer_fields = set(rows_a[0])
        self.assertIn("blind_id", reviewer_fields)
        self.assertNotIn("pair_uid", reviewer_fields)
        self.assertNotIn("reference_subset", reviewer_fields)
        self.assertNotIn("direct_identity_eligible", reviewer_fields)
        self.assertEqual(len(rows_a), 160)
        self.assertEqual(len(rows_b), 160)
        self.assertEqual({row["blind_id"] for row in rows_a}, {row["blind_id"] for row in rows_b})
        self.assertEqual({row["blind_id"] for row in rows_a}, {row["blind_id"] for row in mapping})
        allowed = set(policy["allowed_decisions"])
        self.assertTrue(
            all(row["independent_decision"] in allowed for row in rows_a + rows_b)
        )
        self.assertTrue(
            all(
                row["review_confidence"] and row["review_rationale"]
                for row in rows_a + rows_b
            )
        )

    def test_v3_exposes_raw_occurrences_and_candidate_paths_without_old_labels(self) -> None:
        policy = json.loads(
            (ROOT / "schema" / "step16h_blind_positive_reaudit_v3_policy.json").read_text(
                encoding="utf-8"
            )
        )
        rows = load_csv(ROOT / policy["outputs"]["reviewer_a_queue"])
        raw_occurrence_rows = [
            row for row in rows if json.loads(row["raw_contact_occurrences_json"])
        ]
        component_path_rows = [
            row for row in rows if json.loads(row["component_candidate_path_json"])
        ]
        self.assertGreater(len(raw_occurrence_rows), 0)
        self.assertGreater(len(component_path_rows), 0)
        forbidden_fragments = ("review_label", "paper_evidence_tier", "prob_positive")
        self.assertFalse(
            any(
                fragment in field
                for field in rows[0]
                for fragment in forbidden_fragments
            )
        )


if __name__ == "__main__":
    unittest.main()
