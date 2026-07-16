from __future__ import annotations

import csv
import hashlib
import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
ADJUDICATION_ROOT = (
    ROOT / "reports" / "step16i_retrospective_dev2" / "codex_adjudication_v1_20260716"
)
DEV2_ROOT = (
    ROOT / "reports" / "step16i_retrospective_dev2" / "codex_adjudicated_dev2_v1_20260716"
)
EXCLUSION_PATH = (
    ROOT
    / "reports"
    / "step16i_data_integrity"
    / "step16i_integrity_20260716_v2"
    / "permanent_exclusion_manifest.csv"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


class Step16ICodexAdjudicationContractTests(unittest.TestCase):
    def test_adjudication_is_owner_authorized_but_not_human_gold(self) -> None:
        summary = json.loads((ADJUDICATION_ROOT / "summary.json").read_text(encoding="utf-8"))
        self.assertEqual(summary["row_count"], 160)
        self.assertEqual(
            summary["identity_counts"],
            {"different_controller": 108, "same_controller": 1, "uncertain": 51},
        )
        self.assertTrue(summary["dataset_owner_authorized"])
        self.assertFalse(summary["human_verified_per_row"])
        self.assertFalse(summary["may_be_described_as_two_independent_human_reviews"])
        self.assertFalse(summary["prospective_claim_allowed"])
        self.assertFalse(summary["step5_labels_created_or_modified"])

    def test_frozen_adjudication_hash_and_positive_are_exact(self) -> None:
        summary = json.loads((ADJUDICATION_ROOT / "summary.json").read_text(encoding="utf-8"))
        path = ADJUDICATION_ROOT / "adjudication_by_blind_id.csv"
        self.assertEqual(sha256(path), summary["outputs"]["adjudication_by_blind_id"]["sha256"])
        adjudicated = rows(path)
        self.assertEqual(len(adjudicated), 160)
        self.assertEqual(len({row["blind_id"] for row in adjudicated}), 160)
        positives = [row for row in adjudicated if row["adjudicated_identity_decision"] == "same_controller"]
        self.assertEqual(len(positives), 1)
        self.assertEqual(positives[0]["blind_id"], "retdev2_69bc5acce819b33653f3")
        self.assertEqual(positives[0]["adjudicated_evidence_type"], "same_controller_direct_identifier")
        self.assertTrue(all(row["human_verified_per_row"] == "false" for row in adjudicated))
        self.assertTrue(all(row["step5_supervision_eligible"] == "false" for row in adjudicated))

    def test_materialized_dev2_is_isolated_and_hash_bound(self) -> None:
        summary = json.loads((DEV2_ROOT / "summary.json").read_text(encoding="utf-8"))
        path = DEV2_ROOT / "retrospective_dev2_labels.csv"
        self.assertEqual(sha256(path), summary["outputs"]["retrospective_dev2_labels"]["sha256"])
        materialized = rows(path)
        self.assertEqual(len(materialized), 160)
        self.assertEqual(len({row["pair_uid"] for row in materialized}), 160)
        self.assertEqual(len({row["candidate_component_id"] for row in materialized}), 160)
        self.assertEqual(summary["binary_evaluation_row_count"], 109)
        self.assertEqual(summary["binary_positive_count"], 1)
        self.assertEqual(summary["binary_negative_count"], 108)
        self.assertFalse(summary["paper_primary_benchmark_eligible"])
        self.assertTrue(all(row["retrospective_development_only"] == "true" for row in materialized))
        self.assertTrue(all(row["prospective_final_eligible"] == "false" for row in materialized))
        self.assertTrue(all(row["step5_supervision_eligible"] == "false" for row in materialized))

    def test_materialized_dev2_does_not_overlap_permanent_excluded_pairs(self) -> None:
        excluded_pairs = {
            row["pair_uid"]
            for row in rows(EXCLUSION_PATH)
            if row.get("pair_uid", "").strip()
        }
        materialized_pairs = {
            row["pair_uid"] for row in rows(DEV2_ROOT / "retrospective_dev2_labels.csv")
        }
        self.assertFalse(materialized_pairs & excluded_pairs)


if __name__ == "__main__":
    unittest.main()
