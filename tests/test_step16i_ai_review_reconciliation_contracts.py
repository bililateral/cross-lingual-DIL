from __future__ import annotations

import csv
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import step16i_reconcile_ai_sensitivity_reviews as reconcile  # noqa: E402


FIELDS = [
    "review_index",
    "blind_id",
    "review_scope",
    "raw_evidence",
    "independent_identity_decision",
    "evidence_type_decision",
    "review_confidence",
    "review_rationale",
]


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class Step16IAIReviewReconciliationContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(dir=ROOT)
        self.root = Path(self.temporary.name)
        self.policy_path = self.root / "policy.json"
        self.queue_a_path = self.root / "reviewer_a_queue.csv"
        self.queue_b_path = self.root / "reviewer_b_queue.csv"
        self.completed_a_path = self.root / "reviewer_a_completed.csv"
        self.completed_b_path = self.root / "reviewer_b_completed.csv"
        self.manifest_path = self.root / "preparation_manifest.json"
        self.output_path = self.root / "reconciliation_output"

        self.policy = {
            "blinding": {
                "allowed_identity_decisions": [
                    "same_controller",
                    "different_controller",
                    "uncertain",
                ],
                "allowed_evidence_type_decisions": [
                    "same_controller_direct_identifier",
                    "template_clone_not_controller",
                    "semantic_topic_not_controller",
                    "uncertain_insufficient_evidence",
                ],
            }
        }
        self.policy_path.write_text(json.dumps(self.policy), encoding="utf-8")

        self.queue_a = [
            self.queue_row("1", "blind-1", "evidence one"),
            self.queue_row("2", "blind-2", "evidence two"),
            self.queue_row("3", "blind-3", "evidence three"),
        ]
        self.queue_b = [
            self.queue_row("1", "blind-3", "evidence three"),
            self.queue_row("2", "blind-1", "evidence one"),
            self.queue_row("3", "blind-2", "evidence two"),
        ]
        write_csv(self.queue_a_path, self.queue_a)
        write_csv(self.queue_b_path, self.queue_b)
        self.write_manifest()

        decisions_a = {
            "blind-1": (
                "same_controller",
                "same_controller_direct_identifier",
                "high",
                "A direct evidence",
            ),
            "blind-2": (
                "different_controller",
                "template_clone_not_controller",
                "medium",
                "A template evidence",
            ),
            "blind-3": (
                "uncertain",
                "uncertain_insufficient_evidence",
                "high",
                "A insufficient evidence",
            ),
        }
        decisions_b = {
            "blind-1": (
                "same_controller",
                "same_controller_direct_identifier",
                "high",
                "B direct evidence",
            ),
            "blind-2": (
                "different_controller",
                "semantic_topic_not_controller",
                "high",
                "B semantic evidence",
            ),
            "blind-3": (
                "uncertain",
                "uncertain_insufficient_evidence",
                "low",
                "B insufficient evidence",
            ),
        }
        self.completed_a = self.complete(self.queue_a, decisions_a)
        self.completed_b = self.complete(self.queue_b, decisions_b)
        write_csv(self.completed_a_path, self.completed_a)
        write_csv(self.completed_b_path, self.completed_b)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    @staticmethod
    def queue_row(index: str, blind_id: str, evidence: str) -> dict[str, str]:
        return {
            "review_index": index,
            "blind_id": blind_id,
            "review_scope": "retrospective_development_candidate_only",
            "raw_evidence": evidence,
            "independent_identity_decision": "",
            "evidence_type_decision": "",
            "review_confidence": "",
            "review_rationale": "",
        }

    @staticmethod
    def complete(
        rows: list[dict[str, str]], decisions: dict[str, tuple[str, str, str, str]]
    ) -> list[dict[str, str]]:
        completed = []
        for source in rows:
            row = dict(source)
            identity, evidence, confidence, rationale = decisions[row["blind_id"]]
            row["independent_identity_decision"] = identity
            row["evidence_type_decision"] = evidence
            row["review_confidence"] = confidence
            row["review_rationale"] = rationale
            completed.append(row)
        return completed

    def relative(self, path: Path) -> str:
        return str(path.resolve().relative_to(ROOT.resolve())).replace("\\", "/")

    def write_manifest(self) -> None:
        manifest = {
            "prospective_claim_allowed": False,
            "automatic_identity_labels_assigned": False,
            "inputs": {
                "policy": {
                    "path": self.relative(self.policy_path),
                    "sha256": file_hash(self.policy_path),
                }
            },
            "outputs": {
                "reviewer_a_queue": {
                    "path": self.relative(self.queue_a_path),
                    "sha256": file_hash(self.queue_a_path),
                    "row_count": len(self.queue_a),
                },
                "reviewer_b_queue": {
                    "path": self.relative(self.queue_b_path),
                    "sha256": file_hash(self.queue_b_path),
                    "row_count": len(self.queue_b),
                },
            },
        }
        self.manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    def run_reconciliation(self, output: Path | None = None) -> dict:
        return reconcile.reconcile(
            reviewer_a_queue=self.queue_a_path,
            reviewer_b_queue=self.queue_b_path,
            reviewer_a_completed=self.completed_a_path,
            reviewer_b_completed=self.completed_b_path,
            preparation_manifest=self.manifest_path,
            policy=self.policy_path,
            output_directory=output or self.output_path,
        )

    def test_evidence_column_tampering_is_rejected(self) -> None:
        self.completed_a[0]["raw_evidence"] = "tampered evidence"
        write_csv(self.completed_a_path, self.completed_a)
        with self.assertRaisesRegex(ValueError, "changed immutable field raw_evidence"):
            self.run_reconciliation()

    def test_invalid_decision_is_rejected(self) -> None:
        self.completed_b[0]["independent_identity_decision"] = "probably_same"
        write_csv(self.completed_b_path, self.completed_b)
        with self.assertRaisesRegex(ValueError, "invalid identity decision"):
            self.run_reconciliation()

    def test_original_queue_hash_change_is_rejected(self) -> None:
        self.queue_a[0]["raw_evidence"] = "queue changed after preparation"
        write_csv(self.queue_a_path, self.queue_a)
        with self.assertRaisesRegex(ValueError, "hash does not match"):
            self.run_reconciliation()

    def test_completed_blind_id_order_change_is_rejected(self) -> None:
        self.completed_a[0], self.completed_a[1] = self.completed_a[1], self.completed_a[0]
        write_csv(self.completed_a_path, self.completed_a)
        with self.assertRaisesRegex(ValueError, "blind_id order changed"):
            self.run_reconciliation()

    def test_agreement_counts_and_scope_are_correct(self) -> None:
        summary = self.run_reconciliation()
        self.assertEqual(summary["scope"], "ai_sensitivity_only_not_human_gold")
        self.assertIs(summary["automatic_labels_created"], False)
        self.assertIs(summary["step5_modified"], False)
        self.assertIs(summary["prospective_claim_allowed"], False)
        self.assertIs(summary["validation_contract"]["blind_mapping_read"], False)
        self.assertIs(summary["validation_contract"]["label_inference_performed"], False)
        self.assertEqual(summary["counts"]["reviewed"], 3)
        self.assertEqual(summary["counts"]["exact_identity_agreement"], 3)
        self.assertEqual(summary["counts"]["exact_evidence_agreement"], 2)
        self.assertEqual(
            summary["counts"]["exact_identity_and_evidence_agreement"], 2
        )
        self.assertEqual(summary["counts"]["high_confidence_exact_agreement"], 1)
        self.assertEqual(summary["counts"]["needs_human_adjudication"], 2)

        with (self.output_path / "agreement_by_blind_id.csv").open(
            "r", encoding="utf-8-sig", newline=""
        ) as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual([row["blind_id"] for row in rows], ["blind-1", "blind-2", "blind-3"])
        self.assertEqual(rows[0]["high_confidence_exact_agreement"], "true")
        self.assertEqual(rows[1]["exact_identity_agreement"], "true")
        self.assertEqual(rows[1]["exact_evidence_agreement"], "false")
        self.assertEqual(rows[2]["needs_human_adjudication"], "true")
        emitted = (self.output_path / "agreement_by_blind_id.csv").read_text(
            encoding="utf-8-sig"
        ) + (self.output_path / "summary.json").read_text(encoding="utf-8")
        self.assertNotIn("pair_uid", emitted)
        self.assertNotIn("seller_uid", emitted)

    def test_existing_output_directory_is_never_overwritten(self) -> None:
        self.run_reconciliation()
        with self.assertRaisesRegex(FileExistsError, "Refusing to overwrite"):
            self.run_reconciliation()

    def test_blind_mapping_cannot_be_supplied_as_an_input(self) -> None:
        blind_mapping = self.root / "blind_mapping.csv"
        blind_mapping.write_text("blind_id\nblind-1\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "must never read blind_mapping"):
            reconcile.reconcile(
                reviewer_a_queue=blind_mapping,
                reviewer_b_queue=self.queue_b_path,
                reviewer_a_completed=self.completed_a_path,
                reviewer_b_completed=self.completed_b_path,
                preparation_manifest=self.manifest_path,
                policy=self.policy_path,
                output_directory=self.output_path,
            )


if __name__ == "__main__":
    unittest.main()
