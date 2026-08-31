from __future__ import annotations

import hashlib
import json
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import step28_v13_v1_13_v9_4_1_model_training_common_v2 as common
import step28_v13_v1_13_v9_4_1_replay_english_151_v2 as replay


RESULT_ROOT = (
    ROOT
    / "reports"
    / "step28_model_experiment"
    / "v9_4_1_training_v2_20260830"
    / "english_151_replay_attempt2"
)
EXPECTED_MANIFEST_SIZE = 6696
EXPECTED_MANIFEST_SHA256 = (
    "18d6c8066404fe82437eb07a2d6fa3394867e7e786e87b40f690b60164ab922d"
)
EXPECTED_MANIFEST_SELF_HASH = (
    "6536c11f38dd6552490a4b720aa95c55db61040c8147daab5ee98c21c8037008"
)
EXPECTED_PREDICTION_SIZE = 8089
EXPECTED_PREDICTION_SHA256 = (
    "d6d6d780b71209b2254db5e145b895523d9617d9597917cffb6ebb28152a148b"
)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class English151V2ResultContracts(unittest.TestCase):
    def test_formal_result_bytes_and_self_hash_are_exact(self) -> None:
        manifest_path = RESULT_ROOT / replay.MANIFEST_NAME
        prediction_path = RESULT_ROOT / replay.PREDICTIONS_NAME
        self.assertEqual(manifest_path.stat().st_size, EXPECTED_MANIFEST_SIZE)
        self.assertEqual(sha256_file(manifest_path), EXPECTED_MANIFEST_SHA256)
        self.assertEqual(prediction_path.stat().st_size, EXPECTED_PREDICTION_SIZE)
        self.assertEqual(sha256_file(prediction_path), EXPECTED_PREDICTION_SHA256)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        unsigned = dict(manifest)
        claimed = unsigned.pop("canonical_self_hash")
        self.assertEqual(claimed, EXPECTED_MANIFEST_SELF_HASH)
        self.assertEqual(common.canonical_sha256(unsigned), claimed)

    def test_read_only_validator_reopens_semantic_outputs_without_scoring(self) -> None:
        policy = common.load_policy()
        manifest = replay.validate_output(policy)
        self.assertEqual(manifest["status"], replay.STATUS)
        self.assertTrue(manifest["all_four_exact_matches"])
        self.assertEqual(manifest["valid_pair_count"], 151)
        self.assertFalse(manifest["model_parameters_updated"])
        self.assertFalse(manifest["model_training_or_threshold_selection_performed"])
        for field in (
            "labels_or_identity_evidence_read",
            "controller_or_membership_read",
            "qrels_or_retrieval_truth_read",
            "audit_truth_read",
        ):
            self.assertEqual(manifest[field], 0)

    def test_failed_prepublication_path_is_absent_and_not_reused(self) -> None:
        old_path = RESULT_ROOT.parent / "english_151_replay"
        self.assertFalse(old_path.exists())
        policy = common.load_policy()
        invalidated = policy["invalidated_prepublication_attempts"][0]
        self.assertFalse(invalidated["payload_retained"])
        self.assertFalse(invalidated["path_reuse_allowed"])


if __name__ == "__main__":
    unittest.main()
