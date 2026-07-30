from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import step28_v13_build_fixed_sample_sensitivity as sensitivity  # noqa: E402
import step28_v13_common as common  # noqa: E402


class FixedSampleSensitivityContracts(unittest.TestCase):
    def test_registered_artifact_replays_exactly_and_is_not_power_claim(self) -> None:
        path = (
            ROOT
            / "reports"
            / "step28_synthetic_chinese_dataset"
            / "design_preflights"
            / "training_ready_fixed_sample_sensitivity_v1_20260730.json"
        )
        observed = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(observed, sensitivity.build_artifact())
        self_hash = observed.pop("canonical_self_hash")
        self.assertEqual(self_hash, common.canonical_sha256(observed))
        self.assertFalse(
            observed["claim_boundary"]["confirmatory_power_certified"]
        )
        self.assertTrue(
            observed["claim_boundary"][
                "binary_success_or_failure_from_power_is_forbidden"
            ]
        )
        self.assertTrue(
            observed["parent_confirmatory_power_contract"][
                "old_5000_replicate_monte_carlo_claimed"
            ]
            is False
        )

    def test_mde_increases_with_unknown_world_standard_deviation(self) -> None:
        artifact = sensitivity.build_artifact()
        for comparison in artifact[
            "normal_approximation_sensitivity"
        ]["comparisons"].values():
            detectable = [
                row["minimum_detectable_true_difference"]
                for row in comparison["grid"]
            ]
            self.assertEqual(detectable, sorted(detectable))
            self.assertEqual(len(set(detectable)), len(detectable))

    def test_invalid_sensitivity_parameter_fails_closed(self) -> None:
        with self.assertRaises(common.ContractError):
            sensitivity._minimum_detectable_difference(
                paired_world_standard_deviation=0.0,
                margin=0.03,
                world_count=500,
                familywise_alpha=0.05,
                comparison_count=7,
                target_power=0.8,
            )


if __name__ == "__main__":
    unittest.main()
