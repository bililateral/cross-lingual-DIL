from __future__ import annotations

import copy
import hashlib
import json
import sys
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import step28_v13_v1_13_method_root_quality_attempt2_v9_4 as quality
import step28_v13_v1_13_quality_text_probe_views_v9 as text_views


class MethodRootQualityAttempt2V94Contracts(unittest.TestCase):
    def test_policy_freezes_complete_c_amendment_family(self) -> None:
        policy = quality.verify_policy()
        family = policy["text_probe_family"]
        self.assertEqual(tuple(family["view_names"]), text_views.VIEW_ORDER)
        self.assertEqual(tuple(family["feature_widths"]), text_views.EXPECTED_WIDTHS)
        self.assertEqual(sum(family["feature_widths"]), 346)
        self.assertEqual(family["total_model_count"], 14)
        self.assertEqual(policy["bootstrap"]["replicates"], 9999)
        self.assertEqual(policy["splits"]["eligible_pairs_per_world"], 372)
        self.assertEqual(policy["splits"]["positive_pairs_per_world"], 20)

    def test_policy_self_hash_is_canonical(self) -> None:
        path = quality.POLICY_PATH
        policy = json.loads(path.read_text(encoding="utf-8"))
        claimed = policy.pop("canonical_self_hash")
        observed = hashlib.sha256(quality.canonical_bytes(policy)).hexdigest()
        self.assertEqual(claimed, observed)

    def test_identity_positive_control_cannot_open_private_truth(self) -> None:
        parameters = quality.identity_positive_control.__code__.co_varnames[
            : quality.identity_positive_control.__code__.co_argcount
            + quality.identity_positive_control.__code__.co_kwonlyargcount
        ]
        self.assertNotIn("private_root", parameters)
        source = Path(quality.__file__).read_text(encoding="utf-8")
        function = source.split("def identity_positive_control(", 1)[1].split(
            "\ndef exact_v94_public_replay(", 1
        )[0]
        self.assertNotIn("pair_labels.csv", function)
        self.assertIn("truth_indexes", function)

    def test_counterfactual_second_replay_mutation_is_rejected(self) -> None:
        first = {
            "counterfactual_redacted": [{"x": 1}],
            "counterfactual_profiles": [{"x": 2}],
            "counterfactual_parsed": [{"x": 3}],
            "counterfactual_history": [{"x": 4}],
            "counterfactual_identity33": [{"x": 5}],
            "parser_audit": {"x": 6},
            "derangement_mapping_sha256": "a" * 64,
            "excluded_pair_uids": ("p",),
        }
        quality.assert_counterfactual_replay(first, copy.deepcopy(first))
        changed = copy.deepcopy(first)
        changed["counterfactual_profiles"][0]["x"] = 7
        with self.assertRaisesRegex(
            quality.MethodRootQualityAttempt2Error,
            "Counterfactual independent replay drift",
        ):
            quality.assert_counterfactual_replay(first, changed)


if __name__ == "__main__":
    unittest.main()
