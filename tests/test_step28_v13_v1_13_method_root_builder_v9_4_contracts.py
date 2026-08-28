from __future__ import annotations

import importlib
import json
from pathlib import Path
import re
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

builder = importlib.import_module("step28_v13_v1_13_method_root_builder_v9_4")


class MethodRootBuilderV94Contracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.policy = json.loads(builder.POLICY_PATH.read_text(encoding="utf-8"))
        cls.base_policy = json.loads(builder.BASE_POLICY_PATH.read_text(encoding="utf-8"))
        cls.template = json.loads(builder.TEMPLATE_PATH.read_text(encoding="utf-8"))
        cls.signatures = [
            dict(row) for row in builder.noise_v94.build_noise_signatures().rows
        ]

    def test_policy_keeps_training_and_truth_unsealing_closed(self) -> None:
        self.assertEqual(
            self.policy["authorization"],
            {
                "method_root_build": False,
                "training_qualification": False,
                "audit_truth_unsealing": False,
                "model_training": False,
            },
        )
        self.assertEqual(sum(self.policy["world_counts"].values()), 1004)

    def test_visible_templates_have_no_artificial_code_or_internal_marker(self) -> None:
        values = [*builder.TITLE_PATTERNS, *builder.DESCRIPTION_PATTERNS]
        for value in values:
            self.assertNotRegex(value, builder.ARTIFICIAL_CODE_RE)
            self.assertNotRegex(value, builder.FORBIDDEN_VISIBLE_RE)
            self.assertNotIn("{code}", value)

    def test_mechanism_slot_counts_match_frozen_graphs(self) -> None:
        expected = self.base_policy["identity_design"]["mechanism_assignments"]
        for graph in ("G_A", "G_B"):
            observed: dict[str, int] = {}
            for mechanism in (
                *builder.MECHANISM_SLOTS[graph][3],
                *builder.MECHANISM_SLOTS[graph][2],
            ):
                observed[mechanism] = observed.get(mechanism, 0) + 1
            self.assertEqual(observed, expected[graph])

    def test_one_world_closes_parser_replay_truth_and_identity33(self) -> None:
        world = builder._smoke_world("train")
        value = builder.build_one_world(
            world=world,
            auth=builder.smoke_authorities(),
            base_policy=self.base_policy,
            template=self.template,
            signatures=self.signatures,
        )
        self.assertEqual(len(value["sellers"]), 28)
        self.assertEqual(len(value["endpoints"]), 378)
        self.assertEqual(len(value["identity33"]), 378)
        self.assertEqual(sum(row["label"] for row in value["labels"]), 20)
        self.assertEqual(
            len(value["labels"]) - sum(row["label"] for row in value["labels"]),
            358,
        )
        visible = "\n".join(
            str(row["title"]) + "\n" + str(row["description"])
            for row in value["items"]
        )
        self.assertIsNone(re.search(r"Q[A-P]{10}", visible))
        self.assertIsNone(builder.FORBIDDEN_VISIBLE_RE.search(visible))
        self.assertTrue(value["parser_rows"])
        self.assertGreater(value["generation_audit"]["identity33_active_pair_count"], 0)
        controls = value["generation_audit"]["registered_negative_controls"]
        self.assertEqual(
            [row["control_type"] for row in controls].count(
                "exact_title_clone_negative"
            ),
            2,
        )
        self.assertEqual(
            [row["control_type"] for row in controls].count(
                "high_semantic_similarity_negative"
            ),
            4,
        )


if __name__ == "__main__":
    unittest.main()
