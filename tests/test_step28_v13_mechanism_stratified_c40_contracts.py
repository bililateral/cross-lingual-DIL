from __future__ import annotations

import copy
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import step28_v13_common as common  # noqa: E402
import step28_v13_generate_dataset as generator  # noqa: E402
import step28_v13_mechanism_stratified_c40 as c40  # noqa: E402
import step28_v13_structure as structure  # noqa: E402
import step28_v13_world_builder as world_builder  # noqa: E402


class Step28V13MechanismStratifiedC40Contracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.policy = common.load_policy(mode="development_smoke")
        template, fixture, style = generator._load_release_inputs(
            cls.policy,
            mode="development_smoke",
        )
        record = structure.build_mode_world_pool(
            cls.policy,
            mode="development_smoke",
        )[0]
        cls.world = world_builder.build_world(
            policy=cls.policy,
            template=template,
            fixture=fixture,
            style_profile=style,
            mode="development_smoke",
            world_record=record,
            structure_key_hex=common.structure_key_for_split(
                cls.policy,
                mode="development_smoke",
                split=record["split"],
            ),
        )
        cls.world_uid = str(record["world_uid"])
        cls.key = str(
            cls.policy["randomness"]["formal"]["candidate_key_hex"]
        )

    def build(self, split: str = "train"):
        return c40.build_world_c40(
            split=split,
            candidate_key_hex=self.key,
            world_uid=self.world_uid,
            complete_pair_endpoints=self.world["public"][
                "complete_model_pair_endpoints"
            ],
            controller_membership=self.world["private"][
                "controller_membership"
            ],
            positive_targets=self.world["private"]["positive_targets"],
            negative_flags=self.world["private"]["negative_flags"],
        )

    def test_exact_class_budgets_and_safe_projection(self) -> None:
        for split, positive_count in (
            ("train", 16),
            ("development", 10),
            ("audit_a", 10),
            ("audit_b", 10),
        ):
            rows, audit, summary = self.build(split)
            self.assertEqual(len(rows), 40)
            self.assertEqual(len(audit), 40)
            self.assertTrue(
                all(tuple(row) == c40.SAFE_FIELDS for row in rows)
            )
            self.assertEqual(summary["positive_count"], positive_count)
            self.assertEqual(summary["negative_count"], 40 - positive_count)
            self.assertTrue(summary["all_positive_mechanisms_covered"])
            self.assertTrue(summary["all_negative_flags_covered"])
            self.assertFalse(summary["model_visible_sampling_fields"])

    def test_input_order_does_not_change_output(self) -> None:
        expected = self.build("train")
        observed = c40.build_world_c40(
            split="train",
            candidate_key_hex=self.key,
            world_uid=self.world_uid,
            complete_pair_endpoints=list(
                reversed(
                    self.world["public"][
                        "complete_model_pair_endpoints"
                    ]
                )
            ),
            controller_membership=list(
                reversed(
                    self.world["private"]["controller_membership"]
                )
            ),
            positive_targets=list(
                reversed(self.world["private"]["positive_targets"])
            ),
            negative_flags=list(
                reversed(self.world["private"]["negative_flags"])
            ),
        )
        self.assertEqual(observed, expected)

    def test_selected_rows_use_independent_global_output_order(self) -> None:
        rows, audit, _summary = self.build("train")
        expected = sorted(
            (row["canonical_pair_uid"] for row in rows),
            key=lambda pair_uid: (
                common.hmac_digest(
                    self.key,
                    self.world_uid,
                    "selected_global_rank",
                    pair_uid,
                ),
                pair_uid.encode("utf-8"),
            ),
        )
        self.assertEqual(
            expected,
            [row["canonical_pair_uid"] for row in rows],
        )
        self.assertEqual(
            list(range(1, 41)),
            [int(row["selected_rank"]) for row in audit],
        )
        for row in audit:
            self.assertEqual(
                row["hmac_digest_hex"],
                c40._rank_key(
                    key_hex=self.key,
                    world_uid=self.world_uid,
                    pair_uid=row["canonical_pair_uid"],
                )[0].hex(),
            )

    def test_tampered_target_label_fails_closed(self) -> None:
        targets = copy.deepcopy(self.world["private"]["positive_targets"])
        negative_pair = self.world["private"]["negative_flags"][0][
            "canonical_pair_uid"
        ]
        targets[0]["canonical_pair_uid"] = negative_pair
        with self.assertRaises(common.ContractError):
            c40.build_world_c40(
                split="train",
                candidate_key_hex=self.key,
                world_uid=self.world_uid,
                complete_pair_endpoints=self.world["public"][
                    "complete_model_pair_endpoints"
                ],
                controller_membership=self.world["private"][
                    "controller_membership"
                ],
                positive_targets=targets,
                negative_flags=self.world["private"]["negative_flags"],
            )

    def test_renamed_mechanism_or_flag_fails_closed(self) -> None:
        targets = copy.deepcopy(self.world["private"]["positive_targets"])
        targets[0]["mechanism"] = "invented_mechanism"
        with self.assertRaises(common.ContractError):
            c40.build_world_c40(
                split="train",
                candidate_key_hex=self.key,
                world_uid=self.world_uid,
                complete_pair_endpoints=self.world["public"][
                    "complete_model_pair_endpoints"
                ],
                controller_membership=self.world["private"][
                    "controller_membership"
                ],
                positive_targets=targets,
                negative_flags=self.world["private"]["negative_flags"],
            )
        flags = copy.deepcopy(self.world["private"]["negative_flags"])
        flags[0]["flag"] = "invented_flag"
        with self.assertRaises(common.ContractError):
            c40.build_world_c40(
                split="train",
                candidate_key_hex=self.key,
                world_uid=self.world_uid,
                complete_pair_endpoints=self.world["public"][
                    "complete_model_pair_endpoints"
                ],
                controller_membership=self.world["private"][
                    "controller_membership"
                ],
                positive_targets=self.world["private"][
                    "positive_targets"
                ],
                negative_flags=flags,
            )

    def test_extra_or_missing_input_fields_fail_closed(self) -> None:
        memberships = copy.deepcopy(
            self.world["private"]["controller_membership"]
        )
        memberships[0]["label"] = 1
        with self.assertRaises(common.ContractError):
            c40.build_world_c40(
                split="train",
                candidate_key_hex=self.key,
                world_uid=self.world_uid,
                complete_pair_endpoints=self.world["public"][
                    "complete_model_pair_endpoints"
                ],
                controller_membership=memberships,
                positive_targets=self.world["private"][
                    "positive_targets"
                ],
                negative_flags=self.world["private"]["negative_flags"],
            )
        targets = copy.deepcopy(self.world["private"]["positive_targets"])
        targets[0].pop("mechanism_slot_uid")
        with self.assertRaises(common.ContractError):
            c40.build_world_c40(
                split="train",
                candidate_key_hex=self.key,
                world_uid=self.world_uid,
                complete_pair_endpoints=self.world["public"][
                    "complete_model_pair_endpoints"
                ],
                controller_membership=self.world["private"][
                    "controller_membership"
                ],
                positive_targets=targets,
                negative_flags=self.world["private"]["negative_flags"],
            )


if __name__ == "__main__":
    unittest.main()
