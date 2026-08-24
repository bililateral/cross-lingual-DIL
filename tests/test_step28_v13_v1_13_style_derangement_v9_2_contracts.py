#!/usr/bin/env python3
"""Direct contracts for the public-ID-only V9.2 style mapping generator."""

from __future__ import annotations

import inspect
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import step28_v13_v1_13_style_derangement as subject


SELLERS = tuple(f"seller_{index:02d}" for index in range(28))


class StyleDerangementV92Contracts(unittest.TestCase):
    def test_signature_is_public_id_only(self) -> None:
        self.assertEqual(
            set(inspect.signature(subject.build_style_source_derangement).parameters),
            {"split", "world_uid", "seller_uids"},
        )

    def test_frozen_fixture_digest_bijection_and_no_fixed_point(self) -> None:
        value = subject.build_style_source_derangement(
            split="train",
            world_uid="fixture_world_000",
            seller_uids=SELLERS,
        )
        self.assertEqual(value.attempt, 0)
        self.assertEqual(
            value.seller_set_sha256,
            "cfcd6214cc847bd706f3190a377fa2cce413c97eb9fdecd8bcd07d6be427e259",
        )
        self.assertEqual(
            value.mapping_sha256,
            "b3bb94be0e17d3b6d395f0d28cd648eaf33f16dc169956bd2a03395fbe37ce67",
        )
        mapping = value.as_mapping()
        self.assertEqual(set(mapping), set(SELLERS))
        self.assertEqual(set(mapping.values()), set(SELLERS))
        self.assertTrue(all(target != source for target, source in mapping.items()))

    def test_input_order_cannot_change_the_mapping(self) -> None:
        forward = subject.build_style_source_derangement(
            split="development",
            world_uid="fixture_world_001",
            seller_uids=SELLERS,
        )
        reversed_input = subject.build_style_source_derangement(
            split="development",
            world_uid="fixture_world_001",
            seller_uids=tuple(reversed(SELLERS)),
        )
        self.assertEqual(forward, reversed_input)

    def test_wrong_seller_cardinality_fails_closed(self) -> None:
        with self.assertRaisesRegex(subject.StyleDerangementError, "28 unique"):
            subject.build_style_source_derangement(
                split="train",
                world_uid="fixture_world_002",
                seller_uids=SELLERS[:-1],
            )


if __name__ == "__main__":
    unittest.main()
