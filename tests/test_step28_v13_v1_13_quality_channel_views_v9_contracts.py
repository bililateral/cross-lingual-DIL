from __future__ import annotations

import inspect
import copy
import hashlib
import json
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import step28_v13_v1_13_quality_channel_views_v9 as channel
import step28_v13_v1_13_quality_channel_policy_v9 as channel_policy


CODE_A = "QAAAAAAAAAA"
CODE_B = "QBBBBBBBBBB"
CODE_C = "QCCCCCCCCCC"
CODE_D = "QDDDDDDDDDD"
CODE_E = "QEEEEEEEEEE"
CODE_F = "QFFFFFFFFFF"


def occurrence(
    code: str,
    field: str,
    *,
    own: bool,
) -> channel.CodeOccurrence:
    return channel.CodeOccurrence(
        code=code,
        field=field,
        is_own=own,
    )


def profile_occurrences(
    owned_codes: tuple[str, ...], foreign_codes: tuple[str, ...] = ()
) -> dict[str, tuple[channel.CodeOccurrence, ...]]:
    return {
        field: tuple(
            occurrence(code, field, own=True) for code in owned_codes
        )
        + tuple(occurrence(code, field, own=False) for code in foreign_codes)
        for field in channel.PROFILE_FIELDS
    }


def zero_numeric_deltas() -> dict[str, float]:
    return {name: 0.0 for name in channel.NUMERIC_DELTA_FIELDS}


class QualityChannelSchemaV9Tests(unittest.TestCase):
    def test_feature_schemas_are_exact_and_unique(self) -> None:
        self.assertEqual(len(channel.PUBLIC_FEATURE_NAMES), 2992)
        self.assertEqual(len(set(channel.PUBLIC_FEATURE_NAMES)), 2992)
        self.assertEqual(len(channel.DECODED_FEATURE_NAMES), 388)
        self.assertEqual(len(set(channel.DECODED_FEATURE_NAMES)), 388)
        self.assertEqual(
            sum(name.startswith("seller_slot_pair__") for name in channel.DECODED_FEATURE_NAMES),
            378,
        )

    def test_feature_builders_have_no_label_or_controller_parameters(self) -> None:
        for function in (
            channel.build_public_code_pair_features,
            channel.build_decoded_slot_pair_features,
        ):
            names = set(inspect.signature(function).parameters)
            self.assertFalse(names & {"label", "labels", "controller", "controller_uid"})


class QualityChannelPolicyV9Tests(unittest.TestCase):
    @staticmethod
    def _rehash(value: dict[str, object]) -> None:
        payload = dict(value)
        payload.pop("canonical_self_hash", None)
        canonical = json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        value["canonical_self_hash"] = hashlib.sha256(canonical).hexdigest()

    def test_canonical_policy_is_self_hashed_and_execution_closed(self) -> None:
        policy = channel_policy.load_policy()
        self.assertFalse(policy["authorization"]["design_1004_rebuild"])
        self.assertFalse(policy["authorization"]["quality_audit_run"])
        self.assertFalse(policy["authorization"]["model_training"])

    def test_rehashed_authorization_widening_still_fails(self) -> None:
        policy = copy.deepcopy(channel_policy.load_policy())
        policy["authorization"]["design_1004_rebuild"] = True
        self._rehash(policy)
        with self.assertRaises(channel_policy.QualityChannelPolicyError):
            channel_policy.validate_policy(policy, check_pins=False)

    def test_rehashed_nested_semantic_drift_still_fails_closed(self) -> None:
        mutations = (
            lambda value: value["authorization"].__setitem__(
                "new_data_execution", True
            ),
            lambda value: value.__setitem__("pins", {}),
            lambda value: value["model_views"].__setitem__(
                "materialized_split_files",
                ["private/duplicate.jsonl"] * 9,
            ),
            lambda value: value["model_views"].__setitem__(
                "recompute_profiles_from_each_item_view", False
            ),
            lambda value: value["text_probe_family"].__setitem__(
                "excluded_negative_pairs_per_world", 7
            ),
            lambda value: value["public_code_probe"].__setitem__(
                "forbidden_inputs", ["label"]
            ),
            lambda value: value["decoded_slot_probe"].__setitem__(
                "seller_slot_range", [0, 99]
            ),
            lambda value: value["bootstrap"].__setitem__(
                "sampling", "row_indices_with_replacement"
            ),
            lambda value: value["read_order"].__setitem__(
                "stage_4", "truth_before_features"
            ),
        )
        canonical = channel_policy.load_policy()
        for mutate in mutations:
            policy = copy.deepcopy(canonical)
            mutate(policy)
            self._rehash(policy)
            with self.subTest(mutation=repr(mutate)), self.assertRaises(
                channel_policy.QualityChannelPolicyError
            ):
                channel_policy.validate_policy(policy, check_pins=False)


class LiteralMaskV9Tests(unittest.TestCase):
    def test_registered_codes_are_masked_without_length_or_context_change(self) -> None:
        text = f"商品编号{CODE_A}，复核{CODE_B}。"
        first = text.index(CODE_A)
        second = text.index(CODE_B)
        masked, counts = channel.mask_registered_code_spans(
            text,
            registered_codes=(CODE_A, CODE_B, CODE_C),
            registered_spans=(
                channel.RegisteredCodeSpan(first, first + 11, CODE_A, "title.code.0"),
                channel.RegisteredCodeSpan(second, second + 11, CODE_B, "title.code.1"),
            ),
        )
        self.assertEqual(masked, "商品编号QXXXXXXXXXX，复核QXXXXXXXXXX。")
        self.assertEqual(len(masked), len(text))
        self.assertEqual(counts, {CODE_A: 1, CODE_B: 1, CODE_C: 0})
        self.assertIsNone(channel.CODE_TOKEN_RE.search(masked))

    def test_unregistered_unspanned_code_and_preexisting_mask_fail_closed(self) -> None:
        with self.assertRaises(channel.QualityChannelViewError):
            channel.mask_registered_code_spans(
                f"编号{CODE_D}", registered_codes=(CODE_A,), registered_spans=()
            )
        with self.assertRaises(channel.QualityChannelViewError):
            channel.mask_registered_code_spans(
                "编号QXXXXXXXXXX",
                registered_codes=(CODE_A,),
                registered_spans=(),
            )

    def test_raw_unregistered_code_scan_has_no_lexical_boundary_escape(self) -> None:
        for text in (
            f"X{CODE_A}Y",
            f"0{CODE_A}9",
            f"x{CODE_A}y",
            f"{CODE_A}{CODE_B}",
            f"prefix{CODE_A}suffix",
        ):
            with self.subTest(text=text), self.assertRaises(
                channel.QualityChannelViewError
            ):
                channel.mask_registered_code_spans(
                    text,
                    registered_codes=(CODE_D,),
                    registered_spans=(),
                )

    def test_duplicate_registered_code_inventory_fails_closed(self) -> None:
        with self.assertRaises(channel.QualityChannelViewError):
            channel.mask_registered_code_spans(
                "普通文本",
                registered_codes=(CODE_A, CODE_A),
                registered_spans=(),
            )

    def test_wrong_or_overlapping_ast_span_fails_closed(self) -> None:
        text = f"{CODE_A}{CODE_B}"
        with self.assertRaises(channel.QualityChannelViewError):
            channel.mask_registered_code_spans(
                text,
                registered_codes=(CODE_A, CODE_B),
                registered_spans=(
                    channel.RegisteredCodeSpan(0, 11, CODE_B, "wrong"),
                    channel.RegisteredCodeSpan(10, 21, CODE_B, "overlap"),
                ),
            )


class PublicCodeProbeV9Tests(unittest.TestCase):
    def setUp(self) -> None:
        self.left = channel.SellerCodeView(
            owned_codes=(CODE_A, CODE_B),
            visible_occurrences=(
                occurrence(CODE_A, "title", own=True),
                occurrence(CODE_B, "description", own=True),
                occurrence(CODE_C, "title", own=False),
            ),
            profile_occurrences=profile_occurrences((CODE_A, CODE_B), (CODE_C,)),
            numeric_profile_deltas={
                **zero_numeric_deltas(),
                "title_length_median": 11.0,
            },
        )
        self.right = channel.SellerCodeView(
            owned_codes=(CODE_C,),
            visible_occurrences=(occurrence(CODE_C, "title", own=True),),
            profile_occurrences=profile_occurrences((CODE_C,)),
            numeric_profile_deltas={
                **zero_numeric_deltas(),
                "title_length_median": 5.0,
            },
        )

    def test_public_features_are_exchange_symmetric_and_have_known_overlap(self) -> None:
        forward = channel.build_public_code_pair_features(self.left, self.right)
        reverse = channel.build_public_code_pair_features(self.right, self.left)
        self.assertEqual(forward, reverse)
        self.assertEqual(len(forward), 2992)
        by_name = dict(zip(channel.PUBLIC_FEATURE_NAMES, forward))
        self.assertEqual(by_name["intersection_count__owned_codes"], 0.0)
        self.assertEqual(by_name["intersection_count__visible_codes"], 1.0)
        self.assertAlmostEqual(by_name["jaccard__visible_codes"], 1 / 3)
        self.assertEqual(
            by_name["maximum__cross_code_position_match_ratio"], 1.0
        )
        self.assertEqual(
            by_name["total_variation__owned__payload_position_00"], 1.0
        )
        self.assertEqual(by_name["dot_product__owned__payload_position_00"], 0.0)
        self.assertEqual(
            by_name["absdiff__owned__payload_position_00__symbol_A"], 0.5
        )
        self.assertEqual(
            by_name["absdiff__profile__category_concat_top__foreign_count"],
            1.0,
        )
        self.assertEqual(
            by_name["absdiff__full_minus_neutral__title_length_median"],
            6.0,
        )

    def test_missing_own_code_and_private_cause_fields_are_absent(self) -> None:
        missing = channel.SellerCodeView(
            owned_codes=(CODE_A, CODE_B),
            visible_occurrences=(occurrence(CODE_A, "title", own=True),),
            profile_occurrences=profile_occurrences((CODE_A, CODE_B)),
            numeric_profile_deltas=zero_numeric_deltas(),
        )
        with self.assertRaises(channel.QualityChannelViewError):
            channel.build_public_code_pair_features(missing, self.right)
        wrong_ownership = channel.SellerCodeView(
            owned_codes=(CODE_A,),
            visible_occurrences=(
                occurrence(CODE_A, "title", own=True),
                occurrence(CODE_C, "description", own=True),
            ),
            profile_occurrences=profile_occurrences((CODE_A,)),
            numeric_profile_deltas=zero_numeric_deltas(),
        )
        with self.assertRaises(channel.QualityChannelViewError):
            channel.build_public_code_pair_features(wrong_ownership, self.right)
        self.assertNotIn("clone", channel.SellerCodeView.__dataclass_fields__)
        self.assertNotIn("capacity", channel.SellerCodeView.__dataclass_fields__)

    def test_absolute_composition_detects_relative_summary_counterexample(self) -> None:
        def one_code_view(code: str) -> channel.SellerCodeView:
            return channel.SellerCodeView(
                owned_codes=(code,),
                visible_occurrences=(occurrence(code, "title", own=True),),
                profile_occurrences=profile_occurrences((code,)),
                numeric_profile_deltas=zero_numeric_deltas(),
            )

        all_a = channel.build_public_code_pair_features(
            one_code_view(CODE_A), one_code_view(CODE_A)
        )
        all_b = channel.build_public_code_pair_features(
            one_code_view(CODE_B), one_code_view(CODE_B)
        )
        a_values = dict(zip(channel.PUBLIC_FEATURE_NAMES, all_a))
        b_values = dict(zip(channel.PUBLIC_FEATURE_NAMES, all_b))
        relative_names = [
            "intersection_count__owned_codes",
            "jaccard__owned_codes",
            "intersection_count__visible_codes",
            "jaccard__visible_codes",
            "maximum__cross_code_position_match_ratio",
            "mean__cross_code_position_match_ratio",
        ]
        self.assertEqual(
            [a_values[name] for name in relative_names],
            [b_values[name] for name in relative_names],
        )
        self.assertNotEqual(
            a_values["sum__owned__payload_position_00__symbol_A"],
            b_values["sum__owned__payload_position_00__symbol_A"],
        )

    def test_foreign_code_absolute_characters_cannot_collide(self) -> None:
        def seller(own: str, foreign: str) -> channel.SellerCodeView:
            return channel.SellerCodeView(
                owned_codes=(own,),
                visible_occurrences=(
                    occurrence(own, "title", own=True),
                    occurrence(foreign, "description", own=False),
                ),
                profile_occurrences=profile_occurrences((own,), (foreign,)),
                numeric_profile_deltas=zero_numeric_deltas(),
            )

        pair_a = channel.build_public_code_pair_features(
            seller(CODE_A, CODE_C), seller(CODE_B, CODE_D)
        )
        pair_b = channel.build_public_code_pair_features(
            seller(CODE_A, CODE_E), seller(CODE_B, CODE_F)
        )
        self.assertNotEqual(pair_a, pair_b)
        values_a = dict(zip(channel.PUBLIC_FEATURE_NAMES, pair_a))
        values_b = dict(zip(channel.PUBLIC_FEATURE_NAMES, pair_b))
        witness = "sum__visible_field__description__payload_position_00__symbol_C"
        self.assertNotEqual(values_a[witness], values_b[witness])


class DecodedSlotProbeV9Tests(unittest.TestCase):
    def test_decoded_features_are_exchange_symmetric_and_one_hot(self) -> None:
        forward = channel.build_decoded_slot_pair_features(
            left_seller_slot=1,
            right_seller_slot=3,
            left_item_slots=(0, 1),
            right_item_slots=(0, 1, 2),
        )
        reverse = channel.build_decoded_slot_pair_features(
            left_seller_slot=3,
            right_seller_slot=1,
            left_item_slots=(0, 1, 2),
            right_item_slots=(0, 1),
        )
        self.assertEqual(forward, reverse)
        self.assertEqual(len(forward), 388)
        one_hot = forward[:378]
        self.assertEqual(sum(one_hot), 1.0)
        self.assertEqual(
            one_hot[channel.DECODED_FEATURE_NAMES.index("seller_slot_pair__01__03")],
            1.0,
        )
        by_name = dict(zip(channel.DECODED_FEATURE_NAMES, forward))
        self.assertEqual(by_name["absdiff__seller_slot"], 2.0)
        self.assertEqual(by_name["absdiff__item_slot_count"], 1.0)
        self.assertEqual(by_name["intersection_count__item_slot_mask"], 2.0)
        self.assertEqual(by_name["union_count__item_slot_mask"], 3.0)
        self.assertAlmostEqual(by_name["jaccard__item_slot_mask"], 2 / 3)

    def test_equal_seller_bool_slot_and_item_gap_fail_closed(self) -> None:
        cases = (
            dict(
                left_seller_slot=1,
                right_seller_slot=1,
                left_item_slots=(0,),
                right_item_slots=(0,),
            ),
            dict(
                left_seller_slot=True,
                right_seller_slot=1,
                left_item_slots=(0,),
                right_item_slots=(0,),
            ),
            dict(
                left_seller_slot=0,
                right_seller_slot=1,
                left_item_slots=(0, 2),
                right_item_slots=(0,),
            ),
        )
        for kwargs in cases:
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(channel.QualityChannelViewError):
                    channel.build_decoded_slot_pair_features(**kwargs)


class NeutralizedPermutationV9Tests(unittest.TestCase):
    def test_code_independent_mount_passes_and_code_dependent_mount_fails(self) -> None:
        channel.assert_neutralized_code_permutation_invariant(
            codes=(CODE_A, CODE_B, CODE_C),
            build_model_mount=lambda mapping: b"fixed-neutralized-mount",
        )
        with self.assertRaises(channel.QualityChannelViewError):
            channel.assert_neutralized_code_permutation_invariant(
                codes=(CODE_A, CODE_B, CODE_C),
                build_model_mount=lambda mapping: "|".join(mapping.values()).encode(
                    "ascii"
                ),
            )

    def test_all_16_by_16_modifier_and_tag_states_must_collapse(self) -> None:
        channel.assert_all_derived_symbol_states_collapse(
            build_model_mount=lambda _original_code: b"fixed-neutral-controls"
        )
        with self.assertRaises(channel.QualityChannelViewError):
            channel.assert_all_derived_symbol_states_collapse(
                build_model_mount=lambda original_code: original_code[-2:].encode(
                    "ascii"
                )
            )


if __name__ == "__main__":
    unittest.main()
