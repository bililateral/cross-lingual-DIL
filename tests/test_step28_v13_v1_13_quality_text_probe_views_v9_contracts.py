from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import sys
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import step28_v13_v1_13_quality_text_probe_views_v9 as text_views
import step28_v13_v1_13_quality_probe_preparer_v9 as preparer


EXPECTED_NAME_HASHES = {
    "fs_full": "e7e929d856423d03951612884bbffd57649190ecf5c414b76819e4129265957b",
    "fs_title": "71e72e4c3cf6ea36d78477acb0617f5436d2813c201998b13ae051b55fe9afe8",
    "fs_template_surface": "4ef95fb703e708e59f5334636c1bae539ed44dc35293832d23a908fea9252606",
    "p_full": "1c08e76c0f74ff126a0d3f722afa652c36393d3e30f200077c4c13c91820ec8b",
    "p_topic": "6a201d2afbc4b1579cef20e53ae81703924b69ceba53df2b2913002447d0e891",
    "p_template_surface": "57fd57edf108f3ad09f50e88c9b3ee23644dcc1f3abff53ddc86a2d051cd4156",
    "u_joint_full": "420333af4f991424cd7d65ebeeaeb0aafd43ea612eba8398852c14c25525a745",
}
EXPECTED_GOLDEN_MATRIX_HASHES = {
    "fs_full": "125ebefe6208177182ed7636579efb71793945bf35578e3ad740d6849b5aefea",
    "fs_title": "5c59c4210cda317fbf7cbf2f43422f025e40dba506027e94731a1b37c57c8451",
    "fs_template_surface": "f667b7dc66beb256875df0e43873fe39e86d50d0c3547adde7192fcd8b3f4a3f",
    "p_full": "7957d0da16fe2f6b1043b358d405704e25ae6923073271d15415181d0aea20b3",
    "p_topic": "d42cd426274302763ff4c718efae1eccfddb504a7237e4396b3ee8f735cf4935",
    "p_template_surface": "8dcf2038583f64db60b2f91af5f1a61f98896d1d13b208f04fb9a691de44e580",
    "u_joint_full": "b93aee039521bd2c13638f681cd193dfe03a39783ee7644e4f41a62ddb18634e",
}


def profile(seller_uid: str, marker: str) -> dict[str, object]:
    return {
        "seller_uid": seller_uid,
        "category_concat_top": f"样品{marker}",
        "signature_title_concat": f"签名标题 {marker}",
        "title_concat_top": f"标题集合 {marker} 2026",
        "signature_description_concat": f"签名描述，{marker}",
        "description_concat_top": f"描述集合\n{marker}",
        "item_count": 2,
        "title_length_stats": {"median": 8.0},
        "description_length_stats": {"median": 12.0},
        "style_stats": {
            "digit_ratio_mean": 0.1,
            "punct_ratio_mean": 0.2,
            "repeated_title_share": 0.0,
            "repeated_description_share": 0.0,
            "max_category_share": 1.0,
        },
    }


def item(
    item_uid: str, seller_uid: str, title: str, description: str
) -> dict[str, str]:
    return {
        "item_uid": item_uid,
        "seller_uid": seller_uid,
        "world_uid": "world",
        "title": title,
        "description": description,
    }


class QualityTextProbeViewsV9Tests(unittest.TestCase):
    def setUp(self) -> None:
        self.profiles = [profile("seller_a", "甲"), profile("seller_b", "乙")]
        self.items = [
            item("a0", "seller_a", "甲标题 01", "甲描述，常规"),
            item("a1", "seller_a", "甲标题 02", "甲描述\n更多"),
            item("b0", "seller_b", "乙标题 11", "乙描述，常规"),
            item("b1", "seller_b", "乙标题 12", "乙描述\n更多"),
        ]
        self.profiles = json.loads(
            json.dumps(self.profiles, ensure_ascii=False, sort_keys=True)
        )
        self.items = json.loads(
            json.dumps(self.items, ensure_ascii=False, sort_keys=True)
        )
        self.endpoints = [
            {
                "canonical_pair_uid": "pair",
                "world_uid": "world",
                "seller_uid_left": "seller_a",
                "seller_uid_right": "seller_b",
            }
        ]

    @staticmethod
    def _hash(names: tuple[str, ...]) -> str:
        raw = json.dumps(
            list(names),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        return hashlib.sha256(raw).hexdigest()

    def test_all_seven_views_have_frozen_widths_and_name_hashes(self) -> None:
        views, names = text_views.build_text_probe_views(
            items=self.items, profiles=self.profiles, endpoints=self.endpoints
        )
        self.assertEqual(tuple(views), text_views.VIEW_ORDER)
        self.assertEqual(
            tuple(matrix.shape for matrix in views.values()),
            tuple((1, width) for width in text_views.EXPECTED_WIDTHS),
        )
        self.assertEqual(
            {name: self._hash(values) for name, values in names.items()},
            EXPECTED_NAME_HASHES,
        )
        self.assertEqual(
            {
                name: hashlib.sha256(
                    matrix.astype("<f8", copy=False).tobytes(order="C")
                ).hexdigest()
                for name, matrix in views.items()
            },
            EXPECTED_GOLDEN_MATRIX_HASHES,
        )
        self.assertTrue(all(np.isfinite(matrix).all() for matrix in views.values()))

    def test_pair_exchange_is_value_invariant(self) -> None:
        baseline, _names = text_views.build_text_probe_views(
            items=self.items, profiles=self.profiles, endpoints=self.endpoints
        )
        reverse_endpoint = [dict(self.endpoints[0])]
        reverse_endpoint[0]["seller_uid_left"] = "seller_b"
        reverse_endpoint[0]["seller_uid_right"] = "seller_a"
        reversed_views, _ = text_views.build_text_probe_views(
            items=self.items,
            profiles=self.profiles,
            endpoints=reverse_endpoint,
        )
        for name in text_views.VIEW_ORDER:
            np.testing.assert_array_equal(baseline[name], reversed_views[name])

    def test_item_and_profile_orders_are_separately_value_invariant(self) -> None:
        baseline, _ = text_views.build_text_probe_views(
            items=self.items, profiles=self.profiles, endpoints=self.endpoints
        )
        item_reordered, _ = text_views.build_text_probe_views(
            items=list(reversed(self.items)),
            profiles=self.profiles,
            endpoints=self.endpoints,
        )
        profile_reordered, _ = text_views.build_text_probe_views(
            items=self.items,
            profiles=list(reversed(self.profiles)),
            endpoints=self.endpoints,
        )
        for name in text_views.VIEW_ORDER:
            np.testing.assert_array_equal(baseline[name], item_reordered[name])
            np.testing.assert_array_equal(baseline[name], profile_reordered[name])

    def test_item_uid_rename_cannot_change_features(self) -> None:
        baseline, _ = text_views.build_text_probe_views(
            items=self.items, profiles=self.profiles, endpoints=self.endpoints
        )
        renamed = [dict(row) for row in self.items]
        for index, row in enumerate(renamed):
            row["item_uid"] = f"renamed_{3 - index}"
        observed, _ = text_views.build_text_probe_views(
            items=renamed, profiles=self.profiles, endpoints=self.endpoints
        )
        for name in text_views.VIEW_ORDER:
            np.testing.assert_array_equal(baseline[name], observed[name])

    def test_seller_uid_rename_cannot_change_features(self) -> None:
        baseline, _ = text_views.build_text_probe_views(
            items=self.items, profiles=self.profiles, endpoints=self.endpoints
        )
        replacements = {"seller_a": "renamed_z", "seller_b": "renamed_y"}
        renamed_items = [dict(row) for row in self.items]
        for row in renamed_items:
            row["seller_uid"] = replacements[row["seller_uid"]]
        renamed_profiles = copy.deepcopy(self.profiles)
        for row in renamed_profiles:
            row["seller_uid"] = replacements[row["seller_uid"]]
        renamed_endpoints = [dict(row) for row in self.endpoints]
        for row in renamed_endpoints:
            row["seller_uid_left"] = replacements[row["seller_uid_left"]]
            row["seller_uid_right"] = replacements[row["seller_uid_right"]]
        observed, _ = text_views.build_text_probe_views(
            items=renamed_items,
            profiles=renamed_profiles,
            endpoints=renamed_endpoints,
        )
        for name in text_views.VIEW_ORDER:
            np.testing.assert_array_equal(baseline[name], observed[name])

    def test_world_uid_rename_cannot_change_features(self) -> None:
        baseline, _ = text_views.build_text_probe_views(
            items=self.items, profiles=self.profiles, endpoints=self.endpoints
        )
        renamed_items = [dict(row) for row in self.items]
        for row in renamed_items:
            row["world_uid"] = "renamed_world"
        renamed_endpoints = [dict(row) for row in self.endpoints]
        renamed_endpoints[0]["world_uid"] = "renamed_world"
        observed, _ = text_views.build_text_probe_views(
            items=renamed_items,
            profiles=self.profiles,
            endpoints=renamed_endpoints,
        )
        for name in text_views.VIEW_ORDER:
            np.testing.assert_array_equal(baseline[name], observed[name])

    def test_pair_uid_rename_cannot_change_features(self) -> None:
        baseline, _ = text_views.build_text_probe_views(
            items=self.items, profiles=self.profiles, endpoints=self.endpoints
        )
        renamed_endpoints = [dict(row) for row in self.endpoints]
        renamed_endpoints[0]["canonical_pair_uid"] = "renamed_pair"
        observed, _ = text_views.build_text_probe_views(
            items=self.items,
            profiles=self.profiles,
            endpoints=renamed_endpoints,
        )
        for name in text_views.VIEW_ORDER:
            np.testing.assert_array_equal(baseline[name], observed[name])

    def test_profile_item_count_must_exactly_match_item_rows(self) -> None:
        for invalid in (3, 2.0, True):
            profiles = copy.deepcopy(self.profiles)
            profiles[0]["item_count"] = invalid
            with self.subTest(invalid=invalid):
                with self.assertRaises(text_views.QualityTextProbeViewError):
                    text_views.build_text_probe_views(
                        items=self.items,
                        profiles=profiles,
                        endpoints=self.endpoints,
                    )

    def test_label_or_generator_field_contamination_fails_closed(self) -> None:
        endpoint = [dict(self.endpoints[0])]
        endpoint[0]["label"] = 1
        with self.assertRaises(text_views.QualityTextProbeViewError):
            text_views.build_text_probe_views(
                items=self.items, profiles=self.profiles, endpoints=endpoint
            )
        items = [dict(row) for row in self.items]
        items[0]["override_kind"] = "forbidden"
        with self.assertRaises(text_views.QualityTextProbeViewError):
            text_views.build_text_probe_views(
                items=items, profiles=self.profiles, endpoints=self.endpoints
            )
        profiles = copy.deepcopy(self.profiles)
        profiles[0]["controller_uid"] = "forbidden"
        with self.assertRaises(text_views.QualityTextProbeViewError):
            text_views.build_text_probe_views(
                items=self.items, profiles=profiles, endpoints=self.endpoints
            )
        wrong_type_items = [dict(row) for row in self.items]
        wrong_type_items[0]["title"] = 123
        with self.assertRaises(text_views.QualityTextProbeViewError):
            text_views.build_text_probe_views(
                items=wrong_type_items,
                profiles=self.profiles,
                endpoints=self.endpoints,
            )
        wrong_type_profiles = copy.deepcopy(self.profiles)
        wrong_type_profiles[0]["category_concat_top"] = 123
        with self.assertRaises(text_views.QualityTextProbeViewError):
            text_views.build_text_probe_views(
                items=self.items,
                profiles=wrong_type_profiles,
                endpoints=self.endpoints,
            )

    def test_template_mask_is_character_class_only(self) -> None:
        self.assertEqual(text_views.template_mask("Ab中12，!"), "字字字数数，!")
        self.assertEqual(text_views.word12_tokens("AB-12中文"), ["ab", "12", "中", "文"])

    def test_preparer_freezes_all_seven_surface_matrices(self) -> None:
        sources = (
            preparer.SourceCommitment(
                path="fixture/surface.jsonl",
                size_bytes=1,
                sha256="3" * 64,
            ),
        )
        frozen = preparer.prepare_text_surface_matrices(
            surface="surface_full",
            items=self.items,
            profiles=self.profiles,
            endpoints=self.endpoints,
            ordered_world_uids=("world",),
            sources=sources,
            expected_sellers_per_world=2,
            expected_pairs_per_world=1,
        )
        self.assertEqual(
            tuple(value.view for value in frozen),
            tuple(f"surface_full::{name}" for name in text_views.VIEW_ORDER),
        )
        self.assertEqual(
            tuple(value.values.shape[1] for value in frozen),
            text_views.EXPECTED_WIDTHS,
        )
        self.assertTrue(
            all(
                preparer.verify_frozen_feature_matrix(value)["label_count_read"]
                == 0
                for value in frozen
            )
        )


if __name__ == "__main__":
    unittest.main()
