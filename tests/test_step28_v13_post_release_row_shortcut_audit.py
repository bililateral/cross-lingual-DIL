from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import step28_v13_common as common  # noqa: E402
import step28_v13_post_release_row_shortcut_audit as audit  # noqa: E402


class PostReleaseRowShortcutAuditContracts(unittest.TestCase):
    @staticmethod
    def _item(text: str = "普通商品说明") -> dict[str, str]:
        return {
            "world_uid": "w_1",
            "seller_uid": "sel_1",
            "item_uid": "itm_1",
            "title": "普通标题",
            "description": text,
        }

    @staticmethod
    def _profile(text: str = "普通商品说明") -> dict[str, str]:
        return {"seller_uid": "sel_1", "profile_text": text}

    @staticmethod
    def _slot() -> dict[str, object]:
        return {
            "slot_uid": "slot_1",
            "item_uid": "itm_1",
            "seller_uid": "sel_1",
            "field_name": "description",
            "raw_surface": "tg_secret_handle",
            "downstream_canonical_value": "secret_handle",
        }

    def test_clean_redaction_rows_pass(self) -> None:
        result = audit.scan_redaction_rows(
            items=[self._item()],
            profiles=[self._profile()],
            slots=[self._slot()],
        )
        self.assertTrue(result["hard_leakage_gate_pass"])
        self.assertEqual(result["raw_surface_residual_count"], 0)
        self.assertEqual(result["canonical_value_residual_count"], 0)

    def test_surface_profile_and_internal_markers_fail(self) -> None:
        result = audit.scan_redaction_rows(
            items=[self._item("内部字段 controller_uid=ctl_abcdef1234567890")],
            profiles=[self._profile("仍残留 tg_secret_handle")],
            slots=[self._slot()],
        )
        self.assertFalse(result["hard_leakage_gate_pass"])
        self.assertEqual(result["forbidden_internal_marker_count"], 1)
        self.assertEqual(result["raw_surface_residual_count"], 1)
        self.assertEqual(result["canonical_value_residual_count"], 1)

    def test_duplicate_items_are_rejected(self) -> None:
        with self.assertRaises(common.ContractError):
            audit.scan_redaction_rows(
                items=[self._item(), self._item()],
                profiles=[self._profile()],
                slots=[self._slot()],
            )

    def test_output_must_remain_outside_formal_root(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset = root / "formal"
            dataset.mkdir()
            with self.assertRaises(common.ContractError):
                audit.require_output_outside_dataset(
                    dataset,
                    dataset / "audit.json",
                )
            audit.require_output_outside_dataset(
                dataset,
                root / "audits" / "audit.json",
            )

    def test_candidate_order_requires_independent_global_hmac(self) -> None:
        key = "11" * 32
        world_uid = "w_test"
        pair_uids = [f"pair_{index:02d}" for index in range(40)]
        ordered = sorted(
            pair_uids,
            key=lambda pair_uid: (
                common.hmac_digest(
                    key,
                    world_uid,
                    "selected_global_rank",
                    pair_uid,
                ),
                pair_uid.encode("utf-8"),
            ),
        )
        rows = [
            {
                "canonical_pair_uid": pair_uid,
                "world_uid": world_uid,
                "seller_uid_left": f"left_{pair_uid}",
                "seller_uid_right": f"right_{pair_uid}",
            }
            for pair_uid in ordered
        ]
        self.assertTrue(
            audit.candidate_order_contract(
                rows,
                candidate_key_hex=key,
            )["contract_exact"]
        )
        self.assertFalse(
            audit.candidate_order_contract(
                list(reversed(rows)),
                candidate_key_hex=key,
            )["contract_exact"]
        )


if __name__ == "__main__":
    unittest.main()
