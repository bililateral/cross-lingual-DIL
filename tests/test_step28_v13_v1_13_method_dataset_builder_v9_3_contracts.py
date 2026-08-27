from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import step28_v13_common as common
import step28_v13_v1_13_create_method_random_authority_v9_3 as authority_module
import step28_v13_v1_13_method_dataset_builder_v9_3 as builder


def valid_authority() -> dict:
    values = [f"{index:064x}" for index in range(1, 15)]
    payload = {
        "version": builder.AUTHORITY_VERSION,
        "status": "FROZEN_FRESH_SINGLE_USE",
        "canonical_self_sha256": None,
        "single_use": True,
        "world_counts": dict(builder.WORLD_COUNTS),
        "method_policy_canonical_self_sha256": "f" * 64,
        "keys": {
            "id_namespace_key_hex": values[0],
            "structure_key_hex": values[1],
            "id_key_hex": values[2],
            "identity_value_key_hex": values[3],
            "text_key_hex": values[4],
            "candidate_key_hex": values[5],
            "query_key_hex": values[6],
            "document_variation_key_hex": values[7],
            "anonymous_handle_key_hex": values[8],
            "rewire_key_hexes": values[9:14],
        },
    }
    payload["canonical_self_sha256"] = builder._self_hash(payload)
    return payload


class MethodDatasetBuilderV93Contracts(unittest.TestCase):
    def test_prebuild_gate_uses_remove_field_self_hash_convention(self) -> None:
        payload = {"version": "gate", "scientific_pass": True}
        expected = common.canonical_sha256(payload)
        persisted = {**payload, "canonical_self_sha256": expected}
        self.assertEqual(builder._prebuild_gate_self_hash(persisted), expected)
        self.assertNotEqual(builder._self_hash(persisted), expected)

    def test_authority_schema_is_exact_self_hashed_and_unique(self) -> None:
        payload = valid_authority()
        self.assertEqual(builder._validate_authority(payload), payload)

        duplicate = deepcopy(payload)
        duplicate["keys"]["rewire_key_hexes"][0] = duplicate["keys"][
            "text_key_hex"
        ]
        duplicate["canonical_self_sha256"] = builder._self_hash(duplicate)
        with self.assertRaises(builder.MethodDatasetBuilderV93Error):
            builder._validate_authority(duplicate)

        extra = deepcopy(payload)
        extra["unexpected"] = True
        extra["canonical_self_sha256"] = builder._self_hash(extra)
        with self.assertRaises(builder.MethodDatasetBuilderV93Error):
            builder._validate_authority(extra)

    def test_line_writer_bytes_hash_and_schema_are_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            json_writer = builder._LineWriter(root / "rows.jsonl")
            json_writer.write({"b": 2, "a": "汉字"})
            json_record = json_writer.close(root)
            expected = common.canonical_json_bytes({"b": 2, "a": "汉字"}) + b"\n"
            self.assertEqual((root / "rows.jsonl").read_bytes(), expected)
            self.assertEqual(json_record["sha256"], common.sha256_file(root / "rows.jsonl"))
            self.assertEqual(json_record["row_count"], 1)

            csv_writer = builder._LineWriter(root / "rows.csv", fields=("a", "b"))
            csv_writer.write({"a": "x,y", "b": "换行\n值"})
            with self.assertRaises(builder.MethodDatasetBuilderV93Error):
                csv_writer.write({"b": 1, "a": 2})
            csv_record = csv_writer.close(root)
            self.assertEqual((root / "rows.csv").read_bytes(), b'a,b\n"x,y","\xe6\x8d\xa2\xe8\xa1\x8c\n\xe5\x80\xbc"\n')
            builder._verify_files(root, (json_record, csv_record))

            (root / "rows.jsonl").write_bytes(b"tampered\n")
            with self.assertRaises(builder.MethodDatasetBuilderV93Error):
                builder._verify_files(root, (json_record, csv_record))

    def test_formal_paths_fail_closed_before_any_authority_consumption(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            wrong = Path(temporary)
            with self.assertRaises(builder.MethodDatasetBuilderV93Error):
                builder.build(
                    output_root=wrong / "root",
                    authority_path=wrong / "authority.json",
                    consume_authority=True,
                )
            with self.assertRaises(builder.MethodDatasetBuilderV93Error):
                authority_module.create(wrong / "authority.json")
            self.assertFalse((wrong / "authority.json").exists())

    def test_json_manifest_record_is_in_file_closure(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / "train" / "split_manifest.json"
            path.parent.mkdir(parents=True)
            path.write_text(json.dumps({"ok": True}), encoding="utf-8")
            record = builder._json_file_record(path, root)
            self.assertEqual(record["path"], "train/split_manifest.json")
            self.assertEqual(record["format"], "json")
            builder._verify_files(root, (record,))

    def test_failed_build_deletes_only_new_partial_payloads(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "method_root"
            building = root.with_name(root.name + ".building")

            def fail_after_partial_write(**_kwargs: object) -> dict:
                building.mkdir(parents=True)
                (building / "partial.bin").write_bytes(b"partial")
                root.mkdir(parents=True)
                (root / "partial.bin").write_bytes(b"partial")
                raise builder.MethodDatasetBuilderV93Error("synthetic failure")

            with mock.patch.object(
                builder, "_build_once", side_effect=fail_after_partial_write
            ):
                with self.assertRaisesRegex(
                    builder.MethodDatasetBuilderV93Error, "synthetic"
                ):
                    builder.build(
                        output_root=root,
                        authority_path=Path(temporary) / "authority.json",
                        consume_authority=False,
                    )
            self.assertFalse(building.exists())
            self.assertFalse(root.exists())


if __name__ == "__main__":
    unittest.main()
