from __future__ import annotations

import ast
import csv
import hashlib
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import step28_v13_independent_release_tree_hardening_audit as audit  # noqa: E402


class IndependentReleaseTreeHardeningAuditContracts(
    unittest.TestCase
):
    key_hex = "42" * 32

    @staticmethod
    def write_csv(
        path: Path,
        *,
        fields: tuple[str, ...],
        rows: list[dict[str, str]],
    ) -> None:
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=list(fields),
                lineterminator="\n",
            )
            writer.writeheader()
            writer.writerows(rows)

    def candidate_rows(
        self,
        world_uids: list[str],
    ) -> list[dict[str, str]]:
        output: list[dict[str, str]] = []
        for world_uid in world_uids:
            pairs = []
            for index in range(audit.EXPECTED_CANDIDATES_PER_WORLD):
                left = f"seller_{world_uid}_{index:02d}_a"
                right = f"seller_{world_uid}_{index:02d}_b"
                pair_uid = f"{left}||{right}"
                pairs.append(
                    {
                        "canonical_pair_uid": pair_uid,
                        "world_uid": world_uid,
                        "seller_uid_left": left,
                        "seller_uid_right": right,
                    }
                )
            pairs.sort(
                key=lambda row: (
                    audit.output_order_digest(
                        key_hex=self.key_hex,
                        world_uid=world_uid,
                        pair_uid=row["canonical_pair_uid"],
                    ),
                    row["canonical_pair_uid"].encode("utf-8"),
                )
            )
            output.extend(pairs)
        return output

    def run_candidate_validation(
        self,
        *,
        world_uids: list[str],
        rows: list[dict[str, str]],
    ) -> dict[str, object]:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "candidate_pairs.csv"
            self.write_csv(
                path,
                fields=audit.EXPECTED_CANDIDATE_FIELDS,
                rows=rows,
            )
            with mock.patch.object(
                audit,
                "EXPECTED_WORLD_COUNT",
                len(world_uids),
            ), mock.patch.object(
                audit,
                "EXPECTED_CANDIDATE_COUNT",
                len(world_uids)
                * audit.EXPECTED_CANDIDATES_PER_WORLD,
            ):
                return audit.validate_candidate_rows(
                    path,
                    world_uids=world_uids,
                    candidate_key_hex=self.key_hex,
                )

    def test_complete_contiguous_two_world_order_passes(self) -> None:
        worlds = ["world_a", "world_b"]
        result = self.run_candidate_validation(
            world_uids=worlds,
            rows=self.candidate_rows(worlds),
        )
        self.assertTrue(
            result["independent_selected_global_rank_exact"]
        )
        self.assertTrue(result["world_blocks_contiguous_and_exact"])

    def test_missing_world_fails_closed(self) -> None:
        worlds = ["world_a", "world_b"]
        rows = self.candidate_rows(worlds)[:40]
        with self.assertRaises(audit.AuditError):
            self.run_candidate_validation(
                world_uids=worlds,
                rows=rows,
            )

    def test_interleaved_world_blocks_fail_closed(self) -> None:
        worlds = ["world_a", "world_b"]
        rows = self.candidate_rows(worlds)
        interleaved = rows[:20] + rows[40:60] + rows[20:40] + rows[60:]
        with self.assertRaises(audit.AuditError):
            self.run_candidate_validation(
                world_uids=worlds,
                rows=interleaved,
            )

    def test_wrong_hmac_order_fails_closed(self) -> None:
        worlds = ["world_a"]
        rows = self.candidate_rows(worlds)
        rows[0], rows[1] = rows[1], rows[0]
        with self.assertRaises(audit.AuditError):
            self.run_candidate_validation(
                world_uids=worlds,
                rows=rows,
            )

    def test_empty_candidate_table_fails_closed(self) -> None:
        with self.assertRaises(audit.AuditError):
            self.run_candidate_validation(
                world_uids=["world_a"],
                rows=[],
            )

    def test_canonical_self_hash_tamper_fails_closed(self) -> None:
        document = {"status": "PASS"}
        document["canonical_self_hash"] = audit.canonical_sha256(document)
        self.assertEqual(
            audit.validate_canonical_self_hash(
                document,
                label="test",
            ),
            document["canonical_self_hash"],
        )
        document["status"] = "FAIL"
        with self.assertRaises(audit.AuditError):
            audit.validate_canonical_self_hash(
                document,
                label="test",
            )

    def test_manifest_member_missing_extra_size_and_hash_fail_closed(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            member = root / "observed" / "member.txt"
            member.parent.mkdir()
            member.write_text("abcd", encoding="utf-8")
            digest = hashlib.sha256(member.read_bytes()).hexdigest()
            records = {
                "observed/member.txt": {
                    "model_mount_allowed": True,
                    "path": "observed/member.txt",
                    "sha256": digest,
                    "size_bytes": 4,
                }
            }
            passed = audit.validate_manifest_members(
                root,
                records=records,
            )
            self.assertTrue(
                passed["all_member_sizes_and_sha256_exact"]
            )

            member.write_text("abce", encoding="utf-8")
            with self.assertRaises(audit.AuditError):
                audit.validate_manifest_members(root, records=records)

            member.write_text("abcd", encoding="utf-8")
            extra = root / "extra.txt"
            extra.write_text("extra", encoding="utf-8")
            with self.assertRaises(audit.AuditError):
                audit.validate_manifest_members(root, records=records)

            extra.unlink()
            member.unlink()
            with self.assertRaises(audit.AuditError):
                audit.validate_manifest_members(root, records=records)

    def test_top_level_extra_member_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for name in audit.EXPECTED_TOP_LEVEL:
                path = root / name
                if "." in name:
                    path.write_text("{}", encoding="utf-8")
                else:
                    path.mkdir()
            audit.validate_top_level_member_set(root)
            (root / "unexpected.txt").write_text(
                "unexpected",
                encoding="utf-8",
            )
            with self.assertRaises(audit.AuditError):
                audit.validate_top_level_member_set(root)

    def test_current_release_headers_and_receipts_are_pinned(self) -> None:
        dataset = (
            ROOT
            / "reports"
            / "step28_synthetic_chinese_dataset"
            / audit.EXPECTED_RUN_ID
        )
        audit.validate_top_level_member_set(dataset)
        release_path = dataset / "release_manifest.json"
        self.assertEqual(
            audit.sha256_file(release_path),
            audit.EXPECTED_RELEASE_SHA256,
        )
        release = audit.load_json(release_path)
        self.assertEqual(
            audit.validate_canonical_self_hash(
                release,
                label="release",
            ),
            audit.EXPECTED_RELEASE_SELF_SHA256,
        )
        for split in audit.EXPECTED_SPLITS:
            path = dataset / split / "split_manifest.json"
            self.assertEqual(
                audit.sha256_file(path),
                audit.EXPECTED_SPLIT_MANIFEST_SHA256[split],
            )
            manifest = audit.load_json(path)
            self.assertEqual(
                audit.validate_canonical_self_hash(
                    manifest,
                    label=split,
                ),
                release["split_receipts"][split][
                    "manifest_self_sha256"
                ],
            )
            self.assertEqual(
                audit.sha256_file(path),
                release["split_receipts"][split]["manifest_sha256"],
            )
        equivalence_path = dataset / "repair_equivalence_report.json"
        equivalence = audit.load_json(equivalence_path)
        self.assertEqual(
            audit.sha256_file(equivalence_path),
            release["repair_equivalence_report"]["sha256"],
        )
        self.assertEqual(
            audit.validate_canonical_self_hash(
                equivalence,
                label="equivalence",
            ),
            release["repair_equivalence_report"][
                "canonical_self_hash"
            ],
        )
        base_policy_path = ROOT / release["base_policy"]["path"]
        self.assertEqual(
            audit.sha256_file(base_policy_path),
            release["base_policy"]["sha256"],
        )

    def test_auditor_imports_no_frozen_step28_implementation(self) -> None:
        source = Path(audit.__file__).read_text(encoding="utf-8")
        tree = ast.parse(source)
        imported_roots: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported_roots.update(
                    alias.name.split(".", 1)[0]
                    for alias in node.names
                )
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported_roots.add(node.module.split(".", 1)[0])
        self.assertFalse(
            {
                name
                for name in imported_roots
                if name.startswith("step28")
            }
        )


if __name__ == "__main__":
    unittest.main()
