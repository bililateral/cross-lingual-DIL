from __future__ import annotations

import csv
import hashlib
import json
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import step7_v5_build_english_source_dataset as builder


class Step7V5EnglishSourceUnitTests(unittest.TestCase):
    def test_sql_tuple_parser_handles_commas_quotes_and_terminal_semicolon(self) -> None:
        row = "(7, 'a,b', '''quoted''', 0);"
        self.assertEqual(
            builder.parse_sql_tuple_line(row, 4),
            ["7", "a,b", "'quoted'", "0"],
        )

    def test_style_projection_is_lexically_blind(self) -> None:
        projected = builder.style_projection("Hello HELLO hello hElLo, 12.50! 中文")
        self.assertEqual(projected, "W5T W5U W5L W5M, N00.00! H2")
        for forbidden in ("Hello", "HELLO", "hello", "hElLo", "中文"):
            self.assertNotIn(forbidden, projected)

    def test_conflict_components_close_weak_transitive_edges(self) -> None:
        accounts = {
            "a": {
                "aux_fingerprints": {"A"},
                "weak_fingerprints": {"W_AB"},
                "strong_fingerprints": set(),
                "strong_key_aliases": set(),
            },
            "b": {
                "aux_fingerprints": {"B"},
                "weak_fingerprints": {"W_AB", "W_BC"},
                "strong_fingerprints": set(),
                "strong_key_aliases": {"KEY_ONE", "KEY_TWO"},
            },
            "c": {
                "aux_fingerprints": {"C"},
                "weak_fingerprints": {"W_BC"},
                "strong_fingerprints": set(),
                "strong_key_aliases": set(),
            },
            "d": {
                "aux_fingerprints": {"D"},
                "weak_fingerprints": set(),
                "strong_fingerprints": set(),
                "strong_key_aliases": {"KEY_TWO"},
            },
        }
        components = builder.build_identity_conflict_components(accounts)
        self.assertEqual(components["a"], components["b"])
        self.assertEqual(components["a"], components["c"])
        self.assertEqual(components["a"], components["d"])

    def test_cleaner_removes_source_alias_and_direct_identifiers(self) -> None:
        alias_pattern = builder.compile_alias_pattern({"Seller-Name"}, 3)
        raw = (
            "Seller-Name email a@b.com https://example.com "
            "0123456789ABCDEF0123456789ABCDEF01234567"
        )
        cleaned = builder.clean_visible_text(raw, alias_pattern)
        self.assertNotIn("Seller-Name", cleaned)
        self.assertNotIn("a@b.com", cleaned)
        self.assertNotIn("example.com", cleaned)
        self.assertNotIn("0123456789ABCDEF", cleaned)

    def test_account_alias_variants_are_removed_and_independently_detected(self) -> None:
        aliases = {"ThePurpleLotus", "m-power"}
        pattern = builder.compile_account_alias_pattern(aliases, 3)
        raw = "The PurpleLotus coupon and 50 grM-POWER amnesia"
        self.assertEqual(
            builder.source_alias_residuals(raw, aliases, 3),
            {"thepurplelotus", "m-power"},
        )
        cleaned = pattern.sub(" ", raw)
        self.assertEqual(builder.source_alias_residuals(cleaned, aliases, 3), set())


class Step7V5EnglishSourceArtifactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.policy = builder.load_policy()
        cls.output = ROOT / cls.policy["output_directory"]
        if not cls.output.is_dir():
            raise AssertionError(f"Formal Step7 V5 English source output is missing: {cls.output}")
        cls.manifest = json.loads(
            (cls.output / "manifest.json").read_text(encoding="utf-8")
        )

    def test_manifest_and_file_hashes(self) -> None:
        self.assertEqual(self.manifest["status"], "PASSED")
        self.assertEqual(self.manifest["policy_sha256"], builder.file_sha256(builder.POLICY_PATH))
        self.assertEqual(
            self.manifest["builder_sha256"],
            builder.file_sha256(ROOT / self.manifest["builder_path"]),
        )
        for record in self.manifest["files"]:
            path = self.output / record["path"]
            self.assertTrue(path.is_file(), path)
            self.assertEqual(path.stat().st_size, record["size_bytes"])
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            self.assertEqual(digest, record["sha256"])

    def test_public_files_have_exact_allowlisted_fields(self) -> None:
        with (self.output / "public_pairs.csv").open(
            "r", encoding="utf-8", newline=""
        ) as handle:
            pair_reader = csv.DictReader(handle)
            self.assertEqual(
                pair_reader.fieldnames,
                ["pair_uid", "fold_id", "account_left_uid", "account_right_uid"],
            )
            pairs = list(pair_reader)
        self.assertTrue(pairs)

        expected_views = {
            "public_items_full_clean.jsonl": {
                "item_uid",
                "account_uid",
                "title_clean",
                "description_clean",
            },
            "public_items_style_projection.jsonl": {
                "item_uid",
                "account_uid",
                "title_style",
                "description_style",
            },
        }
        view_rows = {}
        for filename, expected_fields in expected_views.items():
            rows = [
                json.loads(line)
                for line in (self.output / filename).read_text(encoding="utf-8").splitlines()
            ]
            self.assertTrue(rows)
            self.assertTrue(all(set(row) == expected_fields for row in rows))
            self.assertEqual(len(rows), len({row["item_uid"] for row in rows}))
            view_rows[filename] = {row["item_uid"]: row for row in rows}
        self.assertEqual(
            set(view_rows["public_items_full_clean.jsonl"]),
            set(view_rows["public_items_style_projection.jsonl"]),
        )
        for item_uid, full_row in view_rows["public_items_full_clean.jsonl"].items():
            style_row = view_rows["public_items_style_projection.jsonl"][item_uid]
            self.assertEqual(style_row["account_uid"], full_row["account_uid"])
            self.assertEqual(
                style_row["title_style"],
                builder.style_projection(full_row["title_clean"]),
            )
            self.assertEqual(
                style_row["description_style"],
                builder.style_projection(full_row["description_clean"]),
            )

        pair_accounts = {
            row[field]
            for row in pairs
            for field in ("account_left_uid", "account_right_uid")
        }
        self.assertEqual(
            pair_accounts,
            {row["account_uid"] for row in view_rows["public_items_full_clean.jsonl"].values()},
        )

    def test_accounts_and_pairs_are_fold_isolated(self) -> None:
        with (self.output / "public_pairs.csv").open(
            "r", encoding="utf-8", newline=""
        ) as handle:
            pairs = list(csv.DictReader(handle))
        folds_by_account: dict[str, set[str]] = {}
        for row in pairs:
            self.assertNotEqual(row["account_left_uid"], row["account_right_uid"])
            for field in ("account_left_uid", "account_right_uid"):
                folds_by_account.setdefault(row[field], set()).add(row["fold_id"])
        self.assertTrue(all(len(folds) == 1 for folds in folds_by_account.values()))

    def test_all_raw_valid_pgp_key_ids_are_absent_from_public_text(self) -> None:
        all_valid_fingerprints = set()
        with (ROOT / "3z669jwe.sql").open(
            "r", encoding="utf-8", errors="strict"
        ) as handle:
            for line in handle:
                if not line.lstrip().startswith("("):
                    continue
                fields = builder.parse_sql_tuple_line(line, 9)
                fingerprint = builder.normalize_fingerprint(fields[3])
                if len(fingerprint) == 40:
                    all_valid_fingerprints.add(fingerprint)
        known_key_ids = {
            fingerprint[-width:]
            for fingerprint in all_valid_fingerprints
            for width in (8, 16, 40)
        }
        for line in (
            self.output / "public_items_full_clean.jsonl"
        ).read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            for field in ("title_clean", "description_clean"):
                self.assertEqual(
                    builder.source_fingerprint_residuals(
                        row[field], known_key_ids
                    ),
                    set(),
                )

    def test_labels_align_one_to_one_and_quality_gates_pass(self) -> None:
        with (self.output / "public_pairs.csv").open(
            "r", encoding="utf-8", newline=""
        ) as handle:
            pairs = list(csv.DictReader(handle))
        with (self.output / "labels.csv").open(
            "r", encoding="utf-8", newline=""
        ) as handle:
            label_reader = csv.DictReader(handle)
            self.assertEqual(label_reader.fieldnames, ["pair_uid", "label"])
            labels = list(label_reader)
        self.assertEqual(
            {row["pair_uid"] for row in pairs},
            {row["pair_uid"] for row in labels},
        )
        self.assertEqual(len(labels), len({row["pair_uid"] for row in labels}))
        self.assertEqual({row["label"] for row in labels}, {"0", "1"})
        positive_count = sum(row["label"] == "1" for row in labels)
        negative_count = sum(row["label"] == "0" for row in labels)
        self.assertEqual(
            negative_count,
            positive_count * self.policy["construction"]["negative_per_positive"],
        )
        audit = json.loads((self.output / "quality_audit.json").read_text(encoding="utf-8"))
        self.assertEqual(audit["status"], "PASSED")
        self.assertTrue(all(audit["gate_results"].values()))


if __name__ == "__main__":
    unittest.main()
