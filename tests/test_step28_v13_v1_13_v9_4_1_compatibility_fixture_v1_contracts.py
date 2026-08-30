from __future__ import annotations

import copy
import csv
import io
import json
import re
import sys
import tempfile
import unittest
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import step28_v13_v1_13_v9_4_1_model_experiment_common_v1 as common
import step28_v13_v1_13_v9_4_1_prepare_compatibility_fixture_v1 as fixture


def csv_payload(payload: bytes) -> list[dict[str, str]]:
    return list(csv.DictReader(io.StringIO(payload.decode("utf-8-sig"))))


def jsonl_payload(payload: bytes) -> list[dict]:
    return [json.loads(line) for line in payload.decode("utf-8").splitlines()]


class CompatibilityFixtureContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.policy = common.load_policy()
        cls.payload = fixture.build_payload(cls.policy)

    def test_fixture_is_small_label_free_and_opaque(self) -> None:
        audit = self.payload["audit"]
        self.assertEqual(audit["pair_count"], 8)
        self.assertEqual(audit["seller_count"], 16)
        self.assertEqual(audit["unique_text_count"], 32)
        self.assertEqual(audit["multi_chunk_text_count"], 6)
        self.assertFalse(audit["supervised_labels_or_identity_evidence_read"])
        self.assertTrue(audit["frozen_labse_score_values_read"])
        self.assertTrue(audit["canonical_to_opaque_pair_alignment_replayed"])
        self.assertFalse(audit["canonical_seller_or_pair_ids_in_output"])
        self.assertFalse(audit["source_multiplicity_in_output"])
        joined = b"".join(self.payload[name] for name in fixture.OUTPUT_FILES)
        self.assertNotIn(b"seller_raw:", joined)
        self.assertNotIn(b"market_item.xlsx", joined)

    def test_fixture_opaque_schemas_and_multiplicity(self) -> None:
        pairs = csv_payload(self.payload["fixture_pairs.csv"])
        seller_text = jsonl_payload(self.payload["fixture_seller_text_index.jsonl"])
        self.assertEqual(len(pairs), 8)
        for row in pairs:
            self.assertRegex(row["pair_uid"], r"^pair_\d{8}$")
            self.assertRegex(row["seller_uid_left"], r"^seller_\d{8}$")
            self.assertRegex(row["seller_uid_right"], r"^seller_\d{8}$")
        self.assertTrue(seller_text)
        for row in seller_text:
            self.assertEqual(row["multiplicity"], 1)
            self.assertIn(row["field_name"], {"title", "description"})
            self.assertRegex(row["text_uid"], r"^text_\d{8}$")

    def test_fixture_chunks_exactly_reconstruct_all_texts(self) -> None:
        texts = {
            row["text_uid"]: row["text"]
            for row in jsonl_payload(self.payload["fixture_unique_texts.jsonl"])
        }
        grouped = defaultdict(list)
        for row in jsonl_payload(self.payload["fixture_shared_chunks.jsonl"]):
            grouped[row["text_uid"]].append(row)
            self.assertEqual(
                set(row["token_lengths"]),
                {
                    "pcm_multilingual_authorship",
                    "mstyledistance",
                    "multilingual_e5_large",
                    "labse",
                },
            )
            self.assertTrue(all(int(value) <= 256 for value in row["token_lengths"].values()))
        self.assertEqual(set(grouped), set(texts))
        for text_uid, rows in grouped.items():
            rows.sort(key=lambda row: row["chunk_index"])
            self.assertEqual([row["chunk_index"] for row in rows], list(range(len(rows))))
            self.assertEqual("".join(row["text"] for row in rows), texts[text_uid])

    def test_fixture_expected_scores_are_exact_six_labse_features(self) -> None:
        rows = csv_payload(self.payload["fixture_expected_labse_scores.csv"])
        self.assertEqual(len(rows), 8)
        self.assertEqual(
            list(rows[0]), ["pair_uid", *self.policy["feature_contract"]["labse6"]]
        )
        for row in rows:
            self.assertTrue(all(re.fullmatch(r"-?\d+\.\d{12}", row[name]) for name in list(row)[1:]))

    def test_unknown_pinned_pair_hash_fails_closed(self) -> None:
        altered = copy.deepcopy(self.policy)
        hashes = altered["labse_encoding"]["compatibility_fixture"][
            "selected_pair_uid_sha256s"
        ]
        hashes[0] = "0" * 64
        altered["labse_encoding"]["compatibility_fixture"][
            "selected_pair_hash_list_canonical_sha256"
        ] = common.canonical_sha256(hashes)
        paths = fixture._validate_source_pins(altered)
        with self.assertRaisesRegex(common.ModelExperimentContractError, "absent"):
            fixture._selected_public_rows(altered, paths)

    def test_published_fixture_is_exactly_pinned_and_cannot_self_resign(self) -> None:
        root = common.resolve(self.policy["outputs"]["compatibility_fixture"])
        manifest = fixture.validate_published(self.policy, root)
        for name in fixture.OUTPUT_FILES:
            self.assertEqual(self.payload[name], (root / name).read_bytes())
        self.assertEqual(
            manifest["canonical_self_hash"],
            fixture.EXPECTED_MANIFEST_CANONICAL_SELF_HASH,
        )
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory)
            for name in fixture.OUTPUT_FILES:
                (target / name).write_bytes((root / name).read_bytes())
            forged = copy.deepcopy(manifest)
            forged["pair_count"] = 0
            forged["canonical_self_hash"] = common.canonical_sha256(
                {key: value for key, value in forged.items() if key != "canonical_self_hash"}
            )
            (target / "fixture_manifest.json").write_text(
                json.dumps(forged, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
                newline="\n",
            )
            with self.assertRaisesRegex(
                common.ModelExperimentContractError, "exact-byte"
            ):
                fixture.validate_published(self.policy, target)


if __name__ == "__main__":
    unittest.main()
