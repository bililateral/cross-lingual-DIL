from __future__ import annotations

import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import step28_v13_v1_13_document_collision as subject


class Step28V13V113DocumentBytesAndCandidateKey(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.policy = subject.load_policy()

    def test_policy_is_design_only_and_all_formal_authorizations_are_false(self) -> None:
        self.assertEqual(
            self.policy["status"],
            "DESIGN_ONLY_IMPLEMENTATION_IN_PROGRESS_NO_FORMAL_AUTHORIZATION",
        )
        self.assertEqual(
            set(self.policy["formal_authorizations"]),
            set(subject.FORMAL_AUTHORIZATION_KEYS),
        )
        self.assertEqual(set(self.policy["formal_authorizations"].values()), {False})

    def test_item_document_golden_vector_preserves_exact_redactor_output(self) -> None:
        payload = subject.item_document_bytes(
            title=" 标题 ", description=" 说明\n二行 "
        )
        self.assertEqual(
            payload,
            '{"description":" 说明\\n二行 ","title":" 标题 "}'.encode("utf-8"),
        )

    def test_item_document_matches_literal_contract_on_unicode_boundaries(self) -> None:
        vectors = (
            ('引号"反斜杠\\', "\t制表\n换行\u2028行分隔"),
            ("é", "NFC"),
            ("e\u0301", "NFD"),
            ("\x00\x1f", "控制字符"),
            ("表情😀", "𠮷野家"),
        )
        for title, description in vectors:
            expected = json.dumps(
                {"description": description, "title": title},
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
            with self.subTest(title=title, description=description):
                self.assertEqual(
                    subject.item_document_bytes(
                        title=title, description=description
                    ),
                    expected,
                )
        self.assertNotEqual(
            subject.item_document_hash(title="é", description="same"),
            subject.item_document_hash(title="e\u0301", description="same"),
        )
        self.assertEqual(
            subject.item_document_hash(title=" 标题 ", description=" 说明\n二行 "),
            "9ad327b7370067228409a2bdac4c289022615a9039876e25187a859af8858c3c",
        )

    def test_item_document_rejects_non_string_fields(self) -> None:
        with self.assertRaises(subject.CollisionContractError):
            subject.item_document_hash(title="标题", description=None)  # type: ignore[arg-type]

    def test_seller_document_golden_vector_strips_drops_and_orders(self) -> None:
        profile = {
            "description_concat_top": " 普通描述 ",
            "title_concat_top": "普通标题",
            "signature_title_concat": " 签名标题 ",
            "signature_description_concat": "签名描述",
            "category_concat_top": " 类别 ",
            "unrelated": "不得进入文档",
        }
        self.assertEqual(
            subject.seller_document_bytes(profile),
            "类别\n签名标题\n普通标题\n签名描述\n普通描述".encode("utf-8"),
        )
        self.assertEqual(
            subject.seller_document_hash(profile),
            "ce5d379cf8b5073c1e572e9a6a1bd395b5678573121fc47bbeeccf02d6d76866",
        )
        profile["signature_title_concat"] = " \t "
        self.assertNotIn(b"\n\n", subject.seller_document_bytes(profile))

    def test_seller_document_rejects_missing_or_non_string_frozen_field(self) -> None:
        complete = {field: "x" for field in subject.SELLER_DOCUMENT_FIELDS}
        missing = dict(complete)
        missing.pop("title_concat_top")
        with self.assertRaises(subject.CollisionContractError):
            subject.seller_document_bytes(missing)
        wrong_type = dict(complete)
        wrong_type["title_concat_top"] = 1
        with self.assertRaises(subject.CollisionContractError):
            subject.seller_document_bytes(wrong_type)

    def test_row_hash_multiplicity_requires_all_four_cardinalities(self) -> None:
        hashes = ["1" * 64, "2" * 64]
        subject.validate_row_hash_multiplicity(
            row_count=2,
            row_hashes=hashes,
            registry_hashes=frozenset(reversed(hashes)),
            label="items",
        )
        with self.assertRaises(subject.CollisionContractError):
            subject.validate_row_hash_multiplicity(
                row_count=2,
                row_hashes=[hashes[0], hashes[0]],
                registry_hashes={hashes[0]},
                label="items",
            )
        with self.assertRaises(subject.CollisionContractError):
            subject.validate_row_hash_multiplicity(
                row_count=2,
                row_hashes=hashes,
                registry_hashes=[hashes[0], hashes[1], hashes[0]],
                label="items",
            )
        with self.assertRaises(subject.CollisionContractError):
            subject.validate_row_hash_multiplicity(
                row_count=2,
                row_hashes=hashes,
                registry_hashes={hashes[0]: True, hashes[1]: True},
                label="items",
            )
        with self.assertRaises(subject.CollisionContractError):
            subject.validate_row_hash_multiplicity(
                row_count=2,
                row_hashes={hashes[0]: True, hashes[1]: True},  # type: ignore[arg-type]
                registry_hashes=hashes,
                label="items",
            )
        with self.assertRaises(subject.CollisionContractError):
            subject.validate_row_hash_multiplicity(
                row_count=2,
                row_hashes=hashes,
                registry_hashes={hashes[0], "3" * 64},
                label="items",
            )

    def test_candidate_key_matches_frozen_golden_vector(self) -> None:
        key = bytes(range(32))
        original = bytes(key)
        observed = subject.derive_candidate_key(
            key,
            split="train",
            world_uid="w_" + "a" * 64,
            candidate_index=7,
        )
        self.assertEqual(key, original)
        self.assertEqual(
            observed.hex(),
            "e5d0bc7f9d6164119a6dd1164175dbf0f9e123b73ab8362481e7df36fdced4ca",
        )

    def test_candidate_key_is_domain_separated_across_each_input(self) -> None:
        key = b"k" * 32
        baseline = subject.derive_candidate_key(
            key, split="train", world_uid="w_" + "1" * 64, candidate_index=0
        )
        alternatives = {
            subject.derive_candidate_key(
                key,
                split="development",
                world_uid="w_" + "1" * 64,
                candidate_index=0,
            ),
            subject.derive_candidate_key(
                key, split="train", world_uid="w_" + "2" * 64, candidate_index=0
            ),
            subject.derive_candidate_key(
                key, split="train", world_uid="w_" + "1" * 64, candidate_index=1
            ),
        }
        self.assertNotIn(baseline, alternatives)
        self.assertEqual(len(alternatives), 3)

    def test_candidate_key_rejects_noncanonical_or_out_of_range_inputs(self) -> None:
        valid = {
            "document_variation_key": b"k" * 32,
            "split": "train",
            "world_uid": "w_" + "1" * 64,
            "candidate_index": 0,
        }
        for field, bad in (
            ("document_variation_key", b"short"),
            ("split", "valid"),
            ("world_uid", "w_1"),
            ("candidate_index", -1),
            ("candidate_index", 32),
            ("candidate_index", True),
        ):
            case = dict(valid)
            case[field] = bad
            with self.subTest(field=field, bad=bad):
                with self.assertRaises(subject.CollisionContractError):
                    subject.derive_candidate_key(**case)  # type: ignore[arg-type]

    def test_title_missing_attestation_replays_source_values(self) -> None:
        self.assertEqual(
            subject.verify_style_title_missing_attestation(self.policy),
            {key: 0.0 for key in ("0.05", "0.10", "0.25", "0.50", "0.75", "0.90", "0.95")},
        )

    def test_title_missing_attestation_rejects_type_key_and_order_drift(self) -> None:
        spec = self.policy["style_title_missing_attestation"]
        attestation = subject.common.load_json(
            subject.common.repo_path(spec["attestation"]["path"])
        )
        source = subject.common.load_json(subject.common.repo_path(spec["source"]["path"]))
        cases: list[tuple[str, dict, dict]] = []

        false_rows = copy.deepcopy(attestation)
        false_rows["formal_rows_created"] = False
        cases.append(("bool formal row count", false_rows, copy.deepcopy(source)))

        false_attested_value = copy.deepcopy(attestation)
        false_attested_value["observed_values"]["0.05"] = False
        cases.append(
            (
                "bool attested zero",
                false_attested_value,
                copy.deepcopy(source),
            )
        )

        string_zero = copy.deepcopy(source)
        string_zero["seller_equal_weight_quantiles"]["title_missing"]["0.05"] = "0"
        cases.append(("string zero", copy.deepcopy(attestation), string_zero))

        missing_key = copy.deepcopy(source)
        missing_key["seller_equal_weight_quantiles"]["title_missing"].pop("0.95")
        cases.append(("missing key", copy.deepcopy(attestation), missing_key))

        extra_key = copy.deepcopy(source)
        extra_key["seller_equal_weight_quantiles"]["title_missing"]["0.99"] = 0.0
        cases.append(("extra key", copy.deepcopy(attestation), extra_key))

        reversed_keys = copy.deepcopy(source)
        observed = reversed_keys["seller_equal_weight_quantiles"]["title_missing"]
        reversed_keys["seller_equal_weight_quantiles"]["title_missing"] = dict(
            reversed(list(observed.items()))
        )
        cases.append(("key order", copy.deepcopy(attestation), reversed_keys))

        for label, attestation_case, source_case in cases:
            with self.subTest(label=label), mock.patch.object(
                subject,
                "_verify_pinned_json",
                side_effect=[
                    (Path("attestation.json"), attestation_case),
                    (Path("source.json"), source_case),
                ],
            ):
                with self.assertRaises(subject.CollisionContractError):
                    subject.verify_style_title_missing_attestation(self.policy)

    def test_only_canonical_policy_path_can_produce_pass(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            alternate = Path(directory) / "policy.json"
            alternate.write_bytes(subject.DEFAULT_POLICY_PATH.read_bytes())
            with self.assertRaises(subject.CollisionContractError):
                subject.load_policy(alternate)

    def test_policy_canonical_self_hash_rejects_in_memory_drift(self) -> None:
        mutated = copy.deepcopy(self.policy)
        mutated["formal_authorizations"]["formal_seed_ceremony"] = True
        with self.assertRaises(subject.CollisionContractError):
            subject._validate_canonical_self_hash(
                mutated, expected=None, label="mutated policy"
            )

    def test_jsonl_reader_rejects_duplicate_keys_and_blank_rows(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            duplicate = Path(directory) / "duplicate.jsonl"
            duplicate.write_bytes(b'{"title":"a","title":"b"}\n')
            with self.assertRaises(subject.CollisionContractError):
                list(subject._iter_jsonl_objects(duplicate, label="duplicate"))
            blank = Path(directory) / "blank.jsonl"
            blank.write_bytes(b"\n")
            with self.assertRaises(subject.CollisionContractError):
                list(subject._iter_jsonl_objects(blank, label="blank"))

    def test_module_has_no_path_to_labels_or_private_oracle_rows(self) -> None:
        source = Path(subject.__file__).read_text(encoding="utf-8")
        for forbidden in (
            "classification_labels",
            "controller_membership.csv",
            "private_oracle/",
            "same_controller",
        ):
            self.assertNotIn(forbidden, source)
        self.assertNotIn("@lru_cache", source)

    def test_manifest_member_rejects_duplicate_path_and_pin_drift(self) -> None:
        expected = {"sha256": "1" * 64, "size_bytes": 10}
        valid = {
            "path": "observed/redacted_items.jsonl",
            "sha256": "1" * 64,
            "size_bytes": 10,
            "model_mount_allowed": True,
        }
        with self.assertRaises(subject.CollisionContractError):
            subject._manifest_member(
                {"files": [valid, dict(valid)]},
                relative_path="observed/redacted_items.jsonl",
                expected=expected,
            )
        drift = dict(valid)
        drift["sha256"] = "2" * 64
        with self.assertRaises(subject.CollisionContractError):
            subject._manifest_member(
                {"files": [drift]},
                relative_path="observed/redacted_items.jsonl",
                expected=expected,
            )

    def test_sorted_hash_registry_rejects_order_duplicate_and_digest_drift(self) -> None:
        for values, digest in (
            (["2" * 64, "1" * 64], None),
            (["1" * 64, "1" * 64], None),
            (["1" * 64], "2" * 64),
        ):
            with self.subTest(values=values, digest=digest):
                with self.assertRaises(subject.CollisionContractError):
                    subject._validate_sorted_hash_list(
                        values,
                        expected_count=len(values),
                        expected_digest=digest,
                        label="negative registry",
                    )

    def test_v1_2_release_identity_drift_fails_before_row_read(self) -> None:
        spec = self.policy["historical_sources"]["successful_v1_2"]
        bad_release = {
            "canonical_self_hash": spec["release_manifest"]["canonical_self_hash"],
            "version": spec["release_manifest"]["version"],
            "status": "WRONG",
            "run_id": spec["release_manifest"]["run_id"],
        }
        with mock.patch.object(
            subject,
            "_verify_pinned_json",
            return_value=(Path("release.json"), bad_release),
        ):
            with self.assertRaises(subject.CollisionContractError):
                subject._load_successful_v1_2(spec)

    def test_v1_12_uid_hash_method_drift_fails_before_base_load(self) -> None:
        spec = self.policy["historical_sources"]["failed_v1_12"]
        archive = {
            "canonical_self_hash": spec["archive"]["canonical_self_hash"],
            "version": spec["archive"]["version"],
            "status": spec["archive"]["status"],
            "uid_hash_method": "wrong",
            "archive_content_scope": "THIS_HASH_ONLY_ARCHIVE_ONLY_NOT_THE_STILL_EXISTING_FAILED_RUN_PAYLOAD",
            "future_registry_must_not_require_deleted_failed_payloads": True,
            "labels_or_oracle_rows_persisted_in_this_archive": False,
            "raw_identity_values_persisted_in_this_archive": False,
            "raw_item_or_seller_text_persisted_in_this_archive": False,
            "raw_private_keys_persisted_in_this_archive": False,
            "raw_uids_persisted_in_this_archive": False,
            "scientific_metrics_produced": False,
        }
        with mock.patch.object(
            subject,
            "_verify_pinned_json",
            return_value=(Path("archive.json"), archive),
        ):
            with self.assertRaises(subject.CollisionContractError):
                subject._load_failed_v1_12(spec)

    def test_historical_loader_revalidates_sources_on_every_same_process_call(self) -> None:
        overlap = frozenset(f"{index:064x}" for index in range(1, 36))
        archive = {
            "historical_v1_2_item_document_intersection_count": 35,
            "historical_v1_2_item_document_intersection_hashes": sorted(overlap),
            "historical_v1_2_item_document_intersection_hashes_sha256": (
                subject.common.canonical_sha256(sorted(overlap))
            ),
            "historical_v1_2_seller_document_intersection_count": 0,
        }
        policy = {
            "historical_sources": {
                "successful_v1_2": {},
                "failed_v1_12": {},
            }
        }
        failed_result = (
            overlap,
            frozenset(),
            frozenset(),
            {},
            frozenset(),
            archive,
        )
        with (
            mock.patch.object(subject, "load_policy", return_value=policy) as policy_loader,
            mock.patch.object(
                subject,
                "_load_successful_v1_2",
                return_value=(overlap, frozenset(), 35, 0),
            ) as successful_loader,
            mock.patch.object(
                subject, "_load_failed_v1_12", return_value=failed_result
            ) as failed_loader,
        ):
            first = subject.load_historical_exclusion_registries()
            second = subject.load_historical_exclusion_registries()
        self.assertIsNot(first, second)
        self.assertEqual(policy_loader.call_count, 2)
        self.assertEqual(successful_loader.call_count, 2)
        self.assertEqual(failed_loader.call_count, 2)


class Step28V13V113HistoricalRegistryReplay(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.registries = subject.load_historical_exclusion_registries()

    def test_successful_and_failed_document_sources_replay_exactly(self) -> None:
        registry = self.registries
        self.assertEqual(registry.successful_v1_2_item_row_count, 202071)
        self.assertEqual(registry.successful_v1_2_seller_row_count, 56000)
        self.assertEqual(registry.successful_v1_2_item_unique_count, 202043)
        self.assertEqual(registry.successful_v1_2_seller_unique_count, 56000)
        self.assertEqual(registry.failed_v1_12_item_unique_count, 100936)
        self.assertEqual(registry.failed_v1_12_seller_unique_count, 28000)
        self.assertEqual(len(registry.item_document_hashes), 302944)
        self.assertEqual(len(registry.seller_document_hashes), 84000)

    def test_identity_uid_and_capability_exclusions_replay_exactly(self) -> None:
        registry = self.registries
        self.assertEqual(len(registry.identity_value_hashes), 999996)
        self.assertEqual(len(registry.consumed_capability_commitments), 37)
        self.assertEqual(
            subject.load_policy()["historical_sources"]["failed_v1_12"][
                "commitment_counts"
            ],
            {
                "consumed_generator_capability_commitments": 28,
                "consumed_m1_capability_commitments": 5,
                "forbidden_master_seed_commitments": 4,
            },
        )
        self.assertEqual(
            {key: len(value) for key, value in registry.uid_hashes.items()},
            {
                "canonical_pair_uid": 378000,
                "controller_uid": 12000,
                "item_uid": 100946,
                "query_uid": 28000,
                "seller_uid": 28000,
                "world_uid": 1000,
            },
        )

    def test_uid_mapping_is_read_only(self) -> None:
        with self.assertRaises(TypeError):
            self.registries.uid_hashes["new"] = frozenset()  # type: ignore[index]


if __name__ == "__main__":
    unittest.main()
