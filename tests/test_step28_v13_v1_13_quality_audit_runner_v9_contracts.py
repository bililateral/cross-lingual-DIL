from __future__ import annotations

import copy
import csv
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import step28_v13_v1_13_quality_audit_runner_v9 as runner


class QualityAuditRunnerV9Contracts(unittest.TestCase):
    def test_current_policy_refuses_before_any_dataset_lookup(self) -> None:
        with self.assertRaisesRegex(
            runner.QualityAuditRunnerError, "unauthorized"
        ):
            runner.run_formal_quality_audit()

    def test_jsonl_loader_requires_newline_and_exact_row_count(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "rows.jsonl"
            path.write_text('{"a":1}\n{"a":2}\n', encoding="utf-8")
            self.assertEqual(
                runner._load_jsonl(path, expected_rows=2),
                ({"a": 1}, {"a": 2}),
            )
            path.write_text('{"a":1}', encoding="utf-8")
            with self.assertRaises(runner.QualityAuditRunnerError):
                runner._load_jsonl(path, expected_rows=1)

    def test_csv_loader_preserves_exact_header_and_row_order(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "rows.csv"
            with path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=("a", "b"))
                writer.writeheader()
                writer.writerow({"a": "1", "b": "2"})
            self.assertEqual(
                runner._load_csv(
                    path, expected_rows=1, expected_fields=("a", "b")
                ),
                ({"a": "1", "b": "2"},),
            )
            with self.assertRaisesRegex(runner.QualityAuditRunnerError, "header"):
                runner._load_csv(
                    path, expected_rows=1, expected_fields=("b", "a")
                )

    def test_verified_source_binds_bytes_and_relative_path(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            path = root / "train" / "observed" / "x.jsonl"
            path.parent.mkdir(parents=True)
            path.write_text('{"x":1}\n', encoding="utf-8")
            raw = path.read_bytes()
            record = {
                "path": "observed/x.jsonl",
                "size_bytes": len(raw),
                "sha256": hashlib.sha256(raw).hexdigest(),
                "row_count": 1,
            }
            observed_path, source = runner._verified_source(
                dataset_root=root,
                split="train",
                relative="observed/x.jsonl",
                record=record,
            )
            self.assertEqual(observed_path, path.resolve())
            self.assertEqual(source.path, "train/observed/x.jsonl")
            path.write_text('{"x":2}\n', encoding="utf-8")
            with self.assertRaisesRegex(
                runner.QualityAuditRunnerError, "bytes drift"
            ):
                runner._verified_source(
                    dataset_root=root,
                    split="train",
                    relative="observed/x.jsonl",
                    record=record,
                )

    def test_root_pin_cannot_be_supplied_outside_machine_policy(self) -> None:
        import step28_v13_v1_13_quality_channel_policy_v9 as policy_module

        policy = policy_module.load_policy()
        with self.assertRaisesRegex(
            runner.QualityAuditRunnerError, "not pinned"
        ):
            runner._root_pin_from_policy(policy)

    def test_builder_policy_binding_precedes_label_free_split_loading(self) -> None:
        builder_policy = runner.scientific.load_policy()
        policy_path = runner.scientific.DEFAULT_POLICY_PATH.resolve()
        root_manifest = {
            "execution_mode": "design_preflight",
            "scientific_use_forbidden": True,
            "formal_rows_created": 0,
            "training_started": False,
            "builder_policy_canonical_self_hash": builder_policy[
                "canonical_self_hash"
            ],
            "builder_policy_file": {
                "path": policy_path.relative_to(ROOT).as_posix(),
                "size_bytes": policy_path.stat().st_size,
                "sha256": runner._sha256_file(policy_path),
            },
        }
        self.assertEqual(
            runner._validate_builder_policy_binding(root_manifest),
            builder_policy,
        )
        mutated = copy.deepcopy(root_manifest)
        mutated["builder_policy_file"]["sha256"] = "0" * 64
        root_pin = runner.truth_capability.RootManifestPin(
            path="root_manifest.json",
            size_bytes=1,
            sha256="1" * 64,
            canonical_self_hash="2" * 64,
        )
        with (
            patch.object(
                runner,
                "_root_pin_from_policy",
                return_value=(ROOT / "unused-design-root", root_pin),
            ),
            patch.object(
                runner,
                "_load_root_manifests",
                return_value=(mutated, {}),
            ),
            patch.object(runner, "_load_split_label_free") as load_split,
        ):
            state = {"stage": "authorized_entry"}
            with self.assertRaisesRegex(
                runner.DatasetGateFailure, "builder policy binding"
            ):
                runner._run_authorized_formal_quality_audit(
                    policy={}, state=state
                )
        load_split.assert_not_called()
        self.assertEqual(state["stage"], "builder_policy_binding")

    def test_public_uid_registries_close_per_split_and_globally(self) -> None:
        loaded: dict[str, dict[str, object]] = {}
        manifests: dict[str, dict[str, object]] = {}
        global_values = {kind: set() for kind in ("world", "seller", "item", "pair")}
        for split in runner.SPLITS:
            world_uid = f"{split}_world"
            seller_a = f"{split}_seller_a"
            seller_b = f"{split}_seller_b"
            item_a = f"{split}_item_a"
            item_b = f"{split}_item_b"
            pair_uid = f"{split}_pair"
            values = {
                "world": {world_uid},
                "seller": {seller_a, seller_b},
                "item": {item_a, item_b},
                "pair": {pair_uid},
            }
            for kind in values:
                global_values[kind].update(values[kind])
            items = (
                {
                    "item_uid": item_a,
                    "seller_uid": seller_a,
                    "world_uid": world_uid,
                    "title": "甲",
                    "description": "描述甲",
                },
                {
                    "item_uid": item_b,
                    "seller_uid": seller_b,
                    "world_uid": world_uid,
                    "title": "乙",
                    "description": "描述乙",
                },
            )
            profiles = tuple(
                {
                    "seller_uid": seller_uid,
                    "category_concat_top": "类别",
                    "signature_title_concat": "标题",
                    "title_concat_top": "标题",
                    "signature_description_concat": "描述",
                    "description_concat_top": "描述",
                    "item_count": 1,
                    "title_length_stats": {},
                    "description_length_stats": {},
                    "style_stats": {},
                }
                for seller_uid in (seller_a, seller_b)
            )
            split_symbol = chr(ord("A") + runner.SPLITS.index(split))
            codes = (
                "Q" + split_symbol + ("A" * 8) + "A",
                "Q" + split_symbol + ("A" * 8) + "B",
            )
            public_code = tuple(
                {
                    "world_uid": world_uid,
                    "seller_uid": seller_uid,
                    "owned_codes": [code],
                    "item_occurrences": [
                        {"field": "title", "code": code, "is_own": True}
                    ],
                    "profile_occurrences": [],
                    "numeric_profile_deltas": {
                        field: 0.0
                        for field in runner.preparer.channel.NUMERIC_DELTA_FIELDS
                    },
                }
                for seller_uid, code in zip((seller_a, seller_b), codes)
            )
            structure_values: dict[str, object] = {
                "version": "fixture-v9",
                "world_uid": world_uid,
                "item_count": 2,
                "seller_count": 2,
                "registered_code_count": 2,
                "registered_item_occurrence_count": 0,
                "registered_visible_occurrence_expected_count": 0,
                "registered_visible_occurrence_actual_count": 0,
                "clone_directions": [],
                "neutral_receipt": {},
                "full_item_sha256": "a" * 64,
                "masked_item_sha256": "b" * 64,
                "neutral_item_sha256": "c" * 64,
                "full_profile_sha256": "d" * 64,
                "masked_profile_sha256": "e" * 64,
                "neutral_profile_sha256": "f" * 64,
                "forbidden_capability_mounted": {
                    name: False
                    for name in runner.structure_aggregator.FORBIDDEN_CAPABILITY_FIELDS
                },
            }
            structure_values.update(
                {
                    field: 0
                    for field in runner.structure_aggregator.ZERO_TOLERANCE_FIELDS
                }
            )
            structure_row = {
                field: structure_values[field]
                for field in runner.structure_aggregator.STRUCTURE_AUDIT_FIELDS
            }
            loaded[split] = {
                "worlds": ({"world_uid": world_uid, "split_ordinal": 0},),
                "surface_rows": {
                    surface: (items, profiles) for surface in runner.SURFACE_FILES
                },
                "endpoints": (
                    {
                        "canonical_pair_uid": pair_uid,
                        "world_uid": world_uid,
                        "seller_uid_left": seller_a,
                        "seller_uid_right": seller_b,
                    },
                ),
                "public_code": public_code,
                "eligibility": (
                    {
                        "world_uid": world_uid,
                        "canonical_pair_uid": pair_uid,
                        "text_probe_eligible": True,
                    },
                ),
                "structure_audit": (structure_row,),
            }
            for json_key in (
                "worlds",
                "surface_rows",
                "public_code",
                "eligibility",
                "structure_audit",
            ):
                loaded[split][json_key] = json.loads(
                    json.dumps(
                        loaded[split][json_key],
                        ensure_ascii=False,
                        sort_keys=True,
                    )
                )
            manifests[split] = {
                "uid_registries": {
                    kind: {
                        "count": len(entries),
                        "sha256": runner._registry_sha256(entries),
                    }
                    for kind, entries in values.items()
                }
            }
        root_manifest = {
            "uid_registries": {
                kind: {
                    "count": len(entries),
                    "sha256": runner._registry_sha256(entries),
                }
                for kind, entries in global_values.items()
            }
        }
        runner._validate_public_uid_registries(
            root_manifest=root_manifest,
            manifests=manifests,
            loaded=loaded,
            expected_pairs_per_world=1,
            expected_sellers_per_world=2,
            expected_excluded_pairs_per_world=0,
        )
        eligibility_backup = loaded["audit_a"]["eligibility"]
        changed_eligibility = dict(eligibility_backup[0])
        changed_eligibility["text_probe_eligible"] = False
        loaded["audit_a"]["eligibility"] = (changed_eligibility,)
        with self.assertRaisesRegex(
            runner.QualityAuditRunnerError, "eligibility cardinality"
        ):
            runner._validate_public_uid_registries(
                root_manifest=root_manifest,
                manifests=manifests,
                loaded=loaded,
                expected_pairs_per_world=1,
                expected_sellers_per_world=2,
                expected_excluded_pairs_per_world=0,
            )
        loaded["audit_a"]["eligibility"] = eligibility_backup

        surface_backup = loaded["audit_b"]["surface_rows"]
        changed_surfaces = dict(surface_backup)
        full_items, full_profiles = changed_surfaces["surface_code_neutralized"]
        changed_profile = dict(full_profiles[0])
        changed_profile["item_count"] = 2
        changed_surfaces["surface_code_neutralized"] = (
            full_items,
            (changed_profile, *full_profiles[1:]),
        )
        loaded["audit_b"]["surface_rows"] = changed_surfaces
        with self.assertRaisesRegex(
            runner.QualityAuditRunnerError, "profile item-count"
        ):
            runner._validate_public_uid_registries(
                root_manifest=root_manifest,
                manifests=manifests,
                loaded=loaded,
                expected_pairs_per_world=1,
                expected_sellers_per_world=2,
                expected_excluded_pairs_per_world=0,
            )
        loaded["audit_b"]["surface_rows"] = surface_backup

        loaded["development"]["worlds"] = loaded["train"]["worlds"]
        with self.assertRaisesRegex(
            runner.QualityAuditRunnerError, "endpoint closure|join universe|intersection"
        ):
            runner._validate_public_uid_registries(
                root_manifest=root_manifest,
                manifests=manifests,
                loaded=loaded,
                expected_pairs_per_world=1,
                expected_sellers_per_world=2,
                expected_excluded_pairs_per_world=0,
            )

    def test_loaded_six_view_hashes_counts_clones_and_neutral_uids_are_recomputed(self) -> None:
        loaded: dict[str, dict[str, object]] = {}
        for split in runner.SPLITS:
            world_uid = f"{split}_world"
            seller_a = f"{split}_seller_a"
            seller_b = f"{split}_seller_b"
            item_a = f"{split}_item_a"
            item_b = f"{split}_item_b"
            pair_uid = f"{split}_pair"
            items = (
                {
                    "item_uid": item_a,
                    "seller_uid": seller_a,
                    "world_uid": world_uid,
                    "title": "相同标题",
                    "description": "描述甲",
                },
                {
                    "item_uid": item_b,
                    "seller_uid": seller_b,
                    "world_uid": world_uid,
                    "title": "相同标题",
                    "description": "描述乙",
                },
            )
            profiles = (
                {"seller_uid": seller_a, "item_count": 1},
                {"seller_uid": seller_b, "item_count": 1},
            )
            public = (
                {
                    "world_uid": world_uid,
                    "seller_uid": seller_a,
                    "owned_codes": ["QAAAAAAAAAA"],
                    "item_occurrences": [{"field": "title"}],
                    "profile_occurrences": [],
                },
                {
                    "world_uid": world_uid,
                    "seller_uid": seller_b,
                    "owned_codes": ["QAAAAAAAAAB"],
                    "item_occurrences": [{"field": "title"}],
                    "profile_occurrences": [],
                },
            )
            neutral = {
                "per_item_template_mapping": [
                    {"item_uid": item_a},
                    {"item_uid": item_b},
                ],
                "non_code_projection_nodes": [
                    {"item_uid": item_a},
                    {"item_uid": item_b},
                ],
            }
            audit = {
                "world_uid": world_uid,
                "item_count": 2,
                "registered_code_count": 2,
                "registered_item_occurrence_count": 2,
                "registered_visible_occurrence_expected_count": 2,
                "registered_visible_occurrence_actual_count": 2,
                "clone_directions": [
                    {
                        "source_item_uid": item_a,
                        "target_item_uid": item_b,
                    }
                ],
                "neutral_receipt": neutral,
            }
            for surface, (item_hash, profile_hash) in {
                "surface_full": ("full_item_sha256", "full_profile_sha256"),
                "surface_code_masked": (
                    "masked_item_sha256",
                    "masked_profile_sha256",
                ),
                "surface_code_neutralized": (
                    "neutral_item_sha256",
                    "neutral_profile_sha256",
                ),
            }.items():
                audit[item_hash] = runner.common.canonical_sha256(items)
                audit[profile_hash] = runner.common.canonical_sha256(profiles)
            loaded[split] = {
                "surface_rows": {
                    surface: (items, profiles) for surface in runner.SURFACE_FILES
                },
                "public_code": public,
                "endpoints": (
                    {
                        "world_uid": world_uid,
                        "canonical_pair_uid": pair_uid,
                        "seller_uid_left": seller_a,
                        "seller_uid_right": seller_b,
                    },
                ),
                "eligibility": (
                    {
                        "world_uid": world_uid,
                        "canonical_pair_uid": pair_uid,
                        "text_probe_eligible": False,
                    },
                ),
                "structure_audit": (audit,),
            }
        runner._validate_loaded_structure_bindings(
            loaded=loaded, expected_clone_count_per_world=1
        )
        changed = copy.deepcopy(loaded)
        items, profiles = changed["audit_b"]["surface_rows"]["surface_code_masked"]
        altered = dict(items[0])
        altered["description"] = "篡改"
        changed["audit_b"]["surface_rows"]["surface_code_masked"] = (
            (altered, *items[1:]),
            profiles,
        )
        with self.assertRaisesRegex(
            runner.QualityAuditRunnerError, "model-view structure hash"
        ):
            runner._validate_loaded_structure_bindings(
                loaded=changed, expected_clone_count_per_world=1
            )

    def test_builder_seller_authority_covers_all_four_splits(self) -> None:
        id_key = "7" * 64
        loaded: dict[str, dict[str, object]] = {}
        records: dict[str, list[dict[str, object]]] = {}
        for split in runner.SPLITS:
            world_uid = f"{split}_world"
            sellers = [
                runner.structure.base_uid(
                    key_hex=id_key,
                    entity_kind="seller",
                    parent_uid_or_mode=world_uid,
                    ordinal=slot,
                )
                for slot in range(2)
            ]
            items = tuple(
                {
                    "world_uid": world_uid,
                    "seller_uid": seller_uid,
                    "item_uid": f"{split}_item_{slot}",
                }
                for slot, seller_uid in enumerate(sellers)
            )
            profiles = tuple({"seller_uid": seller_uid} for seller_uid in sellers)
            loaded[split] = {
                "endpoints": (
                    {
                        "world_uid": world_uid,
                        "seller_uid_left": sellers[0],
                        "seller_uid_right": sellers[1],
                    },
                ),
                "surface_rows": {
                    surface: (items, profiles) for surface in runner.SURFACE_FILES
                },
                "public_code": tuple(
                    {"world_uid": world_uid, "seller_uid": seller_uid}
                    for seller_uid in sellers
                ),
            }
            records[split] = [{"world_uid": world_uid}]
        runner._validate_builder_seller_authority(
            loaded=loaded,
            records_by_split=records,
            id_key=id_key,
            expected_sellers_per_world=2,
        )
        changed = copy.deepcopy(loaded)
        changed["audit_a"]["public_code"][0]["seller_uid"] = "forged"
        with self.assertRaisesRegex(
            runner.QualityAuditRunnerError, "seller authority"
        ):
            runner._validate_builder_seller_authority(
                loaded=changed,
                records_by_split=records,
                id_key=id_key,
                expected_sellers_per_world=2,
            )

    def test_failure_status_distinguishes_data_gate_from_auditor_failure(self) -> None:
        policy = {
            "authorization": {
                "quality_audit_run": True,
                "metric_generation": True,
            }
        }
        cases = (
            (
                runner.DatasetGateFailure("bad persisted row"),
                "DATASET_INVALIDATED",
                True,
            ),
            (
                runner.AuditorExecutionFailure("disk read failed"),
                "AUDITOR_EXECUTION_FAILED_NO_DATASET_CONCLUSION",
                False,
            ),
            (
                runner.truth_capability.QualityTruthAuditorExecutionError(
                    "truth disk read failed"
                ),
                "AUDITOR_EXECUTION_FAILED_NO_DATASET_CONCLUSION",
                False,
            ),
            (
                runner.structure_aggregator.QualityStructureAggregationError(
                    "malformed persisted structure receipt"
                ),
                "DATASET_INVALIDATED",
                True,
            ),
        )
        for failure, expected_status, cleanup_required in cases:
            with self.subTest(expected_status=expected_status):
                with (
                    patch.object(
                        runner.quality_policy_module,
                        "load_policy",
                        return_value=policy,
                    ),
                    patch.object(
                        runner,
                        "_run_authorized_formal_quality_audit",
                        side_effect=failure,
                    ),
                ):
                    receipt = runner.run_formal_quality_audit()
                self.assertEqual(receipt["status"], expected_status)
                self.assertIs(receipt["cleanup_required"], cleanup_required)
                self.assertEqual(receipt["row_level_labels_returned"], 0)
                self.assertEqual(receipt["row_level_predictions_returned"], 0)
                self.assertEqual(len(receipt["exception_message_sha256"]), 64)


if __name__ == "__main__":
    unittest.main()
