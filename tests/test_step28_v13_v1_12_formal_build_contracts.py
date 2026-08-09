from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import step28_v13_v1_12_exact_shortcut_preflight as shortcut
import step28_v13_v1_12_formal_common as formal
import step28_v13_v1_12_freeze_prelock as freezer
import step28_v13_v1_12_generate_split as generator
import step28_v13_v1_12_historical_identity_coverage as historical_coverage
import step28_v13_v1_12_preceremony as preceremony


class Step28V13V112FormalBuildContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.validated = formal.load_and_validate_draft()
        cls.draft = cls.validated["draft"]
        cls.design_master = bytes.fromhex(
            cls.draft["randomness"]["design_only_master_hex"]
        )

    def test_draft_is_nonauthorizing_and_full378(self) -> None:
        self.assertEqual(set(self.draft["authorizations"].values()), {False})
        shape = self.draft["dataset_shape"]
        self.assertEqual(shape["split_order"], list(formal.SPLITS))
        self.assertEqual(shape["worlds_per_split"], 500)
        self.assertEqual(shape["complete_pairs_per_world"], 378)
        self.assertEqual(shape["positive_pairs_per_world"], 20)
        self.assertEqual(shape["negative_pairs_per_world"], 358)
        self.assertFalse(shape["c40_in_any_premodel_member"])
        self.assertEqual(
            self.draft["release"]["split_manifest_version"],
            "2026-08-03-step28-v13-v1-12-full378-split-manifest-v1",
        )
        self.assertEqual(
            self.draft["release"]["release_manifest_version"],
            "2026-08-03-step28-v13-v1-12-full378-release-manifest-v1",
        )

    def test_historical_v1_2_identity_hashes_are_all_forbidden(self) -> None:
        replay = historical_coverage.build_receipt()
        persisted = preceremony.load_json_strict(
            historical_coverage.OUTPUT_PATH
        )
        self.assertEqual(replay, persisted)
        self.assertEqual(
            replay["historical_v1_2_identity_union"][
                "unique_value_hash_count"
            ],
            170_500,
        )
        self.assertEqual(
            replay["coverage"]["historical_v1_2_missing_count"], 0
        )
        self.assertEqual(replay["coverage"]["old_boundary_missing_count"], 0)
        self.assertNotIn('"identity_value":', json.dumps(replay))

    def test_capabilities_are_split_and_role_separated(self) -> None:
        values: list[str] = []
        for split in formal.SPLITS:
            capabilities = formal.derive_capabilities(
                self.design_master, split=split
            )
            self.assertEqual(
                set(capabilities["generator"]), set(formal.GENERATOR_ROLES)
            )
            self.assertEqual(
                set(capabilities["m1"]),
                set(formal.M1_ROLES) if split == "train" else set(),
            )
            values.extend(capabilities["generator"].values())
            values.extend(capabilities["m1"].values())
        self.assertEqual(len(values), len(set(values)))

    def test_execution_policy_never_contains_master_or_structure_key(self) -> None:
        capabilities = {
            split: formal.derive_capabilities(self.design_master, split=split)
            for split in formal.SPLITS
        }
        commitments = {
            split: formal.capability_commitments(capabilities[split])[
                "generator"
            ]["structure"]
            for split in formal.SPLITS
        }
        policy = formal.build_execution_policy(
            draft=self.draft,
            split="train",
            generator_capabilities=capabilities["train"]["generator"],
            structure_commitments=commitments,
        )
        serialized = json.dumps(policy, sort_keys=True)
        self.assertNotIn(self.design_master.hex(), serialized)
        self.assertNotIn(
            capabilities["train"]["generator"]["structure"], serialized
        )
        self.assertNotIn(
            capabilities["train"]["generator"]["identity_remap"],
            serialized,
        )
        self.assertNotIn(
            capabilities["train"]["generator"]["query"], serialized
        )

    def test_formal_generation_is_unavailable_before_execution_lock(self) -> None:
        with tempfile.TemporaryDirectory(prefix="step28-v1-12-test-") as raw:
            target = Path(raw) / "must_not_exist"
            with self.assertRaises(generator.SplitStageError):
                generator.build_core_stage(
                    output_root=target,
                    split="train",
                    world_count=1,
                    design_only=False,
                    progress_every=0,
                )
            self.assertFalse(target.exists())

    def test_two_world_persisted_replay_consumes_exact_trees(self) -> None:
        receipt = generator.run_two_world_design_replay()
        self.assertEqual(
            receipt["status"],
            "PASS_DESIGN_ONLY_TWO_WORLD_PERSISTED_REPLAY",
        )
        self.assertEqual(receipt["design_world_count"], 2)
        self.assertEqual(receipt["design_pair_count"], 756)
        self.assertEqual(receipt["m1_structural_receipt_count"], 5)
        self.assertFalse(receipt["formal_authorization_used"])
        self.assertEqual(receipt["formal_rows_produced"], 0)

    def test_audit_splits_use_split_specific_identity_asset_counts(self) -> None:
        with tempfile.TemporaryDirectory(
            prefix="step28-v1-12-audit-split-test-"
        ) as raw:
            for split, expected_count in (("audit_a", 84), ("audit_b", 89)):
                with self.subTest(split=split):
                    stage = Path(raw) / split
                    built = generator.build_core_stage(
                        output_root=stage,
                        split=split,
                        world_count=1,
                        design_only=True,
                        progress_every=0,
                    )
                    self.assertEqual(
                        built["generation_receipt"]["aggregate_counts"][
                            "identity_asset_count"
                        ],
                        expected_count,
                    )
                    generator.finalize_stage(
                        output_root=stage,
                        split=split,
                        world_count=1,
                        design_only=True,
                    )
                    validated = generator.validate_design_stage(
                        output_root=stage,
                        split=split,
                        world_count=1,
                    )
                    self.assertEqual(validated["pair_count"], 378)
                    identity_hashes = preceremony.load_json_strict(
                        stage / "private/audit/identity_value_hashes.json"
                    )
                    self.assertEqual(
                        identity_hashes["hash_count"], expected_count
                    )

    def test_stage_validation_rejects_rehashed_manifest_version_drift(self) -> None:
        with tempfile.TemporaryDirectory(
            prefix="step28-v1-12-manifest-version-test-"
        ) as raw:
            stage = Path(raw) / "development"
            generator.build_core_stage(
                output_root=stage,
                split="development",
                world_count=1,
                design_only=True,
                progress_every=0,
            )
            generator.finalize_design_stage(
                output_root=stage,
                split="development",
                world_count=1,
            )
            manifest_path = stage / "public/split_manifest.json"
            manifest = preceremony.load_json_strict(manifest_path)
            manifest.pop("canonical_self_hash")
            manifest["version"] = "wrong-but-self-hashed"
            manifest = preceremony.with_canonical_self_hash(manifest)
            manifest_path.write_text(
                json.dumps(
                    manifest,
                    ensure_ascii=False,
                    sort_keys=True,
                    indent=2,
                    allow_nan=False,
                )
                + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                generator.SplitStageError, "manifest semantic drift"
            ):
                generator.validate_design_stage(
                    output_root=stage,
                    split="development",
                    world_count=1,
                )

    def test_fast_shortcut_path_replays_complete_generator(self) -> None:
        audit = shortcut.validate_fast_path_parity()
        self.assertEqual(
            audit["status"], "PASS_FAST_FULL_NULL_NUISANCE_PARITY"
        )
        self.assertEqual(audit["world_count"], 2)
        self.assertEqual(audit["pair_count"], 756)

    def test_bootstrap_metrics_match_sklearn_with_ties(self) -> None:
        row_world = np.repeat(np.arange(500, dtype=np.int16), 378)
        y = np.tile(
            np.r_[np.ones(20, dtype=np.int8), np.zeros(358, dtype=np.int8)],
            500,
        )
        scores = (
            (np.arange(len(y), dtype=np.int64) * 17 + row_world) % 23
        ).astype(np.float64)
        rng = np.random.Generator(np.random.PCG64DXSM(20260803))
        multiplicities = np.vstack(
            [
                np.bincount(
                    rng.integers(0, 500, size=500), minlength=500
                )
                for _ in range(2)
            ]
        ).astype(np.int16)
        auc_values, ap_values = shortcut._bootstrap_rank_metrics(
            y=y,
            scores=scores,
            multiplicities=multiplicities,
            row_world=row_world,
            replicate_chunk=1,
            row_chunk=997,
        )
        for index in range(2):
            weights = multiplicities[index, row_world]
            reference_auc = roc_auc_score(y, scores, sample_weight=weights)
            reference_ap = average_precision_score(
                y, scores, sample_weight=weights
            )
            self.assertAlmostEqual(
                auc_values[index], max(reference_auc, 1.0 - reference_auc), 12
            )
            self.assertAlmostEqual(ap_values[index], reference_ap, 12)

    def test_exact_logistic_converges_with_frozen_audit(self) -> None:
        rng = np.random.Generator(np.random.PCG64DXSM(20260804))
        x = rng.normal(size=(500, 4))
        y = (x[:, 0] - 0.4 * x[:, 1] + rng.normal(size=500) > 0).astype(
            np.int8
        )
        artifact = shortcut.fit_exact_logistic(
            x,
            y,
            l2=1.0,
            maximum_iterations=100,
            gradient_tolerance=1e-9,
        )
        self.assertTrue(artifact.audit["solver_success"])
        self.assertLess(
            artifact.audit["iteration_count"],
            artifact.audit["maximum_iterations"],
        )
        self.assertLessEqual(
            artifact.audit["normalized_gradient"],
            artifact.audit["gradient_tolerance"],
        )

    def test_prelock_rejects_evidence_from_an_earlier_draft(self) -> None:
        versions = formal.runtime_versions()
        common_sha = preceremony.sha256_file(
            ROOT / "scripts/step28_v13_v1_12_formal_common.py"
        )
        draft_sha = preceremony.sha256_file(formal.DEFAULT_DRAFT_PATH)
        two_world = {
            "producer_path": (
                "scripts/step28_v13_v1_12_generate_split.py"
            ),
            "producer_sha256": preceremony.sha256_file(
                ROOT / "scripts/step28_v13_v1_12_generate_split.py"
            ),
            "formal_common_sha256": common_sha,
            "formal_build_draft_sha256": draft_sha,
            "runtime_versions": versions,
        }
        exact = {
            "producer_path": (
                "scripts/step28_v13_v1_12_exact_shortcut_preflight.py"
            ),
            "producer_sha256": preceremony.sha256_file(
                ROOT
                / "scripts/step28_v13_v1_12_exact_shortcut_preflight.py"
            ),
            "formal_common_sha256": common_sha,
            "formal_build_draft_sha256": draft_sha,
            "runtime_versions": versions,
        }
        closure = {"canonical_sha256": "c" * 64}
        tests = {
            "source_closure_canonical_sha256": closure["canonical_sha256"],
            "runtime_versions": versions,
        }
        freezer.validate_evidence_source_pins(
            two_world=two_world,
            shortcut=exact,
            tests=tests,
            closure=closure,
        )
        two_world["formal_build_draft_sha256"] = "0" * 64
        with self.assertRaises(freezer.FreezeError):
            freezer.validate_evidence_source_pins(
                two_world=two_world,
                shortcut=exact,
                tests=tests,
                closure=closure,
            )
        two_world["formal_build_draft_sha256"] = draft_sha
        mutations = (
            (two_world, "producer_path", "scripts/not-the-producer.py"),
            (two_world, "producer_sha256", "0" * 64),
            (two_world, "formal_common_sha256", "0" * 64),
            (two_world, "runtime_versions", {}),
            (exact, "producer_path", "scripts/not-the-producer.py"),
            (exact, "producer_sha256", "0" * 64),
            (exact, "formal_common_sha256", "0" * 64),
            (exact, "formal_build_draft_sha256", "0" * 64),
            (exact, "runtime_versions", {}),
        )
        for document, field, bad_value in mutations:
            with self.subTest(field=field, receipt=id(document)):
                original = document[field]
                document[field] = bad_value
                with self.assertRaises(freezer.FreezeError):
                    freezer.validate_evidence_source_pins(
                        two_world=two_world,
                        shortcut=exact,
                        tests=tests,
                        closure=closure,
                    )
                document[field] = original

    def test_authorizing_design_receipts_are_postclosure_files(self) -> None:
        expected_two_world = (
            "reports/step28_synthetic_chinese_dataset/design_preflights/"
            "v1_12_cleanroom_20260803/"
            "two_world_persisted_stage_receipt.postclosure_20260809.json"
        )
        expected_shortcut = (
            "reports/step28_synthetic_chinese_dataset/design_preflights/"
            "v1_12_cleanroom_20260803/"
            "exact_shortcut_preflight_receipt.postclosure_20260809.json"
        )
        self.assertEqual(
            freezer.TWO_WORLD_RECEIPT_PATH.relative_to(ROOT).as_posix(),
            expected_two_world,
        )
        self.assertEqual(
            freezer.SHORTCUT_RECEIPT_PATH.relative_to(ROOT).as_posix(),
            expected_shortcut,
        )
        self.assertIn(expected_two_world, freezer.NEW_SOURCE_PATHS)
        self.assertIn(expected_shortcut, freezer.NEW_SOURCE_PATHS)
        self.assertNotIn(
            expected_two_world.replace(".postclosure_20260809", ""),
            freezer.NEW_SOURCE_PATHS,
        )
        self.assertNotIn(
            expected_shortcut.replace(".postclosure_20260809", ""),
            freezer.NEW_SOURCE_PATHS,
        )

    def test_postclosure_design_receipts_bind_current_sources(self) -> None:
        two_world = preceremony.load_json_strict(
            freezer.TWO_WORLD_RECEIPT_PATH
        )
        exact = preceremony.load_json_strict(freezer.SHORTCUT_RECEIPT_PATH)
        preceremony.validate_canonical_self_hash(
            two_world, label="postclosure two-world receipt"
        )
        preceremony.validate_canonical_self_hash(
            exact, label="postclosure exact shortcut receipt"
        )
        closure = {"canonical_sha256": "c" * 64}
        tests = {
            "source_closure_canonical_sha256": closure["canonical_sha256"],
            "runtime_versions": formal.runtime_versions(),
        }
        freezer.validate_evidence_source_pins(
            two_world=two_world,
            shortcut=exact,
            tests=tests,
            closure=closure,
        )

    def test_real_legacy_design_receipts_are_not_authorizing(self) -> None:
        receipt_root = freezer.TWO_WORLD_RECEIPT_PATH.parent
        two_world = preceremony.load_json_strict(
            receipt_root / "two_world_persisted_stage_receipt.json"
        )
        exact = preceremony.load_json_strict(
            receipt_root / "exact_shortcut_preflight_receipt.json"
        )
        required = {
            "producer_path",
            "producer_sha256",
            "formal_common_sha256",
            "formal_build_draft_sha256",
            "runtime_versions",
        }
        self.assertEqual(required - set(two_world), required)
        self.assertEqual(required - set(exact), required)
        closure = {"canonical_sha256": "c" * 64}
        tests = {
            "source_closure_canonical_sha256": closure["canonical_sha256"],
            "runtime_versions": formal.runtime_versions(),
        }
        with self.assertRaisesRegex(
            freezer.FreezeError,
            "Prelock evidence/source closure pin drift",
        ):
            freezer.validate_evidence_source_pins(
                two_world=two_world,
                shortcut=exact,
                tests=tests,
                closure=closure,
            )

    def test_source_closure_uses_only_postclosure_design_receipts(self) -> None:
        closure = freezer.source_closure()
        paths = {record["path"] for record in closure["members"]}
        two_world = freezer.TWO_WORLD_RECEIPT_PATH.relative_to(ROOT).as_posix()
        exact = freezer.SHORTCUT_RECEIPT_PATH.relative_to(ROOT).as_posix()
        self.assertIn(two_world, paths)
        self.assertIn(exact, paths)
        self.assertIn(
            "scripts/step28_v13_v1_12_generate_split.py", paths
        )
        self.assertIn(
            "scripts/step28_v13_v1_12_exact_shortcut_preflight.py", paths
        )
        self.assertNotIn(
            two_world.replace(".postclosure_20260809", ""), paths
        )
        self.assertNotIn(
            exact.replace(".postclosure_20260809", ""), paths
        )
        freezer.require_source_closure_members_tracked(closure)


if __name__ == "__main__":
    unittest.main()
