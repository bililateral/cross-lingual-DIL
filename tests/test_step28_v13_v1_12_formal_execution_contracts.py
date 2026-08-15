from __future__ import annotations

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

import step28_v13_v1_12_formal_common as formal
import step28_v13_v1_12_close_failed_run as failure_closure
import step28_v13_v1_12_finalize_release as release_finalizer
import step28_v13_v1_12_formal_executor as executor
import step28_v13_v1_12_formal_quality_audit as quality
import step28_v13_v1_12_freeze_prelock as freezer
import step28_v13_v1_12_generate_split as generator
import step28_v13_v1_12_preceremony as preceremony
import step28_v13_v1_12_seed_ceremony as ceremony


class Step28V13V112FormalExecutionContracts(unittest.TestCase):
    def test_seed_ceremony_draws_exactly_once_per_split(self) -> None:
        calls: list[int] = []

        def entropy(size: int) -> bytes:
            calls.append(size)
            return bytes([len(calls)]) * size

        material = ceremony.draw_one_shot_material(
            forbidden_master_commitments=set(), random_bytes=entropy
        )
        self.assertEqual(calls, [32, 32, 32, 32])
        self.assertEqual(material["master_draw_count"], 4)
        self.assertEqual(len(set(material["master_commitments"].values())), 4)

    def test_seed_collision_is_not_redrawn(self) -> None:
        calls: list[int] = []

        def colliding_entropy(size: int) -> bytes:
            calls.append(size)
            return b"x" * size

        with self.assertRaises(ceremony.SeedCeremonyError) as caught:
            ceremony.draw_one_shot_material(
                forbidden_master_commitments=set(),
                random_bytes=colliding_entropy,
            )
        self.assertEqual(calls, [32, 32, 32, 32])
        self.assertEqual(
            len(caught.exception.public_details["master_commitments"]), 4
        )

    def test_forbidden_master_commitment_is_not_redrawn(self) -> None:
        values = [bytes([index]) * 32 for index in range(1, 5)]
        calls = 0

        def entropy(size: int) -> bytes:
            nonlocal calls
            value = values[calls]
            calls += 1
            return value

        import hashlib

        forbidden = {hashlib.sha256(values[2]).hexdigest()}
        with self.assertRaises(ceremony.SeedCeremonyError):
            ceremony.draw_one_shot_material(
                forbidden_master_commitments=forbidden,
                random_bytes=entropy,
            )
        self.assertEqual(calls, 4)

    def test_master_never_enters_generator_or_m1_documents(self) -> None:
        material = ceremony.draw_one_shot_material(
            forbidden_master_commitments=set(),
            random_bytes=lambda size, counter=iter(range(11, 15)): bytes(
                [next(counter)]
            )
            * size,
        )
        masters, generators, m1 = ceremony._build_private_documents(
            run_id="unit", material=material
        )
        master_values = {
            document["master_hex"] for document in masters.values()
        }
        for document in [*generators.values(), *m1.values()]:
            serialized = str(document)
            self.assertTrue(all(value not in serialized for value in master_values))
            self.assertFalse(document["master_present"])
        self.assertEqual(len(ceremony._raw_member_paths()), 13)

    def test_source_closure_excludes_c40_and_failed_versions(self) -> None:
        closure = freezer.source_closure()
        paths = [record["path"] for record in closure["members"]]
        self.assertEqual(paths, sorted(paths, key=lambda value: value.encode("utf-8")))
        self.assertEqual(len(paths), len(set(paths)))
        self.assertEqual(closure["baseline_reusable_member_count"], 15)
        self.assertEqual(closure["c40_member_count"], 0)
        self.assertEqual(closure["failed_version_member_count"], 0)
        self.assertFalse(any("c40" in value.casefold() for value in paths))

    def test_audit_authorization_ladder_never_opens_a_and_b_together(self) -> None:
        draft = formal.load_and_validate_draft()["draft"]
        ladder = draft["release"]["audit_generation_ladder"]
        self.assertTrue(ladder["audit_a_authorized_before_audit_b"])
        self.assertTrue(ladder["audit_b_requires_published_audit_a"])
        self.assertFalse(ladder["both_audits_authorized_by_one_lock"])
        self.assertNotEqual(
            formal.DEFAULT_AUDIT_A_LOCK_PATH,
            formal.DEFAULT_AUDIT_B_LOCK_PATH,
        )
        audit_a_lock = {
            "status": "READY_FOR_AUDIT_A_GENERATION_ONLY",
            "authorizations": {
                "formal_audit_a_generation": True,
                "formal_audit_b_generation": False,
            },
        }
        with mock.patch.object(
            formal,
            "load_and_validate_audit_lock",
            return_value={"audit_lock": audit_a_lock},
        ):
            with self.assertRaisesRegex(
                executor.FormalExecutorError, "not authorized: audit_b"
            ):
                executor._load_public_split_authority(
                    split="audit_b",
                    lock_path=formal.DEFAULT_AUDIT_A_LOCK_PATH,
                )

    def test_frozen_m0_text_field_allowlist_is_exact(self) -> None:
        self.assertEqual(
            quality.M0_TEXT_FIELDS,
            (
                "category_concat_top",
                "signature_title_concat",
                "title_concat_top",
                "signature_description_concat",
                "description_concat_top",
            ),
        )
        profile = {field: field for field in quality.M0_TEXT_FIELDS}
        profile.update(
            {
                "seller_uid": "sel_" + "1" * 64,
                "profile_text": "forbidden join UID must not enter model text",
            }
        )
        document = quality._visible_document(profile)
        self.assertNotIn("seller_uid", document)
        self.assertNotIn("profile_text", document)
        self.assertNotIn("sel_", document)

    @unittest.skip(
        "Historical v1.12 prelock is permanently non-executable because its "
        "source closure pins the pre-v1.13 .gitignore bytes; current private "
        "custody exclusions must not be rolled back."
    )
    def test_formal_prelock_does_not_exist_before_freeze(self) -> None:
        # This test is replaced by exact prelock replay after the prelock is
        # created; until then the absence itself is the authorization gate.
        if formal.DEFAULT_PRELOCK_PATH.exists():
            legacy = formal.load_and_validate_prelock(
                formal.DEFAULT_PRELOCK_PATH
            )
            start_path = ROOT / legacy["prelock"]["custody"][
                "seed_ceremony_start_receipt_path"
            ]
            validated = ceremony.authorization.load_and_validate_authorized_prelock(
                formal.DEFAULT_PRELOCK_PATH,
                dereference_waiver_state=not start_path.exists(),
            )
            self.assertEqual(
                validated["prelock"]["status"],
                "READY_FOR_SEED_CEREMONY_ONLY",
            )
        else:
            self.assertFalse(formal.DEFAULT_PRELOCK_PATH.exists())

    def test_design_finalizer_recovers_private_manifest_only_crash(self) -> None:
        with tempfile.TemporaryDirectory(
            prefix="step28-v1-12-finalize-recovery-"
        ) as raw:
            stage = Path(raw) / "development"
            generator.build_core_stage(
                output_root=stage,
                split="development",
                world_count=1,
                design_only=True,
                progress_every=0,
            )
            original = generator._write_json_no_replace

            def interrupt_public_manifest(path: Path, value: object) -> None:
                if path.name == "split_manifest.json":
                    raise RuntimeError("injected interruption")
                original(path, value)

            with mock.patch.object(
                generator,
                "_write_json_no_replace",
                side_effect=interrupt_public_manifest,
            ):
                with self.assertRaisesRegex(RuntimeError, "injected"):
                    generator.finalize_design_stage(
                        output_root=stage,
                        split="development",
                        world_count=1,
                    )
            self.assertTrue(
                (stage / "private/private_manifest.json").exists()
            )
            self.assertFalse((stage / "public/split_manifest.json").exists())
            recovered = generator.finalize_design_stage(
                output_root=stage,
                split="development",
                world_count=1,
            )
            self.assertEqual(
                recovered["status"], "PASS_DESIGN_ONLY_PERSISTED_STAGE"
            )
            replay = generator.finalize_design_stage(
                output_root=stage,
                split="development",
                world_count=1,
            )
            self.assertEqual(replay, recovered)

    def test_quality_state_write_is_replayable_but_not_replaceable(self) -> None:
        with tempfile.TemporaryDirectory(
            prefix="step28-v1-12-quality-recovery-"
        ) as raw:
            path = Path(raw) / "quality.json"
            value = preceremony.with_canonical_self_hash(
                {"status": "PASS_TEST", "deterministic": True}
            )
            quality._publish_or_verify_json(path, value, label="test receipt")
            quality._publish_or_verify_json(path, value, label="test receipt")
            changed = dict(value)
            changed["deterministic"] = False
            with self.assertRaises(quality.FormalQualityError):
                quality._publish_or_verify_json(
                    path, changed, label="test receipt"
                )

    def test_failure_archive_write_is_exactly_recoverable(self) -> None:
        with tempfile.TemporaryDirectory(
            prefix="step28-v1-12-failure-write-"
        ) as raw:
            path = Path(raw) / "archive.json"
            payload = b'{"status":"FAIL_TEST"}\n'
            failure_closure._publish_or_verify(
                path, payload, label="test failure archive"
            )
            failure_closure._publish_or_verify(
                path, payload, label="test failure archive"
            )
            with self.assertRaises(failure_closure.FailureClosureError):
                failure_closure._publish_or_verify(
                    path,
                    b'{"status":"DIFFERENT"}\n',
                    label="test failure archive",
                )

    def test_failure_hash_recovery_only_tolerates_final_partial_line(self) -> None:
        with tempfile.TemporaryDirectory(
            prefix="step28-v1-12-failure-jsonl-"
        ) as raw:
            root = Path(raw)
            path = root / "private/oracle/identity_assets.jsonl"
            path.parent.mkdir(parents=True)
            path.write_text(
                '{"identity_value":"alpha"}\n{"identity_value":',
                encoding="utf-8",
            )
            self.assertEqual(len(failure_closure._identity_hashes(root)), 1)
            path.write_text(
                '{"identity_value":"alpha"}\n{broken\n'
                '{"identity_value":"beta"}\n',
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                failure_closure.FailureClosureError, "before EOF"
            ):
                failure_closure._identity_hashes(root)
            path.write_bytes(b'{"identity_value":"alpha"}\n\xe4\xb8')
            self.assertEqual(len(failure_closure._identity_hashes(root)), 1)
            path.write_bytes(b'{"identity_value":"alpha"}\n{broken\n')
            with self.assertRaisesRegex(
                failure_closure.FailureClosureError, "before EOF"
            ):
                failure_closure._identity_hashes(root)

    def test_publication_cleanup_recovers_validated_markers_only(self) -> None:
        with tempfile.TemporaryDirectory(
            prefix="step28-v1-12-publish-recovery-"
        ) as raw:
            stage = Path(raw) / "_staging" / "train"
            stage.mkdir(parents=True)
            paths = {
                "stage": stage,
                "core_marker": stage / "CORE_COMPLETE.json",
                "finalized_marker": stage / "STAGE_FINALIZED.json",
                "quality_marker": stage / "QUALITY_PASS.json",
            }
            statuses = (
                ("core_marker", "PASS_FORMAL_CORE_STAGE_COMPLETE"),
                ("finalized_marker", "PASS_FORMAL_STAGE_FINALIZED"),
                ("quality_marker", "PASS_FORMAL_STAGE_QUALITY"),
            )
            for key, status in statuses:
                document = preceremony.with_canonical_self_hash(
                    {"status": status, "split": "train"}
                )
                preceremony.write_bytes_no_replace_long_path(
                    paths[key],
                    (
                        json.dumps(
                            document,
                            ensure_ascii=False,
                            sort_keys=True,
                            indent=2,
                        )
                        + "\n"
                    ).encode("utf-8"),
                )
            executor._cleanup_published_stage(paths=paths, split="train")
            self.assertFalse(stage.exists())

    def test_seed_ceremony_temp_bundle_recovers_without_redraw(self) -> None:
        with tempfile.TemporaryDirectory(
            prefix="step28-v1-12-ceremony-recovery-", dir=ROOT
        ) as raw:
            root = Path(raw)
            relative = root.relative_to(ROOT).as_posix()
            prelock = preceremony.with_canonical_self_hash(
                {
                    "run_id": "unit-seed-recovery",
                    "source_closure": {"canonical_sha256": "a" * 64},
                    "custody": {
                        "private_seed_bundle_root": f"{relative}/private/seed_custody",
                        "private_seed_stage_root": f"{relative}/private/stage",
                        "seed_ceremony_start_receipt_path": f"{relative}/public/start.json",
                        "public_ceremony_receipt_path": f"{relative}/public/receipt.json",
                        "train_development_execution_lock_path": f"{relative}/public/lock.json",
                        "permanent_failure_receipt_path": f"{relative}/public/failure.json",
                    },
                }
            )
            prelock_path = root / "prelock.json"
            preceremony.write_bytes_no_replace_long_path(
                prelock_path,
                (
                    json.dumps(
                        prelock,
                        ensure_ascii=False,
                        sort_keys=True,
                        indent=2,
                    )
                    + "\n"
                ).encode("utf-8"),
            )
            validated = {
                "prelock": prelock,
                "baseline": {
                    "forbidden_master_commitments": frozenset()
                },
            }
            calls: list[int] = []

            def entropy(size: int) -> bytes:
                calls.append(size)
                return bytes([len(calls)]) * size

            with mock.patch.object(
                formal, "load_and_validate_prelock", return_value=validated
            ), mock.patch.object(
                formal, "load_and_validate_execution_lock", return_value={}
            ), mock.patch.object(
                ceremony.authorization,
                "validate_authorization_prelock_document",
                return_value={},
            ) as authorization_check:
                first = ceremony.initialize(
                    prelock_path=prelock_path, random_bytes=entropy
                )
                second = ceremony.initialize(
                    prelock_path=prelock_path,
                    random_bytes=lambda _size: self.fail(
                        "recovery must not draw entropy"
                    ),
                )
            self.assertEqual(
                [
                    call.kwargs["dereference_waiver_state"]
                    for call in authorization_check.call_args_list
                ],
                [True, True, False],
            )
            self.assertEqual(calls, [32, 32, 32, 32])
            self.assertEqual(
                first["status"], "PASS_NEW_ONE_SHOT_SEED_CEREMONY"
            )
            self.assertEqual(
                second["status"],
                "PASS_RECOVERED_EXISTING_ONE_SHOT_CEREMONY",
            )
            self.assertTrue((root / "public/start.json").exists())
            finalizer_validated = {
                "draft": {"run_id": prelock["run_id"]},
                "ceremony_receipt": preceremony.load_json_strict(
                    root / "public/receipt.json"
                ),
                "execution_lock": preceremony.load_json_strict(
                    root / "public/lock.json"
                ),
            }
            with mock.patch.object(
                formal,
                "DEFAULT_EXECUTION_LOCK_PATH",
                root / "public/lock.json",
            ):
                custody_pin = release_finalizer._validate_seed_custody_hash_only(
                    validated=finalizer_validated,
                    private_root=root / "private",
                )
            self.assertEqual(
                custody_pin["path"],
                f"{relative}/private/seed_custody/private_manifest.json",
            )
            public_bytes = (root / "public/receipt.json").read_text(
                encoding="utf-8"
            ) + (root / "public/lock.json").read_text(encoding="utf-8")
            for index in range(1, 5):
                self.assertNotIn((bytes([index]) * 32).hex(), public_bytes)

    def test_orphan_public_ceremony_artifact_permanently_closes(self) -> None:
        with tempfile.TemporaryDirectory(
            prefix="step28-v1-12-ceremony-orphan-", dir=ROOT
        ) as raw:
            root = Path(raw)
            relative = root.relative_to(ROOT).as_posix()
            prelock = preceremony.with_canonical_self_hash(
                {
                    "run_id": "unit-seed-orphan",
                    "source_closure": {"canonical_sha256": "a" * 64},
                    "custody": {
                        "private_seed_bundle_root": f"{relative}/private/final",
                        "private_seed_stage_root": f"{relative}/private/stage",
                        "seed_ceremony_start_receipt_path": f"{relative}/public/start.json",
                        "public_ceremony_receipt_path": f"{relative}/public/receipt.json",
                        "train_development_execution_lock_path": f"{relative}/public/lock.json",
                        "permanent_failure_receipt_path": f"{relative}/public/failure.json",
                    },
                }
            )
            prelock_path = root / "prelock.json"
            preceremony.write_bytes_no_replace_long_path(
                prelock_path,
                (json.dumps(prelock, sort_keys=True, indent=2) + "\n").encode(
                    "utf-8"
                ),
            )
            start = ceremony._seed_start_receipt(
                prelock_path=prelock_path, prelock=prelock
            )
            preceremony.write_bytes_no_replace_long_path(
                root / "public/start.json",
                (json.dumps(start, sort_keys=True, indent=2) + "\n").encode(
                    "utf-8"
                ),
            )
            preceremony.write_bytes_no_replace_long_path(
                root / "public/receipt.json", b"{}\n"
            )
            validated = {
                "prelock": prelock,
                "baseline": {"forbidden_master_commitments": frozenset()},
            }
            with mock.patch.object(
                formal, "load_and_validate_prelock", return_value=validated
            ), mock.patch.object(
                ceremony.authorization,
                "validate_authorization_prelock_document",
                return_value={},
            ):
                with self.assertRaisesRegex(
                    ceremony.SeedCeremonyError,
                    "without recoverable private custody",
                ):
                    ceremony.initialize(
                        prelock_path=prelock_path,
                        random_bytes=lambda _size: self.fail(
                            "orphan recovery must not draw entropy"
                        ),
                    )
            failure = preceremony.load_json_strict(
                root / "public/failure.json"
            )
            self.assertEqual(
                failure["status"],
                "FAIL_V1_12_PERMANENTLY_CLOSED_NO_RETRY",
            )
            self.assertEqual(failure["master_draw_count_before_failure"], 0)

    def test_core_start_receipt_blocks_restart_after_missing_stage(self) -> None:
        with tempfile.TemporaryDirectory(
            prefix="step28-v1-12-core-start-", dir=ROOT
        ) as raw:
            root = Path(raw)
            lock = preceremony.with_canonical_self_hash(
                {
                    "status": "READY_FOR_TRAIN_DEVELOPMENT_GENERATION",
                    "run_id": "unit-core-start",
                    "generator_capability_commitments": {
                        split: {
                            role: f"{index + 1:064x}"
                            for index, role in enumerate(formal.GENERATOR_ROLES)
                        }
                        for split in formal.SPLITS
                    },
                }
            )
            lock_path = root / "lock.json"
            preceremony.write_bytes_no_replace_long_path(
                lock_path,
                (json.dumps(lock, sort_keys=True, indent=2) + "\n").encode(
                    "utf-8"
                ),
            )
            stage = root / "private/_staging/train"
            paths = {
                "stage": stage,
                "private_final": root / "private/splits/train",
                "public_final": root / "public/train",
                "core_marker": stage / "CORE_COMPLETE.json",
                "finalized_marker": stage / "STAGE_FINALIZED.json",
                "quality_marker": stage / "QUALITY_PASS.json",
            }
            validated = {"draft": {"unit": True}}
            with mock.patch.object(
                executor,
                "_load_public_split_authority",
                return_value=(validated, lock),
            ), mock.patch.object(
                executor, "_paths", return_value=paths
            ), mock.patch.object(
                executor,
                "_load_split_authority",
                side_effect=RuntimeError("injected after durable start"),
            ) as selected:
                with self.assertRaisesRegex(RuntimeError, "durable start"):
                    executor.generate_core(
                        split="train", execution_lock_path=lock_path
                    )
                self.assertTrue(
                    executor._core_start_path(lock_path, "train").exists()
                )
                with self.assertRaisesRegex(
                    executor.FormalExecutorError,
                    "stage is missing",
                ):
                    executor.generate_core(
                        split="train", execution_lock_path=lock_path
                    )
                self.assertEqual(selected.call_count, 1)

    def test_m1_start_receipt_blocks_rematerialization(self) -> None:
        with tempfile.TemporaryDirectory(
            prefix="step28-v1-12-m1-start-", dir=ROOT
        ) as raw:
            root = Path(raw)
            lock = preceremony.with_canonical_self_hash(
                {
                    "status": "READY_FOR_TRAIN_DEVELOPMENT_GENERATION",
                    "run_id": "unit-m1-start",
                    "generator_capability_commitments": {
                        "train": {
                            role: f"{index + 1:064x}"
                            for index, role in enumerate(formal.GENERATOR_ROLES)
                        }
                    },
                    "m1_capability_commitments": {
                        role: f"{index + 101:064x}"
                        for index, role in enumerate(formal.M1_ROLES)
                    },
                }
            )
            lock_path = root / "lock.json"
            preceremony.write_bytes_no_replace_long_path(
                lock_path,
                (json.dumps(lock, sort_keys=True, indent=2) + "\n").encode(
                    "utf-8"
                ),
            )
            core_start_path = executor._core_start_path(lock_path, "train")
            core_start = executor._expected_core_start(
                split="train", lock_path=lock_path, lock=lock
            )
            preceremony.write_bytes_no_replace_long_path(
                core_start_path,
                (json.dumps(core_start, sort_keys=True, indent=2) + "\n").encode(
                    "utf-8"
                ),
            )
            core_marker = preceremony.with_canonical_self_hash(
                {
                    "status": "PASS_FORMAL_CORE_STAGE_COMPLETE",
                    "split": "train",
                    "core_start_receipt": executor._repo_pin(
                        core_start_path, include_self_hash=True
                    ),
                }
            )
            stage = root / "private/_staging/train"
            paths = {
                "stage": stage,
                "core_marker": stage / "CORE_COMPLETE.json",
                "finalized_marker": stage / "STAGE_FINALIZED.json",
            }
            validated = {"execution_lock": lock, "draft": {"unit": True}}
            with mock.patch.object(
                formal,
                "load_and_validate_execution_lock",
                return_value=validated,
            ), mock.patch.object(
                executor, "_paths", return_value=paths
            ), mock.patch.object(
                executor, "_validate_marker", return_value=core_marker
            ), mock.patch.object(
                formal,
                "load_train_m1_capability",
                side_effect=RuntimeError("injected after M1 start"),
            ) as selected:
                with self.assertRaisesRegex(RuntimeError, "after M1 start"):
                    executor.materialize_train_m1(
                        replicate=1, execution_lock_path=lock_path
                    )
                self.assertTrue(
                    executor._m1_start_path(lock_path, "m1_r01").exists()
                )
                with self.assertRaisesRegex(
                    executor.FormalExecutorError,
                    "rematerialization is forbidden",
                ):
                    executor.materialize_train_m1(
                        replicate=1, execution_lock_path=lock_path
                    )
                self.assertEqual(selected.call_count, 1)


if __name__ == "__main__":
    unittest.main()
