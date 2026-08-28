from __future__ import annotations

import hashlib
import contextlib
import io
import json
from pathlib import Path
from types import SimpleNamespace
import sys
import tempfile
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import step28_v13_v1_13_formal_prebuild_authority_v9_4 as authority_v94
import step28_v13_v1_13_balanced_world_schedule_v9_4 as schedule_v94
import step28_v13_v1_13_joint_noise_signatures_v9_4 as signatures_v94
import step28_v13_v1_13_quality_probe_policy_v9_4 as implementation_v94


def passing_metrics() -> dict[str, object]:
    model_names = (
        f"{implementation_v94.FORMAL_VIEW}::logistic_l2",
        f"{implementation_v94.FORMAL_VIEW}::hist_gradient_boosting_depth2",
    )
    logistic_ap = implementation_v94.FORMAL_AP_BASELINE + 0.005
    tree_ap = implementation_v94.FORMAL_AP_BASELINE + 0.006
    return {
        "single_feature_maximum_symmetric_roc_auc_by_view": {
            implementation_v94.FORMAL_VIEW: 0.51,
        },
        "model_results": {
            model_names[0]: {
                "symmetric_roc_auc": 0.52,
                "average_precision": logistic_ap,
                "score_vector_sha256": "1" * 64,
            },
            model_names[1]: {
                "symmetric_roc_auc": 0.521,
                "average_precision": tree_ap,
                "score_vector_sha256": "2" * 64,
            },
        },
        "maximum_symmetric_roc_auc": 0.521,
        "maximum_average_precision_uplift": (
            tree_ap - implementation_v94.FORMAL_AP_BASELINE
        ),
        "bootstrap": {
            "replicates": 9999,
            "world_count": 500,
            "score_family_size": 2,
            "draws_raw_i8_c_sha256": implementation_v94.FORMAL_BOOTSTRAP[
                "draws_raw_i8_c_sha256"
            ],
            "family_max_symmetric_auc_vector_sha256": "3" * 64,
            "family_max_average_precision_uplift_vector_sha256": "4" * 64,
            "symmetric_auc_95_upper": 0.525,
            "average_precision_uplift_95_upper": 0.01,
        },
    }


class FormalPrebuildAuthorityV94Contracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.time_key = bytes(range(32))
        cls.time_commitment = hashlib.sha256(cls.time_key).hexdigest()
        cls.issuance_claim = authority_v94._build_issuance_claim(
            implementation_commit="a" * 40,
            implementation_tree="b" * 40,
        )
        cls.issuance_receipt = authority_v94._build_issuance_receipt(
            implementation_commit="a" * 40,
            issuance_claim_canonical_sha256=cls.issuance_claim[
                "canonical_self_sha256"
            ],
            time_key_commitment_sha256=cls.time_commitment,
        )
        cls.policy = authority_v94.build_authorization_payload(
            root=ROOT,
            implementation_commit="a" * 40,
            implementation_tree="b" * 40,
            time_key_commitment_sha256=cls.time_commitment,
            key_issuance_claim_canonical_sha256=cls.issuance_claim[
                "canonical_self_sha256"
            ],
            key_issuance_receipt_canonical_sha256=cls.issuance_receipt[
                "canonical_self_sha256"
            ],
            frozen_input_binding={
                "balanced_schedule_version": schedule_v94.VERSION,
                "train_public_design_seed": schedule_v94.PUBLIC_DESIGN_SEEDS[
                    "train"
                ],
                "development_public_design_seed": schedule_v94.PUBLIC_DESIGN_SEEDS[
                    "development"
                ],
                "balanced_schedule_maximum_iterations": schedule_v94.MAX_ITERATIONS,
                "direct_r2_plan_read": False,
                "train_schedule_commitment_sha256": "1" * 64,
                "development_schedule_commitment_sha256": "2" * 64,
                "train_latent_schedule_sha256": "3" * 64,
                "development_latent_schedule_sha256": "4" * 64,
                "schedule_pair_audit_commitment_sha256": "5" * 64,
                "noise_signature_version": signatures_v94.VERSION,
                "noise_signature_source_pins": [
                    ["market_item.xlsx", signatures_v94.WORKBOOK_SHA256],
                    [
                        "reports/step2_content_item_manifest.csv",
                        signatures_v94.MANIFEST_SHA256,
                    ],
                    [
                        "reports/step28_synthetic_chinese_dataset/"
                        "v13_dev_smoke_v1_20260727/reference/"
                        "style_source_train_sellers.csv",
                        signatures_v94.ALLOWLIST_SHA256,
                    ],
                ],
                "noise_signature_rows_sha256": (
                    signatures_v94.EXPECTED_SIGNATURE_ROWS_SHA256
                ),
                "noise_signature_set_commitment_sha256": (
                    signatures_v94.EXPECTED_SIGNATURE_SET_COMMITMENT_SHA256
                ),
            },
        )

    def _write_issuance(self, paths: authority_v94.RuntimePaths) -> None:
        authority_v94._write_new_json(
            paths.authorization_policy,
            self.policy,
        )
        authority_v94._write_new_json(
            paths.issuance_claim,
            self.issuance_claim,
        )
        authority_v94._write_new_json(
            paths.issuance_receipt,
            self.issuance_receipt,
        )

    def _stub_formal_materials(
        self,
    ) -> tuple[dict[str, object], dict[str, object], object]:
        implementation_policy = implementation_v94.load_formal_policy()
        preflight = {
            "implementation_policy": implementation_policy,
            "source_closure": {"registered_file_count": 0},
            "noise_signatures": SimpleNamespace(
                commitment={"signature_set_commitment_sha256": "1" * 64}
            ),
            "train_schedule": SimpleNamespace(
                commitment={"split_schedule_commitment_sha256": "2" * 64}
            ),
            "development_schedule": SimpleNamespace(
                commitment={"split_schedule_commitment_sha256": "3" * 64}
            ),
            "pair_receipt": {"pair_audit_commitment_sha256": "4" * 64},
        }
        prepared_commitment = {
            "prepared_commitment_sha256": "5" * 64,
            "time_key_commitment_sha256": self.time_commitment,
        }
        label_commitment = {"label_commitment_sha256": "6" * 64}
        inputs = {
            "train_prepared": SimpleNamespace(
                matrix=((0.0,),),
                commitment=prepared_commitment,
            ),
            "development_prepared": SimpleNamespace(
                matrix=((0.0,),),
                commitment=prepared_commitment,
            ),
            "train_labels": SimpleNamespace(
                values=(0,),
                row_keys=(("train", 0),),
                commitment=label_commitment,
            ),
            "development_labels": SimpleNamespace(
                values=(0,),
                row_keys=(("development", 0),),
                commitment=label_commitment,
            ),
        }
        return preflight, inputs, implementation_policy

    def test_existing_implementation_policy_cannot_be_authorized_in_place(
        self,
    ) -> None:
        payload = json.loads(implementation_v94.POLICY_PATH.read_text(encoding="utf-8"))
        payload["authorization"]["prebuild_shortcut_gate"] = True
        payload["upstream_contract"][
            "time_key_commitment_sha256"
        ] = self.time_commitment
        with self.assertRaisesRegex(
            implementation_v94.QualityProbePolicyV94Error,
            "Policy authorization drift|Policy upstream drift",
        ):
            implementation_v94.validate_policy_payload(payload)

    def test_authorization_policy_delegates_science_and_keeps_other_gates_closed(
        self,
    ) -> None:
        authority_v94.validate_authorization_payload(
            self.policy,
            root=ROOT,
        )
        self.assertTrue(
            self.policy["authorization"]["prebuild_shortcut_gate"]
        )
        self.assertFalse(self.policy["authorization"]["method_root_build"])
        self.assertFalse(
            self.policy["authorization"]["audit_truth_unsealing"]
        )
        self.assertFalse(self.policy["authorization"]["m0_m1_m2_m3"])
        self.assertTrue(
            self.policy["scientific_contract"][
                "delegated_to_frozen_implementation_policy"
            ]
        )
        self.assertEqual(
            self.policy["implementation_binding"][
                "implementation_policy_sha256"
            ],
            implementation_v94.POLICY_SHA256,
        )

    def test_policy_self_hash_and_source_bytes_are_both_enforced(self) -> None:
        tampered = json.loads(json.dumps(self.policy))
        tampered["authorization"]["method_root_build"] = True
        with self.assertRaisesRegex(
            authority_v94.FormalPrebuildAuthorityV94Error,
            "self-hash drift",
        ):
            authority_v94.validate_authorization_payload(
                tampered,
                root=ROOT,
            )

    def test_authorization_rejects_integer_boolean_substitution(self) -> None:
        for field, replacement in (
            ("prebuild_shortcut_gate", 1),
            ("method_root_build", 0),
            ("train_truth_reads_after_matrix_freeze", True),
        ):
            forged = json.loads(json.dumps(self.policy))
            forged.pop("canonical_self_sha256")
            forged["authorization"][field] = replacement
            forged = authority_v94._with_self_hash(forged)
            with self.subTest(field=field), self.assertRaisesRegex(
                authority_v94.FormalPrebuildAuthorityV94Error,
                "Authorization capability drift",
            ):
                authority_v94.validate_authorization_payload(
                    forged,
                    root=ROOT,
                )
        forged = json.loads(json.dumps(self.policy))
        forged.pop("canonical_self_sha256")
        forged["input_binding"]["train_public_design_seed"] = True
        forged = authority_v94._with_self_hash(forged)
        with self.assertRaisesRegex(
            authority_v94.FormalPrebuildAuthorityV94Error,
            "Scientific authorization drift",
        ):
            authority_v94.validate_authorization_payload(
                forged,
                root=ROOT,
            )

    def test_failed_freeze_retains_issuance_tombstone_and_forbids_second_key(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = authority_v94.runtime_paths(root)
            random_calls: list[int] = []

            def random_bytes(length: int) -> bytes:
                random_calls.append(length)
                return self.time_key

            def fake_git(_root: Path, *arguments: str) -> str:
                if arguments == ("branch", "--show-current"):
                    return authority_v94.EXPECTED_BRANCH
                if arguments == ("rev-parse", "HEAD"):
                    return "a" * 40
                if arguments == ("rev-parse", "HEAD^{tree}"):
                    return "b" * 40
                raise AssertionError(arguments)

            original_write_json = authority_v94._write_new_json

            def fail_policy_write(path: Path, payload: object) -> None:
                if path == paths.authorization_policy:
                    raise OSError("deliberate authorization policy write failure")
                original_write_json(path, payload)

            with (
                mock.patch.object(authority_v94, "_require_clean_repository"),
                mock.patch.object(authority_v94, "_git", side_effect=fake_git),
                mock.patch.object(
                    authority_v94,
                    "_collect_frozen_input_binding",
                    return_value={},
                ),
                mock.patch.object(
                    authority_v94,
                    "build_authorization_payload",
                    return_value=self.policy,
                ),
                mock.patch.object(
                    authority_v94,
                    "_write_new_json",
                    side_effect=fail_policy_write,
                ),
            ):
                with self.assertRaisesRegex(OSError, "policy write failure"):
                    authority_v94.freeze_authorization(
                        root=root,
                        random_bytes=random_bytes,
                    )
                self.assertTrue(paths.issuance_claim.is_file())
                self.assertTrue(paths.issuance_failure.is_file())
                self.assertFalse(paths.unconsumed_key.exists())
                self.assertFalse(paths.issuance_receipt.exists())
                self.assertFalse(paths.authorization_policy.exists())
                with self.assertRaisesRegex(
                    authority_v94.FormalPrebuildAuthorityV94Error,
                    "state already exists",
                ):
                    authority_v94.freeze_authorization(
                        root=root,
                        random_bytes=random_bytes,
                    )
            self.assertEqual(random_calls, [32])

    def test_freeze_removes_stale_unpublished_issuance_claim_building(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = authority_v94.runtime_paths(root)
            paths.private_root.mkdir(parents=True)
            claim_building = paths.issuance_claim.with_name(
                f"{paths.issuance_claim.name}.building"
            )
            claim_building.write_bytes(b"interrupted claim bytes")

            def fake_git(_root: Path, *arguments: str) -> str:
                if arguments == ("branch", "--show-current"):
                    return authority_v94.EXPECTED_BRANCH
                if arguments == ("rev-parse", "HEAD"):
                    return "a" * 40
                if arguments == ("rev-parse", "HEAD^{tree}"):
                    return "b" * 40
                raise AssertionError(arguments)

            def stop_at_random(_length: int) -> bytes:
                raise RuntimeError("stop after issuance claim")

            with (
                mock.patch.object(authority_v94, "_require_clean_repository"),
                mock.patch.object(authority_v94, "_git", side_effect=fake_git),
                mock.patch.object(
                    authority_v94,
                    "_collect_frozen_input_binding",
                    return_value={},
                ),
            ):
                with self.assertRaisesRegex(
                    RuntimeError,
                    "stop after issuance claim",
                ):
                    authority_v94.freeze_authorization(
                        root=root,
                        random_bytes=stop_at_random,
                    )
            self.assertFalse(claim_building.exists())
            self.assertTrue(paths.issuance_claim.is_file())
            self.assertTrue(paths.issuance_failure.is_file())

        forged = json.loads(json.dumps(self.policy))
        forged["implementation_binding"]["source_files"][0]["sha256"] = "f" * 64
        forged["implementation_binding"][
            "source_files_commitment_sha256"
        ] = authority_v94._canonical_sha256(
            forged["implementation_binding"]["source_files"]
        )
        without_self = dict(forged)
        without_self.pop("canonical_self_sha256")
        forged["canonical_self_sha256"] = authority_v94._canonical_sha256(
            without_self
        )
        with self.assertRaisesRegex(
            authority_v94.FormalPrebuildAuthorityV94Error,
            "Pinned source byte drift",
        ):
            authority_v94.validate_authorization_payload(
                forged,
                root=ROOT,
            )

    def test_wrong_private_key_is_consumed_and_closes_claimed_attempt(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            paths = authority_v94.runtime_paths(Path(directory))
            paths.private_root.mkdir(parents=True)
            paths.unconsumed_key.write_bytes(b"x" * 32)
            self._write_issuance(paths)
            authority_v94._validate_fresh_issued_state(
                self.policy,
                paths=paths,
            )
            claim = authority_v94._claim_formal_launch(
                self.policy,
                paths=paths,
            )
            with self.assertRaisesRegex(
                authority_v94.FormalPrebuildAuthorityV94Error,
                "Consumed time-key drift",
            ):
                authority_v94._consume_claimed_key(
                    self.policy,
                    paths=paths,
                    claim=claim,
                    preflight_commitment_sha256="b" * 64,
                )
            self.assertTrue(paths.claim.exists())
            self.assertTrue(paths.consumed_key.exists())
            receipt = json.loads(
                paths.consumption_receipt.read_text(encoding="utf-8")
            )
            self.assertEqual(
                receipt["status"],
                "CONSUMED_FOR_ATTEMPT_COMMITMENT_MISMATCH",
            )

    def test_valid_key_is_claimed_once_and_atomically_consumed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            paths = authority_v94.runtime_paths(Path(directory))
            paths.private_root.mkdir(parents=True)
            paths.unconsumed_key.write_bytes(self.time_key)
            self._write_issuance(paths)
            authority_v94._validate_fresh_issued_state(
                self.policy,
                paths=paths,
            )
            claim = authority_v94._claim_formal_launch(
                self.policy,
                paths=paths,
            )
            key, consumption = authority_v94._consume_claimed_key(
                self.policy,
                paths=paths,
                claim=claim,
                preflight_commitment_sha256="b" * 64,
            )
            self.assertFalse(paths.unconsumed_key.exists())
            self.assertEqual(paths.consumed_key.read_bytes(), self.time_key)
            self.assertEqual(key, self.time_key)
            self.assertTrue(paths.claim.is_file())
            self.assertTrue(paths.launch.is_file())
            self.assertEqual(
                consumption["time_key_commitment_sha256"],
                self.time_commitment,
            )
            authority_v94._validate_method_root_continuation(
                self.policy,
                paths=paths,
                claim=claim,
                consumption_receipt=consumption,
                expected_preflight_commitment_sha256="b" * 64,
            )
            paths.consumed_key.unlink()
            with self.assertRaisesRegex(
                authority_v94.FormalPrebuildAuthorityV94Error,
                "continuation drift",
            ):
                authority_v94._validate_method_root_continuation(
                    self.policy,
                    paths=paths,
                    claim=claim,
                    consumption_receipt=consumption,
                    expected_preflight_commitment_sha256="b" * 64,
                )
            with self.assertRaisesRegex(
                authority_v94.FormalPrebuildAuthorityV94Error,
                "attempt state is not fresh",
            ):
                authority_v94._validate_fresh_issued_state(
                    self.policy,
                    paths=paths,
                )

    def test_continuation_requires_issuance_files_and_exact_preflight(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            paths = authority_v94.runtime_paths(Path(directory))
            paths.private_root.mkdir(parents=True)
            paths.unconsumed_key.write_bytes(self.time_key)
            self._write_issuance(paths)
            claim = authority_v94._claim_formal_launch(
                self.policy,
                paths=paths,
            )
            _, consumption = authority_v94._consume_claimed_key(
                self.policy,
                paths=paths,
                claim=claim,
                preflight_commitment_sha256="b" * 64,
            )
            authority_v94._validate_method_root_continuation(
                self.policy,
                paths=paths,
                claim=claim,
                consumption_receipt=consumption,
                expected_preflight_commitment_sha256="b" * 64,
            )

            issuance_receipt_bytes = paths.issuance_receipt.read_bytes()
            paths.issuance_receipt.unlink()
            with self.assertRaisesRegex(
                authority_v94.FormalPrebuildAuthorityV94Error,
                "issuance receipt drift",
            ):
                authority_v94._validate_method_root_continuation(
                    self.policy,
                    paths=paths,
                    claim=claim,
                    consumption_receipt=consumption,
                    expected_preflight_commitment_sha256="b" * 64,
            )
            paths.issuance_receipt.write_bytes(issuance_receipt_bytes)

            issuance_claim_bytes = paths.issuance_claim.read_bytes()
            paths.issuance_claim.unlink()
            with self.assertRaisesRegex(
                authority_v94.FormalPrebuildAuthorityV94Error,
                "issuance claim drift",
            ):
                authority_v94._validate_method_root_continuation(
                    self.policy,
                    paths=paths,
                    claim=claim,
                    consumption_receipt=consumption,
                    expected_preflight_commitment_sha256="b" * 64,
                )
            paths.issuance_claim.write_bytes(issuance_claim_bytes)

            with self.assertRaisesRegex(
                authority_v94.FormalPrebuildAuthorityV94Error,
                "consumption lineage drift",
            ):
                authority_v94._validate_method_root_continuation(
                    self.policy,
                    paths=paths,
                    claim=claim,
                    consumption_receipt=consumption,
                    expected_preflight_commitment_sha256="c" * 64,
                )

            paths.issuance_failure.write_text("{}", encoding="utf-8")
            with self.assertRaisesRegex(
                authority_v94.FormalPrebuildAuthorityV94Error,
                "continuation drift|consumption lineage drift",
            ):
                authority_v94._validate_method_root_continuation(
                    self.policy,
                    paths=paths,
                    claim=claim,
                    consumption_receipt=consumption,
                    expected_preflight_commitment_sha256="b" * 64,
                )

    def test_consumption_receipt_rejects_noninteger_number_fields(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            paths = authority_v94.runtime_paths(Path(directory))
            paths.private_root.mkdir(parents=True)
            paths.unconsumed_key.write_bytes(self.time_key)
            self._write_issuance(paths)
            claim = authority_v94._claim_formal_launch(
                self.policy,
                paths=paths,
            )
            _, consumption = authority_v94._consume_claimed_key(
                self.policy,
                paths=paths,
                claim=claim,
                preflight_commitment_sha256="b" * 64,
            )
            for field, value in (
                ("observed_consumed_key_length_bytes", 32.0),
                ("consumption_sequence", True),
            ):
                with self.subTest(field=field):
                    tampered = dict(consumption)
                    tampered.pop("canonical_self_sha256")
                    tampered[field] = value
                    tampered = authority_v94._with_self_hash(tampered)
                    paths.consumption_receipt.write_text(
                        json.dumps(tampered),
                        encoding="utf-8",
                    )
                    with self.assertRaisesRegex(
                        authority_v94.FormalPrebuildAuthorityV94Error,
                        "consumption receipt drift",
                    ):
                        authority_v94._load_and_validate_consumption_receipt(
                            self.policy,
                            paths=paths,
                            claim=claim,
                            require_valid_key=True,
                        )

    def test_recovery_receipts_are_bound_to_policy_claim_and_public_launch(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            paths = authority_v94.runtime_paths(Path(directory))
            paths.private_root.mkdir(parents=True)
            paths.unconsumed_key.write_bytes(self.time_key)
            self._write_issuance(paths)
            claim = authority_v94._claim_formal_launch(
                self.policy,
                paths=paths,
            )
            _, consumption = authority_v94._consume_claimed_key(
                self.policy,
                paths=paths,
                claim=claim,
                preflight_commitment_sha256="b" * 64,
            )
            validated = authority_v94._load_and_validate_consumption_receipt(
                self.policy,
                paths=paths,
                claim=claim,
                require_valid_key=True,
            )
            self.assertEqual(validated, consumption)

            tampered_launch = dict(claim)
            tampered_launch.pop("canonical_self_sha256")
            tampered_launch["output_root"] = "reports/wrong"
            tampered_launch = authority_v94._with_self_hash(tampered_launch)
            paths.launch.write_text(
                json.dumps(tampered_launch),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                authority_v94.FormalPrebuildAuthorityV94Error,
                "launch claim drift",
            ):
                authority_v94._load_and_validate_launch_claim(
                    self.policy,
                    paths=paths,
                )

            paths.launch.write_text(json.dumps(claim), encoding="utf-8")
            tampered_consumption = dict(consumption)
            tampered_consumption.pop("canonical_self_sha256")
            tampered_consumption["claim_canonical_sha256"] = "f" * 64
            tampered_consumption = authority_v94._with_self_hash(
                tampered_consumption
            )
            paths.consumption_receipt.write_text(
                json.dumps(tampered_consumption),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                authority_v94.FormalPrebuildAuthorityV94Error,
                "consumption receipt drift",
            ):
                authority_v94._load_and_validate_consumption_receipt(
                    self.policy,
                    paths=paths,
                    claim=claim,
                    require_valid_key=True,
                )

    def test_invalid_recovery_state_is_closed_instead_of_retried(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            paths = authority_v94.runtime_paths(Path(directory))
            paths.private_root.mkdir(parents=True)
            paths.unconsumed_key.write_bytes(self.time_key)
            self._write_issuance(paths)
            claim = authority_v94._claim_formal_launch(
                self.policy,
                paths=paths,
            )
            tampered_launch = dict(claim)
            tampered_launch.pop("canonical_self_sha256")
            tampered_launch["output_root"] = "reports/wrong"
            paths.launch.write_text(
                json.dumps(authority_v94._with_self_hash(tampered_launch)),
                encoding="utf-8",
            )
            terminal = authority_v94._recover_claimed_attempt(
                policy=self.policy,
                paths=paths,
            )
            self.assertEqual(
                terminal["status"],
                "PRE_CONSUMPTION_MECHANICAL_FAILURE_ATTEMPT_CLOSED",
            )
            self.assertEqual(
                terminal["failure_or_completion_stage"],
                "recovery_state_validation",
            )
            self.assertEqual(
                terminal["claim_reference"]["kind"],
                "claim_file_sha256",
            )
            self.assertFalse(terminal["truth_access"]["exact"])
            self.assertFalse(terminal["same_attempt_reusable"])
            self.assertFalse(paths.unconsumed_key.exists())
            with self.assertRaisesRegex(
                authority_v94.FormalPrebuildAuthorityV94Error,
                "already terminal",
            ):
                authority_v94._recover_claimed_attempt(
                    policy=self.policy,
                    paths=paths,
                )

    def test_interrupted_postconsumption_recovery_reports_unknown_reach(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            paths = authority_v94.runtime_paths(Path(directory))
            paths.private_root.mkdir(parents=True)
            paths.unconsumed_key.write_bytes(self.time_key)
            self._write_issuance(paths)
            claim = authority_v94._claim_formal_launch(
                self.policy,
                paths=paths,
            )
            authority_v94._consume_claimed_key(
                self.policy,
                paths=paths,
                claim=claim,
                preflight_commitment_sha256="b" * 64,
            )
            terminal = authority_v94._recover_claimed_attempt(
                policy=self.policy,
                paths=paths,
            )
            self.assertEqual(
                terminal["status"],
                "POST_CONSUMPTION_MECHANICAL_FAILURE_NO_DATASET_CONCLUSION",
            )
            self.assertTrue(terminal["key_consumed"])
            self.assertFalse(terminal["truth_access"]["exact"])
            self.assertIsNone(terminal["truth_access"]["train_reads"])
            self.assertEqual(
                terminal["execution_reach"],
                {
                    "matrices_constructed": None,
                    "truth_join_completed": None,
                    "fixed_models_fit_completed": None,
                    "bootstrap_completed": None,
                },
            )
            self.assertFalse(paths.consumed_key.exists())

    def test_issuance_receipt_is_single_candidate_and_policy_bound(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            paths = authority_v94.runtime_paths(Path(directory))
            paths.private_root.mkdir(parents=True)
            paths.unconsumed_key.write_bytes(self.time_key)
            self._write_issuance(paths)
            receipt = authority_v94.validate_issuance_receipt(
                self.policy,
                paths=paths,
                verify_unconsumed_key=True,
            )
            self.assertEqual(receipt["issuance_sequence"], 1)
            self.assertEqual(receipt["additional_key_candidates_authorized"], 0)
            tampered = dict(receipt)
            tampered["additional_key_candidates_authorized"] = 1
            paths.issuance_receipt.unlink()
            authority_v94._write_new_json(paths.issuance_receipt, tampered)
            with self.assertRaisesRegex(
                authority_v94.FormalPrebuildAuthorityV94Error,
                "self-hash drift|issuance receipt drift",
            ):
                authority_v94.validate_issuance_receipt(
                    self.policy,
                    paths=paths,
                    verify_unconsumed_key=True,
                )

    def test_preconsumption_failure_retires_key_and_records_zero_truth_reads(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            paths = authority_v94.runtime_paths(Path(directory))
            paths.private_root.mkdir(parents=True)
            paths.unconsumed_key.write_bytes(self.time_key)
            self._write_issuance(paths)
            claim = authority_v94._claim_formal_launch(
                self.policy,
                paths=paths,
            )
            authority_v94._publish_mechanical_failure(
                policy=self.policy,
                claim=claim,
                paths=paths,
                stage="truth_free_preflight",
                error=RuntimeError("deliberate preconsumption failure"),
            )
            terminal = json.loads(paths.terminal.read_text(encoding="utf-8"))
            self.assertEqual(
                terminal["status"],
                "PRE_CONSUMPTION_MECHANICAL_FAILURE_ATTEMPT_CLOSED",
            )
            self.assertFalse(terminal["key_consumed"])
            self.assertEqual(terminal["truth_access"]["train_reads"], 0)
            self.assertEqual(terminal["truth_access"]["development_reads"], 0)
            self.assertFalse(paths.unconsumed_key.exists())
            self.assertFalse(paths.consumed_key.exists())

    def test_mechanical_failure_deletes_keys_and_keeps_only_small_terminal(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            paths = authority_v94.runtime_paths(Path(directory))
            paths.private_root.mkdir(parents=True)
            paths.unconsumed_key.write_bytes(self.time_key)
            self._write_issuance(paths)
            claim = authority_v94._claim_formal_launch(
                self.policy,
                paths=paths,
            )
            authority_v94._consume_claimed_key(
                self.policy,
                paths=paths,
                claim=claim,
                preflight_commitment_sha256="b" * 64,
            )
            authority_v94._publish_mechanical_failure(
                policy=self.policy,
                claim=claim,
                paths=paths,
                stage="test_stage",
                error=RuntimeError("deliberate test failure"),
            )
            self.assertFalse(paths.unconsumed_key.exists())
            self.assertFalse(paths.consumed_key.exists())
            self.assertFalse(paths.result.exists())
            terminal = json.loads(paths.terminal.read_text(encoding="utf-8"))
            self.assertEqual(
                terminal["status"],
                "POST_CONSUMPTION_MECHANICAL_FAILURE_NO_DATASET_CONCLUSION",
            )
            self.assertFalse(terminal["same_attempt_reusable"])
            self.assertLess(paths.terminal.stat().st_size, 8192)
            with self.assertRaisesRegex(
                authority_v94.FormalPrebuildAuthorityV94Error,
                "already terminal",
            ):
                authority_v94._recover_claimed_attempt(
                    policy=self.policy,
                    paths=paths,
                )

    def test_mechanical_failure_cleans_unregistered_output_artifacts(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            paths = authority_v94.runtime_paths(Path(directory))
            paths.private_root.mkdir(parents=True)
            paths.unconsumed_key.write_bytes(self.time_key)
            self._write_issuance(paths)
            claim = authority_v94._claim_formal_launch(
                self.policy,
                paths=paths,
            )
            authority_v94._consume_claimed_key(
                self.policy,
                paths=paths,
                claim=claim,
                preflight_commitment_sha256="b" * 64,
            )
            unexpected = paths.output_root / "matrix.npy"
            unexpected.write_bytes(b"not-a-formal-output")
            authority_v94._publish_mechanical_failure(
                policy=self.policy,
                claim=claim,
                paths=paths,
                stage="fit_fixed_probe_and_bootstrap",
                error=RuntimeError("deliberate failed output"),
            )
            self.assertFalse(unexpected.exists())
            self.assertEqual(
                set(authority_v94._output_root_entry_names(paths)),
                {authority_v94.LAUNCH_NAME, authority_v94.TERMINAL_NAME},
            )
            terminal = json.loads(paths.terminal.read_text(encoding="utf-8"))
            self.assertFalse(terminal["large_failed_payloads_retained"])
            self.assertFalse(
                terminal["cleanup_summary"][
                    "unexpected_output_artifacts_retained"
                ]
            )
            self.assertIsNone(
                terminal["cleanup_summary"][
                    "raw_matrix_or_score_payloads_written"
                ]
            )

    def test_consumed_key_without_receipt_is_still_recorded_as_consumed(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            paths = authority_v94.runtime_paths(Path(directory))
            paths.private_root.mkdir(parents=True)
            paths.unconsumed_key.write_bytes(self.time_key)
            self._write_issuance(paths)
            claim = authority_v94._claim_formal_launch(
                self.policy,
                paths=paths,
            )
            paths.unconsumed_key.replace(paths.consumed_key)
            self.assertFalse(paths.consumption_receipt.exists())
            authority_v94._publish_mechanical_failure(
                policy=self.policy,
                claim=claim,
                paths=paths,
                stage="consume_time_key",
                error=RuntimeError("receipt write failed after key move"),
            )
            terminal = json.loads(paths.terminal.read_text(encoding="utf-8"))
            self.assertEqual(
                terminal["status"],
                "POST_CONSUMPTION_MECHANICAL_FAILURE_NO_DATASET_CONCLUSION",
            )
            self.assertTrue(terminal["key_consumed"])
            self.assertIsNone(
                terminal["key_consumption_receipt_file_sha256"]
            )
            self.assertFalse(paths.consumed_key.exists())

    def test_terminal_write_failure_recovers_frozen_consumption_fact(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            paths = authority_v94.runtime_paths(Path(directory))
            paths.private_root.mkdir(parents=True)
            paths.unconsumed_key.write_bytes(self.time_key)
            self._write_issuance(paths)
            claim = authority_v94._claim_formal_launch(
                self.policy,
                paths=paths,
            )
            paths.unconsumed_key.replace(paths.consumed_key)
            original_write = authority_v94._write_new_json
            terminal_failures = 0

            def fail_first_terminal_write(
                path: Path, payload: dict[str, object]
            ) -> None:
                nonlocal terminal_failures
                if path == paths.terminal and terminal_failures == 0:
                    terminal_failures += 1
                    raise OSError("deliberate terminal publication failure")
                original_write(path, payload)

            with mock.patch.object(
                authority_v94,
                "_write_new_json",
                side_effect=fail_first_terminal_write,
            ):
                with self.assertRaisesRegex(
                    OSError,
                    "deliberate terminal publication failure",
                ):
                    authority_v94._publish_mechanical_failure(
                        policy=self.policy,
                        claim=claim,
                        paths=paths,
                        stage="consume_time_key",
                        error=RuntimeError(
                            "receipt write failed after key move"
                        ),
                    )
            self.assertTrue(paths.mechanical_failure_receipt.is_file())
            self.assertFalse(paths.terminal.exists())
            self.assertFalse(paths.consumed_key.exists())
            frozen = json.loads(
                paths.mechanical_failure_receipt.read_text(encoding="utf-8")
            )
            self.assertTrue(frozen["key_consumed_observed"])

            terminal = authority_v94._recover_claimed_attempt(
                policy=self.policy,
                paths=paths,
            )
            self.assertEqual(
                terminal["status"],
                "POST_CONSUMPTION_MECHANICAL_FAILURE_NO_DATASET_CONCLUSION",
            )
            self.assertTrue(terminal["key_consumed"])
            self.assertEqual(
                terminal["mechanical_failure_receipt_canonical_sha256"],
                frozen["canonical_self_sha256"],
            )

    def test_failure_fact_write_failure_preserves_observable_key_state(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            paths = authority_v94.runtime_paths(Path(directory))
            paths.private_root.mkdir(parents=True)
            paths.unconsumed_key.write_bytes(self.time_key)
            self._write_issuance(paths)
            claim = authority_v94._claim_formal_launch(
                self.policy,
                paths=paths,
            )
            paths.unconsumed_key.replace(paths.consumed_key)
            original_write = authority_v94._write_new_json

            def fail_fact_write(
                path: Path, payload: dict[str, object]
            ) -> None:
                if path == paths.mechanical_failure_receipt:
                    raise OSError("deliberate fact receipt failure")
                original_write(path, payload)

            with mock.patch.object(
                authority_v94,
                "_write_new_json",
                side_effect=fail_fact_write,
            ):
                with self.assertRaisesRegex(
                    OSError,
                    "deliberate fact receipt failure",
                ):
                    authority_v94._publish_mechanical_failure(
                        policy=self.policy,
                        claim=claim,
                        paths=paths,
                        stage="consume_time_key",
                        error=RuntimeError(
                            "receipt write failed after key move"
                        ),
                    )
            self.assertTrue(paths.consumed_key.is_file())
            self.assertFalse(paths.mechanical_failure_receipt.exists())
            self.assertFalse(paths.terminal.exists())

    def test_failure_after_input_assembly_records_exact_truth_reads(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            paths = authority_v94.runtime_paths(Path(directory))
            paths.private_root.mkdir(parents=True)
            paths.unconsumed_key.write_bytes(self.time_key)
            self._write_issuance(paths)
            claim = authority_v94._claim_formal_launch(
                self.policy,
                paths=paths,
            )
            authority_v94._consume_claimed_key(
                self.policy,
                paths=paths,
                claim=claim,
                preflight_commitment_sha256="b" * 64,
            )
            authority_v94._publish_mechanical_failure(
                policy=self.policy,
                claim=claim,
                paths=paths,
                stage="fit_fixed_probe_and_bootstrap",
                error=RuntimeError("model fit failed after input assembly"),
            )
            terminal = json.loads(paths.terminal.read_text(encoding="utf-8"))
            self.assertEqual(
                terminal["truth_access"],
                {
                    "exact": True,
                    "train_reads": 1,
                    "development_reads": 1,
                    "audit_a_reads": 0,
                    "audit_b_reads": 0,
                },
            )

    def test_run_once_uses_frozen_implementation_policy_and_publishes_pass(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = authority_v94.runtime_paths(root)
            paths.private_root.mkdir(parents=True)
            paths.unconsumed_key.write_bytes(self.time_key)
            self._write_issuance(paths)
            claim_building = paths.claim.with_name(
                f"{paths.claim.name}.building"
            )
            claim_building.write_bytes(b"interrupted launch claim bytes")
            preflight, inputs, implementation_policy = (
                self._stub_formal_materials()
            )
            captured_policy: list[object] = []

            def evaluate(**kwargs: object) -> dict[str, object]:
                captured_policy.append(kwargs["policy"])
                return passing_metrics()

            with (
                mock.patch.object(
                    authority_v94,
                    "load_authorization_policy",
                    return_value=self.policy,
                ),
                mock.patch.object(authority_v94, "_validate_policy_commit"),
                mock.patch.object(
                    authority_v94,
                    "_preflight_materials",
                    return_value=preflight,
                ),
                mock.patch.object(
                    implementation_v94,
                    "_assemble_formal_inputs_after_authorization",
                    return_value=inputs,
                ),
                mock.patch.object(
                    authority_v94.core_v94,
                    "_evaluate_family",
                    side_effect=evaluate,
                ),
            ):
                with contextlib.redirect_stdout(io.StringIO()):
                    result = authority_v94.run_once(root=root)
            self.assertEqual(
                result["status"],
                "PASSED_PREBUILD_SHORTCUT_GATE",
            )
            self.assertEqual(captured_policy, [implementation_policy])
            self.assertIsNot(captured_policy[0], self.policy)
            self.assertTrue(paths.result.is_file())
            self.assertFalse(claim_building.exists())
            self.assertTrue(
                paths.pass_terminal_validation_pending.is_file()
            )
            self.assertTrue(
                paths.pass_terminal_validation_completion.is_file()
            )
            terminal = json.loads(paths.terminal.read_text(encoding="utf-8"))
            self.assertTrue(terminal["scientific_result_valid"])
            self.assertTrue(terminal["method_root_build_eligible"])
            self.assertTrue(
                terminal["continuation_validated_at_terminal_publication"]
            )
            self.assertTrue(paths.consumed_key.is_file())
            published_result_sha256 = hashlib.sha256(
                paths.result.read_bytes()
            ).hexdigest()
            paths.terminal.unlink()
            paths.consumed_key.unlink()
            with mock.patch.object(
                authority_v94,
                "_preflight_materials",
                return_value=preflight,
            ):
                recovered = authority_v94._recover_claimed_attempt(
                    policy=self.policy,
                    paths=paths,
                )
            self.assertFalse(paths.result.exists())
            self.assertFalse(recovered["scientific_result_valid"])
            self.assertFalse(recovered["method_root_build_eligible"])
            self.assertEqual(
                recovered["invalidated_result_sha256"],
                published_result_sha256,
            )

    def test_run_once_cannot_publish_pass_after_continuation_loss(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = authority_v94.runtime_paths(root)
            paths.private_root.mkdir(parents=True)
            paths.unconsumed_key.write_bytes(self.time_key)
            self._write_issuance(paths)
            preflight, inputs, _ = self._stub_formal_materials()
            original_build_result = authority_v94._build_result

            def lose_continuation(**kwargs: object) -> dict[str, object]:
                paths.consumed_key.unlink()
                return original_build_result(**kwargs)

            with (
                mock.patch.object(
                    authority_v94,
                    "load_authorization_policy",
                    return_value=self.policy,
                ),
                mock.patch.object(authority_v94, "_validate_policy_commit"),
                mock.patch.object(
                    authority_v94,
                    "_preflight_materials",
                    return_value=preflight,
                ),
                mock.patch.object(
                    implementation_v94,
                    "_assemble_formal_inputs_after_authorization",
                    return_value=inputs,
                ),
                mock.patch.object(
                    authority_v94.core_v94,
                    "_evaluate_family",
                    return_value=passing_metrics(),
                ),
                mock.patch.object(
                    authority_v94,
                    "_build_result",
                    side_effect=lose_continuation,
                ),
            ):
                with self.assertRaisesRegex(
                    authority_v94.FormalPrebuildAuthorityV94Error,
                    "continuation drift",
                ):
                    with contextlib.redirect_stdout(io.StringIO()):
                        authority_v94.run_once(root=root)
            self.assertFalse(paths.result.exists())
            terminal = json.loads(paths.terminal.read_text(encoding="utf-8"))
            self.assertFalse(terminal["scientific_result_valid"])
            self.assertFalse(terminal["method_root_build_eligible"])
            self.assertFalse(
                terminal["continuation_validated_at_terminal_publication"]
            )
            self.assertEqual(
                terminal["status"],
                "POST_CONSUMPTION_MECHANICAL_FAILURE_NO_DATASET_CONCLUSION",
            )

    def test_third_continuation_and_failure_receipt_interruption_stays_failed(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = authority_v94.runtime_paths(root)
            paths.private_root.mkdir(parents=True)
            paths.unconsumed_key.write_bytes(self.time_key)
            self._write_issuance(paths)
            preflight, inputs, _ = self._stub_formal_materials()
            original_validate = (
                authority_v94._validate_method_root_continuation
            )
            original_write = authority_v94._write_new_json
            validation_calls = 0

            def fail_third_validation(*args: object, **kwargs: object) -> None:
                nonlocal validation_calls
                validation_calls += 1
                if validation_calls == 3:
                    raise authority_v94.FormalPrebuildAuthorityV94Error(
                        "third continuation failed before failure receipt"
                    )
                original_validate(*args, **kwargs)

            def fail_failure_receipt(path: Path, payload: object) -> None:
                if path == paths.mechanical_failure_receipt:
                    raise OSError("failure receipt publication interrupted")
                original_write(path, payload)

            with (
                mock.patch.object(
                    authority_v94,
                    "load_authorization_policy",
                    return_value=self.policy,
                ),
                mock.patch.object(authority_v94, "_validate_policy_commit"),
                mock.patch.object(
                    authority_v94,
                    "_preflight_materials",
                    return_value=preflight,
                ),
                mock.patch.object(
                    implementation_v94,
                    "_assemble_formal_inputs_after_authorization",
                    return_value=inputs,
                ),
                mock.patch.object(
                    authority_v94.core_v94,
                    "_evaluate_family",
                    return_value=passing_metrics(),
                ),
                mock.patch.object(
                    authority_v94,
                    "_validate_method_root_continuation",
                    side_effect=fail_third_validation,
                ),
                mock.patch.object(
                    authority_v94,
                    "_write_new_json",
                    side_effect=fail_failure_receipt,
                ),
            ):
                with self.assertRaisesRegex(
                    OSError,
                    "failure receipt publication interrupted",
                ):
                    with contextlib.redirect_stdout(io.StringIO()):
                        authority_v94.run_once(root=root)

            self.assertTrue(paths.result.is_file())
            self.assertTrue(paths.pass_terminal_validation_pending.is_file())
            self.assertFalse(
                paths.pass_terminal_validation_completion.exists()
            )
            self.assertFalse(paths.mechanical_failure_receipt.exists())
            self.assertFalse(paths.terminal.exists())

            with mock.patch.object(
                authority_v94,
                "_preflight_materials",
                return_value=preflight,
            ):
                recovered = authority_v94._recover_claimed_attempt(
                    policy=self.policy,
                    paths=paths,
                )
            self.assertFalse(paths.result.exists())
            self.assertEqual(
                recovered["status"],
                "POST_CONSUMPTION_MECHANICAL_FAILURE_NO_DATASET_CONCLUSION",
            )
            self.assertFalse(recovered["scientific_result_valid"])
            self.assertFalse(recovered["method_root_build_eligible"])

    def test_third_continuation_failure_cannot_recover_to_pass(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = authority_v94.runtime_paths(root)
            paths.private_root.mkdir(parents=True)
            paths.unconsumed_key.write_bytes(self.time_key)
            self._write_issuance(paths)
            preflight, inputs, _ = self._stub_formal_materials()
            original_validate = (
                authority_v94._validate_method_root_continuation
            )
            call_count = 0

            def fail_third_continuation(*args: object, **kwargs: object) -> None:
                nonlocal call_count
                call_count += 1
                if call_count == 3:
                    raise authority_v94.FormalPrebuildAuthorityV94Error(
                        "transient third continuation failure"
                    )
                original_validate(*args, **kwargs)

            with (
                mock.patch.object(
                    authority_v94,
                    "load_authorization_policy",
                    return_value=self.policy,
                ),
                mock.patch.object(authority_v94, "_validate_policy_commit"),
                mock.patch.object(
                    authority_v94,
                    "_preflight_materials",
                    return_value=preflight,
                ),
                mock.patch.object(
                    implementation_v94,
                    "_assemble_formal_inputs_after_authorization",
                    return_value=inputs,
                ),
                mock.patch.object(
                    authority_v94.core_v94,
                    "_evaluate_family",
                    return_value=passing_metrics(),
                ),
                mock.patch.object(
                    authority_v94,
                    "_validate_method_root_continuation",
                    side_effect=fail_third_continuation,
                ),
            ):
                with self.assertRaisesRegex(
                    authority_v94.FormalPrebuildAuthorityV94Error,
                    "transient third continuation failure",
                ):
                    with contextlib.redirect_stdout(io.StringIO()):
                        authority_v94.run_once(root=root)
            self.assertEqual(call_count, 3)
            self.assertFalse(paths.result.exists())
            terminal = json.loads(paths.terminal.read_text(encoding="utf-8"))
            self.assertEqual(
                terminal["status"],
                "POST_CONSUMPTION_MECHANICAL_FAILURE_NO_DATASET_CONCLUSION",
            )
            self.assertFalse(terminal["scientific_result_valid"])
            self.assertFalse(terminal["method_root_build_eligible"])

    def test_recovery_handles_terminal_and_failure_buildings_monotonically(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = authority_v94.runtime_paths(root)
            paths.private_root.mkdir(parents=True)
            paths.unconsumed_key.write_bytes(self.time_key)
            self._write_issuance(paths)
            preflight, inputs, _ = self._stub_formal_materials()
            with (
                mock.patch.object(
                    authority_v94,
                    "load_authorization_policy",
                    return_value=self.policy,
                ),
                mock.patch.object(authority_v94, "_validate_policy_commit"),
                mock.patch.object(
                    authority_v94,
                    "_preflight_materials",
                    return_value=preflight,
                ),
                mock.patch.object(
                    implementation_v94,
                    "_assemble_formal_inputs_after_authorization",
                    return_value=inputs,
                ),
                mock.patch.object(
                    authority_v94.core_v94,
                    "_evaluate_family",
                    return_value=passing_metrics(),
                ),
            ):
                with contextlib.redirect_stdout(io.StringIO()):
                    authority_v94.run_once(root=root)
            paths.terminal.unlink()
            terminal_building = paths.terminal.with_name(
                f"{paths.terminal.name}.building"
            )
            terminal_building.write_bytes(b"interrupted terminal bytes")
            with mock.patch.object(
                authority_v94,
                "_preflight_materials",
                return_value=preflight,
            ):
                recovered = authority_v94._recover_claimed_attempt(
                    policy=self.policy,
                    paths=paths,
                )
            self.assertFalse(terminal_building.exists())
            self.assertEqual(
                recovered["status"],
                "PASSED_PREBUILD_SHORTCUT_GATE",
            )
            self.assertTrue(paths.terminal.is_file())
            paths.terminal.unlink()
            failure_building = paths.mechanical_failure_receipt.with_name(
                f"{paths.mechanical_failure_receipt.name}.building"
            )
            failure_building.write_bytes(b"interrupted failure receipt")
            recovered_failure = authority_v94._recover_claimed_attempt(
                policy=self.policy,
                paths=paths,
            )
            self.assertFalse(paths.result.exists())
            self.assertFalse(failure_building.exists())
            self.assertEqual(
                recovered_failure["status"],
                "POST_CONSUMPTION_MECHANICAL_FAILURE_NO_DATASET_CONCLUSION",
            )
            self.assertFalse(recovered_failure["scientific_result_valid"])
            self.assertFalse(recovered_failure["method_root_build_eligible"])

    def test_recovery_validation_and_failure_receipt_interruption_stays_failed(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = authority_v94.runtime_paths(root)
            paths.private_root.mkdir(parents=True)
            paths.unconsumed_key.write_bytes(self.time_key)
            self._write_issuance(paths)
            preflight, inputs, _ = self._stub_formal_materials()
            with (
                mock.patch.object(
                    authority_v94,
                    "load_authorization_policy",
                    return_value=self.policy,
                ),
                mock.patch.object(authority_v94, "_validate_policy_commit"),
                mock.patch.object(
                    authority_v94,
                    "_preflight_materials",
                    return_value=preflight,
                ),
                mock.patch.object(
                    implementation_v94,
                    "_assemble_formal_inputs_after_authorization",
                    return_value=inputs,
                ),
                mock.patch.object(
                    authority_v94.core_v94,
                    "_evaluate_family",
                    return_value=passing_metrics(),
                ),
            ):
                with contextlib.redirect_stdout(io.StringIO()):
                    authority_v94.run_once(root=root)
            paths.terminal.unlink()
            original_write = authority_v94._write_new_json

            def fail_failure_receipt(path: Path, payload: object) -> None:
                if path == paths.mechanical_failure_receipt:
                    raise OSError("recovery failure receipt interrupted")
                original_write(path, payload)

            with (
                mock.patch.object(
                    authority_v94,
                    "_preflight_materials",
                    return_value=preflight,
                ),
                mock.patch.object(
                    authority_v94,
                    "_validate_method_root_continuation",
                    side_effect=authority_v94.FormalPrebuildAuthorityV94Error(
                        "recovery continuation failed"
                    ),
                ),
                mock.patch.object(
                    authority_v94,
                    "_write_new_json",
                    side_effect=fail_failure_receipt,
                ),
            ):
                with self.assertRaisesRegex(
                    OSError,
                    "recovery failure receipt interrupted",
                ):
                    authority_v94._recover_claimed_attempt(
                        policy=self.policy,
                        paths=paths,
                    )

            self.assertTrue(paths.result.is_file())
            self.assertFalse(
                paths.pass_terminal_validation_completion.exists()
            )
            self.assertTrue(
                paths.pass_terminal_validation_revalidating.is_file()
            )
            self.assertFalse(paths.mechanical_failure_receipt.exists())
            self.assertFalse(paths.terminal.exists())

            recovered = authority_v94._recover_claimed_attempt(
                policy=self.policy,
                paths=paths,
            )
            self.assertFalse(paths.result.exists())
            self.assertFalse(
                paths.pass_terminal_validation_revalidating.exists()
            )
            self.assertEqual(
                recovered["status"],
                "POST_CONSUMPTION_MECHANICAL_FAILURE_NO_DATASET_CONCLUSION",
            )
            self.assertFalse(recovered["scientific_result_valid"])
            self.assertFalse(recovered["method_root_build_eligible"])

    def test_launch_claim_validation_failure_is_fenced_before_recovery(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = authority_v94.runtime_paths(root)
            paths.private_root.mkdir(parents=True)
            paths.unconsumed_key.write_bytes(self.time_key)
            self._write_issuance(paths)
            preflight, inputs, _ = self._stub_formal_materials()
            with (
                mock.patch.object(
                    authority_v94,
                    "load_authorization_policy",
                    return_value=self.policy,
                ),
                mock.patch.object(authority_v94, "_validate_policy_commit"),
                mock.patch.object(
                    authority_v94,
                    "_preflight_materials",
                    return_value=preflight,
                ),
                mock.patch.object(
                    implementation_v94,
                    "_assemble_formal_inputs_after_authorization",
                    return_value=inputs,
                ),
                mock.patch.object(
                    authority_v94.core_v94,
                    "_evaluate_family",
                    return_value=passing_metrics(),
                ),
            ):
                with contextlib.redirect_stdout(io.StringIO()):
                    authority_v94.run_once(root=root)
            paths.terminal.unlink()
            original_write = authority_v94._write_new_json

            def fail_failure_receipt(path: Path, payload: object) -> None:
                if path == paths.mechanical_failure_receipt:
                    raise OSError("launch failure receipt interrupted")
                original_write(path, payload)

            with (
                mock.patch.object(
                    authority_v94,
                    "_load_and_validate_launch_claim",
                    side_effect=authority_v94.FormalPrebuildAuthorityV94Error(
                        "transient launch claim validation failure"
                    ),
                ),
                mock.patch.object(
                    authority_v94,
                    "_write_new_json",
                    side_effect=fail_failure_receipt,
                ),
            ):
                with self.assertRaisesRegex(
                    OSError,
                    "launch failure receipt interrupted",
                ):
                    authority_v94._recover_claimed_attempt(
                        policy=self.policy,
                        paths=paths,
                    )

            self.assertTrue(paths.result.is_file())
            self.assertFalse(
                paths.pass_terminal_validation_completion.exists()
            )
            self.assertTrue(
                paths.pass_terminal_validation_revalidating.is_file()
            )
            self.assertFalse(paths.mechanical_failure_receipt.exists())
            self.assertFalse(paths.terminal.exists())

            recovered = authority_v94._recover_claimed_attempt(
                policy=self.policy,
                paths=paths,
            )
            self.assertFalse(paths.result.exists())
            self.assertFalse(
                paths.pass_terminal_validation_revalidating.exists()
            )
            self.assertEqual(
                recovered["status"],
                "POST_CONSUMPTION_MECHANICAL_FAILURE_NO_DATASET_CONCLUSION",
            )
            self.assertFalse(recovered["scientific_result_valid"])
            self.assertFalse(recovered["method_root_build_eligible"])

    def test_revalidation_move_failure_is_fenced_before_recovery(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = authority_v94.runtime_paths(root)
            paths.private_root.mkdir(parents=True)
            paths.unconsumed_key.write_bytes(self.time_key)
            self._write_issuance(paths)
            preflight, inputs, _ = self._stub_formal_materials()
            with (
                mock.patch.object(
                    authority_v94,
                    "load_authorization_policy",
                    return_value=self.policy,
                ),
                mock.patch.object(authority_v94, "_validate_policy_commit"),
                mock.patch.object(
                    authority_v94,
                    "_preflight_materials",
                    return_value=preflight,
                ),
                mock.patch.object(
                    implementation_v94,
                    "_assemble_formal_inputs_after_authorization",
                    return_value=inputs,
                ),
                mock.patch.object(
                    authority_v94.core_v94,
                    "_evaluate_family",
                    return_value=passing_metrics(),
                ),
            ):
                with contextlib.redirect_stdout(io.StringIO()):
                    authority_v94.run_once(root=root)
            paths.terminal.unlink()
            original_write = authority_v94._write_new_json
            original_replace = authority_v94.os.replace

            def fail_failure_receipt(path: Path, payload: object) -> None:
                if path == paths.mechanical_failure_receipt:
                    raise OSError("move failure receipt interrupted")
                original_write(path, payload)

            def fail_completion_move(source: object, target: object) -> None:
                if (
                    Path(source) == paths.pass_terminal_validation_completion
                    and Path(target)
                    == paths.pass_terminal_validation_revalidating
                ):
                    raise OSError("completion move interrupted")
                original_replace(source, target)

            with (
                mock.patch.object(
                    authority_v94.os,
                    "replace",
                    side_effect=fail_completion_move,
                ),
                mock.patch.object(
                    authority_v94,
                    "_write_new_json",
                    side_effect=fail_failure_receipt,
                ),
            ):
                with self.assertRaisesRegex(
                    OSError,
                    "move failure receipt interrupted",
                ):
                    authority_v94._recover_claimed_attempt(
                        policy=self.policy,
                        paths=paths,
                    )

            self.assertTrue(paths.result.is_file())
            self.assertTrue(
                paths.pass_terminal_validation_completion.is_file()
            )
            self.assertTrue(
                paths.pass_terminal_validation_revalidating.is_file()
            )
            marker = json.loads(
                paths.pass_terminal_validation_revalidating.read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(
                marker["status"],
                "PASS_RECOVERY_REVALIDATION_CLAIMED",
            )
            self.assertFalse(paths.mechanical_failure_receipt.exists())
            self.assertFalse(paths.terminal.exists())

            recovered = authority_v94._recover_claimed_attempt(
                policy=self.policy,
                paths=paths,
            )
            self.assertFalse(paths.result.exists())
            self.assertTrue(
                paths.pass_terminal_validation_completion.is_file()
            )
            self.assertFalse(
                paths.pass_terminal_validation_revalidating.exists()
            )
            self.assertEqual(
                recovered["status"],
                "POST_CONSUMPTION_MECHANICAL_FAILURE_NO_DATASET_CONCLUSION",
            )
            self.assertFalse(recovered["scientific_result_valid"])
            self.assertFalse(recovered["method_root_build_eligible"])

    def test_public_result_contains_hashes_not_private_rows_or_vectors(self) -> None:
        metrics = passing_metrics()
        comparison = implementation_v94._compare_formal_gates(
            metrics,
            implementation_v94.load_formal_policy(),
        )
        commitment = {
            "example_sha256": "5" * 64,
            "time_key_commitment_sha256": self.time_commitment,
        }
        inputs = {
            "train_prepared": SimpleNamespace(commitment=commitment),
            "development_prepared": SimpleNamespace(commitment=commitment),
            "train_labels": SimpleNamespace(commitment=commitment),
            "development_labels": SimpleNamespace(commitment=commitment),
        }
        claim = {"canonical_self_sha256": "6" * 64}
        consumption_receipt = {"canonical_self_sha256": "8" * 64}
        preflight = {
            "source_closure": {"file_sha256": []},
            "pair_receipt": {"pair_audit_commitment_sha256": "7" * 64},
        }
        result = authority_v94._build_result(
            policy=self.policy,
            claim=claim,
            consumption_receipt=consumption_receipt,
            preflight=preflight,
            inputs=inputs,
            metrics=metrics,
            comparison=comparison,
        )
        authority_v94._validate_result_context(
            result,
            policy=self.policy,
            claim=claim,
            consumption_receipt=consumption_receipt,
        )
        encoded = json.dumps(result, ensure_ascii=False)
        self.assertNotIn("time_key_hex", encoded)
        self.assertNotIn("controller_groups_by_world", encoded)
        self.assertNotIn("row_labels", encoded)
        self.assertNotIn("matrix_values", encoded)
        self.assertNotIn('"score_vector"', encoded)
        self.assertFalse(result["decision"]["authorizes_training"])
        self.assertFalse(result["decision"]["authorizes_method_root_build"])
        self.assertTrue(result["decision"]["method_root_build_eligible"])
        self.assertTrue(result["time_index_continuation"]["eligible"])
        self.assertEqual(
            result["time_index_continuation"]["commitment_sha256"],
            self.time_commitment,
        )
        tampered = json.loads(json.dumps(result))
        tampered.pop("canonical_self_sha256")
        tampered["authorization_policy_canonical_sha256"] = "f" * 64
        tampered = authority_v94._with_self_hash(tampered)
        with self.assertRaisesRegex(
            authority_v94.FormalPrebuildAuthorityV94Error,
            "result context drift",
        ):
            authority_v94._validate_result_context(
                tampered,
                policy=self.policy,
                claim=claim,
                consumption_receipt=consumption_receipt,
            )


if __name__ == "__main__":
    unittest.main()
