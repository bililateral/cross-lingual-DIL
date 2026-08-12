from __future__ import annotations

import copy
import hashlib
import io
import json
import subprocess
import sys
import tempfile
import threading
import unittest
from dataclasses import replace
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import step28_v13_common as common
import step28_v13_v1_13_counterfactual_text as counterfactual
import step28_v13_v1_13_blind_literal_scan as blind_literal_scan
import step28_v13_v1_13_build_sealed_literal_registry as sealed_registry_builder
import step28_v13_v1_13_document_collision as collision
import step28_v13_v1_13_scientific_common as scientific
import step28_v13_v1_13_scientific_quality_audit as quality
import step28_v13_v1_13_scientific_world as world_module
import run_step28_v13_v1_13_scientific_quality_audit_guarded as quality_guard
from step28_v13_v1_13_style_derangement import build_style_source_derangement


def _write_scan_sidecar(
    dataset_root: Path,
    *,
    extra_forbidden_by_split: dict[str, tuple[str, ...]] | None = None,
) -> Path:
    root_manifest_path = dataset_root / "root_manifest.json"
    builder_policy_path = (
        ROOT / "schema" / "step28_v13_v1_13_scientific_dataset_builder_policy.json"
    )
    builder_policy = common.load_json(builder_policy_path)
    if not root_manifest_path.exists():
        root_manifest = {
            "split_order": list(blind_literal_scan.SPLITS),
            "builder_policy_canonical_self_hash": builder_policy[
                "canonical_self_hash"
            ],
            "canonical_self_hash": None,
        }
        root_manifest["canonical_self_hash"] = (
            blind_literal_scan._canonical_self_hash(root_manifest)
        )
        common.write_json(root_manifest_path, root_manifest)
    root_manifest = common.load_json(root_manifest_path)
    split_registries: dict[str, object] = {}
    extras = extra_forbidden_by_split or {}
    for split in sorted(blind_literal_scan.AUDIT_SPLITS):
        categories: dict[str, list[str]] = {}
        for category in blind_literal_scan.SEALED_REGISTRY_CATEGORIES:
            values = {f"sealed_{split}_{category}"}
            if category == "full_world_forbidden":
                values.update(extras.get(split, ()))
            categories[category] = sorted(
                values, key=lambda value: value.encode("utf-8")
            )
        allowed_noise = [f"visible_noise_{split}"]
        split_registries[split] = {
            "world_count": 1,
            "categories": categories,
            "allowed_noise_raw_surfaces": allowed_noise,
            "forbidden_literal_count": len(
                set().union(*(set(values) for values in categories.values()))
            ),
            "category_commitments": {
                category: {
                    "count": len(values),
                    "sha256": common.canonical_sha256(values),
                }
                for category, values in categories.items()
            },
            "allowed_noise_raw_surface_commitment": {
                "count": len(allowed_noise),
                "sha256": common.canonical_sha256(allowed_noise),
            },
        }
    sidecar = {
        "version": blind_literal_scan.SEALED_REGISTRY_VERSION,
        "status": "SEALED_RELATION_FREE_LITERAL_REGISTRY",
        "transaction_id": "7" * 64,
        "dataset_root_manifest": {
            "path": "root_manifest.json",
            "size_bytes": root_manifest_path.stat().st_size,
            "sha256": common.sha256_file(root_manifest_path),
            "canonical_self_hash": root_manifest["canonical_self_hash"],
        },
        "literal_authority_source": {
            "path": "scripts/step28_v13_v1_13_blind_literal_scan.py",
            "size_bytes": Path(blind_literal_scan.__file__).stat().st_size,
            "sha256": common.sha256_file(Path(blind_literal_scan.__file__)),
        },
        "builder_policy": {
            "path": builder_policy_path.relative_to(ROOT).as_posix(),
            "size_bytes": builder_policy_path.stat().st_size,
            "sha256": common.sha256_file(builder_policy_path),
            "canonical_self_hash": builder_policy["canonical_self_hash"],
        },
        "split_order": sorted(blind_literal_scan.AUDIT_SPLITS),
        "private_relations_persisted": False,
        "labels_persisted": False,
        "labels_opened_for_exact_replay": True,
        "labels_used_for_candidate_selection": False,
        "labels_used_for_literal_selection": False,
        "pair_label_rows_replayed": 39_312,
        "qrels_persisted": False,
        "observed_rows_modified": 0,
        "private_input_files_semantically_replayed": list(
            blind_literal_scan.SEALED_REGISTRY_SEMANTIC_PRIVATE_FILES
        ),
        "split_registries": split_registries,
        "canonical_self_hash": None,
    }
    sidecar["canonical_self_hash"] = blind_literal_scan._canonical_self_hash(
        sidecar
    )
    path = dataset_root / "sealed_registry.json"
    common.write_json(path, sidecar)
    return path


def _write_basic_scan_split(
    dataset_root: Path,
    *,
    split: str,
    visible: str,
    collision_receipt_marker: str = "collision_receipt_secret",
    allocation_receipt_marker: str = "allocation_receipt_secret",
) -> None:
    split_root = dataset_root / split
    (split_root / "private").mkdir(parents=True)
    (split_root / "observed").mkdir()
    private_rows = {
        "controller_membership.jsonl": {
            "world_uid": "world_fixture",
            "seller_uid": "seller_fixture",
            "controller_uid": "controller_secret",
        },
        "qrels.jsonl": {
            "world_uid": "world_fixture",
            "query_uid": "query_secret",
            "query_seller_uid": "seller_fixture",
            "relevant_seller_uids": ["seller_other"],
        },
        "world_generation_audit.jsonl": {
            "world_uid": "world_fixture",
            "identity_assets": [{"identity_value": "identity_secret"}],
        },
        "document_collision_attempts.jsonl": {
            "world_uid": "world_fixture",
            "receipt_marker": collision_receipt_marker,
        },
        "identity_allocation_receipts.jsonl": {
            "world_uid": "world_fixture",
            "receipt_marker": allocation_receipt_marker,
        },
    }
    for name, row in private_rows.items():
        (split_root / "private" / name).write_text(
            common.canonical_json_bytes(row).decode("utf-8") + "\n",
            encoding="utf-8",
        )
    profile = {field: "" for field in quality.VISIBLE_PROFILE_FIELDS}
    profile["seller_uid"] = "seller_fixture"
    (split_root / "observed" / "model_seller_profiles.jsonl").write_text(
        common.canonical_json_bytes(profile).decode("utf-8") + "\n",
        encoding="utf-8",
    )
    (split_root / "observed" / "redacted_items.jsonl").write_text(
        common.canonical_json_bytes(
            {"title": visible, "description": ""}
        ).decode("utf-8")
        + "\n",
        encoding="utf-8",
    )


def _actual_builder_policy_pin() -> dict[str, object]:
    path = (
        ROOT
        / "schema"
        / "step28_v13_v1_13_scientific_dataset_builder_policy.json"
    )
    value = common.load_json(path)
    return {
        "path": path.relative_to(ROOT).as_posix(),
        "size_bytes": path.stat().st_size,
        "sha256": common.sha256_file(path),
        "canonical_self_hash": value["canonical_self_hash"],
    }


def _packer_receipt_base_fixture(
    *,
    root_pin: dict[str, object],
    builder_policy_pin: dict[str, object],
    source_closure: dict[str, dict[str, object]],
) -> dict[str, object]:
    split_commitments: dict[str, object] = {}
    for split in ("audit_a", "audit_b"):
        category_commitments = {
            category: {
                "count": 1,
                "sha256": common.canonical_sha256(
                    [f"{split}_{category}_fixture"]
                ),
            }
            for category in blind_literal_scan.SEALED_REGISTRY_CATEGORIES
        }
        split_commitments[split] = {
            "world_count": 2,
            "forbidden_literal_count": len(category_commitments),
            "category_commitments": category_commitments,
            "allowed_noise_raw_surface_commitment": {
                "count": 1,
                "sha256": common.canonical_sha256([f"{split}_noise"]),
            },
        }
    return {
        "version": sealed_registry_builder.VERSION,
        "status": "PASS_SEALED_LITERAL_REGISTRY_BUILD_NOT_AUDIT_GO",
        "dataset_root_manifest": root_pin,
        "literal_authority_source": source_closure[
            "step28_v13_v1_13_blind_literal_scan"
        ],
        "builder_policy": builder_policy_pin,
        "worlds_replayed": 104,
        "audit_worlds_projected": 4,
        "private_input_files_semantically_replayed": list(
            blind_literal_scan.SEALED_REGISTRY_SEMANTIC_PRIVATE_FILES
        ),
        "private_input_file_count": len(
            blind_literal_scan.SEALED_REGISTRY_SEMANTIC_PRIVATE_FILES
        ),
        "split_commitments": split_commitments,
        "private_values_returned": 0,
        "private_relations_returned": 0,
        "labels_returned": 0,
        "labels_opened_for_exact_replay": True,
        "labels_used_for_candidate_selection": False,
        "labels_used_for_literal_selection": False,
        "pair_label_rows_replayed": 39_312,
        "qrels_returned": 0,
        "observed_rows_modified": 0,
        "candidate_selection_changed": False,
        "derangement_changed": False,
        "quality_probe_run": False,
        "formal_generation_authorized": False,
        "model_training_authorized": False,
        "formal_seed_authorized": False,
        "audit_truth_release_authorized": False,
        "quality_audit_run_authorized": False,
        "formal_500x4_generation_authorized": False,
        "design_dataset_training_qualified": False,
        "source_closure": source_closure,
    }


class QualityAuditUnitContracts(unittest.TestCase):
    def test_policy_is_design_only_and_metrics_are_frozen(self) -> None:
        policy = quality.load_policy()
        self.assertEqual(policy["status"], quality.STATUS)
        self.assertFalse(policy["authorizations"]["formal_seed"])
        self.assertFalse(policy["authorizations"]["model_training"])
        self.assertEqual(policy["bootstrap"]["replicates"], 9999)
        self.assertEqual(policy["bootstrap"]["generator"], "numpy.random.Generator(numpy.random.PCG64)")
        self.assertEqual(policy["bootstrap"]["quantile_method"], "linear")
        derived = int.from_bytes(
            hashlib.sha256(
                b"step28-v13-v1.13-quality-probe-model-v1\x1f"
                + int(policy["bootstrap"]["metadata_design_seed"]).to_bytes(
                    8, "big"
                )
            ).digest()[:4],
            "big",
        )
        self.assertEqual(derived, 793820367)
        self.assertEqual(
            policy["metadata_probe"]["models"]["logistic_l2"]["random_state"],
            derived,
        )
        self.assertFalse(
            policy["row_audit"]["audit_split_private_semantic_open_allowed"]
        )
        self.assertTrue(
            policy["row_audit"]["audit_split_private_byte_integrity_required"]
        )
        self.assertFalse(
            policy["row_audit"]["audit_split_world_reconstruction_allowed"]
        )
        self.assertEqual(
            policy["metadata_probe"]["feature_source_contract"]
            ["title_description_missingness"],
            "observed_redacted_items_exact_model_projection",
        )
        self.assertAlmostEqual(
            policy["metadata_probe"]["average_precision_baseline"], 20 / 378
        )
        self.assertAlmostEqual(
            policy["text_counterfactual"]["average_precision_baseline"], 20 / 372
        )

    def test_every_frozen_scientific_policy_leaf_is_validated(self) -> None:
        policy = quality.load_policy()
        sections = (
            "claim_boundary",
            "input",
            "row_audit",
            "metadata_probe",
            "text_counterfactual",
            "bootstrap",
            "launch_and_failure",
            "authorizations",
            "runtime",
        )

        def leaf_paths(value: object, prefix: tuple[object, ...] = ()):
            if isinstance(value, dict):
                for key, child in value.items():
                    yield from leaf_paths(child, (*prefix, key))
            elif isinstance(value, list):
                for index, child in enumerate(value):
                    yield from leaf_paths(child, (*prefix, index))
            else:
                yield prefix

        def mutate(value: object) -> object:
            if isinstance(value, bool):
                return not value
            if value is None:
                return "MUTATED_NONE"
            if isinstance(value, int):
                return value + 1
            if isinstance(value, float):
                return value + 0.123456789
            if isinstance(value, str):
                return value + "__MUTATED"
            raise AssertionError(type(value))

        checked = 0
        for section in sections:
            for relative in leaf_paths(policy[section]):
                candidate = copy.deepcopy(policy)
                parent: object = candidate[section]
                for part in relative[:-1]:
                    parent = parent[part]  # type: ignore[index]
                final = relative[-1]
                parent[final] = mutate(parent[final])  # type: ignore[index]
                with self.subTest(section=section, path=relative):
                    with self.assertRaises(quality.ScientificQualityAuditError):
                        quality._validate_policy_contract(candidate)
                checked += 1
        self.assertGreater(checked, 150)
        for field in ("version", "status"):
            candidate = copy.deepcopy(policy)
            candidate[field] += "__MUTATED"
            with self.assertRaises(quality.ScientificQualityAuditError):
                quality._validate_policy_contract(candidate)

    def test_every_policy_pin_is_byte_verified_and_direct_run_is_rejected(self) -> None:
        policy = quality.load_policy()
        for name, spec in policy["pins"].items():
            quality._verify_pin(spec, label=name)
            for field in ("path", "size_bytes", "sha256"):
                candidate = copy.deepcopy(spec)
                candidate[field] = (
                    str(candidate[field]) + "__MUTATED"
                    if field in {"path", "sha256"}
                    else int(candidate[field]) + 1
                )
                with self.subTest(pin=name, field=field), self.assertRaises(
                    quality.ScientificQualityAuditError
                ):
                    quality._verify_pin(candidate, label=name)
        with self.assertRaises(quality.ScientificQualityAuditError):
            quality.run_audit()

    def test_quality_policy_binds_actual_relation_free_registry_receipt(self) -> None:
        policy = quality.load_policy()
        receipt_path = common.repo_path(
            policy["pins"]["sealed_literal_registry_receipt"]["path"]
        )
        receipt = common.load_json(receipt_path)
        private_pin = policy["pins"]["sealed_literal_registry"]
        self.assertEqual(
            policy["pins"]["sealed_literal_registry_builder"],
            receipt["builder_source"],
        )
        self.assertEqual(
            private_pin,
            {
                key: receipt["sealed_registry"][key]
                for key in ("path", "size_bytes", "sha256")
            },
        )
        self.assertEqual(
            policy["input"]["builder_policy_canonical_self_hash"],
            receipt["builder_policy"]["canonical_self_hash"],
        )
        self.assertEqual(
            policy["input"]["sealed_registry_source_closure_sha256"],
            common.canonical_sha256(receipt["source_closure"]),
        )
        for name, spec in receipt["source_closure"].items():
            quality._verify_pin(
                spec, label=f"sealed_registry_source_closure/{name}"
            )
        self.assertEqual(receipt["private_values_returned"], 0)
        self.assertEqual(receipt["private_relations_returned"], 0)
        self.assertEqual(receipt["labels_returned"], 0)
        self.assertFalse(receipt["quality_audit_run_authorized"])
        self.assertFalse(receipt["model_training_authorized"])

    def test_external_launch_anchor_and_guard_are_byte_bound(self) -> None:
        release = quality_guard._verify_release_manifest()
        anchor = quality_guard._verify_anchor()
        with self.assertRaises(quality_guard.ExternalReviewPending):
            quality_guard._verify_external_attestation(release)
        release_path = quality.DEFAULT_RELEASE_MANIFEST_PATH
        anchor_path = quality.DEFAULT_LAUNCH_ANCHOR_PATH
        guard_path = quality.DEFAULT_LAUNCH_GUARD_PATH
        release_evidence = {
            "path": release_path.relative_to(ROOT).as_posix(),
            "size_bytes": release_path.stat().st_size,
            "sha256": common.sha256_file(release_path),
            "canonical_self_hash": release["canonical_self_hash"],
        }
        guard_evidence = quality_guard._path_evidence(guard_path)
        quality_evidence = quality_guard._path_evidence(Path(quality.__file__))

        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            temp_root = Path(directory)
            transcript = temp_root / "external_review_transcript.md"
            transcript.write_text(
                "独立审查完成。\n允许清洁运行104-world质量审计\n",
                encoding="utf-8",
            )
            transcript_pin = {
                "path": transcript.relative_to(ROOT).as_posix(),
                "size_bytes": transcript.stat().st_size,
                "sha256": common.sha256_file(transcript),
            }
            scope = {
                "design_dataset_root": "design_preflight_v2_20260811",
                "world_count": 104,
                "quality_audit_run_authorized": True,
                "formal_generation_authorized": False,
                "model_training_authorized": False,
                "audit_truth_release_authorized": False,
            }
            git_provenance = {
                "commit": "1" * 40,
                "tree": "2" * 40,
                "implementation_bytes_committed": True,
            }
            provenance = {
                "provider": "chatgpt.com",
                "model": "GPT-5.6 Sol Pro",
                "conversation_url_sha256": "3" * 64,
                "completed_at_utc": "2026-08-11T00:00:00Z",
            }
            binding = {
                "release_manifest": release_evidence,
                "candidate_git_provenance": git_provenance,
                "external_review_provenance": provenance,
                "review_transcript": transcript_pin,
                "verdict_last_line": quality_guard.EXTERNAL_REVIEW_VERDICT,
                "review_scope": scope,
            }
            attestation = {
                "version": "2026-08-11-step28-v13-v1-13-external-review-attestation-v1",
                "status": "EXTERNAL_REVIEW_GO_DESIGN_QUALITY_AUDIT_ONLY",
                "review_scope": scope,
                "release_manifest": release_evidence,
                "candidate_git_provenance": git_provenance,
                "external_review_provenance": provenance,
                "review_transcript": transcript_pin,
                "verdict_last_line": quality_guard.EXTERNAL_REVIEW_VERDICT,
                "external_review_binding_sha256": common.canonical_sha256(binding),
                "canonical_self_hash": None,
            }
            attestation["canonical_self_hash"] = quality_guard._canonical_sha256(
                attestation
            )
            attestation_path = temp_root / "external_review_attestation.json"
            common.write_json(attestation_path, attestation)
            with (
                patch.object(
                    quality_guard, "EXTERNAL_REVIEW_ATTESTATION", attestation_path
                ),
                patch.object(
                    quality_guard,
                    "_verify_candidate_git_provenance",
                    return_value=None,
                ),
            ):
                self.assertEqual(
                    quality_guard._verify_external_attestation(release), attestation
                )
            evidence = {
                "release_manifest": release_evidence,
                "anchor": {
                    "path": anchor_path.relative_to(ROOT).as_posix(),
                    "size_bytes": anchor_path.stat().st_size,
                    "sha256": common.sha256_file(anchor_path),
                    "canonical_self_hash": anchor["canonical_self_hash"],
                },
                "guard": guard_evidence,
                "external_review_attestation": {
                    "path": attestation_path.relative_to(ROOT).as_posix(),
                    "size_bytes": attestation_path.stat().st_size,
                    "sha256": common.sha256_file(attestation_path),
                    "canonical_self_hash": attestation["canonical_self_hash"],
                },
                "entry": {
                    "main": guard_evidence,
                    "argv0": guard_evidence,
                    "quality_module": quality_evidence,
                },
            }
            main_module = sys.modules["__main__"]
            with (
                patch.object(
                    quality,
                    "DEFAULT_EXTERNAL_REVIEW_ATTESTATION_PATH",
                    attestation_path,
                ),
                patch.object(main_module, "__file__", str(guard_path)),
                patch.object(sys, "argv", [str(guard_path)]),
            ):
                self.assertEqual(
                    quality._verify_launch_evidence(
                        evidence,
                        caller_path=guard_path,
                    ),
                    evidence,
                )
                changed = copy.deepcopy(evidence)
                changed["anchor"]["sha256"] = "0" * 64
                with self.assertRaises(quality.ScientificQualityAuditError):
                    quality._verify_launch_evidence(
                        changed,
                        caller_path=guard_path,
                    )
            spoofed = compile(
                "quality.run_audit(launch_evidence=evidence)",
                str(guard_path),
                "exec",
            )
            with (
                patch.object(
                    quality,
                    "DEFAULT_EXTERNAL_REVIEW_ATTESTATION_PATH",
                    attestation_path,
                ),
                self.assertRaises(quality.AuditLaunchPreflightError),
            ):
                exec(spoofed, {"quality": quality, "evidence": evidence})
            fake_quality = temp_root / "step28_v13_v1_13_scientific_quality_audit.py"
            fake_quality.write_text("# fake module canary\n", encoding="utf-8")
            with self.assertRaises(quality_guard.LaunchGuardError):
                quality_guard._verify_imported_quality_module(fake_quality, release)
            forbidden = copy.deepcopy(attestation)
            forbidden["review_scope"]["model_training_authorized"] = True
            forbidden["canonical_self_hash"] = quality_guard._canonical_sha256(
                forbidden
            )
            forbidden_path = temp_root / "forbidden_external_review_attestation.json"
            common.write_json(forbidden_path, forbidden)
            with (
                patch.object(
                    quality_guard, "EXTERNAL_REVIEW_ATTESTATION", forbidden_path
                ),
                self.assertRaises(quality_guard.LaunchGuardError),
            ):
                quality_guard._verify_external_attestation(release)

            missing = temp_root / "missing-release.json"
            with (
                patch.object(quality_guard, "RELEASE_MANIFEST", missing),
                self.assertRaises(quality_guard.LaunchGuardError),
            ):
                quality_guard._verify_release_manifest()

            mutated_path = temp_root / "mutated-release.json"
            mutated = copy.deepcopy(release)
            mutated["pins"]["quality_audit"]["sha256"] = "0" * 64
            mutated["canonical_self_hash"] = quality_guard._canonical_sha256(mutated)
            common.write_json(mutated_path, mutated)
            with (
                patch.object(quality_guard, "RELEASE_MANIFEST", mutated_path),
                self.assertRaises(quality_guard.LaunchGuardError),
            ):
                quality_guard._verify_release_manifest()

    def test_external_review_transcript_last_line_is_the_actual_go(self) -> None:
        release = common.load_json(quality_guard.RELEASE_MANIFEST)
        cases = {
            "no_verdict": "外部审查结束\n不允许运行\n".encode("utf-8"),
            "negative_appended": (
                "允许清洁运行104-world质量审计\n不允许运行\n"
            ).encode("utf-8"),
            "empty": b" \r\n\t\n",
            "invalid_utf8": b"\xff\xfe",
            "hidden_suffix": (
                quality_guard.EXTERNAL_REVIEW_VERDICT + "\u200b\n"
            ).encode("utf-8"),
        }
        for name, transcript_bytes in cases.items():
            with self.subTest(name=name), tempfile.TemporaryDirectory(
                dir=ROOT
            ) as directory:
                root = Path(directory)
                transcript_path = root / "review.md"
                transcript_path.write_bytes(transcript_bytes)
                transcript_pin = {
                    "path": transcript_path.relative_to(ROOT).as_posix(),
                    "size_bytes": transcript_path.stat().st_size,
                    "sha256": common.sha256_file(transcript_path),
                }
                release_binding = {
                    "path": quality_guard.RELEASE_MANIFEST.relative_to(ROOT).as_posix(),
                    "size_bytes": quality_guard.RELEASE_MANIFEST.stat().st_size,
                    "sha256": common.sha256_file(quality_guard.RELEASE_MANIFEST),
                    "canonical_self_hash": release["canonical_self_hash"],
                }
                git_provenance = {
                    "commit": "1" * 40,
                    "tree": "2" * 40,
                    "implementation_bytes_committed": True,
                }
                provenance = {
                    "provider": "chatgpt.com",
                    "model": "GPT-5.6 Sol Pro",
                    "conversation_url_sha256": "3" * 64,
                    "completed_at_utc": "2026-08-12T00:00:00Z",
                }
                scope = {
                    "design_dataset_root": "design_preflight_v2_20260811",
                    "world_count": 104,
                    "quality_audit_run_authorized": True,
                    "formal_generation_authorized": False,
                    "model_training_authorized": False,
                    "audit_truth_release_authorized": False,
                }
                binding = {
                    "release_manifest": release_binding,
                    "candidate_git_provenance": git_provenance,
                    "external_review_provenance": provenance,
                    "review_transcript": transcript_pin,
                    "verdict_last_line": quality_guard.EXTERNAL_REVIEW_VERDICT,
                    "review_scope": scope,
                }
                attestation = {
                    "version": "test-external-review-attestation",
                    "status": "EXTERNAL_REVIEW_GO_DESIGN_QUALITY_AUDIT_ONLY",
                    "review_scope": scope,
                    "release_manifest": release_binding,
                    "candidate_git_provenance": git_provenance,
                    "external_review_provenance": provenance,
                    "review_transcript": transcript_pin,
                    "verdict_last_line": quality_guard.EXTERNAL_REVIEW_VERDICT,
                    "external_review_binding_sha256": common.canonical_sha256(binding),
                    "canonical_self_hash": None,
                }
                attestation["canonical_self_hash"] = quality_guard._canonical_sha256(
                    attestation
                )
                attestation_path = root / "attestation.json"
                common.write_json(attestation_path, attestation)
                with (
                    patch.object(
                        quality_guard,
                        "EXTERNAL_REVIEW_ATTESTATION",
                        attestation_path,
                    ),
                    patch.object(
                        quality_guard,
                        "_verify_candidate_git_provenance",
                        return_value=None,
                    ),
                    self.assertRaises(quality_guard.LaunchGuardError),
                ):
                    quality_guard._verify_external_attestation(release)

    def test_candidate_git_provenance_uses_real_objects_and_clean_pinned_bytes(self) -> None:
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            root = Path(directory)

            def git(*args: str) -> str:
                completed = subprocess.run(
                    ["git", *args],
                    cwd=root,
                    text=True,
                    encoding="utf-8",
                    capture_output=True,
                    check=True,
                )
                return completed.stdout.strip()

            git("init")
            git("config", "user.name", "Step28 Test")
            git("config", "user.email", "step28-test@example.invalid")
            pinned = root / "pinned.txt"
            pinned.write_bytes(b"reviewed bytes\n")
            git("add", "pinned.txt")
            git("commit", "-m", "test candidate")
            commit = git("rev-parse", "HEAD")
            tree = git("rev-parse", "HEAD^{tree}")
            release = {
                "pins": {
                    "pinned": {
                        "path": "pinned.txt",
                        "size_bytes": pinned.stat().st_size,
                        "sha256": common.sha256_file(pinned),
                    }
                }
            }
            provenance = {
                "commit": commit,
                "tree": tree,
                "implementation_bytes_committed": True,
            }
            quality_guard._verify_candidate_git_provenance(
                provenance, release, root=root
            )
            pinned.write_bytes(b"uncommitted drift\n")
            with self.assertRaises(quality_guard.LaunchGuardError):
                quality_guard._verify_candidate_git_provenance(
                    provenance, release, root=root
                )
            pinned.write_bytes(b"reviewed bytes\n")
            wrong = dict(provenance)
            wrong["tree"] = "0" * 40
            with self.assertRaises(quality_guard.LaunchGuardError):
                quality_guard._verify_candidate_git_provenance(
                    wrong, release, root=root
                )

    def test_launch_failure_receipt_is_safe_and_opens_no_dataset_rows(self) -> None:
        secret = "private_launch_secret_must_not_echo"
        with tempfile.TemporaryDirectory() as directory:
            failure_root = Path(directory) / "launch_failures"
            with patch.object(quality_guard, "LAUNCH_FAILURE_ROOT", failure_root):
                path = quality_guard._write_launch_failure_receipt(
                    quality_guard.LaunchGuardError(secret)
                )
                repeated = quality_guard._write_launch_failure_receipt(
                    quality_guard.LaunchGuardError(secret)
                )
            self.assertEqual(path, repeated)
            serialized = path.read_text(encoding="utf-8")
            receipt = json.loads(serialized)
            self.assertNotIn(secret, serialized)
            self.assertFalse(receipt["dataset_rows_opened"])
            self.assertTrue(receipt["input_dataset_retained"])
            self.assertFalse(receipt["dataset_quality_conclusion_reached"])
            self.assertEqual(
                receipt["canonical_self_hash"],
                quality_guard._canonical_sha256(receipt),
            )

    def test_guard_cleanup_recovery_mode_is_reachable_and_row_blind(self) -> None:
        receipt = {
            "status": "DATASET_INVALIDATION_CLEANUP_COMPLETE",
            "canonical_self_hash": "a" * 64,
        }
        guard_path = Path(quality_guard.__file__).resolve()
        main_module = sys.modules["__main__"]
        output = io.StringIO()
        with (
            patch.object(sys, "argv", [str(guard_path), "--recover-cleanup-receipt"]),
            patch.object(main_module, "__file__", str(guard_path)),
            patch.object(
                quality_guard,
                "_verify_release_manifest",
                return_value={"pins": {"quality_audit": {}}},
            ) as release_check,
            patch.object(quality_guard, "_verify_anchor") as anchor_check,
            patch.object(quality_guard, "_verify_external_attestation") as review_check,
            patch.object(quality_guard, "_verify_imported_quality_module"),
            patch.object(quality, "recover_cleanup_receipt", return_value=receipt) as recover,
            redirect_stdout(output),
        ):
            quality_guard._guarded_recover_cleanup()
        release_check.assert_called_once_with(allow_missing_dataset_manifest=True)
        anchor_check.assert_called_once_with(allow_missing_dataset_manifest=True)
        review_check.assert_called_once()
        recover.assert_called_once_with()
        emitted = json.loads(output.getvalue())
        self.assertEqual(emitted["status"], receipt["status"])
        self.assertFalse(emitted["dataset_rows_opened"])

    def test_guard_post_launch_exception_output_does_not_echo_message(self) -> None:
        private_canary = "private_exception_message_must_not_echo"
        output = io.StringIO()
        with (
            patch.object(sys, "argv", [str(Path(quality_guard.__file__).resolve())]),
            patch.object(
                quality_guard,
                "_guarded_run",
                side_effect=quality_guard.GuardedAuditTerminated(
                    RuntimeError(private_canary)
                ),
            ),
            redirect_stdout(output),
            self.assertRaises(SystemExit) as captured,
        ):
            quality_guard.main()
        self.assertEqual(captured.exception.code, 3)
        self.assertNotIn(private_canary, output.getvalue())
        emitted = json.loads(output.getvalue())
        self.assertEqual(
            emitted["status"], "QUALITY_AUDIT_TERMINATED_SEE_BOUND_RECEIPT"
        )
        self.assertFalse(emitted["raw_exception_message_returned"])

    def test_derangement_is_order_invariant_bijective_and_fixed_point_free(self) -> None:
        sellers = [f"seller_{index:02d}" for index in range(28)]
        first = build_style_source_derangement(
            split="train", world_uid="world_fixture", seller_uids=sellers
        )
        second = build_style_source_derangement(
            split="train", world_uid="world_fixture", seller_uids=list(reversed(sellers))
        )
        self.assertEqual(first, second)
        mapping = first.as_mapping()
        self.assertEqual(set(mapping), set(sellers))
        self.assertEqual(set(mapping.values()), set(sellers))
        self.assertTrue(all(target != source for target, source in mapping.items()))

    def test_vectorized_world_bootstrap_matches_literal_row_replay(self) -> None:
        labels = np.asarray([1, 0, 0, 0] * 3, dtype=np.int8)
        ordered_worlds = ("z_world", "a_world", "m_world")
        worlds = np.asarray(
            [world for world in ordered_worlds for _ in range(4)],
            dtype=object,
        )
        scores = (
            np.asarray([0.8, 0.2, 0.1, 0.3, 0.4, 0.7, 0.2, 0.1, 0.6, 0.5, 0.3, 0.2]),
            np.asarray([0.1, 0.1, 0.2, 0.3, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2]),
        )
        replicates = 31
        seed = 9173
        observed = quality._bootstrap_family_upper(
            labels=labels,
            row_world_uids=worlds,
            ordered_world_uids=ordered_worlds,
            score_family=scores,
            replicates=replicates,
            seed=seed,
            baseline=0.25,
        )
        rng = np.random.Generator(np.random.PCG64(seed))
        draws = rng.integers(0, 3, size=(replicates, 3), endpoint=False, dtype=np.int64)
        indices = [np.flatnonzero(worlds == world) for world in ordered_worlds]
        auc_max = []
        ap_max = []
        for draw in draws:
            row_index = np.concatenate([indices[int(index)] for index in draw])
            auc_values = []
            ap_values = []
            for score in scores:
                auc = float(roc_auc_score(labels[row_index], score[row_index]))
                auc_values.append(max(auc, 1 - auc))
                ap_values.append(
                    float(average_precision_score(labels[row_index], score[row_index]))
                    - 0.25
                )
            auc_max.append(max(auc_values))
            ap_max.append(max(ap_values))
        self.assertAlmostEqual(
            observed["auc_95_upper"],
            float(np.quantile(auc_max, 0.95, method="linear")),
            places=14,
        )
        self.assertAlmostEqual(
            observed["ap_uplift_95_upper"],
            float(np.quantile(ap_max, 0.95, method="linear")),
            places=14,
        )

    def test_bootstrap_family_maximum_includes_fourteenth_model(self) -> None:
        labels = np.asarray([1, 0, 0, 0] * 3, dtype=np.int8)
        ordered_worlds = ("z_world", "a_world", "m_world")
        worlds = np.asarray(
            [world for world in ordered_worlds for _ in range(4)], dtype=object
        )
        weak = tuple(
            np.asarray([0.5] * len(labels), dtype=np.float64) for _ in range(13)
        )
        decisive = labels.astype(np.float64)
        all_models = quality._bootstrap_family_upper(
            labels=labels,
            row_world_uids=worlds,
            ordered_world_uids=ordered_worlds,
            score_family=(*weak, decisive),
            replicates=31,
            seed=9191,
            baseline=0.25,
        )
        first_thirteen = quality._bootstrap_family_upper(
            labels=labels,
            row_world_uids=worlds,
            ordered_world_uids=ordered_worlds,
            score_family=weak,
            replicates=31,
            seed=9191,
            baseline=0.25,
        )
        self.assertEqual(all_models["score_family_size"], 14)
        self.assertEqual(all_models["auc_95_upper"], 1.0)
        self.assertGreater(
            all_models["ap_uplift_95_upper"],
            first_thirteen["ap_uplift_95_upper"],
        )

    def test_linear_bootstrap_quantile_is_not_a_nearest_order_statistic(self) -> None:
        values = np.asarray([0.0, 0.2, 0.4, 0.6, 1.0], dtype=np.float64)
        linear = float(np.quantile(values, 0.95, method="linear"))
        lower = float(np.quantile(values, 0.95, method="lower"))
        self.assertAlmostEqual(linear, 0.92)
        self.assertNotEqual(linear, lower)

    def test_three_path_receipt_rejects_mapping_pair_world_and_mask_mismatch(self) -> None:
        common_binding = {
            "split": "train",
            "split_ordinal": 0,
            "world_uid": "world_fixture",
            "mapping_sha256": "1" * 64,
            "target_source_pairs_sha256": "2" * 64,
        }
        dose = {
            "world_uid": "world_fixture",
            "effective_style_factor_tuple_changed_count": 27,
        }
        path_order = {
            "pair_order_sha256": "3" * 64,
            "world_order_sha256": "4" * 64,
            "eligible_mask_sha256": "5" * 64,
            "eligible_pair_uids_sha256": "6" * 64,
            "dose_sha256": common.canonical_sha256(dose),
        }
        fixed = {**common_binding, **path_order, "slot_keyset_sha256": "7" * 64, "original_items_sha256": "8" * 64, "counterfactual_items_sha256": "9" * 64}
        production = {**common_binding, **path_order, "original_provenance_sha256": "a" * 64, "counterfactual_provenance_sha256": "b" * 64, "original_profiles_sha256": "c" * 64, "counterfactual_profiles_sha256": "d" * 64}
        numeric = {"feature_names_sha256": "e" * 64, "original_matrix_sha256": "f" * 64, "counterfactual_matrix_sha256": "0" * 64}
        joint = {
            **common_binding,
            **path_order,
            "production_binding_sha256": common.canonical_sha256(production),
            "fixed_support_binding_sha256": common.canonical_sha256(fixed),
            "numeric_projection_sha256": common.canonical_sha256(numeric),
        }
        receipt = {
            "version": "2026-08-11-step28-v13-v1-13-three-path-alignment-v1",
            "common_binding": common_binding,
            "fixed_support": fixed,
            "production_step3": production,
            "joint_visible_input": joint,
            "numeric_projection": numeric,
            "dose": dose,
            "canonical_self_hash": None,
        }
        receipt["canonical_self_hash"] = quality._canonical_self_hash(receipt)
        quality._validate_three_path_alignment_receipt(receipt)
        for path_name, field in (
            ("fixed_support", "mapping_sha256"),
            ("production_step3", "pair_order_sha256"),
            ("joint_visible_input", "world_order_sha256"),
            ("fixed_support", "eligible_mask_sha256"),
        ):
            candidate = copy.deepcopy(receipt)
            candidate[path_name][field] = "f" * 64
            candidate["canonical_self_hash"] = quality._canonical_self_hash(candidate)
            with self.subTest(path=path_name, field=field), self.assertRaises(
                quality.ScientificQualityAuditError
            ):
                quality._validate_three_path_alignment_receipt(candidate)
        changed_dose = copy.deepcopy(receipt)
        changed_dose["dose"]["effective_style_factor_tuple_changed_count"] = 26
        changed_dose["canonical_self_hash"] = quality._canonical_self_hash(
            changed_dose
        )
        with self.assertRaises(quality.ScientificQualityAuditError):
            quality._validate_three_path_alignment_receipt(changed_dose)

    def test_frozen_probe_family_really_fits_with_valid_random_state(self) -> None:
        policy = quality.load_policy()
        rng = np.random.Generator(np.random.PCG64(20260811))
        train_x = rng.normal(size=(160, 6))
        train_y = np.asarray([0, 1] * 80, dtype=np.int8)
        development_x = rng.normal(size=(48, 6))
        scores = quality._fit_probe_family(
            train_x,
            train_y,
            development_x,
            config=policy["metadata_probe"]["models"],
        )
        repeated = quality._fit_probe_family(
            train_x,
            train_y,
            development_x,
            config=policy["metadata_probe"]["models"],
        )
        self.assertEqual(set(scores), {"logistic_l2", "shallow_tree"})
        for name, values in scores.items():
            self.assertEqual(values.shape, (48,))
            self.assertTrue(np.isfinite(values).all())
            self.assertTrue(np.logical_and(values >= 0, values <= 1).all())
            self.assertEqual(values.astype("<f8").tobytes(), repeated[name].astype("<f8").tobytes())

    def test_metadata_missingness_uses_observed_text_and_time_uses_source(self) -> None:
        profiles = [
            {"seller_uid": "s0", "item_count": 1},
            {"seller_uid": "s1", "item_count": 1},
        ]
        source_items = [
            {
                "item_uid": "i0",
                "seller_uid": "s0",
                "world_uid": "w0",
                "title": "private title",
                "description": "private description",
                "time_bucket": 1,
            },
            {
                "item_uid": "i1",
                "seller_uid": "s1",
                "world_uid": "w0",
                "title": "private title",
                "description": "private description",
                "time_bucket": 2,
            },
        ]
        observed_items = copy.deepcopy(source_items)
        observed_items[0]["title"] = ""
        observed_items[0]["description"] = ""
        observed_items[0].pop("time_bucket")
        observed_items[1].pop("time_bucket")
        values, names = quality._seller_metadata(
            profiles=profiles,
            observed_items=observed_items,
            source_items=source_items,
        )
        index = {name: position for position, name in enumerate(names)}
        self.assertEqual(values["s0"][index["title_missing_rate"]], 1.0)
        self.assertEqual(values["s0"][index["description_missing_rate"]], 1.0)
        self.assertEqual(values["s0"][index["time_bucket_probability_01"]], 1.0)
        self.assertEqual(values["s1"][index["title_missing_rate"]], 0.0)
        self.assertEqual(values["s1"][index["time_bucket_probability_02"]], 1.0)
        bad_profiles = copy.deepcopy(profiles)
        bad_profiles[0]["item_count"] = 2
        with self.assertRaises(quality.ScientificQualityAuditError):
            quality._seller_metadata(
                profiles=bad_profiles,
                observed_items=observed_items,
                source_items=source_items,
            )
        bad_source = copy.deepcopy(source_items)
        bad_source[0]["seller_uid"] = "s1"
        with self.assertRaises(quality.ScientificQualityAuditError):
            quality._seller_metadata(
                profiles=profiles,
                observed_items=observed_items,
                source_items=bad_source,
            )

    def test_mechanism_neutral_exclusions_require_exact_two_plus_four(self) -> None:
        valid = [
            {"canonical_pair_uid": f"p{index}", "flag": flag}
            for index, flag in enumerate(
                ["exact_title_clone_target"] * 2
                + ["high_semantic_similarity_target"] * 4
            )
        ]
        labels = {f"p{index}": 0 for index in range(6)}
        self.assertEqual(
            quality._mechanism_neutral_exclusions(valid, labels), set(labels)
        )
        three_plus_three = copy.deepcopy(valid)
        three_plus_three[2]["flag"] = "exact_title_clone_target"
        with self.assertRaises(quality.ScientificQualityAuditError):
            quality._mechanism_neutral_exclusions(three_plus_three, labels)
        duplicate = copy.deepcopy(valid)
        duplicate[-1]["canonical_pair_uid"] = "p0"
        with self.assertRaises(quality.ScientificQualityAuditError):
            quality._mechanism_neutral_exclusions(duplicate, labels)

    def test_fixed_support_empty_rates_are_symmetric_numeric_features(self) -> None:
        counts = [(1.0, 0.0, 0.0, 0.0, 0.0, 0.25), (2.0, 0.0, 0.0, 0.0, 0.0, 0.50)]
        seller_row = {"left": 0, "right": 1}
        endpoint = {
            "seller_uid_left": "left",
            "seller_uid_right": "right",
        }
        swapped = {
            "seller_uid_left": "right",
            "seller_uid_right": "left",
        }
        first = quality._fixed_support_surface_pair_features_from_counts(
            counts, seller_row=seller_row, endpoints=[endpoint]
        )
        second = quality._fixed_support_surface_pair_features_from_counts(
            counts, seller_row=seller_row, endpoints=[swapped]
        )
        self.assertEqual(first.astype("<f8").tobytes(), second.astype("<f8").tobytes())
        self.assertAlmostEqual(first[0, -2], 0.25)
        self.assertAlmostEqual(first[0, -1], 0.75)

    def test_fixed_support_never_forms_cross_item_ngrams_or_reads_raw_identity(self) -> None:
        raw_items = [
            {"item_uid": "i0", "seller_uid": "s0", "world_uid": "w0", "title": "AA", "description": "", "raw_identity": "secret_a"},
            {"item_uid": "i1", "seller_uid": "s0", "world_uid": "w0", "title": "BB", "description": "", "raw_identity": "secret_b"},
            {"item_uid": "i2", "seller_uid": "s1", "world_uid": "w0", "title": "AABB", "description": "", "raw_identity": "secret_c"},
        ]
        projected = [
            quality.dataset_builder._project_model_redacted_item(row)
            for row in raw_items
        ]
        changed_private = copy.deepcopy(raw_items)
        for row in changed_private:
            row["raw_identity"] += "_changed"
        reprojected = [
            quality.dataset_builder._project_model_redacted_item(row)
            for row in changed_private
        ]
        self.assertEqual(projected, reprojected)
        endpoints = [{"canonical_pair_uid": "p0", "world_uid": "w0", "seller_uid_left": "s0", "seller_uid_right": "s1"}]
        first, names, _, _ = quality.build_fixed_support_text_views(
            items=projected, endpoints=endpoints
        )
        second, _, _, _ = quality.build_fixed_support_text_views(
            items=reprojected, endpoints=endpoints
        )
        self.assertEqual(first["fs_full"].astype("<f8").tobytes(), second["fs_full"].astype("<f8").tobytes())
        char_title = names["fs_full"].index("char3_cosine__slot_title")
        self.assertEqual(first["fs_full"][0, char_title], 0.0)

    def test_visible_identity_pattern_and_private_literal_canaries_fail(self) -> None:
        blank_profile = {field: "" for field in quality.VISIBLE_PROFILE_FIELDS}
        blank_profile["seller_uid"] = "s0"
        samples = (
            "contact@example.org",
            "@telegram_name",
            "https://example.org/x",
            "example.org/u/alpha",
            "+86 138 1234 5678",
            "0x0123456789abcdef0123456789abcdef",
            "1BoatSLRHtKNngkdXEeobR76b53LETtpyT",
            "请加微信联系",
        )
        for sample in samples:
            with self.subTest(sample=sample):
                item = {"title": sample, "description": ""}
                with self.assertRaises(quality.DatasetInvalidationError):
                    quality._scan_visible_text(
                        profiles=[blank_profile],
                        redacted_items=[item],
                        forbidden_markers=("internal_marker",),
                        forbidden_literals=(),
                    )
        with self.assertRaises(quality.DatasetInvalidationError):
            quality._scan_visible_text(
                profiles=[blank_profile],
                redacted_items=[{"title": "private_literal", "description": ""}],
                forbidden_markers=("internal_marker",),
                forbidden_literals=("private_literal",),
            )

    def test_sealed_scanner_catches_real_uid_and_short_bare_identity_canaries(self) -> None:
        canaries = (
            ("controller_secret", "controller"),
            ("query_secret", "query"),
            ("1234567", "identity"),
            ("wxbare88", "identity"),
            ("batbare88", "identity"),
            ("telebare88", "identity"),
        )
        for visible, source in canaries:
            with self.subTest(visible=visible, source=source), tempfile.TemporaryDirectory() as directory:
                split_root = Path(directory) / "audit_a"
                (split_root / "private").mkdir(parents=True)
                (split_root / "observed").mkdir()
                controller = visible if source == "controller" else "controller_secret"
                query = visible if source == "query" else "query_secret"
                identity = visible if source == "identity" else "identity_secret"
                private_rows = {
                    "controller_membership.jsonl": {
                        "world_uid": "world_fixture",
                        "seller_uid": "seller_fixture",
                        "controller_uid": controller,
                    },
                    "qrels.jsonl": {
                        "world_uid": "world_fixture",
                        "query_uid": query,
                        "query_seller_uid": "seller_fixture",
                        "relevant_seller_uids": ["seller_other"],
                    },
                    "world_generation_audit.jsonl": {
                        "world_uid": "world_fixture",
                        "identity_assets": [{"identity_value": identity}],
                        "identity_slots_audit": [{"raw_surface": identity}],
                    },
                    "document_collision_attempts.jsonl": {
                        "world_uid": "world_fixture",
                        "receipt_marker": "collision_receipt_secret",
                    },
                    "identity_allocation_receipts.jsonl": {
                        "world_uid": "world_fixture",
                        "receipt_marker": "allocation_receipt_secret",
                    },
                }
                for name, row in private_rows.items():
                    (split_root / "private" / name).write_text(
                        common.canonical_json_bytes(row).decode("utf-8") + "\n",
                        encoding="utf-8",
                    )
                profile = {field: "" for field in quality.VISIBLE_PROFILE_FIELDS}
                profile["seller_uid"] = "seller_fixture"
                (split_root / "observed" / "model_seller_profiles.jsonl").write_text(
                    common.canonical_json_bytes(profile).decode("utf-8") + "\n",
                    encoding="utf-8",
                )
                (split_root / "observed" / "redacted_items.jsonl").write_text(
                    common.canonical_json_bytes(
                        {"title": visible, "description": ""}
                    ).decode("utf-8")
                    + "\n",
                    encoding="utf-8",
                )
                registry = _write_scan_sidecar(Path(directory))
                receipt = blind_literal_scan.scan(
                    Path(directory), "audit_a", registry
                )
                self.assertEqual(receipt["status"], "FAIL_PRIVATE_LITERAL_HIT")
                self.assertGreater(receipt["hit_count"], 0)
                serialized = common.canonical_json_bytes(receipt).decode("utf-8")
                self.assertNotIn(visible, serialized)
                self.assertEqual(receipt["private_values_returned"], 0)

    def test_sealed_scanner_catches_private_markers_and_identity_equivalents(self) -> None:
        canaries = (
            ("mechanism", "exact_title_clone", "exact_title_clone"),
            ("flag", "semantic_hard_negative", "semantic_hard_negative"),
            ("override_kind", "forced_description_clone", "forced_description_clone"),
            ("identity_value", "https://www.example.org/u/alpha", "example.org/u/alpha"),
            ("raw_surface", "微信：wx_bare_88", "wx_bare_88"),
            ("dummy_identity_type", "future_contact_88", "future_contact_88"),
        )
        for field, private_value, visible in canaries:
            with self.subTest(field=field), tempfile.TemporaryDirectory() as directory:
                split_root = Path(directory) / "audit_a"
                (split_root / "private").mkdir(parents=True)
                (split_root / "observed").mkdir()
                private_rows = {
                    "controller_membership.jsonl": {
                        "world_uid": "world_fixture",
                        "seller_uid": "seller_fixture",
                        "controller_uid": "controller_secret",
                    },
                    "qrels.jsonl": {
                        "world_uid": "world_fixture",
                        "query_uid": "query_secret",
                        "query_seller_uid": "seller_fixture",
                        "relevant_seller_uids": ["seller_other"],
                    },
                    "world_generation_audit.jsonl": {
                        "world_uid": "world_fixture",
                        "mechanism_assignments": [
                            {"mechanism": private_value if field == "mechanism" else "mechanism_secret"}
                        ],
                        "negative_flags": [
                            {"flag": private_value if field == "flag" else "flag_secret"}
                        ],
                        "override_audit": [
                            {"override_kind": private_value if field == "override_kind" else "override_secret"}
                        ],
                        "identity_assets": [
                            {"identity_value": private_value if field == "identity_value" else "identity_secret"}
                        ],
                        "identity_slots_audit": [
                            {"raw_surface": private_value if field == "raw_surface" else "surface_secret"}
                        ],
                        "future_identity_registry": [
                            {
                                "dummy_identity_type_value": (
                                    private_value
                                    if field == "dummy_identity_type"
                                    else "future_secret"
                                )
                            }
                        ],
                    },
                    "document_collision_attempts.jsonl": {
                        "world_uid": "world_fixture",
                        "receipt_marker": "collision_receipt_secret",
                    },
                    "identity_allocation_receipts.jsonl": {
                        "world_uid": "world_fixture",
                        "receipt_marker": "allocation_receipt_secret",
                    },
                }
                for name, row in private_rows.items():
                    (split_root / "private" / name).write_text(
                        common.canonical_json_bytes(row).decode("utf-8") + "\n",
                        encoding="utf-8",
                    )
                profile = {field: "" for field in quality.VISIBLE_PROFILE_FIELDS}
                profile["seller_uid"] = "seller_fixture"
                (split_root / "observed" / "model_seller_profiles.jsonl").write_text(
                    common.canonical_json_bytes(profile).decode("utf-8") + "\n",
                    encoding="utf-8",
                )
                (split_root / "observed" / "redacted_items.jsonl").write_text(
                    common.canonical_json_bytes(
                        {"title": visible, "description": ""}
                    ).decode("utf-8")
                    + "\n",
                    encoding="utf-8",
                )
                registry = _write_scan_sidecar(Path(directory))
                receipt = blind_literal_scan.scan(
                    Path(directory), "audit_a", registry
                )
                self.assertEqual(receipt["status"], "FAIL_PRIVATE_LITERAL_HIT")
                self.assertGreater(receipt["hit_count"], 0)
                serialized = common.canonical_json_bytes(receipt).decode("utf-8")
                self.assertNotIn(private_value, serialized)
                self.assertNotIn(visible, serialized)

    def test_sealed_scanner_symmetric_normalization_and_future_mapping_keys(self) -> None:
        canaries = (
            ("BAT", "BAT", None),
            ("1234567", "123·4567", None),
            ("1234567", "123(45)67", None),
            ("MixedCaseSecret", "mIXEDcASEsECRET", None),
            ("Caf\u00e9Secret", "Cafe\u0301Secret", None),
            ("identity_placeholder", "seller_secret_88", "seller_secret_88"),
        )
        for private_value, visible, future_key in canaries:
            with self.subTest(visible=visible), tempfile.TemporaryDirectory() as directory:
                split_root = Path(directory) / "audit_a"
                (split_root / "private").mkdir(parents=True)
                (split_root / "observed").mkdir()
                world_audit = {
                    "world_uid": "world_fixture",
                    "identity_assets": [{"identity_value": private_value}],
                    "identity_slots_audit": [{"raw_surface": private_value}],
                }
                if future_key is not None:
                    world_audit["future_identity_registry"] = {future_key: 1}
                private_rows = {
                    "controller_membership.jsonl": {
                        "world_uid": "world_fixture",
                        "seller_uid": "seller_fixture",
                        "controller_uid": "controller_secret",
                    },
                    "qrels.jsonl": {
                        "world_uid": "world_fixture",
                        "query_uid": "query_secret",
                        "query_seller_uid": "seller_fixture",
                        "relevant_seller_uids": ["seller_other"],
                    },
                    "world_generation_audit.jsonl": world_audit,
                    "document_collision_attempts.jsonl": {
                        "world_uid": "world_fixture",
                        "receipt_marker": "collision_receipt_secret",
                    },
                    "identity_allocation_receipts.jsonl": {
                        "world_uid": "world_fixture",
                        "receipt_marker": "allocation_receipt_secret",
                    },
                }
                for name, row in private_rows.items():
                    (split_root / "private" / name).write_text(
                        common.canonical_json_bytes(row).decode("utf-8") + "\n",
                        encoding="utf-8",
                    )
                profile = {field: "" for field in quality.VISIBLE_PROFILE_FIELDS}
                profile["seller_uid"] = "seller_fixture"
                (split_root / "observed" / "model_seller_profiles.jsonl").write_text(
                    common.canonical_json_bytes(profile).decode("utf-8") + "\n",
                    encoding="utf-8",
                )
                (split_root / "observed" / "redacted_items.jsonl").write_text(
                    common.canonical_json_bytes(
                        {"title": visible, "description": ""}
                    ).decode("utf-8")
                    + "\n",
                    encoding="utf-8",
                )
                registry = _write_scan_sidecar(Path(directory))
                receipt = blind_literal_scan.scan(
                    Path(directory), "audit_a", registry
                )
                self.assertEqual(receipt["status"], "FAIL_PRIVATE_LITERAL_HIT")
                serialized = common.canonical_json_bytes(receipt).decode("utf-8")
                self.assertNotIn(private_value, serialized)
                self.assertNotIn(visible, serialized)

    def test_sealed_short_marker_matching_is_boundary_aware(self) -> None:
        self.assertTrue(blind_literal_scan.contains_private_literal("BAT", ("BAT",)))
        self.assertTrue(
            blind_literal_scan.contains_private_literal("请用 BAT：alpha", ("BAT",))
        )
        self.assertFalse(
            blind_literal_scan.contains_private_literal("combat ready", ("BAT",))
        )

    def test_sealed_registry_scans_sidecar_and_receipt_only_literals(self) -> None:
        for source in ("sidecar", "collision_receipt", "allocation_receipt"):
            canary = f"forbidden_{source}_canary"
            with self.subTest(source=source), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                _write_basic_scan_split(
                    root,
                    split="audit_a",
                    visible=canary,
                    collision_receipt_marker=(
                        canary if source == "collision_receipt" else "collision_secret"
                    ),
                    allocation_receipt_marker=(
                        canary if source == "allocation_receipt" else "allocation_secret"
                    ),
                )
                registry = _write_scan_sidecar(
                    root,
                    extra_forbidden_by_split={
                        "audit_a": (canary,) if source == "sidecar" else ()
                    },
                )
                receipt = blind_literal_scan.scan(root, "audit_a", registry)
                self.assertEqual(receipt["status"], "FAIL_PRIVATE_LITERAL_HIT")
                self.assertGreater(receipt["hit_count"], 0)
                self.assertNotIn(
                    canary, common.canonical_json_bytes(receipt).decode("utf-8")
                )

    def test_sealed_registry_tamper_and_missing_category_fail_closed(self) -> None:
        for mutation in (
            "missing_category",
            "bad_commitment",
            "bad_transaction_id",
            "bad_builder_policy",
            "empty_noise_allowlist",
        ):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                _write_basic_scan_split(
                    root, split="audit_a", visible="ordinary visible text"
                )
                registry = _write_scan_sidecar(root)
                value = common.load_json(registry)
                split_value = value["split_registries"]["audit_a"]
                if mutation == "missing_category":
                    del split_value["categories"]["full_world_forbidden"]
                    del split_value["category_commitments"]["full_world_forbidden"]
                elif mutation == "bad_commitment":
                    split_value["category_commitments"]["full_world_forbidden"][
                        "sha256"
                    ] = "0" * 64
                elif mutation == "bad_transaction_id":
                    value["transaction_id"] = "G" * 64
                elif mutation == "bad_builder_policy":
                    value["builder_policy"]["sha256"] = "0" * 64
                else:
                    split_value["allowed_noise_raw_surfaces"] = []
                    split_value["allowed_noise_raw_surface_commitment"] = {
                        "count": 0,
                        "sha256": common.canonical_sha256([]),
                    }
                value["canonical_self_hash"] = blind_literal_scan._canonical_self_hash(
                    value
                )
                registry.write_text(
                    json.dumps(
                        value,
                        ensure_ascii=False,
                        sort_keys=True,
                        indent=2,
                        allow_nan=False,
                    )
                    + "\n",
                    encoding="utf-8",
                )
                with self.assertRaises(blind_literal_scan.BlindLiteralInputError):
                    blind_literal_scan.scan(root, "audit_a", registry)

    def test_sealed_registry_builder_transaction_cleans_only_its_partial_outputs(self) -> None:
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            root = Path(directory)
            private_output = root / "private" / "registry.json"
            public_receipt = root / "public" / "receipt.json"
            transaction_intent = root / "private" / "transaction.json"
            transaction_lock = root / "private" / "transaction.lock.json"
            root_pin = {
                "path": "reports/step28_v13_v1_13_scientific_builder/"
                "design_preflight_v2_20260811/root_manifest.json",
                "size_bytes": sealed_registry_builder.EXPECTED_ROOT_MANIFEST_SIZE_BYTES,
                "sha256": sealed_registry_builder.EXPECTED_ROOT_MANIFEST_SHA256,
                "canonical_self_hash": (
                    sealed_registry_builder.EXPECTED_ROOT_MANIFEST_SELF_HASH
                ),
            }
            source_closure = (
                sealed_registry_builder._capture_runtime_source_closure()
            )
            builder_policy_pin = _actual_builder_policy_pin()
            sidecar = {
                "version": sealed_registry_builder.VERSION,
                "status": "SEALED_RELATION_FREE_LITERAL_REGISTRY",
                "dataset_root_manifest": root_pin,
                "builder_policy": builder_policy_pin,
                "canonical_self_hash": None,
            }
            sidecar["canonical_self_hash"] = (
                sealed_registry_builder._canonical_self_hash(sidecar)
            )
            receipt = _packer_receipt_base_fixture(
                root_pin=root_pin,
                builder_policy_pin=builder_policy_pin,
                source_closure=source_closure,
            )
            original_write = sealed_registry_builder._write_once

            def fail_public(
                path: Path,
                value: object,
                owned_paths: list[Path] | None = None,
                **write_options: object,
            ) -> None:
                if Path(path) == public_receipt:
                    raise OSError("public receipt write canary")
                original_write(  # type: ignore[arg-type]
                    Path(path), value, owned_paths, **write_options
                )

            with (
                patch.object(
                    sealed_registry_builder, "PRIVATE_OUTPUT", private_output
                ),
                patch.object(
                    sealed_registry_builder, "PUBLIC_RECEIPT", public_receipt
                ),
                patch.object(
                    sealed_registry_builder,
                    "TRANSACTION_INTENT",
                    transaction_intent,
                ),
                patch.object(
                    sealed_registry_builder,
                    "TRANSACTION_LOCK",
                    transaction_lock,
                ),
                patch.object(
                    sealed_registry_builder,
                    "_replay_and_collect",
                    return_value=(sidecar, receipt),
                ),
                patch.object(
                    sealed_registry_builder,
                    "_write_once",
                    side_effect=fail_public,
                ),
                self.assertRaises(OSError),
            ):
                sealed_registry_builder.run_build()
            self.assertFalse(private_output.exists())
            self.assertFalse(public_receipt.exists())
            self.assertFalse(transaction_intent.exists())
            self.assertFalse(transaction_lock.exists())

            private_output.parent.mkdir(parents=True, exist_ok=True)
            private_output.write_text("preexisting custody", encoding="utf-8")
            with (
                patch.object(
                    sealed_registry_builder, "PRIVATE_OUTPUT", private_output
                ),
                patch.object(
                    sealed_registry_builder, "PUBLIC_RECEIPT", public_receipt
                ),
                patch.object(
                    sealed_registry_builder,
                    "TRANSACTION_INTENT",
                    transaction_intent,
                ),
                patch.object(
                    sealed_registry_builder,
                    "TRANSACTION_LOCK",
                    transaction_lock,
                ),
                self.assertRaises(
                    sealed_registry_builder.SealedLiteralRegistryBuildError
                ),
            ):
                sealed_registry_builder.run_build()
            self.assertEqual(
                private_output.read_text(encoding="utf-8"), "preexisting custody"
            )

    def test_sealed_registry_write_removes_renamed_file_if_directory_sync_fails(self) -> None:
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            output = Path(directory) / "registry.json"
            with (
                patch.object(
                    sealed_registry_builder,
                    "_fsync_parent_directory",
                    side_effect=OSError("directory sync canary"),
                ),
                self.assertRaises(OSError),
            ):
                sealed_registry_builder._write_once(
                    output,
                    {
                        "status": "fixture",
                        "canonical_self_hash": "6" * 64,
                    },
                )
            self.assertFalse(output.exists())
            self.assertFalse(
                sealed_registry_builder._temporary_path(output).exists()
            )

    def test_sealed_registry_publish_is_atomic_no_replace_and_inode_owned(self) -> None:
        fixture = {
            "status": "fixture",
            "canonical_self_hash": "7" * 64,
        }
        for occupied_kind in ("final", "building"):
            with self.subTest(occupied_kind=occupied_kind), tempfile.TemporaryDirectory(
                dir=ROOT
            ) as directory:
                output = Path(directory) / "registry.json"
                occupied = (
                    output
                    if occupied_kind == "final"
                    else sealed_registry_builder._temporary_path(output)
                )
                occupied.write_text("foreign-owner", encoding="utf-8")
                ledger: list[Path] = []
                with self.assertRaises(
                    sealed_registry_builder.SealedLiteralRegistryBuildError
                ):
                    sealed_registry_builder._write_once(
                        output, fixture, ledger
                    )
                self.assertEqual(
                    occupied.read_text(encoding="utf-8"), "foreign-owner"
                )
                self.assertEqual(ledger, [])

        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            output = Path(directory) / "registry.json"
            original_link = sealed_registry_builder.os.link

            def publish_competing_final(source: object, target: object) -> None:
                Path(target).write_text("foreign-after-check", encoding="utf-8")
                original_link(source, target)

            ledger = []
            with (
                patch.object(
                    sealed_registry_builder.os,
                    "link",
                    side_effect=publish_competing_final,
                ),
                self.assertRaises(
                    sealed_registry_builder.SealedLiteralRegistryBuildError
                ),
            ):
                sealed_registry_builder._write_once(output, fixture, ledger)
            self.assertEqual(
                output.read_text(encoding="utf-8"), "foreign-after-check"
            )
            self.assertFalse(
                sealed_registry_builder._temporary_path(output).exists()
            )
            self.assertEqual(ledger, [])

        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            output = Path(directory) / "registry.json"
            original_link = sealed_registry_builder.os.link

            def interrupt_after_link(source: object, target: object) -> None:
                original_link(source, target)
                raise KeyboardInterrupt("post-link ownership canary")

            ledger = []
            with (
                patch.object(
                    sealed_registry_builder.os,
                    "link",
                    side_effect=interrupt_after_link,
                ),
                self.assertRaises(KeyboardInterrupt),
            ):
                sealed_registry_builder._write_once(output, fixture, ledger)
            self.assertFalse(output.exists())
            self.assertFalse(
                sealed_registry_builder._temporary_path(output).exists()
            )
            self.assertEqual(ledger, [])

    def test_sealed_registry_each_transaction_rename_sync_failure_cleans(self) -> None:
        root_pin = {
            "path": "reports/step28_v13_v1_13_scientific_builder/"
            "design_preflight_v2_20260811/root_manifest.json",
            "size_bytes": sealed_registry_builder.EXPECTED_ROOT_MANIFEST_SIZE_BYTES,
            "sha256": sealed_registry_builder.EXPECTED_ROOT_MANIFEST_SHA256,
            "canonical_self_hash": (
                sealed_registry_builder.EXPECTED_ROOT_MANIFEST_SELF_HASH
            ),
        }
        source_closure = (
            sealed_registry_builder._capture_runtime_source_closure()
        )
        builder_policy_pin = _actual_builder_policy_pin()
        sidecar = {
            "version": sealed_registry_builder.VERSION,
            "status": "SEALED_RELATION_FREE_LITERAL_REGISTRY",
            "dataset_root_manifest": root_pin,
            "builder_policy": builder_policy_pin,
            "canonical_self_hash": None,
        }
        sidecar["canonical_self_hash"] = (
            sealed_registry_builder._canonical_self_hash(sidecar)
        )
        receipt = _packer_receipt_base_fixture(
            root_pin=root_pin,
            builder_policy_pin=builder_policy_pin,
            source_closure=source_closure,
        )
        for failure_type in (OSError, KeyboardInterrupt, SystemExit):
            for failed_stage in ("lock", "intent", "sidecar", "receipt"):
                with self.subTest(
                    failure_type=failure_type.__name__,
                    failed_stage=failed_stage,
                ), tempfile.TemporaryDirectory(dir=ROOT) as directory:
                    root = Path(directory)
                    private_output = root / "private" / "registry.json"
                    public_receipt = root / "public" / "receipt.json"
                    transaction_intent = root / "private" / "transaction.json"
                    transaction_lock = root / "private" / "transaction.lock.json"
                    sentinel = root / "outside-transaction.txt"
                    sentinel.write_text("preserve", encoding="utf-8")
                    targets = {
                        "lock": transaction_lock,
                        "intent": transaction_intent,
                        "sidecar": private_output,
                        "receipt": public_receipt,
                    }

                    def fail_selected(path: Path) -> None:
                        if Path(path) == targets[failed_stage]:
                            raise failure_type("selected directory sync canary")

                    with (
                        patch.object(
                            sealed_registry_builder,
                            "PRIVATE_OUTPUT",
                            private_output,
                        ),
                        patch.object(
                            sealed_registry_builder,
                            "PUBLIC_RECEIPT",
                            public_receipt,
                        ),
                        patch.object(
                            sealed_registry_builder,
                            "TRANSACTION_INTENT",
                            transaction_intent,
                        ),
                        patch.object(
                            sealed_registry_builder,
                            "TRANSACTION_LOCK",
                            transaction_lock,
                        ),
                        patch.object(
                            sealed_registry_builder,
                            "_replay_and_collect",
                            return_value=(sidecar, receipt),
                        ),
                        patch.object(
                            sealed_registry_builder,
                            "_fsync_parent_directory",
                            side_effect=fail_selected,
                        ),
                        self.assertRaises(failure_type),
                    ):
                        sealed_registry_builder.run_build()
                    for path in (
                        transaction_intent,
                        transaction_lock,
                        private_output,
                        public_receipt,
                        sealed_registry_builder._temporary_path(transaction_intent),
                        sealed_registry_builder._temporary_path(transaction_lock),
                        sealed_registry_builder._temporary_path(private_output),
                        sealed_registry_builder._temporary_path(public_receipt),
                    ):
                        self.assertFalse(path.exists())
                    self.assertEqual(sentinel.read_text(encoding="utf-8"), "preserve")

    def test_sealed_registry_each_stage_is_owned_before_publish_returns(self) -> None:
        root_pin = {
            "path": "reports/step28_v13_v1_13_scientific_builder/"
            "design_preflight_v2_20260811/root_manifest.json",
            "size_bytes": sealed_registry_builder.EXPECTED_ROOT_MANIFEST_SIZE_BYTES,
            "sha256": sealed_registry_builder.EXPECTED_ROOT_MANIFEST_SHA256,
            "canonical_self_hash": (
                sealed_registry_builder.EXPECTED_ROOT_MANIFEST_SELF_HASH
            ),
        }
        source_closure = sealed_registry_builder._capture_runtime_source_closure()
        builder_policy_pin = _actual_builder_policy_pin()
        sidecar = {
            "version": sealed_registry_builder.VERSION,
            "status": "SEALED_RELATION_FREE_LITERAL_REGISTRY",
            "dataset_root_manifest": root_pin,
            "builder_policy": builder_policy_pin,
            "canonical_self_hash": None,
        }
        sidecar["canonical_self_hash"] = (
            sealed_registry_builder._canonical_self_hash(sidecar)
        )
        receipt = _packer_receipt_base_fixture(
            root_pin=root_pin,
            builder_policy_pin=builder_policy_pin,
            source_closure=source_closure,
        )
        original_write = sealed_registry_builder._write_once
        for failure_type in (KeyboardInterrupt, SystemExit):
            for failed_stage in ("lock", "intent", "sidecar", "receipt"):
                with self.subTest(
                    failure_type=failure_type.__name__,
                    failed_stage=failed_stage,
                ), tempfile.TemporaryDirectory(dir=ROOT) as directory:
                    root = Path(directory)
                    private_output = root / "private" / "registry.json"
                    public_receipt = root / "public" / "receipt.json"
                    transaction_intent = root / "private" / "transaction.json"
                    transaction_lock = root / "private" / "transaction.lock.json"
                    sentinel = root / "outside-transaction.txt"
                    sentinel.write_text("preserve", encoding="utf-8")
                    targets = {
                        "lock": transaction_lock,
                        "intent": transaction_intent,
                        "sidecar": private_output,
                        "receipt": public_receipt,
                    }

                    def interrupt_after_publish(
                        path: Path,
                        value: object,
                        owned_paths: list[Path] | None = None,
                        **write_options: object,
                    ) -> None:
                        original_write(  # type: ignore[arg-type]
                            Path(path), value, owned_paths, **write_options
                        )
                        if Path(path) == targets[failed_stage]:
                            raise failure_type("post-publish ownership canary")

                    with (
                        patch.object(
                            sealed_registry_builder,
                            "PRIVATE_OUTPUT",
                            private_output,
                        ),
                        patch.object(
                            sealed_registry_builder,
                            "PUBLIC_RECEIPT",
                            public_receipt,
                        ),
                        patch.object(
                            sealed_registry_builder,
                            "TRANSACTION_INTENT",
                            transaction_intent,
                        ),
                        patch.object(
                            sealed_registry_builder,
                            "TRANSACTION_LOCK",
                            transaction_lock,
                        ),
                        patch.object(
                            sealed_registry_builder,
                            "_replay_and_collect",
                            return_value=(sidecar, receipt),
                        ),
                        patch.object(
                            sealed_registry_builder,
                            "_write_once",
                            side_effect=interrupt_after_publish,
                        ),
                        self.assertRaises(failure_type),
                    ):
                        sealed_registry_builder.run_build()
                    for path in (
                        transaction_intent,
                        transaction_lock,
                        private_output,
                        public_receipt,
                        sealed_registry_builder._temporary_path(transaction_intent),
                        sealed_registry_builder._temporary_path(transaction_lock),
                        sealed_registry_builder._temporary_path(private_output),
                        sealed_registry_builder._temporary_path(public_receipt),
                    ):
                        self.assertFalse(path.exists())
                    self.assertEqual(sentinel.read_text(encoding="utf-8"), "preserve")

    def test_sealed_registry_stage_race_preserves_foreign_final_and_building(self) -> None:
        root_pin = {
            "path": "reports/step28_v13_v1_13_scientific_builder/"
            "design_preflight_v2_20260811/root_manifest.json",
            "size_bytes": sealed_registry_builder.EXPECTED_ROOT_MANIFEST_SIZE_BYTES,
            "sha256": sealed_registry_builder.EXPECTED_ROOT_MANIFEST_SHA256,
            "canonical_self_hash": (
                sealed_registry_builder.EXPECTED_ROOT_MANIFEST_SELF_HASH
            ),
        }
        source_closure = sealed_registry_builder._capture_runtime_source_closure()
        builder_policy_pin = _actual_builder_policy_pin()
        sidecar = {
            "version": sealed_registry_builder.VERSION,
            "status": "SEALED_RELATION_FREE_LITERAL_REGISTRY",
            "dataset_root_manifest": root_pin,
            "builder_policy": builder_policy_pin,
            "canonical_self_hash": None,
        }
        sidecar["canonical_self_hash"] = (
            sealed_registry_builder._canonical_self_hash(sidecar)
        )
        receipt = _packer_receipt_base_fixture(
            root_pin=root_pin,
            builder_policy_pin=builder_policy_pin,
            source_closure=source_closure,
        )
        for stage in ("intent", "sidecar", "receipt"):
            for occupied_kind in ("final", "building"):
                with self.subTest(
                    stage=stage, occupied_kind=occupied_kind
                ), tempfile.TemporaryDirectory(dir=ROOT) as directory:
                    root = Path(directory)
                    private_output = root / "private" / "registry.json"
                    public_receipt = root / "public" / "receipt.json"
                    transaction_intent = root / "private" / "transaction.json"
                    transaction_lock = root / "private" / "transaction.lock.json"
                    stage_paths = {
                        "intent": transaction_intent,
                        "sidecar": private_output,
                        "receipt": public_receipt,
                    }
                    occupied = stage_paths[stage]
                    if occupied_kind == "building":
                        occupied = sealed_registry_builder._temporary_path(occupied)

                    def inject_after_clean_replay() -> tuple[object, object]:
                        occupied.parent.mkdir(parents=True, exist_ok=True)
                        occupied.write_text("foreign-stage-owner", encoding="utf-8")
                        return sidecar, receipt

                    with (
                        patch.object(
                            sealed_registry_builder,
                            "PRIVATE_OUTPUT",
                            private_output,
                        ),
                        patch.object(
                            sealed_registry_builder,
                            "PUBLIC_RECEIPT",
                            public_receipt,
                        ),
                        patch.object(
                            sealed_registry_builder,
                            "TRANSACTION_INTENT",
                            transaction_intent,
                        ),
                        patch.object(
                            sealed_registry_builder,
                            "TRANSACTION_LOCK",
                            transaction_lock,
                        ),
                        patch.object(
                            sealed_registry_builder,
                            "_replay_and_collect",
                            side_effect=inject_after_clean_replay,
                        ),
                        self.assertRaises(
                            sealed_registry_builder.SealedLiteralRegistryBuildError
                        ),
                    ):
                        sealed_registry_builder.run_build()
                    self.assertEqual(
                        occupied.read_text(encoding="utf-8"),
                        "foreign-stage-owner",
                    )
                    self.assertFalse(transaction_lock.exists())
                    for owned_candidate in (
                        transaction_intent,
                        private_output,
                        public_receipt,
                    ):
                        if owned_candidate == occupied:
                            continue
                        self.assertFalse(owned_candidate.exists())

    def test_sealed_registry_exclusive_lock_allows_only_one_replay(self) -> None:
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            root = Path(directory)
            private_output = root / "private" / "registry.json"
            public_receipt = root / "public" / "receipt.json"
            transaction_intent = root / "private" / "transaction.json"
            transaction_lock = root / "private" / "transaction.lock.json"
            root_pin = {
                "path": "reports/step28_v13_v1_13_scientific_builder/"
                "design_preflight_v2_20260811/root_manifest.json",
                "size_bytes": sealed_registry_builder.EXPECTED_ROOT_MANIFEST_SIZE_BYTES,
                "sha256": sealed_registry_builder.EXPECTED_ROOT_MANIFEST_SHA256,
                "canonical_self_hash": (
                    sealed_registry_builder.EXPECTED_ROOT_MANIFEST_SELF_HASH
                ),
            }
            source_closure = sealed_registry_builder._capture_runtime_source_closure()
            builder_policy_pin = _actual_builder_policy_pin()
            sidecar = {
                "version": sealed_registry_builder.VERSION,
                "status": "SEALED_RELATION_FREE_LITERAL_REGISTRY",
                "dataset_root_manifest": root_pin,
                "builder_policy": builder_policy_pin,
                "canonical_self_hash": None,
            }
            sidecar["canonical_self_hash"] = (
                sealed_registry_builder._canonical_self_hash(sidecar)
            )
            receipt = _packer_receipt_base_fixture(
                root_pin=root_pin,
                builder_policy_pin=builder_policy_pin,
                source_closure=source_closure,
            )
            replay_entered = threading.Event()
            replay_release = threading.Event()
            results: list[dict[str, object]] = []
            errors: list[BaseException] = []

            def delayed_replay() -> tuple[object, object]:
                replay_entered.set()
                if not replay_release.wait(timeout=30):
                    raise AssertionError("concurrent replay release timeout")
                return sidecar, receipt

            def invoke() -> None:
                try:
                    results.append(sealed_registry_builder.run_build())
                except BaseException as exc:
                    errors.append(exc)

            with (
                patch.object(
                    sealed_registry_builder, "PRIVATE_OUTPUT", private_output
                ),
                patch.object(
                    sealed_registry_builder, "PUBLIC_RECEIPT", public_receipt
                ),
                patch.object(
                    sealed_registry_builder,
                    "TRANSACTION_INTENT",
                    transaction_intent,
                ),
                patch.object(
                    sealed_registry_builder,
                    "TRANSACTION_LOCK",
                    transaction_lock,
                ),
                patch.object(
                    sealed_registry_builder,
                    "_replay_and_collect",
                    side_effect=delayed_replay,
                ),
            ):
                first = threading.Thread(target=invoke)
                first.start()
                self.assertTrue(replay_entered.wait(timeout=30))
                lock_bytes = transaction_lock.read_bytes()
                try:
                    with self.assertRaises(
                        sealed_registry_builder.SealedLiteralRegistryBuildError
                    ):
                        sealed_registry_builder.recover_interrupted_transaction()
                    self.assertEqual(transaction_lock.read_bytes(), lock_bytes)
                    second = threading.Thread(target=invoke)
                    second.start()
                    second.join(timeout=30)
                    self.assertFalse(second.is_alive())
                    self.assertEqual(transaction_lock.read_bytes(), lock_bytes)
                finally:
                    replay_release.set()
                    first.join(timeout=30)
                self.assertFalse(first.is_alive())
            self.assertEqual(len(results), 1)
            self.assertEqual(len(errors), 1)
            self.assertIsInstance(
                errors[0], sealed_registry_builder.SealedLiteralRegistryBuildError
            )
            self.assertTrue(private_output.is_file())
            self.assertTrue(public_receipt.is_file())
            self.assertFalse(transaction_intent.exists())
            self.assertFalse(transaction_lock.exists())

    def test_sealed_registry_automatic_recovery_is_disabled_and_read_only(self) -> None:
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            root = Path(directory)
            private_output = root / "private" / "registry.json"
            public_receipt = root / "public" / "receipt.json"
            transaction_intent = root / "private" / "transaction.json"
            transaction_lock = root / "private" / "transaction.lock.json"
            artifacts = (
                transaction_lock,
                sealed_registry_builder._temporary_path(transaction_lock),
                transaction_intent,
                private_output,
                public_receipt,
            )
            for index, path in enumerate(artifacts):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(f"preserve-{index}".encode("ascii"))
            before = {path: path.read_bytes() for path in artifacts}
            with (
                patch.object(
                    sealed_registry_builder, "PRIVATE_OUTPUT", private_output
                ),
                patch.object(
                    sealed_registry_builder, "PUBLIC_RECEIPT", public_receipt
                ),
                patch.object(
                    sealed_registry_builder,
                    "TRANSACTION_INTENT",
                    transaction_intent,
                ),
                patch.object(
                    sealed_registry_builder,
                    "TRANSACTION_LOCK",
                    transaction_lock,
                ),
                self.assertRaises(
                    sealed_registry_builder.SealedLiteralRegistryBuildError
                ),
            ):
                sealed_registry_builder.recover_interrupted_transaction()
            self.assertEqual(
                {path: path.read_bytes() for path in artifacts}, before
            )

    def test_sealed_registry_run_build_commits_once_and_refuses_overwrite(self) -> None:
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            root = Path(directory)
            private_output = root / "private" / "registry.json"
            public_receipt = root / "public" / "receipt.json"
            transaction_intent = root / "private" / "transaction.json"
            transaction_lock = root / "private" / "transaction.lock.json"
            root_pin = {
                "path": "reports/step28_v13_v1_13_scientific_builder/"
                "design_preflight_v2_20260811/root_manifest.json",
                "size_bytes": sealed_registry_builder.EXPECTED_ROOT_MANIFEST_SIZE_BYTES,
                "sha256": sealed_registry_builder.EXPECTED_ROOT_MANIFEST_SHA256,
                "canonical_self_hash": (
                    sealed_registry_builder.EXPECTED_ROOT_MANIFEST_SELF_HASH
                ),
            }
            source_closure = (
                sealed_registry_builder._capture_runtime_source_closure()
            )
            builder_policy_pin = _actual_builder_policy_pin()
            sidecar = {
                "version": sealed_registry_builder.VERSION,
                "status": "SEALED_RELATION_FREE_LITERAL_REGISTRY",
                "dataset_root_manifest": root_pin,
                "builder_policy": builder_policy_pin,
                "canonical_self_hash": None,
            }
            sidecar["canonical_self_hash"] = (
                sealed_registry_builder._canonical_self_hash(sidecar)
            )
            receipt = _packer_receipt_base_fixture(
                root_pin=root_pin,
                builder_policy_pin=builder_policy_pin,
                source_closure=source_closure,
            )
            with (
                patch.object(
                    sealed_registry_builder, "PRIVATE_OUTPUT", private_output
                ),
                patch.object(
                    sealed_registry_builder, "PUBLIC_RECEIPT", public_receipt
                ),
                patch.object(
                    sealed_registry_builder,
                    "TRANSACTION_INTENT",
                    transaction_intent,
                ),
                patch.object(
                    sealed_registry_builder,
                    "TRANSACTION_LOCK",
                    transaction_lock,
                ),
                patch.object(
                    sealed_registry_builder,
                    "_replay_and_collect",
                    return_value=(sidecar, receipt),
                ),
            ):
                committed = sealed_registry_builder.run_build()
                self.assertEqual(
                    committed["status"],
                    "PASS_SEALED_LITERAL_REGISTRY_BUILD_NOT_AUDIT_GO",
                )
                self.assertTrue(private_output.is_file())
                self.assertTrue(public_receipt.is_file())
                self.assertFalse(transaction_intent.exists())
                self.assertFalse(transaction_lock.exists())
                with self.assertRaises(
                    sealed_registry_builder.SealedLiteralRegistryBuildError
                ):
                    sealed_registry_builder.run_build()

    def test_sealed_registry_packer_rejects_projection_universe_drift(self) -> None:
        exact = {
            category: [f"value_{category}"]
            for category in blind_literal_scan.SEALED_REGISTRY_CATEGORIES
        }
        sealed_registry_builder._validate_split_projection(
            split="audit_a",
            categories=exact,
            allowed_noise=["visible_noise"],
            world_count=2,
            expected_world_count=2,
        )
        mutations = []
        missing = dict(exact)
        del missing[blind_literal_scan.SEALED_REGISTRY_CATEGORIES[0]]
        mutations.append((missing, ["visible_noise"], 2))
        extra = dict(exact)
        extra["unknown_category"] = ["unknown"]
        mutations.append((extra, ["visible_noise"], 2))
        reversed_order = dict(reversed(tuple(exact.items())))
        mutations.append((reversed_order, ["visible_noise"], 2))
        empty_category = dict(exact)
        empty_category[blind_literal_scan.SEALED_REGISTRY_CATEGORIES[0]] = []
        mutations.append((empty_category, ["visible_noise"], 2))
        mutations.append((exact, [], 2))
        mutations.append((exact, ["visible_noise"], 1))
        mutations.append((exact, ["visible_noise"], 3))
        for categories, allowed_noise, world_count in mutations:
            with self.assertRaises(
                sealed_registry_builder.SealedLiteralRegistryBuildError
            ):
                sealed_registry_builder._validate_split_projection(
                    split="audit_a",
                    categories=categories,
                    allowed_noise=allowed_noise,
                    world_count=world_count,
                    expected_world_count=2,
                )

    def test_sealed_registry_source_and_root_raw_closure_fail_closed(self) -> None:
        closure = sealed_registry_builder._capture_runtime_source_closure()
        self.assertEqual(len(closure), 25)
        for name in (
            "step3_build_seller_profiles",
            "step7_v3_1_source_data",
            "step7_v4_common",
        ):
            self.assertIn(name, closure)
        sealed_registry_builder._verify_runtime_source_closure(closure)
        changed = copy.deepcopy(closure)
        changed["step28_v13_v1_13_blind_literal_scan"]["sha256"] = "0" * 64
        with self.assertRaises(
            sealed_registry_builder.SealedLiteralRegistryBuildError
        ):
            sealed_registry_builder._verify_runtime_source_closure(changed)
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            fake = Path(directory) / "step28_v13_v1_13_blind_literal_scan.py"
            fake.write_text("# fake module path\n", encoding="utf-8")
            with (
                patch.object(blind_literal_scan, "__file__", str(fake)),
                self.assertRaises(
                    sealed_registry_builder.SealedLiteralRegistryBuildError
                ),
            ):
                sealed_registry_builder._capture_runtime_source_closure()

            dataset_root = Path(directory) / "dataset"
            dataset_root.mkdir()
            actual = common.load_json(
                sealed_registry_builder.DATASET_ROOT / "root_manifest.json"
            )
            (dataset_root / "root_manifest.json").write_text(
                json.dumps(
                    actual,
                    ensure_ascii=False,
                    sort_keys=False,
                    indent=4,
                    allow_nan=False,
                )
                + "\n",
                encoding="utf-8",
            )
            with (
                patch.object(
                    sealed_registry_builder, "DATASET_ROOT", dataset_root
                ),
                self.assertRaises(
                    sealed_registry_builder.SealedLiteralRegistryBuildError
                ),
            ):
                sealed_registry_builder._root_manifest_record()
            policy_path = Path(directory) / "builder_policy.json"
            actual_policy = scientific.load_policy()
            policy_path.write_text(
                json.dumps(
                    actual_policy,
                    ensure_ascii=False,
                    sort_keys=False,
                    indent=4,
                    allow_nan=False,
                )
                + "\n",
                encoding="utf-8",
            )
            with (
                patch.object(
                    scientific, "DEFAULT_POLICY_PATH", policy_path
                ),
                self.assertRaises(
                    sealed_registry_builder.SealedLiteralRegistryBuildError
                ),
            ):
                sealed_registry_builder._builder_policy_record(actual_policy)
        policy = scientific.load_policy()
        root_manifest = common.load_json(
            sealed_registry_builder.DATASET_ROOT / "root_manifest.json"
        )
        self.assertEqual(
            root_manifest["builder_policy_canonical_self_hash"],
            policy["canonical_self_hash"],
        )

    def test_sealed_registry_source_closure_is_recursively_derived(self) -> None:
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            scripts_root = Path(directory) / "scripts"
            scripts_root.mkdir()
            entry = scripts_root / "entry.py"
            entry.write_text("import module_b\n", encoding="utf-8")
            (scripts_root / "module_b.py").write_text(
                "import module_c\n", encoding="utf-8"
            )
            (scripts_root / "module_c.py").write_text(
                "import importlib\n"
                "importlib.import_module('module_d')\n",
                encoding="utf-8",
            )
            (scripts_root / "module_d.py").write_text(
                "VALUE = 1\n", encoding="utf-8"
            )
            (scripts_root / "unrelated.py").write_text(
                "VALUE = 2\n", encoding="utf-8"
            )
            closure = (
                sealed_registry_builder._discover_repo_local_module_paths(
                    entry_path=entry,
                    scripts_root=scripts_root,
                )
            )
            self.assertEqual(
                tuple(closure),
                ("module_b", "module_c", "module_d", "packer"),
            )
            self.assertNotIn("unrelated", closure)
            (scripts_root / "module_b.py").write_text(
                "import module_c\nimport unrelated\n", encoding="utf-8"
            )
            changed = (
                sealed_registry_builder._discover_repo_local_module_paths(
                    entry_path=entry,
                    scripts_root=scripts_root,
                )
            )
            self.assertIn("unrelated", changed)

    def test_sealed_registry_disabled_recovery_preserves_every_artifact(self) -> None:
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            root = Path(directory)
            private_output = root / "private" / "registry.json"
            public_receipt = root / "public" / "receipt.json"
            intent_path = root / "private" / "transaction.json"
            lock_path = root / "private" / "transaction.lock.json"
            paths = (
                lock_path,
                sealed_registry_builder._temporary_path(lock_path),
                intent_path,
                sealed_registry_builder._temporary_path(intent_path),
                private_output,
                sealed_registry_builder._temporary_path(private_output),
                public_receipt,
                sealed_registry_builder._temporary_path(public_receipt),
            )
            for populated_count in range(len(paths) + 1):
                with self.subTest(populated_count=populated_count):
                    for path in paths:
                        if path.exists():
                            path.unlink()
                    for index, path in enumerate(paths[:populated_count]):
                        path.parent.mkdir(parents=True, exist_ok=True)
                        path.write_bytes(f"audit-{index}".encode("ascii"))
                    before = {
                        path: (path.is_file(), path.read_bytes() if path.is_file() else None)
                        for path in paths
                    }
                    with (
                        patch.object(
                            sealed_registry_builder,
                            "PRIVATE_OUTPUT",
                            private_output,
                        ),
                        patch.object(
                            sealed_registry_builder,
                            "PUBLIC_RECEIPT",
                            public_receipt,
                        ),
                        patch.object(
                            sealed_registry_builder,
                            "TRANSACTION_INTENT",
                            intent_path,
                        ),
                        patch.object(
                            sealed_registry_builder,
                            "TRANSACTION_LOCK",
                            lock_path,
                        ),
                        self.assertRaises(
                            sealed_registry_builder.SealedLiteralRegistryBuildError
                        ),
                    ):
                        sealed_registry_builder.recover_interrupted_transaction()
                    after = {
                        path: (path.is_file(), path.read_bytes() if path.is_file() else None)
                        for path in paths
                    }
                    self.assertEqual(after, before)

    def test_sealed_registry_pair_label_projection_never_persists_labels(self) -> None:
        rows = [
            {
                "canonical_pair_uid": "pair_positive",
                "world_uid": "world_fixture",
                "label": "1",
            },
            {
                "canonical_pair_uid": "pair_negative",
                "world_uid": "world_fixture",
                "label": "0",
            },
        ]
        projection = sealed_registry_builder._pair_label_literal_projection(rows)
        self.assertEqual(
            projection,
            [
                {
                    "canonical_pair_uid": "pair_positive",
                    "world_uid": "world_fixture",
                },
                {
                    "canonical_pair_uid": "pair_negative",
                    "world_uid": "world_fixture",
                },
            ],
        )
        serialized = common.canonical_json_bytes(projection).decode("utf-8")
        self.assertNotIn('"label"', serialized)
        bad = [dict(rows[0], extra_private_field="canary")]
        with self.assertRaises(
            sealed_registry_builder.SealedLiteralRegistryBuildError
        ):
            sealed_registry_builder._pair_label_literal_projection(bad)

    def test_sealed_registry_isolation_closes_three_private_universes(self) -> None:
        def build_fixture(root: Path, duplicate_kind: str | None = None) -> None:
            split_values: dict[str, dict[str, set[str]]] = {}
            split_hashes: dict[str, str] = {}
            for split in blind_literal_scan.SPLITS:
                controller = f"controller_{split}"
                query = f"query_{split}"
                identity = f"identity_{split}"
                if split == "audit_a" and duplicate_kind == "controller_uid":
                    controller = "controller_train"
                if split == "audit_a" and duplicate_kind == "query_uid":
                    query = "query_train"
                if split == "audit_a" and duplicate_kind == "identity_value":
                    identity = "identity_train"
                split_root = root / split
                (split_root / "private").mkdir(parents=True)
                rows = {
                    "private/controller_membership.jsonl": [
                        {
                            "world_uid": f"world_{split}",
                            "seller_uid": f"seller_{split}",
                            "controller_uid": controller,
                        }
                    ],
                    "private/qrels.jsonl": [
                        {
                            "world_uid": f"world_{split}",
                            "query_uid": query,
                            "query_seller_uid": f"seller_{split}",
                            "relevant_seller_uids": [],
                        }
                    ],
                    "private/world_generation_audit.jsonl": [
                        {
                            "world_uid": f"world_{split}",
                            "identity_assets": [{"identity_value": identity}],
                        }
                    ],
                    "private/document_collision_attempts.jsonl": [
                        {
                            "world_uid": f"world_{split}",
                            "receipt_marker": f"collision_{split}",
                        }
                    ],
                    "private/identity_allocation_receipts.jsonl": [
                        {
                            "world_uid": f"world_{split}",
                            "receipt_marker": f"allocation_{split}",
                        }
                    ],
                }
                records = []
                for relative, values in rows.items():
                    path = split_root / relative
                    path.write_text(
                        "".join(
                            common.canonical_json_bytes(value).decode("utf-8") + "\n"
                            for value in values
                        ),
                        encoding="utf-8",
                    )
                    records.append(
                        {
                            "path": relative,
                            "row_count": len(values),
                            "size_bytes": path.stat().st_size,
                            "sha256": common.sha256_file(path),
                        }
                    )
                identity_hash = blind_literal_scan._identity_value_hash(identity)
                values = {
                    "identity_value": {identity_hash},
                    "controller_uid": {controller},
                    "query_uid": {query},
                }
                split_values[split] = values
                manifest = {
                    "split": split,
                    "files": records,
                    "identity_value_registry_count": 1,
                    "identity_value_registry_sha256": common.canonical_sha256(
                        sorted(values["identity_value"])
                    ),
                    "uid_registries": {
                        "controller": {
                            "count": 1,
                            "sha256": common.canonical_sha256(
                                sorted(values["controller_uid"])
                            ),
                        },
                        "query": {
                            "count": 1,
                            "sha256": common.canonical_sha256(
                                sorted(values["query_uid"])
                            ),
                        },
                    },
                    "canonical_self_hash": None,
                }
                manifest["canonical_self_hash"] = blind_literal_scan._canonical_self_hash(
                    manifest
                )
                common.write_json(split_root / "split_manifest.json", manifest)
                split_hashes[split] = manifest["canonical_self_hash"]
            unions = {
                kind: set().union(
                    *(split_values[split][kind] for split in blind_literal_scan.SPLITS)
                )
                for kind in ("identity_value", "controller_uid", "query_uid")
            }
            root_manifest = {
                "split_order": list(blind_literal_scan.SPLITS),
                "split_manifest_self_hashes": split_hashes,
                "identity_value_registry_count": len(unions["identity_value"]),
                "identity_value_registry_sha256": common.canonical_sha256(
                    sorted(unions["identity_value"])
                ),
                "uid_registries": {
                    "controller": {
                        "count": len(unions["controller_uid"]),
                        "sha256": common.canonical_sha256(
                            sorted(unions["controller_uid"])
                        ),
                    },
                    "query": {
                        "count": len(unions["query_uid"]),
                        "sha256": common.canonical_sha256(
                            sorted(unions["query_uid"])
                        ),
                    },
                },
                "canonical_self_hash": None,
            }
            root_manifest["canonical_self_hash"] = blind_literal_scan._canonical_self_hash(
                root_manifest
            )
            common.write_json(root / "root_manifest.json", root_manifest)

        with tempfile.TemporaryDirectory() as directory:
            fixture_root = Path(directory)
            build_fixture(fixture_root)
            receipt = blind_literal_scan.scan_registry_isolation(fixture_root)
            self.assertEqual(
                receipt["status"], "PASS_PRIVATE_REGISTRY_SPLIT_ISOLATION"
            )
            self.assertEqual(
                receipt["cross_split_overlap_counts"],
                {"controller_uid": 0, "identity_value": 0, "query_uid": 0},
            )
            self.assertEqual(receipt["private_values_returned"], 0)
            counters = quality._new_blind_boundary_counters()
            wrapped = quality._run_blind_registry_isolation_scan(
                policy={
                    "pins": {
                        "blind_literal_scan": {
                            "path": "scripts/step28_v13_v1_13_blind_literal_scan.py"
                        }
                    }
                },
                dataset_root=fixture_root,
                root_manifest=common.load_json(fixture_root / "root_manifest.json"),
                split_manifests={
                    split: common.load_json(
                        fixture_root / split / "split_manifest.json"
                    )
                    for split in blind_literal_scan.SPLITS
                },
                blind_counters=counters,
            )
            self.assertEqual(wrapped["canonical_self_hash"], receipt["canonical_self_hash"])
            for split in ("audit_a", "audit_b"):
                self.assertEqual(counters[split]["sealed_registry_isolation_calls"], 1)
                self.assertEqual(counters[split]["truth_read_requests"], 0)
                self.assertEqual(counters[split]["private_payload_open_requests"], 0)
        for kind in ("identity_value", "controller_uid", "query_uid"):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as directory:
                build_fixture(Path(directory), duplicate_kind=kind)
                with self.assertRaises(blind_literal_scan.BlindLiteralInputError):
                    blind_literal_scan.scan_registry_isolation(Path(directory))

    def test_blind_visible_numeric_profile_replay_uses_redacted_items_only(self) -> None:
        items = [
            {
                "seller_uid": "s0",
                "title": " A  12 ",
                "description": "甲!",
            },
            {
                "seller_uid": "s0",
                "title": "A 12",
                "description": "",
            },
        ]
        expected = quality._blind_visible_numeric_profiles(items)["s0"]
        self.assertEqual(expected["item_count"], 2)
        self.assertEqual(expected["title_length_median"], 4)
        self.assertEqual(expected["description_length_median"], 1)
        self.assertEqual(expected["repeated_title_share"], 0.5)
        self.assertEqual(expected["repeated_description_share"], 0.5)

    def test_blind_counter_schema_and_final_closure_share_one_authority(self) -> None:
        counters = quality._new_blind_boundary_counters()
        for split in ("audit_a", "audit_b"):
            counters[split]["sealed_literal_scan_calls"] += 1
            counters[split]["sealed_registry_isolation_calls"] += 1
        quality._validate_final_blind_boundary_counters(counters)
        mutations = []
        missing = copy.deepcopy(counters)
        del missing["audit_a"]["sealed_registry_isolation_calls"]
        mutations.append(missing)
        zero = copy.deepcopy(counters)
        zero["audit_a"]["sealed_registry_isolation_calls"] = 0
        mutations.append(zero)
        double = copy.deepcopy(counters)
        double["audit_b"]["sealed_registry_isolation_calls"] = 2
        mutations.append(double)
        extra = copy.deepcopy(counters)
        extra["audit_b"]["unknown_counter"] = 0
        mutations.append(extra)
        for mutated in mutations:
            with self.subTest(mutated=mutated), self.assertRaises(
                quality.ScientificQualityAuditError
            ):
                quality._validate_final_blind_boundary_counters(mutated)

    def test_blind_description_repeat_replay_uses_production_280_char_snippet(self) -> None:
        shared_prefix = "甲" * 280
        items = [
            {
                "seller_uid": "s0",
                "title": "标题一",
                "description": shared_prefix + "后缀一",
            },
            {
                "seller_uid": "s0",
                "title": "标题二",
                "description": shared_prefix + "后缀二",
            },
        ]
        expected = quality._blind_visible_numeric_profiles(items)["s0"]
        self.assertEqual(expected["description_length_median"], 283)
        self.assertEqual(expected["repeated_description_share"], 0.5)

    def test_counterfactual_scan_failure_is_not_dataset_invalidation(self) -> None:
        profile = {field: "" for field in quality.VISIBLE_PROFILE_FIELDS}
        profile["seller_uid"] = "seller_fixture"
        with self.assertRaises(quality.ScientificQualityAuditError) as captured:
            quality._scan_visible_text(
                profiles=[profile],
                redacted_items=[{"title": "PrivateSecret", "description": ""}],
                forbidden_markers=("internal_marker",),
                forbidden_literals=("privatesecret",),
                failure_domain="auditor_counterfactual",
            )
        self.assertNotIsInstance(captured.exception, quality.DatasetInvalidationError)
        self.assertEqual(
            quality._failure_classification(captured.exception),
            "AUDITOR_EXECUTION_FAILED_NO_DATASET_CONCLUSION",
        )

    def test_counterfactual_invariant_and_true_low_dose_are_classified_separately(self) -> None:
        contract = {
            "required_source_seller_changed_count": 28,
            "minimum_effective_style_uid_changed_count": 27,
            "minimum_effective_style_factor_tuple_changed_count": 27,
            "minimum_seller_profile_text_changed_count": 1,
            "minimum_visible_seller_changed_count": 1,
            "maximum_zero_dose_seller_count": 1,
            "maximum_zero_visible_dose_seller_count": 1,
        }
        dose = {
            "source_seller_changed_count": 28,
            "effective_style_uid_changed_count": 28,
            "effective_style_factor_tuple_changed_count": 28,
            "seller_profile_text_changed_count": 28,
            "visible_seller_changed_count": 28,
            "zero_dose_seller_count": 0,
            "zero_visible_dose_seller_count": 0,
        }
        audit = {
            "original_style_uid_multiset_sha256": "a" * 64,
            "mapped_style_uid_multiset_sha256": "a" * 64,
            "original_style_factor_multiset_sha256": "b" * 64,
            "mapped_style_factor_multiset_sha256": "b" * 64,
        }
        bad_mapping = dict(audit)
        bad_mapping["mapped_style_uid_multiset_sha256"] = "c" * 64
        with self.assertRaises(quality.ScientificQualityAuditError) as captured:
            quality._validate_counterfactual_dose(
                dose=dose,
                counterfactual_audit=bad_mapping,
                dose_contract=contract,
            )
        self.assertNotIsInstance(captured.exception, quality.DatasetInvalidationError)
        low_dose = dict(dose)
        low_dose["visible_seller_changed_count"] = 0
        with self.assertRaises(quality.DatasetInvalidationError):
            quality._validate_counterfactual_dose(
                dose=low_dose,
                counterfactual_audit=audit,
                dose_contract=contract,
            )

    def test_sealed_scanner_classifies_malformed_dataset_input_without_echo(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            split_root = Path(directory) / "audit_a"
            (split_root / "private").mkdir(parents=True)
            (split_root / "observed").mkdir()
            secret = "private_secret_must_not_echo"
            (split_root / "private" / "controller_membership.jsonl").write_text(
                "{" + secret,
                encoding="utf-8",
            )
            valid_private = {
                "qrels.jsonl": {
                    "world_uid": "world_fixture",
                    "query_uid": "query_secret",
                },
                "world_generation_audit.jsonl": {
                    "world_uid": "world_fixture",
                    "identity_assets": [{"identity_value": "identity_secret"}],
                },
                "document_collision_attempts.jsonl": {
                    "world_uid": "world_fixture",
                    "receipt_marker": "collision_receipt_secret",
                },
                "identity_allocation_receipts.jsonl": {
                    "world_uid": "world_fixture",
                    "receipt_marker": "allocation_receipt_secret",
                },
            }
            for name, row in valid_private.items():
                (split_root / "private" / name).write_text(
                    common.canonical_json_bytes(row).decode("utf-8") + "\n",
                    encoding="utf-8",
                )
            profile = {name: "" for name in quality.VISIBLE_PROFILE_FIELDS}
            profile["seller_uid"] = "seller_fixture"
            (split_root / "observed" / "model_seller_profiles.jsonl").write_text(
                common.canonical_json_bytes(profile).decode("utf-8") + "\n",
                encoding="utf-8",
            )
            (split_root / "observed" / "redacted_items.jsonl").write_text(
                common.canonical_json_bytes(
                    {"title": "visible", "description": ""}
                ).decode("utf-8")
                + "\n",
                encoding="utf-8",
            )
            registry = _write_scan_sidecar(Path(directory))
            completed = subprocess.run(
                [
                    sys.executable,
                    "-I",
                    "-S",
                    "-B",
                    str(SCRIPTS / "step28_v13_v1_13_blind_literal_scan.py"),
                    "--dataset-root",
                    directory,
                    "--split",
                    "audit_a",
                    "--sealed-registry",
                    str(registry),
                ],
                cwd=ROOT,
                text=True,
                encoding="utf-8",
                capture_output=True,
                timeout=30,
                check=False,
            )
            self.assertEqual(completed.returncode, 3)
            receipt = json.loads(completed.stdout)
            self.assertEqual(receipt["status"], "FAIL_SEALED_INPUT_INVALID")
            self.assertNotIn(secret, completed.stdout)
            self.assertEqual(receipt["private_values_returned"], 0)

    def test_duplicate_csv_header_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "duplicate.csv"
            path.write_text("a,b,a\n1,2,3\n", encoding="utf-8")
            with self.assertRaises(quality.DatasetInvalidationError):
                quality._read_csv(path)

    def test_extra_dataset_file_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "expected.txt").write_text("ok", encoding="utf-8")
            quality._assert_exact_file_universe(root, {"expected.txt"})
            (root / "unexpected.txt").write_text("extra", encoding="utf-8")
            with self.assertRaises(quality.DatasetInvalidationError):
                quality._assert_exact_file_universe(root, {"expected.txt"})


class QualityAuditProductionReplay(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.audit_policy = quality.load_policy()
        cls.builder_policy = scientific.load_policy()
        cls.context = scientific.build_execution_context(
            cls.builder_policy, execution_mode="design_preflight"
        )
        cls.template, fixture, style_profile = scientific.load_release_inputs(cls.context)
        historical = collision.load_historical_exclusion_registries()
        cls.historical = historical
        record = sorted(
            cls.context.world_records,
            key=lambda row: (
                scientific.SPLITS.index(str(row["split"])),
                int(row["split_ordinal"]),
            ),
        )[0]
        split = str(record["split"])
        cls.accepted = world_module.build_scientific_world(
            policy=cls.context.effective_policy,
            template=cls.template,
            fixture=fixture,
            style_profile=style_profile,
            mode=cls.context.base_mode,
            world_record=record,
            structure_key_hex=common.structure_key_for_split(
                cls.context.effective_policy,
                mode=cls.context.base_mode,
                split=split,
            ),
            document_variation_key=cls.context.document_variation_key,
            anonymous_handle_key=cls.context.anonymous_handle_key,
            historical_item_hashes=historical.item_document_hashes,
            historical_seller_hashes=historical.seller_document_hashes,
            historical_identity_hashes=historical.identity_value_hashes,
            current_item_hashes=set(),
            current_seller_hashes=set(),
            current_identity_hashes=set(),
        )
        cls.counterfactual = counterfactual.rerender_counterfactual_world(
            cls.context.effective_policy,
            mode=cls.context.base_mode,
            split=split,
            template=cls.template,
            sellers=cls.accepted.world["public"]["sellers"],
            items=cls.accepted.world["public"]["items"],
            identity_slots_audit=cls.accepted.world["private"]["identity_slots_audit"],
            noise_slots_audit=cls.accepted.world["private"]["noise_slots_audit"],
            render_asts=cls.accepted.world["private"]["render_asts"],
            override_audit=cls.accepted.world["private"]["override_audit"],
        )

    def test_real_noise_uid_is_forbidden_while_raw_noise_surface_remains_allowed(self) -> None:
        noise_rows = self.accepted.world["private"]["noise_slots_audit"]
        self.assertTrue(noise_rows)
        noise_uid = str(noise_rows[0]["noise_slot_uid"])
        raw_surface = str(noise_rows[0]["raw_surface"])
        categories, allowed_noise = sealed_registry_builder._collect_audit_world_literals(
            self.accepted,
            persisted_world_audit=sealed_registry_builder.dataset_builder._private_world_audit_row(
                self.accepted
            ),
            persisted_collision=sealed_registry_builder._expected_collision_row(
                self.accepted
            ),
            persisted_identity_allocation=sealed_registry_builder._expected_identity_allocation_row(
                self.accepted
            ),
            persisted_controller_membership=list(
                self.accepted.controller_membership
            ),
            persisted_qrels=list(self.accepted.qrels),
            persisted_pair_labels=sealed_registry_builder._expected_pair_label_rows(
                self.accepted
            ),
        )
        self.assertIn(noise_uid, categories["full_world_forbidden"])
        self.assertIn(raw_surface, allowed_noise)
        self.assertNotIn(raw_surface, categories["full_world_forbidden"])

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write_basic_scan_split(root, split="audit_a", visible=raw_surface)
            registry = _write_scan_sidecar(
                root,
                extra_forbidden_by_split={"audit_a": (noise_uid,)},
            )
            allowed_receipt = blind_literal_scan.scan(root, "audit_a", registry)
            self.assertEqual(allowed_receipt["status"], "PASS_NO_PRIVATE_LITERAL_HIT")
            item_path = root / "audit_a" / "observed" / "redacted_items.jsonl"
            item_path.write_text(
                common.canonical_json_bytes(
                    {"title": noise_uid, "description": ""}
                ).decode("utf-8")
                + "\n",
                encoding="utf-8",
            )
            blocked_receipt = blind_literal_scan.scan(root, "audit_a", registry)
            self.assertEqual(blocked_receipt["status"], "FAIL_PRIVATE_LITERAL_HIT")

    def test_counterfactual_changes_only_style_and_preserves_identity33(self) -> None:
        profiles, redacted, provenance, delta = quality._recompute_counterfactual_identity33(
            policy=self.context.effective_policy,
            mode=self.context.base_mode,
            split=self.accepted.split,
            template=self.template,
            accepted=self.accepted,
            counterfactual=self.counterfactual,
        )
        self.assertEqual(len(profiles), 28)
        self.assertEqual(len(redacted), len(self.accepted.redacted_items))
        self.assertNotEqual(
            common.canonical_sha256(provenance),
            self.accepted.profile_provenance_sha256,
        )
        self.assertEqual(delta["original_contribution_row_count"], 576)
        self.assertEqual(delta["counterfactual_contribution_row_count"], 574)
        self.assertEqual(
            delta["canonical_self_hash"],
            common.canonical_sha256(
                {key: value for key, value in delta.items() if key != "canonical_self_hash"}
            ),
        )
        self.assertGreater(self.counterfactual["audit"]["changed_title_count"], 0)
        self.assertGreater(self.counterfactual["audit"]["changed_description_count"], 0)
        self.assertEqual(
            self.counterfactual["audit"]["source_seller_changed_count"], 28
        )
        self.assertEqual(
            self.counterfactual["audit"]["effective_style_uid_changed_count"],
            self.counterfactual["audit"]
            ["effective_style_factor_tuple_changed_count"],
        )
        self.assertEqual(
            self.counterfactual["audit"]["zero_dose_seller_count"],
            28
            - self.counterfactual["audit"]
            ["effective_style_factor_tuple_changed_count"],
        )
        self.assertEqual(self.counterfactual["audit"]["zero_dose_seller_count"], 1)
        self.assertEqual(
            self.counterfactual["audit"]["original_style_uid_multiset_sha256"],
            self.counterfactual["audit"]["mapped_style_uid_multiset_sha256"],
        )
        self.assertEqual(
            self.counterfactual["audit"]["original_style_factor_multiset_sha256"],
            self.counterfactual["audit"]["mapped_style_factor_multiset_sha256"],
        )
        self.assertLessEqual(
            self.counterfactual["audit"]["zero_dose_seller_count"],
            self.audit_policy["text_counterfactual"]["intervention_dose"]
            ["maximum_zero_dose_seller_count"],
        )
        original_profiles = {
            str(row["seller_uid"]): row for row in self.accepted.seller_profiles
        }
        counterfactual_profiles = {
            str(row["seller_uid"]): row for row in profiles
        }
        changed_profile_count = sum(
            any(
                str(original_profiles[seller_uid][field])
                != str(counterfactual_profiles[seller_uid][field])
                for field in quality.VISIBLE_PROFILE_FIELDS
            )
            for seller_uid in original_profiles
        )
        self.assertGreaterEqual(
            changed_profile_count,
            self.audit_policy["text_counterfactual"]["intervention_dose"]
            ["minimum_seller_profile_text_changed_count"],
        )

    def test_three_path_receipt_uses_each_feature_builder_returned_order(self) -> None:
        profiles, redacted, _provenance, delta = (
            quality._recompute_counterfactual_identity33(
                policy=self.context.effective_policy,
                mode=self.context.base_mode,
                split=self.accepted.split,
                template=self.template,
                accepted=self.accepted,
                counterfactual=self.counterfactual,
            )
        )
        original_profiles = [
            quality.dataset_builder._project_model_seller_profile(row)
            for row in self.accepted.seller_profiles
        ]
        counterfactual_profiles = [
            quality.dataset_builder._project_model_seller_profile(row)
            for row in profiles
        ]
        original_items = [
            quality.dataset_builder._project_model_redacted_item(row)
            for row in self.accepted.redacted_items
        ]
        counterfactual_items = [
            quality.dataset_builder._project_model_redacted_item(row)
            for row in redacted
        ]
        endpoints = sorted(
            self.accepted.world["public"]["complete_model_pair_endpoints"],
            key=lambda row: str(row["canonical_pair_uid"]).encode("utf-8"),
        )
        label_index = {
            str(row["canonical_pair_uid"]): int(row["label"])
            for row in self.accepted.pair_labels
        }
        excluded = quality._mechanism_neutral_exclusions(
            self.accepted.world["private"]["negative_flags"],
            label_index,
        )
        fixed_support = quality._fixed_support_slot_contract(
            split=self.accepted.split,
            world_uid=self.accepted.world_uid,
            original_items=original_items,
            counterfactual_items=counterfactual_items,
        )
        kwargs = {
            "split": self.accepted.split,
            "split_ordinal": self.accepted.split_ordinal,
            "world_uid": self.accepted.world_uid,
            "target_source_pairs": self.counterfactual["audit"][
                "target_source_pairs"
            ],
            "expected_mapping_sha256": self.counterfactual["audit"][
                "mapping_sha256"
            ],
            "fixed_support": fixed_support,
            "provenance_delta": delta,
            "original_profiles": original_profiles,
            "counterfactual_profiles": counterfactual_profiles,
            "original_items": original_items,
            "counterfactual_items": counterfactual_items,
            "endpoints": endpoints,
            "excluded_pair_uids": excluded,
            "dose": {"world_uid": self.accepted.world_uid},
        }
        receipt = quality._world_three_path_alignment_receipt(**kwargs)
        self.assertNotIn("pair_order_sha256", receipt["common_binding"])
        self.assertEqual(
            receipt["fixed_support"]["pair_order_sha256"],
            receipt["production_step3"]["pair_order_sha256"],
        )

        real_builder = quality.build_fixed_support_text_views

        def reordered_fixed_support(*args: object, **call_kwargs: object):
            views, names, pair_uids, world_uids = real_builder(
                *args,
                **call_kwargs,
            )
            pair_uids = pair_uids.copy()
            pair_uids[[0, 1]] = pair_uids[[1, 0]]
            return views, names, pair_uids, world_uids

        with (
            patch.object(
                quality,
                "build_fixed_support_text_views",
                side_effect=reordered_fixed_support,
            ),
            self.assertRaises(quality.ScientificQualityAuditError),
        ):
            quality._world_three_path_alignment_receipt(**kwargs)

    def test_counterfactual_provenance_canary_fails(self) -> None:
        forged = replace(self.accepted, profile_provenance_sha256="0" * 64)
        with self.assertRaises(quality.ScientificQualityAuditError):
            quality._recompute_counterfactual_identity33(
                policy=self.context.effective_policy,
                mode=self.context.base_mode,
                split=self.accepted.split,
                template=self.template,
                accepted=forged,
                counterfactual=self.counterfactual,
            )
        changed_structure = copy.deepcopy(self.counterfactual)
        source_item = changed_structure["public"]["raw_items"][0]
        source_item["time_bucket"] = (int(source_item["time_bucket"]) + 1) % 4
        with self.assertRaises(quality.ScientificQualityAuditError):
            quality._recompute_counterfactual_identity33(
                policy=self.context.effective_policy,
                mode=self.context.base_mode,
                split=self.accepted.split,
                template=self.template,
                accepted=self.accepted,
                counterfactual=changed_structure,
            )
        _profiles, provenance, _identity33, _redacted = (
            quality.world_module._build_profiles_and_identity33(
                policy=self.context.effective_policy,
                mode=self.context.base_mode,
                split=self.accepted.split,
                template=self.template,
                world=self.accepted.world,
            )
        )
        tampered = copy.deepcopy(provenance)
        tampered["rows"][0]["source_item_uids"] = ["unknown_item"]
        tampered["rows_sha256"] = common.canonical_sha256(tampered["rows"])
        support = {}
        for row in self.accepted.world["public"]["items"]:
            support.setdefault(str(row["seller_uid"]), set()).add(
                str(row["item_uid"])
            )
        with self.assertRaises(quality.ScientificQualityAuditError):
            quality._profile_provenance_delta_receipt(
                world_uid=self.accepted.world_uid,
                original=provenance,
                counterfactual=tampered,
                support_by_seller=support,
                original_items=self.accepted.redacted_items,
                counterfactual_items=self.accepted.redacted_items,
                expected_original_sha256=common.canonical_sha256(provenance),
                expected_counterfactual_sha256=common.canonical_sha256(
                    provenance
                ),
            )

        row_mutations = {
            "world_uid": lambda row: row.__setitem__("world_uid", "wrong_world"),
            "seller_uid": lambda row: row.__setitem__("seller_uid", "wrong_seller"),
            "output_field": lambda row: row.__setitem__("output_field", "unknown"),
            "aggregation_role": lambda row: row.__setitem__("aggregation_role", "unknown"),
            "output_rank": lambda row: row.__setitem__("output_rank", 0),
            "source_item_uids": lambda row: row.__setitem__("source_item_uids", ["unknown_item"]),
            "source_item_uids_sha256": lambda row: row.__setitem__("source_item_uids_sha256", "0" * 64),
            "source_item_count": lambda row: row.__setitem__("source_item_count", int(row["source_item_count"]) + 1),
            "first_seen_position": lambda row: row.__setitem__("first_seen_position", 0),
            "item_uid": lambda row: row.__setitem__("item_uid", "unknown_item"),
            "extracted_segment_ordinal": lambda row: row.__setitem__("extracted_segment_ordinal", 0),
            "seller_df": lambda row: row.__setitem__("seller_df", int(row["seller_df"]) + 1),
            "seller_df_seller_count": lambda row: row.__setitem__("seller_df_seller_count", int(row["seller_df_seller_count"]) + 1),
            "seller_df_seller_uids_sha256": lambda row: row.__setitem__("seller_df_seller_uids_sha256", "0" * 64),
        }
        for field, mutate in row_mutations.items():
            candidate = copy.deepcopy(provenance)
            mutate(candidate["rows"][0])
            candidate["rows_sha256"] = common.canonical_sha256(candidate["rows"])
            with self.subTest(field=field), self.assertRaises(
                quality.ScientificQualityAuditError
            ):
                quality._profile_provenance_delta_receipt(
                    world_uid=self.accepted.world_uid,
                    original=provenance,
                    counterfactual=candidate,
                    support_by_seller=support,
                    original_items=self.accepted.redacted_items,
                    counterfactual_items=self.accepted.redacted_items,
                    expected_original_sha256=common.canonical_sha256(provenance),
                    expected_counterfactual_sha256=common.canonical_sha256(candidate),
                )
        extra = copy.deepcopy(provenance)
        extra["rows"][0]["unexpected"] = 1
        extra["rows_sha256"] = common.canonical_sha256(extra["rows"])
        with self.assertRaises(quality.ScientificQualityAuditError):
            quality._profile_provenance_delta_receipt(
                world_uid=self.accepted.world_uid,
                original=provenance,
                counterfactual=extra,
                support_by_seller=support,
                original_items=self.accepted.redacted_items,
                counterfactual_items=self.accepted.redacted_items,
                expected_original_sha256=common.canonical_sha256(provenance),
                expected_counterfactual_sha256=common.canonical_sha256(extra),
            )

        signature_index = next(
            index
            for index, row in enumerate(provenance["rows"])
            if str(row["output_field"]).startswith("signature_")
        )

        def forge_self_consistent_wrong_df() -> dict[str, object]:
            forged_provenance = copy.deepcopy(provenance)
            row = forged_provenance["rows"][signature_index]
            wrong_count = 1 if int(row["seller_df"]) != 1 else 2
            row["seller_df"] = wrong_count
            row["seller_df_seller_count"] = wrong_count
            row["seller_df_seller_uids_sha256"] = "1" * 64
            forged_provenance["rows_sha256"] = common.canonical_sha256(
                forged_provenance["rows"]
            )
            return forged_provenance

        counterfactual_df_forge = forge_self_consistent_wrong_df()
        with self.assertRaises(quality.ScientificQualityAuditError) as caught:
            quality._profile_provenance_delta_receipt(
                world_uid=self.accepted.world_uid,
                original=provenance,
                counterfactual=counterfactual_df_forge,
                support_by_seller=support,
                original_items=self.accepted.redacted_items,
                counterfactual_items=self.accepted.redacted_items,
                expected_original_sha256=common.canonical_sha256(provenance),
                expected_counterfactual_sha256=common.canonical_sha256(
                    counterfactual_df_forge
                ),
            )
        self.assertNotIsInstance(caught.exception, quality.DatasetInvalidationError)

        original_df_forge = forge_self_consistent_wrong_df()
        with self.assertRaises(quality.DatasetInvalidationError):
            quality._profile_provenance_delta_receipt(
                world_uid=self.accepted.world_uid,
                original=original_df_forge,
                counterfactual=provenance,
                support_by_seller=support,
                original_items=self.accepted.redacted_items,
                counterfactual_items=self.accepted.redacted_items,
                expected_original_sha256=common.canonical_sha256(
                    original_df_forge
                ),
                expected_counterfactual_sha256=common.canonical_sha256(
                    provenance
                ),
            )

    def test_private_literal_inventory_covers_identity_variants_and_query_uid(self) -> None:
        literals = quality._private_leak_literals(self.accepted)
        asset = self.accepted.world["private"]["identity_assets"][0]
        value = str(asset["identity_value"])
        self.assertIn(value, literals)
        self.assertIn(value.casefold(), literals)
        self.assertIn(str(self.accepted.qrels[0]["query_uid"]), literals)

    def test_train_loader_includes_all_three_frozen_truth_tables(self) -> None:
        loaded = quality._load_persisted_split(
            self.context.output_root, "train", read_truth=True
        )
        self.assertIn("membership", loaded)
        self.assertIn("labels", loaded)
        self.assertIn("qrels", loaded)
        self.assertEqual(len(loaded["membership"]), 50 * 28)
        self.assertEqual(len(loaded["labels"]), 50 * 378)
        self.assertEqual(len(loaded["qrels"]), 50 * 28)

    def test_frozen_text_views_and_metadata_feature_order_close(self) -> None:
        profiles = [
            quality.dataset_builder._project_model_seller_profile(row)
            for row in self.accepted.seller_profiles
        ]
        endpoints = list(self.accepted.world["public"]["complete_model_pair_endpoints"])
        p_views, p_names, pairs, worlds = quality.build_text_views(
            profiles=profiles, endpoints=endpoints
        )
        items = [
            quality.dataset_builder._project_model_redacted_item(row)
            for row in self.accepted.redacted_items
        ]
        f_views, f_names, f_pairs, f_worlds = quality.build_fixed_support_text_views(
            items=items, endpoints=endpoints
        )
        numeric, numeric_names, n_pairs, n_worlds = (
            quality.build_production_numeric_matrix(
                profiles=profiles, endpoints=endpoints
            )
        )
        views = {**p_views, **f_views}
        names = {**p_names, **f_names}
        views["u_joint_full"] = np.column_stack(
            (p_views["p_full"], f_views["fs_full"], numeric)
        )
        names["u_joint_full"] = tuple(
            [f"p::{name}" for name in p_names["p_full"]]
            + [f"fs::{name}" for name in f_names["fs_full"]]
            + [f"numeric::{name}" for name in numeric_names]
        )
        self.assertTrue(np.array_equal(pairs, f_pairs))
        self.assertTrue(np.array_equal(pairs, n_pairs))
        self.assertTrue(np.array_equal(worlds, f_worlds))
        self.assertTrue(np.array_equal(worlds, n_worlds))
        for view_name, contract in self.audit_policy["text_counterfactual"]["views"].items():
            self.assertEqual(views[view_name].shape, (378, contract["expected_width"]))
            self.assertEqual(
                common.canonical_sha256(list(names[view_name])),
                contract["feature_names_canonical_sha256"],
            )
        matrix, metadata_names, _pairs, _worlds = quality.build_metadata_matrix(
            profiles=profiles,
            observed_items=[
                quality.dataset_builder._project_model_redacted_item(row)
                for row in self.accepted.redacted_items
            ],
            source_items=self.accepted.world["public"]["items"],
            endpoints=endpoints,
        )
        self.assertEqual(matrix.shape, (378, 67))
        self.assertEqual(
            metadata_names,
            tuple(self.audit_policy["metadata_probe"]["feature_names_in_order"]),
        )

    def test_fixed_support_is_order_and_item_uid_name_invariant(self) -> None:
        endpoints = list(self.accepted.world["public"]["complete_model_pair_endpoints"])
        original = [
            quality.dataset_builder._project_model_redacted_item(row)
            for row in self.accepted.redacted_items
        ]
        first, first_names, _, _ = quality.build_fixed_support_text_views(
            items=original, endpoints=endpoints
        )
        shuffled = list(reversed(copy.deepcopy(original)))
        second, second_names, _, _ = quality.build_fixed_support_text_views(
            items=shuffled, endpoints=endpoints
        )
        renamed = copy.deepcopy(original)
        for index, row in enumerate(renamed):
            row["item_uid"] = f"renamed_{len(renamed)-index:04d}"
        third, third_names, _, _ = quality.build_fixed_support_text_views(
            items=renamed, endpoints=endpoints
        )
        self.assertEqual(first_names, second_names)
        self.assertEqual(first_names, third_names)
        for name in first:
            self.assertEqual(first[name].astype("<f8").tobytes(), second[name].astype("<f8").tobytes())
            self.assertEqual(first[name].astype("<f8").tobytes(), third[name].astype("<f8").tobytes())

    def test_fixed_support_is_endpoint_symmetric_with_unequal_item_counts(self) -> None:
        items = [
            {
                "item_uid": "i0",
                "seller_uid": "s0",
                "world_uid": "w0",
                "title": "甲",
                "description": "",
            },
            {
                "item_uid": "i1",
                "seller_uid": "s0",
                "world_uid": "w0",
                "title": "甲二",
                "description": "描述",
            },
            {
                "item_uid": "i2",
                "seller_uid": "s1",
                "world_uid": "w0",
                "title": "乙",
                "description": "描述乙",
            },
        ]
        forward = [{"canonical_pair_uid": "p", "world_uid": "w0", "seller_uid_left": "s0", "seller_uid_right": "s1"}]
        reverse = [{"canonical_pair_uid": "p", "world_uid": "w0", "seller_uid_left": "s1", "seller_uid_right": "s0"}]
        first, names, _, _ = quality.build_fixed_support_text_views(
            items=items, endpoints=forward
        )
        second, reverse_names, _, _ = quality.build_fixed_support_text_views(
            items=items, endpoints=reverse
        )
        self.assertEqual(names, reverse_names)
        self.assertEqual(
            {name: matrix.shape[1] for name, matrix in first.items()},
            {"fs_full": 33, "fs_title": 14, "fs_template_surface": 30},
        )
        for name in first:
            self.assertEqual(
                first[name].astype("<f8").tobytes(),
                second[name].astype("<f8").tobytes(),
            )

    def test_fixed_support_slot_closure_canaries_fail(self) -> None:
        original = [
            quality.dataset_builder._project_model_redacted_item(row)
            for row in self.accepted.redacted_items
        ]
        changed = [
            quality.dataset_builder._project_model_redacted_item(row)
            for row in self.counterfactual["public"]["redacted_items"]
        ]
        receipt = quality._fixed_support_slot_contract(
            split=self.accepted.split,
            world_uid=self.accepted.world_uid,
            original_items=original, counterfactual_items=changed
        )
        self.assertEqual(receipt["slot_count"], 2 * len(original))
        with self.assertRaises(quality.ScientificQualityAuditError):
            quality._fixed_support_slot_contract(
                split=self.accepted.split,
                world_uid=self.accepted.world_uid,
                original_items=original, counterfactual_items=changed[:-1]
            )
        duplicate = copy.deepcopy(changed)
        duplicate[-1]["item_uid"] = duplicate[0]["item_uid"]
        duplicate[-1]["seller_uid"] = duplicate[0]["seller_uid"]
        duplicate[-1]["world_uid"] = duplicate[0]["world_uid"]
        with self.assertRaises(quality.ScientificQualityAuditError):
            quality._fixed_support_slot_contract(
                split=self.accepted.split,
                world_uid=self.accepted.world_uid,
                original_items=original, counterfactual_items=duplicate
            )
        moved = copy.deepcopy(changed)
        different_seller = next(
            row["seller_uid"]
            for row in moved
            if row["seller_uid"] != moved[0]["seller_uid"]
        )
        moved[0]["seller_uid"] = different_seller
        with self.assertRaises(quality.ScientificQualityAuditError):
            quality._fixed_support_slot_contract(
                split=self.accepted.split,
                world_uid=self.accepted.world_uid,
                original_items=original, counterfactual_items=moved
            )
        empty_drift = copy.deepcopy(changed)
        empty_drift[0]["title"] = "" if original[0]["title"] else "nonempty"
        with self.assertRaises(quality.ScientificQualityAuditError):
            quality._fixed_support_slot_contract(
                split=self.accepted.split,
                world_uid=self.accepted.world_uid,
                original_items=original, counterfactual_items=empty_drift
            )

    def test_audit_split_loader_does_not_open_any_private_payload(self) -> None:
        with (
            patch.object(quality, "_read_jsonl", return_value=[]) as read_jsonl,
            patch.object(quality, "_read_csv", return_value=[]) as read_csv,
        ):
            counters = quality._new_blind_boundary_counters()
            loaded = quality._load_persisted_split(
                self.context.output_root,
                "audit_a",
                read_truth=False,
                blind_counters=counters,
            )
        touched = [
            Path(call.args[0]).as_posix()
            for mocked in (read_jsonl, read_csv)
            for call in mocked.call_args_list
        ]
        self.assertFalse(loaded["truth_read"])
        self.assertNotIn("labels", loaded)
        self.assertNotIn("membership", loaded)
        self.assertTrue(touched)
        self.assertTrue(all("/observed/" in path for path in touched))
        self.assertTrue(all("/private/" not in path for path in touched))
        self.assertTrue(all(value == 0 for value in counters["audit_a"].values()))
        with self.assertRaises(quality.ScientificQualityAuditError):
            quality._load_persisted_split(
                self.context.output_root,
                "audit_a",
                read_truth=True,
                blind_counters=counters,
            )
        self.assertEqual(counters["audit_a"]["truth_read_requests"], 1)
        self.assertEqual(counters["audit_a"]["private_payload_open_requests"], 1)

    def test_audit_split_privileged_reconstruction_canary_fails(self) -> None:
        counters = quality._new_blind_boundary_counters()
        with self.assertRaises(quality.ScientificQualityAuditError):
            quality._reject_blind_privileged_replay("audit_b", counters)
        self.assertEqual(counters["audit_b"]["world_reconstruction_requests"], 1)
        self.assertEqual(counters["audit_b"]["build_private_truth_calls"], 1)
        self.assertEqual(counters["audit_b"]["qrels_generations"], 1)

    def test_input_tree_verifier_hashes_but_never_semantically_opens_audit_private_payloads(self) -> None:
        root = self.context.output_root
        root_manifest = common.load_json(root / "root_manifest.json")
        expected: dict[Path, tuple[str, int]] = {}
        for split in scientific.SPLITS:
            manifest = common.load_json(root / split / "split_manifest.json")
            for record in manifest["files"]:
                expected[(root / split / record["path"]).resolve()] = (
                    str(record["sha256"]),
                    int(record["row_count"]),
                )
        touched: list[Path] = []

        def fake_sha256(path: Path) -> str:
            resolved = Path(path).resolve()
            touched.append(resolved)
            return expected[resolved][0]

        def fake_row_count(path: Path) -> int:
            return expected[Path(path).resolve()][1]

        with (
            patch.object(common, "sha256_file", side_effect=fake_sha256),
            patch.object(
                quality.dataset_builder,
                "_count_file_rows",
                side_effect=fake_row_count,
            ),
            patch.object(quality, "_read_jsonl") as semantic_jsonl,
            patch.object(quality, "_read_csv") as semantic_csv,
        ):
            quality._verify_quality_input_tree(root, root_manifest)
        relative = [path.relative_to(root).as_posix() for path in touched]
        self.assertTrue(any(path.startswith("audit_a/observed/") for path in relative))
        self.assertTrue(
            any(
                path.startswith(("audit_a/private/", "audit_b/private/"))
                for path in relative
            )
        )
        semantic_jsonl.assert_not_called()
        semantic_csv.assert_not_called()

        audit_private = (
            root / "audit_a" / "private" / "controller_membership.jsonl"
        ).resolve()

        def mutated_sha256(path: Path) -> str:
            resolved = Path(path).resolve()
            if resolved == audit_private:
                return "0" * 64
            return expected[resolved][0]

        with (
            patch.object(common, "sha256_file", side_effect=mutated_sha256),
            patch.object(
                quality.dataset_builder,
                "_count_file_rows",
                side_effect=fake_row_count,
            ),
            patch.object(quality, "_read_jsonl") as semantic_jsonl,
            patch.object(quality, "_read_csv") as semantic_csv,
            self.assertRaises(quality.ScientificQualityAuditError),
        ):
            quality._verify_quality_input_tree(root, root_manifest)
        semantic_jsonl.assert_not_called()
        semantic_csv.assert_not_called()

    def test_blind_observed_audit_closes_without_world_reconstruction(self) -> None:
        data = quality._load_persisted_split(
            self.context.output_root, "audit_a", read_truth=False
        )
        record = next(
            row
            for row in self.context.world_records
            if row["split"] == "audit_a" and int(row["split_ordinal"]) == 0
        )
        world_uid = str(record["world_uid"])
        items = quality._group(data["items"], "world_uid")[world_uid]
        seller_uids = {str(row["seller_uid"]) for row in items}
        profiles = [
            row for row in data["profiles"] if str(row["seller_uid"]) in seller_uids
        ]
        endpoint_fields = tuple(
            self.context.effective_policy["relational_integrity"]
            ["pair_projection_contract"]["complete_model_pair_endpoints_schema"]
        )
        identity_fields = (
            "canonical_pair_uid",
            "world_uid",
            *tuple(self.context.effective_policy["history_features"]["feature_names"]),
        )
        with patch.object(
            quality.world_module,
            "build_scientific_world",
            side_effect=AssertionError("blind audit must not reconstruct a world"),
        ):
            scanned = quality._audit_blind_observed_world(
                policy=self.audit_policy,
                record=record,
                profiles=profiles,
                items=items,
                endpoints=quality._group(data["endpoints"], "world_uid")[world_uid],
                identity33=quality._group(data["identity33"], "world_uid")[world_uid],
                endpoint_fields=endpoint_fields,
                identity_fields=identity_fields,
                historical=self.historical,
                current_item_hashes=set(),
                current_seller_hashes=set(),
            )
        self.assertGreater(scanned, 0)
        missing_column = copy.deepcopy(
            quality._group(data["identity33"], "world_uid")[world_uid]
        )
        missing_column[0].pop(identity_fields[-1])
        with self.assertRaises(quality.ScientificQualityAuditError):
            quality._audit_blind_observed_world(
                policy=self.audit_policy,
                record=record,
                profiles=profiles,
                items=items,
                endpoints=quality._group(data["endpoints"], "world_uid")[world_uid],
                identity33=missing_column,
                endpoint_fields=endpoint_fields,
                identity_fields=identity_fields,
                historical=self.historical,
                current_item_hashes=set(),
                current_seller_hashes=set(),
            )

    def test_blind_observed_audit_rejects_self_consistent_profile_count_tamper(self) -> None:
        data = quality._load_persisted_split(
            self.context.output_root, "audit_a", read_truth=False
        )
        record = next(
            row
            for row in self.context.world_records
            if row["split"] == "audit_a" and int(row["split_ordinal"]) == 0
        )
        world_uid = str(record["world_uid"])
        items = quality._group(data["items"], "world_uid")[world_uid]
        seller_uids = {str(row["seller_uid"]) for row in items}
        profiles = [
            copy.deepcopy(row)
            for row in data["profiles"]
            if str(row["seller_uid"]) in seller_uids
        ]
        profiles[0]["item_count"] = int(profiles[0]["item_count"]) + 1
        endpoint_fields = tuple(
            self.context.effective_policy["relational_integrity"]
            ["pair_projection_contract"]["complete_model_pair_endpoints_schema"]
        )
        identity_fields = (
            "canonical_pair_uid",
            "world_uid",
            *tuple(self.context.effective_policy["history_features"]["feature_names"]),
        )
        with self.assertRaises(quality.DatasetInvalidationError):
            quality._audit_blind_observed_world(
                policy=self.audit_policy,
                record=record,
                profiles=profiles,
                items=items,
                endpoints=quality._group(data["endpoints"], "world_uid")[world_uid],
                identity33=quality._group(data["identity33"], "world_uid")[world_uid],
                endpoint_fields=endpoint_fields,
                identity_fields=identity_fields,
                historical=self.historical,
                current_item_hashes=set(),
                current_seller_hashes=set(),
            )

    def test_failure_receipt_is_small_bound_and_immutable(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "quality_v2"
            counters = quality._new_blind_boundary_counters()
            failure_root = quality._write_failure_receipt(
                policy=self.audit_policy,
                output_root=output,
                stage="unit_canary",
                error=quality.ScientificQualityAuditError("canary"),
                blind_counters=counters,
                partial_results={"completed": False},
                classification="AUDITOR_EXECUTION_FAILED_NO_DATASET_CONCLUSION",
                launch_evidence={"anchor": {}, "guard": {}},
            )
            receipt = common.load_json(failure_root / "decision_receipt.json")
            self.assertEqual(
                receipt["canonical_self_hash"], quality._canonical_self_hash(receipt)
            )
            self.assertEqual(receipt["failure_stage"], "unit_canary")
            self.assertTrue(receipt["input_dataset_retained_at_decision"])
            self.assertEqual(
                receipt["input_dataset_state_at_decision"],
                "PRESENT_AND_PRESERVED",
            )
            self.assertFalse(receipt["dataset_quality_conclusion_reached"])
            self.assertIn("quality_policy", receipt["evidence_binding"])
            self.assertNotIn("exception_message", receipt)
            self.assertEqual(
                receipt["exception_message_sha256"],
                common.sha256_bytes(b"canary"),
            )
            with self.assertRaises(quality.ScientificQualityAuditError):
                quality._write_failure_receipt(
                    policy=self.audit_policy,
                    output_root=output,
                    stage="overwrite",
                    error=quality.ScientificQualityAuditError("overwrite"),
                    blind_counters=counters,
                    partial_results={},
                    classification="AUDITOR_EXECUTION_FAILED_NO_DATASET_CONCLUSION",
                    launch_evidence={"anchor": {}, "guard": {}},
                )

    def test_dataset_invalidation_cleanup_deletes_only_exact_safe_design_root(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fake_root = Path(directory)
            dataset_root = (
                fake_root
                / "reports"
                / "step28_v13_v1_13_scientific_builder"
                / "design_preflight_unit"
            )
            dataset_root.mkdir(parents=True)
            manifest = dataset_root / "root_manifest.json"
            manifest.write_text('{"unit":true}\n', encoding="utf-8")
            failure_root = dataset_root.parent / "quality_unit_FAILED"
            failure_root.mkdir()
            decision = {
                "status": "DATASET_INVALIDATED",
                "cleanup_required": True,
                "canonical_self_hash": None,
            }
            decision["canonical_self_hash"] = quality._canonical_self_hash(decision)
            common.write_json(failure_root / "decision_receipt.json", decision)
            policy = copy.deepcopy(self.audit_policy)
            policy["input"]["dataset_root"] = "DATASET"
            policy["pins"]["dataset_root_manifest"] = {
                "path": "MANIFEST",
                "size_bytes": manifest.stat().st_size,
                "sha256": common.sha256_file(manifest),
            }

            def fake_repo_path(value: str) -> Path:
                if value == "DATASET":
                    return dataset_root
                if value == "MANIFEST":
                    return manifest
                raise AssertionError(value)

            with (
                patch.object(quality, "ROOT", fake_root),
                patch.object(quality.common, "repo_path", side_effect=fake_repo_path),
            ):
                receipt = quality._cleanup_invalidated_dataset(
                    policy=policy, failure_root=failure_root
                )
            self.assertTrue(receipt["input_dataset_deleted"])
            self.assertFalse(dataset_root.exists())
            self.assertTrue((failure_root / "cleanup_intent.json").is_file())
            self.assertTrue((failure_root / "cleanup_receipt.json").is_file())

    def test_quality_gate_and_model_fit_failures_have_different_classifications(self) -> None:
        self.assertEqual(
            quality._failure_classification(
                quality.DatasetInvalidationError("metadata passed=false")
            ),
            "DATASET_INVALIDATED",
        )
        self.assertEqual(
            quality._failure_classification(MemoryError("model fit")),
            "AUDITOR_EXECUTION_FAILED_NO_DATASET_CONCLUSION",
        )

    def test_dataset_invalidation_decision_does_not_claim_cleanup_early(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "quality_v2"
            failure_root = quality._write_failure_receipt(
                policy=self.audit_policy,
                output_root=output,
                stage="row_replay",
                error=quality.DatasetInvalidationError("row canary"),
                blind_counters=quality._new_blind_boundary_counters(),
                partial_results={},
                classification="DATASET_INVALIDATED",
                launch_evidence={"anchor": {}, "guard": {}},
            )
            receipt = common.load_json(failure_root / "decision_receipt.json")
            self.assertTrue(receipt["input_dataset_retained_at_decision"])
            self.assertEqual(
                receipt["input_dataset_state_at_decision"],
                "PRESENT_PENDING_CLEANUP",
            )
            self.assertTrue(receipt["cleanup_required"])
            self.assertNotIn("input_dataset_deleted", receipt)

    def test_dataset_invalidation_cleanup_reports_delete_failure_honestly(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fake_root = Path(directory)
            dataset_root = fake_root / "reports" / "step28_v13_v1_13_scientific_builder" / "design_preflight_unit"
            dataset_root.mkdir(parents=True)
            manifest = dataset_root / "root_manifest.json"
            manifest.write_text('{"unit":true}\n', encoding="utf-8")
            failure_root = dataset_root.parent / "quality_unit_FAILED"
            failure_root.mkdir()
            decision = {
                "status": "DATASET_INVALIDATED",
                "cleanup_required": True,
                "canonical_self_hash": None,
            }
            decision["canonical_self_hash"] = quality._canonical_self_hash(decision)
            common.write_json(failure_root / "decision_receipt.json", decision)
            policy = copy.deepcopy(self.audit_policy)
            policy["input"]["dataset_root"] = "DATASET"
            policy["pins"]["dataset_root_manifest"] = {"path": "MANIFEST", "size_bytes": manifest.stat().st_size, "sha256": common.sha256_file(manifest)}

            def fake_repo_path(value: str) -> Path:
                return dataset_root if value == "DATASET" else manifest

            with (
                patch.object(quality, "ROOT", fake_root),
                patch.object(quality.common, "repo_path", side_effect=fake_repo_path),
                patch.object(quality.shutil, "rmtree", side_effect=OSError("canary")),
            ):
                receipt = quality._cleanup_invalidated_dataset(
                    policy=policy, failure_root=failure_root
                )
            self.assertFalse(receipt["input_dataset_deleted"])
            self.assertEqual(receipt["deletion_error_type"], "OSError")
            self.assertTrue(dataset_root.exists())

    def test_cleanup_intent_survives_receipt_write_failure_and_recovers(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fake_root = Path(directory)
            dataset_root = (
                fake_root
                / "reports"
                / "step28_v13_v1_13_scientific_builder"
                / "design_preflight_unit"
            )
            dataset_root.mkdir(parents=True)
            manifest = dataset_root / "root_manifest.json"
            manifest.write_text('{"unit":true}\n', encoding="utf-8")
            failure_root = dataset_root.parent / "quality_unit_FAILED"
            failure_root.mkdir()
            decision = {
                "status": "DATASET_INVALIDATED",
                "cleanup_required": True,
                "canonical_self_hash": None,
            }
            decision["canonical_self_hash"] = quality._canonical_self_hash(decision)
            common.write_json(failure_root / "decision_receipt.json", decision)
            policy = copy.deepcopy(self.audit_policy)
            policy["input"]["dataset_root"] = "DATASET"
            policy["pins"]["dataset_root_manifest"] = {
                "path": "MANIFEST",
                "size_bytes": manifest.stat().st_size,
                "sha256": common.sha256_file(manifest),
            }

            def fake_repo_path(value: str) -> Path:
                return dataset_root if value == "DATASET" else manifest

            original_write_json = common.write_json

            def fail_completion(path: Path, value: object) -> None:
                if Path(path).name == "cleanup_receipt.json":
                    raise OSError("completion receipt canary")
                original_write_json(path, value)

            with (
                patch.object(quality, "ROOT", fake_root),
                patch.object(quality.common, "repo_path", side_effect=fake_repo_path),
                patch.object(quality.common, "write_json", side_effect=fail_completion),
                self.assertRaises(OSError),
            ):
                quality._cleanup_invalidated_dataset(
                    policy=policy, failure_root=failure_root
                )
            self.assertFalse(dataset_root.exists())
            self.assertTrue((failure_root / "cleanup_intent.json").is_file())
            self.assertFalse((failure_root / "cleanup_receipt.json").exists())
            with (
                patch.object(quality, "ROOT", fake_root),
                patch.object(quality.common, "repo_path", side_effect=fake_repo_path),
            ):
                receipt = quality._cleanup_invalidated_dataset(
                    policy=policy, failure_root=failure_root
                )
            self.assertTrue(receipt["input_dataset_deleted"])
            self.assertTrue((failure_root / "cleanup_receipt.json").is_file())


if __name__ == "__main__":
    unittest.main()
