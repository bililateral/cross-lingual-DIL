#!/usr/bin/env python3
"""Validate the mandatory v1.12 authorization overlay before seed access."""

from __future__ import annotations

import math
import os
import re
import subprocess
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import step28_v13_v1_12_formal_common as formal
import step28_v13_v1_12_preceremony as preceremony


ROOT = Path(__file__).resolve().parents[1]
RUN_ID = "v13_training_ready_v1_12_cleanroom_20260803"
DESIGN_EVIDENCE_ROLE = (
    "LEGACY_COMPATIBILITY_PREREQUISITES_NOT_SUFFICIENT_FOR_SEED_AUTHORIZATION"
)
AUTHORIZATION_EVIDENCE_ROLE = "MANDATORY_CURRENT_SEED_AUTHORIZATION_EVIDENCE"
DESIGN_ROOT = (
    ROOT
    / "reports"
    / "step28_synthetic_chinese_dataset"
    / "design_preflights"
    / "v1_12_cleanroom_20260803"
)
OVERLAY_PATH = (
    ROOT
    / "docs"
    / "STEP28_V13_V1_12_FORMAL_AUTHORIZATION_OVERLAY_20260809.zh.md"
)
TEXT_RECEIPT_PATH = DESIGN_ROOT / "text_shortcut_preflight_receipt.json"
INTERRUPTION_RECEIPT_PATH = DESIGN_ROOT / "external_interruption_receipt.json"
WAIVER_RECEIPT_PATH = DESIGN_ROOT / "historical_manifest_waiver_receipt.json"

OVERLAY_PIN = {
    "path": "docs/STEP28_V13_V1_12_FORMAL_AUTHORIZATION_OVERLAY_20260809.zh.md",
    "size_bytes": 6167,
    "sha256": "a45a0488b91b5819bbf47090cd70a89674fe0dfdb15452a0309b65b73790245d",
}
TEXT_RECEIPT_PIN = {
    "path": (
        "reports/step28_synthetic_chinese_dataset/design_preflights/"
        "v1_12_cleanroom_20260803/text_shortcut_preflight_receipt.json"
    ),
    "size_bytes": 105744,
    "sha256": "e371a460d65560f06e071a6961bf67c22d1f39a5e5e3a37431af5b1c88dd152e",
    "canonical_self_hash": (
        "ffe86c9ad07f77609c3463e831108e73504493b591c73815359c0419e58c4a20"
    ),
}
INTERRUPTION_RECEIPT_PIN = {
    "path": (
        "reports/step28_synthetic_chinese_dataset/design_preflights/"
        "v1_12_cleanroom_20260803/external_interruption_receipt.json"
    ),
    "size_bytes": 5220,
    "sha256": "be6e78c84de6d242f7812656f7dd7e2d9cb43ed472d061fe12d5f237c7ac7587",
    "canonical_self_hash": (
        "503291c8749de2a4b8e2668f45dad6de7e0607c760fedebb0d7ae0fed5d396a5"
    ),
}
WAIVER_RECEIPT_PIN = {
    "path": (
        "reports/step28_synthetic_chinese_dataset/design_preflights/"
        "v1_12_cleanroom_20260803/historical_manifest_waiver_receipt.json"
    ),
    "size_bytes": 3338,
    "sha256": "d753d1e647b071320f7f8fcecb0c9e5d9144aca7add66dc4df3b9dc0e015e513",
    "canonical_self_hash": (
        "56da4c9174c348a5f8f2b9e0f0adbc030271d107d3ea1456d235b5f72a9ec05f"
    ),
}
TEXT_SOURCE_CLOSURE_SHA256 = (
    "75c3e940a873a2674300337d0d3f1042fa41626e7e0aa2e97f640744175da2a3"
)
WAIVED_TEST_ID = (
    "test_step28_v12_application_contracts.Step28V12ApplicationContracts."
    "test_sync_manifest_is_closed_and_hashes_match"
)
EXPECTED_STARTED_SKIPPED = [
    {
        "id": (
            "test_step28_v11_application_contracts.Step28V11ApplicationContracts."
            "test_blind_packet_has_no_model_outputs_or_pair_uid"
        ),
        "reason": "v11 was withdrawn after final audit; v12 owns the current contracts",
    },
    {
        "id": (
            "test_step28_v11_application_contracts.Step28V11ApplicationContracts."
            "test_current_outcome_is_zero_queue_abstention"
        ),
        "reason": "v11 was withdrawn after final audit; v12 owns the current contracts",
    },
    {
        "id": (
            "test_step28_v11_application_contracts.Step28V11ApplicationContracts."
            "test_observable_state_support_recomputes_from_train_and_development"
        ),
        "reason": "v11 was withdrawn after final audit; v12 owns the current contracts",
    },
    {
        "id": (
            "test_step28_v11_application_contracts.Step28V11ApplicationContracts."
            "test_reviewed_registry_is_uid_only_and_excluded_everywhere"
        ),
        "reason": "v11 was withdrawn after final audit; v12 owns the current contracts",
    },
    {
        "id": (
            "test_step28_v11_application_contracts.Step28V11ApplicationContracts."
            "test_sync_manifest_is_closed_and_hashes_match"
        ),
        "reason": "v11 was withdrawn after final audit; v12 owns the current contracts",
    },
    {
        "id": (
            "test_step28_v11_application_contracts.Step28V11ApplicationContracts."
            "test_threshold_selection_excludes_train_ambiguous_states"
        ),
        "reason": "v11 was withdrawn after final audit; v12 owns the current contracts",
    },
]
EXPECTED_FIXTURE_SKIPPED = [
    {
        "id": (
            "setUpClass (test_step28_v13_metadata_shortcut_audit_contracts."
            "Step28V13MetadataShortcutContracts)"
        ),
        "reason": (
            "The immutable, execution-blocked metadata-shortcut lock belongs to the "
            "superseded dataset_smoke_v3 parent contract and policy. Current "
            "training_ready shortcut execution is covered by "
            "test_step28_v13_training_ready_builder_contracts.py."
        ),
    }
]
FALSE_AUTHORIZATIONS = {
    "formal_seed_ceremony": False,
    "formal_train_generation": False,
    "formal_development_generation": False,
    "formal_audit_a_generation": False,
    "formal_audit_b_generation": False,
    "model_training": False,
    "audit_truth_unsealing": False,
}
HEX_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
GIT_OID_RE = re.compile(r"^(?:[0-9a-f]{40}|[0-9a-f]{64})$")
TEXT_TOP_LEVEL_KEYS = {
    "version",
    "status",
    "design_only",
    "run_id",
    "stage_statuses",
    "assignment_null",
    "counterfactual_fast_full_parity",
    "original_fast_full_parity",
    "counterfactual_visible_text_hard_gate",
    "original_visible_text_descriptive_only",
    "rowwise_design_audit",
    "source_files",
    "formal_authorizations_after_preflight",
    "formal_seed_or_key_access",
    "formal_dataset_rows_produced",
    "formal_dataset_rows_audited",
    "raw_design_worlds_or_matrices_persisted",
    "model_training_authorized",
    "audit_truth_unsealed",
    "claim_boundary",
    "canonical_self_hash",
}
TEXT_SOURCE_KEYS = {
    "assignment_null",
    "contract",
    "counterfactual_text",
    "exact_logistic",
    "formal_build_draft",
    "formal_common",
    "history_common",
    "history_features",
    "identity_plan",
    "identity_values",
    "nonidentity",
    "policy",
    "preceremony",
    "preflight",
    "production_chain",
    "profiles",
    "runner",
    "step28_common",
    "structure",
    "style_derangement",
    "text_renderer",
    "v13_common",
    "world_builder",
}
PRELOCK_KEYS = {
    "version",
    "status",
    "run_id",
    "authorizations",
    "formal_build_draft",
    "design_evidence_role",
    "design_evidence",
    "authorization_evidence_role",
    "authorization_overlay_contract",
    "authorization_evidence",
    "source_closure",
    "dependency_versions",
    "custody",
    "git_head",
    "formal_master_or_capability_created",
    "canonical_self_hash",
}
STRUCTURED_RESULT_KEYS = {
    "version",
    "tests_run",
    "started_test_ids",
    "success_ids",
    "failure_ids",
    "error_ids",
    "skipped",
    "fixture_skipped",
    "expected_failure_ids",
    "unexpected_success_ids",
    "failed_subtest_ids",
    "skipped_subtest_ids",
    "was_successful",
    "wall_seconds",
}
FULL_TEST_RECEIPT_KEYS = {
    "version",
    "status",
    "status_semantics",
    "count_semantics",
    "command",
    "git_head",
    "source_closure_canonical_sha256",
    "source_closure_member_count",
    "test_suite_source_closure",
    "test_count",
    "passed_count",
    "skipped_count",
    "failed_count",
    "error_count",
    "raw_test_count",
    "raw_passed_count",
    "raw_skipped_count",
    "raw_fixture_skipped_count",
    "raw_failed_count",
    "raw_error_count",
    "raw_failure_ids",
    "raw_error_ids",
    "raw_skipped_ids",
    "raw_fixture_skipped_ids",
    "raw_expected_failure_ids",
    "raw_unexpected_success_ids",
    "raw_failed_subtest_ids",
    "raw_skipped_subtest_ids",
    "accepted_waived_failure_count",
    "accepted_waived_failure_ids",
    "historical_manifest_waiver",
    "current_v1_12_targeted_tests",
    "subprocess_return_code",
    "raw_subprocess_return_code",
    "test_runner_reported_seconds",
    "wall_seconds",
    "raw_structured_result",
    "structured_result_sha256",
    "captured_output_sha256",
    "runtime_versions",
    "warning_policy",
    "formal_seed_or_key_access",
    "formal_rows_produced",
    "canonical_self_hash",
}


class AuthorizationEvidenceError(ValueError):
    """Raised before any irreversible seed-ceremony side effect."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AuthorizationEvidenceError(message)


def _exact_keys(value: Any, expected: set[str], *, label: str) -> Mapping[str, Any]:
    _require(isinstance(value, Mapping), f"{label} must be an object")
    _require(set(value) == expected, f"{label} keyset drift")
    return value


def _verify_exact_pin(spec: Mapping[str, Any], expected: Mapping[str, Any], *, label: str) -> Path:
    _require(dict(spec) == dict(expected), f"{label} pin drift")
    return preceremony.verify_file_pin(spec, label=label)


def _load_exact_self_hashed_receipt(
    spec: Mapping[str, Any], expected: Mapping[str, Any], *, label: str
) -> dict[str, Any]:
    path = _verify_exact_pin(spec, expected, label=label)
    document = preceremony.load_json_strict(path)
    preceremony.validate_canonical_self_hash(document, label=label)
    _require(
        document.get("canonical_self_hash") == expected["canonical_self_hash"],
        f"{label} canonical self-hash drift",
    )
    return document


def load_current_authorization_receipts_exact(
    *, dereference_waiver_state: bool
) -> dict[str, dict[str, Any]]:
    """Load the three immutable receipts before any expensive test/run."""

    text_receipt = _load_exact_self_hashed_receipt(
        TEXT_RECEIPT_PIN,
        TEXT_RECEIPT_PIN,
        label="final text shortcut receipt",
    )
    interruption = _load_exact_self_hashed_receipt(
        INTERRUPTION_RECEIPT_PIN,
        INTERRUPTION_RECEIPT_PIN,
        label="external interruption receipt",
    )
    waiver = _load_exact_self_hashed_receipt(
        WAIVER_RECEIPT_PIN,
        WAIVER_RECEIPT_PIN,
        label="historical manifest waiver receipt",
    )
    validate_text_receipt(text_receipt)
    validate_interruption_receipt(interruption)
    validate_waiver_receipt(
        waiver, dereference_current_state=dereference_waiver_state
    )
    return {
        "text_receipt": text_receipt,
        "interruption_receipt": interruption,
        "waiver_receipt": waiver,
    }


def test_suite_source_closure() -> dict[str, Any]:
    """Hash every Python file discoverable below tests/, independent of formal source."""

    tests_root = (ROOT / "tests").resolve()
    paths = sorted(
        (
            path
            for path in tests_root.rglob("*.py")
            if path.is_file()
        ),
        key=lambda path: path.relative_to(ROOT).as_posix().encode("utf-8"),
    )
    _require(bool(paths), "test-suite source tree is empty")
    members: list[dict[str, Any]] = []
    for path in paths:
        resolved = path.resolve()
        _require(not path.is_symlink(), "test-suite source may not be a symlink")
        try:
            relative = resolved.relative_to(ROOT.resolve()).as_posix()
        except ValueError as exc:
            raise AuthorizationEvidenceError(
                "test-suite source escapes repository"
            ) from exc
        members.append(
            {
                "path": relative,
                "size_bytes": preceremony.stat_long_path(resolved).st_size,
                "sha256": preceremony.sha256_file(resolved),
            }
        )
    return {
        "member_count": len(members),
        "canonical_sha256": preceremony.canonical_sha256(members),
        "members": members,
    }


def _git_run(arguments: list[str]) -> str:
    environment = {**os.environ, "GIT_OPTIONAL_LOCKS": "0"}
    return subprocess.run(
        ["git", *arguments],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
        env=environment,
    ).stdout


def current_git_head() -> str:
    value = _git_run(["rev-parse", "HEAD"]).strip()
    _require(GIT_OID_RE.fullmatch(value) is not None, "Git HEAD is malformed")
    return value


def require_tracked_worktree_matches_head() -> None:
    """Require HEAD, index, and worktree tracked bytes to match separately."""

    environment = {**os.environ, "GIT_OPTIONAL_LOCKS": "0"}
    checks = (
        (["git", "diff", "--quiet", "--"], "worktree differs from index"),
        (
            ["git", "diff", "--cached", "--quiet", "HEAD", "--"],
            "index differs from Git HEAD",
        ),
    )
    for command, failure in checks:
        result = subprocess.run(
            command,
            cwd=ROOT,
            check=False,
            env=environment,
        )
        if result.returncode == 1:
            raise AuthorizationEvidenceError(
                f"tracked repository {failure}"
            )
        if result.returncode != 0:
            raise AuthorizationEvidenceError(
                "unable to verify tracked repository state: "
                f"{result.returncode}"
            )


def require_paths_git_tracked(paths: set[str], *, label: str) -> None:
    _require(bool(paths), f"{label} path set is empty")
    tracked = {
        line.strip().replace("\\", "/")
        for line in _git_run(
            ["ls-files", "--", *sorted(paths, key=lambda item: item.encode("utf-8"))]
        ).splitlines()
        if line.strip()
    }
    _require(tracked == paths, f"every {label} member must be Git-tracked")


def require_committed_clean_test_tree(
    closure: Mapping[str, Any] | None = None,
) -> None:
    """Reject dirty, untracked, ignored, or missing discoverable test sources."""

    observed = test_suite_source_closure() if closure is None else closure
    paths = {str(row["path"]) for row in observed["members"]}
    status = _git_run(["status", "--porcelain=v1", "--", "tests"])
    _require(not status.strip(), "test-suite source tree must be clean")
    tracked = {
        line.strip().replace("\\", "/")
        for line in _git_run(["ls-files", "--", "tests"]).splitlines()
        if line.strip().endswith(".py")
    }
    _require(paths == tracked, "discoverable test-suite sources must all be tracked")


def _validate_test_suite_closure_record(value: Any) -> Mapping[str, Any]:
    closure = _exact_keys(
        value,
        {"member_count", "canonical_sha256", "members"},
        label="test-suite source closure",
    )
    members = closure["members"]
    _require(isinstance(members, list), "test-suite members must be a list")
    paths: list[str] = []
    for index, member in enumerate(members):
        pin = _exact_keys(
            member,
            {"path", "size_bytes", "sha256"},
            label=f"test-suite source pin {index}",
        )
        path = str(pin["path"])
        _require(path.startswith("tests/") and path.endswith(".py"), "test-suite path boundary drift")
        _require(type(pin["size_bytes"]) is int and pin["size_bytes"] >= 0, "test-suite size drift")
        _require(HEX_SHA256_RE.fullmatch(str(pin["sha256"])) is not None, "test-suite hash drift")
        paths.append(path)
    _require(
        paths == sorted(paths, key=lambda item: item.encode("utf-8"))
        and len(paths) == len(set(paths))
        and int(closure["member_count"]) == len(members)
        and preceremony.canonical_sha256(members) == closure["canonical_sha256"],
        "test-suite source closure registry drift",
    )
    return closure


def _validate_closed_authorizations(document: Mapping[str, Any], field: str) -> None:
    _require(document.get(field) == FALSE_AUTHORIZATIONS, f"{field} drift")
    _require(
        all(type(value) is bool for value in document[field].values()),
        f"{field} contains a non-boolean",
    )
    _require(document.get("formal_seed_or_key_access") is False, "formal seed access drift")
    _require(
        type(document.get("formal_dataset_rows_produced")) is int
        and document["formal_dataset_rows_produced"] == 0,
        "formal row production drift",
    )
    if "formal_dataset_rows_audited" in document:
        _require(
            type(document.get("formal_dataset_rows_audited")) is int
            and document["formal_dataset_rows_audited"] == 0,
            "formal row audit drift",
        )
    _require(document.get("model_training_authorized") is False, "model training boundary drift")


def validate_text_receipt(document: Mapping[str, Any]) -> None:
    """Replay the final text audit's immutable source and scientific boundary."""

    _exact_keys(document, TEXT_TOP_LEVEL_KEYS, label="final text receipt")
    _require(
        document.get("version")
        == "2026-08-08-step28-v13-v1-12-text-shortcut-preflight-v1"
        and document.get("status")
        == "PASS_DESIGN_TEXT_SHORTCUT_PREFLIGHT_NO_FORMAL_AUTHORIZATION"
        and document.get("design_only") is True
        and document.get("run_id") == RUN_ID,
        "final text receipt identity/status drift",
    )
    _validate_closed_authorizations(document, "formal_authorizations_after_preflight")
    _require(document.get("audit_truth_unsealed") is False, "audit truth boundary drift")
    _require(document.get("raw_design_worlds_or_matrices_persisted") is False, "raw design persistence drift")

    expected_stages = {
        "assignment_null": "PASS_ASSIGNMENT_NULL_GATES",
        "counterfactual_fast_full_parity": "PASS_TEXT_FAST_FULL_REDACTED_PARITY",
        "original_fast_full_parity": "PASS_ORIGINAL_FAST_FULL_REDACTED_PARITY",
        "original_text_isolation": "PASS_DESIGN_ORIGINAL_TEXT_ISOLATION_DESCRIPTIVE_ONLY",
        "rowwise_design_audit": "PASS_DESIGN_WORLDS_ROW_BY_ROW_RECOMPUTED_IN_MEMORY",
        "visible_text_shortcut": "PASS_VISIBLE_TEXT_SHORTCUT_GATES",
    }
    _require(document.get("stage_statuses") == expected_stages, "text audit stage status drift")

    sources = _exact_keys(document.get("source_files"), TEXT_SOURCE_KEYS, label="text source closure")
    _require(preceremony.canonical_sha256(sources) == TEXT_SOURCE_CLOSURE_SHA256, "text source closure hash drift")
    for name in sorted(sources):
        pin = sources[name]
        expected_pin_keys = {"path", "size_bytes", "sha256"}
        if name in {"policy", "formal_build_draft"}:
            expected_pin_keys.add("canonical_self_hash")
        _exact_keys(pin, expected_pin_keys, label=f"text source pin {name}")
        path = preceremony.verify_file_pin(pin, label=f"text source {name}")
        if "canonical_self_hash" in pin:
            nested = preceremony.load_json_strict(path)
            preceremony.validate_canonical_self_hash(nested, label=f"text source {name}")
            _require(nested["canonical_self_hash"] == pin["canonical_self_hash"], f"text source self-hash drift: {name}")

    assignment = document["assignment_null"]
    for split in ("train", "development"):
        description = assignment[f"{split}_description"]
        _require(int(description.get("world_count", -1)) == 500, f"assignment {split} world count drift")
        _require(int(description.get("pair_count", -1)) == 186000, f"assignment {split} pair count drift")
    hard_assignment = assignment["development_hard_gate"]
    expected_assignment_gates = {
        "development_bootstrap_95_upper_family_max_symmetric_auc": True,
        "development_maximum_direct_symmetric_auc": True,
    }
    _require(
        hard_assignment.get("status") == "PASS_ASSIGNMENT_NULL_GATES"
        and int(hard_assignment.get("bootstrap_replicates", -1)) == 9999
        and hard_assignment.get("hard_gates") == expected_assignment_gates
        and all(
            type(value) is bool
            for value in hard_assignment["hard_gates"].values()
        )
        and hard_assignment.get("bootstrap_draw_sha256")
        == "dfad1a38ab9ead6befc977af24e545d49e1de8e49cf4f633c12c21f3cfc059dd"
        and float(hard_assignment.get("point_maximum_direct_symmetric_auc", 1.0)) <= 0.52
        and float(hard_assignment.get("bootstrap_95_upper_family_max_symmetric_auc", 1.0)) <= 0.53,
        "assignment null gate replay failed",
    )
    _require(
        assignment["train_description"].get("classifier_fitted") is False
        and assignment["development_description"].get("classifier_fitted") is False,
        "assignment classifier boundary drift",
    )

    visible = document["counterfactual_visible_text_hard_gate"]
    expected_visible_gates = {
        "development_bootstrap_family_max_average_precision_uplift": True,
        "development_bootstrap_family_max_symmetric_auc": True,
        "development_family_model_average_precision_uplift": True,
        "development_family_model_symmetric_auc": True,
        "development_single_feature_symmetric_auc": True,
    }
    _require(
        visible.get("status") == "PASS_VISIBLE_TEXT_SHORTCUT_GATES"
        and int(visible.get("bootstrap_replicates", -1)) == 9999
        and visible.get("models_refit_inside_bootstrap") is False
        and visible.get("development_only_used_for_hard_gates") is True
        and visible.get("hard_gates") == expected_visible_gates
        and all(type(value) is bool for value in visible["hard_gates"].values())
        and visible.get("bootstrap_draw_sha256")
        == "755f6dfb7a5c1d12842e9702b7a3949ebcd0339fa10f716e3fa0dee7e7a49a51"
        and float(visible.get("point_maximum_single_feature_symmetric_auc", 1.0)) <= 0.52
        and float(visible.get("point_maximum_family_model_symmetric_auc", 1.0)) <= 0.53
        and float(visible.get("point_maximum_family_model_average_precision_uplift", 1.0)) <= 0.01
        and float(visible.get("bootstrap_95_upper_family_max_symmetric_auc", 1.0)) <= 0.53
        and float(visible.get("bootstrap_95_upper_family_max_average_precision_uplift", 1.0)) <= 0.015,
        "visible text gate replay failed",
    )

    expected_intersections = {
        "world_uid": 0,
        "seller_uid": 0,
        "item_uid": 0,
        "canonical_pair_uid": 0,
        "controller_uid": 0,
        "item_document_hash": 0,
        "seller_document_hash": 0,
        "seller_five_field_record_hash": 0,
    }
    original = document["original_visible_text_descriptive_only"]
    _require(original.get("cross_split_exact_intersection_counts") == expected_intersections, "cross-split intersection drift")
    _require(original.get("visible_forbidden_residue_counts") == {"development": 0, "train": 0}, "visible forbidden residue drift")
    _require(original.get("formal_gate") is False, "original branch formal-gate drift")
    for split in ("train", "development"):
        row = original[split]
        strata = row["registered_mechanism_strata"]
        _require(
            int(row.get("pair_count", -1)) == 189000
            and int(row.get("positive_count", -1)) == 10000
            and row.get("threshold_or_model_fitted") is False
            and {name: int(value["row_count"]) for name, value in strata.items()}
            == {
                "exact_title_clone_target": 1000,
                "high_semantic_similarity_target": 2000,
                "other_negative": 176000,
                "positive": 10000,
            },
            f"original {split} count drift",
        )
    rowwise = document["rowwise_design_audit"]
    _require(rowwise.get("status") == "PASS_DESIGN_WORLDS_ROW_BY_ROW_RECOMPUTED_IN_MEMORY", "rowwise audit drift")
    for split in ("train", "development"):
        row = rowwise["splits"][split]
        _require(
            int(row.get("world_count", -1)) == 500
            and int(row.get("original_pair_rows_recomputed", -1)) == 189000
            and int(row.get("counterfactual_pair_rows_recomputed_after_neutral_mask", -1)) == 186000
            and int(row.get("counterfactual_positive_rows", -1)) == 10000
            and int(row.get("original_seller_profile_count", -1)) == 14000
            and row.get("raw_rows_persisted") is False,
            f"rowwise {split} count drift",
        )
        _require(
            HEX_SHA256_RE.fullmatch(str(row.get("original_world_audit_sha256", "")))
            is not None
            and HEX_SHA256_RE.fullmatch(
                str(row.get("counterfactual_world_audit_sha256", ""))
            )
            is not None,
            f"rowwise {split} audit hash drift",
        )
    for field, status, pair_count in (
        ("counterfactual_fast_full_parity", "PASS_TEXT_FAST_FULL_REDACTED_PARITY", 372),
        ("original_fast_full_parity", "PASS_ORIGINAL_FAST_FULL_REDACTED_PARITY", 378),
    ):
        parity = document[field]
        _require(
            parity.get("status") == status
            and int(parity.get("world_count", -1)) == 2
            and [row.get("split") for row in parity.get("rows", [])] == ["train", "development"]
            and all(int(row.get("pair_count", -1)) == pair_count for row in parity.get("rows", [])),
            f"{field} drift",
        )


def validate_interruption_receipt(document: Mapping[str, Any]) -> None:
    expected_keys = {
        "version", "status", "design_only", "run_id", "original_process_start_local",
        "interruption_detected_date_local", "original_target_receipt_path",
        "original_target_receipt_existed_after_abort", "interruption_classification",
        "scientific_or_registered_gate_failure_observed", "last_entered_stage",
        "last_observed_progress", "registered_gate_control_flow_observation",
        "temporary_log_evidence", "residual_process_check", "persisted_scientific_payload_check",
        "frozen_equivalence_evidence", "runtime_environment", "replacement_policy",
        "if_replacement_is_interrupted_again", "formal_authorizations_after_interruption",
        "formal_seed_or_key_access", "formal_dataset_rows_produced", "formal_dataset_rows_audited",
        "model_training_authorized", "audit_truth_unsealed", "canonical_self_hash",
    }
    _exact_keys(document, expected_keys, label="external interruption receipt")
    _require(
        document.get("version") == "2026-08-09-step28-v13-v1-12-external-interruption-v1"
        and document.get("status") == "INTERRUPTED_EXTERNAL_INFRASTRUCTURE_NO_SCIENTIFIC_FAILURE"
        and document.get("design_only") is True
        and document.get("run_id") == RUN_ID
        and document.get("scientific_or_registered_gate_failure_observed") is False
        and document.get("original_target_receipt_existed_after_abort") is False
        and document.get("residual_process_check") == {"python_process_remaining": False},
        "external interruption classification drift",
    )
    _validate_closed_authorizations(document, "formal_authorizations_after_interruption")
    _require(document.get("audit_truth_unsealed") is False, "interruption audit truth drift")
    persisted = document["persisted_scientific_payload_check"]
    _require(
        persisted == {"raw_worlds": 0, "matrices": 0, "checkpoints": 0, "final_receipts": 0, "only_temporary_logs": True},
        "interrupted payload boundary drift",
    )
    policy = document["replacement_policy"]
    _require(
        policy.get("exact_replacement_authorized") is True
        and int(policy.get("maximum_replacement_reruns", -1)) == 1
        and policy.get("replacement_requires_full_restart_from_stage_zero") is True
        and policy.get("checkpoint_resume_forbidden") is True
        and policy.get("prior_intermediate_artifact_reuse_forbidden") is True
        and policy.get("same_run_id_required") is True
        and policy.get("same_final_output_path_required") is True
        and policy.get("authorization_consumed_when_replacement_process_starts") is True
        and policy.get("second_replacement_forbidden") is True,
        "exact replacement discipline drift",
    )
    _require(
        document.get("if_replacement_is_interrupted_again")
        == {"no_third_attempt_under_v1_12": True, "require_new_version_and_refreeze_before_any_further_execution": True},
        "no-third-attempt boundary drift",
    )
    frozen = document["frozen_equivalence_evidence"]
    _require(frozen.get("source_closure_canonical_sha256") == TEXT_SOURCE_CLOSURE_SHA256, "interruption source closure drift")
    _require(
        frozen.get("historical_manifest_waiver_receipt")
        == {key: WAIVER_RECEIPT_PIN[key] for key in ("size_bytes", "sha256", "canonical_self_hash")},
        "interruption waiver pin drift",
    )


def _git_clean_for(relative_path: str) -> bool:
    return not _git_run(
        ["status", "--porcelain=v1", "--", relative_path]
    ).strip()


def _last_commit(relative_path: str) -> str:
    return _git_run(
        ["log", "-1", "--format=%H", "--", relative_path]
    ).strip()


def validate_waiver_receipt(document: Mapping[str, Any], *, dereference_current_state: bool) -> None:
    expected_keys = {
        "version", "status", "design_only", "waiver_scope", "full_suite_observation",
        "current_step28_v13_v1_12_targeted_tests", "historical_manifest",
        "historical_expected_record", "current_committed_document", "historical_test_current_bytes",
        "classification", "waiver_conditions", "current_freeze", "formal_authorizations_after_waiver",
        "formal_seed_or_key_access", "formal_dataset_rows_produced", "formal_dataset_rows_audited",
        "model_training_authorized", "canonical_self_hash",
    }
    _exact_keys(document, expected_keys, label="historical manifest waiver")
    _require(
        document.get("version") == "2026-08-08-step28-v13-v1-12-historical-manifest-waiver-v1"
        and document.get("status") == "PASS_EXACT_HISTORICAL_MANIFEST_FAILURE_WAIVER_NO_FORMAL_AUTHORIZATION"
        and document.get("design_only") is True
        and document.get("waiver_scope")
        == {"exact_test": WAIVED_TEST_ID, "allowed_failure_count": 1, "additional_failure_invalidates_waiver": True},
        "historical waiver scope drift",
    )
    _validate_closed_authorizations(document, "formal_authorizations_after_waiver")
    observed = document["full_suite_observation"]
    targeted = document["current_step28_v13_v1_12_targeted_tests"]
    _require(
        observed.get("tests_run") == 464
        and observed.get("passed") == 456
        and observed.get("skipped") == 7
        and observed.get("failed") == 1
        and observed.get("sole_failure_matches_waiver_scope") is True
        and targeted.get("tests_run") == targeted.get("passed") == 39
        and targeted.get("skipped") == targeted.get("failed") == 0,
        "historical waiver observation drift",
    )
    _require(
        document.get("current_freeze")
        == {
            "text_shortcut_contract_sha256": "7e17220f7dea073640ad810a48070a8321bd75bba1c95c7f8018df69780af126",
            "text_shortcut_policy_canonical_self_hash": "21e918e3106d817fa866594188e389815f899066671f59318e38283183c5dd9b",
        },
        "waiver current freeze drift",
    )
    if not dereference_current_state:
        return
    manifest_pin = document["historical_manifest"]
    manifest_path = preceremony.verify_file_pin(manifest_pin, label="waived historical manifest")
    manifest = preceremony.load_json_strict(manifest_path)
    expected_record = document["historical_expected_record"]
    matching = [row for row in manifest.get("artifacts", []) if row.get("path") == expected_record.get("path")]
    _require(matching == [expected_record], "historical manifest expected record drift")
    current = document["current_committed_document"]
    current_path = preceremony.verify_file_pin(current, label="current PROJECT_PROGRESS")
    current_relative = current_path.relative_to(ROOT).as_posix()
    _require(current.get("worktree_clean_for_path") is True and _git_clean_for(current_relative), "current PROJECT_PROGRESS is dirty")
    _require(_last_commit(current_relative) == current.get("last_commit"), "current PROJECT_PROGRESS commit drift")
    historical_test = document["historical_test_current_bytes"]
    test_path = preceremony.verify_file_pin(historical_test, label="waived historical test")
    test_relative = test_path.relative_to(ROOT).as_posix()
    _require(historical_test.get("worktree_clean_for_path") is True and _git_clean_for(test_relative), "waived historical test is dirty")


def validate_full_test_receipt(tests: Mapping[str, Any], waiver: Mapping[str, Any]) -> None:
    _exact_keys(tests, FULL_TEST_RECEIPT_KEYS, label="full-test receipt")
    structured = _exact_keys(
        tests.get("raw_structured_result"),
        STRUCTURED_RESULT_KEYS,
        label="raw structured unittest result",
    )

    def strict_ids(field: str) -> list[str]:
        values = structured.get(field)
        _require(
            isinstance(values, list)
            and all(isinstance(value, str) and value for value in values)
            and values == sorted(values)
            and len(values) == len(set(values)),
            f"structured unittest {field} drift",
        )
        return values

    started_ids = strict_ids("started_test_ids")
    success_ids = strict_ids("success_ids")
    failure_ids = strict_ids("failure_ids")
    error_ids = strict_ids("error_ids")
    expected_failure_ids = strict_ids("expected_failure_ids")
    unexpected_success_ids = strict_ids("unexpected_success_ids")
    failed_subtest_ids = strict_ids("failed_subtest_ids")
    skipped_subtest_ids = strict_ids("skipped_subtest_ids")

    def strict_skip_rows(field: str) -> list[tuple[str, str]]:
        rows = structured.get(field)
        _require(
            isinstance(rows, list),
            f"structured unittest {field} rows drift",
        )
        pairs: list[tuple[str, str]] = []
        for index, row in enumerate(rows):
            record = _exact_keys(
                row,
                {"id", "reason"},
                label=f"structured {field} row {index}",
            )
            _require(
                isinstance(record["id"], str)
                and bool(record["id"])
                and isinstance(record["reason"], str),
                f"structured {field} row value drift",
            )
            pairs.append((record["id"], record["reason"]))
        _require(
            pairs == sorted(pairs)
            and len({test_id for test_id, _reason in pairs}) == len(pairs),
            f"structured {field} rows are not sorted and unique by test",
        )
        return pairs

    skipped_pairs = strict_skip_rows("skipped")
    fixture_skipped_pairs = strict_skip_rows("fixture_skipped")
    skipped_ids = [test_id for test_id, _reason in skipped_pairs]
    fixture_skipped_ids = [
        test_id for test_id, _reason in fixture_skipped_pairs
    ]
    test_count = structured.get("tests_run")
    _require(
        structured.get("version")
        == "2026-08-09-step28-v13-v1-12-unittest-json-v2"
        and type(test_count) is int
        and test_count > 0
        and len(started_ids) == test_count
        and structured.get("was_successful") is False
        and isinstance(structured.get("wall_seconds"), (int, float))
        and not isinstance(structured.get("wall_seconds"), bool)
        and math.isfinite(float(structured["wall_seconds"]))
        and float(structured["wall_seconds"]) >= 0.0,
        "structured unittest identity/count drift",
    )
    partitions = [set(success_ids), set(skipped_ids), set(failure_ids)]
    _require(
        failure_ids == [WAIVED_TEST_ID]
        and structured["skipped"] == EXPECTED_STARTED_SKIPPED
        and structured["fixture_skipped"] == EXPECTED_FIXTURE_SKIPPED
        and error_ids == []
        and expected_failure_ids == []
        and unexpected_success_ids == []
        and failed_subtest_ids == []
        and skipped_subtest_ids == []
        and all(
            not left.intersection(right)
            for index, left in enumerate(partitions)
            for right in partitions[index + 1 :]
        )
        and set().union(*partitions) == set(started_ids)
        and not set(fixture_skipped_ids).intersection(started_ids)
        and not set(fixture_skipped_ids).intersection(
            set().union(*partitions, set(error_ids))
        ),
        "structured unittest exact outcome partition drift",
    )
    raw_tests = test_count
    raw_passed = len(success_ids)
    raw_skipped = len(skipped_ids)
    raw_fixture_skipped = len(fixture_skipped_ids)
    raw_failed = len(failure_ids)
    raw_errors = len(error_ids)
    count_fields = (
        "test_count",
        "passed_count",
        "skipped_count",
        "failed_count",
        "error_count",
        "raw_test_count",
        "raw_passed_count",
        "raw_skipped_count",
        "raw_fixture_skipped_count",
        "raw_failed_count",
        "raw_error_count",
        "accepted_waived_failure_count",
        "subprocess_return_code",
        "raw_subprocess_return_code",
        "source_closure_member_count",
        "formal_rows_produced",
    )
    _require(
        all(type(tests.get(field)) is int for field in count_fields),
        "full-test receipt count type drift",
    )
    _require(
        tests.get("structured_result_sha256")
        == preceremony.canonical_sha256(structured),
        "structured unittest result hash drift",
    )
    _validate_test_suite_closure_record(tests.get("test_suite_source_closure"))
    _require(
        tests.get("version") == "2026-08-09-step28-v13-v1-12-full-tests-v3"
        and tests.get("status") == "PASS_FULL_REPOSITORY_TESTS"
        and tests.get("status_semantics") == "PASS_WITH_ONE_EXACT_HISTORICAL_MANIFEST_FAILURE_WAIVER"
        and tests.get("count_semantics")
        == (
            "raw_tests_use_unittest_testsRun; raw_skips_cover_started_tests; "
            "raw_fixture_skips_are_nonstarted_events; compatibility_skipped_"
            "count_includes_one_accepted_waiver"
        )
        and tests.get("accepted_waived_failure_count") == 1
        and tests.get("accepted_waived_failure_ids") == [WAIVED_TEST_ID]
        and tests.get("raw_failure_ids") == failure_ids
        and tests.get("raw_error_ids") == error_ids
        and tests.get("raw_skipped_ids") == skipped_ids
        and tests.get("raw_fixture_skipped_ids") == fixture_skipped_ids
        and tests.get("raw_failed_subtest_ids") == failed_subtest_ids
        and tests.get("raw_skipped_subtest_ids") == skipped_subtest_ids
        and tests.get("raw_expected_failure_ids") == expected_failure_ids
        and tests.get("raw_unexpected_success_ids") == unexpected_success_ids
        and tests.get("raw_test_count") == raw_tests
        and tests.get("raw_passed_count") == raw_passed
        and tests.get("raw_skipped_count") == raw_skipped
        and tests.get("raw_fixture_skipped_count") == raw_fixture_skipped
        and tests.get("raw_failed_count") == raw_failed == 1
        and tests.get("raw_error_count") == raw_errors == 0
        and int(tests.get("raw_subprocess_return_code", -1)) == 1
        and int(tests.get("subprocess_return_code", -1)) == 1
        and tests.get("test_count") == raw_tests
        and tests.get("passed_count") == raw_passed
        and tests.get("skipped_count") == raw_skipped + 1
        and tests.get("failed_count") == 0
        and tests.get("error_count") == 0
        and tests.get("test_runner_reported_seconds")
        == structured["wall_seconds"],
        "full-test waiver projection drift",
    )
    _require(
        GIT_OID_RE.fullmatch(str(tests.get("git_head", ""))) is not None
        and HEX_SHA256_RE.fullmatch(
            str(tests.get("source_closure_canonical_sha256", ""))
        )
        is not None
        and HEX_SHA256_RE.fullmatch(
            str(tests.get("captured_output_sha256", ""))
        )
        is not None
        and tests.get("formal_seed_or_key_access") is False
        and tests.get("formal_rows_produced") == 0,
        "full-test immutable boundary drift",
    )
    targeted_ids = sorted(
        test_id
        for test_id in started_ids
        if "test_step28_v13_v1_12_" in test_id
    )
    targeted_successes = sorted(
        test_id
        for test_id in success_ids
        if "test_step28_v13_v1_12_" in test_id
    )
    targeted = tests.get("current_v1_12_targeted_tests", {})
    _require(
        targeted_ids
        and targeted_successes == targeted_ids
        and targeted
        == {
            "tests_run": len(targeted_ids),
            "passed": len(targeted_successes),
            "skipped_ids": [],
            "failure_ids": [],
            "error_ids": [],
        },
        "current v1.12 targeted test boundary drift",
    )
    waiver_pin = tests.get("historical_manifest_waiver")
    _require(waiver_pin == WAIVER_RECEIPT_PIN, "full-test waiver pin drift")
    _require(waiver["waiver_scope"]["exact_test"] == WAIVED_TEST_ID, "waiver/test correspondence drift")


def validate_current_full_test_environment(tests: Mapping[str, Any]) -> None:
    """Reject stale receipts before the first irreversible seed start."""

    expected = _validate_test_suite_closure_record(
        tests.get("test_suite_source_closure")
    )
    observed = test_suite_source_closure()
    _require(observed == expected, "test-suite source closure changed after full tests")
    require_tracked_worktree_matches_head()
    require_committed_clean_test_tree(observed)
    _require(current_git_head() == tests.get("git_head"), "full-test receipt Git HEAD is stale")


def validate_authorization_prelock_document(
    prelock: Mapping[str, Any],
    *,
    legacy_validation: Mapping[str, Any],
    dereference_waiver_state: bool,
) -> dict[str, Any]:
    """Validate the overlay already embedded in one prelock document."""

    _exact_keys(prelock, PRELOCK_KEYS, label="composite formal prelock")
    if "prelock" in legacy_validation:
        _require(
            legacy_validation["prelock"] == prelock,
            "legacy/composite prelock object drift",
        )
    _require(prelock.get("design_evidence_role") == DESIGN_EVIDENCE_ROLE, "design evidence role drift")
    _require(prelock.get("authorization_evidence_role") == AUTHORIZATION_EVIDENCE_ROLE, "authorization evidence role drift")
    _verify_exact_pin(prelock["authorization_overlay_contract"], OVERLAY_PIN, label="authorization overlay")
    evidence = _exact_keys(
        prelock.get("authorization_evidence"),
        {"final_text_shortcut_preflight", "external_interruption", "historical_manifest_waiver"},
        label="authorization evidence",
    )
    text_receipt = _load_exact_self_hashed_receipt(
        evidence["final_text_shortcut_preflight"], TEXT_RECEIPT_PIN, label="final text shortcut receipt"
    )
    interruption = _load_exact_self_hashed_receipt(
        evidence["external_interruption"], INTERRUPTION_RECEIPT_PIN, label="external interruption receipt"
    )
    waiver = _load_exact_self_hashed_receipt(
        evidence["historical_manifest_waiver"], WAIVER_RECEIPT_PIN, label="historical manifest waiver receipt"
    )
    validate_text_receipt(text_receipt)
    validate_interruption_receipt(interruption)
    validate_waiver_receipt(waiver, dereference_current_state=dereference_waiver_state)
    tests = legacy_validation.get("test_receipt")
    _require(isinstance(tests, Mapping), "legacy full-test receipt is unavailable")
    validate_full_test_receipt(tests, waiver)
    _require(
        prelock.get("git_head") == tests.get("git_head"),
        "prelock/full-test Git HEAD drift",
    )
    if dereference_waiver_state:
        validate_current_full_test_environment(tests)
    source_paths = {str(row.get("path", "")) for row in legacy_validation.get("source_members", [])}
    if dereference_waiver_state:
        require_paths_git_tracked(
            source_paths, label="formal source closure"
        )
    required_paths = {
        OVERLAY_PIN["path"], TEXT_RECEIPT_PIN["path"], INTERRUPTION_RECEIPT_PIN["path"],
        WAIVER_RECEIPT_PIN["path"], "scripts/step28_v13_v1_12_prelock_evidence.py",
        "scripts/step28_v13_v1_12_unittest_json_runner.py",
        "scripts/step28_v13_v1_12_freeze_prelock.py", "scripts/step28_v13_v1_12_seed_ceremony.py",
        "tests/test_step28_v13_v1_12_authorization_overlay_contracts.py",
        "tests/test_step28_v13_v1_12_formal_execution_contracts.py",
    }
    _require(required_paths.issubset(source_paths), "authorization source closure is incomplete")
    return {
        "prelock": prelock,
        "legacy_validation": legacy_validation,
        "text_receipt": text_receipt,
        "interruption_receipt": interruption,
        "waiver_receipt": waiver,
        "dereferenced_waiver_state": dereference_waiver_state,
    }


def load_and_validate_authorized_prelock(
    path: Path = formal.DEFAULT_PRELOCK_PATH,
    *,
    dereference_waiver_state: bool,
) -> dict[str, Any]:
    """Run frozen compatibility validation and then the mandatory overlay."""

    legacy = formal.load_and_validate_prelock(path)
    return validate_authorization_prelock_document(
        legacy["prelock"],
        legacy_validation=legacy,
        dereference_waiver_state=dereference_waiver_state,
    )
