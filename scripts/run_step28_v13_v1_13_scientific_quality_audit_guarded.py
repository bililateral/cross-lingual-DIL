#!/usr/bin/env python3
"""Reviewed external launch guard for the Step28-v13 v1.13 quality audit."""

from __future__ import annotations

import hashlib
import __main__
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
RELEASE_MANIFEST = (
    ROOT
    / "schema"
    / "step28_v13_v1_13_scientific_quality_audit_release_manifest.json"
)
ANCHOR = (
    ROOT
    / "schema"
    / "step28_v13_v1_13_scientific_quality_audit_launch_anchor.json"
)
EXTERNAL_REVIEW_ATTESTATION = (
    ROOT
    / "schema"
    / "step28_v13_v1_13_scientific_quality_audit_external_review_attestation.json"
)
EXTERNAL_REVIEW_VERDICT = "允许清洁运行104-world质量审计"
GIT_OBJECT_ID = re.compile(r"^[0-9a-f]{40}$")
LAUNCH_FAILURE_ROOT = (
    ROOT
    / "reports"
    / "step28_v13_v1_13_scientific_builder"
    / "quality_audit_launch_failures_v7"
)


class LaunchGuardError(RuntimeError):
    """Fail-closed launch error."""


class ExternalReviewPending(LaunchGuardError):
    """The exact candidate has not received an independent external GO."""


class GuardedAuditTerminated(RuntimeError):
    """A post-launch audit exception that must not escape as a traceback."""

    def __init__(self, error: BaseException) -> None:
        super().__init__("guarded audit terminated")
        self.exception_type = type(error).__name__


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    payload = dict(value)
    payload.pop("canonical_self_hash", None)
    return hashlib.sha256(
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _git_command(root: Path, *arguments: str) -> bytes:
    completed = subprocess.run(
        ["git", *arguments],
        cwd=root,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=30,
        check=False,
    )
    if completed.returncode != 0:
        raise LaunchGuardError("Candidate Git provenance command failed")
    return completed.stdout


def _verify_candidate_git_provenance(
    git_provenance: dict[str, Any],
    release: dict[str, Any],
    *,
    root: Path = ROOT,
    allow_missing_dataset_manifest: bool = False,
) -> None:
    commit = _git_command(root, "rev-parse", "HEAD").decode("ascii").strip()
    tree = _git_command(root, "rev-parse", "HEAD^{tree}").decode("ascii").strip()
    if (
        commit != git_provenance["commit"]
        or tree != git_provenance["tree"]
        or _git_command(root, "cat-file", "-t", commit).strip() != b"commit"
        or _git_command(root, "cat-file", "-t", tree).strip() != b"tree"
    ):
        raise LaunchGuardError("External review Git object binding drift")
    for name, spec in release["pins"].items():
        relative = str(spec["path"])
        if (
            Path(relative).is_absolute()
            or Path(relative).drive
            or any(part in {"", ".", ".."} for part in Path(relative).parts)
        ):
            raise LaunchGuardError("Release Git path contract drift")
        committed = _git_command(root, "show", f"{commit}:{relative}")
        if (
            len(committed) != spec["size_bytes"]
            or hashlib.sha256(committed).hexdigest() != spec["sha256"]
        ):
            raise LaunchGuardError(f"Release Git bytes drift: {name}")
        path = root / relative
        if allow_missing_dataset_manifest and name == "dataset_root_manifest":
            if path.exists():
                raise LaunchGuardError(
                    "Cleanup recovery requires the dataset manifest to be absent"
                )
            continue
        status = _git_command(root, "status", "--porcelain=v1", "--", relative)
        if status:
            raise LaunchGuardError(f"Release working-tree bytes are not clean: {name}")


def _verify_release_manifest(
    *, allow_missing_dataset_manifest: bool = False
) -> dict[str, Any]:
    if not RELEASE_MANIFEST.is_file():
        raise LaunchGuardError("External release manifest is missing")
    release = json.loads(RELEASE_MANIFEST.read_text(encoding="utf-8"))
    if (
        not isinstance(release, dict)
        or set(release) != {"version", "status", "pins", "canonical_self_hash"}
        or release.get("canonical_self_hash") != _canonical_sha256(release)
        or release.get("status")
        != "EXTERNAL_REVIEW_CANDIDATE_NOT_FORMAL_AUTHORIZATION"
        or tuple(release["pins"])
        != (
            "launch_anchor",
            "quality_audit_c_amendment",
            "quality_policy",
            "quality_audit",
            "counterfactual_text",
            "blind_literal_scan",
            "sealed_literal_registry_builder",
            "sealed_literal_registry_receipt",
            "quality_tests",
            "dataset_root_manifest",
            "launch_guard",
        )
    ):
        raise LaunchGuardError("External release manifest contract drift")
    for name, spec in release["pins"].items():
        if set(spec) != {"path", "size_bytes", "sha256"}:
            raise LaunchGuardError(f"Release pin schema drift: {name}")
        path = (ROOT / str(spec["path"])).resolve()
        missing_recovery_manifest = (
            allow_missing_dataset_manifest
            and name == "dataset_root_manifest"
            and not path.exists()
        )
        if ROOT.resolve() not in path.parents or (
            not missing_recovery_manifest
            and (
                not path.is_file()
                or path.stat().st_size != spec["size_bytes"]
                or _sha256_file(path) != spec["sha256"]
            )
        ):
            raise LaunchGuardError(f"Release pin bytes drift: {name}")
    guard_pin = release["pins"]["launch_guard"]
    guard_path = Path(__file__).resolve()
    if (
        (ROOT / str(guard_pin["path"])).resolve() != guard_path
        or guard_pin["size_bytes"] != guard_path.stat().st_size
        or guard_pin["sha256"] != _sha256_file(guard_path)
    ):
        raise LaunchGuardError("Release manifest does not pin this guard")
    anchor_pin = release["pins"]["launch_anchor"]
    if (ROOT / str(anchor_pin["path"])).resolve() != ANCHOR.resolve():
        raise LaunchGuardError("Release manifest anchor path drift")
    return release


def _verify_anchor(*, allow_missing_dataset_manifest: bool = False) -> dict[str, Any]:
    if not ANCHOR.is_file():
        raise LaunchGuardError("Reviewed launch anchor is missing")
    anchor = json.loads(ANCHOR.read_text(encoding="utf-8"))
    if (
        not isinstance(anchor, dict)
        or set(anchor)
        != {
            "version",
            "status",
            "pins",
            "runtime",
            "quality_policy_canonical_self_hash",
            "dataset_root_manifest_canonical_self_hash",
            "canonical_self_hash",
        }
        or anchor.get("canonical_self_hash") != _canonical_sha256(anchor)
        or anchor.get("status")
        != "EXTERNAL_REVIEW_ATTESTATION_REQUIRED_BEFORE_EXECUTION"
    ):
        raise LaunchGuardError("Reviewed launch anchor contract drift")
    if tuple(anchor["pins"]) != (
        "quality_audit_c_amendment",
        "quality_policy",
        "quality_audit",
        "counterfactual_text",
        "blind_literal_scan",
        "sealed_literal_registry_builder",
        "sealed_literal_registry_receipt",
        "quality_tests",
        "dataset_root_manifest",
        "launch_guard",
    ):
        raise LaunchGuardError("Reviewed launch anchor pin order drift")
    for name, spec in anchor["pins"].items():
        if set(spec) != {"path", "size_bytes", "sha256"}:
            raise LaunchGuardError(f"Anchor pin schema drift: {name}")
        path = (ROOT / str(spec["path"])).resolve()
        missing_recovery_manifest = (
            allow_missing_dataset_manifest
            and name == "dataset_root_manifest"
            and not path.exists()
        )
        if ROOT.resolve() not in path.parents or (
            not missing_recovery_manifest
            and (
                not path.is_file()
                or path.stat().st_size != spec["size_bytes"]
                or _sha256_file(path) != spec["sha256"]
            )
        ):
            raise LaunchGuardError(f"Anchor pin bytes drift: {name}")
    guard_pin = anchor["pins"]["launch_guard"]
    guard_path = Path(__file__).resolve()
    if (
        (ROOT / str(guard_pin["path"])).resolve() != guard_path
        or guard_pin["size_bytes"] != guard_path.stat().st_size
        or guard_pin["sha256"] != _sha256_file(guard_path)
    ):
        raise LaunchGuardError("Launch guard self-pin drift")
    return anchor


def _verify_external_attestation(
    release: dict[str, Any],
    *,
    allow_missing_dataset_manifest: bool = False,
) -> dict[str, Any]:
    if not EXTERNAL_REVIEW_ATTESTATION.is_file():
        raise ExternalReviewPending("External review GO attestation is absent")
    attestation = json.loads(EXTERNAL_REVIEW_ATTESTATION.read_text(encoding="utf-8"))
    if (
        not isinstance(attestation, dict)
        or set(attestation)
        != {
            "version",
            "status",
            "review_scope",
            "release_manifest",
            "candidate_git_provenance",
            "external_review_provenance",
            "review_transcript",
            "verdict_last_line",
            "external_review_binding_sha256",
            "canonical_self_hash",
        }
        or attestation.get("canonical_self_hash") != _canonical_sha256(attestation)
        or attestation.get("status")
        != "EXTERNAL_REVIEW_GO_DESIGN_QUALITY_AUDIT_ONLY"
        or attestation.get("verdict_last_line") != EXTERNAL_REVIEW_VERDICT
        or attestation.get("review_scope")
        != {
            "design_dataset_root": "design_preflight_v2_20260811",
            "world_count": 104,
            "quality_audit_run_authorized": True,
            "formal_generation_authorized": False,
            "model_training_authorized": False,
            "audit_truth_release_authorized": False,
        }
    ):
        raise LaunchGuardError("External review attestation contract drift")
    release_binding = attestation.get("release_manifest")
    expected_release_binding = {
        "path": RELEASE_MANIFEST.relative_to(ROOT).as_posix(),
        "size_bytes": RELEASE_MANIFEST.stat().st_size,
        "sha256": _sha256_file(RELEASE_MANIFEST),
        "canonical_self_hash": release["canonical_self_hash"],
    }
    if release_binding != expected_release_binding:
        raise LaunchGuardError("External review release binding drift")
    git_provenance = attestation.get("candidate_git_provenance")
    if (
        not isinstance(git_provenance, dict)
        or set(git_provenance) != {"commit", "tree", "implementation_bytes_committed"}
        or GIT_OBJECT_ID.fullmatch(str(git_provenance.get("commit", ""))) is None
        or GIT_OBJECT_ID.fullmatch(str(git_provenance.get("tree", ""))) is None
        or git_provenance.get("implementation_bytes_committed") is not True
    ):
        raise LaunchGuardError("External review Git provenance drift")
    _verify_candidate_git_provenance(
        git_provenance,
        release,
        allow_missing_dataset_manifest=allow_missing_dataset_manifest,
    )
    provenance = attestation.get("external_review_provenance")
    if (
        not isinstance(provenance, dict)
        or set(provenance)
        != {
            "provider",
            "model",
            "conversation_url_sha256",
            "completed_at_utc",
        }
        or provenance.get("provider") != "chatgpt.com"
        or provenance.get("model") != "GPT-5.6 Sol Pro"
        or re.fullmatch(r"[0-9a-f]{64}", str(provenance.get("conversation_url_sha256", "")))
        is None
        or not isinstance(provenance.get("completed_at_utc"), str)
        or not str(provenance["completed_at_utc"]).endswith("Z")
    ):
        raise LaunchGuardError("External review provenance drift")
    transcript = attestation.get("review_transcript")
    if not isinstance(transcript, dict) or set(transcript) != {
        "path",
        "size_bytes",
        "sha256",
    }:
        raise LaunchGuardError("External review transcript pin drift")
    transcript_path = (ROOT / str(transcript["path"])).resolve()
    if (
        ROOT.resolve() not in transcript_path.parents
        or not transcript_path.is_file()
        or transcript_path.stat().st_size != transcript["size_bytes"]
        or _sha256_file(transcript_path) != transcript["sha256"]
    ):
        raise LaunchGuardError("External review transcript bytes drift")
    try:
        transcript_text = transcript_path.read_bytes().decode("utf-8", errors="strict")
    except (OSError, UnicodeError) as exc:
        raise LaunchGuardError("External review transcript is not strict UTF-8") from exc
    nonempty_lines = [line for line in transcript_text.splitlines() if line.strip()]
    if not nonempty_lines or nonempty_lines[-1] != EXTERNAL_REVIEW_VERDICT:
        raise LaunchGuardError("External review transcript verdict drift")
    binding = {
        "release_manifest": release_binding,
        "candidate_git_provenance": git_provenance,
        "external_review_provenance": provenance,
        "review_transcript": transcript,
        "verdict_last_line": attestation["verdict_last_line"],
        "review_scope": attestation["review_scope"],
    }
    if attestation.get("external_review_binding_sha256") != hashlib.sha256(
        json.dumps(
            binding,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest():
        raise LaunchGuardError("External review binding digest drift")
    return attestation


def _path_evidence(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    if ROOT.resolve() not in resolved.parents or not resolved.is_file():
        raise LaunchGuardError("Launch entry path is unsafe or missing")
    return {
        "path": resolved.relative_to(ROOT).as_posix(),
        "size_bytes": resolved.stat().st_size,
        "sha256": _sha256_file(resolved),
    }


def _verify_imported_quality_module(
    module_file: str | Path, release: dict[str, Any]
) -> dict[str, Any]:
    observed = _path_evidence(Path(module_file))
    if observed != release["pins"]["quality_audit"]:
        raise LaunchGuardError("Imported quality module is not the reviewed source")
    return observed


def _safe_optional_file_binding(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {"exists": False, "size_bytes": None, "sha256": None}
    return {
        "exists": True,
        "size_bytes": path.stat().st_size,
        "sha256": _sha256_file(path),
    }


def _write_launch_failure_receipt(error: LaunchGuardError) -> Path:
    bindings = {
        "release_manifest": _safe_optional_file_binding(RELEASE_MANIFEST),
        "launch_anchor": _safe_optional_file_binding(ANCHOR),
        "launch_guard": _safe_optional_file_binding(Path(__file__).resolve()),
        "external_review_attestation": _safe_optional_file_binding(
            EXTERNAL_REVIEW_ATTESTATION
        ),
    }
    receipt = {
        "version": "2026-08-11-step28-v13-v1-13-quality-launch-failure-v1",
        "status": "AUDITOR_LAUNCH_FAILED_NO_DATASET_CONCLUSION",
        "reason_code": "RESEARCH_LAUNCH_CONTRACT_FAILED",
        "exception_type": type(error).__name__,
        "dataset_rows_opened": False,
        "input_dataset_retained": True,
        "dataset_quality_conclusion_reached": False,
        "formal_generation_authorized": False,
        "model_training_authorized": False,
        "evidence_binding": bindings,
        "canonical_self_hash": None,
    }
    receipt["canonical_self_hash"] = _canonical_sha256(receipt)
    LAUNCH_FAILURE_ROOT.mkdir(parents=True, exist_ok=True)
    path = LAUNCH_FAILURE_ROOT / f"{receipt['canonical_self_hash']}.json"
    payload = (
        json.dumps(
            receipt,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    if path.exists():
        if path.read_bytes() != payload:
            raise LaunchGuardError("Immutable launch failure receipt drift")
        return path
    with path.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    return path


def _guarded_run() -> None:
    if len(sys.argv) != 1:
        raise LaunchGuardError("The frozen launch guard accepts no arguments")
    release = _verify_release_manifest()
    anchor = _verify_anchor()
    external_review = _verify_external_attestation(release)
    guard_path = Path(__file__).resolve()
    main_path = Path(str(getattr(__main__, "__file__", ""))).resolve()
    argv0_path = Path(sys.argv[0]).resolve()
    if main_path != guard_path or argv0_path != guard_path:
        raise LaunchGuardError("Launch process entry is not the reviewed guard")
    evidence = {
        "release_manifest": {
            "path": RELEASE_MANIFEST.relative_to(ROOT).as_posix(),
            "size_bytes": RELEASE_MANIFEST.stat().st_size,
            "sha256": _sha256_file(RELEASE_MANIFEST),
            "canonical_self_hash": release["canonical_self_hash"],
        },
        "anchor": {
            "path": ANCHOR.relative_to(ROOT).as_posix(),
            "size_bytes": ANCHOR.stat().st_size,
            "sha256": _sha256_file(ANCHOR),
            "canonical_self_hash": anchor["canonical_self_hash"],
        },
        "guard": {
            "path": guard_path.relative_to(ROOT).as_posix(),
            "size_bytes": guard_path.stat().st_size,
            "sha256": _sha256_file(guard_path),
        },
        "external_review_attestation": {
            "path": EXTERNAL_REVIEW_ATTESTATION.relative_to(ROOT).as_posix(),
            "size_bytes": EXTERNAL_REVIEW_ATTESTATION.stat().st_size,
            "sha256": _sha256_file(EXTERNAL_REVIEW_ATTESTATION),
            "canonical_self_hash": external_review["canonical_self_hash"],
        },
    }
    if str(ROOT / "scripts") not in sys.path:
        sys.path.insert(0, str(ROOT / "scripts"))
    import step28_v13_v1_13_scientific_quality_audit as quality

    quality_path = Path(str(quality.__file__)).resolve()
    quality_evidence = _verify_imported_quality_module(quality_path, release)
    evidence["entry"] = {
        "main": _path_evidence(main_path),
        "argv0": _path_evidence(argv0_path),
        "quality_module": quality_evidence,
    }

    try:
        result = quality.run_audit(launch_evidence=evidence)
    except quality.AuditLaunchPreflightError as exc:
        raise LaunchGuardError("Quality launch preflight rejected execution") from exc
    except Exception as exc:
        raise GuardedAuditTerminated(exc) from None
    print(
        json.dumps(
            {
                "status": result["status"],
                "canonical_self_hash": result["canonical_self_hash"],
            },
            ensure_ascii=False,
        )
    )


def _guarded_recover_cleanup() -> None:
    if len(sys.argv) != 2 or sys.argv[1] != "--recover-cleanup-receipt":
        raise LaunchGuardError("Cleanup recovery invocation drift")
    release = _verify_release_manifest(allow_missing_dataset_manifest=True)
    _verify_anchor(allow_missing_dataset_manifest=True)
    _verify_external_attestation(
        release, allow_missing_dataset_manifest=True
    )
    guard_path = Path(__file__).resolve()
    main_path = Path(str(getattr(__main__, "__file__", ""))).resolve()
    argv0_path = Path(sys.argv[0]).resolve()
    if main_path != guard_path or argv0_path != guard_path:
        raise LaunchGuardError("Cleanup recovery entry is not the reviewed guard")
    if str(ROOT / "scripts") not in sys.path:
        sys.path.insert(0, str(ROOT / "scripts"))
    import step28_v13_v1_13_scientific_quality_audit as quality

    _verify_imported_quality_module(Path(str(quality.__file__)), release)
    try:
        receipt = quality.recover_cleanup_receipt()
    except Exception as exc:
        raise GuardedAuditTerminated(exc) from None
    print(
        json.dumps(
            {
                "status": receipt["status"],
                "canonical_self_hash": receipt["canonical_self_hash"],
                "dataset_rows_opened": False,
            },
            ensure_ascii=False,
            sort_keys=True,
        ),
        flush=True,
    )


def main() -> None:
    try:
        if len(sys.argv) == 2 and sys.argv[1] == "--recover-cleanup-receipt":
            _guarded_recover_cleanup()
        else:
            _guarded_run()
    except ExternalReviewPending:
        print(
            json.dumps(
                {
                    "status": "EXTERNAL_REVIEW_PENDING_NO_RUN",
                    "dataset_rows_opened": False,
                },
                ensure_ascii=False,
                sort_keys=True,
            ),
            flush=True,
        )
        raise SystemExit(4)
    except GuardedAuditTerminated as exc:
        print(
            json.dumps(
                {
                    "status": "QUALITY_AUDIT_TERMINATED_SEE_BOUND_RECEIPT",
                    "exception_type": exc.exception_type,
                    "raw_exception_message_returned": False,
                },
                ensure_ascii=False,
                sort_keys=True,
            ),
            flush=True,
        )
        raise SystemExit(3)
    except LaunchGuardError as exc:
        try:
            receipt_path = _write_launch_failure_receipt(exc)
        except Exception as receipt_error:
            print(
                json.dumps(
                    {
                        "status": "AUDITOR_GUARD_FAILURE_RECEIPT_WRITE_FAILED",
                        "exception_type": type(receipt_error).__name__,
                        "raw_exception_message_returned": False,
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                ),
                flush=True,
            )
            raise SystemExit(5)
        print(
            json.dumps(
                {
                    "status": "AUDITOR_LAUNCH_FAILED_NO_DATASET_CONCLUSION",
                    "receipt_sha256": _sha256_file(receipt_path),
                },
                ensure_ascii=False,
                sort_keys=True,
            ),
            flush=True,
        )
        raise SystemExit(2)
    except Exception as exc:
        print(
            json.dumps(
                {
                    "status": "AUDITOR_GUARD_INTERNAL_FAILURE_NO_DATASET_CONCLUSION",
                    "exception_type": type(exc).__name__,
                    "raw_exception_message_returned": False,
                },
                ensure_ascii=False,
                sort_keys=True,
            ),
            flush=True,
        )
        raise SystemExit(6)


if __name__ == "__main__":
    main()
