#!/usr/bin/env python3
"""Archive hash-only v1.12 failure lineage and delete failed payload trees."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import step28_v13_identity_values as identity_values
import step28_v13_v1_12_formal_common as formal
import step28_v13_v1_12_preceremony as preceremony


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ARCHIVE = (
    ROOT
    / "reports"
    / "step28_synthetic_chinese_dataset"
    / "failure_records"
    / "step28_v13_v1_12_failure_exclusions_20260803.json"
)


class FailureClosureError(ValueError):
    """Raised when failed v1.12 cannot be closed without losing lineage."""


def _identity_hashes(root: Path) -> set[str]:
    hashes: set[str] = set()
    if not preceremony.exists_long_path(root):
        return hashes
    for path in preceremony.walk_files_long_path(root):
        relative = path.relative_to(root).as_posix()
        if relative.endswith("identity_value_hashes.json"):
            document = preceremony.load_json_strict(path)
            preceremony.validate_canonical_self_hash(
                document, label="failed identity hash document"
            )
            values = document.get("hashes", [])
            if (
                not isinstance(values, list)
                or int(document.get("hash_count", -1)) != len(values)
                or values != sorted(values)
                or len(values) != len(set(values))
                or any(
                    not isinstance(value, str)
                    or preceremony.HEX_SHA256_RE.fullmatch(value) is None
                    for value in values
                )
            ):
                raise FailureClosureError("Failed identity hash document drift")
            hashes.update(values)
        elif relative.endswith("identity_assets.jsonl"):
            with open(
                preceremony._filesystem_path(path), "rb"
            ) as handle:
                for line_number, raw_line in enumerate(handle, start=1):
                    try:
                        line = raw_line.decode("utf-8")
                        row = json.loads(line)
                    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                        # Only a genuinely unterminated final fragment may be
                        # discarded.  A malformed newline-terminated row, or
                        # any later byte, means earlier hashes are not a safe
                        # complete account of the failed payload.
                        if raw_line.endswith((b"\n", b"\r")) or handle.read(1):
                            raise FailureClosureError(
                                "Malformed failed identity JSONL before EOF"
                            ) from exc
                        break
                    value = (
                        row.get("identity_value")
                        if isinstance(row, dict)
                        else None
                    )
                    if not isinstance(value, str) or not value.strip():
                        raise FailureClosureError(
                            "Parsed failed identity row lacks identity_value: "
                            f"{relative}:{line_number}"
                        )
                    hashes.add(identity_values.value_hash(value))
    malformed = [
        value
        for value in hashes
        if preceremony.HEX_SHA256_RE.fullmatch(value) is None
    ]
    if malformed:
        raise FailureClosureError("Malformed failed identity hash")
    return hashes


def _safe_delete_tree(path: Path, *, expected_parent: Path) -> None:
    resolved = path.resolve()
    parent = expected_parent.resolve()
    try:
        resolved.relative_to(parent)
    except ValueError as exc:
        raise FailureClosureError("Failure cleanup target escapes run parent") from exc
    if resolved == parent:
        raise FailureClosureError("Refusing to delete failure parent itself")
    if preceremony.exists_long_path(path):
        shutil.rmtree(preceremony._filesystem_path(path))


def _json_bytes(value: dict[str, Any]) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        + b"\n"
    )


def _publish_or_verify(path: Path, payload: bytes, *, label: str) -> None:
    if preceremony.exists_long_path(path):
        if preceremony.read_bytes_long_path(path) != payload:
            raise FailureClosureError(f"Existing {label} differs from replay")
        return
    preceremony.write_bytes_no_replace_long_path(path, payload)


def close_failed_run(
    *, phase: str, error_type: str, archive_path: Path = DEFAULT_ARCHIVE
) -> dict[str, Any]:
    if not phase or not error_type:
        raise FailureClosureError("Failure phase/type must be explicit")
    formal.require_canonical_path(
        archive_path,
        DEFAULT_ARCHIVE,
        label="v1.12 failure exclusion archive",
    )
    validated = formal.load_and_validate_prelock()
    draft = validated["draft"]
    public_root = preceremony._repo_path(str(draft["release"]["public_root"]))
    private_root = preceremony._repo_path(str(draft["release"]["private_root"]))
    ceremony_path = preceremony._repo_path(
        str(validated["prelock"]["custody"]["public_ceremony_receipt_path"])
    )
    master_commitments: dict[str, str] = {}
    if preceremony.exists_long_path(ceremony_path):
        ceremony = preceremony.load_json_strict(ceremony_path)
        preceremony.validate_canonical_self_hash(
            ceremony, label="failed-run seed ceremony receipt"
        )
        master_commitments = {
            str(key): str(value)
            for key, value in ceremony.get("master_commitments", {}).items()
        }
        if (
            ceremony.get("status") != "PASS_ONE_SHOT_SEED_CEREMONY"
            or set(master_commitments) != set(formal.SPLITS)
            or any(
                preceremony.HEX_SHA256_RE.fullmatch(value) is None
                for value in master_commitments.values()
            )
        ):
            raise FailureClosureError("Failed-run master commitment receipt drift")
    if preceremony.exists_long_path(archive_path):
        archive = preceremony.load_json_strict(archive_path)
        preceremony.validate_canonical_self_hash(
            archive, label="v1.12 failure exclusion archive"
        )
        archived_values = archive.get("failed_identity_value_hashes", [])
        if (
            archive.get("version")
            != "2026-08-03-step28-v13-v1-12-failure-closure-v1"
            or archive.get("status")
            != "FAIL_V1_12_PERMANENTLY_CLOSED_HASHES_ARCHIVED"
            or archive.get("run_id") != draft["run_id"]
            or archive.get("failure_phase") != phase
            or archive.get("error_type") != error_type
            or not isinstance(archived_values, list)
            or archived_values != sorted(archived_values)
            or len(archived_values) != len(set(archived_values))
            or int(archive.get("failed_identity_value_hash_count", -1))
            != len(archived_values)
            or any(
                preceremony.HEX_SHA256_RE.fullmatch(str(value)) is None
                for value in archived_values
            )
            or not isinstance(
                archive.get("forbidden_master_seed_commitments"), dict
            )
            or (
                archive.get("forbidden_master_seed_commitments")
                and (
                    set(archive["forbidden_master_seed_commitments"])
                    != set(formal.SPLITS)
                    or any(
                        preceremony.HEX_SHA256_RE.fullmatch(str(value)) is None
                        for value in archive[
                            "forbidden_master_seed_commitments"
                        ].values()
                    )
                )
            )
            or archive.get("raw_identity_values_persisted") is not False
            or archive.get("raw_private_keys_persisted") is not False
            or archive.get("replacement_seed_or_retry_forbidden") is not True
        ):
            raise FailureClosureError("Existing failure archive semantic drift")
        hashes = set(archived_values)
        if preceremony.exists_long_path(private_root):
            observed_hashes = _identity_hashes(private_root)
            if not observed_hashes <= hashes:
                raise FailureClosureError(
                    "Retained payload contains identity hashes absent from archive"
                )
        if master_commitments and archive.get(
            "forbidden_master_seed_commitments"
        ) != master_commitments:
            raise FailureClosureError(
                "Existing failure archive master commitments drift"
            )
        _publish_or_verify(
            archive_path,
            _json_bytes(archive),
            label="v1.12 failure archive",
        )
    else:
        hashes = _identity_hashes(private_root)
        archive = preceremony.with_canonical_self_hash(
            {
                "version": "2026-08-03-step28-v13-v1-12-failure-closure-v1",
                "status": "FAIL_V1_12_PERMANENTLY_CLOSED_HASHES_ARCHIVED",
                "run_id": draft["run_id"],
                "failure_phase": phase,
                "error_type": error_type,
                "failed_identity_value_hash_count": len(hashes),
                "failed_identity_value_hashes": sorted(hashes),
                "forbidden_master_seed_commitments": master_commitments,
                "raw_identity_values_persisted": False,
                "raw_private_keys_persisted": False,
                "replacement_seed_or_retry_forbidden": True,
                "failed_public_and_private_payload_cleanup_required": True,
                "failed_run_code_requires_documentation_then_git_deletion": True,
                "scientific_metrics_produced": False,
            }
        )
        _publish_or_verify(
            archive_path,
            _json_bytes(archive),
            label="v1.12 failure archive",
        )
    _safe_delete_tree(public_root, expected_parent=public_root.parent)
    _safe_delete_tree(private_root, expected_parent=private_root.parent)
    cleanup = preceremony.with_canonical_self_hash(
        {
            "version": "2026-08-03-step28-v13-v1-12-failure-cleanup-v1",
            "status": "PASS_FAILED_V1_12_PAYLOAD_TREES_DELETED",
            "run_id": draft["run_id"],
            "archive_path": archive_path.relative_to(ROOT).as_posix(),
            "archive_sha256": preceremony.sha256_file(archive_path),
            "archive_canonical_self_hash": archive["canonical_self_hash"],
            "public_payload_root_exists": preceremony.exists_long_path(
                public_root
            ),
            "private_payload_root_exists": preceremony.exists_long_path(
                private_root
            ),
            "raw_identity_values_or_private_keys_retained": False,
        }
    )
    if cleanup["public_payload_root_exists"] or cleanup["private_payload_root_exists"]:
        raise FailureClosureError("Failed v1.12 payload cleanup did not close")
    cleanup_path = archive_path.with_name(
        archive_path.stem + ".cleanup_receipt.json"
    )
    _publish_or_verify(
        cleanup_path,
        _json_bytes(cleanup),
        label="v1.12 failure cleanup receipt",
    )
    return {**cleanup, "failed_identity_value_hash_count": len(hashes)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", required=True)
    parser.add_argument("--error-type", required=True)
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument(
        "--confirm-permanent-close",
        action="store_true",
        help="Required acknowledgement; this deletes failed v1.12 payload trees.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.confirm_permanent_close:
        raise FailureClosureError("--confirm-permanent-close is required")
    archive = close_failed_run(
        phase=args.phase,
        error_type=args.error_type,
        archive_path=args.archive.resolve(),
    )
    print(
        archive["status"],
        archive["failed_identity_value_hash_count"],
        archive["canonical_self_hash"],
    )


if __name__ == "__main__":
    main()
