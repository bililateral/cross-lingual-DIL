#!/usr/bin/env python3
"""Prove that the v1.12 deny archive covers every successful-v1.2 identity.

This is a privileged, pre-seed boundary audit.  It reads the already compromised
historical v1.2 identity assets, but its deterministic public receipt contains
only file pins, counts, and digests.  No raw identity value is persisted or
printed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import unicodedata
from pathlib import Path
from typing import Any, Iterable, Mapping

import step28_v13_v1_12_preceremony as preceremony


ROOT = Path(__file__).resolve().parents[1]
BASELINE_POLICY_PATH = (
    ROOT / "schema/step28_v13_v1_12_cleanroom_preceremony_policy.json"
)
HISTORICAL_RELEASE_SPEC = {
    "path": (
        "reports/step28_synthetic_chinese_dataset/"
        "v13_training_ready_v1_2_order_repair_20260731/release_manifest.json"
    ),
    "sha256": "81ca7d9d2040d500b3bcb2ffc9af6aeb72c581754dbd075b94dd6cf8904b8275",
    "size_bytes": 21536,
    "canonical_self_hash": (
        "59001459bc9b3a908ab0efa1f9f46a6c821bf6078ba3dc5f3308f910d0c5e00b"
    ),
}
OUTPUT_PATH = (
    ROOT
    / "reports/step28_synthetic_chinese_dataset/design_preflights/"
    "v1_12_cleanroom_20260803/"
    "historical_identity_exclusion_coverage_receipt.json"
)
SPLITS = ("train", "development", "audit_a", "audit_b")
EXPECTED_SPLIT_COUNTS = {
    "train": 42_000,
    "development": 42_000,
    "audit_a": 42_000,
    "audit_b": 44_500,
}
NORMALIZATION = "SHA256(UTF-8(NFC(casefold(strip(value)))))"


class HistoricalIdentityCoverageError(RuntimeError):
    """Raised when the historical deny-lineage proof cannot be reproduced."""


def _strict_object_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for key, value in pairs:
        if key in output:
            raise HistoricalIdentityCoverageError(
                f"Duplicate JSON key in historical identity asset: {key}"
            )
        output[key] = value
    return output


def _value_hash(value: Any) -> str:
    if not isinstance(value, str):
        raise HistoricalIdentityCoverageError(
            "Historical identity_value must be a string"
        )
    normalized = unicodedata.normalize("NFC", value.strip().casefold())
    if not normalized:
        raise HistoricalIdentityCoverageError(
            "Historical identity_value cannot be empty"
        )
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _ordered_hash_digest(values: Iterable[str]) -> str:
    digest = hashlib.sha256()
    for value in sorted(values, key=lambda item: item.encode("utf-8")):
        digest.update(value.encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def _validate_hash_list(
    values: Any, *, expected_count: int, label: str
) -> frozenset[str]:
    if not isinstance(values, list) or len(values) != expected_count:
        raise HistoricalIdentityCoverageError(f"{label} count drift")
    previous: str | None = None
    for value in values:
        if (
            not isinstance(value, str)
            or preceremony.HEX_SHA256_RE.fullmatch(value) is None
            or (previous is not None and value <= previous)
        ):
            raise HistoricalIdentityCoverageError(
                f"{label} is not strictly sorted unique SHA-256"
            )
        previous = value
    return frozenset(values)


def _load_original_deny_registry(
    baseline_policy: Mapping[str, Any],
) -> tuple[dict[str, Any], frozenset[str], dict[str, Any]]:
    base_policy_spec = next(
        (
            record
            for record in baseline_policy["design_only_base_inputs"]
            if str(record["path"]).endswith(
                "step28_v13_synthetic_chinese_dataset_policy.json"
            )
        ),
        None,
    )
    if base_policy_spec is None:
        raise HistoricalIdentityCoverageError("Pinned base DGP policy is absent")
    base_policy_path = preceremony.verify_file_pin(
        base_policy_spec, label="historical identity coverage base DGP policy"
    )
    base_policy = preceremony.load_json_strict(base_policy_path)
    deny_spec = base_policy["identity_design"]["identity_value_generation"][
        "salt_selection"
    ]["deny_hash_artifact"]
    deny_path = preceremony._repo_path(str(deny_spec["path"]))
    if preceremony.sha256_file(deny_path) != str(deny_spec["sha256"]):
        raise HistoricalIdentityCoverageError(
            "Original identity deny registry SHA-256 drift"
        )
    deny_size = preceremony.stat_long_path(deny_path).st_size
    deny = preceremony.load_json_strict(deny_path)
    preceremony.validate_canonical_self_hash(
        deny, label="original identity deny registry"
    )
    expected_count = int(deny.get("unique_value_hash_count", -1))
    hashes = _validate_hash_list(
        deny.get("value_hashes"),
        expected_count=expected_count,
        label="original identity deny registry",
    )
    if (
        deny.get("status") != "PASS_BOUNDARY_ONLY"
        or deny.get("normalization") != NORMALIZATION
        or expected_count != 112_996
    ):
        raise HistoricalIdentityCoverageError(
            "Original identity deny registry semantic drift"
        )
    record = {
        "path": deny_spec["path"],
        "sha256": deny_spec["sha256"],
        "size_bytes": deny_size,
        "canonical_self_hash": deny["canonical_self_hash"],
        "unique_value_hash_count": len(hashes),
        "ordered_value_hash_digest": _ordered_hash_digest(hashes),
    }
    return base_policy, hashes, record


def _load_historical_release() -> tuple[Path, dict[str, Any]]:
    path = preceremony.verify_file_pin(
        HISTORICAL_RELEASE_SPEC,
        label="historical successful v1.2 release manifest",
    )
    release = preceremony.load_json_strict(path)
    preceremony.validate_canonical_self_hash(
        release, label="historical successful v1.2 release manifest"
    )
    if (
        release.get("canonical_self_hash")
        != HISTORICAL_RELEASE_SPEC["canonical_self_hash"]
        or release.get("status") != "PASS_DATASET_ONLY_READY_FOR_M0_M1_M2"
        or release.get("run_id")
        != "v13_training_ready_v1_2_order_repair_20260731"
        or set(release.get("split_receipts", {})) != set(SPLITS)
    ):
        raise HistoricalIdentityCoverageError(
            "Historical successful v1.2 release semantic drift"
        )
    return path, release


def _stream_split_identity_hashes(
    *, release_root: Path, release: Mapping[str, Any], split: str
) -> tuple[frozenset[str], dict[str, Any]]:
    split_manifest_path = release_root / split / "split_manifest.json"
    split_receipt = release["split_receipts"][split]
    if (
        preceremony.sha256_file(split_manifest_path)
        != split_receipt["manifest_sha256"]
    ):
        raise HistoricalIdentityCoverageError(
            f"Historical {split} split manifest SHA-256 drift"
        )
    manifest = preceremony.load_json_strict(split_manifest_path)
    preceremony.validate_canonical_self_hash(
        manifest, label=f"historical {split} split manifest"
    )
    if (
        manifest.get("canonical_self_hash")
        != split_receipt["manifest_self_sha256"]
        or manifest.get("split") != split
        or manifest.get("run_id")
        != "v13_training_ready_v1_2_order_repair_20260731"
    ):
        raise HistoricalIdentityCoverageError(
            f"Historical {split} split manifest semantic drift"
        )
    records = [
        record
        for record in manifest.get("files", [])
        if record.get("path") == "private_oracle/identity_assets.jsonl"
    ]
    if len(records) != 1 or records[0].get("model_mount_allowed") is not False:
        raise HistoricalIdentityCoverageError(
            f"Historical {split} identity asset manifest record drift"
        )
    identity_record = records[0]
    identity_path = release_root / split / identity_record["path"]
    if (
        preceremony.sha256_file(identity_path) != identity_record["sha256"]
        or preceremony.stat_long_path(identity_path).st_size
        != int(identity_record["size_bytes"])
    ):
        raise HistoricalIdentityCoverageError(
            f"Historical {split} identity asset bytes drift"
        )

    hashes: set[str] = set()
    row_count = 0
    with open(
        preceremony._filesystem_path(identity_path),
        "r",
        encoding="utf-8",
        newline="",
    ) as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                raise HistoricalIdentityCoverageError(
                    f"Blank historical identity row: {split}:{line_number}"
                )
            try:
                row = json.loads(line, object_pairs_hook=_strict_object_pairs)
            except json.JSONDecodeError as exc:
                raise HistoricalIdentityCoverageError(
                    f"Malformed historical identity row: {split}:{line_number}"
                ) from exc
            if not isinstance(row, dict) or "identity_value" not in row:
                raise HistoricalIdentityCoverageError(
                    f"Missing historical identity_value: {split}:{line_number}"
                )
            value_hash = _value_hash(row["identity_value"])
            if value_hash in hashes:
                raise HistoricalIdentityCoverageError(
                    f"Duplicate historical identity value hash in {split}"
                )
            hashes.add(value_hash)
            row_count += 1
    expected_count = EXPECTED_SPLIT_COUNTS[split]
    if row_count != expected_count or len(hashes) != expected_count:
        raise HistoricalIdentityCoverageError(
            f"Historical {split} identity count drift"
        )
    return frozenset(hashes), {
        "split_manifest": {
            "path": split_manifest_path.relative_to(ROOT).as_posix(),
            "sha256": split_receipt["manifest_sha256"],
            "size_bytes": preceremony.stat_long_path(
                split_manifest_path
            ).st_size,
            "canonical_self_hash": manifest["canonical_self_hash"],
        },
        "identity_assets": {
            "path": identity_path.relative_to(ROOT).as_posix(),
            "sha256": identity_record["sha256"],
            "size_bytes": int(identity_record["size_bytes"]),
        },
        "row_count": row_count,
        "unique_value_hash_count": len(hashes),
        "ordered_value_hash_digest": _ordered_hash_digest(hashes),
    }


def build_receipt() -> dict[str, Any]:
    validated = preceremony.validate_policy(BASELINE_POLICY_PATH)
    baseline_policy = validated["policy"]
    _, original_hashes, original_record = _load_original_deny_registry(
        baseline_policy
    )
    release_path, release = _load_historical_release()
    release_root = release_path.parent

    split_records: dict[str, Any] = {}
    historical_hashes: set[str] = set()
    for split in SPLITS:
        split_hashes, split_record = _stream_split_identity_hashes(
            release_root=release_root,
            release=release,
            split=split,
        )
        if historical_hashes & split_hashes:
            raise HistoricalIdentityCoverageError(
                "Historical v1.2 identity hashes overlap across splits"
            )
        historical_hashes.update(split_hashes)
        split_records[split] = split_record

    failed_hashes = frozenset(validated["failed_identity_hashes"])
    old_boundary = original_hashes | historical_hashes
    if (
        len(historical_hashes) != 170_500
        or original_hashes & historical_hashes
        or len(old_boundary) != 283_496
        or not original_hashes <= failed_hashes
        or not historical_hashes <= failed_hashes
        or not old_boundary <= failed_hashes
    ):
        raise HistoricalIdentityCoverageError(
            "Historical identity exclusion coverage is incomplete"
        )

    archive_spec = baseline_policy["failed_identity_exclusion_archive"]
    archive = preceremony.load_json_strict(
        preceremony.verify_file_pin(
            archive_spec, label="combined failed identity exclusion archive"
        )
    )
    producer_path = Path(__file__).resolve()
    receipt = preceremony.with_canonical_self_hash(
        {
            "version": (
                "2026-08-03-step28-v13-v1-12-historical-identity-"
                "coverage-receipt-v1"
            ),
            "status": (
                "PASS_HISTORICAL_IDENTITY_EXCLUSION_COVERAGE_"
                "NO_FORMAL_AUTHORIZATION"
            ),
            "run_id": baseline_policy["run_id"],
            "normalization": NORMALIZATION,
            "formal_seed_or_key_access": False,
            "formal_dataset_rows_produced": 0,
            "scientific_metrics_produced": False,
            "raw_identity_values_persisted_in_receipt": False,
            "producer": {
                "path": producer_path.relative_to(ROOT).as_posix(),
                "sha256": preceremony.sha256_file(producer_path),
                "size_bytes": preceremony.stat_long_path(producer_path).st_size,
            },
            "historical_release_manifest": {
                **HISTORICAL_RELEASE_SPEC,
                "run_id": release["run_id"],
            },
            "original_identity_deny_registry": original_record,
            "historical_v1_2_splits": split_records,
            "historical_v1_2_identity_union": {
                "unique_value_hash_count": len(historical_hashes),
                "ordered_value_hash_digest": _ordered_hash_digest(
                    historical_hashes
                ),
                "intersection_with_original_deny_count": len(
                    original_hashes & historical_hashes
                ),
            },
            "original_deny_plus_historical_v1_2": {
                "unique_value_hash_count": len(old_boundary),
                "ordered_value_hash_digest": _ordered_hash_digest(old_boundary),
            },
            "combined_exclusion_archive": {
                "path": archive_spec["path"],
                "sha256": archive_spec["sha256"],
                "size_bytes": int(archive_spec["size_bytes"]),
                "canonical_self_hash": archive["canonical_self_hash"],
                "base_unique_value_hash_count": int(
                    archive["base_unique_value_hash_count"]
                ),
                "combined_unique_value_hash_count": len(failed_hashes),
            },
            "coverage": {
                "original_deny_present_count": len(
                    original_hashes & failed_hashes
                ),
                "original_deny_missing_count": len(
                    original_hashes - failed_hashes
                ),
                "historical_v1_2_present_count": len(
                    historical_hashes & failed_hashes
                ),
                "historical_v1_2_missing_count": len(
                    historical_hashes - failed_hashes
                ),
                "old_boundary_present_count": len(old_boundary & failed_hashes),
                "old_boundary_missing_count": len(old_boundary - failed_hashes),
                "all_historical_identity_values_forbidden": True,
            },
            "formal_authorizations_after_audit": dict(
                baseline_policy["authorizations"]
            ),
        }
    )
    return receipt


def publish_receipt(path: Path, receipt: Mapping[str, Any]) -> str:
    payload = json.dumps(
        receipt,
        ensure_ascii=False,
        sort_keys=True,
        indent=2,
        allow_nan=False,
    ).encode("utf-8") + b"\n"
    if preceremony.exists_long_path(path):
        if preceremony.read_bytes_long_path(path) != payload:
            raise HistoricalIdentityCoverageError(
                "Existing historical coverage receipt has different bytes"
            )
        return "EXACT_EXISTING_RECEIPT_REUSED"
    preceremony.write_bytes_no_replace_long_path(path, payload)
    return "NEW_RECEIPT_PUBLISHED_NO_REPLACE"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write-receipt", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    receipt = build_receipt()
    disposition = "RECEIPT_NOT_WRITTEN"
    if args.write_receipt:
        disposition = publish_receipt(OUTPUT_PATH, receipt)
    print(
        receipt["status"],
        receipt["historical_v1_2_identity_union"]["unique_value_hash_count"],
        receipt["coverage"]["historical_v1_2_missing_count"],
        disposition,
    )


if __name__ == "__main__":
    main()
