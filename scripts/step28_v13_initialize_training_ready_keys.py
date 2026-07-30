#!/usr/bin/env python3
"""Initialize four split-private Step28-v13 training-ready structure keys.

The command is a one-shot ceremony.  It publishes the private key directory
with an atomic no-replace rename and writes a public, key-free commitment
receipt.  If interruption occurs after the private directory is published,
rerunning recovers only the identical public receipt from its private copy.
"""

from __future__ import annotations

import argparse
import os
import stat
import uuid
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Callable

import step28_v13_build_training_ready_dataset as builder
import step28_v13_common as common


SPLITS = builder.SPLITS
RECEIPT_VERSION = (
    "2026-07-29-step28-v13-training-ready-key-ceremony-v1"
)
KEY_DOCUMENT_VERSION = (
    "2026-07-29-step28-v13-training-ready-split-key-v1"
)
PUBLIC_RECEIPT_FILENAME = "ceremony_receipt.public_copy.json"
PUBLIC_RECEIPT_KEYS = {
    "version",
    "status",
    "run_id",
    "pre_ceremony_overlay_path",
    "pre_ceremony_overlay_sha256",
    "initializer_path",
    "initializer_sha256",
    "commitments",
    "commitments_unique",
    "forbidden_commitment_intersection_count",
    "one_split_key_per_file",
    "raw_structure_keys_serialized",
    "os_custody_attested",
    "canonical_self_hash",
}


def _public_key_commitments(base: Mapping[str, Any]) -> set[str]:
    commitments: set[str] = set()
    for namespace in ("formal", "development_smoke"):
        stream = base["randomness"][namespace]
        values = [
            stream["id_namespace_key_hex"],
            stream["id_key_hex"],
            stream["identity_value_key_hex"],
            stream["text_key_hex"],
            stream["candidate_key_hex"],
            stream["query_key_hex"],
            *stream["rewire_key_hexes"],
        ]
        if namespace == "development_smoke":
            values.append(stream["structure_key_hex"])
        for value in values:
            commitments.add(
                common.sha256_bytes(bytes.fromhex(str(value)))
            )
    commitments.add(
        common.sha256_bytes(
            bytes.fromhex(builder.DESIGN_ONLY_STRUCTURE_KEY_HEX)
        )
    )
    return commitments


def _forbidden_commitments(
    overlay: Mapping[str, Any],
) -> set[str]:
    base = builder._load_pinned_base(overlay)
    forbidden = _public_key_commitments(base)
    compromised = base["randomness"]["formal"][
        "label_bearing_structure_keys"
    ]["compromised_draft_key_commitments_forbidden"]
    if (
        not isinstance(compromised, list)
        or any(
            not isinstance(value, str)
            or len(value) != 64
            or value.lower() != value
            for value in compromised
        )
    ):
        raise common.ContractError(
            "Compromised structure-key commitment registry drift"
        )
    forbidden.update(compromised)
    return forbidden


def _has_reparse_attribute(path: Path) -> bool:
    attributes = getattr(os.lstat(path), "st_file_attributes", 0)
    marker = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
    return bool(marker and attributes & marker)


def _require_plain_directory(path: Path, *, label: str) -> None:
    if (
        path.is_symlink()
        or not path.is_dir()
        or _has_reparse_attribute(path)
    ):
        raise common.ContractError(
            f"{label} is not a plain non-reparse directory"
        )


def _require_plain_file(path: Path, *, label: str) -> None:
    if (
        path.is_symlink()
        or not path.is_file()
        or _has_reparse_attribute(path)
    ):
        raise common.ContractError(
            f"{label} is not a plain non-reparse regular file"
        )


def _generate_documents(
    *,
    run_id: str,
    forbidden_commitments: set[str],
    random_bytes: Callable[[int], bytes] = os.urandom,
) -> tuple[dict[str, dict[str, str]], dict[str, str]]:
    documents: dict[str, dict[str, str]] = {}
    commitments: dict[str, str] = {}
    for split in SPLITS:
        for _attempt in range(128):
            raw = random_bytes(32)
            if type(raw) is not bytes or len(raw) != 32:
                raise common.ContractError(
                    "Structure-key entropy provider returned invalid bytes"
                )
            commitment = common.sha256_bytes(raw)
            if (
                commitment in forbidden_commitments
                or commitment in commitments.values()
            ):
                continue
            key_hex = raw.hex()
            documents[split] = {
                "version": KEY_DOCUMENT_VERSION,
                "run_id": run_id,
                "split": split,
                "key_hex": key_hex,
                "sha256_commitment": commitment,
            }
            commitments[split] = commitment
            break
        else:
            raise common.ContractError(
                "Could not draw a collision-free split structure key"
            )
    return documents, commitments


def _validate_private_bundle(
    *,
    directory: Path,
    overlay: Mapping[str, Any],
    overlay_path: Path,
) -> tuple[dict[str, str], dict[str, Any]]:
    custody = overlay["private_structure_key_custody"]
    _require_plain_directory(
        directory,
        label="Private structure-key bundle",
    )
    filenames = {
        split: str(custody["key_filename_pattern"]).format(split=split)
        for split in SPLITS
    }
    expected_members = set(filenames.values()) | {
        PUBLIC_RECEIPT_FILENAME
    }
    observed_members = {path.name for path in directory.iterdir()}
    if observed_members != expected_members:
        raise common.ContractError(
            "Private structure-key bundle member set drift"
        )
    commitments: dict[str, str] = {}
    for split in SPLITS:
        path = directory / filenames[split]
        _require_plain_file(
            path,
            label=f"Private structure-key document {split}",
        )
        document = common.load_json(path)
        if (
            set(document)
            != {
                "version",
                "run_id",
                "split",
                "key_hex",
                "sha256_commitment",
            }
            or document["version"] != KEY_DOCUMENT_VERSION
            or document["run_id"] != overlay["run_id"]
            or document["split"] != split
        ):
            raise common.ContractError(
                f"Private structure-key schema drift: {split}"
            )
        try:
            raw = bytes.fromhex(str(document["key_hex"]))
        except ValueError as exc:
            raise common.ContractError(
                f"Private structure key is not hex: {split}"
            ) from exc
        commitment = common.sha256_bytes(raw)
        if (
            len(raw) != 32
            or str(document["key_hex"]) != raw.hex()
            or document["sha256_commitment"] != commitment
        ):
            raise common.ContractError(
                f"Private structure-key commitment mismatch: {split}"
            )
        commitments[split] = commitment
    if len(set(commitments.values())) != len(SPLITS):
        raise common.ContractError(
            "Private structure-key commitments are not unique"
        )
    forbidden = _forbidden_commitments(overlay)
    intersection = set(commitments.values()) & forbidden
    if intersection:
        raise common.ContractError(
            "Private structure-key commitment is forbidden"
        )
    receipt_path = directory / PUBLIC_RECEIPT_FILENAME
    _require_plain_file(
        receipt_path,
        label="Private ceremony receipt copy",
    )
    receipt = common.load_json(receipt_path)
    initializer_path = (
        common.ROOT
        / "scripts"
        / "step28_v13_initialize_training_ready_keys.py"
    )
    expected_overlay_relative = overlay_path.relative_to(
        common.ROOT
    ).as_posix()
    if (
        set(receipt) != PUBLIC_RECEIPT_KEYS
        or receipt.get("version") != RECEIPT_VERSION
        or receipt.get("status")
        != "PASS_SPLIT_PRIVATE_KEY_CEREMONY"
        or receipt.get("run_id") != overlay["run_id"]
        or receipt.get("pre_ceremony_overlay_path")
        != expected_overlay_relative
        or receipt.get("pre_ceremony_overlay_sha256")
        != common.sha256_file(overlay_path)
        or receipt.get("initializer_path")
        != "scripts/step28_v13_initialize_training_ready_keys.py"
        or receipt.get("initializer_sha256")
        != common.sha256_file(initializer_path)
        or receipt.get("commitments") != commitments
        or receipt.get("commitments_unique") is not True
        or int(
            receipt.get(
                "forbidden_commitment_intersection_count",
                -1,
            )
        )
        != 0
        or receipt.get("one_split_key_per_file") is not True
        or receipt.get("raw_structure_keys_serialized") is not False
        or receipt.get("os_custody_attested") is not False
        or receipt.get("canonical_self_hash")
        != common.canonical_sha256(
            {
                key: value
                for key, value in receipt.items()
                if key != "canonical_self_hash"
            }
        )
    ):
        raise common.ContractError("Private ceremony receipt copy drift")
    return commitments, receipt


def _publish_public_receipt(
    receipt_path: Path,
    receipt: Mapping[str, Any],
) -> None:
    if receipt_path.exists():
        _require_plain_file(
            receipt_path,
            label="Public ceremony receipt",
        )
        observed = common.load_json(receipt_path)
        if (
            common.canonical_json_bytes(observed)
            != common.canonical_json_bytes(receipt)
        ):
            raise common.ContractError(
                "Existing public ceremony receipt has different bytes"
            )
        return
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    _require_plain_directory(
        receipt_path.parent,
        label="Public ceremony receipt parent",
    )
    temporary = (
        receipt_path.parent / f".receipt-{uuid.uuid4().hex[:10]}.tmp"
    )
    common.write_json(temporary, dict(receipt))
    common.atomic_rename_no_replace(temporary, receipt_path)


def initialize(overlay_path: Path) -> dict[str, Any]:
    overlay = builder.load_overlay(
        overlay_path,
        require_generation_frozen=False,
    )
    if (
        overlay.get("status") != "READY_FOR_KEY_CEREMONY"
        or overlay.get("generation_enabled") is not False
        or overlay.get("self_sha256")
        != builder._canonical_self_hash(overlay)
    ):
        raise common.ContractError(
            "Overlay is not frozen at READY_FOR_KEY_CEREMONY"
        )
    custody = overlay["private_structure_key_custody"]
    if (
        custody.get("one_split_key_per_file") is not True
        or custody.get("generator_process_reads_exactly_one_split_key")
        is not True
        or custody.get("key_files_must_not_be_model_mounted") is not True
        or custody.get("key_files_must_not_be_committed") is not True
        or any(custody["commitments"].get(split) is not None for split in SPLITS)
        or custody.get("ceremony_receipt") is not None
    ):
        raise common.ContractError(
            "Pre-ceremony custody registry is not empty and exact"
        )
    key_directory = common.repo_path(str(custody["key_directory"]))
    receipt_path = common.repo_path(str(custody["ceremony_receipt_path"]))
    if key_directory.exists():
        commitments, receipt = _validate_private_bundle(
            directory=key_directory,
            overlay=overlay,
            overlay_path=overlay_path,
        )
        _publish_public_receipt(receipt_path, receipt)
        return {
            "status": "PASS_RECOVERED_OR_VERIFIED_EXISTING_CEREMONY",
            "commitments": commitments,
            "public_receipt_path": receipt_path.relative_to(
                common.ROOT
            ).as_posix(),
            "raw_keys_returned": False,
        }
    if receipt_path.exists():
        raise common.ContractError(
            "Public ceremony receipt exists without its private key bundle"
        )
    key_directory.parent.mkdir(parents=True, exist_ok=True)
    _require_plain_directory(
        key_directory.parent,
        label="Private structure-key custody parent",
    )
    stale_stages = [
        path
        for path in key_directory.parent.iterdir()
        if path.name.startswith(".keys-")
    ]
    if stale_stages:
        raise common.ContractError(
            "A prior key-ceremony stage exists; do not draw replacement "
            "keys until it is audited and recovered or declared invalid"
        )
    forbidden = _forbidden_commitments(overlay)
    documents, commitments = _generate_documents(
        run_id=str(overlay["run_id"]),
        forbidden_commitments=forbidden,
    )
    source_path = (
        common.ROOT
        / "scripts"
        / "step28_v13_initialize_training_ready_keys.py"
    )
    receipt: dict[str, Any] = {
        "version": RECEIPT_VERSION,
        "status": "PASS_SPLIT_PRIVATE_KEY_CEREMONY",
        "run_id": overlay["run_id"],
        "pre_ceremony_overlay_path": overlay_path.relative_to(
            common.ROOT
        ).as_posix(),
        "pre_ceremony_overlay_sha256": common.sha256_file(overlay_path),
        "initializer_path": (
            "scripts/step28_v13_initialize_training_ready_keys.py"
        ),
        "initializer_sha256": common.sha256_file(source_path),
        "commitments": commitments,
        "commitments_unique": len(set(commitments.values())) == 4,
        "forbidden_commitment_intersection_count": 0,
        "one_split_key_per_file": True,
        "raw_structure_keys_serialized": False,
        "os_custody_attested": False,
    }
    receipt["canonical_self_hash"] = common.canonical_sha256(receipt)
    stage = (
        key_directory.parent / f".keys-{uuid.uuid4().hex[:10]}"
    )
    stage.mkdir(parents=False)
    try:
        for split, document in documents.items():
            filename = str(custody["key_filename_pattern"]).format(
                split=split
            )
            common.write_json(stage / filename, document)
        common.write_json(
            stage / PUBLIC_RECEIPT_FILENAME,
            receipt,
        )
        replayed, replayed_receipt = _validate_private_bundle(
            directory=stage,
            overlay=overlay,
            overlay_path=overlay_path,
        )
        if (
            replayed != commitments
            or common.canonical_json_bytes(replayed_receipt)
            != common.canonical_json_bytes(receipt)
        ):
            raise common.ContractError(
                "Staged private key ceremony replay failed"
            )
        common.atomic_rename_no_replace(stage, key_directory)
    except Exception:
        if stage.exists():
            common.write_json(
                stage / "FAILURE.json",
                {
                    "status": "INVALID_KEY_CEREMONY_STAGE_DO_NOT_USE",
                    "raw_keys_may_be_present": True,
                },
            )
        raise
    _publish_public_receipt(receipt_path, receipt)
    _validate_private_bundle(
        directory=key_directory,
        overlay=overlay,
        overlay_path=overlay_path,
    )
    return {
        "status": "PASS_NEW_SPLIT_PRIVATE_KEY_CEREMONY",
        "commitments": commitments,
        "public_receipt_path": receipt_path.relative_to(
            common.ROOT
        ).as_posix(),
        "raw_keys_returned": False,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--overlay",
        type=Path,
        default=builder.DEFAULT_OVERLAY,
    )
    return parser.parse_args()


def main() -> None:
    result = initialize(parse_args().overlay.resolve())
    print(common.canonical_json_bytes(result).decode("utf-8"))


if __name__ == "__main__":
    main()
