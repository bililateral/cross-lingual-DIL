#!/usr/bin/env python3
"""Run or recover the one-shot Step28-v13 v1.12 seed ceremony.

The ceremony is the only process allowed to open a master seed.  It draws
exactly one 32-byte value for each split, derives capability-scoped children,
and publishes only commitments plus a train/development execution lock.
"""

from __future__ import annotations

import argparse
import errno
import hashlib
import json
import os
import shutil
import stat
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import step28_v13_v1_12_formal_common as formal
import step28_v13_v1_12_prelock_evidence as authorization
import step28_v13_v1_12_preceremony as preceremony


ROOT = Path(__file__).resolve().parents[1]
MASTER_DOCUMENT_VERSION = (
    "2026-08-03-step28-v13-v1-12-private-master-v1"
)
GENERATOR_DOCUMENT_VERSION = (
    "2026-08-03-step28-v13-v1-12-generator-capabilities-v1"
)
M1_DOCUMENT_VERSION = (
    "2026-08-03-step28-v13-v1-12-m1-capability-v1"
)
PRIVATE_MANIFEST_VERSION = (
    "2026-08-03-step28-v13-v1-12-seed-custody-manifest-v1"
)
PUBLIC_RECEIPT_VERSION = (
    "2026-08-03-step28-v13-v1-12-seed-ceremony-receipt-v1"
)
START_RECEIPT_VERSION = (
    "2026-08-03-step28-v13-v1-12-seed-ceremony-start-v1"
)
EXECUTION_LOCK_VERSION = (
    "2026-08-03-step28-v13-v1-12-train-development-lock-v1"
)


class SeedCeremonyError(ValueError):
    """Raised when the one-shot ceremony cannot safely publish."""

    def __init__(
        self, message: str, *, public_details: Mapping[str, Any] | None = None
    ) -> None:
        super().__init__(message)
        self.public_details = dict(public_details or {})


def _has_reparse_attribute(path: Path) -> bool:
    attributes = getattr(os.lstat(path), "st_file_attributes", 0)
    marker = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
    return bool(marker and attributes & marker)


def _require_plain_directory(path: Path, *, label: str) -> None:
    if path.is_symlink() or not path.is_dir() or _has_reparse_attribute(path):
        raise SeedCeremonyError(f"{label} is not a plain directory")


def _require_plain_file(path: Path, *, label: str) -> None:
    if path.is_symlink() or not path.is_file() or _has_reparse_attribute(path):
        raise SeedCeremonyError(f"{label} is not a plain regular file")


def _json_bytes(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _write_json_no_replace(path: Path, value: Mapping[str, Any]) -> None:
    preceremony.write_bytes_no_replace_long_path(path, _json_bytes(value))


def _repo_relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError as exc:
        raise SeedCeremonyError("Ceremony path escapes repository") from exc


def _pin_for_bytes(
    *, path: Path, payload: bytes, canonical_self_hash: str
) -> dict[str, Any]:
    return {
        "path": _repo_relative(path),
        "size_bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "canonical_self_hash": canonical_self_hash,
    }


def _pin_for_file(path: Path, *, include_self_hash: bool) -> dict[str, Any]:
    result: dict[str, Any] = {
        "path": _repo_relative(path),
        "size_bytes": preceremony.stat_long_path(path).st_size,
        "sha256": preceremony.sha256_file(path),
    }
    if include_self_hash:
        document = preceremony.load_json_strict(path)
        preceremony.validate_canonical_self_hash(
            document, label=f"private ceremony document {path.name}"
        )
        result["canonical_self_hash"] = document["canonical_self_hash"]
    return result


def _seed_start_receipt(
    *, prelock_path: Path, prelock: Mapping[str, Any]
) -> dict[str, Any]:
    producer_path = ROOT / "scripts" / "step28_v13_v1_12_seed_ceremony.py"
    return preceremony.with_canonical_self_hash(
        {
            "version": START_RECEIPT_VERSION,
            "status": "SEED_CEREMONY_STARTED_NO_REDRAW",
            "run_id": prelock["run_id"],
            "prelock": _pin_for_file(prelock_path, include_self_hash=True),
            "producer_path": _repo_relative(producer_path),
            "producer_sha256": preceremony.sha256_file(producer_path),
            "master_draw_count_at_start_receipt": 0,
            "raw_master_or_capability_present": False,
            "missing_stage_after_this_receipt_permanently_closes_run": True,
            "replacement_draw_or_seed_screening_forbidden": True,
        }
    )


def draw_one_shot_material(
    *,
    forbidden_master_commitments: frozenset[str] | set[str],
    random_bytes: Callable[[int], bytes] = os.urandom,
) -> dict[str, Any]:
    """Draw exactly four masters; collisions fail instead of being redrawn."""

    masters: dict[str, bytes] = {}
    for split in formal.SPLITS:
        raw = random_bytes(32)
        if type(raw) is not bytes or len(raw) != 32:
            raise SeedCeremonyError(
                f"Entropy provider returned invalid bytes for {split}"
            )
        masters[split] = raw
    master_commitments = {
        split: hashlib.sha256(raw).hexdigest() for split, raw in masters.items()
    }
    if (
        len(set(masters.values())) != len(formal.SPLITS)
        or len(set(master_commitments.values())) != len(formal.SPLITS)
        or set(master_commitments.values()) & set(forbidden_master_commitments)
    ):
        raise SeedCeremonyError(
            "One-shot master collision/forbidden commitment closes v1.12",
            public_details={"master_commitments": master_commitments},
        )
    capabilities = {
        split: formal.derive_capabilities(masters[split], split=split)
        for split in formal.SPLITS
    }
    raw_capabilities = [
        value
        for split in formal.SPLITS
        for namespace in ("generator", "m1")
        for value in capabilities[split][namespace].values()
    ]
    if len(raw_capabilities) != len(set(raw_capabilities)):
        raise SeedCeremonyError(
            "Derived capability collision closes v1.12",
            public_details={"master_commitments": master_commitments},
        )
    commitments = {
        split: formal.capability_commitments(capabilities[split])
        for split in formal.SPLITS
    }
    return {
        "masters": masters,
        "master_commitments": master_commitments,
        "capabilities": capabilities,
        "commitments": commitments,
        "master_draw_count": 4,
    }


def _build_private_documents(
    *, run_id: str, material: Mapping[str, Any]
) -> tuple[
    dict[str, dict[str, Any]],
    dict[str, dict[str, Any]],
    dict[str, dict[str, Any]],
]:
    master_documents: dict[str, dict[str, Any]] = {}
    generator_documents: dict[str, dict[str, Any]] = {}
    m1_documents: dict[str, dict[str, Any]] = {}
    for split in formal.SPLITS:
        master_documents[split] = preceremony.with_canonical_self_hash(
            {
                "version": MASTER_DOCUMENT_VERSION,
                "run_id": run_id,
                "split": split,
                "master_hex": material["masters"][split].hex(),
                "sha256_commitment": material["master_commitments"][split],
                "generator_or_model_mount_forbidden": True,
            }
        )
        generator_documents[split] = preceremony.with_canonical_self_hash(
            {
                "version": GENERATOR_DOCUMENT_VERSION,
                "run_id": run_id,
                "split": split,
                "generator_capabilities": material["capabilities"][split][
                    "generator"
                ],
                "generator_capability_commitments": material["commitments"][
                    split
                ]["generator"],
                "master_present": False,
                "m1_capability_present": False,
            }
        )
    for role in formal.M1_ROLES:
        value = material["capabilities"]["train"]["m1"][role]
        m1_documents[role] = preceremony.with_canonical_self_hash(
            {
                "version": M1_DOCUMENT_VERSION,
                "run_id": run_id,
                "split": "train",
                "role": role,
                "rewire_key_hex": value,
                "sha256_commitment": material["commitments"]["train"][
                    "m1"
                ][role],
                "master_present": False,
                "other_m1_capability_present": False,
            }
        )
    return master_documents, generator_documents, m1_documents


def _raw_member_paths() -> list[str]:
    return sorted(
        [
            *(f"masters/{split}.json" for split in formal.SPLITS),
            *(f"generator/{split}.json" for split in formal.SPLITS),
            *(f"m1/train/{role}.json" for role in formal.M1_ROLES),
        ],
        key=lambda value: value.encode("utf-8"),
    )


def _publish_or_verify(path: Path, payload: bytes) -> None:
    if preceremony.exists_long_path(path):
        _require_plain_file(path, label=f"existing public {path.name}")
        if preceremony.read_bytes_long_path(path) != payload:
            raise SeedCeremonyError(
                f"Existing public ceremony artifact differs: {path}"
            )
        return
    preceremony.write_bytes_no_replace_long_path(path, payload)


def _validate_private_bundle(
    *,
    bundle_root: Path,
    prelock_path: Path,
    validated_prelock: Mapping[str, Any],
    logical_bundle_root: Path | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    _require_plain_directory(bundle_root, label="seed custody bundle")
    expected = set(_raw_member_paths()) | {
        "private_manifest.json",
        "public_receipt_copy.json",
        "execution_lock_copy.json",
    }
    observed = {
        path.relative_to(bundle_root).as_posix()
        for path in preceremony.walk_files_long_path(bundle_root)
    }
    if observed != expected:
        raise SeedCeremonyError("Seed custody member set drift")
    for relative, label in (
        ("masters", "private masters directory"),
        ("generator", "private generator directory"),
        ("m1", "private M1 directory"),
        ("m1/train", "private train M1 directory"),
    ):
        _require_plain_directory(bundle_root / relative, label=label)
    prelock = validated_prelock["prelock"]
    masters: dict[str, bytes] = {}
    master_commitments: dict[str, str] = {}
    generator_documents: dict[str, dict[str, Any]] = {}
    all_commitments: dict[str, dict[str, Any]] = {}
    for split in formal.SPLITS:
        master_path = bundle_root / "masters" / f"{split}.json"
        generator_path = bundle_root / "generator" / f"{split}.json"
        for path, label in (
            (master_path, f"private master {split}"),
            (generator_path, f"private generator capabilities {split}"),
        ):
            _require_plain_file(path, label=label)
        master = preceremony.load_json_strict(master_path)
        generator = preceremony.load_json_strict(generator_path)
        preceremony.validate_canonical_self_hash(master, label=f"master {split}")
        preceremony.validate_canonical_self_hash(
            generator, label=f"generator capabilities {split}"
        )
        try:
            raw = bytes.fromhex(str(master["master_hex"]))
        except (KeyError, ValueError) as exc:
            raise SeedCeremonyError("Private master is malformed") from exc
        commitment = hashlib.sha256(raw).hexdigest()
        if (
            len(raw) != 32
            or master.get("version") != MASTER_DOCUMENT_VERSION
            or master.get("run_id") != prelock["run_id"]
            or master.get("split") != split
            or master.get("sha256_commitment") != commitment
            or master.get("generator_or_model_mount_forbidden") is not True
        ):
            raise SeedCeremonyError("Private master document drift")
        derived = formal.derive_capabilities(raw, split=split)
        if (
            generator.get("version") != GENERATOR_DOCUMENT_VERSION
            or generator.get("run_id") != prelock["run_id"]
            or generator.get("split") != split
            or generator.get("generator_capabilities")
            != derived["generator"]
            or generator.get("generator_capability_commitments")
            != formal.capability_commitments(derived)["generator"]
            or generator.get("master_present") is not False
            or generator.get("m1_capability_present") is not False
        ):
            raise SeedCeremonyError("Private generator capability drift")
        masters[split] = raw
        master_commitments[split] = commitment
        generator_documents[split] = generator
        all_commitments[split] = formal.capability_commitments(derived)
    if (
        len(set(masters.values())) != 4
        or set(master_commitments.values())
        & set(validated_prelock["baseline"]["forbidden_master_commitments"])
    ):
        raise SeedCeremonyError("Private master commitment closure failed")
    m1_commitments: dict[str, str] = {}
    for role in formal.M1_ROLES:
        path = bundle_root / "m1" / "train" / f"{role}.json"
        _require_plain_file(path, label=f"private M1 capability {role}")
        document = preceremony.load_json_strict(path)
        preceremony.validate_canonical_self_hash(
            document, label=f"private M1 capability {role}"
        )
        derived_value = formal.derive_capabilities(
            masters["train"], split="train"
        )["m1"][role]
        if (
            document.get("version") != M1_DOCUMENT_VERSION
            or document.get("run_id") != prelock["run_id"]
            or document.get("split") != "train"
            or document.get("role") != role
            or document.get("rewire_key_hex") != derived_value
            or document.get("sha256_commitment")
            != hashlib.sha256(bytes.fromhex(derived_value)).hexdigest()
            or document.get("master_present") is not False
            or document.get("other_m1_capability_present") is not False
        ):
            raise SeedCeremonyError("Private M1 capability drift")
        m1_commitments[role] = str(document["sha256_commitment"])

    manifest_path = bundle_root / "private_manifest.json"
    manifest = preceremony.load_json_strict(manifest_path)
    preceremony.validate_canonical_self_hash(
        manifest, label="seed custody private manifest"
    )
    expected_records = []
    for relative in _raw_member_paths():
        pin = _pin_for_file(
            bundle_root / relative, include_self_hash=True
        )
        if logical_bundle_root is not None:
            pin["path"] = _repo_relative(logical_bundle_root / relative)
        expected_records.append(pin)
    expected_records.sort(
        key=lambda row: str(row["path"]).encode("utf-8")
    )
    if (
        manifest.get("version") != PRIVATE_MANIFEST_VERSION
        or manifest.get("status") != "PASS_PRIVATE_SEED_CUSTODY"
        or manifest.get("run_id") != prelock["run_id"]
        or int(manifest.get("master_document_count", -1)) != 4
        or int(manifest.get("generator_document_count", -1)) != 4
        or int(manifest.get("m1_document_count", -1)) != 5
        or manifest.get("files") != expected_records
        or manifest.get("prelock_sha256")
        != preceremony.sha256_file(prelock_path)
    ):
        raise SeedCeremonyError("Private seed custody manifest drift")
    receipt = preceremony.load_json_strict(
        bundle_root / "public_receipt_copy.json"
    )
    lock = preceremony.load_json_strict(bundle_root / "execution_lock_copy.json")
    preceremony.validate_canonical_self_hash(
        receipt, label="private seed ceremony receipt copy"
    )
    preceremony.validate_canonical_self_hash(
        lock, label="private execution lock copy"
    )
    producer_path = ROOT / "scripts" / "step28_v13_v1_12_seed_ceremony.py"
    expected_receipt = preceremony.with_canonical_self_hash(
        {
            "version": PUBLIC_RECEIPT_VERSION,
            "status": "PASS_ONE_SHOT_SEED_CEREMONY",
            "run_id": prelock["run_id"],
            "prelock": _pin_for_file(prelock_path, include_self_hash=True),
            "ceremony_start_receipt": _pin_for_file(
                preceremony._repo_path(
                    str(
                        prelock["custody"][
                            "seed_ceremony_start_receipt_path"
                        ]
                    )
                ),
                include_self_hash=True,
            ),
            "producer_path": _repo_relative(producer_path),
            "producer_sha256": preceremony.sha256_file(producer_path),
            "master_draw_count": 4,
            "master_commitments": master_commitments,
            "generator_capability_commitments": {
                split: all_commitments[split]["generator"]
                for split in formal.SPLITS
            },
            "m1_capability_commitments": m1_commitments,
            "master_commitments_unique": True,
            "forbidden_master_commitment_intersection_count": 0,
            "one_draw_per_split_no_retry": True,
            "raw_master_or_capability_serialized_publicly": False,
            "master_mounted_to_generator": False,
            "master_mounted_to_model": False,
            "private_manifest_sha256": preceremony.sha256_file(manifest_path),
            "private_manifest_canonical_self_hash": manifest[
                "canonical_self_hash"
            ],
        }
    )
    if receipt != expected_receipt:
        raise SeedCeremonyError("Private ceremony receipt copy replay drift")

    generator_pins = {
        Path(str(record["path"])).stem: dict(record)
        for record in expected_records
        if "/generator/" in "/" + str(record["path"])
    }
    m1_pins = {
        Path(str(record["path"])).stem: dict(record)
        for record in expected_records
        if "/m1/train/" in "/" + str(record["path"])
    }
    receipt_target = preceremony._repo_path(
        str(prelock["custody"]["public_ceremony_receipt_path"])
    )
    expected_lock = preceremony.with_canonical_self_hash(
        {
            "version": EXECUTION_LOCK_VERSION,
            "status": "READY_FOR_TRAIN_DEVELOPMENT_GENERATION",
            "run_id": prelock["run_id"],
            "authorizations": {
                "formal_seed_ceremony": False,
                "formal_train_generation": True,
                "formal_development_generation": True,
                "formal_audit_a_generation": False,
                "formal_audit_b_generation": False,
                "model_training": False,
                "audit_truth_unsealing": False,
            },
            "prelock": _pin_for_file(prelock_path, include_self_hash=True),
            "ceremony_receipt": _pin_for_bytes(
                path=receipt_target,
                payload=_json_bytes(receipt),
                canonical_self_hash=receipt["canonical_self_hash"],
            ),
            "source_closure_canonical_sha256": prelock["source_closure"][
                "canonical_sha256"
            ],
            "master_commitments": master_commitments,
            "generator_capability_commitments": expected_receipt[
                "generator_capability_commitments"
            ],
            "m1_capability_commitments": m1_commitments,
            "private_generator_capability_files": generator_pins,
            "private_m1_capability_files": m1_pins,
            "formal_split_order": ["train", "development"],
            "seed_or_capability_replacement_forbidden": True,
        }
    )
    if lock != expected_lock:
        raise SeedCeremonyError("Private execution lock copy replay drift")
    return receipt, lock


def _build_and_stage_bundle(
    *,
    stage: Path,
    final_bundle: Path,
    prelock_path: Path,
    validated_prelock: Mapping[str, Any],
    material: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    prelock = validated_prelock["prelock"]
    run_id = str(prelock["run_id"])
    masters, generators, m1_documents = _build_private_documents(
        run_id=run_id, material=material
    )
    for split, document in masters.items():
        _write_json_no_replace(stage / "masters" / f"{split}.json", document)
    for split, document in generators.items():
        _write_json_no_replace(
            stage / "generator" / f"{split}.json", document
        )
    for role, document in m1_documents.items():
        _write_json_no_replace(
            stage / "m1" / "train" / f"{role}.json", document
        )

    raw_records: list[dict[str, Any]] = []
    generator_pins: dict[str, dict[str, Any]] = {}
    m1_pins: dict[str, dict[str, Any]] = {}
    for relative in _raw_member_paths():
        staged_path = stage / relative
        final_path = final_bundle / relative
        pin = _pin_for_file(staged_path, include_self_hash=True)
        pin["path"] = _repo_relative(final_path)
        raw_records.append(pin)
        if relative.startswith("generator/"):
            generator_pins[Path(relative).stem] = dict(pin)
        elif relative.startswith("m1/train/"):
            m1_pins[Path(relative).stem] = dict(pin)
    raw_records.sort(key=lambda row: str(row["path"]).encode("utf-8"))
    private_manifest = preceremony.with_canonical_self_hash(
        {
            "version": PRIVATE_MANIFEST_VERSION,
            "status": "PASS_PRIVATE_SEED_CUSTODY",
            "run_id": run_id,
            "prelock_sha256": preceremony.sha256_file(prelock_path),
            "master_document_count": 4,
            "generator_document_count": 4,
            "m1_document_count": 5,
            "files": raw_records,
        }
    )
    manifest_path = stage / "private_manifest.json"
    _write_json_no_replace(manifest_path, private_manifest)

    producer_path = ROOT / "scripts" / "step28_v13_v1_12_seed_ceremony.py"
    receipt = preceremony.with_canonical_self_hash(
        {
            "version": PUBLIC_RECEIPT_VERSION,
            "status": "PASS_ONE_SHOT_SEED_CEREMONY",
            "run_id": run_id,
            "prelock": _pin_for_file(prelock_path, include_self_hash=True),
            "ceremony_start_receipt": _pin_for_file(
                preceremony._repo_path(
                    str(
                        prelock["custody"][
                            "seed_ceremony_start_receipt_path"
                        ]
                    )
                ),
                include_self_hash=True,
            ),
            "producer_path": _repo_relative(producer_path),
            "producer_sha256": preceremony.sha256_file(producer_path),
            "master_draw_count": int(material["master_draw_count"]),
            "master_commitments": dict(material["master_commitments"]),
            "generator_capability_commitments": {
                split: material["commitments"][split]["generator"]
                for split in formal.SPLITS
            },
            "m1_capability_commitments": material["commitments"]["train"][
                "m1"
            ],
            "master_commitments_unique": True,
            "forbidden_master_commitment_intersection_count": 0,
            "one_draw_per_split_no_retry": True,
            "raw_master_or_capability_serialized_publicly": False,
            "master_mounted_to_generator": False,
            "master_mounted_to_model": False,
            "private_manifest_sha256": preceremony.sha256_file(manifest_path),
            "private_manifest_canonical_self_hash": private_manifest[
                "canonical_self_hash"
            ],
        }
    )
    receipt_target = preceremony._repo_path(
        str(prelock["custody"]["public_ceremony_receipt_path"])
    )
    receipt_payload = _json_bytes(receipt)
    receipt_pin = _pin_for_bytes(
        path=receipt_target,
        payload=receipt_payload,
        canonical_self_hash=receipt["canonical_self_hash"],
    )
    lock = preceremony.with_canonical_self_hash(
        {
            "version": EXECUTION_LOCK_VERSION,
            "status": "READY_FOR_TRAIN_DEVELOPMENT_GENERATION",
            "run_id": run_id,
            "authorizations": {
                "formal_seed_ceremony": False,
                "formal_train_generation": True,
                "formal_development_generation": True,
                "formal_audit_a_generation": False,
                "formal_audit_b_generation": False,
                "model_training": False,
                "audit_truth_unsealing": False,
            },
            "prelock": _pin_for_file(prelock_path, include_self_hash=True),
            "ceremony_receipt": receipt_pin,
            "source_closure_canonical_sha256": prelock["source_closure"][
                "canonical_sha256"
            ],
            "master_commitments": dict(material["master_commitments"]),
            "generator_capability_commitments": receipt[
                "generator_capability_commitments"
            ],
            "m1_capability_commitments": receipt[
                "m1_capability_commitments"
            ],
            "private_generator_capability_files": generator_pins,
            "private_m1_capability_files": m1_pins,
            "formal_split_order": ["train", "development"],
            "seed_or_capability_replacement_forbidden": True,
        }
    )
    _write_json_no_replace(stage / "public_receipt_copy.json", receipt)
    _write_json_no_replace(stage / "execution_lock_copy.json", lock)
    return receipt, lock


def _ceremony_paths(prelock: Mapping[str, Any]) -> dict[str, Path]:
    custody = prelock["custody"]
    return {
        "private_bundle": preceremony._repo_path(
            str(custody["private_seed_bundle_root"])
        ),
        "private_stage": preceremony._repo_path(
            str(custody["private_seed_stage_root"])
        ),
        "start_receipt": preceremony._repo_path(
            str(custody["seed_ceremony_start_receipt_path"])
        ),
        "public_receipt": preceremony._repo_path(
            str(custody["public_ceremony_receipt_path"])
        ),
        "execution_lock": preceremony._repo_path(
            str(custody["train_development_execution_lock_path"])
        ),
        "failure_receipt": preceremony._repo_path(
            str(custody["permanent_failure_receipt_path"])
        ),
    }


def _publish_permanent_failure(
    *,
    path: Path,
    prelock_path: Path,
    prelock: Mapping[str, Any],
    error: BaseException,
    master_draw_count: int,
) -> None:
    details = (
        error.public_details
        if isinstance(error, SeedCeremonyError)
        else {}
    )
    receipt = preceremony.with_canonical_self_hash(
        {
            "version": (
                "2026-08-03-step28-v13-v1-12-permanent-failure-v1"
            ),
            "status": "FAIL_V1_12_PERMANENTLY_CLOSED_NO_RETRY",
            "run_id": prelock["run_id"],
            "phase": "seed_ceremony",
            "error_type": type(error).__name__,
            "prelock_sha256": preceremony.sha256_file(prelock_path),
            "master_draw_count_before_failure": master_draw_count,
            "master_commitments": details.get("master_commitments", {}),
            "raw_master_or_capability_persisted_publicly": False,
            "formal_rows_produced": 0,
            "replacement_draw_or_rerun_forbidden": True,
            "failed_private_stage_cleanup_required_after_receipt": True,
        }
    )
    _publish_or_verify(path, _json_bytes(receipt))


@contextmanager
def _exclusive_ceremony_invocation(prelock_path: Path) -> Iterator[None]:
    """Hold one process-lifetime lock keyed by the immutable prelock path."""

    _require_plain_file(prelock_path, label="formal ceremony prelock")
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.CreateMutexW.argtypes = (
            wintypes.LPVOID,
            wintypes.BOOL,
            wintypes.LPCWSTR,
        )
        kernel32.CreateMutexW.restype = wintypes.HANDLE
        kernel32.WaitForSingleObject.argtypes = (
            wintypes.HANDLE,
            wintypes.DWORD,
        )
        kernel32.WaitForSingleObject.restype = wintypes.DWORD
        kernel32.ReleaseMutex.argtypes = (wintypes.HANDLE,)
        kernel32.ReleaseMutex.restype = wintypes.BOOL
        kernel32.CloseHandle.argtypes = (wintypes.HANDLE,)
        kernel32.CloseHandle.restype = wintypes.BOOL
        mutex_key = hashlib.sha256(
            str(prelock_path.resolve(strict=True)).casefold().encode("utf-8")
        ).hexdigest()
        handle = kernel32.CreateMutexW(
            None,
            False,
            f"Global\\Step28V13V112SeedCeremony-{mutex_key}",
        )
        if not handle:
            raise SeedCeremonyError(
                "unable to create ceremony invocation mutex"
            )
        acquired = False
        try:
            wait_result = int(kernel32.WaitForSingleObject(handle, 0))
            if wait_result == 258:  # WAIT_TIMEOUT
                raise SeedCeremonyError(
                    "ceremony invocation already active"
                )
            if wait_result not in {0, 128}:  # OBJECT_0 or ABANDONED_0
                raise SeedCeremonyError(
                    "unable to acquire ceremony invocation mutex"
                )
            acquired = True
            yield
        finally:
            try:
                if acquired and not kernel32.ReleaseMutex(handle):
                    raise SeedCeremonyError(
                        "unable to release ceremony invocation mutex"
                    )
            finally:
                kernel32.CloseHandle(handle)
        return

    filesystem_path = preceremony._filesystem_path(prelock_path)
    descriptor = os.open(filesystem_path, os.O_RDONLY)
    locked = False
    try:
        import fcntl

        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            if exc.errno in {errno.EACCES, errno.EAGAIN}:
                raise SeedCeremonyError(
                    "ceremony invocation already active"
                ) from exc
            raise SeedCeremonyError(
                "unable to acquire ceremony invocation lock"
            ) from exc
        locked = True
        yield
    finally:
        try:
            if locked:
                import fcntl

                fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)


def _initialize_locked(
    prelock_path: Path = formal.DEFAULT_PRELOCK_PATH,
    *,
    random_bytes: Callable[[int], bytes] = os.urandom,
) -> dict[str, Any]:
    """Create or recover the immutable ceremony without replacement draws."""

    validated = formal.load_and_validate_prelock(prelock_path)
    prelock = validated["prelock"]
    paths = _ceremony_paths(prelock)
    start_exists = preceremony.exists_long_path(paths["start_receipt"])
    authorization.validate_authorization_prelock_document(
        prelock,
        legacy_validation=validated,
        dereference_waiver_state=not start_exists,
    )
    final_bundle = paths["private_bundle"]
    stage = paths["private_stage"]
    start_target = paths["start_receipt"]
    receipt_target = paths["public_receipt"]
    lock_target = paths["execution_lock"]
    failure_target = paths["failure_receipt"]

    if preceremony.exists_long_path(failure_target):
        raise SeedCeremonyError(
            "Permanent v1.12 failure receipt exists; seed redraw is forbidden"
        )

    expected_start = _seed_start_receipt(
        prelock_path=prelock_path, prelock=prelock
    )
    final_exists = preceremony.exists_long_path(final_bundle)
    stage_exists = preceremony.exists_long_path(stage)
    receipt_exists = preceremony.exists_long_path(receipt_target)
    lock_exists = preceremony.exists_long_path(lock_target)
    prior_state_exists = any(
        (final_exists, stage_exists, receipt_exists, lock_exists)
    )
    if prior_state_exists and not start_exists:
        error = SeedCeremonyError(
            "Ceremony state exists without its one-shot start receipt"
        )
        _publish_permanent_failure(
            path=failure_target,
            prelock_path=prelock_path,
            prelock=prelock,
            error=error,
            master_draw_count=0,
        )
        raise error
    if start_exists:
        try:
            _publish_or_verify(start_target, _json_bytes(expected_start))
        except Exception as exc:
            _publish_permanent_failure(
                path=failure_target,
                prelock_path=prelock_path,
                prelock=prelock,
                error=exc,
                master_draw_count=0,
            )
            raise
        if not prior_state_exists:
            error = SeedCeremonyError(
                "Ceremony start receipt exists without recoverable state"
            )
            _publish_permanent_failure(
                path=failure_target,
                prelock_path=prelock_path,
                prelock=prelock,
                error=error,
                master_draw_count=0,
            )
            raise error
    else:
        _publish_or_verify(start_target, _json_bytes(expected_start))
        try:
            revalidated = formal.load_and_validate_prelock(prelock_path)
            if revalidated["prelock"] != prelock:
                raise SeedCeremonyError(
                    "Formal prelock changed after durable ceremony start"
                )
            authorization.validate_authorization_prelock_document(
                prelock,
                legacy_validation=revalidated,
                dereference_waiver_state=True,
            )
            validated = revalidated
        except Exception as exc:
            _publish_permanent_failure(
                path=failure_target,
                prelock_path=prelock_path,
                prelock=prelock,
                error=exc,
                master_draw_count=0,
            )
            raise

    if final_exists:
        try:
            receipt, lock = _validate_private_bundle(
                bundle_root=final_bundle,
                prelock_path=prelock_path,
                validated_prelock=validated,
            )
            _publish_or_verify(receipt_target, _json_bytes(receipt))
            _publish_or_verify(lock_target, _json_bytes(lock))
            formal.load_and_validate_execution_lock(lock_target)
        except Exception as exc:
            _publish_permanent_failure(
                path=failure_target,
                prelock_path=prelock_path,
                prelock=prelock,
                error=exc,
                master_draw_count=0,
            )
            raise
        return {
            "status": "PASS_RECOVERED_EXISTING_ONE_SHOT_CEREMONY",
            "raw_master_or_capability_returned": False,
            "master_draw_count_this_invocation": 0,
            "ceremony_receipt_sha256": preceremony.sha256_file(receipt_target),
            "execution_lock_sha256": preceremony.sha256_file(lock_target),
        }
    if receipt_exists or lock_exists:
        error = SeedCeremonyError(
            "Public ceremony artifact exists without recoverable private custody"
        )
        _publish_permanent_failure(
            path=failure_target,
            prelock_path=prelock_path,
            prelock=prelock,
            error=error,
            master_draw_count=0,
        )
        raise error
    if stage_exists:
        try:
            receipt, lock = _validate_private_bundle(
                bundle_root=stage,
                prelock_path=prelock_path,
                validated_prelock=validated,
                logical_bundle_root=final_bundle,
            )
        except Exception as exc:
            _publish_permanent_failure(
                path=failure_target,
                prelock_path=prelock_path,
                prelock=prelock,
                error=exc,
                master_draw_count=0,
            )
            shutil.rmtree(preceremony._filesystem_path(stage))
            raise SeedCeremonyError(
                "Incomplete prior seed stage closes v1.12; never redraw"
            ) from exc
        os.rename(
            preceremony._filesystem_path(stage),
            preceremony._filesystem_path(final_bundle),
        )
        _publish_or_verify(receipt_target, _json_bytes(receipt))
        _publish_or_verify(lock_target, _json_bytes(lock))
        formal.load_and_validate_execution_lock(lock_target)
        return {
            "status": "PASS_RECOVERED_COMPLETE_ONE_SHOT_STAGE",
            "raw_master_or_capability_returned": False,
            "master_draw_count_this_invocation": 0,
            "ceremony_receipt_sha256": preceremony.sha256_file(receipt_target),
            "execution_lock_sha256": preceremony.sha256_file(lock_target),
        }

    master_draw_count = 0
    material: dict[str, Any] | None = None

    def counted_random_bytes(size: int) -> bytes:
        nonlocal master_draw_count
        master_draw_count += 1
        return random_bytes(size)

    try:
        os.makedirs(
            preceremony._filesystem_path(stage.parent), exist_ok=True
        )
        _require_plain_directory(
            stage.parent, label="private ceremony stage parent"
        )
        os.mkdir(preceremony._filesystem_path(stage))
        material = draw_one_shot_material(
            forbidden_master_commitments=validated["baseline"][
                "forbidden_master_commitments"
            ],
            random_bytes=counted_random_bytes,
        )
        receipt, lock = _build_and_stage_bundle(
            stage=stage,
            final_bundle=final_bundle,
            prelock_path=prelock_path,
            validated_prelock=validated,
            material=material,
        )
        _validate_private_bundle(
            bundle_root=stage,
            prelock_path=prelock_path,
            validated_prelock=validated,
            logical_bundle_root=final_bundle,
        )
        if preceremony.exists_long_path(final_bundle):
            raise SeedCeremonyError("Private seed bundle appeared during ceremony")
        os.rename(
            preceremony._filesystem_path(stage),
            preceremony._filesystem_path(final_bundle),
        )
    except Exception as exc:
        try:
            _publish_permanent_failure(
                path=failure_target,
                prelock_path=prelock_path,
                prelock=prelock,
                error=exc,
                master_draw_count=master_draw_count,
            )
        except Exception:
            # Retain the stage as the no-redraw marker if the public failure
            # receipt itself cannot be published.
            raise
        if preceremony.exists_long_path(stage):
            shutil.rmtree(preceremony._filesystem_path(stage))
        raise
    _publish_or_verify(receipt_target, _json_bytes(receipt))
    _publish_or_verify(lock_target, _json_bytes(lock))
    formal.load_and_validate_execution_lock(lock_target)
    return {
        "status": "PASS_NEW_ONE_SHOT_SEED_CEREMONY",
        "raw_master_or_capability_returned": False,
        "master_draw_count_this_invocation": 4,
        "ceremony_receipt_sha256": preceremony.sha256_file(receipt_target),
        "execution_lock_sha256": preceremony.sha256_file(lock_target),
    }


def initialize(
    prelock_path: Path = formal.DEFAULT_PRELOCK_PATH,
    *,
    random_bytes: Callable[[int], bytes] = os.urandom,
) -> dict[str, Any]:
    """Serialize the complete one-shot ceremony and its recovery state."""

    with _exclusive_ceremony_invocation(prelock_path):
        return _initialize_locked(
            prelock_path=prelock_path,
            random_bytes=random_bytes,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--prelock", type=Path, default=formal.DEFAULT_PRELOCK_PATH
    )
    return parser.parse_args()


def main() -> None:
    result = initialize(parse_args().prelock.resolve())
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))


if __name__ == "__main__":
    main()
