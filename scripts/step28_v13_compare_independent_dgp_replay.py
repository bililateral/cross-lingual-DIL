#!/usr/bin/env python3
"""Compare Step 28-v13 producer and independent typed projections.

This entrypoint is development-only.  It accepts no structure key and enforces
the minimal projection schema, but the development process itself is not an OS
sandbox: a caller that controls its path arguments could make it open a
misnamed file before schema rejection.  Formal comparison remains disabled
until an allow-listed custody launcher and immutable parent seals exist.
"""

from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import os
import uuid
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import step28_v13_independent_dgp_comparator as comparator


POLICY_VERSION = (
    "2026-07-29-step28-v13-synthetic-chinese-dataset-v13-draft"
)
REPLAY_RECEIPT_VERSION = (
    "2026-07-28-step28-v13-independent-replay-receipt-v2-draft"
)
PRODUCER_MANIFEST_VERSION = (
    "2026-07-28-step28-v13-producer-typed-dgp-"
    "projection-manifest-v1-draft"
)
AGGREGATE_VERSION = (
    "2026-07-28-step28-v13-dgp-comparison-receipt-v3-draft"
)
SPLITS = ("train", "development", "audit_a", "audit_b")
FIELD_SEPARATOR = b"\x1f"


class ComparatorLauncherError(ValueError):
    """Fail-closed input or development-custody error."""


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _reject_duplicate_pairs(
    pairs: list[tuple[str, Any]],
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for key, value in pairs:
        if key in output:
            raise ComparatorLauncherError(
                f"DGP_COMPARATOR_DUPLICATE_JSON_KEY:{key}"
            )
        output[key] = value
    return output


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(
            handle,
            object_pairs_hook=_reject_duplicate_pairs,
        )
    if not isinstance(value, dict):
        raise ComparatorLauncherError(
            "DGP_COMPARATOR_JSON_OBJECT_REQUIRED"
        )
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                raise ComparatorLauncherError(
                    "DGP_COMPARATOR_EMPTY_JSONL_ROW:"
                    f"{path.name}:{line_number}"
                )
            value = json.loads(
                line,
                object_pairs_hook=_reject_duplicate_pairs,
            )
            if not isinstance(value, dict):
                raise ComparatorLauncherError(
                    "DGP_COMPARATOR_JSONL_OBJECT_REQUIRED:"
                    f"{path.name}:{line_number}"
                )
            output.append(value)
    return output


def _verify_self_hash(
    value: Mapping[str, Any],
    *,
    label: str,
) -> None:
    claimed = value.get("canonical_self_hash")
    payload = dict(value)
    payload.pop("canonical_self_hash", None)
    if (
        not isinstance(claimed, str)
        or len(claimed) != 64
        or _canonical_sha256(payload) != claimed
    ):
        raise ComparatorLauncherError(
            f"DGP_COMPARATOR_PARENT_SELF_HASH_INVALID:{label}"
        )


def _key_bytes(value: Any, *, label: str) -> bytes:
    text = str(value)
    if len(text) != 64 or text != text.lower():
        raise ComparatorLauncherError(
            f"DGP_COMPARATOR_PUBLIC_KEY_ENCODING_INVALID:{label}"
        )
    try:
        raw = bytes.fromhex(text)
    except ValueError as exc:
        raise ComparatorLauncherError(
            f"DGP_COMPARATOR_PUBLIC_KEY_ENCODING_INVALID:{label}"
        ) from exc
    if len(raw) != 32:
        raise ComparatorLauncherError(
            f"DGP_COMPARATOR_PUBLIC_KEY_ENCODING_INVALID:{label}"
        )
    return raw


def _registered_world_uids(
    policy: Mapping[str, Any],
    *,
    mode: str,
    split: str,
) -> list[str]:
    if policy.get("version") != POLICY_VERSION:
        raise ComparatorLauncherError(
            "DGP_COMPARATOR_POLICY_VERSION_INVALID"
        )
    counts = policy["modes"][mode]["world_counts"]
    if not isinstance(counts, Mapping) or set(counts) != set(SPLITS):
        raise ComparatorLauncherError(
            "DGP_COMPARATOR_WORLD_COUNT_SCHEMA_INVALID"
        )
    exact_counts: dict[str, int] = {}
    for split_name in SPLITS:
        raw = counts[split_name]
        if isinstance(raw, bool):
            raise ComparatorLauncherError(
                "DGP_COMPARATOR_WORLD_COUNT_INVALID"
            )
        count = int(raw)
        if count <= 0 or str(count) != str(raw):
            raise ComparatorLauncherError(
                "DGP_COMPARATOR_WORLD_COUNT_INVALID"
            )
        exact_counts[split_name] = count
    stream = policy["randomness"][mode]
    id_key = _key_bytes(stream["id_key_hex"], label="id_key_hex")
    namespace_key = _key_bytes(
        stream["id_namespace_key_hex"],
        label="id_namespace_key_hex",
    )
    world_pool: list[str] = []
    for ordinal in range(sum(exact_counts.values())):
        message = FIELD_SEPARATOR.join(
            (
                b"step28-v13",
                b"world",
                mode.encode("utf-8"),
                str(ordinal).encode("ascii"),
            )
        )
        world_pool.append(
            "w_" + hmac.new(id_key, message, hashlib.sha256).hexdigest()
        )
    ranked = sorted(
        world_pool,
        key=lambda world_uid: (
            hmac.new(
                namespace_key,
                b"world_split_assignment"
                + FIELD_SEPARATOR
                + world_uid.encode("utf-8"),
                hashlib.sha256,
            ).digest(),
            world_uid.encode("utf-8"),
        ),
    )
    cursor = 0
    selected: dict[str, list[str]] = {}
    for split_name in SPLITS:
        count = exact_counts[split_name]
        selected[split_name] = sorted(
            ranked[cursor : cursor + count],
            key=lambda value: value.encode("utf-8"),
        )
        cursor += count
    if cursor != len(ranked):
        raise ComparatorLauncherError(
            "DGP_COMPARATOR_WORLD_SPLIT_SLICING_INVALID"
        )
    return selected[split]


def _reject_registered_key_environment(
    policy: Mapping[str, Any],
) -> None:
    custody = policy["randomness"]["formal"][
        "label_bearing_structure_keys"
    ]
    names = [
        str(custody[split]["environment_variable"]) for split in SPLITS
    ]
    if len(names) != len(set(names)):
        raise ComparatorLauncherError(
            "DGP_COMPARATOR_KEY_ENV_REGISTRY_INVALID"
        )
    if any(os.environ.get(name) is not None for name in names):
        raise ComparatorLauncherError(
            "DGP_COMPARATOR_REGISTERED_STRUCTURE_KEY_ENV_PRESENT"
        )


def _validate_replay_parent(
    *,
    receipt: Mapping[str, Any],
    receipt_path: Path,
    ledger_path: Path,
    policy_path: Path,
    expected_world_uids: list[str],
    mode: str,
    split: str,
) -> None:
    required = {
        "version",
        "mode",
        "split",
        "evidence_level",
        "formal_custody_seal",
        "world_count",
        "registered_split_world_count",
        "complete_registered_world_set_exact",
        "registered_world_uids_sha256",
        "input_records",
        "key_audit",
        "structure_key_serialized",
        "producer_oracle_input_used",
        "source_records",
        "world_replay_ledger_sha256",
        "output_ledger_size_bytes",
        "output_ledger_file_sha256",
        "canonical_self_hash",
    }
    if set(receipt) != required:
        raise ComparatorLauncherError(
            "DGP_COMPARATOR_REPLAY_RECEIPT_SCHEMA_INVALID"
        )
    _verify_self_hash(receipt, label="replay_receipt")
    if (
        receipt["version"] != REPLAY_RECEIPT_VERSION
        or receipt["mode"] != mode
        or receipt["split"] != split
        or receipt["evidence_level"]
        != (
            "DEVELOPMENT_INTEGRATION_COMPLETE_SPLIT_"
            "NOT_FORMAL_CUSTODY_SEAL"
        )
        or receipt["formal_custody_seal"] is not False
        or receipt["structure_key_serialized"] is not False
        or receipt["producer_oracle_input_used"] is not False
        or receipt["complete_registered_world_set_exact"] is not True
        or int(receipt["world_count"]) != len(expected_world_uids)
        or int(receipt["registered_split_world_count"])
        != len(expected_world_uids)
        or receipt["registered_world_uids_sha256"]
        != _canonical_sha256(expected_world_uids)
    ):
        raise ComparatorLauncherError(
            "DGP_COMPARATOR_REPLAY_RECEIPT_CONTENT_INVALID"
        )
    if (
        int(receipt["output_ledger_size_bytes"])
        != ledger_path.stat().st_size
        or receipt["output_ledger_file_sha256"]
        != _sha256_file(ledger_path)
    ):
        raise ComparatorLauncherError(
            "DGP_COMPARATOR_REPLAY_LEDGER_PARENT_MISMATCH"
        )
    inputs = receipt["input_records"]
    expected_roles = (
        "public_policy",
        "world_uid_pool",
        "seller_uid_pool",
        "all_item_uid_pool",
        "nonempty_title_item_uid_pool",
        "nonempty_description_item_uid_pool",
    )
    if (
        not isinstance(inputs, list)
        or [
            str(row.get("role"))
            for row in inputs
            if isinstance(row, Mapping)
        ]
        != list(expected_roles)
    ):
        raise ComparatorLauncherError(
            "DGP_COMPARATOR_REPLAY_INPUT_RECORDS_INVALID"
        )
    by_role = {
        str(row["role"]): row
        for row in inputs
        if isinstance(row, Mapping)
        and set(row)
        == {"role", "path_basename", "size_bytes", "sha256"}
    }
    if len(by_role) != len(inputs) or set(by_role) != set(expected_roles):
        raise ComparatorLauncherError(
            "DGP_COMPARATOR_REPLAY_INPUT_RECORDS_INVALID"
        )
    policy_record = by_role["public_policy"]
    if (
        policy_record["path_basename"] != policy_path.name
        or int(policy_record["size_bytes"]) != policy_path.stat().st_size
        or policy_record["sha256"] != _sha256_file(policy_path)
    ):
        raise ComparatorLauncherError(
            "DGP_COMPARATOR_REPLAY_POLICY_PARENT_MISMATCH"
        )
    source_records = receipt["source_records"]
    expected_source_paths = (
        (
            "independent_replay_launcher",
            Path(__file__).resolve().with_name(
                "step28_v13_run_independent_dgp_replay.py"
            ),
        ),
        (
            "independent_replay_implementation",
            Path(comparator.__file__).resolve().with_name(
                "step28_v13_independent_private_dgp_replay.py"
            ),
        ),
    )
    if (
        not isinstance(source_records, list)
        or len(source_records) != 2
        or [
            str(row.get("role"))
            for row in source_records
            if isinstance(row, Mapping)
            and set(row) == {"role", "path_basename", "sha256"}
        ]
        != [role for role, _path in expected_source_paths]
    ):
        raise ComparatorLauncherError(
            "DGP_COMPARATOR_REPLAY_SOURCE_RECORDS_INVALID"
        )
    if any(
        row["path_basename"] != source_path.name
        or row["sha256"] != _sha256_file(source_path)
        for row, (_role, source_path) in zip(
            source_records,
            expected_source_paths,
            strict=True,
        )
    ):
        raise ComparatorLauncherError(
            "DGP_COMPARATOR_REPLAY_SOURCE_CLOSURE_MISMATCH"
        )
    if receipt_path.stat().st_size <= 0:
        raise ComparatorLauncherError(
            "DGP_COMPARATOR_REPLAY_RECEIPT_EMPTY"
        )


def _validate_producer_parent(
    *,
    manifest: Mapping[str, Any],
    projection_path: Path,
    policy_path: Path,
    expected_world_uids: list[str],
    mode: str,
    split: str,
) -> None:
    required = {
        "version",
        "mode",
        "split",
        "evidence_level",
        "formal_custody_seal",
        "policy_sha256",
        "world_count",
        "registered_split_world_count",
        "complete_registered_world_set_exact",
        "registered_world_uids_sha256",
        "projection_file",
        "source_record",
        "canonical_self_hash",
    }
    if set(manifest) != required:
        raise ComparatorLauncherError(
            "DGP_COMPARATOR_PRODUCER_MANIFEST_SCHEMA_INVALID"
        )
    _verify_self_hash(manifest, label="producer_projection_manifest")
    if (
        manifest["version"] != PRODUCER_MANIFEST_VERSION
        or manifest["mode"] != mode
        or manifest["split"] != split
        or manifest["evidence_level"]
        != (
            "DEVELOPMENT_PRODUCER_PRIVATE_PROJECTION_"
            "NOT_FORMAL_CUSTODY_SEAL"
        )
        or manifest["formal_custody_seal"] is not False
        or manifest["policy_sha256"] != _sha256_file(policy_path)
        or int(manifest["world_count"]) != len(expected_world_uids)
        or int(manifest["registered_split_world_count"])
        != len(expected_world_uids)
        or manifest["complete_registered_world_set_exact"] is not True
        or manifest["registered_world_uids_sha256"]
        != _canonical_sha256(expected_world_uids)
    ):
        raise ComparatorLauncherError(
            "DGP_COMPARATOR_PRODUCER_MANIFEST_CONTENT_INVALID"
        )
    record = manifest["projection_file"]
    if (
        not isinstance(record, Mapping)
        or set(record) != {"role", "path", "size_bytes", "sha256"}
        or record["role"]
        != "private_producer_typed_dgp_projection"
        or record["path"]
        != "oracle/producer_typed_dgp_projection.private.jsonl"
        or int(record["size_bytes"]) != projection_path.stat().st_size
        or record["sha256"] != _sha256_file(projection_path)
    ):
        raise ComparatorLauncherError(
            "DGP_COMPARATOR_PRODUCER_PROJECTION_PARENT_MISMATCH"
        )
    source = manifest["source_record"]
    producer_source_path = Path(__file__).resolve().with_name(
        "step28_v13_producer_dgp_projection.py"
    )
    if (
        not isinstance(source, Mapping)
        or set(source) != {"role", "path_basename", "sha256"}
        or source["role"] != "producer_typed_dgp_projector"
        or source["path_basename"] != producer_source_path.name
    ):
        raise ComparatorLauncherError(
            "DGP_COMPARATOR_PRODUCER_SOURCE_RECORD_INVALID"
        )
    if source["sha256"] != _sha256_file(producer_source_path):
        raise ComparatorLauncherError(
            "DGP_COMPARATOR_PRODUCER_SOURCE_CLOSURE_MISMATCH"
        )


def _write_fsynced(path: Path, payload: bytes) -> None:
    with path.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def _fsync_directory(path: Path) -> None:
    """Persist directory entries on Linux; formal mode is Linux-only."""

    if os.name == "nt":
        return
    descriptor = os.open(
        path,
        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
    )
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _cleanup_failed_stage(stage: Path, members: tuple[Path, ...]) -> None:
    """Remove only the two known comparator staging files."""

    if not stage.exists():
        return
    for member in members:
        try:
            member.unlink()
        except FileNotFoundError:
            pass
    try:
        stage.rmdir()
    except FileNotFoundError:
        pass


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.mode != "development_smoke":
        raise ComparatorLauncherError(
            "DGP_COMPARATOR_FORMAL_CAPABILITY_NOT_IMPLEMENTED"
        )
    policy_path = Path(args.policy).resolve()
    if not policy_path.is_file():
        raise ComparatorLauncherError(
            "DGP_COMPARATOR_POLICY_PATH_INVALID"
        )
    policy = _read_json(policy_path)
    if (
        policy.get("version") != POLICY_VERSION
        or policy.get("status") != "DRAFT_SMOKE_ONLY"
        or policy.get("formal_generation_enabled") is not False
    ):
        raise ComparatorLauncherError(
            "DGP_COMPARATOR_DEVELOPMENT_POLICY_STATE_INVALID"
        )
    _reject_registered_key_environment(policy)
    if args.validate_config_only:
        return {
            "version": AGGREGATE_VERSION,
            "mode": args.mode,
            "split": args.split,
            "evidence_level": (
                "STATIC_DEVELOPMENT_CONFIGURATION_ONLY_NOT_COMPARISON"
            ),
            "configuration_valid": True,
            "private_inputs_opened": False,
            "formal_custody_seal": False,
            "registered_key_environment_names_present": False,
            "policy_sha256": _sha256_file(policy_path),
        }

    paths = {
        "replay_ledgers": Path(args.replay_ledgers).resolve(),
        "replay_receipt": Path(args.replay_receipt).resolve(),
        "producer_projections": Path(args.producer_projections).resolve(),
        "producer_manifest": Path(args.producer_manifest).resolve(),
    }
    if len(set(paths.values())) != len(paths) or any(
        not path.is_file() for path in paths.values()
    ):
        raise ComparatorLauncherError(
            "DGP_COMPARATOR_INPUT_PATH_INVALID"
        )
    expected_names = {
        "replay_ledgers": "world_replay_ledgers.private.jsonl",
        "replay_receipt": "replay_receipt.private.json",
        "producer_projections": (
            "producer_typed_dgp_projection.private.jsonl"
        ),
        "producer_manifest": (
            "producer_typed_dgp_projection_manifest.private.json"
        ),
    }
    if any(
        paths[role].name != expected_name
        for role, expected_name in expected_names.items()
    ) or (
        paths["replay_ledgers"].parent
        != paths["replay_receipt"].parent
        or paths["producer_projections"].parent
        != paths["producer_manifest"].parent
    ):
        raise ComparatorLauncherError(
            "DGP_COMPARATOR_INPUT_LAYOUT_INVALID"
        )
    expected_world_uids = _registered_world_uids(
        policy,
        mode=args.mode,
        split=args.split,
    )
    replay_receipt = _read_json(paths["replay_receipt"])
    _validate_replay_parent(
        receipt=replay_receipt,
        receipt_path=paths["replay_receipt"],
        ledger_path=paths["replay_ledgers"],
        policy_path=policy_path,
        expected_world_uids=expected_world_uids,
        mode=args.mode,
        split=args.split,
    )
    ledgers = _read_jsonl(paths["replay_ledgers"])
    ledger_world_uids = [str(row.get("world_uid", "")) for row in ledgers]
    if ledger_world_uids != expected_world_uids:
        raise ComparatorLauncherError(
            "DGP_COMPARATOR_REPLAY_COMPLETE_WORLD_SET_MISMATCH"
        )
    if _canonical_sha256(ledgers) != replay_receipt[
        "world_replay_ledger_sha256"
    ]:
        raise ComparatorLauncherError(
            "DGP_COMPARATOR_REPLAY_LEDGER_CONTENT_HASH_MISMATCH"
        )

    producer_manifest = _read_json(paths["producer_manifest"])
    _validate_producer_parent(
        manifest=producer_manifest,
        projection_path=paths["producer_projections"],
        policy_path=policy_path,
        expected_world_uids=expected_world_uids,
        mode=args.mode,
        split=args.split,
    )
    projections = _read_jsonl(paths["producer_projections"])
    projection_world_uids = [
        str(row.get("world_uid", "")) for row in projections
    ]
    if projection_world_uids != expected_world_uids:
        raise ComparatorLauncherError(
            "DGP_COMPARATOR_PRODUCER_COMPLETE_WORLD_SET_MISMATCH"
        )
    expected_graph = str(
        policy["identity_design"]["mechanism_by_split"][args.split]
    )
    receipts: list[dict[str, Any]] = []
    for ledger, projection, world_uid in zip(
        ledgers,
        projections,
        expected_world_uids,
        strict=True,
    ):
        if (
            ledger.get("mode") != args.mode
            or ledger.get("split") != args.split
            or ledger.get("world_uid") != world_uid
            or ledger.get("graph_name") != expected_graph
            or projection.get("mode") != args.mode
            or projection.get("split") != args.split
            or projection.get("world_uid") != world_uid
            or projection.get("graph_name") != expected_graph
        ):
            raise ComparatorLauncherError(
                "DGP_COMPARATOR_WORLD_ENVELOPE_MISMATCH"
            )
        receipts.append(
            comparator.compare_typed_dgp(
                expected_replay=ledger,
                producer_projection=projection,
            )
        )

    component_names = sorted(
        receipts[0]["component_receipts"],
        key=lambda value: value.encode("utf-8"),
    )
    component_summary: dict[str, dict[str, Any]] = {}
    for component in component_names:
        component_summary[component] = {
            "producer_row_count": sum(
                int(row["component_receipts"][component][
                    "producer_row_count"
                ])
                for row in receipts
            ),
            "replayer_row_count": sum(
                int(row["component_receipts"][component][
                    "replayer_row_count"
                ])
                for row in receipts
            ),
            "producer_world_hashes_sha256": _canonical_sha256(
                [
                    {
                        "world_uid": row["world_uid"],
                        "sha256": row["component_receipts"][component][
                            "producer_sha256"
                        ],
                    }
                    for row in receipts
                ]
            ),
            "replayer_world_hashes_sha256": _canonical_sha256(
                [
                    {
                        "world_uid": row["world_uid"],
                        "sha256": row["component_receipts"][component][
                            "replayer_sha256"
                        ],
                    }
                    for row in receipts
                ]
            ),
            "all_exact": True,
        }
    aggregate: dict[str, Any] = {
        "version": AGGREGATE_VERSION,
        "mode": args.mode,
        "split": args.split,
        "evidence_level": (
            "DEVELOPMENT_COMPLETE_SPLIT_SELF_HASH_MANIFEST_BOUND_"
            "COMPARISON_"
            "NOT_FORMAL_CUSTODY_SEAL"
        ),
        "world_count": len(receipts),
        "registered_split_world_count": len(expected_world_uids),
        "complete_registered_world_set_exact": True,
        "registered_world_uids_sha256": _canonical_sha256(
            expected_world_uids
        ),
        "all_worlds_exact": all(
            row["full_typed_projection_exact"] is True
            for row in receipts
        ),
        "component_summary": component_summary,
        "formal_custody_seal": False,
        "structure_key_input_count": 0,
        "registered_key_environment_names_present": False,
        "policy_sha256": _sha256_file(policy_path),
        "parent_records": {
            "replay_receipt": {
                "file_sha256": _sha256_file(paths["replay_receipt"]),
                "canonical_self_hash": replay_receipt[
                    "canonical_self_hash"
                ],
            },
            "producer_projection_manifest": {
                "file_sha256": _sha256_file(
                    paths["producer_manifest"]
                ),
                "canonical_self_hash": producer_manifest[
                    "canonical_self_hash"
                ],
            },
        },
        "comparator_source_records": [
            {
                "role": "development_comparator_launcher",
                "path_basename": Path(__file__).name,
                "sha256": _sha256_file(Path(__file__).resolve()),
            },
            {
                "role": "development_comparator_implementation",
                "path_basename": Path(comparator.__file__).name,
                "sha256": _sha256_file(
                    Path(comparator.__file__).resolve()
                ),
            },
        ],
        "private_world_receipts_sha256": _canonical_sha256(receipts),
    }

    output_root = Path(args.output_root).resolve()
    if output_root.exists():
        raise FileExistsError(
            f"Refusing to overwrite comparator output root: {output_root}"
        )
    if not output_root.parent.is_dir():
        raise ComparatorLauncherError(
            "DGP_COMPARATOR_OUTPUT_PARENT_MISSING"
        )
    stage = output_root.parent / (
        f".{output_root.name}.staging-{uuid.uuid4().hex}"
    )
    stage.mkdir()
    private_path = stage / "world_comparison_receipts.private.jsonl"
    public_path = stage / "aggregate_comparison_receipt.json"
    published = False
    try:
        _write_fsynced(
            private_path,
            b"".join(_canonical_json(row) + b"\n" for row in receipts),
        )
        aggregate["private_receipt_file_sha256"] = _sha256_file(
            private_path
        )
        aggregate["private_receipt_size_bytes"] = private_path.stat().st_size
        aggregate["canonical_self_hash"] = _canonical_sha256(aggregate)
        _write_fsynced(
            public_path,
            (
                json.dumps(
                    aggregate,
                    ensure_ascii=False,
                    sort_keys=True,
                    indent=2,
                    allow_nan=False,
                )
                + "\n"
            ).encode("utf-8"),
        )
        _fsync_directory(stage)
        os.replace(stage, output_root)
        published = True
        try:
            _fsync_directory(output_root.parent)
        except OSError as error:
            raise ComparatorLauncherError(
                "DGP_COMPARATOR_OUTPUT_PUBLISHED_PARENT_FSYNC_FAILED:"
                f"{output_root}"
            ) from error
    except BaseException:
        if not published:
            _cleanup_failed_stage(stage, (public_path, private_path))
        raise
    return aggregate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", required=True)
    parser.add_argument(
        "--mode",
        choices=("development_smoke", "formal"),
        required=True,
    )
    parser.add_argument("--split", choices=SPLITS, required=True)
    parser.add_argument("--replay-ledgers")
    parser.add_argument("--replay-receipt")
    parser.add_argument("--producer-projections")
    parser.add_argument("--producer-manifest")
    parser.add_argument("--output-root")
    parser.add_argument("--validate-config-only", action="store_true")
    args = parser.parse_args()
    if not args.validate_config_only:
        required = (
            "replay_ledgers",
            "replay_receipt",
            "producer_projections",
            "producer_manifest",
            "output_root",
        )
        missing = [name for name in required if not getattr(args, name)]
        if missing:
            parser.error(
                "required outside --validate-config-only: "
                + ", ".join(f"--{name.replace('_', '-')}" for name in missing)
            )
    return args


def main() -> None:
    receipt = run(parse_args())
    if receipt.get("configuration_valid") is True:
        print(
            "Step28-v13 DGP comparator static development "
            "configuration PASS"
        )
        return
    print(
        "Step28-v13 independent DGP comparison "
        "DEVELOPMENT_INTEGRATION_PASS: "
        f"split={receipt['split']} worlds={receipt['world_count']}"
    )


if __name__ == "__main__":
    main()
