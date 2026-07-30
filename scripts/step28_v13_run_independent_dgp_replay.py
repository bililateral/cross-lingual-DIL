#!/usr/bin/env python3
"""Run the split-private independent Step 28-v13 DGP replayer.

The CLI deliberately has no key-value argument.  A formal key is read only
from the exact split environment variable registered in the public policy.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import uuid
from collections import defaultdict
from pathlib import Path
from typing import Any

import step28_v13_independent_private_dgp_replay as replay


SPLITS = ("train", "development", "audit_a", "audit_b")
RECEIPT_VERSION = (
    "2026-07-28-step28-v13-independent-replay-receipt-v2-draft"
)
EVIDENCE_LEVEL = (
    "DEVELOPMENT_INTEGRATION_COMPLETE_SPLIT_NOT_FORMAL_CUSTODY_SEAL"
)


class ReplayLauncherError(ValueError):
    """Fail-closed launcher error that never includes secret material."""


def _read_json(path: Path) -> dict[str, Any]:
    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        output: dict[str, Any] = {}
        for key, value in pairs:
            if key in output:
                raise ReplayLauncherError(
                    f"REPLAY_LAUNCHER_DUPLICATE_JSON_KEY:{key}"
                )
            output[key] = value
        return output

    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle, object_pairs_hook=reject_duplicates)
    if not isinstance(value, dict):
        raise ReplayLauncherError("REPLAY_LAUNCHER_POLICY_NOT_OBJECT")
    return value


def _read_csv(path: Path, fields: tuple[str, ...]) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != list(fields):
            raise ReplayLauncherError(
                f"REPLAY_LAUNCHER_CSV_SCHEMA_MISMATCH:{path.name}"
            )
        rows = [{field: str(row[field]) for field in fields} for row in reader]
    return rows


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


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


def _write_fsynced(path: Path, payload: bytes) -> None:
    with path.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def _fsync_directory(path: Path) -> None:
    """Persist directory entries on the Linux formal target.

    Windows does not expose the same directory-fsync primitive.  Development
    smoke still gets atomic visibility from ``os.replace`` there; formal mode
    remains disabled and is required to run under the Linux custody launcher.
    """

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
    """Remove only the known files from this launcher's private stage."""

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


def _split_key(
    policy: dict[str, Any], *, mode: str, split: str
) -> tuple[str, dict[str, Any]]:
    stream = policy["randomness"][mode]
    if mode == "development_smoke":
        return str(stream["structure_key_hex"]), {
            "key_source": "public_development_smoke_policy",
            "other_split_key_env_present": False,
        }
    if mode != "formal":
        raise ReplayLauncherError("REPLAY_LAUNCHER_MODE_INVALID")
    custody = stream["label_bearing_structure_keys"]
    expected_name = str(custody[split]["environment_variable"])
    other_names = [
        str(custody[name]["environment_variable"])
        for name in SPLITS
        if name != split
    ]
    if any(os.environ.get(name) is not None for name in other_names):
        raise ReplayLauncherError(
            "REPLAY_LAUNCHER_OTHER_SPLIT_KEY_ENV_PRESENT"
        )
    value = os.environ.get(expected_name)
    if value is None:
        raise ReplayLauncherError("REPLAY_LAUNCHER_SPLIT_KEY_ENV_MISSING")
    return value, {
        "key_source": "registered_split_environment_variable",
        "key_environment_variable": expected_name,
        "other_split_key_env_present": False,
    }


def _group_rows(
    rows: list[dict[str, str]], *, world_uids: set[str]
) -> dict[str, list[dict[str, str]]]:
    output: defaultdict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        world_uid = row["world_uid"]
        if world_uid not in world_uids:
            raise ReplayLauncherError(
                "REPLAY_LAUNCHER_POOL_WORLD_FOREIGN_KEY"
            )
        output[world_uid].append(row)
    return dict(output)


def _require_canonical_unique_rows(
    rows: list[dict[str, str]],
    *,
    fields: tuple[str, ...],
    label: str,
) -> None:
    keys = [tuple(row[field] for field in fields) for row in rows]
    expected = sorted(
        keys,
        key=lambda key: tuple(value.encode("utf-8") for value in key),
    )
    if keys != expected:
        raise ReplayLauncherError(
            f"REPLAY_LAUNCHER_POOL_ORDER_NONCANONICAL:{label}"
        )
    if len(keys) != len(set(keys)):
        raise ReplayLauncherError(
            f"REPLAY_LAUNCHER_POOL_ROW_DUPLICATE:{label}"
        )


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.mode != "development_smoke":
        raise ReplayLauncherError(
            "REPLAY_LAUNCHER_FORMAL_CAPABILITY_NOT_IMPLEMENTED"
        )
    policy_path = Path(args.policy).resolve()
    if not policy_path.is_file():
        raise ReplayLauncherError("REPLAY_LAUNCHER_POLICY_PATH_INVALID")
    policy = _read_json(policy_path)
    if (
        policy.get("version") != replay.POLICY_VERSION
        or policy.get("status") != "DRAFT_SMOKE_ONLY"
        or policy.get("formal_generation_enabled") is not False
    ):
        raise ReplayLauncherError(
            "REPLAY_LAUNCHER_DEVELOPMENT_POLICY_STATE_INVALID"
        )
    if args.validate_config_only:
        return {
            "version": RECEIPT_VERSION,
            "mode": args.mode,
            "split": args.split,
            "evidence_level": (
                "STATIC_DEVELOPMENT_CONFIGURATION_ONLY_NOT_DGP_REPLAY"
            ),
            "configuration_valid": True,
            "input_data_opened": False,
            "structure_key_loaded": False,
            "formal_custody_seal": False,
            "policy_sha256": _sha256_file(policy_path),
        }
    world_path = Path(args.world_pool).resolve()
    seller_path = Path(args.seller_pool).resolve()
    all_item_path = Path(args.all_item_pool).resolve()
    title_path = Path(args.nonempty_title_pool).resolve()
    description_path = Path(args.nonempty_description_pool).resolve()
    input_paths = (
        policy_path,
        world_path,
        seller_path,
        all_item_path,
        title_path,
        description_path,
    )
    if len(set(input_paths)) != len(input_paths) or any(
        not path.is_file() for path in input_paths
    ):
        raise ReplayLauncherError("REPLAY_LAUNCHER_INPUT_PATH_INVALID")
    worlds = _read_csv(world_path, ("world_uid",))
    world_uids = [row["world_uid"] for row in worlds]
    if not world_uids or len(world_uids) != len(set(world_uids)):
        raise ReplayLauncherError("REPLAY_LAUNCHER_WORLD_POOL_INVALID")
    expected_world_uids = replay.registered_world_uids_for_split(
        policy,
        mode=args.mode,
        split=args.split,
    )
    if world_uids != expected_world_uids:
        raise ReplayLauncherError(
            "REPLAY_LAUNCHER_COMPLETE_WORLD_SET_MISMATCH"
        )
    world_set = set(world_uids)
    seller_rows = _read_csv(
        seller_path,
        ("world_uid", "seller_uid"),
    )
    all_item_rows = _read_csv(
        all_item_path,
        ("world_uid", "seller_uid", "item_uid"),
    )
    title_item_rows = _read_csv(
        title_path,
        ("world_uid", "seller_uid", "item_uid"),
    )
    description_item_rows = _read_csv(
        description_path,
        ("world_uid", "seller_uid", "item_uid"),
    )
    for rows, fields, label in (
        (seller_rows, ("world_uid", "seller_uid"), "seller_uid_pool"),
        (
            all_item_rows,
            ("world_uid", "seller_uid", "item_uid"),
            "all_item_uid_pool",
        ),
        (
            title_item_rows,
            ("world_uid", "seller_uid", "item_uid"),
            "nonempty_title_item_uid_pool",
        ),
        (
            description_item_rows,
            ("world_uid", "seller_uid", "item_uid"),
            "nonempty_description_item_uid_pool",
        ),
    ):
        _require_canonical_unique_rows(
            rows,
            fields=fields,
            label=label,
        )
    sellers = _group_rows(
        seller_rows,
        world_uids=world_set,
    )
    all_items = _group_rows(
        all_item_rows,
        world_uids=world_set,
    )
    title_items = _group_rows(
        title_item_rows,
        world_uids=world_set,
    )
    description_items = _group_rows(
        description_item_rows,
        world_uids=world_set,
    )
    for grouped, label in (
        (sellers, "SELLER"),
        (all_items, "ALL_ITEM"),
        (title_items, "NONEMPTY_TITLE"),
        (description_items, "NONEMPTY_DESCRIPTION"),
    ):
        if set(grouped) != world_set:
            raise ReplayLauncherError(
                f"REPLAY_LAUNCHER_{label}_POOL_WORLD_SET_MISMATCH"
            )
    structure_key_hex, key_audit = _split_key(
        policy,
        mode=args.mode,
        split=args.split,
    )
    ledgers = [
        replay.replay_typed_dgp(
            policy,
            mode=args.mode,
            split=args.split,
            world_uid=world_uid,
            observed_seller_uids=[
                row["seller_uid"] for row in sellers[world_uid]
            ],
            observed_all_item_uid_rows=all_items.get(world_uid, []),
            observed_nonempty_title_item_uid_rows=title_items.get(
                world_uid, []
            ),
            observed_nonempty_description_item_uid_rows=(
                description_items.get(world_uid, [])
            ),
            structure_key_hex=structure_key_hex,
        )
        for world_uid in sorted(
            world_uids, key=lambda value: value.encode("utf-8")
        )
    ]
    input_records = [
        {
            "role": role,
            "path_basename": path.name,
            "size_bytes": path.stat().st_size,
            "sha256": _sha256_file(path),
        }
        for role, path in zip(
            (
                "public_policy",
                "world_uid_pool",
                "seller_uid_pool",
                "all_item_uid_pool",
                "nonempty_title_item_uid_pool",
                "nonempty_description_item_uid_pool",
            ),
            input_paths,
            strict=True,
        )
    ]
    receipt = {
        "version": RECEIPT_VERSION,
        "mode": args.mode,
        "split": args.split,
        "evidence_level": EVIDENCE_LEVEL,
        "formal_custody_seal": False,
        "world_count": len(ledgers),
        "registered_split_world_count": len(expected_world_uids),
        "complete_registered_world_set_exact": (
            world_uids == expected_world_uids
        ),
        "registered_world_uids_sha256": _canonical_sha256(
            expected_world_uids
        ),
        "input_records": input_records,
        "key_audit": key_audit,
        "structure_key_serialized": False,
        "producer_oracle_input_used": False,
        "source_records": [
            {
                "role": "independent_replay_launcher",
                "path_basename": Path(__file__).name,
                "sha256": _sha256_file(Path(__file__).resolve()),
            },
            {
                "role": "independent_replay_implementation",
                "path_basename": Path(replay.__file__).name,
                "sha256": _sha256_file(Path(replay.__file__).resolve()),
            },
        ],
        "world_replay_ledger_sha256": _canonical_sha256(ledgers),
    }
    output_root = Path(args.output_root).resolve()
    if output_root.exists():
        raise FileExistsError(
            f"Refusing to overwrite replay output root: {output_root}"
        )
    if not output_root.parent.is_dir():
        raise ReplayLauncherError("REPLAY_LAUNCHER_OUTPUT_PARENT_MISSING")
    stage = output_root.parent / (
        f".{output_root.name}.staging-{uuid.uuid4().hex}"
    )
    stage.mkdir()
    ledger_path = stage / "world_replay_ledgers.private.jsonl"
    receipt_path = stage / "replay_receipt.private.json"
    published = False
    try:
        _write_fsynced(
            ledger_path,
            b"".join(
                _canonical_json(ledger) + b"\n" for ledger in ledgers
            ),
        )
        receipt["output_ledger_size_bytes"] = ledger_path.stat().st_size
        receipt["output_ledger_file_sha256"] = _sha256_file(ledger_path)
        receipt["canonical_self_hash"] = _canonical_sha256(receipt)
        _write_fsynced(
            receipt_path,
            (
                json.dumps(
                    receipt,
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
            raise ReplayLauncherError(
                "REPLAY_LAUNCHER_OUTPUT_PUBLISHED_PARENT_FSYNC_FAILED:"
                f"{output_root}"
            ) from error
    except BaseException:
        if not published:
            _cleanup_failed_stage(stage, (receipt_path, ledger_path))
        raise
    return receipt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", required=True)
    parser.add_argument(
        "--mode",
        choices=("development_smoke", "formal"),
        required=True,
    )
    parser.add_argument("--split", choices=SPLITS, required=True)
    parser.add_argument("--world-pool")
    parser.add_argument("--seller-pool")
    parser.add_argument("--all-item-pool")
    parser.add_argument("--nonempty-title-pool")
    parser.add_argument("--nonempty-description-pool")
    parser.add_argument("--output-root")
    parser.add_argument("--validate-config-only", action="store_true")
    args = parser.parse_args()
    if not args.validate_config_only:
        required = (
            "world_pool",
            "seller_pool",
            "all_item_pool",
            "nonempty_title_pool",
            "nonempty_description_pool",
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
            "Step28-v13 independent DGP replayer static development "
            "configuration PASS"
        )
        return
    print(
        "Step28-v13 independent DGP replay DEVELOPMENT_INTEGRATION_PASS: "
        f"mode={receipt['mode']} split={receipt['split']} "
        f"worlds={receipt['world_count']}"
    )


if __name__ == "__main__":
    main()
