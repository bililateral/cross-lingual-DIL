#!/usr/bin/env python3
"""Select and replay collision-free Step 28-v13 synthetic identity values."""

from __future__ import annotations

import argparse
import hashlib
import hmac
import math
from pathlib import Path
from typing import Any

import step28_v13_common as common


BASE36_ALPHABET = "0123456789abcdefghijklmnopqrstuvwxyz"
HANDLE_TYPES = frozenset({"telegram", "bat", "wechat"})
HANDLE_ENCODINGS = {
    "legacy_base36_v1": (BASE36_ALPHABET, 36**14),
    "parser_safe_hex_v2": ("0123456789abcdef", 16**14),
}
NON_HANDLE_DOMAIN_SIZES = {
    "email": 16**16,
    "qq": 9 * 10**8,
    "phone": 10**9,
    "crypto_wallet": 16**40,
    "external_url": 16**20,
}


def domain_size(identity_type: str, handle_encoding: str) -> int:
    if identity_type in HANDLE_TYPES:
        try:
            return HANDLE_ENCODINGS[handle_encoding][1]
        except KeyError as exc:
            raise common.ContractError(
                f"Unknown handle encoding: {handle_encoding}"
            ) from exc
    try:
        return NON_HANDLE_DOMAIN_SIZES[identity_type]
    except KeyError as exc:
        raise common.ContractError(
            f"Unknown identity type: {identity_type}"
        ) from exc


def _hmac_integer(key_hex: str, domain: str, identity_type: str, salt: int) -> int:
    if salt < 0:
        raise common.ContractError("Identity salt cannot be negative")
    message = (
        domain.encode("ascii")
        + common.FIELD_SEPARATOR
        + identity_type.encode("ascii")
        + common.FIELD_SEPARATOR
        + str(salt).encode("ascii")
    )
    return int.from_bytes(
        hmac.new(bytes.fromhex(key_hex), message, hashlib.sha256).digest(),
        "big",
        signed=False,
    )


def affine_parameters(
    key_hex: str,
    identity_type: str,
    salt: int,
    *,
    handle_encoding: str = "legacy_base36_v1",
) -> tuple[int, int, int]:
    modulus = domain_size(identity_type, handle_encoding)
    coefficient = _hmac_integer(
        key_hex, "affine_a", identity_type, salt
    ) % modulus
    while math.gcd(coefficient, modulus) != 1:
        coefficient = (coefficient + 1) % modulus
    offset = _hmac_integer(
        key_hex, "affine_b", identity_type, salt
    ) % modulus
    return coefficient, offset, modulus


def _base_n(value: int, width: int, alphabet: str) -> str:
    if value < 0:
        raise common.ContractError("Cannot encode a negative integer")
    base = len(alphabet)
    if base < 2 or len(set(alphabet)) != base:
        raise common.ContractError("Identity alphabet is invalid")
    digits: list[str] = []
    current = value
    while current:
        current, remainder = divmod(current, base)
        digits.append(alphabet[remainder])
    output = "".join(reversed(digits or ["0"]))
    if len(output) > width:
        raise common.ContractError("Encoded identity exceeds its fixed width")
    return output.rjust(width, "0")


def encode_identity_value(
    identity_type: str,
    value: int,
    *,
    handle_encoding: str = "legacy_base36_v1",
) -> str:
    modulus = domain_size(identity_type, handle_encoding)
    if not 0 <= value < modulus:
        raise common.ContractError("Identity integer is outside its type domain")
    handle_alphabet = HANDLE_ENCODINGS.get(handle_encoding, (None, None))[0]
    if identity_type == "telegram":
        return "tg" + _base_n(value, 14, str(handle_alphabet))
    if identity_type == "email":
        return "u" + f"{value:016x}" + "@id.invalid"
    if identity_type == "bat":
        return "bt" + _base_n(value, 14, str(handle_alphabet))
    if identity_type == "qq":
        return str(1 + value // 10**8) + f"{value % 10**8:08d}"
    if identity_type == "wechat":
        return "wx" + _base_n(value, 14, str(handle_alphabet))
    if identity_type == "phone":
        return "13" + f"{value:09d}"
    if identity_type == "crypto_wallet":
        return "0x" + f"{value:040x}"
    if identity_type == "external_url":
        encoded = f"{value:020x}"
        return f"s{encoded[:12]}.example/path/{encoded[12:]}"
    raise common.ContractError(f"Unknown identity type: {identity_type}")


def identity_value(
    *,
    key_hex: str,
    identity_type: str,
    salt: int,
    global_asset_index: int,
    handle_encoding: str = "legacy_base36_v1",
) -> str:
    coefficient, offset, modulus = affine_parameters(
        key_hex,
        identity_type,
        salt,
        handle_encoding=handle_encoding,
    )
    if global_asset_index < 0:
        raise common.ContractError("Global identity-asset index cannot be negative")
    return encode_identity_value(
        identity_type,
        (coefficient * global_asset_index + offset) % modulus,
        handle_encoding=handle_encoding,
    )


def value_hash(value: str) -> str:
    normalized = __import__("unicodedata").normalize(
        "NFC", value.strip().casefold()
    )
    if not normalized:
        raise common.ContractError("Cannot hash an empty identity value")
    return common.sha256_bytes(normalized.encode("utf-8"))


def _load_hash_artifact(spec: dict[str, Any], *, label: str) -> dict[str, Any]:
    path = common.verify_file_pin(spec, label=label)
    artifact = common.load_json(path)
    claimed = artifact.get("canonical_self_hash")
    payload = dict(artifact)
    payload.pop("canonical_self_hash", None)
    if claimed != common.canonical_sha256(payload):
        raise common.ContractError(f"{label} canonical self-hash drift")
    return artifact


def _candidate_hashes(
    *,
    key_hex: str,
    identity_type: str,
    salt: int,
    candidate_count: int,
    handle_encoding: str,
) -> set[str]:
    values = {
        identity_value(
            key_hex=key_hex,
            identity_type=identity_type,
            salt=salt,
            global_asset_index=index,
            handle_encoding=handle_encoding,
        )
        for index in range(candidate_count)
    }
    if len(values) != candidate_count:
        raise common.ContractError(
            f"Identity affine pool is not bijective for {identity_type}"
        )
    if (
        handle_encoding == "parser_safe_hex_v2"
        and identity_type in HANDLE_TYPES
    ):
        # Local import keeps the value utility lightweight while binding the
        # formal gate to the exact frozen parser semantics used downstream.
        import step3_build_seller_profiles as step3

        collisions = [
            value
            for value in values
            if step3.PRODUCT_DATA_RISK_RE.search(value) is not None
        ]
        if collisions:
            raise common.ContractError(
                "Formal handle encoding is not Step3 parser-safe: "
                f"{identity_type} example={common.utf8_sort(collisions)[0]}"
            )
    return {value_hash(value) for value in values}


def build_salt_artifact(
    policy: dict[str, Any], *, mode: str
) -> dict[str, Any]:
    if mode not in {"development_smoke", "formal"}:
        raise common.ContractError(f"Unsupported identity salt mode: {mode}")
    salt_policy = policy["identity_design"]["identity_value_generation"][
        "salt_selection"
    ]
    deny = _load_hash_artifact(
        salt_policy["deny_hash_artifact"],
        label="identity value deny-hash artifact",
    )
    forbidden = set(deny["value_hashes"])
    if len(forbidden) != int(deny["unique_value_hash_count"]):
        raise common.ContractError("Identity deny hash count drift")

    if mode == "formal":
        smoke_spec = salt_policy["salt_artifacts"]["development_smoke"]
        smoke = _load_hash_artifact(
            smoke_spec, label="development-smoke identity salt artifact"
        )
        forbidden.update(smoke["candidate_value_hashes"])
        if policy["modes"]["formal"].get("power_design_path") is None:
            raise common.ContractError(
                "Formal identity salts cannot be selected before power design"
            )

    stream = policy["randomness"][mode]
    key_hex = str(stream["identity_value_key_hex"])
    handle_encoding = str(
        policy["identity_design"]["identity_value_generation"][
            "handle_encoding_by_mode"
        ][mode]
    )
    world_count = sum(
        int(value) for value in policy["modes"][mode]["world_counts"].values()
    )
    candidate_count = world_count * int(
        policy["identity_design"]["slot_feasibility"][
            "identity_asset_uid_pool"
        ]["count_per_world"]
    )
    identity_types = list(policy["identity_design"]["identity_types"])
    maximum_salt = int(salt_policy["maximum_salt"])
    selected: dict[str, int] = {}
    per_type_hashes: dict[str, list[str]] = {}
    same_mode_union: set[str] = set()

    for identity_type in identity_types:
        for salt in range(maximum_salt + 1):
            hashes = _candidate_hashes(
                key_hex=key_hex,
                identity_type=identity_type,
                salt=salt,
                candidate_count=candidate_count,
                handle_encoding=handle_encoding,
            )
            if hashes & forbidden or hashes & same_mode_union:
                continue
            selected[identity_type] = salt
            ordered_hashes = common.utf8_sort(hashes)
            per_type_hashes[identity_type] = ordered_hashes
            same_mode_union.update(hashes)
            break
        else:
            raise common.ContractError(
                f"No collision-free identity salt through {maximum_salt}: {identity_type}"
            )

    configured = salt_policy[f"{mode}_per_type_salt_counters"]
    if configured is not None and configured != selected:
        raise common.ContractError(
            f"Configured identity salts drift from first admissible salts for {mode}"
        )
    config_projection = {
        "mode": mode,
        "identity_types": identity_types,
        "identity_value_key_commitment": common.sha256_bytes(
            bytes.fromhex(key_hex)
        ),
        "world_counts": policy["modes"][mode]["world_counts"],
        "candidate_count_per_type": candidate_count,
        "maximum_salt": maximum_salt,
        "handle_encoding": handle_encoding,
        "deny_artifact_sha256": salt_policy["deny_hash_artifact"]["sha256"],
        "other_mode_artifact_sha256": (
            salt_policy["salt_artifacts"]["development_smoke"]["sha256"]
            if mode == "formal"
            else None
        ),
    }
    artifact: dict[str, Any] = {
        "version": f"2026-07-27-step28-v13-identity-value-salts-{mode}-v1",
        "status": "PASS_BOUNDARY_ONLY",
        "mode": mode,
        "scientific_metrics_produced": False,
        "config_projection": config_projection,
        "config_projection_sha256": common.canonical_sha256(config_projection),
        "salt_counters": selected,
        "candidate_count_per_type": candidate_count,
        "candidate_value_hashes_by_type_sha256": {
            identity_type: common.canonical_sha256(per_type_hashes[identity_type])
            for identity_type in identity_types
        },
        "candidate_value_hashes": common.utf8_sort(same_mode_union),
        "candidate_value_hash_count": len(same_mode_union),
        "deny_intersection_count": 0,
        "same_mode_cross_type_intersection_count": 0,
    }
    artifact["canonical_self_hash"] = common.canonical_sha256(artifact)
    return artifact


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    common.add_policy_argument(parser)
    parser.add_argument(
        "--mode",
        choices=("development_smoke", "formal"),
        default="development_smoke",
    )
    parser.add_argument("--validate-config-only", action="store_true")
    args = parser.parse_args()
    policy = common.load_policy(args.policy, mode="development_smoke")
    if args.validate_config_only:
        _load_hash_artifact(
            policy["identity_design"]["identity_value_generation"][
                "salt_selection"
            ]["deny_hash_artifact"],
            label="identity value deny-hash artifact",
        )
        print("Step28-v13 identity-value salt configuration is valid")
        return
    artifact = build_salt_artifact(policy, mode=args.mode)
    output = common.repo_path(
        policy["identity_design"]["identity_value_generation"]["salt_selection"][
            "salt_artifacts"
        ][args.mode]["path"]
    )
    common.write_json(output, artifact)
    print(
        "Step28-v13 identity salts selected:",
        args.mode,
        artifact["salt_counters"],
        artifact["candidate_value_hash_count"],
    )


if __name__ == "__main__":
    main()
