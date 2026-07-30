#!/usr/bin/env python3
"""Shared deterministic and fail-closed utilities for Step 28-v13.

This module deliberately has no dependency on labels, model packages, or prior
Step 28 generators.  Experiment stages import it before importing any
production parser/model helper.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import hmac
import json
import math
import os
import sys
import tempfile
from collections.abc import Iterable, Mapping, Sequence
from decimal import Decimal, ROUND_FLOOR, localcontext
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_POLICY_PATH = ROOT / "schema" / "step28_v13_synthetic_chinese_dataset_policy.json"
HEX_SHA256_LENGTH = 64
FIELD_SEPARATOR = b"\x1f"
POLICY_VERSION = "2026-07-29-step28-v13-synthetic-chinese-dataset-v13-draft"


class ContractError(ValueError):
    """Raised when an input violates a frozen Step 28-v13 contract."""


def canonical_json_bytes(value: Any) -> bytes:
    """Return the one canonical JSON representation used for content hashes."""

    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def canonical_rows_sha256(
    rows: Sequence[Mapping[str, Any]],
    *,
    order_fields: Sequence[str],
) -> str:
    """Hash a table after an explicit, input-order-independent UTF-8 sort."""

    fields = list(order_fields)
    if not fields or len(fields) != len(set(fields)):
        raise ContractError("Canonical table order fields are empty or duplicated")
    materialized = [dict(row) for row in rows]
    if any(any(field not in row for field in fields) for row in materialized):
        raise ContractError("Canonical table row lacks an order field")
    materialized.sort(
        key=lambda row: tuple(
            (
                str(row[field]).encode("utf-8")
                if not isinstance(row[field], (int, float, bool))
                else canonical_json_bytes(row[field])
            )
            for field in fields
        )
    )
    return canonical_sha256(materialized)


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def filesystem_path(path: Path) -> str:
    """Return an absolute OS path, using Win32 extended-length syntax."""

    value = os.path.abspath(os.fspath(path))
    if os.name != "nt" or value.startswith("\\\\?\\"):
        return value
    if value.startswith("\\\\"):
        return "\\\\?\\UNC\\" + value[2:]
    return "\\\\?\\" + value


def atomic_rename_no_replace(source: Path, target: Path) -> None:
    """Atomically rename a file or directory and fail if target exists."""

    source_value = filesystem_path(source)
    target_value = filesystem_path(target)
    if os.name == "nt":
        # Win32 rename is atomic and raises FileExistsError when dst exists.
        os.rename(source_value, target_value)
        return
    if os.name == "posix" and sys.platform.startswith("linux"):
        import ctypes

        libc = ctypes.CDLL(None, use_errno=True)
        renameat2 = getattr(libc, "renameat2", None)
        if renameat2 is None:
            raise ContractError(
                "Linux atomic no-replace publish requires libc renameat2"
            )
        renameat2.argtypes = (
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        )
        renameat2.restype = ctypes.c_int
        ctypes.set_errno(0)
        result = renameat2(
            -100,
            os.fsencode(source_value),
            -100,
            os.fsencode(target_value),
            1,
        )
        if result != 0:
            error = ctypes.get_errno()
            raise OSError(error, os.strerror(error), os.fspath(target))
        return
    raise ContractError(
        "Atomic no-replace publish is unsupported on this operating system"
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(filesystem_path(path), "rb") as handle:
        while True:
            chunk = handle.read(8 * 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _reject_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for key, value in pairs:
        if key in output:
            raise ContractError(f"Duplicate JSON object key is forbidden: {key}")
        output[key] = value
    return output


def load_json(path: Path) -> Any:
    with open(filesystem_path(path), "r", encoding="utf-8") as handle:
        return json.load(handle, object_pairs_hook=_reject_duplicate_pairs)


def repo_path(relative_path: str) -> Path:
    """Resolve a policy path and reject escape/symlink tricks."""

    candidate = Path(relative_path)
    if candidate.is_absolute() or candidate.drive:
        raise ContractError(f"Absolute or drive-qualified policy path is forbidden: {relative_path}")
    if any(part in {"", ".", ".."} for part in candidate.parts):
        raise ContractError(f"Non-canonical policy path is forbidden: {relative_path}")
    resolved_root = ROOT.resolve()
    resolved = (ROOT / candidate).resolve()
    try:
        resolved.relative_to(resolved_root)
    except ValueError as exc:
        raise ContractError(f"Policy path escapes the repository: {relative_path}") from exc
    cursor = ROOT
    for part in candidate.parts:
        cursor = cursor / part
        if cursor.exists() and cursor.is_symlink():
            raise ContractError(f"Symlink is forbidden in policy path: {relative_path}")
    return resolved


def verify_file_pin(spec: Mapping[str, Any], *, label: str) -> Path:
    path = repo_path(str(spec["path"]))
    if not path.is_file():
        raise FileNotFoundError(f"Missing frozen {label}: {path}")
    expected = str(spec["sha256"]).lower()
    if len(expected) != HEX_SHA256_LENGTH:
        raise ContractError(f"Invalid SHA-256 length for {label}: {expected}")
    observed = sha256_file(path)
    if observed != expected:
        raise ContractError(
            f"Frozen input drift for {label}: expected={expected} observed={observed}"
        )
    return path


def _validate_hex_key(value: object, *, label: str) -> str:
    text_value = str(value).lower()
    if len(text_value) != HEX_SHA256_LENGTH:
        raise ContractError(f"{label} must contain exactly 32 bytes as lowercase hex")
    try:
        raw = bytes.fromhex(text_value)
    except ValueError as exc:
        raise ContractError(f"{label} is not hexadecimal") from exc
    if len(raw) != 32 or text_value != raw.hex():
        raise ContractError(f"{label} must be canonical lowercase 32-byte hex")
    return text_value


def _validate_release_reference(
    spec: Mapping[str, Any],
    *,
    label: str,
    frozen: bool,
) -> Path:
    path = repo_path(str(spec["path"]))
    if not path.is_file():
        raise FileNotFoundError(f"Missing {label}: {path}")
    expected = spec.get("sha256")
    if frozen:
        if not isinstance(expected, str):
            raise ContractError(f"Frozen {label} requires a SHA-256 pin")
        verify_file_pin(spec, label=label)
    elif expected is not None:
        _validate_hex_key(expected, label=f"{label} SHA-256")
        verify_file_pin(spec, label=label)
    return path


def _validate_release_reference_shape(
    spec: Mapping[str, Any],
    *,
    label: str,
    frozen: bool,
) -> None:
    """Validate a release reference without opening the referenced file."""

    path_value = str(spec.get("path", ""))
    if not path_value:
        raise ContractError(f"Missing path for {label}")
    repo_path(path_value)
    expected = spec.get("sha256")
    if frozen and not isinstance(expected, str):
        raise ContractError(f"Frozen {label} requires a SHA-256 pin")
    if expected is not None:
        _validate_hex_key(expected, label=f"{label} SHA-256")


def validate_policy(policy: Mapping[str, Any], *, mode: str) -> None:
    """Validate policy structure without opening any referenced input.

    This validator is intentionally capability-neutral.  A worker must
    separately verify only the external files that its own capability is
    allowed to open.
    """

    if policy.get("version") != POLICY_VERSION:
        raise ContractError("Unexpected Step 28-v13 policy version")
    if mode not in policy["modes"]:
        raise ContractError(f"Unknown Step 28-v13 mode: {mode}")
    frozen = policy.get("status") == "FROZEN"
    if mode == "formal" and (not frozen or not policy.get("formal_generation_enabled")):
        raise ContractError("Formal generation is disabled until a frozen policy is released")

    _validate_release_reference_shape(
        policy["contract"],
        label="construction contract",
        frozen=frozen,
    )
    _validate_release_reference_shape(
        policy["template_library"],
        label="template library",
        frozen=frozen,
    )
    fixture_spec = policy["identity_design"]["role_template_parser_flag_fixture"]
    _validate_release_reference_shape(
        fixture_spec,
        label="parser/template fixture",
        frozen=frozen,
    )
    _validate_release_reference_shape(
        policy["security"]["dataset_custody_deployment"],
        label="dataset custody deployment",
        frozen=frozen,
    )

    for label, spec in policy["frozen_inputs"].items():
        path_value = str(spec.get("path", ""))
        hash_value = str(spec.get("sha256", "")).lower()
        if not path_value or len(hash_value) != HEX_SHA256_LENGTH:
            raise ContractError(f"Malformed frozen-input pin for {label}")
        repo_path(path_value)
        _validate_hex_key(hash_value, label=f"{label} SHA-256")

    counts = policy["modes"][mode]["world_counts"]
    if set(counts) != {"train", "development", "audit_a", "audit_b"}:
        raise ContractError("World-count split set drift")
    if any(int(value) <= 0 for value in counts.values()):
        raise ContractError("Every split must contain a positive number of worlds")

    release = policy["development_complete_release"]
    expected_release_keys = {
        "release_name",
        "manifest_filename",
        "manifest_version",
        "required_status",
        "required_split_order",
        "parent_binds_child_direction",
        "fixture_result_must_be_opened_and_validated_before_generation",
        "fixture_result_exact_file_must_be_bound_by_parent",
        "atomic_publish_semantics",
        "release_parent_symlink_junction_or_escape_forbidden",
        "m0_exact_mount_allowlist_per_split",
        "m0_observed_directory_mount_forbidden",
        "supersedes_development_release",
    }
    if (
        set(release) != expected_release_keys
        or release["release_name"] != "dataset_smoke_v3"
        or release["manifest_filename"] != "release_manifest.json"
        or list(release["required_split_order"])
        != ["train", "development", "audit_a", "audit_b"]
        or release["fixture_result_must_be_opened_and_validated_before_generation"]
        is not True
        or release["fixture_result_exact_file_must_be_bound_by_parent"]
        is not True
        or release["release_parent_symlink_junction_or_escape_forbidden"]
        is not True
        or release["m0_observed_directory_mount_forbidden"] is not True
        or list(release["m0_exact_mount_allowlist_per_split"])
        != [
            "observed/complete_model_pair_endpoints.csv",
            "observed/redacted_items.jsonl",
            "observed/seller_profiles.jsonl",
        ]
    ):
        raise ContractError("Development complete-release contract drift")

    design = policy["world_design"]
    if (
        int(design["controllers_per_world"]) != 12
        or int(design["dyad_controller_count"]) != 8
        or int(design["triad_controller_count"]) != 4
        or int(design["sellers_per_world"]) != 28
    ):
        raise ContractError("World topology drift")
    if (
        int(design["dyad_controller_count"]) * 2
        + int(design["triad_controller_count"]) * 3
        != int(design["sellers_per_world"])
    ):
        raise ContractError("Controller-size arithmetic is inconsistent")

    identity = policy["identity_design"]
    for graph_name in ("G_A", "G_B"):
        if sum(int(value) for value in identity["mechanism_assignments"][graph_name].values()) != 12:
            raise ContractError(f"{graph_name} does not assign exactly 12 controllers")
    features = policy["history_features"]["feature_names"]
    if len(features) != 33 or len(set(features)) != 33:
        raise ContractError("History feature list must contain 33 unique names")

    random_cfg = policy["randomness"]
    all_public_values: list[str] = []
    for namespace in ("formal", "development_smoke"):
        stream = random_cfg[namespace]
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
        canonical_values = [
            _validate_hex_key(value, label=f"{namespace} random key")
            for value in values
        ]
        if len(canonical_values) != len(set(canonical_values)):
            raise ContractError(f"Reused random key in {namespace}")
        all_public_values.extend(canonical_values)
    if len(all_public_values) != len(set(all_public_values)):
        raise ContractError("Formal and smoke random-key namespaces overlap")
    if mode == "training_ready":
        training_stream = random_cfg.get("training_ready")
        if not isinstance(training_stream, Mapping):
            raise ContractError("Training-ready random stream is missing")
        public_fields = (
            "id_namespace_key_hex",
            "id_key_hex",
            "identity_value_key_hex",
            "text_key_hex",
            "candidate_key_hex",
            "query_key_hex",
            "rewire_key_hexes",
        )
        if any(
            canonical_json_bytes(training_stream.get(field))
            != canonical_json_bytes(random_cfg["formal"][field])
            for field in public_fields
        ):
            raise ContractError(
                "Training-ready public randomness must equal the formal stream"
            )
        training_structure_key = _validate_hex_key(
            training_stream.get("structure_key_hex"),
            label="training-ready split-private structure key",
        )
        if (
            training_structure_key in set(all_public_values)
            or sha256_bytes(bytes.fromhex(training_structure_key))
            in {
                str(value).lower()
                for value in random_cfg["formal"][
                    "label_bearing_structure_keys"
                ]["compromised_draft_key_commitments_forbidden"]
            }
        ):
            raise ContractError(
                "Training-ready structure key collides or is compromised"
            )

    formal_structure = random_cfg["formal"]["label_bearing_structure_keys"]
    if int(formal_structure["key_bytes"]) != 32:
        raise ContractError("Formal structure keys must contain 32 bytes")
    commitments: list[str] = []
    for split in ("train", "development", "audit_a", "audit_b"):
        split_spec = formal_structure[split]
        environment_variable = str(split_spec.get("environment_variable", ""))
        if environment_variable != f"STEP28_V13_{split.upper()}_STRUCTURE_KEY_HEX":
            raise ContractError(f"Formal structure-key environment name drift for {split}")
        commitment = split_spec.get("sha256_commitment")
        if frozen:
            commitments.append(
                _validate_hex_key(
                    commitment,
                    label=f"formal {split} structure-key commitment",
                )
            )
        elif commitment is not None:
            commitments.append(
                _validate_hex_key(
                    commitment,
                    label=f"draft {split} structure-key commitment",
                )
            )
    deny = {
        _validate_hex_key(value, label="compromised structure-key commitment")
        for value in formal_structure["compromised_draft_key_commitments_forbidden"]
    }
    public_key_commitments = {
        sha256_bytes(bytes.fromhex(value)) for value in all_public_values
    }
    if commitments and (
        len(commitments) != len(set(commitments))
        or set(commitments) & deny
        or set(commitments) & public_key_commitments
    ):
        raise ContractError(
            "Formal structure-key commitments are reused, compromised, or collide "
            "with a public random key"
        )
    if frozen and len(commitments) != 4:
        raise ContractError("Frozen release requires four committed structure keys")

    formal_keys = set(random_cfg["formal"]["rewire_key_hexes"])
    smoke_keys = set(random_cfg["development_smoke"]["rewire_key_hexes"])
    expected_replicates = int(policy["placebo"]["replicates"])
    if (
        expected_replicates != 5
        or len(random_cfg["formal"]["rewire_key_hexes"])
        != expected_replicates
        or len(random_cfg["development_smoke"]["rewire_key_hexes"])
        != expected_replicates
        or len(formal_keys) != expected_replicates
        or len(smoke_keys) != expected_replicates
    ):
        raise ContractError(
            "Formal and smoke must each register exactly five rewire keys"
        )
    if formal_keys & smoke_keys:
        raise ContractError("Formal and smoke rewire keys overlap")

    if frozen:
        formal_mode = policy["modes"]["formal"]
        for key in ("power_design_path", "power_design_sha256"):
            if formal_mode.get(key) is None:
                raise ContractError(f"Frozen formal mode is missing {key}")


def validate_policy_release_documents(
    policy: Mapping[str, Any],
    *,
    mode: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Open and validate only the public contract/template/fixture documents."""

    validate_policy(policy, mode=mode)
    frozen = policy.get("status") == "FROZEN"
    contract_path = _validate_release_reference(
        policy["contract"],
        label="construction contract",
        frozen=frozen,
    )
    template_path = _validate_release_reference(
        policy["template_library"],
        label="template library",
        frozen=frozen,
    )
    fixture_spec = policy["identity_design"]["role_template_parser_flag_fixture"]
    fixture_path = _validate_release_reference(
        fixture_spec,
        label="parser/template fixture",
        frozen=frozen,
    )
    template = load_json(template_path)
    fixture = load_json(fixture_path)
    if template.get("version") != policy["template_library"]["required_version"]:
        raise ContractError("Template-library version drift")
    if fixture.get("version") != fixture_spec["required_version"]:
        raise ContractError("Parser/template fixture version drift")
    if frozen and fixture.get("status") != "FROZEN_FIXTURE":
        raise ContractError(
            "Frozen release requires a frozen executable parser fixture"
        )
    if not contract_path.read_text(encoding="utf-8").startswith(
        "# Step 28-v13 中文合成身份数据集正式构建合同"
    ):
        raise ContractError("Construction-contract title drift")

    if len(template["style_prototypes"]) != int(
        policy["template_library"]["expected_style_prototypes"]
    ):
        raise ContractError("Style-prototype count drift")
    categories = template["generic_lexicon"]["categories"]
    category_products = template["generic_lexicon"]["category_products"]
    title_modifiers = template["generic_lexicon"].get("title_modifiers")
    if list(category_products) != categories:
        raise ContractError("Category/product mapping order or keyset drift")
    if any(not category_products[category] for category in categories):
        raise ContractError("Every generic category needs at least one product")
    if (
        not isinstance(title_modifiers, list)
        or len(title_modifiers) != 16
        or any(not isinstance(value, str) or not value for value in title_modifiers)
        or len(set(title_modifiers)) != len(title_modifiers)
    ):
        raise ContractError("Title-modifier domain must have 16 unique strings")
    if (
        len(template["generic_lexicon"]["delivery"]) != 6
        or len(template["generic_lexicon"]["service"]) != 6
    ):
        raise ContractError("Fixture-covered delivery/service domains must each have six values")
    expected_splits = {"train", "development", "audit_a", "audit_b"}
    if set(template["split_libraries"]) != expected_splits:
        raise ContractError("Template split-library keyset drift")
    for split, library in template["split_libraries"].items():
        if (
            len(library["title_skeletons"])
            != int(policy["template_library"]["expected_title_skeletons_per_split"])
            or len(library["description_skeletons"])
            != int(policy["template_library"]["expected_description_skeletons_per_split"])
        ):
            raise ContractError(f"Template skeleton count drift for {split}")
        title_skeletons = library["title_skeletons"]
        if (
            sum("{code}" in value for value in title_skeletons) != 4
            or sum("{title_modifier}" in value for value in title_skeletons) != 4
            or any(
                ("{code}" in value) == ("{title_modifier}" in value)
                for value in title_skeletons
            )
        ):
            raise ContractError(
                f"Every {split} title library needs four disjoint code and modifier skeletons"
            )
        suffix = "{noise_clause}{context_guard}{identity_clause}"
        if any(not value.endswith(suffix) for value in library["description_skeletons"]):
            raise ContractError(f"Description skeleton suffix drift for {split}")
    return template, fixture


def validate_independent_replay_public_domains(
    policy: Mapping[str, Any],
    *,
    template: Mapping[str, Any],
    style_profile: Mapping[str, Any],
) -> None:
    """Bind duplicated public replay domains to the producer release inputs."""

    template_contract = policy["template_library"]
    style_ids = [
        str(row["style_id"]) for row in template["style_prototypes"]
    ]
    if list(template_contract["style_prototype_ids"]) != style_ids:
        raise ContractError(
            "Independent replay style-ID domain differs from template"
        )
    domains = policy["independent_replay_public_domains"]
    expected_keys = {
        "purpose",
        "categories_in_registered_order",
        "anonymous_category_rank_probability",
        "category_products",
        "attributes",
        "title_skeleton_count_by_split",
    }
    if set(domains) != expected_keys:
        raise ContractError("Independent replay public-domain schema drift")
    lexicon = template["generic_lexicon"]
    comparisons = (
        (
            domains["categories_in_registered_order"],
            lexicon["categories"],
            "category order",
        ),
        (
            domains["category_products"],
            lexicon["category_products"],
            "category/product mapping",
        ),
        (domains["attributes"], lexicon["attributes"], "attribute order"),
        (
            domains["anonymous_category_rank_probability"],
            style_profile["anonymous_category_rank_probability"],
            "category probabilities",
        ),
        (
            domains["title_skeleton_count_by_split"],
            {
                split: len(library["title_skeletons"])
                for split, library in template["split_libraries"].items()
            },
            "title skeleton counts",
        ),
    )
    for observed, expected, label in comparisons:
        if canonical_json_bytes(observed) != canonical_json_bytes(expected):
            raise ContractError(
                f"Independent replay public {label} differs from producer input"
            )


def load_policy(
    path: Path = DEFAULT_POLICY_PATH,
    *,
    mode: str = "development_smoke",
) -> dict[str, Any]:
    policy = load_json(path)
    validate_policy(policy, mode=mode)
    return policy


def structure_key_for_split(
    policy: Mapping[str, Any],
    *,
    mode: str,
    split: str,
) -> str:
    """Return only the requested split structure key and verify formal custody."""

    if split not in {"train", "development", "audit_a", "audit_b"}:
        raise ContractError(f"Unknown split for structure-key access: {split}")
    stream = policy["randomness"][mode]
    if mode in {"development_smoke", "training_ready"}:
        return _validate_hex_key(
            stream["structure_key_hex"],
            label=f"{mode} structure key",
        )
    if mode != "formal":
        raise ContractError(f"Unsupported structure-key mode: {mode}")
    if policy.get("status") != "FROZEN" or not policy.get("formal_generation_enabled"):
        raise ContractError("Formal structure-key access is disabled before release freeze")

    custody = stream["label_bearing_structure_keys"]
    split_spec = custody[split]
    variable = str(split_spec["environment_variable"])
    raw_value = os.environ.get(variable)
    if raw_value is None:
        raise ContractError(f"Formal split custody did not provide {variable}")
    key_hex = _validate_hex_key(raw_value, label=f"formal {split} structure key")
    public_key_hexes: set[str] = set()
    for namespace in ("formal", "development_smoke"):
        public_stream = policy["randomness"][namespace]
        public_key_hexes.update(
            {
                str(public_stream["id_namespace_key_hex"]).lower(),
                str(public_stream["id_key_hex"]).lower(),
                str(public_stream["identity_value_key_hex"]).lower(),
                str(public_stream["text_key_hex"]).lower(),
                str(public_stream["candidate_key_hex"]).lower(),
                str(public_stream["query_key_hex"]).lower(),
                *(
                    str(value).lower()
                    for value in public_stream["rewire_key_hexes"]
                ),
            }
        )
        if namespace == "development_smoke":
            public_key_hexes.add(str(public_stream["structure_key_hex"]).lower())
    if key_hex in public_key_hexes:
        raise ContractError(
            f"Formal {split} structure key collides with a public random key"
        )
    commitment = sha256_bytes(bytes.fromhex(key_hex))
    if commitment != str(split_spec["sha256_commitment"]).lower():
        raise ContractError(f"Formal {split} structure-key commitment mismatch")
    compromised = {
        str(value).lower()
        for value in custody["compromised_draft_key_commitments_forbidden"]
    }
    if commitment in compromised:
        raise ContractError(f"Formal {split} structure key uses a compromised draft key")

    other_variables = {
        str(custody[name]["environment_variable"])
        for name in ("train", "development", "audit_a", "audit_b")
        if name != split
    }
    present = sorted(name for name in other_variables if os.environ.get(name) is not None)
    if present:
        raise ContractError(
            f"Split custody process contains other formal structure keys: {present}"
        )
    return key_hex


class DeterministicRng:
    """Counter-mode HMAC random stream with unbiased bounded integers."""

    def __init__(self, key_hex: str, *context: str) -> None:
        self._key = bytes.fromhex(key_hex)
        if len(self._key) != 32:
            raise ContractError("HMAC key must be 32 bytes")
        if not context or any(FIELD_SEPARATOR in item.encode("utf-8") for item in context):
            raise ContractError("Random stream context is missing or contains the separator")
        self._context = FIELD_SEPARATOR.join(item.encode("utf-8") for item in context)
        self._counter = 0

    def _block(self) -> bytes:
        counter = self._counter
        self._counter += 1
        message = self._context + FIELD_SEPARATOR + counter.to_bytes(16, "big")
        return hmac.new(self._key, message, hashlib.sha256).digest()

    def randbelow(self, upper: int) -> int:
        if upper <= 0:
            raise ContractError("randbelow upper bound must be positive")
        limit = (1 << 256) - ((1 << 256) % upper)
        while True:
            value = int.from_bytes(self._block(), "big")
            if value < limit:
                return value % upper

    def uint64(self) -> int:
        """Return an exact unsigned 64-bit draw."""

        return int.from_bytes(self._block()[:8], "big")

    def uniform01_decimal(self) -> Decimal:
        """Return the exact rational uint64/2^64 as a high-precision Decimal."""

        with localcontext() as context:
            context.prec = 80
            return Decimal(self.uint64()) / Decimal(1 << 64)

    def bernoulli(self, probability: float | str | Decimal) -> bool:
        try:
            exact_probability = Decimal(str(probability))
        except Exception as exc:  # pragma: no cover - defensive input boundary
            raise ContractError("Invalid Bernoulli probability") from exc
        if (
            not exact_probability.is_finite()
            or exact_probability < 0
            or exact_probability > 1
        ):
            raise ContractError("Invalid Bernoulli probability")
        with localcontext() as context:
            context.prec = 80
            threshold = int(
                (exact_probability * Decimal(1 << 64)).to_integral_value(
                    rounding=ROUND_FLOOR
                )
            )
        return self.uint64() < threshold

    def choice(self, values: Sequence[Any]) -> Any:
        if not values:
            raise ContractError("Cannot choose from an empty sequence")
        return values[self.randbelow(len(values))]

    def sample(self, values: Sequence[Any], count: int) -> list[Any]:
        if count < 0 or count > len(values):
            raise ContractError("Invalid deterministic sample size")
        pool = list(values)
        for index in range(count):
            swap_index = index + self.randbelow(len(pool) - index)
            pool[index], pool[swap_index] = pool[swap_index], pool[index]
        return pool[:count]

    def shuffled(self, values: Sequence[Any]) -> list[Any]:
        return self.sample(values, len(values))


def hmac_digest(key_hex: str, *parts: str) -> bytes:
    if not parts or any(FIELD_SEPARATOR in part.encode("utf-8") for part in parts):
        raise ContractError("HMAC message parts are missing or contain the separator")
    return hmac.new(
        bytes.fromhex(key_hex),
        FIELD_SEPARATOR.join(part.encode("utf-8") for part in parts),
        hashlib.sha256,
    ).digest()


def opaque_uid(prefix: str, key_hex: str, *parts: str) -> str:
    if not prefix or "|" in prefix:
        raise ContractError("Invalid opaque UID prefix")
    return f"{prefix}_{hmac_digest(key_hex, *parts).hex()}"


def utf8_sort(values: Iterable[str]) -> list[str]:
    return sorted(values, key=lambda value: value.encode("utf-8"))


def canonical_pair_uid(left_uid: str, right_uid: str) -> str:
    if left_uid == right_uid:
        raise ContractError("Self-pairs are forbidden")
    if "|" in left_uid or "|" in right_uid:
        raise ContractError("Seller UID contains the reserved pair delimiter")
    left, right = utf8_sort([left_uid, right_uid])
    return f"{left}||{right}"


def query_uid(world_uid: str, seller_uid: str) -> str:
    return "qry_" + sha256_bytes(
        world_uid.encode("utf-8") + FIELD_SEPARATOR + seller_uid.encode("utf-8")
    )


def source_dataset_name(
    policy: Mapping[str, Any],
    *,
    mode: str,
    split: str,
) -> str:
    """Return the registered synthetic source name for parser/profile rows."""

    mode_spec = policy["modes"][mode]
    prefix = mode_spec.get("source_dataset_prefix")
    if prefix is None:
        prefix = f"step28_v13_{mode}"
    prefix = str(prefix)
    if (
        not prefix
        or any(character.isspace() for character in prefix)
        or split not in {"train", "development", "audit_a", "audit_b"}
    ):
        raise ContractError("Synthetic source-dataset name contract drift")
    return f"{prefix}_{split}"


def relation_uid(query_identifier: str, gallery_seller_uid: str) -> str:
    return "rel_" + sha256_bytes(
        query_identifier.encode("utf-8")
        + FIELD_SEPARATOR
        + gallery_seller_uid.encode("utf-8")
    )


def _atomic_replace(path: Path, payload: bytes) -> None:
    parent = filesystem_path(path.parent)
    target = filesystem_path(path)
    os.makedirs(parent, exist_ok=True)
    if os.path.exists(target):
        with open(target, "rb") as handle:
            observed = handle.read()
        if observed == payload:
            return
        raise FileExistsError(f"Refusing to overwrite an existing artifact with new bytes: {path}")
    descriptor, temporary_name = tempfile.mkstemp(
        # Keep the temporary basename independent of the destination name.
        # Repeating a long audit filename here pushes an otherwise valid
        # staging target beyond the legacy Windows MAX_PATH boundary.
        prefix=".tmp-",
        suffix=".part",
        dir=parent,
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        atomic_rename_no_replace(Path(temporary_name), path)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


def write_json(path: Path, value: Any) -> None:
    payload = (
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        + "\n"
    ).encode("utf-8")
    _atomic_replace(path, payload)


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    payload = b"".join(canonical_json_bytes(dict(row)) + b"\n" for row in rows)
    _atomic_replace(path, payload)


def csv_bytes(rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str]) -> bytes:
    import io

    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(
        buffer,
        fieldnames=list(fieldnames),
        extrasaction="raise",
        lineterminator="\n",
    )
    writer.writeheader()
    for row in rows:
        if set(row) != set(fieldnames):
            missing = set(fieldnames) - set(row)
            extra = set(row) - set(fieldnames)
            raise ContractError(f"CSV schema mismatch missing={sorted(missing)} extra={sorted(extra)}")
        writer.writerow(row)
    return buffer.getvalue().encode("utf-8")


def write_csv(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
    fieldnames: Sequence[str],
) -> None:
    _atomic_replace(path, csv_bytes(rows, fieldnames))


def artifact_record(path: Path, *, role: str, root: Path) -> dict[str, Any]:
    resolved_root = root.resolve()
    resolved_path = path.resolve()
    relative = resolved_path.relative_to(resolved_root).as_posix()
    return {
        "role": role,
        "path": relative,
        "size_bytes": os.stat(filesystem_path(path)).st_size,
        "sha256": sha256_file(path),
    }


def mode_output_root(policy: Mapping[str, Any], mode: str) -> Path:
    output = repo_path(str(policy["modes"][mode]["output_root"]))
    if output.exists() and output.is_symlink():
        raise ContractError("Output root may not be a symlink")
    return output


def add_mode_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--mode",
        choices=("development_smoke", "formal"),
        default="development_smoke",
    )


def add_policy_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--policy",
        type=Path,
        default=DEFAULT_POLICY_PATH,
    )
