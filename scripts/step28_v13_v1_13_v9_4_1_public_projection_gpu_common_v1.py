#!/usr/bin/env python3
"""Minimal split-blind contract used inside the isolated GPU workspace."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[1]
POLICY_PATH = (
    ROOT
    / "schema/step28_v13_v1_13_v9_4_1_public_projection_gpu_policy_v1.json"
)
POLICY_SIZE_BYTES = 3571
POLICY_SHA256 = "80559b3594ae5a7d9beb5358d0cc0802cc81ee38ba09fb0afc67e2060fb57863"
POLICY_CANONICAL_SELF_HASH = (
    "c9d45ec1b5781c4cce3f7209aa6adb8072f82e676b96db2d7a0439d56606831f"
)


class GPUProjectionContractError(ValueError):
    """Raised when the isolated opaque encoding contract is violated."""


# Keep the encoder's generic error spelling without importing the full public
# policy module (which contains semantic split names).
PublicProjectionContractError = GPUProjectionContractError


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve(relative: str) -> Path:
    path = (ROOT / relative).resolve()
    try:
        path.relative_to(ROOT.resolve())
    except ValueError as exc:
        raise GPUProjectionContractError("GPU path escapes the workspace") from exc
    return path


def load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise GPUProjectionContractError(f"Invalid UTF-8 JSON: {path}") from exc
    if not isinstance(value, dict):
        raise GPUProjectionContractError(f"JSON is not an object: {path}")
    return value


def verify_file_record(spec: Mapping[str, Any], *, label: str) -> Path:
    path = resolve(str(spec["path"]))
    if (
        not path.is_file()
        or path.stat().st_size != int(spec["size_bytes"])
        or sha256_file(path) != spec["sha256"]
    ):
        raise GPUProjectionContractError(f"{label} file pin drift")
    return path


def _validate_semantics(policy: Mapping[str, Any]) -> None:
    if (
        policy.get("version")
        != "step28-v13-v1.13-v9.4.1-public-projection-gpu-v1"
        or policy.get("status") != "OPAQUE_FOUR_PART_ENCODING_IMPLEMENTATION_ONLY"
        or policy.get("parent_training_core_commit")
        != "49ffff89e01818e4f1de58a641195a0d8ef95c3e"
        or any(policy["permissions"].values())
    ):
        raise GPUProjectionContractError("GPU policy identity/permission drift")
    parts = policy["part_contract"]
    if (
        parts["part_ids"] != [f"part_{index:03d}" for index in range(4)]
        or parts["seller_count_per_part"] != 14000
        or parts["pair_count_per_part"] != 189000
        or parts["allowed_root_files"] != ["transfer_manifest.json"]
        or parts["allowed_part_files"]
        != [
            "opaque_unique_texts.jsonl",
            "opaque_seller_text_index.jsonl",
            "opaque_pair_endpoints.csv",
        ]
        or parts["return_root_files"] != ["gpu_return_manifest.json"]
        or parts["return_part_files"] != ["labse6.npy", "labse6_manifest.json"]
    ):
        raise GPUProjectionContractError("GPU part/file universe drift")
    labse = policy["labse_contract"]
    if (
        len(labse["feature_names"]) != 6
        or canonical_sha256(labse["feature_names"])
        != labse["feature_name_canonical_sha256"]
        or labse["output_shape_per_part"] != [189000, 6]
        or labse["output_dtype"] != "<f8"
        or labse["serialized_decimal_places"] != 12
        or labse["temporary_chunks_retained"] is not False
        or labse["temporary_embeddings_retained"] is not False
    ):
        raise GPUProjectionContractError("GPU LaBSE contract drift")
    step7_path = verify_file_record(policy["step7_policy"], label="Step7 policy")
    step7 = load_json(step7_path)
    payloads = policy["model_payloads"]
    if tuple(payloads) != tuple(step7["embedding_models"]):
        raise GPUProjectionContractError("GPU tokenizer payload registry drift")
    for key, pin in payloads.items():
        cfg = step7["embedding_models"][key]
        if (
            cfg["local_path"] != pin["path"]
            or cfg["expected_file_count"] != pin["file_count"]
            or cfg["expected_total_size_bytes"] != pin["total_size_bytes"]
            or cfg["expected_content_sha256"] != pin["content_sha256"]
        ):
            raise GPUProjectionContractError(f"GPU model payload pin drift for {key}")


def load_policy(path: Path = POLICY_PATH) -> dict[str, Any]:
    if path.resolve() != POLICY_PATH.resolve():
        raise GPUProjectionContractError("Only the default GPU policy is valid")
    raw = path.read_bytes()
    if (
        len(raw) != POLICY_SIZE_BYTES
        or hashlib.sha256(raw).hexdigest() != POLICY_SHA256
    ):
        raise GPUProjectionContractError("GPU policy raw-byte pin drift")
    policy = load_json(path)
    claimed = policy.get("canonical_self_hash")
    body = dict(policy)
    body.pop("canonical_self_hash", None)
    if claimed != POLICY_CANONICAL_SELF_HASH or canonical_sha256(body) != claimed:
        raise GPUProjectionContractError("GPU policy canonical self-hash drift")
    _validate_semantics(policy)
    return policy
