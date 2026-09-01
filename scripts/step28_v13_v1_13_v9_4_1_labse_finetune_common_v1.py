#!/usr/bin/env python3
"""Shared, label-free data contracts for direct Chinese LaBSE fine-tuning."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
POLICY_PATH = (
    ROOT
    / "schema"
    / "step28_v13_v1_13_v9_4_1_labse_direct_finetune_v1_policy.json"
)
EXPECTED_VERSION = "step28-v13-v1.13-v9.4.1-labse-direct-finetune-v1"
FIELDS = ("title", "description")
REDACTED_ITEM_FIELDS = ("world_uid", "seller_uid", "item_uid", "title", "description")
PREFERRED_BOUNDARY_PATTERN = re.compile(r"[\n\r\t ，。！？；：、,.!?;:)]")


class LabseFinetuneContractError(ValueError):
    """Raised when direct-fine-tune public inputs violate their contract."""


def canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def canonical_self_hash(value: Mapping[str, Any]) -> str:
    payload = dict(value)
    payload.pop("canonical_self_hash", None)
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_policy(path: Path = POLICY_PATH) -> dict[str, Any]:
    if path.resolve() != POLICY_PATH.resolve():
        raise LabseFinetuneContractError("Only the default fine-tune policy is valid")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("version") != EXPECTED_VERSION:
        raise LabseFinetuneContractError("Fine-tune policy identity drift")
    if value.get("canonical_self_hash") != canonical_self_hash(value):
        raise LabseFinetuneContractError("Fine-tune policy canonical self-hash drift")
    if (
        value.get("formal_gpu_training_authorized") is not False
        or value.get("audit_a_truth_authorized") is not False
        or value.get("audit_b_truth_authorized") is not False
        or value.get("forbidden_splits") != ["audit_a", "audit_b"]
        or value.get("allowed_splits") != ["train", "development"]
    ):
        raise LabseFinetuneContractError("Fine-tune truth or split boundary drift")
    text = value["text_input"]
    if (
        text["file_role"] != "redacted_items.jsonl"
        or text["fields"] != list(FIELDS)
        or text["identifier_redacted_text_only"] is not True
        or text["whole_document_truncation_allowed"] is not False
        or text["maximum_chunks_per_text"] is not None
        or text["exact_character_reconstruction_required"] is not True
        or int(text["token_budget_including_special_tokens"]) != 256
    ):
        raise LabseFinetuneContractError("Fine-tune full-text contract drift")
    budgets = value["training_budgets"]
    if (
        budgets["world_counts"] != [5, 25, 50, 125, 500]
        or budgets["percent_labels"] != [1, 5, 10, 25, 100]
        or budgets["labels_or_results_used_for_world_selection"] is not False
    ):
        raise LabseFinetuneContractError("Fine-tune budget contract drift")
    for label, spec in value["frozen_inputs"].items():
        pinned = ROOT / str(spec["path"])
        if (
            not pinned.is_file()
            or pinned.stat().st_size != int(spec["size_bytes"])
            or sha256_file(pinned) != str(spec["sha256"])
        ):
            raise LabseFinetuneContractError(f"Fine-tune frozen input drift: {label}")
    return value


def verify_labse_payload(policy: Mapping[str, Any]) -> dict[str, Any]:
    """Verify the exact local LaBSE bytes used by the frozen English baseline."""

    import step7_v4_common as step7_common

    spec = policy["labse_model"]
    observed = step7_common.model_content_fingerprint(
        ROOT / str(spec["path"])
    )
    expected = {
        "file_count": int(spec["file_count"]),
        "total_size_bytes": int(spec["total_size_bytes"]),
        "content_sha256": str(spec["content_sha256"]),
    }
    for field, expected_value in expected.items():
        if observed[field] != expected_value:
            raise LabseFinetuneContractError(
                f"LaBSE payload drift: {field} expected={expected_value} "
                f"observed={observed[field]}"
            )
    return observed


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise LabseFinetuneContractError(
                    f"Invalid JSONL at {path}:{line_number}"
                ) from exc
            if not isinstance(value, dict):
                raise LabseFinetuneContractError(
                    f"Non-object JSONL row at {path}:{line_number}"
                )
            yield value


def _root_manifest(policy: Mapping[str, Any]) -> dict[str, Any]:
    spec = policy["frozen_inputs"]["formal_root_manifest"]
    return json.loads((ROOT / str(spec["path"])).read_text(encoding="utf-8"))


def verified_redacted_items_path(
    policy: Mapping[str, Any], split: str
) -> Path:
    if split not in policy["allowed_splits"]:
        raise LabseFinetuneContractError("Fine-tune attempted a forbidden split")
    relative = f"{split}/observed/redacted_items.jsonl"
    manifest = _root_manifest(policy)
    entries = {
        str(record["path"]): record
        for record in manifest.get("public_files", [])
    }
    if relative not in entries:
        raise LabseFinetuneContractError(
            f"Formal root does not register {relative}"
        )
    path = ROOT / str(policy["public_dataset_root"]) / relative
    spec = entries[relative]
    if (
        not path.is_file()
        or path.stat().st_size != int(spec["size_bytes"])
        or sha256_file(path) != str(spec["sha256"])
    ):
        raise LabseFinetuneContractError(f"Redacted text input drift: {split}")
    return path


def nested_world_order(world_uids: Sequence[str]) -> tuple[str, ...]:
    values = tuple(str(value) for value in world_uids)
    if len(values) != len(set(values)) or any(not value for value in values):
        raise LabseFinetuneContractError("World identifiers are empty or duplicated")
    return tuple(
        sorted(
            values,
            key=lambda value: (
                hashlib.sha256(value.encode("utf-8")).digest(),
                value.encode("utf-8"),
            ),
        )
    )


def nested_world_subsets(
    world_uids: Sequence[str], world_counts: Sequence[int]
) -> dict[int, tuple[str, ...]]:
    order = nested_world_order(world_uids)
    counts = tuple(int(value) for value in world_counts)
    if (
        not counts
        or tuple(sorted(set(counts))) != counts
        or counts[0] <= 0
        or counts[-1] != len(order)
    ):
        raise LabseFinetuneContractError("Nested world-count contract is invalid")
    output = {count: order[:count] for count in counts}
    previous: set[str] = set()
    for count in counts:
        current = set(output[count])
        if not previous <= current or len(current) != count:
            raise AssertionError("World subsets are not nested")
        previous = current
    return output


def build_redacted_text_index(
    rows: Iterable[Mapping[str, Any]],
    *,
    expected_worlds: int,
    expected_sellers_per_world: int,
    expected_items_per_world: int,
) -> dict[str, dict[str, dict[str, tuple[str, ...]]]]:
    texts: dict[str, dict[str, dict[str, set[str]]]] = defaultdict(
        lambda: defaultdict(lambda: {field: set() for field in FIELDS})
    )
    item_uids: set[str] = set()
    item_counts: Counter[str] = Counter()
    for row_number, row in enumerate(rows, start=1):
        if tuple(row) != REDACTED_ITEM_FIELDS:
            raise LabseFinetuneContractError(
                f"Redacted item schema/order drift at row {row_number}"
            )
        world_uid = str(row["world_uid"])
        seller_uid = str(row["seller_uid"])
        item_uid = str(row["item_uid"])
        if not world_uid or not seller_uid or not item_uid or item_uid in item_uids:
            raise LabseFinetuneContractError("Redacted item identity drift")
        item_uids.add(item_uid)
        item_counts[world_uid] += 1
        for field in FIELDS:
            value = row[field]
            if not isinstance(value, str):
                raise LabseFinetuneContractError("Redacted item text is not a string")
            if value.strip():
                texts[world_uid][seller_uid][field].add(value)
    if len(texts) != expected_worlds:
        raise LabseFinetuneContractError("Redacted text world count drift")
    output: dict[str, dict[str, dict[str, tuple[str, ...]]]] = {}
    for world_uid in sorted(texts, key=lambda value: value.encode("utf-8")):
        sellers = texts[world_uid]
        if (
            len(sellers) != expected_sellers_per_world
            or item_counts[world_uid] != expected_items_per_world
        ):
            raise LabseFinetuneContractError(
                f"Redacted text layout drift in {world_uid}"
            )
        output[world_uid] = {}
        for seller_uid in sorted(sellers, key=lambda value: value.encode("utf-8")):
            fields = sellers[seller_uid]
            if any(not fields[field] for field in FIELDS):
                raise LabseFinetuneContractError(
                    f"Seller lacks a nonempty redacted field: {seller_uid}"
                )
            output[world_uid][seller_uid] = {
                field: tuple(
                    sorted(
                        fields[field],
                        key=lambda value: (
                            hashlib.sha256(value.encode("utf-8")).digest(),
                            value.encode("utf-8"),
                        ),
                    )
                )
                for field in FIELDS
            }
    if len(item_uids) != expected_worlds * expected_items_per_world:
        raise LabseFinetuneContractError("Redacted item total count drift")
    return output


def tokenizer_length(tokenizer: Any, text: str) -> int:
    encoded = tokenizer(
        text,
        add_special_tokens=True,
        truncation=False,
        return_attention_mask=False,
        return_token_type_ids=False,
    )
    input_ids = encoded["input_ids"]
    if input_ids and isinstance(input_ids[0], list):
        if len(input_ids) != 1:
            raise LabseFinetuneContractError("Tokenizer returned multiple rows")
        input_ids = input_ids[0]
    return len(input_ids)


def _maximum_fitting_prefix(tokenizer: Any, text: str, budget: int) -> int:
    low, high = 1, len(text)
    if tokenizer_length(tokenizer, text, ) <= budget:
        return len(text)
    while low < high:
        middle = (low + high + 1) // 2
        if tokenizer_length(tokenizer, text[:middle]) <= budget:
            low = middle
        else:
            high = middle - 1
    if tokenizer_length(tokenizer, text[:low]) > budget:
        raise LabseFinetuneContractError("No character fits the tokenizer budget")
    return low


def chunk_text_exact(tokenizer: Any, text: str, budget: int) -> tuple[str, ...]:
    if not isinstance(text, str) or not text.strip() or budget <= 2:
        raise LabseFinetuneContractError("Cannot chunk empty text or invalid budget")
    chunks: list[str] = []
    position = 0
    while position < len(text):
        remaining = text[position:]
        maximum = _maximum_fitting_prefix(tokenizer, remaining, budget)
        if maximum == len(remaining):
            end = maximum
        else:
            candidates = [
                match.end()
                for match in PREFERRED_BOUNDARY_PATTERN.finditer(remaining[:maximum])
                if remaining[: match.end()].strip()
            ]
            end = candidates[-1] if candidates else maximum
        chunk = remaining[:end]
        if (
            not chunk
            or not chunk.strip()
            or tokenizer_length(tokenizer, chunk) > budget
        ):
            raise LabseFinetuneContractError("Fine-tune chunk violates its budget")
        chunks.append(chunk)
        position += len(chunk)
    if "".join(chunks) != text or any(
        tokenizer_length(tokenizer, chunk) > budget for chunk in chunks
    ):
        raise LabseFinetuneContractError("Fine-tune chunks do not reconstruct text")
    return tuple(chunks)


def _unit(vector: np.ndarray) -> np.ndarray:
    value = np.asarray(vector, dtype=np.float64)
    norm = float(np.linalg.norm(value))
    if value.ndim != 1 or not math.isfinite(norm) or norm <= 1e-12:
        raise LabseFinetuneContractError("Cannot normalize a zero/non-finite vector")
    return value / norm


def unit_mean(matrix: np.ndarray) -> np.ndarray:
    values = np.asarray(matrix, dtype=np.float64)
    if values.ndim != 2 or len(values) == 0 or not np.isfinite(values).all():
        raise LabseFinetuneContractError("Unit mean needs a finite nonempty matrix")
    return _unit(np.mean(np.vstack([_unit(row) for row in values]), axis=0))


def symmetric_top_k(left: np.ndarray, right: np.ndarray, k: int) -> float:
    left_values = np.vstack([_unit(row) for row in np.asarray(left)])
    right_values = np.vstack([_unit(row) for row in np.asarray(right)])
    similarities = left_values @ right_values.T
    left_k = min(k, similarities.shape[1])
    right_k = min(k, similarities.shape[0])
    left_scores = np.mean(np.partition(similarities, -left_k, axis=1)[:, -left_k:])
    right_scores = np.mean(
        np.partition(similarities.T, -right_k, axis=1)[:, -right_k:]
    )
    return float((left_scores + right_scores) / 2.0)


def six_pair_aggregates(
    left: Mapping[str, np.ndarray], right: Mapping[str, np.ndarray], top_k: int = 3
) -> np.ndarray:
    values: dict[str, tuple[float, float]] = {}
    for field in FIELDS:
        left_matrix = np.asarray(left[field], dtype=np.float64)
        right_matrix = np.asarray(right[field], dtype=np.float64)
        values[field] = (
            float(_unit(unit_mean(left_matrix)) @ _unit(unit_mean(right_matrix))),
            symmetric_top_k(left_matrix, right_matrix, top_k),
        )
    result = np.asarray(
        [
            np.mean([values[field][0] for field in FIELDS]),
            np.mean([values[field][1] for field in FIELDS]),
            values["title"][0],
            values["title"][1],
            values["description"][0],
            values["description"][1],
        ],
        dtype="<f8",
    )
    if result.shape != (6,) or not np.isfinite(result).all():
        raise LabseFinetuneContractError("Six-feature aggregation failed")
    return result


def torch_unit_mean(matrix: Any) -> Any:
    """Differentiable equivalent of :func:`unit_mean`."""

    import torch
    import torch.nn.functional as functional

    if matrix.ndim != 2 or matrix.shape[0] <= 0:
        raise LabseFinetuneContractError("Torch unit mean needs a nonempty matrix")
    normalized = functional.normalize(matrix, p=2, dim=1, eps=1e-12)
    return functional.normalize(normalized.mean(dim=0), p=2, dim=0, eps=1e-12)


def torch_symmetric_top_k(left: Any, right: Any, k: int) -> Any:
    import torch.nn.functional as functional

    if left.ndim != 2 or right.ndim != 2 or len(left) == 0 or len(right) == 0:
        raise LabseFinetuneContractError("Torch top-k needs two nonempty matrices")
    left_values = functional.normalize(left, p=2, dim=1, eps=1e-12)
    right_values = functional.normalize(right, p=2, dim=1, eps=1e-12)
    similarities = left_values @ right_values.transpose(0, 1)
    left_k = min(int(k), similarities.shape[1])
    right_k = min(int(k), similarities.shape[0])
    left_score = similarities.topk(left_k, dim=1).values.mean()
    right_score = similarities.transpose(0, 1).topk(right_k, dim=1).values.mean()
    return (left_score + right_score) / 2.0


def torch_six_pair_aggregates(
    left: Mapping[str, Any], right: Mapping[str, Any], top_k: int = 3
) -> Any:
    import torch

    values: dict[str, tuple[Any, Any]] = {}
    for field in FIELDS:
        left_matrix = left[field]
        right_matrix = right[field]
        left_centroid = torch_unit_mean(left_matrix)
        right_centroid = torch_unit_mean(right_matrix)
        values[field] = (
            torch.sum(left_centroid * right_centroid),
            torch_symmetric_top_k(left_matrix, right_matrix, top_k),
        )
    result = torch.stack(
        (
            torch.stack([values[field][0] for field in FIELDS]).mean(),
            torch.stack([values[field][1] for field in FIELDS]).mean(),
            values["title"][0],
            values["title"][1],
            values["description"][0],
            values["description"][1],
        )
    )
    if result.shape != (6,):
        raise LabseFinetuneContractError("Torch six-feature aggregation failed")
    return result
