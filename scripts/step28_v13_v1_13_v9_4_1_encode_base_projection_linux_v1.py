#!/usr/bin/env python3
"""Encode an opaque four-part LaBSE transfer in an isolated Linux workspace.

The command line remains validation-only until a later exact-commit one-time
authorization wrapper is frozen.  The encoder never receives canonical split,
world, seller, item, pair, identity, or supervision fields.
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import math
import os
import platform
import re
import shutil
import tempfile
from pathlib import Path
from typing import Any, Mapping

import numpy as np

import step28_v13_v1_13_v9_4_1_public_projection_gpu_common_v1 as common
import step7_v4_common as step7_common
import step7_v4_encode_item_models as step7_encoder


PART_PATTERN = re.compile(r"part_[0-9]{3}\Z")
SELLER_PATTERN = re.compile(r"seller_[0-9]{6}\Z")
PAIR_PATTERN = re.compile(r"pair_[0-9]{6}\Z")
GPU_TEXT_FIELDS = ("text_uid", "text", "text_sha256")
GPU_SELLER_TEXT_FIELDS = ("seller_uid", "field_name", "text_uid", "multiplicity")
GPU_PAIR_FIELDS = ("pair_uid", "seller_uid_left", "seller_uid_right")


def load_json(path: Path) -> dict[str, Any]:
    return common.load_json(path)


def verify_self_hash(value: Mapping[str, Any], *, label: str) -> None:
    claimed = value.get("canonical_self_hash")
    body = dict(value)
    body.pop("canonical_self_hash", None)
    if not isinstance(claimed, str) or common.canonical_sha256(body) != claimed:
        raise common.PublicProjectionContractError(f"{label} self-hash drift")


def iter_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                raise common.PublicProjectionContractError(
                    f"Blank opaque JSONL line at {path}:{line_number}"
                )
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise common.PublicProjectionContractError(
                    f"Invalid opaque JSONL at {path}:{line_number}"
                ) from exc
            if not isinstance(value, dict):
                raise common.PublicProjectionContractError("Opaque JSONL row is not an object")
            rows.append(value)
    return rows


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != GPU_PAIR_FIELDS:
            raise common.PublicProjectionContractError("Opaque pair CSV schema drift")
        return list(reader)


def file_record(path: Path, root: Path) -> dict[str, Any]:
    return {
        "path": path.relative_to(root).as_posix(),
        "size_bytes": path.stat().st_size,
        "sha256": common.sha256_file(path),
    }


def render_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def matrix_value_sha256(matrix: np.ndarray) -> str:
    return hashlib.sha256(
        np.ascontiguousarray(matrix).tobytes(order="C")
    ).hexdigest()


def _verify_record(root: Path, spec: Mapping[str, Any], expected: str) -> Path:
    if spec.get("path") != expected:
        raise common.PublicProjectionContractError("Opaque transfer path drift")
    path = (root / expected).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError as exc:
        raise common.PublicProjectionContractError("Opaque path escapes transfer root") from exc
    if (
        not path.is_file()
        or path.stat().st_size != int(spec.get("size_bytes", -1))
        or common.sha256_file(path) != spec.get("sha256")
    ):
        raise common.PublicProjectionContractError("Opaque transfer file pin drift")
    return path


def validate_transfer(
    policy: Mapping[str, Any], transfer_root: Path
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if not transfer_root.is_dir():
        raise FileNotFoundError(f"Opaque transfer root is missing: {transfer_root}")
    manifest_path = transfer_root / "transfer_manifest.json"
    manifest = load_json(manifest_path)
    verify_self_hash(manifest, label="opaque transfer manifest")
    if (
        manifest.get("status")
        != "FROZEN_OPAQUE_LABEL_FREE_TRANSFER_NO_CANONICAL_IDENTITIES"
        or manifest.get("gpu_policy_canonical_self_hash")
        != policy["canonical_self_hash"]
        or re.fullmatch(
            r"[0-9a-f]{64}",
            str(manifest.get("public_policy_canonical_self_hash", "")),
        )
        is None
        or manifest.get("part_count") != 4
        or manifest.get("split_names_or_canonical_identifiers_present") is not False
        or manifest.get("labels_controllers_membership_qrels_or_audit_truth_present")
        is not False
        or manifest.get("identity33_or_legacy18_present") is not False
    ):
        raise common.PublicProjectionContractError("Opaque transfer manifest drift")
    parts = manifest.get("parts")
    if not isinstance(parts, list) or [row.get("part_id") for row in parts] != [
        f"part_{index:03d}" for index in range(4)
    ]:
        raise common.PublicProjectionContractError("Opaque transfer part order drift")
    expected_paths = {"transfer_manifest.json"}
    validated = []
    expected_counts = policy["part_contract"]
    for part in parts:
        part_id = str(part["part_id"])
        if PART_PATTERN.fullmatch(part_id) is None:
            raise common.PublicProjectionContractError("Invalid opaque part ID")
        records = part.get("files")
        if not isinstance(records, dict) or tuple(records) != (
            "opaque_unique_texts",
            "opaque_seller_text_index",
            "opaque_pair_endpoints",
        ):
            raise common.PublicProjectionContractError("Opaque transfer registry drift")
        paths = {
            role: _verify_record(
                transfer_root,
                spec,
                f"{part_id}/{filename}",
            )
            for (role, filename), spec in zip(
                (
                    ("opaque_unique_texts", "opaque_unique_texts.jsonl"),
                    ("opaque_seller_text_index", "opaque_seller_text_index.jsonl"),
                    ("opaque_pair_endpoints", "opaque_pair_endpoints.csv"),
                ),
                records.values(),
                strict=True,
            )
        }
        expected_paths.update(str(spec["path"]) for spec in records.values())
        text_rows = iter_jsonl(paths["opaque_unique_texts"])
        seller_rows = iter_jsonl(paths["opaque_seller_text_index"])
        pair_rows = read_csv(paths["opaque_pair_endpoints"])
        if (
            len(text_rows) != int(part["unique_text_count"])
            or len(seller_rows) != int(part["seller_text_row_count"])
            or len(pair_rows) != int(part["opaque_pair_count"])
            or len(pair_rows) != expected_counts["pair_count_per_part"]
            or int(part["opaque_seller_count"])
            != expected_counts["seller_count_per_part"]
        ):
            raise common.PublicProjectionContractError("Opaque workload count drift")
        text_uids: set[str] = set()
        for row in text_rows:
            if tuple(row) != GPU_TEXT_FIELDS:
                raise common.PublicProjectionContractError("Opaque text schema drift")
            text_uid = str(row["text_uid"])
            text = row["text"]
            if (
                not isinstance(text, str)
                or not text.strip()
                or row["text_sha256"] != text_uid
                or hashlib.sha256(text.encode("utf-8")).hexdigest() != text_uid
                or text_uid in text_uids
            ):
                raise common.PublicProjectionContractError("Opaque text identity drift")
            text_uids.add(text_uid)
        seller_uids: set[str] = set()
        seller_text_keys: set[tuple[str, str, str]] = set()
        for row in seller_rows:
            if tuple(row) != GPU_SELLER_TEXT_FIELDS:
                raise common.PublicProjectionContractError("Opaque seller-text schema drift")
            seller_uid = str(row["seller_uid"])
            key = (seller_uid, str(row["field_name"]), str(row["text_uid"]))
            if (
                SELLER_PATTERN.fullmatch(seller_uid) is None
                or row["field_name"] not in ("title", "description")
                or row["text_uid"] not in text_uids
                or row["multiplicity"] != 1
                or key in seller_text_keys
            ):
                raise common.PublicProjectionContractError("Opaque seller-text identity drift")
            seller_uids.add(seller_uid)
            seller_text_keys.add(key)
        if len(seller_uids) != expected_counts["seller_count_per_part"]:
            raise common.PublicProjectionContractError("Opaque seller universe drift")
        pair_uids: set[str] = set()
        for ordinal, row in enumerate(pair_rows, start=1):
            expected_pair_uid = f"pair_{ordinal:06d}"
            if (
                tuple(row) != GPU_PAIR_FIELDS
                or row["pair_uid"] != expected_pair_uid
                or PAIR_PATTERN.fullmatch(row["pair_uid"]) is None
                or row["pair_uid"] in pair_uids
                or row["seller_uid_left"] not in seller_uids
                or row["seller_uid_right"] not in seller_uids
                or row["seller_uid_left"] == row["seller_uid_right"]
            ):
                raise common.PublicProjectionContractError("Opaque pair identity/order drift")
            pair_uids.add(row["pair_uid"])
        validated.append(
            {
                "part_id": part_id,
                "text_rows": text_rows,
                "seller_rows": seller_rows,
                "pair_rows": pair_rows,
                "input_files": {
                    role: file_record(path, transfer_root)
                    for role, path in paths.items()
                },
            }
        )
    actual_paths = {
        path.relative_to(transfer_root).as_posix()
        for path in transfer_root.rglob("*")
        if path.is_file()
    }
    if actual_paths != expected_paths:
        raise common.PublicProjectionContractError("Opaque transfer file universe drift")
    return manifest, validated


def preprocess_digest_and_lengths(
    model, texts: list[str], prefix: str, *, batch_size: int = 128
) -> tuple[str, list[int]]:
    preprocess = getattr(model, "preprocess", None)
    if not callable(preprocess):
        raise common.PublicProjectionContractError("SentenceTransformer lacks preprocess()")
    digest = hashlib.sha256()
    lengths: list[int] = []
    for start in range(0, len(texts), int(batch_size)):
        batch = [prefix + text for text in texts[start : start + int(batch_size)]]
        features = preprocess(batch)
        if not isinstance(features, Mapping) or not {
            "input_ids",
            "attention_mask",
        }.issubset(features):
            raise common.PublicProjectionContractError("Preprocess output drift")
        input_ids = step7_encoder._tensor_to_numpy(features["input_ids"])
        attention = step7_encoder._tensor_to_numpy(features["attention_mask"])
        if input_ids.ndim != 2 or attention.shape != input_ids.shape:
            raise common.PublicProjectionContractError("Preprocess tensor shape drift")
        for ids, mask in zip(input_ids, attention, strict=True):
            active = np.flatnonzero(np.asarray(mask) != 0)
            if active.size <= 0 or not np.array_equal(
                active, np.arange(active[0], active[-1] + 1)
            ):
                raise common.PublicProjectionContractError("Attention-mask layout drift")
            values = [int(value) for value in np.asarray(ids)[active]]
            lengths.append(len(values))
            digest.update(len(values).to_bytes(8, "little", signed=False))
            for value in values:
                digest.update(value.to_bytes(8, "little", signed=True))
    return digest.hexdigest(), lengths


def exact_runtime(policy: Mapping[str, Any], torch, transformers, sentence_transformers) -> dict[str, Any]:
    expected = policy["exact_runtime"]
    step7_encoder.configure_deterministic_gpu(torch, step7_common.load_policy())
    observed = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "torch_cuda_runtime": str(torch.version.cuda),
        "cudnn_runtime": torch.backends.cudnn.version(),
        "transformers": transformers.__version__,
        "sentence_transformers": sentence_transformers.__version__,
        "gpu_name": torch.cuda.get_device_name(0),
        "gpu_compute_capability": list(torch.cuda.get_device_capability(0)),
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "tokenizers_parallelism": os.environ.get("TOKENIZERS_PARALLELISM"),
        "cuda_matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
        "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
        "deterministic_algorithms_enabled": bool(
            torch.are_deterministic_algorithms_enabled()
        ),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
    }
    if observed != expected:
        raise common.PublicProjectionContractError(
            f"Exact Linux GPU runtime drift: expected={expected} observed={observed}"
        )
    return observed


def score_part(
    policy: Mapping[str, Any],
    part: Mapping[str, Any],
    step7_policy: Mapping[str, Any],
    tokenizers: Mapping[str, Any],
    model: Any,
) -> tuple[np.ndarray, dict[str, Any]]:
    text_rows = list(part["text_rows"])
    chunks, chunk_audit = step7_encoder.build_shared_chunks(
        step7_policy, text_rows, dict(tokenizers)
    )
    cfg = step7_policy["embedding_models"]["labse"]
    texts = [str(row["text"]) for row in chunks]
    runtime_digest, runtime_lengths = preprocess_digest_and_lengths(
        model, texts, str(cfg["text_prefix"])
    )
    tokenizer_digest, tokenizer_lengths = step7_encoder.tokenizer_digest_and_lengths(
        tokenizers["labse"], texts, str(cfg["text_prefix"])
    )
    expected_lengths = [int(row["token_lengths"]["labse"]) for row in chunks]
    maximum_length = max(runtime_lengths, default=0)
    if (
        runtime_digest != tokenizer_digest
        or runtime_lengths != tokenizer_lengths
        or runtime_lengths != expected_lengths
        or maximum_length
        > int(
            step7_policy["shared_chunking"][
                "token_budget_including_model_prefix_and_special_tokens"
            ]
        )
        or maximum_length > int(cfg["native_max_seq_length"])
    ):
        raise common.PublicProjectionContractError(
            "Runtime LaBSE token IDs, lengths, or window drift"
        )
    matrix = model.encode(
        [str(cfg["text_prefix"]) + text for text in texts],
        batch_size=int(cfg["batch_size"]),
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
        prompt=str(cfg["sentence_transformer_prompt"]),
    )
    matrix = np.ascontiguousarray(matrix, dtype=np.float32)
    if (
        matrix.shape != (len(chunks), int(cfg["expected_dimension"]))
        or not np.isfinite(matrix).all()
    ):
        raise common.PublicProjectionContractError("LaBSE embedding matrix drift")
    score_rows = step7_common.compute_pair_score_rows(
        list(part["pair_rows"]),
        list(part["seller_rows"]),
        chunks,
        matrix,
        cfg,
        top_k=int(step7_policy["aggregation"]["top_k_item_matches"]),
        decimal_places=int(step7_policy["aggregation"]["serialized_decimal_places"]),
        similarity_block_rows=int(step7_policy["aggregation"]["similarity_block_rows"]),
    )
    names = list(policy["labse_contract"]["feature_names"])
    values = []
    for expected_pair, row in zip(part["pair_rows"], score_rows, strict=True):
        if row["pair_uid"] != expected_pair["pair_uid"]:
            raise common.PublicProjectionContractError("LaBSE pair order drift")
        output_row = []
        for name in names:
            serialized = row[name]
            if serialized == "":
                output_row.append(float("nan"))
            elif (
                not isinstance(serialized, str)
                or re.fullmatch(r"-?[0-9]+\.[0-9]{12}", serialized) is None
            ):
                raise common.PublicProjectionContractError("LaBSE decimal contract drift")
            else:
                output_row.append(float(serialized))
        values.append(output_row)
    result = np.ascontiguousarray(values, dtype="<f8")
    if result.shape != (len(part["pair_rows"]), 6) or np.isinf(result).any():
        raise common.PublicProjectionContractError("LaBSE six-feature matrix drift")
    audit = {
        "part_id": part["part_id"],
        "chunk_audit": chunk_audit,
        "shared_labse_token_id_stream_sha256": tokenizer_digest,
        "runtime_labse_token_id_stream_sha256": runtime_digest,
        "runtime_token_id_stream_replays_shared_tokenizer": True,
        "embedding_shape": list(matrix.shape),
        "embedding_dtype": matrix.dtype.str,
        "embedding_value_sha256": matrix_value_sha256(matrix),
        "maximum_unit_norm_error": float(
            np.max(np.abs(np.linalg.norm(matrix, axis=1) - 1.0))
        ),
    }
    del matrix, chunks, score_rows
    gc.collect()
    return result, audit


def validate_gpu_return(
    policy: Mapping[str, Any],
    transfer_manifest: Mapping[str, Any],
    transfer_parts: list[Mapping[str, Any]],
    return_root: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    root_path = return_root / "gpu_return_manifest.json"
    root_manifest = load_json(root_path)
    verify_self_hash(root_manifest, label="GPU return manifest")
    if (
        root_manifest.get("status")
        != "FROZEN_OPAQUE_FOUR_PART_LABSE6_RETURN_NO_MODEL_TRAINING"
        or root_manifest.get("gpu_policy_canonical_self_hash")
        != policy["canonical_self_hash"]
        or root_manifest.get("public_policy_canonical_self_hash")
        != transfer_manifest["public_policy_canonical_self_hash"]
        or root_manifest.get("transfer_manifest_canonical_self_hash")
        != transfer_manifest["canonical_self_hash"]
        or root_manifest.get("exact_runtime") != policy["exact_runtime"]
        or root_manifest.get("loaded_model_state")
        != policy["loaded_model_state"]
        or root_manifest.get("temporary_chunks_retained") is not False
        or root_manifest.get("temporary_embeddings_retained") is not False
        or root_manifest.get("canonical_identifiers_or_split_names_read") is not False
        or root_manifest.get(
            "labels_controllers_membership_qrels_or_audit_truth_read"
        )
        is not False
        or root_manifest.get("model_parameters_updated") is not False
    ):
        raise common.PublicProjectionContractError("GPU return root lineage drift")
    part_registry = root_manifest.get("parts")
    if not isinstance(part_registry, list) or [row.get("part_id") for row in part_registry] != [
        row["part_id"] for row in transfer_parts
    ]:
        raise common.PublicProjectionContractError("GPU return part registry drift")
    expected_paths = {"gpu_return_manifest.json"}
    validated: list[dict[str, Any]] = []
    transfer_by_id = {str(row["part_id"]): row for row in transfer_parts}
    for record in part_registry:
        part_id = str(record["part_id"])
        expected_manifest_path = f"{part_id}/labse6_manifest.json"
        manifest_spec = record.get("manifest_file", {})
        manifest_path = _verify_record(return_root, manifest_spec, expected_manifest_path)
        expected_paths.add(expected_manifest_path)
        part_manifest = load_json(manifest_path)
        verify_self_hash(part_manifest, label=f"GPU return {part_id}")
        if (
            part_manifest.get("canonical_self_hash")
            != record.get("manifest_canonical_self_hash")
            or part_manifest.get("status")
            != "FROZEN_OPAQUE_LABSE6_PART_NO_MODEL_TRAINING"
            or part_manifest.get("part_id") != part_id
            or part_manifest.get("gpu_policy_canonical_self_hash")
            != policy["canonical_self_hash"]
            or part_manifest.get("public_policy_canonical_self_hash")
            != transfer_manifest["public_policy_canonical_self_hash"]
            or part_manifest.get("transfer_manifest_canonical_self_hash")
            != transfer_manifest["canonical_self_hash"]
            or part_manifest.get("input_files")
            != transfer_by_id[part_id]["input_files"]
            or part_manifest.get("canonical_identifiers_or_split_names_read") is not False
            or part_manifest.get(
                "labels_controllers_membership_qrels_or_audit_truth_read"
            )
            is not False
            or part_manifest.get("identity33_or_legacy18_read") is not False
            or part_manifest.get("model_parameters_updated") is not False
        ):
            raise common.PublicProjectionContractError("GPU return part lineage drift")
        value_spec = part_manifest.get("labse6_file", {})
        expected_value_path = f"{part_id}/labse6.npy"
        value_path = _verify_record(return_root, value_spec, expected_value_path)
        expected_paths.add(expected_value_path)
        values = np.load(value_path, allow_pickle=False)
        if (
            not isinstance(values, np.ndarray)
            or values.shape != tuple(policy["labse_contract"]["output_shape_per_part"])
            or values.dtype.str != "<f8"
            or not values.flags.c_contiguous
            or np.isinf(values).any()
            or part_manifest.get("labse6_shape") != list(values.shape)
            or part_manifest.get("labse6_dtype") != values.dtype.str
            or part_manifest.get("labse6_value_sha256")
            != matrix_value_sha256(values)
        ):
            raise common.PublicProjectionContractError("GPU return LaBSE matrix drift")
        audit = part_manifest.get("audit", {})
        norm_error = audit.get("maximum_unit_norm_error")
        shared_token_digest = audit.get(
            "shared_labse_token_id_stream_sha256"
        )
        runtime_token_digest = audit.get(
            "runtime_labse_token_id_stream_sha256"
        )
        if (
            audit.get("part_id") != part_id
            or re.fullmatch(r"[0-9a-f]{64}", str(shared_token_digest)) is None
            or runtime_token_digest != shared_token_digest
            or audit.get("runtime_token_id_stream_replays_shared_tokenizer")
            is not True
            or not isinstance(norm_error, (int, float))
            or isinstance(norm_error, bool)
            or not math.isfinite(float(norm_error))
            or not 0.0 <= float(norm_error) <= 1e-3
        ):
            raise common.PublicProjectionContractError("GPU return audit drift")
        validated.append(
            {
                "part_id": part_id,
                "values": np.ascontiguousarray(values, dtype="<f8"),
                "manifest": part_manifest,
                "manifest_file": dict(manifest_spec),
            }
        )
    actual_paths = {
        path.relative_to(return_root).as_posix()
        for path in return_root.rglob("*")
        if path.is_file()
    }
    if actual_paths != expected_paths:
        raise common.PublicProjectionContractError("GPU return file universe drift")
    return root_manifest, validated


def encode_transfer_to_temporary(
    policy: Mapping[str, Any], transfer_root: Path, return_root: Path
) -> dict[str, Any]:
    """Build an unpublished GPU return for a future authorized wrapper."""

    if return_root.exists():
        raise common.PublicProjectionContractError("GPU return root already exists")
    transfer_manifest, parts = validate_transfer(policy, transfer_root)
    step7_policy = step7_common.load_policy()
    for model_key, pin in policy["model_payloads"].items():
        observed = step7_common.validate_model_payload(
            model_key, step7_policy["embedding_models"][model_key]
        )
        if {
            "file_count": observed["file_count"],
            "total_size_bytes": observed["total_size_bytes"],
            "content_sha256": observed["content_sha256"],
        } != {
            "file_count": pin["file_count"],
            "total_size_bytes": pin["total_size_bytes"],
            "content_sha256": pin["content_sha256"],
        }:
            raise common.PublicProjectionContractError(
                f"Opaque encoder model payload drift for {model_key}"
            )
    transformers_cpu, tokenizer_cls = step7_encoder.require_tokenizer_stack()
    transformers_cpu.logging.set_verbosity_error()
    tokenizers = step7_encoder.load_tokenizers(step7_policy, tokenizer_cls)
    torch, transformers, sentence_transformers, sentence_transformer_cls = (
        step7_encoder.require_gpu_stack(step7_policy)
    )
    runtime = exact_runtime(policy, torch, transformers, sentence_transformers)
    cfg = step7_policy["embedding_models"]["labse"]
    model, loaded_state = step7_encoder.create_sentence_transformer(
        sentence_transformer_cls, cfg
    )
    return_root.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".gpu_return_v1.", dir=return_root.parent))
    try:
        records = []
        for part in parts:
            part_id = str(part["part_id"])
            values, audit = score_part(policy, part, step7_policy, tokenizers, model)
            part_root = temporary / part_id
            part_root.mkdir()
            value_path = part_root / "labse6.npy"
            np.save(value_path, values, allow_pickle=False)
            part_manifest = {
                "step": "step28_v13_v1_13_v9_4_1_labse6_part_v1",
                "status": "FROZEN_OPAQUE_LABSE6_PART_NO_MODEL_TRAINING",
                "part_id": part_id,
                "gpu_policy_canonical_self_hash": policy["canonical_self_hash"],
                "public_policy_canonical_self_hash": transfer_manifest[
                    "public_policy_canonical_self_hash"
                ],
                "transfer_manifest_canonical_self_hash": transfer_manifest[
                    "canonical_self_hash"
                ],
                "input_files": part["input_files"],
                "labse6_file": file_record(value_path, temporary),
                "labse6_shape": list(values.shape),
                "labse6_dtype": values.dtype.str,
                "labse6_value_sha256": matrix_value_sha256(values),
                "audit": audit,
                "canonical_identifiers_or_split_names_read": False,
                "labels_controllers_membership_qrels_or_audit_truth_read": False,
                "identity33_or_legacy18_read": False,
                "model_parameters_updated": False,
            }
            part_manifest["canonical_self_hash"] = common.canonical_sha256(
                part_manifest
            )
            manifest_path = part_root / "labse6_manifest.json"
            render_json(manifest_path, part_manifest)
            records.append(
                {
                    "part_id": part_id,
                    "manifest_file": file_record(manifest_path, temporary),
                    "manifest_canonical_self_hash": part_manifest[
                        "canonical_self_hash"
                    ],
                }
            )
            del values
            gc.collect()
        del model, tokenizers
        gc.collect()
        torch.cuda.empty_cache()
        root_manifest = {
            "step": "step28_v13_v1_13_v9_4_1_base_gpu_return_v1",
            "status": "FROZEN_OPAQUE_FOUR_PART_LABSE6_RETURN_NO_MODEL_TRAINING",
            "gpu_policy_canonical_self_hash": policy["canonical_self_hash"],
            "public_policy_canonical_self_hash": transfer_manifest[
                "public_policy_canonical_self_hash"
            ],
            "transfer_manifest_canonical_self_hash": transfer_manifest[
                "canonical_self_hash"
            ],
            "parts": records,
            "exact_runtime": runtime,
            "loaded_model_state": loaded_state,
            "temporary_chunks_retained": False,
            "temporary_embeddings_retained": False,
            "canonical_identifiers_or_split_names_read": False,
            "labels_controllers_membership_qrels_or_audit_truth_read": False,
            "model_parameters_updated": False,
        }
        root_manifest["canonical_self_hash"] = common.canonical_sha256(root_manifest)
        render_json(temporary / "gpu_return_manifest.json", root_manifest)
        validate_gpu_return(policy, transfer_manifest, parts, temporary)
        temporary.replace(return_root)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return root_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("validate-contract",))
    parser.parse_args()
    policy = common.load_policy()
    print(
        json.dumps(
            {
                "status": "PASSED_LINUX_ENCODER_CONTRACT_NO_FORMAL_EXECUTION",
                "gpu_policy_canonical_self_hash": policy["canonical_self_hash"],
                "formal_projection_executed": False,
                "supervision_or_audit_truth_read": False,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
