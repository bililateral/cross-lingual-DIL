#!/usr/bin/env python3
"""Encode complete redacted item documents and publish compact Step 7-v4 scores."""

from __future__ import annotations

import argparse
import gc
import hashlib
import inspect
import json
import math
import os
import platform
import unicodedata
from bisect import bisect_right
from collections.abc import Mapping
from pathlib import Path

import numpy as np

import step7_v4_build_sync_manifest as sync_builder
import step7_v4_common as common


ENCODER_SCRIPT = Path(__file__).resolve()
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def require_tokenizer_stack():
    try:
        import transformers
        from transformers import AutoTokenizer
    except ImportError as error:
        raise RuntimeError(
            "Step7-v4 GPU encoding requires transformers in the active Linux environment"
        ) from error
    return transformers, AutoTokenizer


def validate_sentence_transformers_version(policy: dict, observed: object) -> str:
    required = policy["gpu_execution"][
        "required_sentence_transformers_version"
    ]
    if required != common.REQUIRED_SENTENCE_TRANSFORMERS_VERSION:
        raise RuntimeError("Step7-v4 audited library-version contract drift")
    if not isinstance(observed, str) or observed != required:
        raise RuntimeError(
            "Step7-v4 sentence-transformers version drift: "
            f"expected={required} observed={observed}"
        )
    return observed


def require_gpu_stack(policy: dict):
    try:
        import sentence_transformers
        import torch
        import transformers
        from sentence_transformers import SentenceTransformer
    except ImportError as error:
        raise RuntimeError(
            "Step7-v4 GPU encoding requires torch, transformers, and sentence-transformers"
        ) from error
    if not torch.cuda.is_available():
        raise RuntimeError("Step7-v4 formal encoding requires CUDA")
    validate_sentence_transformers_version(
        policy, sentence_transformers.__version__
    )
    return torch, transformers, sentence_transformers, SentenceTransformer


def configure_deterministic_gpu(torch, policy: dict) -> dict:
    cfg = policy["gpu_execution"]
    seed = int(cfg["random_seed"])
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("highest")
    observed = {
        "random_seed": seed,
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "tokenizers_parallelism": os.environ.get("TOKENIZERS_PARALLELISM"),
        "deterministic_algorithms_enabled": bool(
            torch.are_deterministic_algorithms_enabled()
        ),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "cuda_matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
        "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
    }
    if observed != cfg["expected_runtime"]:
        raise ValueError("Step7-v4 deterministic GPU runtime contract drift")
    return observed


def verify_source_preparation(
    policy: dict, sync_manifest: dict
) -> tuple[dict, list[dict], list[dict], list[dict]]:
    outputs = policy["outputs"]
    manifest_records = {record["path"]: record for record in sync_manifest["files"]}
    for role in ("gpu_pair_manifest", "unique_text_corpus", "gpu_seller_text_index"):
        path = common.resolve(outputs[role])
        record = manifest_records.get(common.relative(path))
        if record is None or common.verify_file_record(record, role) != path:
            raise ValueError(f"Step7-v4 GPU source payload drift: {role}")

    pair_rows = common.load_csv(common.resolve(outputs["gpu_pair_manifest"]))
    pair_schema = ["pair_uid", "seller_uid_left", "seller_uid_right"]
    if (
        not pair_rows
        or any(list(row) != pair_schema for row in pair_rows)
        or len(pair_rows)
        != int(policy["supervision_boundary"]["expected_counts"]["total"])
        or any(
            row["pair_uid"] != f"pair_{index:06d}"
            for index, row in enumerate(pair_rows, start=1)
        )
    ):
        raise ValueError("Step7-v4 GPU opaque pair manifest drift")
    unique_rows = common.load_jsonl(common.resolve(outputs["unique_text_corpus"]))
    unique_schema = ["text_uid", "text", "text_sha256"]
    if not unique_rows or any(list(row) != unique_schema for row in unique_rows):
        raise ValueError("Step7-v4 GPU unique-text schema drift")
    text_uids = set()
    for row in unique_rows:
        if (
            not row["text"]
            or row["text_uid"] in text_uids
            or row["text_uid"] != row["text_sha256"]
            or common.sha256_text(row["text"]) != row["text_sha256"]
        ):
            raise ValueError("Step7-v4 GPU unique-text content/hash drift")
        text_uids.add(row["text_uid"])

    seller_rows = common.load_jsonl(common.resolve(outputs["gpu_seller_text_index"]))
    seller_schema = [
        "seller_uid",
        "field_name",
        "text_uid",
        "multiplicity",
    ]
    if not seller_rows or any(list(row) != seller_schema for row in seller_rows):
        raise ValueError("Step7-v4 GPU seller-text schema drift")
    seen = set()
    pair_sellers = {
        row[endpoint]
        for row in pair_rows
        for endpoint in ("seller_uid_left", "seller_uid_right")
    }
    mapped_sellers = set()
    for row in seller_rows:
        key = (row["seller_uid"], row["field_name"], row["text_uid"])
        if (
            key in seen
            or not row["seller_uid"].startswith("seller_")
            or len(row["seller_uid"]) != len("seller_000000")
            or not row["seller_uid"][len("seller_") :].isdigit()
            or row["field_name"] not in common.FIELD_NAMES
            or row["text_uid"] not in text_uids
            or int(row["multiplicity"]) <= 0
        ):
            raise ValueError("Step7-v4 GPU seller-text mapping drift")
        seen.add(key)
        mapped_sellers.add(row["seller_uid"])
    if mapped_sellers != pair_sellers:
        raise ValueError("Step7-v4 GPU seller universe lacks usable clean text")
    source_contract = {
        "manifest_file_sha256": sync_manifest[
            "source_preparation_manifest_file_sha256"
        ],
        "manifest_content_sha256": sync_manifest[
            "source_preparation_manifest_content_sha256"
        ],
        "complete_preparation_manifest_present_in_gpu_workspace": False,
    }
    return source_contract, pair_rows, unique_rows, seller_rows


def verify_label_free_gpu_sync(policy: dict) -> tuple[dict, dict[str, dict]]:
    path = common.resolve(policy["outputs"]["gpu_sync_manifest"])
    if not path.is_file():
        raise FileNotFoundError("Build and transfer the Step7-v4 GPU sync manifest first")
    manifest = common.load_json(path)
    common.verify_implementation_files(policy, sync_builder.GPU_IMPLEMENTATION_ROLES)
    content_hash = manifest.get("manifest_content_sha256")
    without_hash = dict(manifest)
    without_hash.pop("manifest_content_sha256", None)
    expected_records = [
        common.file_record(item)
        for item in sync_builder.payload_paths(policy, common.DEFAULT_POLICY)
    ]
    expected_model_directories = {
        model_key: {
            "path": cfg["local_path"],
            "file_count": int(cfg["expected_file_count"]),
            "total_size_bytes": int(cfg["expected_total_size_bytes"]),
            "content_sha256": cfg["expected_content_sha256"],
        }
        for model_key, cfg in policy["embedding_models"].items()
    }
    if (
        content_hash != common.canonical_hash(without_hash)
        or manifest.get("step") != "step7_v4_label_free_gpu_sync"
        or manifest.get("version") != policy["version"]
        or manifest.get("policy_sha256") != common.sha256_file(common.DEFAULT_POLICY)
        or manifest.get("policy_contract_sha256") != common.canonical_hash(policy)
        or manifest.get("files") != expected_records
        or manifest.get("file_count") != len(expected_records)
        or manifest.get("total_file_bytes")
        != sum(record["size_bytes"] for record in expected_records)
        or manifest.get("model_directories") != expected_model_directories
        or manifest.get("forbidden_workspace_paths")
        != sync_builder.FORBIDDEN_WORKSPACE_PATHS
        or manifest.get("expected_gpu_outputs_to_sync_back")
        != sync_builder.expected_gpu_output_paths(policy)
        or not isinstance(
            manifest.get("source_preparation_manifest_file_sha256"), str
        )
        or len(manifest["source_preparation_manifest_file_sha256"]) != 64
        or not isinstance(
            manifest.get("source_preparation_manifest_content_sha256"), str
        )
        or len(manifest["source_preparation_manifest_content_sha256"]) != 64
    ):
        raise ValueError("Step7-v4 GPU sync manifest contract drift")
    if (
        manifest.get("pair_level_label_or_evidence_value_files_in_payload")
        is not False
        or manifest.get(
            "raw_source_workbook_or_item_manifest_file_bytes_in_payload"
        )
        is not False
        or manifest.get(
            "aggregate_supervision_counts_and_evidence_vocabulary_in_policy_only"
        )
        is not True
    ):
        raise ValueError("Step7-v4 GPU sync is not label/raw-source isolated")
    present = [
        value
        for value in manifest["forbidden_workspace_paths"]
        if common.resolve(value).exists()
    ]
    if present:
        raise ValueError(
            "Step7-v4 formal GPU workspace contains a forbidden path: " + present[0]
        )
    fingerprints = {}
    for model_key, cfg in policy["embedding_models"].items():
        observed = common.validate_model_payload(model_key, cfg)
        expected_fingerprint = manifest["model_directories"][model_key]
        if expected_fingerprint != {
            "path": cfg["local_path"],
            "file_count": observed["file_count"],
            "total_size_bytes": observed["total_size_bytes"],
            "content_sha256": observed["content_sha256"],
        }:
            raise ValueError(f"Step7-v4 GPU model/sync fingerprint drift: {model_key}")
        fingerprints[model_key] = observed
    return manifest, fingerprints


def tokenizer_result_ids(result: dict) -> list[int]:
    values = result["input_ids"]
    if values and isinstance(values[0], list):
        if len(values) != 1:
            raise ValueError("Step7-v4 expected one tokenizer row")
        values = values[0]
    return [int(value) for value in values]


def token_length(tokenizer, text: str) -> int:
    return len(
        tokenizer_result_ids(
            tokenizer(
                text,
                add_special_tokens=True,
                padding=False,
                truncation=False,
            )
        )
    )


def all_token_lengths(tokenizers: dict[str, object], policy: dict, text: str) -> dict[str, int]:
    return {
        model_key: token_length(
            tokenizer, policy["embedding_models"][model_key]["text_prefix"] + text
        )
        for model_key, tokenizer in tokenizers.items()
    }


def preferred_boundaries(text: str) -> list[int]:
    boundaries = [
        index
        for index, character in enumerate(text, start=1)
        if character.isspace() or unicodedata.category(character).startswith("P")
    ]
    if not boundaries or boundaries[-1] != len(text):
        boundaries.append(len(text))
    return boundaries


def conservative_tokenizer_end(tokenizer, prefix: str, remaining: str, budget: int) -> int:
    complete = prefix + remaining
    if token_length(tokenizer, complete) <= budget:
        return len(remaining)
    try:
        encoded = tokenizer(
            complete,
            add_special_tokens=False,
            padding=False,
            truncation=False,
            return_offsets_mapping=True,
        )
        offsets = encoded["offset_mapping"]
        if offsets and isinstance(offsets[0], list):
            if len(offsets) != 1:
                raise ValueError("unexpected batched offset mapping")
            offsets = offsets[0]
        available = budget - int(tokenizer.num_special_tokens_to_add(pair=False))
        if available <= 0 or len(offsets) <= available:
            raise ValueError("invalid tokenizer offset capacity")
        usable = [
            int(stop) - len(prefix)
            for _start, stop in offsets[:available]
            if int(stop) > len(prefix)
        ]
        if not usable:
            raise ValueError("token budget does not reach chunk content")
        return max(1, min(len(remaining), max(usable)))
    except (KeyError, TypeError, ValueError, NotImplementedError):
        # Exact checks below remain authoritative.  Binary search is only a
        # conservative starting point when a tokenizer has no offset mapping.
        low, high = 1, len(remaining)
        while low < high:
            middle = (low + high + 1) // 2
            if token_length(tokenizer, prefix + remaining[:middle]) <= budget:
                low = middle
            else:
                high = middle - 1
        return low


def choose_shared_chunk(
    tokenizers: dict[str, object], policy: dict, remaining: str
) -> tuple[str, dict[str, int]]:
    if not remaining:
        raise ValueError("Step7-v4 cannot chunk an empty remainder")
    budget = int(
        policy["shared_chunking"]["token_budget_including_model_prefix_and_special_tokens"]
    )
    complete_lengths = all_token_lengths(tokenizers, policy, remaining)
    if max(complete_lengths.values()) <= budget:
        return remaining, complete_lengths
    conservative = min(
        conservative_tokenizer_end(
            tokenizers[model_key],
            policy["embedding_models"][model_key]["text_prefix"],
            remaining,
            budget,
        )
        for model_key in common.MODEL_KEYS
    )
    boundaries = preferred_boundaries(remaining)
    boundary_index = bisect_right(boundaries, conservative) - 1
    end = boundaries[boundary_index] if boundary_index >= 0 else conservative
    end = max(1, min(end, len(remaining)))
    while end > 1:
        candidate = remaining[:end]
        lengths = all_token_lengths(tokenizers, policy, candidate)
        if candidate.strip() and max(lengths.values()) <= budget:
            break
        boundary_index = bisect_right(boundaries, end - 1) - 1
        end = boundaries[boundary_index] if boundary_index >= 0 else end - 1
    candidate = remaining[:end]
    lengths = all_token_lengths(tokenizers, policy, candidate)
    if not candidate.strip() or max(lengths.values()) > budget:
        raise ValueError("Step7-v4 cannot form a nonempty shared chunk")
    for next_end in boundaries[bisect_right(boundaries, end) :]:
        expanded = remaining[:next_end]
        expanded_lengths = all_token_lengths(tokenizers, policy, expanded)
        if max(expanded_lengths.values()) > budget:
            break
        candidate, lengths, end = expanded, expanded_lengths, next_end
    return candidate, lengths


def validate_shared_chunks(
    policy: dict, unique_rows: list[dict], chunk_rows: list[dict]
) -> dict:
    expected_schema = [
        "chunk_uid",
        "text_uid",
        "chunk_index",
        "char_start",
        "char_end",
        "text",
        "text_sha256",
        "token_lengths",
    ]
    if not chunk_rows or any(list(row) != expected_schema for row in chunk_rows):
        raise ValueError("Step7-v4 shared chunk schema drift")
    grouped: dict[str, list[dict]] = {}
    seen_chunk_uids = set()
    for row in chunk_rows:
        if row["chunk_uid"] in seen_chunk_uids:
            raise ValueError("Step7-v4 shared chunks contain a duplicate UID")
        seen_chunk_uids.add(row["chunk_uid"])
        grouped.setdefault(row["text_uid"], []).append(row)
    if list(grouped) != [row["text_uid"] for row in unique_rows]:
        raise ValueError("Step7-v4 shared chunks changed unique-text order/universe")

    counts = []
    token_lengths: dict[str, list[int]] = {key: [] for key in common.MODEL_KEYS}
    budget = int(
        policy["shared_chunking"]["token_budget_including_model_prefix_and_special_tokens"]
    )
    for source_row in unique_rows:
        rows = grouped[source_row["text_uid"]]
        position = 0
        pieces = []
        for index, row in enumerate(rows):
            identity = {
                "text_uid": source_row["text_uid"],
                "chunk_index": index,
                "char_start": position,
                "char_end": position + len(row["text"]),
                "text_sha256": common.sha256_text(row["text"]),
            }
            if (
                int(row["chunk_index"]) != index
                or int(row["char_start"]) != position
                or int(row["char_end"]) != position + len(row["text"])
                or row["text_sha256"] != identity["text_sha256"]
                or row["chunk_uid"] != common.canonical_hash(identity)
                or not row["text"].strip()
                or list(row["token_lengths"]) != common.MODEL_KEYS
            ):
                raise ValueError("Step7-v4 shared chunk identity/order drift")
            for model_key in common.MODEL_KEYS:
                length = int(row["token_lengths"][model_key])
                if length <= 0 or length > budget:
                    raise ValueError("Step7-v4 shared chunk exceeds a tokenizer budget")
                token_lengths[model_key].append(length)
            pieces.append(row["text"])
            position = int(row["char_end"])
        if "".join(pieces) != source_row["text"]:
            raise ValueError("Step7-v4 shared chunks do not exactly reconstruct text")
        counts.append(len(rows))
    return {
        "exact_character_reconstruction": True,
        "unique_text_count": len(unique_rows),
        "chunk_count": len(chunk_rows),
        "chunk_count_minimum": int(min(counts)),
        "chunk_count_median": float(np.median(counts)),
        "chunk_count_p95": float(np.quantile(counts, 0.95)),
        "chunk_count_maximum": int(max(counts)),
        "whole_text_within_shared_budget_count": int(
            sum(count == 1 for count in counts)
        ),
        "text_requiring_chunking_count": int(
            sum(count > 1 for count in counts)
        ),
        "token_length_by_model": {
            model_key: {
                "minimum": int(np.min(values)),
                "median": float(np.median(values)),
                "p95": float(np.quantile(values, 0.95)),
                "maximum": int(np.max(values)),
                "over_budget_count": int(np.sum(np.asarray(values) > budget)),
            }
            for model_key, values in token_lengths.items()
        },
    }


def build_shared_chunks(
    policy: dict, unique_rows: list[dict], tokenizers: dict[str, object]
) -> tuple[list[dict], dict]:
    source_texts = [row["text"] for row in unique_rows]
    complete_lengths_by_model = {}
    for model_key, tokenizer in tokenizers.items():
        _digest, lengths = tokenizer_digest_and_lengths(
            tokenizer,
            source_texts,
            policy["embedding_models"][model_key]["text_prefix"],
        )
        if len(lengths) != len(unique_rows):
            raise AssertionError("Step7-v4 batched whole-text token lengths are incomplete")
        complete_lengths_by_model[model_key] = lengths
    budget = int(
        policy["shared_chunking"][
            "token_budget_including_model_prefix_and_special_tokens"
        ]
    )
    rows = []
    complete_text_count = 0
    for source_index, source in enumerate(unique_rows):
        text = source["text"]
        position = 0
        chunk_index = 0
        complete_lengths = {
            model_key: int(complete_lengths_by_model[model_key][source_index])
            for model_key in common.MODEL_KEYS
        }
        if max(complete_lengths.values()) <= budget:
            chunks = [(text, complete_lengths)]
            complete_text_count += 1
        else:
            chunks = []
            remaining_position = 0
            while remaining_position < len(text):
                chunk, lengths = choose_shared_chunk(
                    tokenizers, policy, text[remaining_position:]
                )
                chunks.append((chunk, lengths))
                remaining_position += len(chunk)
        for chunk, lengths in chunks:
            end = position + len(chunk)
            identity = {
                "text_uid": source["text_uid"],
                "chunk_index": chunk_index,
                "char_start": position,
                "char_end": end,
                "text_sha256": common.sha256_text(chunk),
            }
            rows.append(
                {
                    "chunk_uid": common.canonical_hash(identity),
                    "text_uid": source["text_uid"],
                    "chunk_index": chunk_index,
                    "char_start": position,
                    "char_end": end,
                    "text": chunk,
                    "text_sha256": identity["text_sha256"],
                    "token_lengths": lengths,
                }
            )
            position = end
            chunk_index += 1
    audit = validate_shared_chunks(policy, unique_rows, rows)
    if audit["whole_text_within_shared_budget_count"] != complete_text_count:
        raise AssertionError("Step7-v4 batched whole-text budget audit drift")
    return rows, audit


def load_tokenizers(policy: dict, tokenizer_cls) -> dict[str, object]:
    return {
        model_key: tokenizer_cls.from_pretrained(
            str(common.resolve(cfg["local_path"])),
            local_files_only=True,
            use_fast=True,
        )
        for model_key, cfg in policy["embedding_models"].items()
    }


def tokenizer_digest_and_lengths(
    tokenizer, texts: list[str], prefix: str, *, batch_size: int = 128
) -> tuple[str, list[int]]:
    digest = hashlib.sha256()
    lengths = []
    for start in range(0, len(texts), int(batch_size)):
        batch = [prefix + text for text in texts[start : start + int(batch_size)]]
        encoded = tokenizer(
            batch,
            add_special_tokens=True,
            padding=False,
            truncation=False,
        )["input_ids"]
        if len(encoded) != len(batch):
            raise ValueError("Step7-v4 tokenizer batch row-count drift")
        for ids in encoded:
            values = [int(value) for value in ids]
            lengths.append(len(values))
            digest.update(len(values).to_bytes(8, "little", signed=False))
            for value in values:
                digest.update(value.to_bytes(8, "little", signed=True))
    return digest.hexdigest(), lengths


def _tensor_to_numpy(value) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


def sentence_transformer_tokenizer_digest_and_lengths(
    model, texts: list[str], prefix: str, *, batch_size: int = 128
) -> tuple[str, list[int]]:
    """Hash the actual SentenceTransformer input IDs before encode()."""

    digest = hashlib.sha256()
    lengths = []
    for start in range(0, len(texts), int(batch_size)):
        batch = [prefix + text for text in texts[start : start + int(batch_size)]]
        features = model.tokenize(batch)
        if (
            not isinstance(features, Mapping)
            or "input_ids" not in features
            or "attention_mask" not in features
        ):
            raise ValueError(
                "Step7-v4 SentenceTransformer tokenize() lacks IDs or attention mask"
            )
        input_ids = _tensor_to_numpy(features["input_ids"])
        attention_mask = _tensor_to_numpy(features["attention_mask"])
        if (
            input_ids.ndim != 2
            or attention_mask.shape != input_ids.shape
            or input_ids.shape[0] != len(batch)
        ):
            raise ValueError(
                "Step7-v4 SentenceTransformer token batch shape drift"
            )
        for ids, mask in zip(input_ids, attention_mask, strict=True):
            active = np.asarray(mask) != 0
            active_indices = np.flatnonzero(active)
            if active_indices.size <= 0 or not np.array_equal(
                active_indices,
                np.arange(active_indices[0], active_indices[-1] + 1),
            ):
                raise ValueError(
                    "Step7-v4 SentenceTransformer attention-mask layout drift"
                )
            values = [int(value) for value in np.asarray(ids)[active_indices]]
            lengths.append(len(values))
            digest.update(len(values).to_bytes(8, "little", signed=False))
            for value in values:
                digest.update(value.to_bytes(8, "little", signed=True))
    return digest.hexdigest(), lengths


def normalize_loaded_prompts(model) -> dict[str, str]:
    raw = getattr(model, "prompts", None)
    if raw is None:
        return {}
    if not isinstance(raw, dict) or any(
        not isinstance(key, str) or not isinstance(value, str)
        for key, value in raw.items()
    ):
        raise ValueError("Step7-v4 loaded SentenceTransformer prompts are malformed")
    return dict(sorted(raw.items()))


def validate_sentence_transformer_class(sentence_transformer_cls) -> None:
    encode_parameters = inspect.signature(sentence_transformer_cls.encode).parameters
    if "prompt" not in encode_parameters:
        raise RuntimeError(
            "Step7-v4 requires SentenceTransformer.encode(prompt=...) so hidden "
            "model-default prompts can be disabled explicitly"
        )


def create_sentence_transformer(sentence_transformer_cls, cfg: dict):
    validate_sentence_transformer_class(sentence_transformer_cls)
    path = str(common.resolve(cfg["local_path"]))
    try:
        model = sentence_transformer_cls(
            path, device="cuda", local_files_only=True
        )
    except TypeError as error:
        raise RuntimeError(
            "Step7-v4 requires a sentence-transformers version supporting local_files_only"
        ) from error
    loaded_prompt_state = {
        "loaded_default_prompt_name": getattr(model, "default_prompt_name", None),
        "loaded_prompts": normalize_loaded_prompts(model),
        "loaded_native_max_seq_length": getattr(model, "max_seq_length", None),
    }
    if loaded_prompt_state["loaded_default_prompt_name"] is not None and not isinstance(
        loaded_prompt_state["loaded_default_prompt_name"], str
    ):
        raise ValueError(
            "Step7-v4 loaded SentenceTransformer default_prompt_name is malformed"
        )
    model.default_prompt_name = None
    if getattr(model, "default_prompt_name", None) is not None:
        raise ValueError(
            "Step7-v4 could not clear SentenceTransformer default_prompt_name"
        )
    loaded_max = loaded_prompt_state["loaded_native_max_seq_length"]
    if (
        isinstance(loaded_max, bool)
        or not isinstance(loaded_max, (int, np.integer))
        or int(loaded_max) != int(cfg["native_max_seq_length"])
    ):
        raise ValueError(
            "Step7-v4 loaded native max_seq_length drift: "
            f"expected={cfg['native_max_seq_length']} observed={loaded_max}"
        )
    loaded_prompt_state["loaded_native_max_seq_length"] = int(loaded_max)
    model.eval()
    return model, loaded_prompt_state


def smoke_test_one_model(
    policy: dict,
    model_key: str,
    cfg: dict,
    tokenizer,
    chunk_rows: list[dict],
    torch,
    sentence_transformer_cls,
) -> dict:
    """Load and deterministically encode one real, longest shared chunk twice."""

    if not chunk_rows:
        raise ValueError("Step7-v4 smoke test has no shared chunks")
    index = max(
        range(len(chunk_rows)),
        key=lambda value: int(chunk_rows[value]["token_lengths"][model_key]),
    )
    row = chunk_rows[index]
    text = row["text"]
    expected_digest, expected_lengths = tokenizer_digest_and_lengths(
        tokenizer, [text], cfg["text_prefix"], batch_size=1
    )
    model, loaded_state = create_sentence_transformer(
        sentence_transformer_cls, cfg
    )
    observed_digest, observed_lengths = (
        sentence_transformer_tokenizer_digest_and_lengths(
            model, [text], cfg["text_prefix"], batch_size=1
        )
    )
    registered_length = int(row["token_lengths"][model_key])
    if (
        expected_digest != observed_digest
        or expected_lengths != observed_lengths
        or observed_lengths != [registered_length]
        or registered_length
        > int(
            policy["shared_chunking"][
                "token_budget_including_model_prefix_and_special_tokens"
            ]
        )
        or registered_length > int(cfg["native_max_seq_length"])
    ):
        raise ValueError(
            f"Step7-v4 real model tokenizer smoke drift: {model_key}"
        )
    sentences = [cfg["text_prefix"] + text]
    matrices = []
    for _repeat in range(2):
        encoded = model.encode(
            sentences,
            batch_size=1,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
            prompt=cfg["sentence_transformer_prompt"],
        )
        matrix = np.ascontiguousarray(encoded, dtype=np.float32)
        expected_shape = (1, int(cfg["expected_dimension"]))
        if matrix.shape != expected_shape or not np.all(np.isfinite(matrix)):
            raise ValueError(
                f"Step7-v4 real model smoke embedding drift: {model_key}"
            )
        matrices.append(matrix)
    if not np.array_equal(matrices[0], matrices[1]):
        raise ValueError(
            f"Step7-v4 repeated deterministic smoke encoding drift: {model_key}"
        )
    norm_error = float(
        np.max(np.abs(np.linalg.norm(matrices[0], axis=1) - 1.0))
    )
    if norm_error > 1e-3:
        raise ValueError(
            f"Step7-v4 real model smoke returned non-unit vector: {model_key}"
        )
    parameter_dtypes = sorted({str(parameter.dtype) for parameter in model.parameters()})
    if not parameter_dtypes:
        raise ValueError(f"Step7-v4 smoke model has no parameters: {model_key}")
    result = {
        "model_key": model_key,
        **loaded_state,
        "native_max_seq_length": int(cfg["native_max_seq_length"]),
        "longest_shared_chunk_token_length": registered_length,
        "actual_sentence_transformer_tokenizer_digest": observed_digest,
        "embedding_shape": list(matrices[0].shape),
        "embedding_dtype": str(matrices[0].dtype),
        "model_parameter_dtypes": parameter_dtypes,
        "maximum_unit_norm_error": norm_error,
        "repeated_encoding_byte_identical": True,
    }
    del matrices, model
    gc.collect()
    torch.cuda.empty_cache()
    return result


def matrix_content_sha256(matrix: np.ndarray) -> str:
    values = np.ascontiguousarray(matrix)
    digest = hashlib.sha256()
    digest.update(memoryview(values).cast("B"))
    return digest.hexdigest()


def encode_one_model(
    policy: dict,
    model_key: str,
    cfg: dict,
    model_fingerprint: dict,
    shared_tokenizer_digest: str,
    chunk_rows: list[dict],
    pair_rows: list[dict],
    seller_rows: list[dict],
    sync_manifest: dict,
    source_manifest: dict,
    torch,
    transformers,
    sentence_transformers,
    sentence_transformer_cls,
) -> dict:
    deterministic_runtime = configure_deterministic_gpu(torch, policy)
    model, loaded_prompt_state = create_sentence_transformer(
        sentence_transformer_cls, cfg
    )
    texts = [row["text"] for row in chunk_rows]
    runtime_digest, runtime_lengths = sentence_transformer_tokenizer_digest_and_lengths(
        model, texts, cfg["text_prefix"]
    )
    registered_lengths = [
        int(row["token_lengths"][model_key]) for row in chunk_rows
    ]
    if runtime_digest != shared_tokenizer_digest or runtime_lengths != registered_lengths:
        raise ValueError(f"Step7-v4 runtime tokenizer drift: {model_key}")
    if max(runtime_lengths) > int(
        policy["shared_chunking"]["token_budget_including_model_prefix_and_special_tokens"]
    ):
        raise ValueError(f"Step7-v4 runtime tokenizer exceeds shared budget: {model_key}")
    if max(runtime_lengths) > int(cfg["native_max_seq_length"]):
        raise ValueError(
            f"Step7-v4 runtime tokenizer exceeds native model window: {model_key}"
        )
    parameter_dtypes = sorted({str(parameter.dtype) for parameter in model.parameters()})
    if not parameter_dtypes:
        raise ValueError(f"Step7-v4 loaded model has no parameters: {model_key}")

    matrix = model.encode(
        [cfg["text_prefix"] + text for text in texts],
        batch_size=int(cfg["batch_size"]),
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
        prompt=cfg["sentence_transformer_prompt"],
    )
    matrix = np.ascontiguousarray(matrix, dtype=np.float32)
    expected_shape = (len(chunk_rows), int(cfg["expected_dimension"]))
    if matrix.shape != expected_shape or not np.all(np.isfinite(matrix)):
        raise ValueError(
            f"Step7-v4 unexpected embedding matrix for {model_key}: {matrix.shape}"
        )
    norm_error = float(np.max(np.abs(np.linalg.norm(matrix, axis=1) - 1.0)))
    if norm_error > 1e-3:
        raise ValueError(f"Step7-v4 model returned non-unit vectors: {model_key}")
    matrix_hash = matrix_content_sha256(matrix)
    aggregation = policy["aggregation"]
    pair_scores = common.compute_pair_score_rows(
        pair_rows,
        seller_rows,
        chunk_rows,
        matrix,
        cfg,
        top_k=int(aggregation["top_k_item_matches"]),
        decimal_places=int(aggregation["serialized_decimal_places"]),
        similarity_block_rows=int(aggregation["similarity_block_rows"]),
    )
    score_path = common.resolve(
        policy["outputs"]["pair_scores_template"].format(model_key=model_key)
    )
    common.write_csv_immutable(score_path, pair_scores)
    runtime = {
        "step": "step7_v4_encode_complete_item_shared_chunks",
        "version": policy["version"],
        "model_key": model_key,
        "role": cfg["role"],
        "repo_id": cfg["repo_id"],
        "revision": cfg["revision"],
        "local_path": cfg["local_path"],
        "model_fingerprint": model_fingerprint,
        "encoder_parameters_updated": False,
        "feature_generation_reads_label_values": False,
        "label_or_raw_source_files_present_in_gpu_workspace": False,
        "same_exact_shared_chunks_for_all_models": True,
        "text_prefix": cfg["text_prefix"],
        "sentence_transformer_prompt": cfg["sentence_transformer_prompt"],
        "explicit_sentence_transformer_prompt_argument_used": True,
        "default_prompt_name_cleared_before_encoding": True,
        **loaded_prompt_state,
        "native_max_seq_length": int(cfg["native_max_seq_length"]),
        "batch_size": int(cfg["batch_size"]),
        "shared_tokenizer_digest": shared_tokenizer_digest,
        "runtime_sentence_transformer_tokenizer_digest": runtime_digest,
        "runtime_token_lengths_replay_shared_manifest": True,
        "embedding_matrix_published": False,
        "embedding_matrix_ephemeral": True,
        "embedding_matrix_shape": list(matrix.shape),
        "embedding_matrix_dtype": str(matrix.dtype),
        "model_parameter_dtypes": parameter_dtypes,
        "embedding_matrix_content_sha256": matrix_hash,
        "maximum_unit_norm_error": norm_error,
        "pair_count": len(pair_scores),
        "aggregate_feature_names": common.encoder_feature_names(cfg),
        "multiplicity_audit_feature_names": common.frequency_audit_feature_names(cfg),
        "pair_scores": common.file_record(score_path),
        "shared_chunks_sha256": common.sha256_file(
            common.resolve(policy["outputs"]["shared_chunks"])
        ),
        "gpu_sync_manifest_sha256": common.sha256_file(
            common.resolve(policy["outputs"]["gpu_sync_manifest"])
        ),
        "source_preparation_manifest_file_sha256": source_manifest[
            "manifest_file_sha256"
        ],
        "source_preparation_manifest_content_sha256": source_manifest[
            "manifest_content_sha256"
        ],
        "policy_sha256": common.sha256_file(common.DEFAULT_POLICY),
        "policy_contract_sha256": common.canonical_hash(policy),
        "generator_script_sha256": common.sha256_file(ENCODER_SCRIPT),
        "device": "cuda",
        "gpu_name": torch.cuda.get_device_name(0),
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "torch_version": torch.__version__,
        "torch_cuda_runtime_version": str(torch.version.cuda),
        "cudnn_runtime_version": torch.backends.cudnn.version(),
        "gpu_compute_capability": list(torch.cuda.get_device_capability(0)),
        "transformers_version": transformers.__version__,
        "sentence_transformers_version": sentence_transformers.__version__,
        "deterministic_gpu_runtime": deterministic_runtime,
        "sync_manifest_content_sha256": sync_manifest["manifest_content_sha256"],
    }
    runtime["runtime_content_sha256"] = common.canonical_hash(runtime)
    runtime_path = common.resolve(
        policy["outputs"]["model_runtime_manifest_template"].format(model_key=model_key)
    )
    common.write_json_immutable(runtime_path, runtime)
    del pair_scores, matrix, model
    gc.collect()
    torch.cuda.empty_cache()
    return runtime


def output_bundle(policy: dict, sync_manifest: dict) -> dict:
    expected = sync_manifest["expected_gpu_outputs_to_sync_back"]
    bundle_path = policy["outputs"]["gpu_output_manifest"]
    records = [
        common.file_record(common.resolve(path))
        for path in expected
        if path != bundle_path
    ]
    bundle = {
        "step": "step7_v4_label_free_gpu_output_bundle",
        "version": policy["version"],
        "label_or_raw_source_files_present_in_gpu_workspace": False,
        "embedding_matrices_published": False,
        "files": records,
        "file_count": len(records),
        "total_file_bytes": sum(record["size_bytes"] for record in records),
        "gpu_sync_manifest_sha256": common.sha256_file(
            common.resolve(policy["outputs"]["gpu_sync_manifest"])
        ),
        "policy_sha256": common.sha256_file(common.DEFAULT_POLICY),
        "generator_script_sha256": common.sha256_file(ENCODER_SCRIPT),
    }
    bundle["bundle_content_sha256"] = common.canonical_hash(bundle)
    common.write_json_immutable(common.resolve(bundle_path), bundle)
    return bundle


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--validate-inputs-only",
        action="store_true",
        help=(
            "Validate the isolated contract, build shared chunks in memory, and "
            "run one real deterministic model smoke per encoder without full-corpus output."
        ),
    )
    args = parser.parse_args()
    policy = common.load_policy()
    sync_manifest, model_fingerprints = verify_label_free_gpu_sync(policy)
    source_manifest, pair_rows, unique_rows, seller_rows = verify_source_preparation(
        policy, sync_manifest
    )
    if args.validate_inputs_only:
        transformers_for_tokenizer, tokenizer_cls = require_tokenizer_stack()
        torch, transformers, sentence_transformers, sentence_transformer_cls = (
            require_gpu_stack(policy)
        )
        validate_sentence_transformer_class(sentence_transformer_cls)
        deterministic_runtime = configure_deterministic_gpu(torch, policy)
        tokenizers = load_tokenizers(policy, tokenizer_cls)
        chunk_rows, chunk_audit = build_shared_chunks(
            policy, unique_rows, tokenizers
        )
        smoke_results = {}
        for model_key, cfg in policy["embedding_models"].items():
            smoke_results[model_key] = smoke_test_one_model(
                policy,
                model_key,
                cfg,
                tokenizers[model_key],
                chunk_rows,
                torch,
                sentence_transformer_cls,
            )
        print(
            json.dumps(
                {
                    "status": "pass",
                    "pair_count": len(pair_rows),
                    "unique_text_count": len(unique_rows),
                    "seller_mapping_count": len(seller_rows),
                    "model_keys": common.MODEL_KEYS,
                    "cuda_device_name": torch.cuda.get_device_name(0),
                    "transformers_tokenizer_version": (
                        transformers_for_tokenizer.__version__
                    ),
                    "transformers_runtime_version": transformers.__version__,
                    "sentence_transformers_version": (
                        sentence_transformers.__version__
                    ),
                    "torch_version": torch.__version__,
                    "explicit_sentence_transformer_prompt_supported": True,
                    "deterministic_gpu_runtime": deterministic_runtime,
                    "shared_chunk_count": len(chunk_rows),
                    "shared_chunk_audit": chunk_audit,
                    "real_model_smoke_results": smoke_results,
                    "smoke_numerical_execution_performed": True,
                    "full_corpus_encoding_performed": False,
                    "formal_output_files_written": False,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return

    transformers_for_tokenizer, tokenizer_cls = require_tokenizer_stack()
    torch, transformers, sentence_transformers, sentence_transformer_cls = (
        require_gpu_stack(policy)
    )
    validate_sentence_transformer_class(sentence_transformer_cls)
    tokenizers = load_tokenizers(policy, tokenizer_cls)
    chunk_rows, chunk_audit = build_shared_chunks(policy, unique_rows, tokenizers)
    chunk_path = common.resolve(policy["outputs"]["shared_chunks"])
    common.write_jsonl_immutable(chunk_path, chunk_rows)
    texts = [row["text"] for row in chunk_rows]
    tokenizer_digests = {}
    for model_key, tokenizer in tokenizers.items():
        digest, lengths = tokenizer_digest_and_lengths(
            tokenizer, texts, policy["embedding_models"][model_key]["text_prefix"]
        )
        if lengths != [int(row["token_lengths"][model_key]) for row in chunk_rows]:
            raise ValueError(f"Step7-v4 shared tokenizer length replay drift: {model_key}")
        tokenizer_digests[model_key] = digest
    chunk_manifest = {
        "step": "step7_v4_build_complete_shared_item_chunks",
        "version": policy["version"],
        "labels_or_evidence_types_read": False,
        "raw_source_workbooks_present": False,
        "chunking_contract": policy["shared_chunking"],
        "model_input_contracts": {
            model_key: {
                "text_prefix": cfg["text_prefix"],
                "sentence_transformer_prompt": cfg[
                    "sentence_transformer_prompt"
                ],
                "native_max_seq_length": int(cfg["native_max_seq_length"]),
            }
            for model_key, cfg in policy["embedding_models"].items()
        },
        "chunk_audit": chunk_audit,
        "tokenizer_digests": tokenizer_digests,
        "model_fingerprints": model_fingerprints,
        "shared_chunks": common.file_record(chunk_path),
        "unique_text_corpus_sha256": common.sha256_file(
            common.resolve(policy["outputs"]["unique_text_corpus"])
        ),
        "source_preparation_manifest_file_sha256": sync_manifest[
            "source_preparation_manifest_file_sha256"
        ],
        "source_preparation_manifest_content_sha256": sync_manifest[
            "source_preparation_manifest_content_sha256"
        ],
        "gpu_sync_manifest_sha256": common.sha256_file(
            common.resolve(policy["outputs"]["gpu_sync_manifest"])
        ),
        "policy_sha256": common.sha256_file(common.DEFAULT_POLICY),
        "generator_script_sha256": common.sha256_file(ENCODER_SCRIPT),
        "transformers_version": transformers_for_tokenizer.__version__,
    }
    chunk_manifest["manifest_content_sha256"] = common.canonical_hash(chunk_manifest)
    common.write_json_immutable(
        common.resolve(policy["outputs"]["shared_chunks_manifest"]), chunk_manifest
    )
    del tokenizers
    gc.collect()

    runtimes = {}
    for model_key, cfg in policy["embedding_models"].items():
        runtimes[model_key] = encode_one_model(
            policy,
            model_key,
            cfg,
            model_fingerprints[model_key],
            tokenizer_digests[model_key],
            chunk_rows,
            pair_rows,
            seller_rows,
            sync_manifest,
            source_manifest,
            torch,
            transformers,
            sentence_transformers,
            sentence_transformer_cls,
        )
    bundle = output_bundle(policy, sync_manifest)
    print(
        json.dumps(
            {
                "status": "pass",
                "chunk_count": len(chunk_rows),
                "model_keys": list(runtimes),
                "embedding_matrices_published": False,
                "gpu_output_file_count": bundle["file_count"] + 1,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
