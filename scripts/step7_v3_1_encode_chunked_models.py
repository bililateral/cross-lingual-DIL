#!/usr/bin/env python3
"""Encode complete Step7-v3.1 seller histories through shared, field-aware chunks."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import platform
import string
from bisect import bisect_right
from pathlib import Path

import numpy as np

import step7_v3_1_source_data as source
import step7_v3_1_common as common


ENCODER_SCRIPT = Path(__file__).resolve()
PREFERRED_PUNCTUATION = frozenset(string.punctuation)


def require_tokenizer_stack():
    try:
        import transformers  # type: ignore
        from transformers import AutoTokenizer  # type: ignore
    except ModuleNotFoundError as exc:  # pragma: no cover - runtime dependent
        raise SystemExit("Step7-v3.1 tokenizer preflight requires transformers.") from exc
    return transformers, AutoTokenizer


def require_gpu_stack():
    try:
        import torch  # type: ignore
        import transformers  # type: ignore
        from sentence_transformers import SentenceTransformer  # type: ignore
        from transformers import AutoTokenizer  # type: ignore
    except ModuleNotFoundError as exc:  # pragma: no cover - runtime dependent
        raise SystemExit(
            "Step7-v3.1 formal encoding requires torch, transformers, and sentence-transformers."
        ) from exc
    if not torch.cuda.is_available():
        raise SystemExit(
            "Step7-v3.1 formal encoding requires the Linux CUDA host; Windows is for contracts only."
        )
    return torch, transformers, SentenceTransformer, AutoTokenizer


def verify_source_preparation(policy: dict) -> tuple[dict, list[dict]]:
    paths = common.validate_source_encoding_artifacts(policy)
    manifest = common.load_json(paths["preparation_manifest"])
    corpus_path = paths["field_corpus"]
    rows = common.load_jsonl(corpus_path)
    common.validate_field_corpus_rows(policy, rows)
    return manifest, rows


def verify_label_free_gpu_sync(
    policy: dict, policy_path: Path
) -> tuple[dict, dict[str, dict]]:
    sync_path = common.resolve(policy["outputs"]["gpu_sync_manifest"])
    if not sync_path.is_file():
        raise FileNotFoundError("Build and transfer the Step7-v3.1 GPU sync manifest first")
    manifest = common.load_json(sync_path)
    if (
        manifest.get("step") != "step7_v3_1_label_free_gpu_sync"
        or manifest.get("version") != policy["version"]
        or manifest.get("policy_sha256") != common.sha256_file(policy_path)
        or manifest.get("policy_contract_sha256") != common.canonical_hash(policy)
        or manifest.get("label_files_included") is not False
        or manifest.get("raw_source_files_included") is not False
    ):
        raise ValueError("Step7-v3.1 GPU sync role, policy, or isolation drift")
    for record in manifest.get("files", []):
        common.verify_file_record(record, "transferred GPU payload")
    present_forbidden = [
        value
        for value in manifest.get("forbidden_workspace_paths", [])
        if common.resolve(value).exists()
    ]
    if present_forbidden:
        raise ValueError(
            "Step7-v3.1 formal GPU workspace contains a forbidden source/label file: "
            f"{present_forbidden[0]}"
        )
    fingerprints = {}
    for model_key, cfg in policy["embedding_models"].items():
        observed = source.validate_model_content_pin(model_key, cfg)
        expected = manifest.get("model_directories", {}).get(model_key)
        if expected != {"path": cfg["local_path"], **observed}:
            raise ValueError(f"Step7-v3.1 GPU model fingerprint drift: {model_key}")
        fingerprints[model_key] = observed
    return manifest, fingerprints


def tokenizer_result_ids(result: dict) -> list[int]:
    values = result["input_ids"]
    if values and isinstance(values[0], list):
        if len(values) != 1:
            raise ValueError("Step7-v3.1 expected one tokenizer row")
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


def all_token_lengths(
    tokenizers: dict[str, object], policy: dict, text: str
) -> dict[str, int]:
    return {
        key: token_length(tokenizer, policy["embedding_models"][key]["text_prefix"] + text)
        for key, tokenizer in tokenizers.items()
    }


def preferred_boundaries(text: str) -> list[int]:
    output = [
        index
        for index, character in enumerate(text, start=1)
        if character.isspace() or character in PREFERRED_PUNCTUATION
    ]
    if not output or output[-1] != len(text):
        output.append(len(text))
    return output


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
        if offsets and isinstance(offsets[0], list) and offsets[0] and isinstance(
            offsets[0][0], (list, tuple)
        ):
            if len(offsets) != 1:
                raise ValueError("unexpected batched offset mapping")
            offsets = offsets[0]
        special_count = int(tokenizer.num_special_tokens_to_add(pair=False))
        available = budget - special_count
        if available <= 0 or len(offsets) <= available:
            raise ValueError("invalid tokenizer offset capacity")
        usable = [
            int(stop) - len(prefix)
            for start, stop in offsets[:available]
            if int(stop) > len(prefix)
        ]
        if not usable:
            raise ValueError("token budget does not reach chunk content")
        return max(1, min(len(remaining), max(usable)))
    except (KeyError, TypeError, ValueError, NotImplementedError):
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
        raise ValueError("Step7-v3.1 cannot chunk an empty remainder")
    budget = int(
        policy["shared_chunking"][
            "token_budget_including_model_prefix_and_special_tokens"
        ]
    )
    lengths = all_token_lengths(tokenizers, policy, remaining)
    if max(lengths.values()) <= budget:
        return remaining, lengths
    conservative = min(
        conservative_tokenizer_end(
            tokenizers[key],
            policy["embedding_models"][key]["text_prefix"],
            remaining,
            budget,
        )
        for key in tokenizers
    )
    boundaries = preferred_boundaries(remaining)
    boundary_index = bisect_right(boundaries, conservative) - 1
    end = boundaries[boundary_index] if boundary_index >= 0 else conservative
    end = max(1, min(end, len(remaining)))
    while end > 1:
        candidate = remaining[:end]
        candidate_lengths = all_token_lengths(tokenizers, policy, candidate)
        if candidate.strip() and max(candidate_lengths.values()) <= budget:
            break
        boundary_index = bisect_right(boundaries, end - 1) - 1
        end = boundaries[boundary_index] if boundary_index >= 0 else end - 1
    candidate = remaining[:end]
    candidate_lengths = all_token_lengths(tokenizers, policy, candidate)
    if not candidate.strip() or max(candidate_lengths.values()) > budget:
        raise ValueError("Step7-v3.1 cannot form a nonempty common-token-budget chunk")
    # The offset-derived endpoint is conservative.  Expand through immediately
    # following preferred boundaries while the exact five-tokenizer check passes.
    for next_end in boundaries[bisect_right(boundaries, end) :]:
        expanded = remaining[:next_end]
        expanded_lengths = all_token_lengths(tokenizers, policy, expanded)
        if max(expanded_lengths.values()) > budget:
            break
        candidate, candidate_lengths, end = expanded, expanded_lengths, next_end
    return candidate, candidate_lengths


def field_group_map(policy: dict) -> dict[str, str]:
    return {
        field: group
        for group, fields in policy["clean_text_contract"]["field_groups"].items()
        for field in fields
    }


def build_shared_chunks(
    policy: dict, field_rows: list[dict], tokenizers: dict[str, object]
) -> tuple[list[dict], dict]:
    output = []
    groups = field_group_map(policy)
    for seller in field_rows:
        for field in policy["clean_text_contract"]["fields_in_order"]:
            text = seller["field_texts"][field]
            position = 0
            chunk_index = 0
            while position < len(text):
                chunk, lengths = choose_shared_chunk(tokenizers, policy, text[position:])
                end = position + len(chunk)
                identity = {
                    "seller_uid": seller["seller_uid"],
                    "field_name": field,
                    "chunk_index": chunk_index,
                    "char_start": position,
                    "char_end": end,
                    "text_sha256": common.sha256_text(chunk),
                }
                output.append(
                    {
                        "chunk_uid": common.canonical_hash(identity),
                        "seller_uid": seller["seller_uid"],
                        "split_name": seller["split_name"],
                        "field_name": field,
                        "field_group": groups[field],
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
    audit = common.validate_shared_chunk_rows(policy, field_rows, output)
    token_values = {
        model_key: np.asarray(
            [row["token_lengths"][model_key] for row in output], dtype=np.int64
        )
        for model_key in policy["embedding_models"]
    }
    audit["token_length_diagnostics"] = {
        model_key: {
            "minimum": int(np.min(values)),
            "median": float(np.median(values)),
            "p90": float(np.quantile(values, 0.90)),
            "p95": float(np.quantile(values, 0.95)),
            "maximum": int(np.max(values)),
            "over_budget_count": int(np.sum(values > 480)),
        }
        for model_key, values in token_values.items()
    }
    return output, audit


def validate_expected_chunk_preflight(
    policy: dict,
    field_rows: list[dict],
    chunk_rows: list[dict],
    chunk_audit: dict,
    transformers_version: str,
) -> dict:
    expected = policy["shared_chunking"]["expected_label_free_tokenizer_preflight"]
    observed = {
        "reference_transformers_version": transformers_version,
        "field_corpus_sha256": common.sha256_file(
            common.resolve(policy["outputs"]["field_corpus"])
        ),
        "shared_chunk_rows_canonical_sha256": common.canonical_hash(chunk_rows),
        "seller_count": chunk_audit["seller_count"],
        "chunk_count": chunk_audit["chunk_count"],
        "nonempty_field_count": chunk_audit["nonempty_field_count"],
        "chunk_count_min": chunk_audit["chunk_count_min"],
        "chunk_count_median": chunk_audit["chunk_count_median"],
        "chunk_count_p90": chunk_audit["chunk_count_p90"],
        "chunk_count_p95": chunk_audit["chunk_count_p95"],
        "chunk_count_max": chunk_audit["chunk_count_max"],
        "missing_group_seller_counts_audit_only": chunk_audit[
            "missing_group_seller_counts_audit_only"
        ],
        "token_length_maximum_by_model": {
            key: value["maximum"]
            for key, value in chunk_audit["token_length_diagnostics"].items()
        },
        "over_budget_count_by_model": {
            key: value["over_budget_count"]
            for key, value in chunk_audit["token_length_diagnostics"].items()
        },
    }
    if observed != expected or len(field_rows) != int(expected["seller_count"]):
        raise ValueError(
            "Step7-v3.1 shared chunk corpus does not replay the frozen label-free preflight"
        )
    return observed


def load_tokenizers(policy: dict, tokenizer_cls) -> dict[str, object]:
    output = {}
    for model_key, cfg in policy["embedding_models"].items():
        output[model_key] = tokenizer_cls.from_pretrained(
            str(common.resolve(cfg["local_path"])),
            local_files_only=True,
            trust_remote_code=bool(cfg.get("trust_remote_code", False)),
            use_fast=True,
        )
    return output


def create_sentence_transformer(sentence_transformer_cls, cfg: dict):
    kwargs = {
        "device": "cuda",
        "trust_remote_code": bool(cfg.get("trust_remote_code", False)),
    }
    model_path = str(common.resolve(cfg["local_path"]))
    try:
        model = sentence_transformer_cls(model_path, local_files_only=True, **kwargs)
    except TypeError:
        model = sentence_transformer_cls(model_path, **kwargs)
    model.max_seq_length = int(cfg["max_length"])
    return model


def tokenizer_digest_and_lengths(
    tokenizer, texts: list[str], prefix: str
) -> tuple[str, list[int]]:
    digest = hashlib.sha256()
    lengths = []
    batch_size = 128
    for start in range(0, len(texts), batch_size):
        batch = [prefix + text for text in texts[start : start + batch_size]]
        encoded = tokenizer(
            batch,
            add_special_tokens=True,
            padding=False,
            truncation=False,
        )["input_ids"]
        if len(encoded) != len(batch) or any(
            not isinstance(ids, list) for ids in encoded
        ):
            raise ValueError("Step7-v3.1 tokenizer digest batch schema drift")
        for ids in encoded:
            lengths.append(len(ids))
            digest.update(len(ids).to_bytes(8, "big"))
            for value in ids:
                digest.update(int(value).to_bytes(8, "big", signed=True))
    if len(lengths) != len(texts):
        raise ValueError("Step7-v3.1 tokenizer digest length replay is incomplete")
    return digest.hexdigest(), lengths


def tokenizer_digest(tokenizer, texts: list[str], prefix: str) -> str:
    return tokenizer_digest_and_lengths(tokenizer, texts, prefix)[0]


def encode_model(
    policy: dict,
    model_key: str,
    chunk_rows: list[dict],
    pair_rows: list[dict],
    shared_tokenizer,
    sentence_transformer_cls,
    torch_module,
    model_fingerprint: dict,
    provenance: dict,
) -> dict:
    cfg = policy["embedding_models"][model_key]
    layout = source.validate_sentence_transformer_layout(model_key, cfg)
    texts = [row["text"] for row in chunk_rows]
    prefixed = [cfg["text_prefix"] + text for text in texts]
    model = create_sentence_transformer(sentence_transformer_cls, cfg)
    shared_digest, shared_lengths = tokenizer_digest_and_lengths(
        shared_tokenizer, texts, cfg["text_prefix"]
    )
    runtime_digest, runtime_lengths = tokenizer_digest_and_lengths(
        model.tokenizer, texts, cfg["text_prefix"]
    )
    if shared_digest != runtime_digest:
        raise ValueError(f"Step7-v3.1 chunk/runtime tokenizer drift: {model_key}")
    budget = int(
        policy["shared_chunking"][
            "token_budget_including_model_prefix_and_special_tokens"
        ]
    )
    registered_lengths = [row["token_lengths"][model_key] for row in chunk_rows]
    if (
        shared_lengths != registered_lengths
        or runtime_lengths != registered_lengths
        or max(runtime_lengths) > budget
    ):
        raise ValueError(f"Step7-v3.1 runtime chunk length replay failed: {model_key}")
    embeddings = np.asarray(
        model.encode(
            prefixed,
            batch_size=int(cfg["batch_size"]),
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,
        ),
        dtype=np.float32,
    )
    if embeddings.ndim != 2 or embeddings.shape[0] != len(chunk_rows):
        raise ValueError(f"Step7-v3.1 invalid chunk embedding shape: {model_key}")
    if not np.all(np.isfinite(embeddings)):
        raise ValueError(f"Step7-v3.1 non-finite chunk embeddings: {model_key}")
    norms = np.linalg.norm(embeddings, axis=1)
    if float(np.max(np.abs(norms - 1.0))) > 1e-3:
        raise ValueError(f"Step7-v3.1 chunk embeddings are not normalized: {model_key}")
    score_rows = common.compute_pair_score_rows(
        policy, cfg, embeddings, chunk_rows, pair_rows
    )
    outputs = policy["outputs"]
    matrix_path = common.resolve(
        outputs["embedding_matrix_template"].format(model_key=model_key)
    )
    score_path = common.resolve(
        outputs["embedding_pair_scores_template"].format(model_key=model_key)
    )
    manifest_path = common.resolve(
        outputs["embedding_manifest_template"].format(model_key=model_key)
    )
    common.write_npy_immutable(matrix_path, embeddings)
    common.write_csv_immutable(score_path, score_rows)
    manifest = {
        "step": "step7_v3_1_encode_full_text_shared_chunks",
        "version": policy["version"],
        "model_key": model_key,
        "repo_id": cfg["repo_id"],
        "local_path": cfg["local_path"],
        "aggregate_feature_names": common.aggregate_feature_names(cfg),
        "primary_raw_encoder_feature_name": common.primary_feature_name(cfg),
        "layout_validation": layout,
        "model_fingerprint": model_fingerprint,
        **provenance,
        "feature_generation_reads_label_values": False,
        "label_or_raw_source_files_present_in_gpu_workspace": False,
        "same_shared_chunks_for_all_models": True,
        "text_prefix": cfg["text_prefix"],
        "max_length": int(cfg["max_length"]),
        "shared_chunk_token_budget": budget,
        "batch_size": int(cfg["batch_size"]),
        "chunk_uids": [row["chunk_uid"] for row in chunk_rows],
        "shape": list(embeddings.shape),
        "pair_count": len(score_rows),
        "maximum_unit_norm_error": float(np.max(np.abs(norms - 1.0))),
        "shared_tokenizer_digest": shared_digest,
        "runtime_sentence_transformer_tokenizer_digest": runtime_digest,
        "runtime_token_lengths_replay_shared_manifest": True,
        "shared_chunks_sha256": common.sha256_file(
            common.resolve(outputs["shared_chunks"])
        ),
        "shared_chunks_manifest_sha256": common.sha256_file(
            common.resolve(outputs["shared_chunks_manifest"])
        ),
        "pair_manifest_sha256": common.sha256_file(
            common.resolve(policy["outputs"]["pair_manifest"])
        ),
        "embedding_matrix_sha256": common.sha256_file(matrix_path),
        "pair_scores_sha256": common.sha256_file(score_path),
        "device": "cuda",
        "gpu_name": torch_module.cuda.get_device_name(0),
        "torch_version": torch_module.__version__,
        "transformers_version": importlib.metadata.version("transformers"),
        "sentence_transformers_version": importlib.metadata.version(
            "sentence-transformers"
        ),
    }
    common.write_json_immutable(manifest_path, manifest)
    del model, embeddings
    torch_module.cuda.empty_cache()
    return manifest


def output_record(path_value: str) -> dict:
    path = common.resolve(path_value)
    if not path.is_file():
        raise FileNotFoundError(f"Step7-v3.1 expected GPU output is missing: {path}")
    return {
        "path": common.relative(path),
        "size_bytes": path.stat().st_size,
        "sha256": common.sha256_file(path),
    }


def expected_payload_paths(policy: dict) -> list[str]:
    outputs = policy["outputs"]
    paths = [outputs["shared_chunks"], outputs["shared_chunks_manifest"]]
    for model_key in policy["embedding_models"]:
        paths.extend(
            [
                outputs["embedding_matrix_template"].format(model_key=model_key),
                outputs["embedding_manifest_template"].format(model_key=model_key),
                outputs["embedding_pair_scores_template"].format(model_key=model_key),
            ]
        )
    return paths


def write_output_bundle(policy: dict, provenance: dict) -> dict:
    records = [output_record(path) for path in expected_payload_paths(policy)]
    payload = {
        "step": "step7_v3_1_label_free_gpu_output_bundle",
        "version": policy["version"],
        **provenance,
        "label_or_raw_source_files_present_in_gpu_workspace": False,
        "file_count": len(records),
        "total_file_bytes": sum(record["size_bytes"] for record in records),
        "files": records,
    }
    common.write_json_immutable(
        common.resolve(policy["outputs"]["gpu_output_manifest"]), payload
    )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", default=str(common.DEFAULT_POLICY))
    parser.add_argument("--embedding-model", action="append", dest="embedding_models")
    parser.add_argument("--validate-config-only", action="store_true")
    parser.add_argument("--validate-shared-chunking-only", action="store_true")
    args = parser.parse_args()
    if args.validate_config_only and args.validate_shared_chunking_only:
        raise SystemExit("Choose only one Step7-v3.1 validation mode")

    policy_path = common.resolve(args.policy)
    policy = common.load_json(policy_path)
    common.validate_policy(policy)
    source_manifest, field_rows = verify_source_preparation(policy)
    selected_models = args.embedding_models or list(policy["embedding_models"])
    unknown = sorted(set(selected_models) - set(policy["embedding_models"]))
    if unknown:
        raise ValueError(f"Unknown Step7-v3.1 encoder keys: {unknown}")
    layouts = {
        key: source.validate_sentence_transformer_layout(
            key, policy["embedding_models"][key]
        )
        for key in selected_models
    }
    if args.validate_config_only:
        print(
            json.dumps(
                {
                    "status": "pass",
                    "selected_embedding_models": selected_models,
                    "model_layouts": layouts,
                    "field_seller_count": len(field_rows),
                    "source_preparation_manifest_version": source_manifest["version"],
                    "formal_execution_requires_linux_cuda": True,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return

    if args.validate_shared_chunking_only:
        transformers_module, tokenizer_cls = require_tokenizer_stack()
        tokenizers = load_tokenizers(policy, tokenizer_cls)
        chunk_rows, chunk_audit = build_shared_chunks(policy, field_rows, tokenizers)
        preflight = validate_expected_chunk_preflight(
            policy,
            field_rows,
            chunk_rows,
            chunk_audit,
            transformers_module.__version__,
        )
        print(
            json.dumps(
                {
                    "status": "pass",
                    "role": "cpu_label_free_tokenizer_preflight_not_formal_encoding",
                    "labels_or_evidence_types_read": False,
                    "frozen_preflight_replay": preflight,
                    "chunk_audit": chunk_audit,
                    "transformers_version": transformers_module.__version__,
                    "gpu_required": False,
                    "artifacts_written": False,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return

    gpu_sync, model_fingerprints = verify_label_free_gpu_sync(policy, policy_path)
    sync_path = common.resolve(policy["outputs"]["gpu_sync_manifest"])
    provenance = {
        "policy_sha256": common.sha256_file(policy_path),
        "policy_contract_sha256": common.canonical_hash(policy),
        "generator_script_path": common.relative(ENCODER_SCRIPT),
        "generator_script_sha256": common.sha256_file(ENCODER_SCRIPT),
        "gpu_sync_manifest_sha256": common.sha256_file(sync_path),
        "source_preparation_manifest_sha256": common.sha256_file(
            common.resolve(policy["outputs"]["preparation_manifest"])
        ),
    }
    if gpu_sync["source_preparation_manifest_sha256"] != provenance[
        "source_preparation_manifest_sha256"
    ]:
        raise ValueError("Step7-v3.1 GPU sync/source preparation drift")
    torch_module, transformers_module, sentence_transformer_cls, tokenizer_cls = (
        require_gpu_stack()
    )
    tokenizers = load_tokenizers(policy, tokenizer_cls)
    chunk_rows, chunk_audit = build_shared_chunks(policy, field_rows, tokenizers)
    preflight = validate_expected_chunk_preflight(
        policy,
        field_rows,
        chunk_rows,
        chunk_audit,
        transformers_module.__version__,
    )
    outputs = policy["outputs"]
    chunks_path = common.resolve(outputs["shared_chunks"])
    common.write_jsonl_immutable(chunks_path, chunk_rows)
    chunk_manifest = {
        "step": "step7_v3_1_build_complete_shared_chunks",
        "version": policy["version"],
        **provenance,
        "labels_or_evidence_types_read": False,
        "chunking_contract": policy["shared_chunking"],
        "field_group_contract": policy["clean_text_contract"]["field_groups"],
        "chunk_audit": chunk_audit,
        "frozen_label_free_tokenizer_preflight_replay": preflight,
        "field_corpus_sha256": common.sha256_file(
            common.resolve(outputs["field_corpus"])
        ),
        "shared_chunks": output_record(outputs["shared_chunks"]),
        "model_fingerprints": model_fingerprints,
        "transformers_version": transformers_module.__version__,
    }
    common.write_json_immutable(
        common.resolve(outputs["shared_chunks_manifest"]), chunk_manifest
    )
    source_policy = common.source_policy(policy)
    pair_rows = source.load_csv(common.resolve(outputs["pair_manifest"]))
    source.validate_public_pair_rows(source_policy, pair_rows)
    manifests = {
        model_key: encode_model(
            policy,
            model_key,
            chunk_rows,
            pair_rows,
            tokenizers[model_key],
            sentence_transformer_cls,
            torch_module,
            model_fingerprints[model_key],
            provenance,
        )
        for model_key in selected_models
    }
    bundle = None
    if set(selected_models) == set(policy["embedding_models"]):
        bundle = write_output_bundle(policy, provenance)
    print(
        json.dumps(
            {
                "status": "pass",
                "platform": platform.platform(),
                "chunk_count": len(chunk_rows),
                "embedding_models": list(manifests),
                "gpu_output_bundle_written": bundle is not None,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
