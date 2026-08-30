#!/usr/bin/env python3
"""Replay the complete frozen Step7 LaBSE workload on the Linux GPU."""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import importlib.metadata
import io
import json
import math
import os
import platform
import shutil
import tempfile
from pathlib import Path
from typing import Any, Mapping

import numpy as np

import step28_v13_v1_13_v9_4_1_model_experiment_common_v1 as base
import step7_v4_common as step7_common
import step7_v4_encode_item_models as step7_encoder


POLICY_PATH = (
    base.ROOT
    / "schema"
    / "step28_v13_v1_13_v9_4_1_full_english_compatibility_policy_v2.json"
)
RUNNER_PATH = (
    base.ROOT
    / "scripts"
    / "run_step28_v13_v1_13_v9_4_1_full_english_compatibility_v2_linux_20260830.sh"
)
EXPECTED_POLICY_SIZE_BYTES = 6015
EXPECTED_POLICY_SHA256 = (
    "1aa14e95f9cf939dba454327ffab81b2ead3ca2d214f083fca92dc22ea7a3722"
)
EXPECTED_POLICY_SELF_HASH = (
    "b269b6654f34f4bf1717c8dc38cc9011ec0b3c1db4e3ff3f2e294666cbb531f1"
)
EXPECTED_VERSION = (
    "2026-08-30-step28-v13-v1.13-v9.4.1-full-english-compatibility-v2"
)
EXPECTED_STATUS = "IMPLEMENTATION_ONLY_NO_GPU_REPLAY_NO_MODEL_TRAINING"
RESULT_STEP = "step28_v13_v1_13_v9_4_1_full_english_compatibility_linux_v2"
RESULT_STATUS = "PASSED_FULL_ENGLISH_LABSE_EXACT_COMPATIBILITY_REPLAY"


class FullCompatibilityError(base.ModelExperimentContractError):
    """Raised when the complete compatibility workload does not replay."""


def _exact_file_record(path: Path) -> dict[str, Any]:
    return {
        "path": path.relative_to(base.ROOT).as_posix(),
        "size_bytes": path.stat().st_size,
        "sha256": base.sha256_file(path),
    }


def _implementation_records() -> dict[str, dict[str, Any]]:
    paths = {
        "compatibility_policy": POLICY_PATH,
        "linux_replay": Path(__file__).resolve(),
        "linux_runner": RUNNER_PATH,
    }
    if any(not path.is_file() for path in paths.values()):
        raise FullCompatibilityError("Compatibility-v2 implementation file is missing")
    return {
        role: _exact_file_record(path.resolve())
        for role, path in sorted(paths.items())
    }


def load_policy() -> dict[str, Any]:
    raw = POLICY_PATH.read_bytes()
    if len(raw) != EXPECTED_POLICY_SIZE_BYTES:
        raise FullCompatibilityError("Compatibility-v2 policy byte-size drift")
    if hashlib.sha256(raw).hexdigest() != EXPECTED_POLICY_SHA256:
        raise FullCompatibilityError("Compatibility-v2 policy SHA-256 drift")
    try:
        policy = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise FullCompatibilityError("Compatibility-v2 policy is not UTF-8 JSON") from exc
    if not isinstance(policy, dict):
        raise FullCompatibilityError("Compatibility-v2 policy is not an object")
    base.verify_self_hash(policy, label="compatibility-v2 policy")
    if (
        policy.get("canonical_self_hash") != EXPECTED_POLICY_SELF_HASH
        or policy.get("version") != EXPECTED_VERSION
        or policy.get("status") != EXPECTED_STATUS
    ):
        raise FullCompatibilityError("Compatibility-v2 policy identity drift")
    permissions = policy.get("permissions")
    if not isinstance(permissions, dict) or any(permissions.values()):
        raise FullCompatibilityError("Compatibility-v2 policy grants forbidden authority")
    replay = policy["full_replay"]
    if (
        replay.get("score_file_exact_byte_match_required") is not True
        or replay.get("embedding_matrix_exact_byte_match_required") is not True
        or replay.get("numeric_tolerance_or_fixture_reselection_allowed") is not False
        or replay.get("complete_workload_must_be_rebuilt_from_unique_texts") is not True
    ):
        raise FullCompatibilityError("Compatibility-v2 exact replay contract drift")
    return policy


def validate_inputs(policy: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Path]]:
    paths: dict[str, Path] = {}
    for role, spec in policy["authority"].items():
        paths[role] = base.verify_file_pin(spec, label=f"compatibility authority {role}")
    for role, spec in policy["frozen_label_free_inputs"].items():
        paths[role] = base.verify_file_pin(spec, label=f"label-free replay input {role}")

    base_policy = base.load_policy(paths["base_model_experiment_policy"])
    runtime_gate = base.validate_encoding_runtime(base_policy)
    if runtime_gate != {
        "sentence_transformers": "5.6.0",
        "step7_policy_sha256": policy["frozen_label_free_inputs"]["step7_policy"][
            "sha256"
        ],
        "payload_count": 4,
    }:
        raise FullCompatibilityError("Compatibility-v2 four-payload gate drift")
    _validate_original_runtime_contract(policy, paths, base_policy)
    return base_policy, paths


def _validate_original_runtime_contract(
    policy: Mapping[str, Any],
    paths: Mapping[str, Path],
    base_policy: Mapping[str, Any],
) -> None:
    original = json.loads(paths["original_labse_runtime"].read_text(encoding="utf-8"))
    deterministic = original.get("deterministic_gpu_runtime", {})
    expected_runtime = {
        "python": original.get("python_version"),
        "numpy": original.get("numpy_version"),
        "torch": original.get("torch_version"),
        "torch_cuda_runtime": str(original.get("torch_cuda_runtime_version")),
        "cudnn_runtime": original.get("cudnn_runtime_version"),
        "transformers": original.get("transformers_version"),
        "sentence_transformers": original.get("sentence_transformers_version"),
        "gpu_name": original.get("gpu_name"),
        "gpu_compute_capability": original.get("gpu_compute_capability"),
        "cublas_workspace_config": deterministic.get("cublas_workspace_config"),
        "tokenizers_parallelism": deterministic.get("tokenizers_parallelism"),
        "cuda_matmul_allow_tf32": deterministic.get("cuda_matmul_allow_tf32"),
        "cudnn_allow_tf32": deterministic.get("cudnn_allow_tf32"),
        "deterministic_algorithms_enabled": deterministic.get(
            "deterministic_algorithms_enabled"
        ),
        "cudnn_benchmark": deterministic.get("cudnn_benchmark"),
        "cudnn_deterministic": deterministic.get("cudnn_deterministic"),
    }
    replay = policy["full_replay"]
    expected_shape = [replay["shared_chunk_count"], replay["labse_dimension"]]
    cfg = json.loads(paths["step7_policy"].read_text(encoding="utf-8"))[
        "embedding_models"
    ]["labse"]
    if (
        expected_runtime != policy["exact_runtime"]
        or original.get("embedding_matrix_shape") != expected_shape
        or original.get("embedding_matrix_dtype") != replay["embedding_dtype"]
        or original.get("embedding_matrix_content_sha256")
        != replay["embedding_matrix_sha256"]
        or original.get("runtime_sentence_transformer_tokenizer_digest")
        != replay["labse_token_id_stream_sha256"]
        or original.get("batch_size") != replay["labse_batch_size"]
        or original.get("pair_count") != replay["opaque_pair_count"]
        or cfg.get("batch_size") != replay["labse_batch_size"]
        or cfg.get("expected_dimension") != replay["labse_dimension"]
        or cfg.get("expected_content_sha256")
        != base_policy["labse_encoding"]["model_content_sha256"]
    ):
        raise FullCompatibilityError("Compatibility-v2 original runtime binding drift")


def _require_schema(rows: list[Mapping[str, Any]], expected: list[str], label: str) -> None:
    if not rows or any(list(row) != expected for row in rows):
        raise FullCompatibilityError(f"Compatibility-v2 {label} schema drift")


def _load_and_rebuild_workload(
    policy: Mapping[str, Any], paths: Mapping[str, Path], tokenizer_cls
) -> tuple[dict[str, Any], list[dict], list[dict], list[dict], dict[str, Any]]:
    step7_policy = step7_common.load_policy()
    unique_rows = step7_common.load_jsonl(paths["unique_texts"])
    expected_chunks = step7_common.load_jsonl(paths["shared_chunks"])
    seller_rows = step7_common.load_jsonl(paths["opaque_seller_text_index"])
    pair_rows = step7_common.load_csv(paths["opaque_pairs"])
    _require_schema(unique_rows, ["text_uid", "text", "text_sha256"], "text")
    _require_schema(
        seller_rows,
        ["seller_uid", "field_name", "text_uid", "multiplicity"],
        "seller-text",
    )
    _require_schema(
        pair_rows,
        ["pair_uid", "seller_uid_left", "seller_uid_right"],
        "pair",
    )
    replay = policy["full_replay"]
    if (
        len(unique_rows) != int(replay["unique_text_count"])
        or len(expected_chunks) != int(replay["shared_chunk_count"])
        or len(seller_rows) != int(replay["seller_text_row_count"])
        or len({row["seller_uid"] for row in seller_rows})
        != int(replay["opaque_seller_count"])
        or len(pair_rows) != int(replay["opaque_pair_count"])
    ):
        raise FullCompatibilityError("Compatibility-v2 complete workload count drift")

    tokenizers = step7_encoder.load_tokenizers(step7_policy, tokenizer_cls)
    rebuilt_chunks, chunk_audit = step7_encoder.build_shared_chunks(
        step7_policy, unique_rows, tokenizers
    )
    if rebuilt_chunks != expected_chunks:
        raise FullCompatibilityError(
            "Compatibility-v2 complete shared-chunk byte/field replay drift"
        )
    tokenizer_audit: dict[str, Any] = {}
    texts = [str(row["text"]) for row in rebuilt_chunks]
    for model_key, cfg in step7_policy["embedding_models"].items():
        digest, lengths = step7_encoder.tokenizer_digest_and_lengths(
            tokenizers[model_key], texts, str(cfg["text_prefix"])
        )
        expected_lengths = [
            int(row["token_lengths"][model_key]) for row in rebuilt_chunks
        ]
        if lengths != expected_lengths:
            raise FullCompatibilityError(
                f"Compatibility-v2 full tokenizer length drift for {model_key}"
            )
        tokenizer_audit[model_key] = {
            "token_id_stream_sha256": digest,
            "row_count": len(lengths),
            "maximum_token_length": max(lengths),
        }
    expected_manifest = json.loads(paths["shared_chunks_manifest"].read_text("utf-8"))
    if tokenizer_audit["labse"]["token_id_stream_sha256"] != replay[
        "labse_token_id_stream_sha256"
    ]:
        raise FullCompatibilityError("Compatibility-v2 frozen LaBSE token stream drift")
    if expected_manifest.get("tokenizer_digests", {}).get("labse") != replay[
        "labse_token_id_stream_sha256"
    ]:
        raise FullCompatibilityError("Compatibility-v2 chunk manifest token pin drift")
    del tokenizers
    gc.collect()
    return step7_policy, rebuilt_chunks, seller_rows, pair_rows, {
        "chunk_audit": chunk_audit,
        "tokenizer_audit": tokenizer_audit,
    }


def _preprocess_digest_and_lengths(
    model, texts: list[str], prefix: str, *, batch_size: int = 128
) -> tuple[str, list[int]]:
    preprocess = getattr(model, "preprocess", None)
    if not callable(preprocess):
        raise FullCompatibilityError(
            "sentence-transformers 5.6.0 model lacks preprocess()"
        )
    digest = hashlib.sha256()
    lengths: list[int] = []
    for start in range(0, len(texts), int(batch_size)):
        batch = [prefix + text for text in texts[start : start + int(batch_size)]]
        features = preprocess(batch)
        if not isinstance(features, Mapping) or not {
            "input_ids",
            "attention_mask",
        }.issubset(features):
            raise FullCompatibilityError("SentenceTransformer preprocess output drift")
        input_ids = step7_encoder._tensor_to_numpy(features["input_ids"])
        attention_mask = step7_encoder._tensor_to_numpy(features["attention_mask"])
        if (
            input_ids.ndim != 2
            or attention_mask.shape != input_ids.shape
            or input_ids.shape[0] != len(batch)
        ):
            raise FullCompatibilityError("SentenceTransformer preprocess shape drift")
        for ids, mask in zip(input_ids, attention_mask, strict=True):
            active_indices = np.flatnonzero(np.asarray(mask) != 0)
            if active_indices.size <= 0 or not np.array_equal(
                active_indices,
                np.arange(active_indices[0], active_indices[-1] + 1),
            ):
                raise FullCompatibilityError("SentenceTransformer attention-mask drift")
            values = [int(value) for value in np.asarray(ids)[active_indices]]
            lengths.append(len(values))
            digest.update(len(values).to_bytes(8, "little", signed=False))
            for value in values:
                digest.update(value.to_bytes(8, "little", signed=True))
    return digest.hexdigest(), lengths


def _exact_runtime(
    policy: Mapping[str, Any], torch, transformers, sentence_transformers
) -> dict[str, Any]:
    deterministic = step7_encoder.configure_deterministic_gpu(
        torch, step7_common.load_policy()
    )
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
    if observed != policy["exact_runtime"]:
        raise FullCompatibilityError(
            "Compatibility-v2 exact runtime drift: "
            f"expected={policy['exact_runtime']} observed={observed}"
        )
    if deterministic["random_seed"] != 20260722:
        raise FullCompatibilityError("Compatibility-v2 deterministic seed drift")
    return observed


def _score_diagnostics(expected: bytes, observed: bytes) -> dict[str, Any]:
    expected_rows = list(csv.DictReader(io.StringIO(expected.decode("utf-8"))))
    observed_rows = list(csv.DictReader(io.StringIO(observed.decode("utf-8"))))
    output: dict[str, Any] = {
        "expected_sha256": hashlib.sha256(expected).hexdigest(),
        "observed_sha256": hashlib.sha256(observed).hexdigest(),
        "expected_row_count": len(expected_rows),
        "observed_row_count": len(observed_rows),
        "mismatched_numeric_cell_count": 0,
        "maximum_absolute_difference": None,
        "first_mismatch": None,
    }
    if not expected_rows or not observed_rows or len(expected_rows) != len(observed_rows):
        return output
    maximum = 0.0
    for row_index, (expected_row, observed_row) in enumerate(
        zip(expected_rows, observed_rows, strict=True)
    ):
        if list(expected_row) != list(observed_row):
            return output
        for name in expected_row:
            if name == "pair_uid":
                if expected_row[name] != observed_row[name] and output["first_mismatch"] is None:
                    output["first_mismatch"] = {
                        "row_index": row_index,
                        "column": name,
                        "expected": expected_row[name],
                        "observed": observed_row[name],
                    }
                continue
            if expected_row[name] == observed_row[name]:
                continue
            output["mismatched_numeric_cell_count"] += 1
            difference = abs(float(expected_row[name]) - float(observed_row[name]))
            maximum = max(maximum, difference)
            if output["first_mismatch"] is None:
                output["first_mismatch"] = {
                    "row_index": row_index,
                    "column": name,
                    "expected": expected_row[name],
                    "observed": observed_row[name],
                    "absolute_difference": difference,
                }
    output["maximum_absolute_difference"] = maximum
    return output


def run_replay(policy: Mapping[str, Any]) -> tuple[dict[str, Any], bytes]:
    implementation_records = _implementation_records()
    base_policy, paths = validate_inputs(policy)
    transformers_cpu, tokenizer_cls = step7_encoder.require_tokenizer_stack()
    transformers_cpu.logging.set_verbosity_error()
    step7_policy, chunks, seller_rows, pair_rows, workload_audit = (
        _load_and_rebuild_workload(policy, paths, tokenizer_cls)
    )
    torch, transformers, sentence_transformers, sentence_transformer_cls = (
        step7_encoder.require_gpu_stack(step7_policy)
    )
    runtime = _exact_runtime(policy, torch, transformers, sentence_transformers)
    cfg = step7_policy["embedding_models"]["labse"]
    model, loaded_state = step7_encoder.create_sentence_transformer(
        sentence_transformer_cls, cfg
    )
    texts = [str(row["text"]) for row in chunks]
    runtime_digest, runtime_lengths = _preprocess_digest_and_lengths(
        model, texts, str(cfg["text_prefix"])
    )
    expected_lengths = [int(row["token_lengths"]["labse"]) for row in chunks]
    if (
        runtime_digest != policy["full_replay"]["labse_token_id_stream_sha256"]
        or runtime_lengths != expected_lengths
    ):
        raise FullCompatibilityError("Compatibility-v2 runtime LaBSE token replay drift")

    matrix = model.encode(
        [str(cfg["text_prefix"]) + text for text in texts],
        batch_size=int(cfg["batch_size"]),
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
        prompt=str(cfg["sentence_transformer_prompt"]),
    )
    matrix = np.ascontiguousarray(matrix, dtype=np.float32)
    expected_shape = (
        int(policy["full_replay"]["shared_chunk_count"]),
        int(policy["full_replay"]["labse_dimension"]),
    )
    if matrix.shape != expected_shape or not np.isfinite(matrix).all():
        raise FullCompatibilityError("Compatibility-v2 embedding shape/finite drift")
    matrix_sha256 = step7_encoder.matrix_content_sha256(matrix)
    aggregation = step7_policy["aggregation"]
    score_rows = step7_common.compute_pair_score_rows(
        pair_rows,
        seller_rows,
        chunks,
        matrix,
        cfg,
        top_k=int(aggregation["top_k_item_matches"]),
        decimal_places=int(aggregation["serialized_decimal_places"]),
        similarity_block_rows=int(aggregation["similarity_block_rows"]),
    )
    observed_score_bytes = step7_common.render_csv(score_rows)
    expected_score_bytes = paths["expected_labse_scores"].read_bytes()
    score_diagnostics = _score_diagnostics(expected_score_bytes, observed_score_bytes)
    matrix_exact = matrix_sha256 == policy["full_replay"]["embedding_matrix_sha256"]
    scores_exact = observed_score_bytes == expected_score_bytes
    if not matrix_exact or not scores_exact:
        raise FullCompatibilityError(
            "Compatibility-v2 full workload drift: "
            f"matrix_expected={policy['full_replay']['embedding_matrix_sha256']} "
            f"matrix_observed={matrix_sha256} score_diagnostics="
            f"{json.dumps(score_diagnostics, ensure_ascii=True, sort_keys=True)}"
        )
    if _implementation_records() != implementation_records:
        raise FullCompatibilityError(
            "Compatibility-v2 implementation bytes changed during replay"
        )

    norm_error = float(np.max(np.abs(np.linalg.norm(matrix, axis=1) - 1.0)))
    del matrix, model
    gc.collect()
    torch.cuda.empty_cache()
    input_records = {
        name: _exact_file_record(path)
        for name, path in sorted(paths.items())
    }
    result = {
        "step": RESULT_STEP,
        "status": RESULT_STATUS,
        "policy_canonical_self_hash": policy["canonical_self_hash"],
        "base_model_experiment_policy_canonical_self_hash": base_policy[
            "canonical_self_hash"
        ],
        "input_records": input_records,
        "implementation_records": implementation_records,
        "exact_runtime": runtime,
        "loaded_model_state": loaded_state,
        "workload_audit": workload_audit,
        "runtime_labse_token_id_stream_sha256": runtime_digest,
        "embedding_matrix_shape": list(expected_shape),
        "embedding_matrix_dtype": "float32",
        "embedding_matrix_sha256": matrix_sha256,
        "embedding_matrix_exact_byte_match": True,
        "maximum_unit_norm_error": norm_error,
        "observed_score_file": {
            "path": policy["outputs"]["observed_score_file"],
            "size_bytes": len(observed_score_bytes),
            "sha256": hashlib.sha256(observed_score_bytes).hexdigest(),
        },
        "expected_score_sha256": policy["frozen_label_free_inputs"][
            "expected_labse_scores"
        ]["sha256"],
        "complete_733_pair_score_file_exact_byte_match": True,
        "numeric_tolerance_used": False,
        "fixture_reselection_used": False,
        "supervised_labels_or_identity_evidence_read": False,
        "identity33_read": False,
        "controller_or_membership_read": False,
        "qrels_or_retrieval_truth_read": False,
        "audit_truth_read": False,
        "model_parameters_updated": False,
        "model_training_or_threshold_selection_performed": False,
        "m0_m1_m2_m3_training_authorized": False,
    }
    return result, observed_score_bytes


def _output_root(policy: Mapping[str, Any]) -> Path:
    return base.resolve(policy["outputs"]["success_root"])


def validate_published(policy: Mapping[str, Any], root: Path) -> dict[str, Any]:
    expected_files = sorted(
        [policy["outputs"]["observed_score_file"], policy["outputs"]["success_manifest"]]
    )
    if not root.is_dir() or sorted(path.name for path in root.iterdir()) != expected_files:
        raise FullCompatibilityError("Compatibility-v2 published file universe drift")
    manifest_path = root / policy["outputs"]["success_manifest"]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    base.verify_self_hash(manifest, label="compatibility-v2 success manifest")
    score_path = root / policy["outputs"]["observed_score_file"]
    score_record = manifest.get("observed_score_file", {})
    replay = policy["full_replay"]
    expected_matrix_shape = [replay["shared_chunk_count"], replay["labse_dimension"]]
    expected_input_records: dict[str, dict[str, Any]] = {}
    for registry_name in ("authority", "frozen_label_free_inputs"):
        for role, spec in policy[registry_name].items():
            source = base.verify_file_pin(
                spec, label=f"compatibility-v2 published input {role}"
            )
            expected_input_records[role] = _exact_file_record(source)
    expected_implementation_records = _implementation_records()
    original_runtime = json.loads(
        base.verify_file_pin(
            policy["frozen_label_free_inputs"]["original_labse_runtime"],
            label="compatibility-v2 original runtime revalidation",
        ).read_text(encoding="utf-8")
    )
    expected_loaded_state = {
        "loaded_default_prompt_name": original_runtime.get(
            "loaded_default_prompt_name"
        ),
        "loaded_prompts": original_runtime.get("loaded_prompts"),
        "loaded_native_max_seq_length": original_runtime.get(
            "loaded_native_max_seq_length"
        ),
    }
    norm_error = manifest.get("maximum_unit_norm_error")
    if (
        manifest.get("step") != RESULT_STEP
        or manifest.get("status") != RESULT_STATUS
        or manifest.get("policy_canonical_self_hash") != policy["canonical_self_hash"]
        or manifest.get("base_model_experiment_policy_canonical_self_hash")
        != base.load_policy(
            base.verify_file_pin(
                policy["authority"]["base_model_experiment_policy"],
                label="compatibility-v2 base policy revalidation",
            )
        )["canonical_self_hash"]
        or manifest.get("input_records") != expected_input_records
        or manifest.get("implementation_records")
        != expected_implementation_records
        or manifest.get("exact_runtime") != policy["exact_runtime"]
        or manifest.get("loaded_model_state") != expected_loaded_state
        or manifest.get("runtime_labse_token_id_stream_sha256")
        != replay["labse_token_id_stream_sha256"]
        or manifest.get("embedding_matrix_shape") != expected_matrix_shape
        or manifest.get("embedding_matrix_dtype") != replay["embedding_dtype"]
        or manifest.get("embedding_matrix_sha256")
        != replay["embedding_matrix_sha256"]
        or manifest.get("embedding_matrix_exact_byte_match") is not True
        or not isinstance(norm_error, (int, float))
        or isinstance(norm_error, bool)
        or not math.isfinite(float(norm_error))
        or not (0.0 <= float(norm_error) <= 1e-3)
        or manifest.get("complete_733_pair_score_file_exact_byte_match") is not True
        or manifest.get("expected_score_sha256")
        != policy["frozen_label_free_inputs"]["expected_labse_scores"]["sha256"]
        or manifest.get("numeric_tolerance_used") is not False
        or manifest.get("fixture_reselection_used") is not False
        or manifest.get("supervised_labels_or_identity_evidence_read") is not False
        or manifest.get("identity33_read") is not False
        or manifest.get("controller_or_membership_read") is not False
        or manifest.get("qrels_or_retrieval_truth_read") is not False
        or manifest.get("audit_truth_read") is not False
        or manifest.get("model_parameters_updated") is not False
        or manifest.get("model_training_or_threshold_selection_performed") is not False
        or manifest.get("m0_m1_m2_m3_training_authorized") is not False
        or score_record.get("path") != score_path.name
        or score_record.get("size_bytes") != score_path.stat().st_size
        or score_record.get("sha256") != base.sha256_file(score_path)
        or score_path.read_bytes()
        != base.verify_file_pin(
            policy["frozen_label_free_inputs"]["expected_labse_scores"],
            label="compatibility-v2 expected score revalidation",
        ).read_bytes()
    ):
        raise FullCompatibilityError("Compatibility-v2 published result drift")
    return manifest


def publish(policy: Mapping[str, Any]) -> dict[str, Any]:
    output_root = _output_root(policy)
    if output_root.exists():
        raise FullCompatibilityError(
            "Compatibility-v2 formal replay output already exists; "
            "use validate-output for artifact integrity only"
        )
    output_root.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=".full_english_compatibility_v2.", dir=output_root.parent)
    )
    try:
        result, score_bytes = run_replay(policy)
        (temporary / policy["outputs"]["observed_score_file"]).write_bytes(score_bytes)
        result["canonical_self_hash"] = base.canonical_sha256(result)
        (temporary / policy["outputs"]["success_manifest"]).write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        validate_published(policy, temporary)
        temporary.replace(output_root)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return validate_published(policy, output_root)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        nargs="?",
        choices=("run", "validate-inputs", "validate-output"),
        default="run",
    )
    args = parser.parse_args()
    policy = load_policy()
    if args.command == "validate-inputs":
        base_policy, paths = validate_inputs(policy)
        result = {
            "status": "PASSED_LABEL_FREE_FULL_REPLAY_INPUT_PREFLIGHT",
            "policy_canonical_self_hash": policy["canonical_self_hash"],
            "base_policy_canonical_self_hash": base_policy["canonical_self_hash"],
            "input_count": len(paths),
            "m0_m1_m2_m3_training_authorized": False,
        }
    elif args.command == "validate-output":
        manifest = validate_published(policy, _output_root(policy))
        result = {
            "status": "PASSED_EXISTING_ARTIFACT_INTEGRITY_ONLY_NO_GPU_EXECUTION",
            "validated_manifest_canonical_self_hash": manifest[
                "canonical_self_hash"
            ],
            "formal_gpu_replay_executed_by_this_command": False,
            "m0_m1_m2_m3_training_authorized": False,
        }
    else:
        result = publish(policy)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
