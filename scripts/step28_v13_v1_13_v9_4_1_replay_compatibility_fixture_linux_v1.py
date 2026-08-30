#!/usr/bin/env python3
"""Replay the frozen label-free LaBSE fixture on the formal Linux GPU stack."""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import io
import json
import platform
import shutil
import tempfile
from pathlib import Path
from typing import Any, Mapping

import numpy as np

import step28_v13_v1_13_v9_4_1_model_experiment_common_v1 as common
import step28_v13_v1_13_v9_4_1_prepare_compatibility_fixture_v1 as fixture
import step7_v4_common as step7_common
import step7_v4_encode_item_models as step7_encoder


OBSERVED_SCORES = "observed_labse_scores.csv"
RESULT_MANIFEST = "linux_replay_manifest.json"
CHUNK_COMPARE_FIELDS = (
    "text_uid",
    "chunk_index",
    "char_start",
    "char_end",
    "text",
    "token_lengths",
)
STEP7_IMPLEMENTATION_ROLES = ("common", "sync_builder", "encoder")
STEP28_IMPLEMENTATION_FILES = {
    "common": "scripts/step28_v13_v1_13_v9_4_1_model_experiment_common_v1.py",
    "fixture": "scripts/step28_v13_v1_13_v9_4_1_prepare_compatibility_fixture_v1.py",
    "linux_replay": "scripts/step28_v13_v1_13_v9_4_1_replay_compatibility_fixture_linux_v1.py",
    "linux_runner": "scripts/run_step28_v13_v1_13_v9_4_1_compatibility_linux_20260830.sh",
}


def render_csv(rows: list[Mapping[str, Any]]) -> bytes:
    if not rows:
        raise common.ModelExperimentContractError("Linux fixture score rows are empty")
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(
        buffer, fieldnames=list(rows[0]), lineterminator="\n", extrasaction="raise"
    )
    writer.writeheader()
    writer.writerows(rows)
    return ("\ufeff" + buffer.getvalue()).encode("utf-8")


def _step28_implementation_file_records() -> dict[str, dict[str, Any]]:
    records = {}
    for role, relative in STEP28_IMPLEMENTATION_FILES.items():
        path = common.resolve(relative)
        if not path.is_file():
            raise FileNotFoundError(f"Missing Step28 Linux implementation file: {path}")
        records[role] = {
            "path": relative,
            "size_bytes": path.stat().st_size,
            "sha256": common.sha256_file(path),
        }
    return records


def _load_fixture(policy: Mapping[str, Any]) -> dict[str, Any]:
    root = common.resolve(policy["outputs"]["compatibility_fixture"])
    manifest = fixture.validate_published(policy, root)
    return {
        "root": root,
        "manifest": manifest,
        "pairs": fixture.read_csv(root / "fixture_pairs.csv"),
        "texts": fixture.read_jsonl(root / "fixture_unique_texts.jsonl"),
        "sellers": fixture.read_jsonl(root / "fixture_seller_text_index.jsonl"),
        "chunks": fixture.read_jsonl(root / "fixture_shared_chunks.jsonl"),
        "expected": fixture.read_csv(root / "fixture_expected_labse_scores.csv"),
    }


def _project_chunk_rows(rows: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [
        {field: row[field] for field in CHUNK_COMPARE_FIELDS}
        for row in rows
    ]


def _rebuild_and_verify_shared_chunks(
    step7_policy: Mapping[str, Any], payload: Mapping[str, Any], tokenizer_cls
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    tokenizers = step7_encoder.load_tokenizers(dict(step7_policy), tokenizer_cls)
    rebuilt, chunk_audit = step7_encoder.build_shared_chunks(
        dict(step7_policy), list(payload["texts"]), tokenizers
    )
    if _project_chunk_rows(rebuilt) != _project_chunk_rows(payload["chunks"]):
        raise common.ModelExperimentContractError(
            "Fixture Linux shared-chunk replay drift"
        )
    if (
        len(rebuilt) != 49
        or chunk_audit.get("unique_text_count") != 32
        or chunk_audit.get("chunk_count") != 49
        or chunk_audit.get("text_requiring_chunking_count") != 6
        or chunk_audit.get("exact_character_reconstruction") is not True
    ):
        raise common.ModelExperimentContractError(
            "Fixture Linux shared-chunk coverage drift"
        )
    texts = [str(row["text"]) for row in rebuilt]
    output = {}
    for model_key, cfg in step7_policy["embedding_models"].items():
        digest, lengths = step7_encoder.tokenizer_digest_and_lengths(
            tokenizers[model_key], texts, str(cfg["text_prefix"])
        )
        expected_lengths = [
            int(row["token_lengths"][model_key]) for row in rebuilt
        ]
        if lengths != expected_lengths:
            raise common.ModelExperimentContractError(
                f"Fixture shared-tokenizer length drift for {model_key}"
            )
        if max(lengths) > int(
            step7_policy["shared_chunking"][
                "token_budget_including_model_prefix_and_special_tokens"
            ]
        ):
            raise common.ModelExperimentContractError(
                f"Fixture tokenizer budget drift for {model_key}"
            )
        output[model_key] = {
            "token_id_stream_sha256": digest,
            "maximum_token_length": max(lengths),
            "row_count": len(lengths),
        }
    del tokenizers
    gc.collect()
    return rebuilt, chunk_audit, output


def _score_rows(
    step7_policy: Mapping[str, Any], payload: Mapping[str, Any], matrix: np.ndarray
) -> list[dict[str, Any]]:
    cfg = step7_policy["embedding_models"]["labse"]
    rows = step7_common.compute_pair_score_rows(
        payload["pairs"],
        payload["sellers"],
        payload["chunks"],
        matrix,
        cfg,
        top_k=int(step7_policy["aggregation"]["top_k_item_matches"]),
        decimal_places=int(step7_policy["aggregation"]["serialized_decimal_places"]),
        similarity_block_rows=int(step7_policy["aggregation"]["similarity_block_rows"]),
    )
    names = list(payload["expected"][0])
    observed = [{name: row[name] for name in names} for row in rows]
    if observed != payload["expected"]:
        for index, (actual, expected) in enumerate(
            zip(observed, payload["expected"], strict=True)
        ):
            if actual != expected:
                raise common.ModelExperimentContractError(
                    f"Fixture LaBSE twelve-decimal score drift at row {index}"
                )
        raise common.ModelExperimentContractError("Fixture LaBSE score count drift")
    return observed


def run_replay(policy: Mapping[str, Any]) -> tuple[dict[str, Any], bytes]:
    payload = _load_fixture(policy)
    runtime_gate = common.validate_encoding_runtime(policy)
    step7_policy = step7_common.load_policy()
    implementation_files = step7_common.verify_implementation_files(
        step7_policy, STEP7_IMPLEMENTATION_ROLES
    )
    _transformers, tokenizer_cls = step7_encoder.require_tokenizer_stack()
    replayed_chunks, shared_chunk_audit, tokenizer_audit = (
        _rebuild_and_verify_shared_chunks(step7_policy, payload, tokenizer_cls)
    )
    torch, transformers_gpu, sentence_transformers, sentence_transformer_cls = (
        step7_encoder.require_gpu_stack(step7_policy)
    )
    deterministic = step7_encoder.configure_deterministic_gpu(torch, step7_policy)
    cfg = step7_policy["embedding_models"]["labse"]
    model, loaded_state = step7_encoder.create_sentence_transformer(
        sentence_transformer_cls, cfg
    )
    texts = [str(row["text"]) for row in replayed_chunks]
    runtime_digest, runtime_lengths = (
        step7_encoder.sentence_transformer_tokenizer_digest_and_lengths(
            model, texts, str(cfg["text_prefix"])
        )
    )
    if runtime_lengths != [
        int(row["token_lengths"]["labse"]) for row in replayed_chunks
    ]:
        raise common.ModelExperimentContractError(
            "Fixture SentenceTransformer tokenizer replay drift"
        )
    matrices = []
    for _repeat in range(2):
        encoded = model.encode(
            [str(cfg["text_prefix"]) + text for text in texts],
            batch_size=int(cfg["batch_size"]),
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
            prompt=str(cfg["sentence_transformer_prompt"]),
        )
        matrix = np.ascontiguousarray(encoded, dtype=np.float32)
        if matrix.shape != (len(texts), int(cfg["expected_dimension"])):
            raise common.ModelExperimentContractError("Fixture embedding shape drift")
        if not np.isfinite(matrix).all():
            raise common.ModelExperimentContractError("Fixture embedding is non-finite")
        matrices.append(matrix)
    if not np.array_equal(matrices[0], matrices[1]):
        raise common.ModelExperimentContractError(
            "Fixture repeated encoding is not byte-identical"
        )
    norm_error = float(
        np.max(np.abs(np.linalg.norm(matrices[0], axis=1) - 1.0))
    )
    if norm_error > 1e-3:
        raise common.ModelExperimentContractError("Fixture embedding norm drift")
    score_payload = dict(payload)
    score_payload["chunks"] = replayed_chunks
    observed_rows = _score_rows(step7_policy, score_payload, matrices[0])
    observed_bytes = render_csv(observed_rows)
    matrix_sha256 = hashlib.sha256(
        memoryview(np.ascontiguousarray(matrices[0])).cast("B")
    ).hexdigest()
    result = {
        "step": "step28_v13_v1_13_v9_4_1_linux_compatibility_replay_v1",
        "status": "PASSED_LABEL_FREE_LINUX_LABSE_COMPATIBILITY_REPLAY",
        "policy_canonical_self_hash": policy["canonical_self_hash"],
        "fixture_manifest_canonical_self_hash": payload["manifest"][
            "canonical_self_hash"
        ],
        "runtime_gate": runtime_gate,
        "step7_implementation_files": implementation_files,
        "step28_implementation_files": _step28_implementation_file_records(),
        "shared_chunk_audit": shared_chunk_audit,
        "tokenizer_audit": tokenizer_audit,
        "sentence_transformer_tokenizer_id_stream_sha256": runtime_digest,
        "standalone_and_sentence_transformer_labse_token_ids_identical": (
            runtime_digest == tokenizer_audit["labse"]["token_id_stream_sha256"]
        ),
        "text_prefix": cfg["text_prefix"],
        "sentence_transformer_prompt": cfg["sentence_transformer_prompt"],
        "embedding_shape": list(matrices[0].shape),
        "embedding_dtype": str(matrices[0].dtype),
        "embedding_matrix_sha256": matrix_sha256,
        "maximum_unit_norm_error": norm_error,
        "repeated_encoding_byte_identical": True,
        "loaded_model_state": loaded_state,
        "default_prompt_name_cleared_before_encoding": (
            getattr(model, "default_prompt_name", None) is None
        ),
        "observed_score_file": {
            "path": OBSERVED_SCORES,
            "size_bytes": len(observed_bytes),
            "sha256": hashlib.sha256(observed_bytes).hexdigest(),
        },
        "all_eight_pairs_match_six_scores_at_twelve_decimals": True,
        "labels_or_identity_evidence_read": False,
        "audit_truth_read": False,
        "model_parameters_updated": False,
        "model_training_or_threshold_selection_performed": False,
        "device": "cuda",
        "gpu_name": torch.cuda.get_device_name(0),
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "torch_version": torch.__version__,
        "torch_cuda_runtime_version": str(torch.version.cuda),
        "cudnn_runtime_version": torch.backends.cudnn.version(),
        "transformers_version": transformers_gpu.__version__,
        "sentence_transformers_version": sentence_transformers.__version__,
        "deterministic_gpu_runtime": deterministic,
    }
    result["canonical_self_hash"] = common.canonical_sha256(result)
    del observed_rows, matrices, model
    gc.collect()
    torch.cuda.empty_cache()
    return result, observed_bytes


def _revalidate_actual_runtime_state(
    policy: Mapping[str, Any],
    step7_policy: Mapping[str, Any],
    manifest: Mapping[str, Any],
) -> None:
    """Independently reopen the runtime, implementation, and model-load gates."""

    observed_runtime = common.validate_encoding_runtime(policy)
    observed_implementation = step7_common.verify_implementation_files(
        dict(step7_policy), STEP7_IMPLEMENTATION_ROLES
    )
    if (
        manifest.get("runtime_gate") != observed_runtime
        or manifest.get("step7_implementation_files") != observed_implementation
        or manifest.get("step28_implementation_files")
        != _step28_implementation_file_records()
    ):
        raise common.ModelExperimentContractError(
            "Linux fixture replay runtime or Step7 implementation drift"
        )
    torch, _transformers, _sentence_transformers, sentence_transformer_cls = (
        step7_encoder.require_gpu_stack(dict(step7_policy))
    )
    deterministic = step7_encoder.configure_deterministic_gpu(
        torch, dict(step7_policy)
    )
    cfg = step7_policy["embedding_models"]["labse"]
    model = None
    try:
        model, loaded_state = step7_encoder.create_sentence_transformer(
            sentence_transformer_cls, dict(cfg)
        )
        prompt_cleared = getattr(model, "default_prompt_name", None) is None
        if (
            manifest.get("deterministic_gpu_runtime") != deterministic
            or manifest.get("loaded_model_state") != loaded_state
            or manifest.get("default_prompt_name_cleared_before_encoding")
            is not prompt_cleared
            or not prompt_cleared
            or loaded_state.get("loaded_native_max_seq_length")
            != int(cfg["native_max_seq_length"])
        ):
            raise common.ModelExperimentContractError(
                "Linux fixture replay deterministic runtime or model-load drift"
            )
    finally:
        if model is not None:
            del model
        gc.collect()
        torch.cuda.empty_cache()


def validate_published(policy: Mapping[str, Any], output_root: Path) -> dict[str, Any]:
    manifest_path = output_root / RESULT_MANIFEST
    score_path = output_root / OBSERVED_SCORES
    if not manifest_path.is_file() or not score_path.is_file():
        raise FileNotFoundError("Linux fixture replay output is incomplete")
    manifest = common.load_json(manifest_path)
    common.verify_self_hash(manifest, label="Linux fixture replay manifest")
    fixture_payload = _load_fixture(policy)
    step7_policy = step7_common.load_policy()
    cfg = step7_policy["embedding_models"]["labse"]
    expected_runtime = {
        "sentence_transformers": policy["runtime"]["encoding"][
            "sentence_transformers"
        ],
        "step7_policy_sha256": policy["labse_encoding"]["step7_policy"]["sha256"],
        "payload_count": 4,
    }
    tokenizers = manifest.get("tokenizer_audit", {})
    tokenizer_ok = (
        list(tokenizers) == list(step7_policy["embedding_models"])
        and all(
            isinstance(value, dict)
            and value.get("row_count") == 49
            and isinstance(value.get("maximum_token_length"), int)
            and 0 < value["maximum_token_length"] <= 256
            and isinstance(value.get("token_id_stream_sha256"), str)
            and len(value["token_id_stream_sha256"]) == 64
            for value in tokenizers.values()
        )
    )
    chunk_audit = manifest.get("shared_chunk_audit", {})
    if (
        manifest.get("step")
        != "step28_v13_v1_13_v9_4_1_linux_compatibility_replay_v1"
        or manifest.get("status")
        != "PASSED_LABEL_FREE_LINUX_LABSE_COMPATIBILITY_REPLAY"
        or manifest.get("policy_canonical_self_hash") != policy["canonical_self_hash"]
        or manifest.get("fixture_manifest_canonical_self_hash")
        != fixture_payload["manifest"]["canonical_self_hash"]
        or manifest.get("labels_or_identity_evidence_read") is not False
        or manifest.get("audit_truth_read") is not False
        or manifest.get("model_parameters_updated") is not False
        or manifest.get("model_training_or_threshold_selection_performed") is not False
        or manifest.get("repeated_encoding_byte_identical") is not True
        or manifest.get("all_eight_pairs_match_six_scores_at_twelve_decimals") is not True
        or manifest.get("standalone_and_sentence_transformer_labse_token_ids_identical")
        is not True
        or manifest.get("runtime_gate") != expected_runtime
        or manifest.get("text_prefix") != cfg["text_prefix"]
        or manifest.get("sentence_transformer_prompt")
        != cfg["sentence_transformer_prompt"]
        or manifest.get("embedding_shape") != [49, int(cfg["expected_dimension"])]
        or manifest.get("embedding_dtype") != "float32"
        or not isinstance(manifest.get("embedding_matrix_sha256"), str)
        or len(manifest["embedding_matrix_sha256"]) != 64
        or not isinstance(manifest.get("maximum_unit_norm_error"), (int, float))
        or not 0.0 <= float(manifest["maximum_unit_norm_error"]) <= 1e-3
        or manifest.get("device") != "cuda"
        or manifest.get("deterministic_gpu_runtime")
        != step7_policy["gpu_execution"]["expected_runtime"]
        or manifest.get("default_prompt_name_cleared_before_encoding") is not True
        or not isinstance(manifest.get("loaded_model_state"), dict)
        or manifest["loaded_model_state"].get("loaded_native_max_seq_length")
        != int(cfg["native_max_seq_length"])
        or not isinstance(manifest.get("step7_implementation_files"), dict)
        or list(manifest["step7_implementation_files"])
        != sorted(STEP7_IMPLEMENTATION_ROLES)
        or not isinstance(manifest.get("step28_implementation_files"), dict)
        or list(manifest["step28_implementation_files"])
        != list(STEP28_IMPLEMENTATION_FILES)
        or not tokenizer_ok
        or tokenizers.get("labse", {}).get("token_id_stream_sha256")
        != manifest.get("sentence_transformer_tokenizer_id_stream_sha256")
        or chunk_audit.get("exact_character_reconstruction") is not True
        or chunk_audit.get("unique_text_count") != 32
        or chunk_audit.get("chunk_count") != 49
        or chunk_audit.get("text_requiring_chunking_count") != 6
    ):
        raise common.ModelExperimentContractError("Linux fixture replay boundary drift")
    _revalidate_actual_runtime_state(policy, step7_policy, manifest)
    common.verify_file_pin(
        {**manifest["observed_score_file"], "path": str(score_path)},
        label="Linux fixture observed scores",
    )
    if score_path.read_bytes() != (
        fixture_payload["root"] / "fixture_expected_labse_scores.csv"
    ).read_bytes():
        raise common.ModelExperimentContractError(
            "Linux observed scores differ from the frozen expected bytes"
        )
    actual_files = sorted(path.name for path in output_root.iterdir() if path.is_file())
    if actual_files != sorted([OBSERVED_SCORES, RESULT_MANIFEST]):
        raise common.ModelExperimentContractError(
            "Linux fixture replay contains an unregistered file"
        )
    return manifest


def publish(policy: Mapping[str, Any]) -> dict[str, Any]:
    output_root = common.resolve(policy["outputs"]["compatibility_fixture_linux_replay"])
    if output_root.exists():
        return validate_published(policy, output_root)
    output_root.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".linux_fixture_replay.", dir=output_root.parent))
    try:
        result, score_bytes = run_replay(policy)
        (temporary / OBSERVED_SCORES).write_bytes(score_bytes)
        (temporary / RESULT_MANIFEST).write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        validated = validate_published(policy, temporary)
        temporary.replace(output_root)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return validated


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    policy = common.load_policy()
    result = publish(policy)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
