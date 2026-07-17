#!/usr/bin/env python3
"""Encode pair-specific decontaminated sides with the frozen Step24 style models."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import step24_build_style_embedding_cache as step24_cache
import step24_common as step24
import step25_common as step25_v1
import step25_v2_common as common


def validate_existing_cache(
    matrix_path: Path,
    metadata_path: Path,
    expected_keys: list[str],
    expected_corpus_sha256: str,
    expected_model_fingerprint: dict,
    expected_metadata_contract: dict,
) -> bool:
    if not matrix_path.exists() and not metadata_path.exists():
        return False
    if not matrix_path.is_file() or not metadata_path.is_file():
        raise ValueError(f"Step25-v2 found a partial pair-side cache: {matrix_path}")
    index, matrix, metadata = common.load_pair_embedding_cache(metadata_path, matrix_path)
    if list(index) != expected_keys:
        raise ValueError(f"Step25-v2 existing pair-side key order differs: {metadata_path}")
    expected = {
        "pair_local_text_corpus_sha256": expected_corpus_sha256,
        "model_directory_fingerprint": expected_model_fingerprint,
        "identifier_redacted": True,
        "pair_local_decontaminated": True,
        "valid_test_pair_side_count": 0,
        "encoder_parameters_updated": False,
        **expected_metadata_contract,
    }
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise ValueError(f"Step25-v2 existing cache differs: {metadata_path}:{key}")
    if matrix.shape[0] != len(expected_keys):
        raise ValueError(f"Step25-v2 existing cache row count differs: {matrix_path}")
    return True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", default=str(common.DEFAULT_POLICY))
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--validate-config-only", action="store_true")
    args = parser.parse_args()
    policy_path, policy, step24_policy, _step25_v1_policy = common.load_policy(args.policy)
    model_records = {}
    for key, cfg in policy["frozen_style_encoders"].items():
        path = common.resolve(cfg["local_path"])
        model_records[key] = {
            "path": cfg["local_path"],
            "available": path.is_dir() and any(path.iterdir()),
        }
    if args.validate_config_only:
        print(
            json.dumps(
                {
                    "status": "pass",
                    "models": model_records,
                    "pair_side_encoding": True,
                    "valid_test_pair_side_count": 0,
                    "numerical_execution_performed": False,
                },
                indent=2,
            )
        )
        return

    output_root = common.resolve(policy["outputs_root"])
    detector_summary_path = output_root / policy["outputs"]["detector_summary"]
    detector_summary = json.loads(detector_summary_path.read_text(encoding="utf-8"))
    if detector_summary.get("labels_evidence_types_or_scores_read_by_detector") is not False:
        raise ValueError("Step25-v2 detector isolation failed")
    prepared = {}
    for pool_name in step24_policy["pools"]:
        path = common.pair_text_path(policy, pool_name)
        rows = step25_v1.load_jsonl(path)
        keys = []
        texts = []
        for row in rows:
            forbidden = {
                "review_label",
                "evidence_type",
                "prob_positive",
                "model_score",
            } & set(row)
            if forbidden:
                raise ValueError(
                    f"Step25-v2 pair text leaked supervision fields: {pool_name}:{sorted(forbidden)}"
                )
            if row.get("pool") != pool_name or row.get("split_name") != "train":
                raise ValueError(f"Step25-v2 pair text boundary differs: {pool_name}")
            keys.extend(
                [
                    common.pair_side_key(row["pair_uid"], "left"),
                    common.pair_side_key(row["pair_uid"], "right"),
                ]
            )
            texts.extend([row["left_clean_text"], row["right_clean_text"]])
        if len(keys) != len(set(keys)):
            raise ValueError(f"Step25-v2 duplicate pair-side key: {pool_name}")
        corpus_hash = step24.canonical_hash(list(zip(keys, texts, strict=True)))
        prepared[pool_name] = {
            "keys": keys,
            "texts": texts,
            "corpus_hash": corpus_hash,
            "pair_text_path": path,
        }

    device = step24_cache.choose_device(args.device)
    encoder_records = {}
    policy_sha256 = step24.sha256_file(policy_path)
    producer_sha256 = step24.sha256_file(Path(__file__).resolve())
    for encoder_key, cfg in policy["frozen_style_encoders"].items():
        model_path = common.resolve(cfg["local_path"])
        provenance = step24.validate_model_provenance(model_path, cfg)
        provenance_path = model_path / "step24_model_provenance.json"
        fingerprint = step24.directory_fingerprint(model_path)
        all_existing = True
        for pool_name, item in prepared.items():
            matrix_path, metadata_path = common.embedding_paths(policy, encoder_key, pool_name)
            metadata_contract = {
                "encoder_key": encoder_key,
                "repo_id": cfg["repo_id"],
                "revision": cfg["revision"],
                "pool": pool_name,
                "encoded_split": "train",
                "pair_text_sha256": step24.sha256_file(item["pair_text_path"]),
                "policy_sha256": policy_sha256,
                "producer_sha256": producer_sha256,
            }
            all_existing = validate_existing_cache(
                matrix_path,
                metadata_path,
                item["keys"],
                item["corpus_hash"],
                fingerprint,
                metadata_contract,
            ) and all_existing
        if not all_existing:
            model = step24_cache.load_sentence_transformer(model_path, device)
            model.max_seq_length = int(cfg["maximum_sequence_length"])
            model.eval()
            for pool_name, item in prepared.items():
                matrix_path, metadata_path = common.embedding_paths(
                    policy, encoder_key, pool_name
                )
                if matrix_path.exists() or metadata_path.exists():
                    continue
                unique_texts = list(dict.fromkeys(item["texts"]))
                unique_matrix = model.encode(
                    unique_texts,
                    batch_size=int(cfg["batch_size"]),
                    show_progress_bar=True,
                    convert_to_numpy=True,
                    normalize_embeddings=bool(cfg["normalize_embeddings"]),
                )
                unique_matrix = np.asarray(unique_matrix, dtype=np.float32)
                expected_shape = (len(unique_texts), int(cfg["expected_dimension"]))
                if unique_matrix.shape != expected_shape:
                    raise ValueError(
                        f"Step25-v2 unexpected {encoder_key} unique matrix shape: "
                        f"{unique_matrix.shape} != {expected_shape}"
                    )
                text_index = {text: index for index, text in enumerate(unique_texts)}
                matrix = np.asarray(
                    [unique_matrix[text_index[text]] for text in item["texts"]],
                    dtype=np.float32,
                )
                if not np.all(np.isfinite(matrix)):
                    raise ValueError(
                        f"Step25-v2 {encoder_key} returned non-finite embeddings"
                    )
                norms = np.linalg.norm(matrix, axis=1)
                if np.max(np.abs(norms - 1.0)) > 1e-3:
                    raise ValueError(
                        f"Step25-v2 {encoder_key} embeddings are not normalized"
                    )
                step24.write_npy_immutable(matrix_path, matrix)
                metadata = {
                    "step": "step25_v2_pair_local_style_embedding",
                    "version": policy["version"],
                    "encoder_key": encoder_key,
                    "repo_id": cfg["repo_id"],
                    "revision": cfg["revision"],
                    "model_local_path": cfg["local_path"],
                    "model_provenance_sha256": step24.sha256_file(provenance_path),
                    "model_directory_fingerprint": fingerprint,
                    "pool": pool_name,
                    "encoded_split": "train",
                    "pair_side_keys": item["keys"],
                    "shape": list(matrix.shape),
                    "pair_local_text_corpus_sha256": item["corpus_hash"],
                    "pair_text_sha256": step24.sha256_file(item["pair_text_path"]),
                    "identifier_redacted": True,
                    "pair_local_decontaminated": True,
                    "valid_test_pair_side_count": 0,
                    "synthetic_text_count": 0,
                    "encoder_parameters_updated": False,
                    "normalized_embeddings": True,
                    "unique_text_count": len(unique_texts),
                    "policy_sha256": step24.sha256_file(policy_path),
                    "producer_sha256": step24.sha256_file(Path(__file__).resolve()),
                }
                metadata["metadata_sha256"] = step24.canonical_hash(metadata)
                step24.write_json_immutable(metadata_path, metadata)
            del model

        # Build a deterministic manifest from the persisted cache facts. Runtime
        # reuse state and device choice are operational details and must not make
        # an identical replay conflict with the immutable manifest.
        pool_outputs = {}
        for pool_name, item in prepared.items():
            matrix_path, metadata_path = common.embedding_paths(
                policy, encoder_key, pool_name
            )
            if not validate_existing_cache(
                matrix_path,
                metadata_path,
                item["keys"],
                item["corpus_hash"],
                fingerprint,
                {
                    "encoder_key": encoder_key,
                    "repo_id": cfg["repo_id"],
                    "revision": cfg["revision"],
                    "pool": pool_name,
                    "encoded_split": "train",
                    "pair_text_sha256": step24.sha256_file(item["pair_text_path"]),
                    "policy_sha256": policy_sha256,
                    "producer_sha256": producer_sha256,
                },
            ):
                raise ValueError(
                    f"Step25-v2 failed to materialize pair-side cache: {encoder_key}:{pool_name}"
                )
            _index, matrix, metadata = common.load_pair_embedding_cache(
                metadata_path, matrix_path
            )
            pool_outputs[pool_name] = {
                "matrix": str(matrix_path.relative_to(common.ROOT)).replace("\\", "/"),
                "metadata": str(metadata_path.relative_to(common.ROOT)).replace("\\", "/"),
                "shape": list(matrix.shape),
                "unique_text_count": int(metadata["unique_text_count"]),
                "matrix_sha256": step24.sha256_file(matrix_path),
                "metadata_sha256": step24.sha256_file(metadata_path),
                "cache_contract_verified": True,
            }
        encoder_records[encoder_key] = {
            "repo_id": cfg["repo_id"],
            "revision": cfg["revision"],
            "provenance": provenance,
            "provenance_sha256": step24.sha256_file(provenance_path),
            "model_directory_fingerprint": fingerprint,
            "pools": pool_outputs,
        }

    manifest = {
        "step": "step25_v2_encode_pair_local_style",
        "version": policy["version"],
        "status": "pass",
        "boundary": policy["boundary"]["name"],
        "identifier_redacted": True,
        "pair_local_decontaminated": True,
        "valid_test_pair_side_count": 0,
        "synthetic_text_count": 0,
        "encoder_parameters_updated": False,
        "runtime_device_recorded_in_manifest": False,
        "encoders": encoder_records,
        "detector_summary_sha256": step24.sha256_file(detector_summary_path),
        "policy_sha256": step24.sha256_file(policy_path),
        "producer_sha256": step24.sha256_file(Path(__file__).resolve()),
    }
    manifest["manifest_sha256"] = step24.canonical_hash(manifest)
    manifest_path = output_root / policy["outputs"]["embedding_manifest"]
    step24.write_json_immutable(manifest_path, manifest)
    print(
        json.dumps(
            {
                "status": "pass",
                "device": device,
                "encoders": list(encoder_records),
                "manifest": str(manifest_path.relative_to(common.ROOT)).replace("\\", "/"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
