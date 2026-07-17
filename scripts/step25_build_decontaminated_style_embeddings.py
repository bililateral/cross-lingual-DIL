#!/usr/bin/env python3
"""Encode Step25 component-cross-fitted decontaminated train texts."""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import numpy as np

import step24_build_style_embedding_cache as step24_cache
import step24_common as step24
import step25_common as common


def validate_existing(
    matrix_path: Path,
    metadata_path: Path,
    seller_uids: list[str],
    encoder_key: str,
    text_sha256: str,
    model_fingerprint: dict,
) -> bool:
    if not matrix_path.exists() and not metadata_path.exists():
        return False
    if not matrix_path.is_file() or not metadata_path.is_file():
        raise ValueError(f"Step25 found a partial embedding cache: {matrix_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    expected = {
        "encoder_key": encoder_key,
        "seller_uids": seller_uids,
        "decontaminated_texts_sha256": text_sha256,
        "model_directory_fingerprint": model_fingerprint,
        "matrix_sha256": step24.sha256_file(matrix_path),
    }
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise ValueError(f"Step25 existing embedding cache differs: {encoder_key}:{key}")
    step24.load_normalized_cache(metadata_path, matrix_path)
    return True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", default=str(common.DEFAULT_POLICY))
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--validate-config-only", action="store_true")
    args = parser.parse_args()
    policy_path, policy, step24_policy = common.load_policy(args.policy)
    if args.validate_config_only:
        print(
            json.dumps(
                {
                    "status": "pass",
                    "model_paths": {
                        key: cfg["local_path"]
                        for key, cfg in policy["frozen_style_encoders"].items()
                    },
                    "model_directory_checked": False,
                    "numerical_execution_performed": False,
                },
                indent=2,
            )
        )
        return

    output_root = common.resolve(policy["outputs_root"])
    prepared = {}
    for pool_name in step24_policy["pools"]:
        _, text_path, summary_path = common.template_output_paths(output_root, pool_name)
        records = common.load_jsonl(text_path)
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        seller_uids = [row["seller_uid"] for row in records]
        if len(seller_uids) != len(set(seller_uids)) or not seller_uids:
            raise ValueError(f"Step25 decontaminated seller index is invalid: {pool_name}")
        if summary["valid_test_seller_count"] != 0 or summary[
            "review_label_read_by_template_detector"
        ]:
            raise ValueError(f"Step25 template summary violates isolation: {pool_name}")
        prepared[pool_name] = {
            "seller_uids": seller_uids,
            "texts": [row["decontaminated_text"] for row in records],
            "reliable_flags": [int(row["decontaminated_text_reliable"]) for row in records],
            "text_path": text_path,
            "text_sha256": step24.sha256_file(text_path),
            "summary_sha256": step24.sha256_file(summary_path),
        }

    device = step24_cache.choose_device(args.device)
    encoder_records = {}
    for encoder_key, encoder_cfg in policy["frozen_style_encoders"].items():
        model_path = common.resolve(encoder_cfg["local_path"])
        provenance = step24.validate_model_provenance(model_path, encoder_cfg)
        provenance_path = model_path / "step24_model_provenance.json"
        fingerprint = step24.directory_fingerprint(model_path)
        existing_all = True
        for pool_name, item in prepared.items():
            matrix_path, metadata_path = common.embedding_output_paths(
                output_root, encoder_key, pool_name
            )
            existing_all = validate_existing(
                matrix_path,
                metadata_path,
                item["seller_uids"],
                encoder_key,
                item["text_sha256"],
                fingerprint,
            ) and existing_all
        if not existing_all:
            model = step24_cache.load_sentence_transformer(model_path, device)
            model.max_seq_length = int(encoder_cfg["maximum_sequence_length"])
            model.eval()
            for pool_name, item in prepared.items():
                matrix_path, metadata_path = common.embedding_output_paths(
                    output_root, encoder_key, pool_name
                )
                if matrix_path.exists() or metadata_path.exists():
                    continue
                matrix = model.encode(
                    item["texts"],
                    batch_size=int(encoder_cfg["batch_size"]),
                    show_progress_bar=True,
                    convert_to_numpy=True,
                    normalize_embeddings=bool(encoder_cfg["normalize_embeddings"]),
                )
                matrix = np.asarray(matrix, dtype=np.float32)
                expected_shape = (
                    len(item["seller_uids"]),
                    int(encoder_cfg["expected_dimension"]),
                )
                if matrix.shape != expected_shape or not np.all(np.isfinite(matrix)):
                    raise ValueError(
                        f"Step25 unexpected {encoder_key} matrix: {matrix.shape}; "
                        f"expected={expected_shape}"
                    )
                norms = np.linalg.norm(matrix, axis=1)
                if np.max(np.abs(norms - 1.0)) > 1e-3:
                    raise ValueError(f"Step25 {encoder_key} embeddings are not normalized")
                step24.write_npy_immutable(matrix_path, matrix)
                metadata = {
                    "step": "step25_decontaminated_style_embedding",
                    "version": policy["version"],
                    "encoder_key": encoder_key,
                    "repo_id": encoder_cfg["repo_id"],
                    "revision": encoder_cfg["revision"],
                    "model_local_path": encoder_cfg["local_path"],
                    "model_provenance_sha256": step24.sha256_file(provenance_path),
                    "model_directory_fingerprint": fingerprint,
                    "seller_uids": item["seller_uids"],
                    "reliable_flags": item["reliable_flags"],
                    "shape": list(matrix.shape),
                    "dtype": str(matrix.dtype),
                    "normalized": True,
                    "identifier_redacted": True,
                    "template_decontaminated": True,
                    "component_cross_fitted": True,
                    "locally_finetuned": False,
                    "encoded_split": "train",
                    "valid_test_seller_encoded_count": 0,
                    "decontaminated_texts_sha256": item["text_sha256"],
                    "decontamination_summary_sha256": item["summary_sha256"],
                    "matrix_sha256": step24.sha256_file(matrix_path),
                    "policy_sha256": step24.sha256_file(policy_path),
                    "producer_sha256": step24.sha256_file(Path(__file__).resolve()),
                }
                metadata["metadata_payload_sha256"] = step24.canonical_hash(metadata)
                step24.write_json_immutable(metadata_path, metadata)
            del model
            gc.collect()
            try:
                import torch

                if device == "cuda":
                    torch.cuda.empty_cache()
            except ImportError:
                pass
        encoder_records[encoder_key] = {
            "repo_id": encoder_cfg["repo_id"],
            "revision": encoder_cfg["revision"],
            "model_provenance": provenance,
            "model_provenance_sha256": step24.sha256_file(provenance_path),
            "model_directory_fingerprint": fingerprint,
            "reused_all_identical_caches": existing_all,
        }

    outputs = {}
    for encoder_key in policy["frozen_style_encoders"]:
        for pool_name in step24_policy["pools"]:
            matrix_path, metadata_path = common.embedding_output_paths(
                output_root, encoder_key, pool_name
            )
            outputs[f"{encoder_key}:{pool_name}"] = {
                "matrix": str(matrix_path.relative_to(common.ROOT)).replace("\\", "/"),
                "metadata": str(metadata_path.relative_to(common.ROOT)).replace("\\", "/"),
                "matrix_sha256": step24.sha256_file(matrix_path),
                "metadata_sha256": step24.sha256_file(metadata_path),
            }
    manifest = {
        "step": "step25_build_decontaminated_style_embeddings",
        "version": policy["version"],
        "device": device,
        "encoder_parameters_updated": False,
        "local_synthetic_label_count": 0,
        "valid_test_seller_encoded_count": 0,
        "encoders": encoder_records,
        "outputs": outputs,
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
                "valid_test_seller_encoded_count": 0,
                "manifest": str(manifest_path.relative_to(common.ROOT)).replace("\\", "/"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
