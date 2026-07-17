#!/usr/bin/env python3
"""Encode canonical-train sellers with frozen multilingual authorship models."""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import numpy as np

import step24_common as common


def output_paths(output_root: Path, encoder_key: str, pool_name: str) -> tuple[Path, Path]:
    stem = f"{encoder_key}.{pool_name}"
    return output_root / "embeddings" / f"{stem}.npy", output_root / "embeddings" / f"{stem}.json"


def validate_existing_cache(
    matrix_path: Path,
    metadata_path: Path,
    seller_uids: list[str],
    encoder_key: str,
    model_fingerprint: dict,
    clean_text_sha256: str,
) -> bool:
    if not matrix_path.exists() and not metadata_path.exists():
        return False
    if not matrix_path.is_file() or not metadata_path.is_file():
        raise ValueError(f"Step24 found a partial style cache: {matrix_path} / {metadata_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    required = {
        "encoder_key": encoder_key,
        "seller_uids": seller_uids,
        "model_directory_fingerprint": model_fingerprint,
        "clean_text_corpus_sha256": clean_text_sha256,
        "matrix_sha256": common.sha256_file(matrix_path),
    }
    for key, expected in required.items():
        if metadata.get(key) != expected:
            raise ValueError(
                f"Existing Step24 cache is not an identical replay for {encoder_key}; "
                f"field={key}"
            )
    common.load_normalized_cache(metadata_path, matrix_path)
    return True


def load_sentence_transformer(model_path: Path, device: str):
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise RuntimeError(
            "Step24 requires sentence-transformers on Linux; install it in the active environment"
        ) from exc
    try:
        return SentenceTransformer(str(model_path), device=device, local_files_only=True)
    except TypeError as exc:
        raise RuntimeError(
            "The installed sentence-transformers version lacks local_files_only support; "
            "upgrade it rather than allowing an implicit network download"
        ) from exc


def choose_device(requested: str) -> str:
    if requested != "auto":
        return requested
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("Step24 embedding requires PyTorch on Linux") from exc
    return "cuda" if torch.cuda.is_available() else "cpu"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", default=str(common.DEFAULT_POLICY))
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--validate-config-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    policy_path = common.resolve(args.policy)
    policy = json.loads(policy_path.read_text(encoding="utf-8"))
    common.validate_policy(policy)
    output_root = common.resolve(policy["outputs_root"])
    model_records = {}
    for encoder_key, encoder_cfg in policy["frozen_style_encoders"].items():
        model_path = common.resolve(encoder_cfg["local_path"])
        model_records[encoder_key] = {
            "repo_id": encoder_cfg["repo_id"],
            "local_path": encoder_cfg["local_path"],
            "available": model_path.is_dir() and any(model_path.iterdir()),
        }
    if args.validate_config_only:
        print(
            json.dumps(
                {
                    "status": "pass",
                    "policy": str(policy_path.relative_to(common.ROOT)).replace("\\", "/"),
                    "models": model_records,
                    "numerical_execution_performed": False,
                },
                indent=2,
            )
        )
        return

    rows_by_pool = common.load_canonical_train_rows(policy)
    text_cfg = policy["clean_text_contract"]
    prepared = {}
    clean_manifest_records = {}
    for pool_name, pool_cfg in policy["pools"].items():
        e5_metadata_path = common.resolve(pool_cfg["identifier_redacted_e5_metadata"])
        e5_matrix_path = common.resolve(pool_cfg["identifier_redacted_e5_matrix"])
        _, _, e5_metadata = common.load_normalized_cache(e5_metadata_path, e5_matrix_path)
        if e5_metadata.get("identifier_redacted") is not True:
            raise ValueError(f"Step24 E5 input is not identifier-redacted: {pool_name}")
        all_seller_uids = list(e5_metadata["seller_uids"])
        all_clean_texts, all_diagnostics = common.replay_v7_clean_texts(
            pool_cfg, text_cfg, all_seller_uids
        )
        expected_corpus_hash = e5_metadata.get("redaction_diagnostics", {}).get(
            "clean_text_corpus_sha256"
        )
        if expected_corpus_hash != all_diagnostics["clean_text_corpus_sha256"]:
            raise ValueError(
                f"Step24 clean-text replay differs from the frozen v7 E5 corpus: {pool_name}"
            )
        all_text_index = {uid: index for index, uid in enumerate(all_seller_uids)}
        selected_sellers = common.train_sellers(rows_by_pool[pool_name])
        missing = sorted(set(selected_sellers) - set(all_text_index))
        if missing:
            raise ValueError(f"Step24 train seller is missing from v7 E5 cache: {missing[0]}")
        selected_texts = [all_clean_texts[all_text_index[uid]] for uid in selected_sellers]
        selected_hash = common.canonical_hash(
            list(zip(selected_sellers, selected_texts, strict=True))
        )
        prepared[pool_name] = {
            "seller_uids": selected_sellers,
            "clean_texts": selected_texts,
            "clean_text_corpus_sha256": selected_hash,
        }
        clean_manifest_records[pool_name] = {
            "canonical_train_pair_count": len(rows_by_pool[pool_name]),
            "canonical_train_seller_count": len(selected_sellers),
            "encoded_split": "train",
            "valid_test_seller_encoded_count": 0,
            "selected_clean_text_corpus_sha256": selected_hash,
            "full_v7_clean_text_corpus_sha256_verified": expected_corpus_hash,
            "v7_e5_metadata_sha256": common.sha256_file(e5_metadata_path),
            "v7_e5_matrix_sha256": common.sha256_file(e5_matrix_path),
            "profile_sha256": common.sha256_file(common.resolve(pool_cfg["seller_profiles"])),
            "identity_signal_sha256": common.sha256_file(
                common.resolve(pool_cfg["item_identity_signals"])
            ),
        }
    clean_manifest = {
        "step": "step24_clean_text_replay",
        "version": policy["version"],
        "identifier_redacted": True,
        "synthetic_text_count": 0,
        "valid_test_scores_or_labels_read": False,
        "records": clean_manifest_records,
        "policy_sha256": common.sha256_file(policy_path),
        "producer_sha256": common.sha256_file(Path(__file__).resolve()),
    }
    clean_manifest["manifest_sha256"] = common.canonical_hash(clean_manifest)
    common.write_json_immutable(
        output_root / policy["outputs"]["clean_text_manifest"], clean_manifest
    )

    device = choose_device(args.device)
    embedding_records = {}
    for encoder_key, encoder_cfg in policy["frozen_style_encoders"].items():
        model_path = common.resolve(encoder_cfg["local_path"])
        provenance = common.validate_model_provenance(model_path, encoder_cfg)
        provenance_path = model_path / "step24_model_provenance.json"
        model_fingerprint = common.directory_fingerprint(model_path)
        existing_all = True
        for pool_name, item in prepared.items():
            matrix_path, metadata_path = output_paths(output_root, encoder_key, pool_name)
            existing_all = validate_existing_cache(
                matrix_path,
                metadata_path,
                item["seller_uids"],
                encoder_key,
                model_fingerprint,
                item["clean_text_corpus_sha256"],
            ) and existing_all
        if existing_all:
            embedding_records[encoder_key] = {
                "repo_id": encoder_cfg["repo_id"],
                "revision": encoder_cfg["revision"],
                "provenance": provenance,
                "provenance_sha256": common.sha256_file(provenance_path),
                "model_directory_fingerprint": model_fingerprint,
                "reused_identical_cache": True,
            }
            continue

        model = load_sentence_transformer(model_path, device)
        model.max_seq_length = int(encoder_cfg["maximum_sequence_length"])
        model.eval()
        for pool_name, item in prepared.items():
            matrix_path, metadata_path = output_paths(output_root, encoder_key, pool_name)
            if matrix_path.exists() or metadata_path.exists():
                continue
            matrix = model.encode(
                item["clean_texts"],
                batch_size=int(encoder_cfg["batch_size"]),
                show_progress_bar=True,
                convert_to_numpy=True,
                normalize_embeddings=bool(encoder_cfg["normalize_embeddings"]),
            )
            matrix = np.asarray(matrix, dtype=np.float32)
            if matrix.shape != (len(item["seller_uids"]), int(encoder_cfg["expected_dimension"])):
                raise ValueError(
                    f"Step24 unexpected {encoder_key} matrix shape: {matrix.shape}"
                )
            if not np.all(np.isfinite(matrix)):
                raise ValueError(f"Step24 {encoder_key} returned non-finite embeddings")
            norms = np.linalg.norm(matrix, axis=1)
            if np.max(np.abs(norms - 1.0)) > 1e-3:
                raise ValueError(f"Step24 {encoder_key} embeddings are not unit-normalized")
            common.write_npy_immutable(matrix_path, matrix)
            metadata = {
                "step": "step24_frozen_style_embedding",
                "version": policy["version"],
                "encoder_key": encoder_key,
                "repo_id": encoder_cfg["repo_id"],
                "revision": encoder_cfg["revision"],
                "model_provenance_sha256": common.sha256_file(provenance_path),
                "model_local_path": encoder_cfg["local_path"],
                "model_directory_fingerprint": model_fingerprint,
                "seller_uids": item["seller_uids"],
                "shape": list(matrix.shape),
                "dtype": str(matrix.dtype),
                "normalized": True,
                "identifier_redacted": True,
                "locally_finetuned": False,
                "encoded_split": "train",
                "valid_test_seller_encoded_count": 0,
                "clean_text_corpus_sha256": item["clean_text_corpus_sha256"],
                "matrix_sha256": common.sha256_file(matrix_path),
                "policy_sha256": common.sha256_file(policy_path),
                "producer_sha256": common.sha256_file(Path(__file__).resolve()),
            }
            metadata["metadata_payload_sha256"] = common.canonical_hash(metadata)
            common.write_json_immutable(metadata_path, metadata)
        del model
        gc.collect()
        try:
            import torch

            if device == "cuda":
                torch.cuda.empty_cache()
        except ImportError:
            pass
        embedding_records[encoder_key] = {
            "repo_id": encoder_cfg["repo_id"],
            "revision": encoder_cfg["revision"],
            "provenance": provenance,
            "provenance_sha256": common.sha256_file(provenance_path),
            "model_directory_fingerprint": model_fingerprint,
            "reused_identical_cache": False,
        }

    manifest = {
        "step": "step24_build_style_embedding_cache",
        "version": policy["version"],
        "device": device,
        "encoder_parameters_updated": False,
        "local_synthetic_label_count": 0,
        "valid_test_seller_encoded_count": 0,
        "encoders": embedding_records,
        "outputs": {},
        "policy_sha256": common.sha256_file(policy_path),
        "producer_sha256": common.sha256_file(Path(__file__).resolve()),
    }
    for encoder_key in policy["frozen_style_encoders"]:
        for pool_name in policy["pools"]:
            matrix_path, metadata_path = output_paths(output_root, encoder_key, pool_name)
            manifest["outputs"][f"{encoder_key}:{pool_name}"] = {
                "matrix": str(matrix_path.relative_to(common.ROOT)).replace("\\", "/"),
                "metadata": str(metadata_path.relative_to(common.ROOT)).replace("\\", "/"),
                "matrix_sha256": common.sha256_file(matrix_path),
                "metadata_sha256": common.sha256_file(metadata_path),
            }
    manifest["manifest_sha256"] = common.canonical_hash(manifest)
    common.write_json_immutable(
        output_root / policy["outputs"]["embedding_manifest"], manifest
    )
    print(
        json.dumps(
            {
                "status": "pass",
                "device": device,
                "encoded_split": "train",
                "valid_test_seller_encoded_count": 0,
                "manifest": str(
                    (output_root / policy["outputs"]["embedding_manifest"]).relative_to(
                        common.ROOT
                    )
                ).replace("\\", "/"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
