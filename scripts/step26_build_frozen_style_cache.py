#!/usr/bin/env python3
"""Blindly encode the frozen Step26 valid/test sellers with frozen Step24 models."""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import numpy as np

import step24_build_style_embedding_cache as step24_cache
import step24_common as step24
import step26_common as common


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", default=str(common.DEFAULT_POLICY))
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--validate-config-only", action="store_true")
    return parser.parse_args()


def cache_paths(output_root: Path, encoder_key: str) -> tuple[Path, Path]:
    base = output_root / "embeddings" / f"{encoder_key}.evaluation"
    return Path(f"{base}.npy"), Path(f"{base}.json")


def validate_existing(
    matrix_path: Path,
    metadata_path: Path,
    seller_uids: list[str],
    clean_hash: str,
    model_fingerprint: dict,
) -> bool:
    if not matrix_path.exists() and not metadata_path.exists():
        return False
    if not matrix_path.is_file() or not metadata_path.is_file():
        raise ValueError(f"Step26 found a partial cache: {matrix_path} / {metadata_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    expected = {
        "seller_uids": seller_uids,
        "clean_text_corpus_sha256": clean_hash,
        "model_directory_fingerprint": model_fingerprint,
        "matrix_sha256": common.sha256(matrix_path),
        "labels_or_evidence_read_before_encoding": False,
        "encoder_parameters_updated": False,
    }
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise ValueError(f"Existing Step26 cache is not an identical replay: {key}")
    step24.load_normalized_cache(metadata_path, matrix_path)
    return True


def main() -> None:
    args = parse_args()
    policy_path, policy, step24_policy = common.load_policy(args.policy)
    source_validation = common.validate_frozen_sources(policy)
    model_status = {}
    for key in policy["frozen_models"]["encoder_keys"]:
        cfg = step24_policy["frozen_style_encoders"][key]
        model_path = common.resolve(cfg["local_path"])
        model_status[key] = {
            "path": cfg["local_path"],
            "available": model_path.is_dir() and any(model_path.iterdir()),
        }
    if args.validate_config_only:
        print(
            json.dumps(
                {
                    "status": "pass",
                    "policy": str(policy_path.relative_to(common.ROOT)).replace("\\", "/"),
                    "models": model_status,
                    "numerical_execution_performed": False,
                },
                indent=2,
            )
        )
        return

    blind_allowlists = common.load_blind_pair_allowlists(policy)
    seller_uids = sorted(
        {
            seller
            for pair_uids in blind_allowlists.values()
            for pair_uid in pair_uids
            for seller in common.pair_uid_sellers(pair_uid)
        }
    )
    data_cfg = policy["evaluation_data"]
    e5_metadata_path = common.resolve(data_cfg["identifier_redacted_e5_metadata"])
    e5_matrix_path = common.resolve(data_cfg["identifier_redacted_e5_matrix"])
    e5_index, _, e5_metadata = step24.load_normalized_cache(e5_metadata_path, e5_matrix_path)
    if e5_metadata.get("identifier_redacted") is not True:
        raise ValueError("Step26 E5 cache is not identifier-redacted")
    missing = sorted(set(seller_uids) - set(e5_index))
    if missing:
        raise ValueError(f"Step26 evaluation seller is absent from the frozen E5 cache: {missing[0]}")

    pool_cfg = {
        "seller_profiles": data_cfg["seller_profiles"],
        "item_identity_signals": data_cfg["item_identity_signals"],
    }
    all_sellers = list(e5_metadata["seller_uids"])
    all_clean_texts, replay = step24.replay_v7_clean_texts(
        pool_cfg, step24_policy["clean_text_contract"], all_sellers
    )
    expected_full_hash = e5_metadata.get("redaction_diagnostics", {}).get(
        "clean_text_corpus_sha256"
    )
    if replay["clean_text_corpus_sha256"] != expected_full_hash:
        raise ValueError("Step26 clean-text replay differs from the frozen v7 E5 corpus")
    text_index = {uid: index for index, uid in enumerate(all_sellers)}
    clean_texts = [all_clean_texts[text_index[uid]] for uid in seller_uids]
    clean_hash = common.canonical_hash(list(zip(seller_uids, clean_texts, strict=True)))
    output_root = common.resolve(policy["outputs_root"])
    clean_manifest = {
        "step": "step26_blind_clean_text_replay",
        "version": policy["version"],
        "pair_count": sum(len(value) for value in blind_allowlists.values()),
        "seller_count": len(seller_uids),
        "seller_uids": seller_uids,
        "pair_allowlist_sha256": common.canonical_hash(blind_allowlists),
        "clean_text_corpus_sha256": clean_hash,
        "full_v7_clean_text_corpus_sha256_verified": expected_full_hash,
        "identifier_redacted": True,
        "labels_or_evidence_read_before_encoding": False,
        "valid_test_scores_used_for_model_selection": False,
        "step24_source_validation_hashes": source_validation["hashes"],
        "policy_sha256": common.sha256(policy_path),
        "producer_sha256": common.sha256(Path(__file__).resolve()),
    }
    clean_manifest["manifest_sha256"] = common.canonical_hash(clean_manifest)
    common.write_json_immutable(
        output_root / policy["outputs"]["clean_text_manifest"], clean_manifest
    )

    device = step24_cache.choose_device(args.device)
    records = {}
    for encoder_key in policy["frozen_models"]["encoder_keys"]:
        cfg = step24_policy["frozen_style_encoders"][encoder_key]
        model_path = common.resolve(cfg["local_path"])
        provenance = step24.validate_model_provenance(model_path, cfg)
        fingerprint = step24.directory_fingerprint(model_path)
        matrix_path, metadata_path = cache_paths(output_root, encoder_key)
        reused = validate_existing(
            matrix_path, metadata_path, seller_uids, clean_hash, fingerprint
        )
        if not reused:
            model = step24_cache.load_sentence_transformer(model_path, device)
            model.max_seq_length = int(cfg["maximum_sequence_length"])
            model.eval()
            matrix = np.asarray(
                model.encode(
                    clean_texts,
                    batch_size=int(cfg["batch_size"]),
                    show_progress_bar=True,
                    convert_to_numpy=True,
                    normalize_embeddings=True,
                ),
                dtype=np.float32,
            )
            expected_shape = (len(seller_uids), int(cfg["expected_dimension"]))
            if matrix.shape != expected_shape or not np.all(np.isfinite(matrix)):
                raise ValueError(
                    f"Step26 unexpected/non-finite {encoder_key} matrix: {matrix.shape}"
                )
            if np.max(np.abs(np.linalg.norm(matrix, axis=1) - 1.0)) > 1e-3:
                raise ValueError(f"Step26 {encoder_key} matrix is not unit-normalized")
            step24.write_npy_immutable(matrix_path, matrix)
            metadata = {
                "step": "step26_frozen_style_embedding",
                "version": policy["version"],
                "encoder_key": encoder_key,
                "repo_id": cfg["repo_id"],
                "revision": cfg["revision"],
                "seller_uids": seller_uids,
                "shape": list(matrix.shape),
                "dtype": str(matrix.dtype),
                "normalized": True,
                "identifier_redacted": True,
                "locally_finetuned": False,
                "encoder_parameters_updated": False,
                "labels_or_evidence_read_before_encoding": False,
                "clean_text_corpus_sha256": clean_hash,
                "model_directory_fingerprint": fingerprint,
                "model_provenance": provenance,
                "matrix_sha256": common.sha256(matrix_path),
                "policy_sha256": common.sha256(policy_path),
                "producer_sha256": common.sha256(Path(__file__).resolve()),
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
        records[encoder_key] = {
            "matrix": str(matrix_path.relative_to(common.ROOT)).replace("\\", "/"),
            "metadata": str(metadata_path.relative_to(common.ROOT)).replace("\\", "/"),
            "matrix_sha256": common.sha256(matrix_path),
            "metadata_sha256": common.sha256(metadata_path),
            "reused_identical_cache": reused,
        }
    embedding_manifest = {
        "step": "step26_style_embedding_manifest",
        "version": policy["version"],
        "device": device,
        "seller_count": len(seller_uids),
        "encoder_parameters_updated": False,
        "labels_or_evidence_read_before_encoding": False,
        "records": records,
        "policy_sha256": common.sha256(policy_path),
        "producer_sha256": common.sha256(Path(__file__).resolve()),
    }
    embedding_manifest["manifest_sha256"] = common.canonical_hash(embedding_manifest)
    common.write_json_immutable(
        output_root / policy["outputs"]["embedding_manifest"], embedding_manifest
    )
    print(
        json.dumps(
            {
                "status": "pass",
                "seller_count": len(seller_uids),
                "device": device,
                "output_root": str(output_root.relative_to(common.ROOT)).replace("\\", "/"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
