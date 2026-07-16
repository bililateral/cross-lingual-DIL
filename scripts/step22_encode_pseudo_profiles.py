#!/usr/bin/env python3
"""Encode Step22 pseudo profiles with the frozen identifier-redacted E5 encoder."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

import step7_build_semantic_pair_features as semantic
import step15_build_v7_clean_embedding_cache as redaction


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_POLICY = ROOT / "schema" / "step22_same_seller_split_policy.json"


def resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", default=str(DEFAULT_POLICY))
    parser.add_argument("--device", default=None)
    parser.add_argument("--force-recompute", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    policy_path = resolve(args.policy)
    policy = json.loads(policy_path.read_text(encoding="utf-8"))
    output_root = resolve(policy["outputs_root"])
    profile_path = output_root / policy["outputs"]["profiles"]
    generation_summary_path = output_root / policy["outputs"]["summary"]
    if not profile_path.is_file() or not generation_summary_path.is_file():
        raise FileNotFoundError("Run step22_build_same_seller_split_augmentation.py first")
    matrix_path = output_root / policy["outputs"]["embedding_matrix"]
    metadata_path = output_root / policy["outputs"]["embedding_metadata"]
    semantic_policy_path = resolve(policy["inputs"]["semantic_model_policy"])
    semantic_policy = json.loads(semantic_policy_path.read_text(encoding="utf-8"))
    model_key = "multilingual_e5_large"
    model_cfg = dict(semantic_policy["embedding_models"][model_key])
    model_dir = semantic.resolve_local_model_dir(model_key, model_cfg)
    model_fingerprint = redaction.directory_fingerprint(model_dir)
    producer_path = Path(__file__).resolve()
    profiles = semantic.load_jsonl(profile_path)
    seller_uids = [row["seller_uid"] for row in profiles]
    if len(set(seller_uids)) != len(seller_uids):
        raise ValueError("Step22 pseudo seller UIDs are not unique")
    expected = {
        "profile_sha256": sha256(profile_path),
        "generation_summary_sha256": sha256(generation_summary_path),
        "producer_sha256": sha256(producer_path),
        "semantic_producer_sha256": sha256(Path(semantic.__file__).resolve()),
        "redaction_producer_sha256": sha256(Path(redaction.__file__).resolve()),
        "semantic_policy_sha256": sha256(semantic_policy_path),
        "model_fingerprint": model_fingerprint,
        "seller_uids": seller_uids,
    }
    if matrix_path.is_file() and metadata_path.is_file() and not args.force_recompute:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        for key, value in expected.items():
            if metadata.get(key) != value:
                raise ValueError(f"Existing Step22 embedding cache mismatch: {key}")
        matrix = np.load(matrix_path, mmap_mode="r")
        if list(matrix.shape) != list(metadata.get("shape", [])):
            raise ValueError("Existing Step22 embedding cache shape mismatch")
        print(json.dumps({"status": "verified_existing", "shape": list(matrix.shape)}, indent=2))
        return
    torch_module, tokenizer_cls, model_cls, _ = semantic.require_torch_and_transformers()
    device = semantic.choose_device(torch_module, semantic_policy["device_preference"], args.device)
    texts = [str(row.get("profile_text", "")) for row in profiles]
    if any(not text.strip() for text in texts):
        raise ValueError("Step22 contains an empty pseudo profile")
    matrix = semantic.encode_texts(
        model_key, model_cfg, texts, device, torch_module, tokenizer_cls, model_cls
    )
    np.save(matrix_path, matrix)
    metadata = {
        "step": "step22_pseudo_profile_encoding",
        "policy_version": policy["version"],
        "model_key": model_key,
        "model_repo_id": model_cfg["repo_id"],
        "model_local_path": model_cfg["local_path"],
        **expected,
        "shape": list(matrix.shape),
        "device": str(device),
        "identifier_redacted": True,
        "synthetic_train_only": True,
        "benchmark_eligible": False,
    }
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"status": "encoded", "device": str(device), "shape": list(matrix.shape)}, indent=2))


if __name__ == "__main__":
    main()
