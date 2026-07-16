#!/usr/bin/env python3
"""Encode Step21 identifier-redacted synthetic profiles with frozen Multilingual-E5."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

import step7_build_semantic_pair_features as semantic
import step15_build_v7_clean_embedding_cache as redaction


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_POLICY = ROOT / "schema" / "step21_synthetic_train_only_policy.json"


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
    parser.add_argument("--track", action="append", dest="tracks")
    parser.add_argument("--device", default=None)
    parser.add_argument("--force-recompute", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    policy_path = resolve(args.policy)
    policy = json.loads(policy_path.read_text(encoding="utf-8"))
    output_root = resolve(policy["outputs_root"])
    generation_summary_path = output_root / policy["outputs"]["summary"]
    if not generation_summary_path.exists():
        raise FileNotFoundError(
            f"Run step21_build_synthetic_zh_train.py first: {generation_summary_path}"
        )
    generation_summary = json.loads(generation_summary_path.read_text(encoding="utf-8"))
    semantic_policy_path = resolve(policy["inputs"]["semantic_model_policy"])
    semantic_policy = json.loads(semantic_policy_path.read_text(encoding="utf-8"))
    model_key = "multilingual_e5_large"
    model_cfg = dict(semantic_policy["embedding_models"][model_key])
    producer_path = Path(__file__).resolve()
    model_dir = semantic.resolve_local_model_dir(model_key, model_cfg)
    model_fingerprint = redaction.directory_fingerprint(model_dir)
    tracks = args.tracks or list(generation_summary["tracks"])
    unknown = sorted(set(tracks) - set(generation_summary["tracks"]))
    if unknown:
        raise ValueError(f"Unknown generated Step21 tracks: {unknown}")

    torch_module, tokenizer_cls, model_cls, _ = semantic.require_torch_and_transformers()
    device = semantic.choose_device(
        torch_module, semantic_policy["device_preference"], args.device
    )
    encoded_tracks = {}
    for track_name in tracks:
        track_root = output_root / policy["outputs"]["tracks_directory"] / track_name
        profile_path = track_root / "synthetic_seller_profiles.jsonl"
        matrix_path = track_root / "synthetic_e5_identifier_redacted.npy"
        metadata_path = track_root / "synthetic_e5_identifier_redacted.json"
        profiles = semantic.load_jsonl(profile_path)
        seller_uids = [row["seller_uid"] for row in profiles]
        if len(set(seller_uids)) != len(seller_uids):
            raise ValueError(f"Step21 synthetic seller UIDs are not unique: {track_name}")
        if (
            matrix_path.exists()
            and metadata_path.exists()
            and not args.force_recompute
        ):
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            matrix = np.load(matrix_path, mmap_mode="r")
            if metadata.get("profile_sha256") != sha256(profile_path):
                raise ValueError(
                    f"Existing Step21 embedding cache has different profile content: {track_name}"
                )
            if metadata.get("seller_uids") != seller_uids:
                raise ValueError(f"Existing Step21 embedding cache UID mismatch: {track_name}")
            if metadata.get("producer_sha256") != sha256(producer_path):
                raise ValueError(f"Existing Step21 embedding cache producer mismatch: {track_name}")
            if metadata.get("semantic_policy_sha256") != sha256(semantic_policy_path):
                raise ValueError(f"Existing Step21 embedding cache policy mismatch: {track_name}")
            if metadata.get("model_fingerprint") != model_fingerprint:
                raise ValueError(f"Existing Step21 embedding cache model mismatch: {track_name}")
            if list(matrix.shape) != list(metadata.get("shape", [])):
                raise ValueError(f"Existing Step21 embedding cache shape mismatch: {track_name}")
        else:
            texts = [str(row.get("profile_text", "")) for row in profiles]
            if any(not text.strip() for text in texts):
                raise ValueError(f"Step21 track contains an empty synthetic profile: {track_name}")
            matrix = semantic.encode_texts(
                model_key,
                model_cfg,
                texts,
                device,
                torch_module,
                tokenizer_cls,
                model_cls,
            )
            np.save(matrix_path, matrix)
            metadata = {
                "step": "step21_synthetic_profile_encoding",
                "track": track_name,
                "model_key": model_key,
                "model_repo_id": model_cfg["repo_id"],
                "model_local_path": model_cfg["local_path"],
                "producer_sha256": sha256(producer_path),
                "semantic_policy_sha256": sha256(semantic_policy_path),
                "model_fingerprint": model_fingerprint,
                "identifier_redacted": True,
                "synthetic_train_only": True,
                "profile_path": str(profile_path.relative_to(ROOT)).replace("\\", "/"),
                "profile_sha256": sha256(profile_path),
                "seller_uids": seller_uids,
                "shape": list(matrix.shape),
                "device": str(device),
            }
            metadata_path.write_text(
                json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
        encoded_tracks[track_name] = {
            "seller_count": len(seller_uids),
            "shape": list(matrix.shape),
            "matrix": str(matrix_path.relative_to(ROOT)).replace("\\", "/"),
            "metadata": str(metadata_path.relative_to(ROOT)).replace("\\", "/"),
        }
    print(
        json.dumps(
            {
                "status": "encoded_train_only_synthetic_profiles",
                "device": str(device),
                "tracks": encoded_tracks,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
