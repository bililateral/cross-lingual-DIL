#!/usr/bin/env python3
"""Encode Step23 selected real train items with frozen Multilingual-E5."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

import step7_build_semantic_pair_features as semantic
import step15_build_v7_clean_embedding_cache as redaction


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_POLICY = ROOT / "schema" / "step23_item_multi_instance_policy.json"


def resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", default=str(DEFAULT_POLICY))
    parser.add_argument("--device", default=None)
    parser.add_argument("--force-recompute", action="store_true")
    args = parser.parse_args()

    policy_path = resolve(args.policy)
    policy = json.loads(policy_path.read_text(encoding="utf-8"))
    output_root = resolve(policy["outputs_root"])
    item_path = output_root / policy["outputs"]["selected_items"]
    selection_summary_path = output_root / policy["outputs"]["item_selection_summary"]
    selection_manifest_path = output_root / policy["outputs"]["item_selection_manifest"]
    matrix_path = output_root / policy["outputs"]["item_embedding_matrix"]
    metadata_path = output_root / policy["outputs"]["item_embedding_metadata"]
    for path in (item_path, selection_summary_path, selection_manifest_path):
        if not path.is_file():
            raise FileNotFoundError(f"Run step23_build_item_text_cache.py first: {path}")

    semantic_policy_path = resolve(policy["inputs"]["semantic_model_policy"])
    semantic_policy = json.loads(semantic_policy_path.read_text(encoding="utf-8"))
    model_key = policy["embedding"]["model_key"]
    model_cfg = dict(semantic_policy["embedding_models"][model_key])
    model_directory = semantic.resolve_local_model_dir(model_key, model_cfg)
    model_fingerprint = redaction.directory_fingerprint(model_directory)
    items = semantic.load_jsonl(item_path)
    item_uids = [row["item_uid"] for row in items]
    if not items or len(item_uids) != len(set(item_uids)):
        raise ValueError("Step23 selected item list is empty or contains duplicate item UIDs")
    if any(row.get("split_name") != "train" or row.get("identifier_redacted") is not True for row in items):
        raise ValueError("Step23 encoder received a non-train or non-redacted item")
    expected = {
        "policy_sha256": sha256_file(policy_path),
        "selected_items_sha256": sha256_file(item_path),
        "selection_summary_sha256": sha256_file(selection_summary_path),
        "selection_manifest_sha256": sha256_file(selection_manifest_path),
        "producer_sha256": sha256_file(Path(__file__)),
        "semantic_producer_sha256": sha256_file(Path(semantic.__file__).resolve()),
        "redaction_producer_sha256": sha256_file(Path(redaction.__file__).resolve()),
        "semantic_policy_sha256": sha256_file(semantic_policy_path),
        "model_fingerprint": model_fingerprint,
        "item_uids": item_uids,
    }
    if matrix_path.exists() != metadata_path.exists() and not args.force_recompute:
        raise FileExistsError("Incomplete Step23 item embedding cache exists")
    if matrix_path.is_file() and metadata_path.is_file() and not args.force_recompute:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        for key, value in expected.items():
            if metadata.get(key) != value:
                raise ValueError(f"Existing Step23 item embedding cache mismatch: {key}")
        matrix = np.load(matrix_path, mmap_mode="r")
        if list(matrix.shape) != list(metadata.get("shape", [])):
            raise ValueError("Existing Step23 item embedding shape mismatch")
        print(json.dumps({"status": "verified_existing", "shape": list(matrix.shape)}, indent=2))
        return

    texts = [str(row.get("clean_text", "")) for row in items]
    if any(not text.strip() for text in texts):
        raise ValueError("Step23 selected item contains empty clean text")
    torch_module, tokenizer_cls, model_cls, _ = semantic.require_torch_and_transformers()
    device = semantic.choose_device(torch_module, semantic_policy["device_preference"], args.device)
    matrix = semantic.encode_texts(
        model_key, model_cfg, texts, device, torch_module, tokenizer_cls, model_cls
    )
    np.save(matrix_path, matrix)
    metadata = {
        "step": "step23_selected_real_item_encoding",
        "policy_version": policy["version"],
        "model_key": model_key,
        "model_repo_id": model_cfg["repo_id"],
        "model_local_path": model_cfg["local_path"],
        **expected,
        "shape": list(matrix.shape),
        "device": str(device),
        "identifier_redacted": True,
        "selected_real_train_items_only": True,
        "valid_test_items_encoded": False,
        "synthetic_item_count": 0,
    }
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "encoded", "device": str(device), "shape": list(matrix.shape)}, indent=2))


if __name__ == "__main__":
    main()
