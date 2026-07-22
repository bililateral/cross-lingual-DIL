#!/usr/bin/env python3
"""Encode the Step7-v3 clean corpus with five model-native encoders and one symmetric reranker."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
import platform
from pathlib import Path

import numpy as np

import step7_v3_common as common


ENCODER_SCRIPT = Path(__file__).resolve()


def require_gpu_stack():
    try:
        import torch  # type: ignore
        import transformers  # type: ignore
        from sentence_transformers import SentenceTransformer  # type: ignore
        from transformers import AutoModelForSequenceClassification, AutoTokenizer  # type: ignore
    except ModuleNotFoundError as exc:  # pragma: no cover - runtime dependent
        raise SystemExit(
            "Step7-v3 formal encoding requires torch, transformers, and sentence-transformers."
        ) from exc
    if not torch.cuda.is_available():
        raise SystemExit(
            "Step7-v3 formal encoding requires a CUDA GPU. Prepare and test contracts on Windows, "
            "then run this command on the Linux GPU host."
        )
    return torch, transformers, SentenceTransformer, AutoTokenizer, AutoModelForSequenceClassification


def verify_public_preparation(policy: dict) -> dict:
    outputs = policy["outputs"]
    manifest_path = common.resolve(outputs["preparation_manifest"])
    if not manifest_path.is_file():
        raise FileNotFoundError(
            "Step7-v3 public preparation manifest is missing; run the public stage first"
        )
    manifest = common.load_json(manifest_path)
    if manifest.get("version") != policy["version"]:
        raise ValueError("Step7-v3 preparation manifest version mismatch")
    if manifest.get("step") != "step7_v3_prepare_public_label_free_data":
        raise ValueError("Step7-v3 encoder requires the label-free public preparation")
    if manifest.get("feature_generation_uses_review_label_values") is not False:
        raise ValueError("Step7-v3 public preparation is not label isolated")
    if manifest.get("pair_feature_roles") != policy["pair_feature_roles"] or manifest.get(
        "shortcut_features_eligible_for_model_training_or_selection"
    ) is not False:
        raise ValueError("Step7-v3 shortcut feature roles are not audit-only")
    if manifest.get("common_script_sha256") != common.sha256_file(
        common.resolve("scripts/step7_v3_common.py")
    ):
        raise ValueError("Step7-v3 public preparation/common script drift")
    if manifest.get("redaction_dependency_script_sha256") != common.sha256_file(
        common.resolve("scripts/step3_build_seller_profiles.py")
    ):
        raise ValueError("Step7-v3 public preparation/Step3 dependency drift")
    residue_scan = manifest.get("identity_residue_scan", {})
    if residue_scan.get("status") != "pass" or residue_scan.get(
        "total_residue_count"
    ) != 0:
        raise ValueError("Step7-v3 public corpus identity-residue audit did not pass")
    common.validate_content_fidelity_manifest(policy, manifest)
    common.validate_global_identity_audit_manifest(policy, manifest)
    checks = {
        "pair_manifest": common.resolve(outputs["pair_manifest"]),
        "clean_corpus": common.resolve(outputs["clean_corpus"]),
    }
    for key, path in checks.items():
        record = manifest["output_files"][key]
        if common.sha256_file(path) != record["sha256"]:
            raise ValueError(f"Step7-v3 prepared artifact drift: {key}")
    return manifest


def verify_label_free_gpu_sync(policy: dict, policy_path: Path) -> tuple[dict, dict]:
    outputs = policy["outputs"]
    sync_path = common.resolve(outputs["gpu_sync_manifest"])
    if not sync_path.is_file():
        raise FileNotFoundError("Build and transfer the Step7-v3 GPU sync manifest first")
    manifest = common.load_json(sync_path)
    if manifest.get("step") != "step7_v3_label_free_windows_to_linux_gpu_sync":
        raise ValueError("Step7-v3 GPU sync manifest has the wrong role")
    if manifest.get("version") != policy["version"]:
        raise ValueError("Step7-v3 GPU sync manifest version mismatch")
    if manifest.get("policy_sha256") != common.sha256_file(policy_path):
        raise ValueError("Step7-v3 GPU sync policy hash drift")
    if manifest.get("policy_contract_sha256") != common.canonical_hash(policy):
        raise ValueError("Step7-v3 GPU sync policy contract drift")
    if manifest.get("label_files_included") is not False:
        raise ValueError("Step7-v3 GPU sync manifest includes labels")
    if manifest.get("raw_source_files_included") is not False:
        raise ValueError("Step7-v3 GPU sync manifest includes raw source inputs")
    for record in manifest.get("files", []):
        path = common.resolve(record["path"])
        if not path.is_file():
            raise FileNotFoundError(f"Step7-v3 transferred GPU file missing: {path}")
        if path.stat().st_size != int(record["size_bytes"]):
            raise ValueError(f"Step7-v3 transferred GPU file size drift: {record['path']}")
        if common.sha256_file(path) != record["sha256"]:
            raise ValueError(f"Step7-v3 transferred GPU file hash drift: {record['path']}")
    present_forbidden = [
        path_value
        for path_value in manifest.get("forbidden_workspace_paths", [])
        if common.resolve(path_value).exists()
    ]
    if present_forbidden:
        raise ValueError(
            "Step7-v3 formal GPU workspace is not label/raw-source isolated: "
            f"{present_forbidden[0]}. Run "
            "bash scripts/run_step7_v3_clean_source_linux_20260722.sh from the "
            "source repository; the runner creates the isolated workspace."
        )
    model_fingerprints = {}
    for model_key, cfg in {
        **policy["embedding_models"],
        policy["shared_reranker"]["model_key"]: policy["shared_reranker"],
    }.items():
        observed = common.validate_model_content_pin(model_key, cfg)
        registered = manifest["model_directories"].get(model_key)
        if registered is None or registered != {"path": cfg["local_path"], **observed}:
            raise ValueError(f"Step7-v3 GPU model fingerprint drift: {model_key}")
        model_fingerprints[model_key] = observed
    return manifest, model_fingerprints


def corpus_and_pairs(policy: dict) -> tuple[list[dict], list[dict]]:
    corpus = common.load_jsonl(common.resolve(policy["outputs"]["clean_corpus"]))
    pairs = common.load_csv(common.resolve(policy["outputs"]["pair_manifest"]))
    common.validate_clean_corpus_rows(corpus)
    common.validate_public_pair_rows(policy, pairs)
    seller_uids = [row["seller_uid"] for row in corpus]
    seller_set = set(seller_uids)
    missing = sorted(
        {
            row[key]
            for row in pairs
            for key in ("seller_uid_left", "seller_uid_right")
        }
        - seller_set
    )
    if missing:
        raise ValueError(f"Step7-v3 pair endpoint missing from clean corpus: {missing[0]}")
    if len(pairs) != int(policy["supervision_boundary"]["expected_counts"]["total"]):
        raise ValueError("Step7-v3 pair manifest row count drift")
    return corpus, pairs


def create_sentence_transformer(sentence_transformer_cls, model_dir: Path, cfg: dict, device: str):
    kwargs = {
        "device": device,
        "trust_remote_code": bool(cfg.get("trust_remote_code", False)),
    }
    try:
        model = sentence_transformer_cls(str(model_dir), local_files_only=True, **kwargs)
    except TypeError:
        # sentence-transformers 2.x does not expose local_files_only. A concrete local path
        # still prevents a Hub model lookup for the base model payload.
        model = sentence_transformer_cls(str(model_dir), **kwargs)
    model.max_seq_length = int(cfg["max_length"])
    return model


def token_length_diagnostics(
    tokenizer,
    first_texts: list[str],
    max_length: int,
    second_texts: list[str] | None = None,
    batch_size: int = 32,
) -> dict:
    lengths: list[int] = []
    for start, stop in batch_ranges(len(first_texts), batch_size):
        first = first_texts[start:stop]
        second = second_texts[start:stop] if second_texts is not None else None
        encoded = tokenizer(
            first,
            second,
            add_special_tokens=True,
            padding=False,
            truncation=False,
        )
        lengths.extend(len(ids) for ids in encoded["input_ids"])
    values = np.asarray(lengths, dtype=np.int64)
    if len(values) != len(first_texts) or np.any(values <= 0):
        raise ValueError("Step7-v3 tokenizer length audit is incomplete")
    return {
        "row_count": len(lengths),
        "max_length_contract": int(max_length),
        "truncated_row_count": int(np.sum(values > int(max_length))),
        "truncated_row_fraction": float(np.mean(values > int(max_length))),
        "token_length_min": int(np.min(values)),
        "token_length_median": float(np.median(values)),
        "token_length_p90": float(np.quantile(values, 0.90)),
        "token_length_p95": float(np.quantile(values, 0.95)),
        "token_length_max": int(np.max(values)),
    }


def encode_embedding_model(
    policy: dict,
    model_key: str,
    corpus: list[dict],
    pairs: list[dict],
    torch_module,
    sentence_transformer_cls,
    model_fingerprint: dict,
    provenance: dict,
) -> dict:
    cfg = policy["embedding_models"][model_key]
    layout = common.validate_sentence_transformer_layout(model_key, cfg)
    model_dir = common.resolve(cfg["local_path"])
    seller_uids = [row["seller_uid"] for row in corpus]
    texts = [cfg["text_prefix"] + row["model_text"] for row in corpus]
    corpus_sha256 = common.sha256_file(common.resolve(policy["outputs"]["clean_corpus"]))
    pair_manifest_sha256 = common.sha256_file(
        common.resolve(policy["outputs"]["pair_manifest"])
    )

    model = create_sentence_transformer(
        sentence_transformer_cls, model_dir, cfg, device="cuda"
    )
    length_diagnostics = token_length_diagnostics(
        model.tokenizer, texts, int(cfg["max_length"])
    )
    embeddings = model.encode(
        texts,
        batch_size=int(cfg["batch_size"]),
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    embeddings = np.asarray(embeddings, dtype=np.float32)
    if embeddings.ndim != 2 or embeddings.shape[0] != len(corpus):
        raise ValueError(f"Step7-v3 invalid embedding shape for {model_key}: {embeddings.shape}")
    if not np.all(np.isfinite(embeddings)):
        raise ValueError(f"Step7-v3 non-finite embeddings for {model_key}")
    norms = np.linalg.norm(embeddings, axis=1)
    if float(np.max(np.abs(norms - 1.0))) > 1e-3:
        raise ValueError(f"Step7-v3 embeddings are not normalized for {model_key}")

    index = {seller_uid: idx for idx, seller_uid in enumerate(seller_uids)}
    pair_scores = []
    for pair in pairs:
        left = embeddings[index[pair["seller_uid_left"]]]
        right = embeddings[index[pair["seller_uid_right"]]]
        score = float(np.dot(left, right))
        if not math.isfinite(score):
            raise ValueError(f"Step7-v3 non-finite cosine for {model_key}")
        pair_scores.append(
            {
                "pair_uid": pair["pair_uid"],
                cfg["feature_name"]: f"{score:.12f}",
            }
        )

    outputs = policy["outputs"]
    matrix_path = common.resolve(
        outputs["embedding_matrix_template"].format(model_key=model_key)
    )
    manifest_path = common.resolve(
        outputs["embedding_manifest_template"].format(model_key=model_key)
    )
    score_path = common.resolve(
        outputs["embedding_pair_scores_template"].format(model_key=model_key)
    )
    common.write_npy_immutable(matrix_path, embeddings)
    common.write_csv_immutable(score_path, pair_scores)
    manifest = {
        "step": "step7_v3_encode_clean_embedding",
        "version": policy["version"],
        "model_key": model_key,
        "repo_id": cfg["repo_id"],
        "local_path": cfg["local_path"],
        "feature_name": cfg["feature_name"],
        "pooling_contract": cfg["pooling_contract"],
        "layout_validation": layout,
        "model_fingerprint": model_fingerprint,
        **provenance,
        "feature_generation_reads_label_values": False,
        "label_or_raw_source_files_present_in_gpu_workspace": False,
        "text_prefix": cfg["text_prefix"],
        "max_length": int(cfg["max_length"]),
        "batch_size": int(cfg["batch_size"]),
        "seller_uids": seller_uids,
        "shape": list(embeddings.shape),
        "pair_count": len(pair_scores),
        "maximum_unit_norm_error": float(np.max(np.abs(norms - 1.0))),
        "token_length_diagnostics": length_diagnostics,
        "clean_corpus_sha256": corpus_sha256,
        "pair_manifest_sha256": pair_manifest_sha256,
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


def batch_ranges(length: int, batch_size: int):
    for start in range(0, length, batch_size):
        yield start, min(start + batch_size, length)


def logits_to_scores(logits, torch_module):
    if logits.ndim == 1 or logits.shape[-1] == 1:
        return torch_module.sigmoid(logits.reshape(-1))
    return torch_module.softmax(logits, dim=-1)[:, -1]


def encode_reranker(
    policy: dict,
    corpus: list[dict],
    pairs: list[dict],
    torch_module,
    transformers_module,
    tokenizer_cls,
    reranker_cls,
    model_fingerprint: dict,
    provenance: dict,
) -> dict:
    cfg = policy["shared_reranker"]
    model_dir = common.resolve(cfg["local_path"])
    layout = common.validate_reranker_layout(cfg["model_key"], cfg)
    text_by_uid = {row["seller_uid"]: row["model_text"] for row in corpus}
    tokenizer = tokenizer_cls.from_pretrained(
        str(model_dir),
        local_files_only=True,
        trust_remote_code=bool(cfg.get("trust_remote_code", False)),
    )
    model = reranker_cls.from_pretrained(
        str(model_dir),
        local_files_only=True,
        trust_remote_code=bool(cfg.get("trust_remote_code", False)),
        torch_dtype="auto",
    )
    model.to("cuda")
    model.eval()

    all_left_texts = [text_by_uid[row["seller_uid_left"]] for row in pairs]
    all_right_texts = [text_by_uid[row["seller_uid_right"]] for row in pairs]
    length_diagnostics = token_length_diagnostics(
        tokenizer,
        all_left_texts,
        int(cfg["max_length"]),
        second_texts=all_right_texts,
    )

    forward_scores = np.empty(len(pairs), dtype=np.float64)
    reverse_scores = np.empty(len(pairs), dtype=np.float64)
    with torch_module.inference_mode():
        for start, stop in batch_ranges(len(pairs), int(cfg["batch_size"])):
            batch = pairs[start:stop]
            left_texts = [text_by_uid[row["seller_uid_left"]] for row in batch]
            right_texts = [text_by_uid[row["seller_uid_right"]] for row in batch]
            for target, first, second in (
                (forward_scores, left_texts, right_texts),
                (reverse_scores, right_texts, left_texts),
            ):
                encoded = tokenizer(
                    first,
                    second,
                    padding=True,
                    truncation=True,
                    max_length=int(cfg["max_length"]),
                    return_tensors="pt",
                )
                encoded = {key: value.to("cuda") for key, value in encoded.items()}
                values = logits_to_scores(model(**encoded).logits, torch_module)
                target[start:stop] = values.detach().float().cpu().numpy()
    scores = (forward_scores + reverse_scores) / 2.0
    if not np.all(np.isfinite(scores)):
        raise ValueError("Step7-v3 reranker produced non-finite scores")
    rows = [
        {
            "pair_uid": pair["pair_uid"],
            cfg["feature_name"]: f"{float(score):.12f}",
        }
        for pair, score in zip(pairs, scores, strict=True)
    ]
    score_path = common.resolve(policy["outputs"]["reranker_pair_scores"])
    common.write_csv_immutable(score_path, rows)
    manifest = {
        "step": "step7_v3_encode_clean_reranker",
        "version": policy["version"],
        "model_key": cfg["model_key"],
        "repo_id": cfg["repo_id"],
        "local_path": cfg["local_path"],
        "feature_name": cfg["feature_name"],
        "layout_validation": layout,
        "model_fingerprint": model_fingerprint,
        **provenance,
        "feature_generation_reads_label_values": False,
        "label_or_raw_source_files_present_in_gpu_workspace": False,
        "pair_symmetrization": cfg["pair_symmetrization"],
        "single_logit_transform": cfg["single_logit_transform"],
        "max_length": int(cfg["max_length"]),
        "batch_size": int(cfg["batch_size"]),
        "forward_reverse_mean_absolute_gap": float(np.mean(np.abs(forward_scores - reverse_scores))),
        "token_length_diagnostics": length_diagnostics,
        "clean_corpus_sha256": common.sha256_file(
            common.resolve(policy["outputs"]["clean_corpus"])
        ),
        "pair_manifest_sha256": common.sha256_file(
            common.resolve(policy["outputs"]["pair_manifest"])
        ),
        "pair_scores_sha256": common.sha256_file(score_path),
        "pair_count": len(rows),
        "device": "cuda",
        "gpu_name": torch_module.cuda.get_device_name(0),
        "transformers_version": transformers_module.__version__,
        "torch_version": torch_module.__version__,
    }
    common.write_json_immutable(
        common.resolve(policy["outputs"]["reranker_manifest"]), manifest
    )
    del model
    torch_module.cuda.empty_cache()
    return manifest


def gpu_output_record(path_value: str) -> dict:
    path = common.resolve(path_value)
    if not path.is_file():
        raise FileNotFoundError(f"Step7-v3 expected GPU output is missing: {path}")
    return {
        "path": str(path.relative_to(common.ROOT)).replace("\\", "/"),
        "size_bytes": path.stat().st_size,
        "sha256": common.sha256_file(path),
    }


def write_gpu_output_manifest(policy: dict, provenance: dict) -> dict:
    outputs = policy["outputs"]
    paths = []
    for model_key in policy["embedding_models"]:
        paths.extend(
            [
                outputs["embedding_matrix_template"].format(model_key=model_key),
                outputs["embedding_manifest_template"].format(model_key=model_key),
                outputs["embedding_pair_scores_template"].format(model_key=model_key),
            ]
        )
    paths.extend([outputs["reranker_pair_scores"], outputs["reranker_manifest"]])
    records = [gpu_output_record(path) for path in paths]
    payload = {
        "step": "step7_v3_label_free_gpu_output_bundle",
        "version": policy["version"],
        **provenance,
        "label_or_raw_source_files_present_in_gpu_workspace": False,
        "file_count": len(records),
        "total_file_bytes": sum(record["size_bytes"] for record in records),
        "files": records,
    }
    common.write_json_immutable(common.resolve(outputs["gpu_output_manifest"]), payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", default=str(common.DEFAULT_POLICY))
    parser.add_argument("--embedding-model", action="append", dest="embedding_models")
    parser.add_argument("--skip-reranker", action="store_true")
    parser.add_argument("--validate-config-only", action="store_true")
    args = parser.parse_args()

    policy_path = common.resolve(args.policy)
    policy = common.load_json(policy_path)
    common.validate_policy(policy)
    selected_models = args.embedding_models or list(policy["embedding_models"])
    unknown = sorted(set(selected_models) - set(policy["embedding_models"]))
    if unknown:
        raise ValueError(f"Unknown Step7-v3 embedding model keys: {unknown}")
    preparation = verify_public_preparation(policy)
    layouts = {
        key: common.validate_sentence_transformer_layout(key, policy["embedding_models"][key])
        for key in selected_models
    }
    reranker_layout = common.validate_reranker_layout(
        policy["shared_reranker"]["model_key"], policy["shared_reranker"]
    )
    if args.validate_config_only:
        print(
            json.dumps(
                {
                    "status": "pass",
                    "selected_embedding_models": selected_models,
                    "run_shared_reranker": not args.skip_reranker,
                    "model_layouts": layouts,
                    "reranker_layout": reranker_layout,
                    "preparation_manifest_version": preparation["version"],
                    "formal_execution_requires_linux_cuda": True,
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
        "generator_script_path": str(ENCODER_SCRIPT.relative_to(common.ROOT)).replace("\\", "/"),
        "generator_script_sha256": common.sha256_file(ENCODER_SCRIPT),
        "gpu_sync_manifest_sha256": common.sha256_file(sync_path),
        "public_preparation_manifest_sha256": common.sha256_file(
            common.resolve(policy["outputs"]["preparation_manifest"])
        ),
    }
    if gpu_sync["public_preparation_manifest_sha256"] != provenance[
        "public_preparation_manifest_sha256"
    ]:
        raise ValueError("Step7-v3 GPU sync/public preparation provenance drift")
    (
        torch_module,
        transformers_module,
        sentence_transformer_cls,
        tokenizer_cls,
        reranker_cls,
    ) = require_gpu_stack()
    corpus, pairs = corpus_and_pairs(policy)
    manifests = {
        key: encode_embedding_model(
            policy,
            key,
            corpus,
            pairs,
            torch_module,
            sentence_transformer_cls,
            model_fingerprints[key],
            provenance,
        )
        for key in selected_models
    }
    reranker_manifest = None
    if not args.skip_reranker:
        reranker_manifest = encode_reranker(
            policy,
            corpus,
            pairs,
            torch_module,
            transformers_module,
            tokenizer_cls,
            reranker_cls,
            model_fingerprints[policy["shared_reranker"]["model_key"]],
            provenance,
        )
    output_bundle = None
    if set(selected_models) == set(policy["embedding_models"]) and reranker_manifest is not None:
        output_bundle = write_gpu_output_manifest(policy, provenance)
    print(
        json.dumps(
            {
                "status": "pass",
                "platform": platform.platform(),
                "embedding_models": list(manifests),
                "reranker_completed": reranker_manifest is not None,
                "gpu_output_bundle_written": output_bundle is not None,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
