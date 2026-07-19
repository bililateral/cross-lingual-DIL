#!/usr/bin/env python3
"""Re-redact and encode every Step27 real/synthetic profile with frozen E5."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

import step15_build_v7_clean_embedding_cache as redaction
import step27_common as common
import step7_build_semantic_pair_features as semantic


def verified_contract_input(policy: dict, path_key: str, hash_key: str) -> Path:
    inputs = dict(policy.get("inputs") or {})
    if not inputs.get(path_key) or not inputs.get(hash_key):
        raise ValueError(f"Step27 encoder contract is missing {path_key}/{hash_key}")
    path = common.resolve(inputs[path_key])
    if not path.is_file() or common.sha256_file(path) != inputs[hash_key]:
        raise ValueError(f"Step27 frozen encoder contract hash mismatch: {path}")
    return path


def load_real_replay_contract(policy: dict, fields: list[str]) -> dict:
    bundle = common.frozen_step24_bundle(policy)
    v7_policy_path = verified_contract_input(
        policy,
        "identifier_redaction_policy",
        "identifier_redaction_policy_sha256",
    )
    redaction_producer_path = verified_contract_input(
        policy,
        "identifier_redaction_producer",
        "identifier_redaction_producer_sha256",
    )
    if redaction_producer_path.resolve() != Path(redaction.__file__).resolve():
        raise ValueError("Step27 imported redaction producer differs from the frozen contract")
    v7_policy = common.load_json(v7_policy_path)
    clean_cfg = dict(v7_policy["clean_semantic_encoder"])
    if list(clean_cfg.get("text_fields", [])) != fields:
        raise ValueError("Step27 text fields do not exactly replay the Step15-v7 field order")
    pool_cfg = dict(v7_policy["pools"]["zh_target_strict"])
    metadata_path = common.resolve(pool_cfg["clean_e5_cache_metadata"])
    matrix_path = common.resolve(pool_cfg["clean_e5_cache_matrix"])

    step24_policy_path = bundle["paths"]["policy"]
    step24_policy = common.load_json(step24_policy_path)
    clean_manifest_path = bundle["paths"]["clean_text_manifest"]
    clean_manifest = bundle["clean_text_manifest"]
    expected = dict(clean_manifest["records"]["zh_target_strict"])
    if common.sha256_file(metadata_path) != expected["v7_e5_metadata_sha256"]:
        raise ValueError("Step27 frozen v7 E5 metadata differs from the Step24 replay contract")
    if common.sha256_file(matrix_path) != expected["v7_e5_matrix_sha256"]:
        raise ValueError("Step27 frozen v7 E5 matrix differs from the Step24 replay contract")
    metadata = common.load_json(metadata_path)
    if metadata.get("producer_sha256") != common.sha256_file(redaction_producer_path):
        raise ValueError("Step27 frozen v7 cache producer differs from the encoder contract")
    observed_full_hash = metadata.get("redaction_diagnostics", {}).get(
        "clean_text_corpus_sha256"
    )
    if observed_full_hash != expected["full_v7_clean_text_corpus_sha256_verified"]:
        raise ValueError("Step27 frozen v7 clean-text corpus hash differs from Step24")
    summary_record = dict(
        bundle["pair_feature_summary"].get("records", {}).get("zh_target_strict")
        or {}
    )
    if (
        summary_record.get("e5_metadata_sha256") != expected["v7_e5_metadata_sha256"]
        or summary_record.get("e5_matrix_sha256") != expected["v7_e5_matrix_sha256"]
    ):
        raise ValueError("Step27 Step24 clean-text and pair-feature E5 anchors disagree")
    return {
        "v7_policy_path": v7_policy_path,
        "v7_policy": v7_policy,
        "redaction_producer_path": redaction_producer_path,
        "clean_cfg": clean_cfg,
        "metadata_path": metadata_path,
        "matrix_path": matrix_path,
        "metadata": metadata,
        "step24_policy_path": step24_policy_path,
        "clean_manifest_path": clean_manifest_path,
        "expected": expected,
        "step24_bundle": bundle,
    }


def clean_real_profiles(
    policy: dict, fields: list[str], replay: dict
) -> tuple[list[dict], dict]:
    canonical_path = common.parent_root(policy) / "canonical_pairs.csv"
    canonical = common.load_csv(canonical_path)
    seller_uids = sorted(
        {
            row[field]
            for row in canonical
            for field in ("seller_uid_left", "seller_uid_right")
        }
    )
    train_seller_uids = sorted(
        {
            row[field]
            for row in canonical
            if row.get("split_name") == "train"
            for field in ("seller_uid_left", "seller_uid_right")
        }
    )
    profiles_path = common.policy_input(policy, "seller_profiles", "zh_seller_profiles")
    signals_path = common.policy_input(policy, "item_identity_signals", "zh_item_identity_signals")
    profiles = common.load_profiles_index(profiles_path)
    literals_by_seller, signal_summary = redaction.signal_literals_by_seller(signals_path)
    rows = []
    counts: Counter[str] = Counter()
    exact_text_by_uid: dict[str, str] = {}
    clean_cfg = replay["clean_cfg"]
    for seller_uid in seller_uids:
        profile = profiles.get(seller_uid)
        if profile is None:
            raise ValueError(f"Step27 canonical seller profile is missing: {seller_uid}")
        literals = common.profile_literals(profile, literals_by_seller)
        source_text = redaction.build_content_text(profile, clean_cfg)
        exact_text, exact_diagnostics = redaction.redact_identifiers(source_text, literals)
        redaction.assert_no_known_identifier_residue(exact_text, literals, seller_uid)
        if not exact_text:
            exact_text = "content unavailable"
            counts["empty_after_redaction_count"] += 1
        exact_text_by_uid[seller_uid] = exact_text
        counts.update({f"exact_replay_{key}": value for key, value in exact_diagnostics.items()})

        # Retain separately redacted fields only for Step27's explicit residual
        # features. The frozen E5 input always uses exact_text above.
        cleaned, diagnostics = common.clean_profile_fields(profile, fields, literals_by_seller)
        cleaned, second_pass = common.redact_transformed_fields(cleaned, literals, seller_uid)
        counts.update(diagnostics)
        counts.update({f"second_pass_{key}": value for key, value in second_pass.items()})
        rows.append(
            {
                "seller_uid": seller_uid,
                "data_bucket": "zh_target_strict",
                **cleaned,
                "profile_text": exact_text,
                "identifier_redacted": True,
                "synthetic_train_only": False,
                "source_market_raw": "",
                "source_seller_raw": "",
            }
        )
    selected_hash = common.canonical_hash(
        [(uid, exact_text_by_uid[uid]) for uid in train_seller_uids]
    )
    expected_hash = replay["expected"]["selected_clean_text_corpus_sha256"]
    if selected_hash != expected_hash:
        raise ValueError(
            "Step27 reconstructed train clean text does not exactly replay Step24: "
            f"expected={expected_hash} observed={selected_hash}"
        )
    common.assert_exact_real_text_replay(
        exact_text_by_uid,
        {str(row["seller_uid"]): str(row["profile_text"]) for row in rows},
    )
    return rows, {
        "redaction": dict(sorted(counts.items())),
        "signals": signal_summary,
        "exact_replay": {
            "train_seller_count": len(train_seller_uids),
            "all_canonical_seller_count": len(seller_uids),
            "selected_clean_text_corpus_sha256": selected_hash,
            "expected_selected_clean_text_corpus_sha256": expected_hash,
            "verified": True,
        },
    }


def clean_synthetic_profiles(path: Path, fields: list[str]) -> tuple[list[dict], dict]:
    source_rows = common.load_jsonl(path)
    output = []
    counts: Counter[str] = Counter()
    for source in source_rows:
        seller_uid = str(source["seller_uid"])
        field_order = source.get("synthetic_field_order")
        if not isinstance(field_order, list) or set(field_order) != set(fields) or len(
            field_order
        ) != len(fields):
            raise ValueError(f"Step27 synthetic profile lost its transformed field order: {seller_uid}")
        cleaned = {}
        for field in field_order:
            value, diagnostics = redaction.redact_identifiers(str(source.get(field, "") or ""), [])
            redaction.assert_no_known_identifier_residue(value, [], seller_uid)
            cleaned[field] = value
            counts.update(diagnostics)
        text = common.render_profile_text(cleaned)
        text, joined_diagnostics = redaction.redact_identifiers(text, [])
        redaction.assert_no_known_identifier_residue(text, [], seller_uid)
        counts.update({f"joined_{key}": value for key, value in joined_diagnostics.items()})
        if not text:
            raise ValueError(f"Step27 synthetic profile became empty after re-redaction: {seller_uid}")
        lineage = source.get("synthetic_lineage")
        if not isinstance(lineage, dict):
            raise ValueError(f"Step27 synthetic profile has no lineage: {seller_uid}")
        if source.get("source_market_raw") or source.get("source_dataset"):
            raise ValueError(f"Step27 synthetic profile fabricates source provenance: {seller_uid}")
        if int(source.get("contact_token_count_total", 0)) != 0 or any(
            source.get("contact_signals", {}).get(key) for key in source.get("contact_signals", {})
        ):
            raise ValueError(f"Step27 synthetic profile contains identifiers: {seller_uid}")
        output.append(
            {
                "seller_uid": seller_uid,
                "data_bucket": "zh_synthetic_train_only",
                **cleaned,
                "profile_text": text,
                "identifier_redacted": True,
                "synthetic_train_only": True,
                "source_market_raw": "",
                "source_seller_raw": "",
                "synthetic_lineage": lineage,
            }
        )
    if len({row["seller_uid"] for row in output}) != len(output):
        raise ValueError(f"Step27 synthetic profile UIDs are duplicated: {path}")
    return sorted(output, key=lambda row: row["seller_uid"]), dict(sorted(counts.items()))


def tokenizer_truncation_audit(tokenizer, texts: list[str], model_cfg: dict) -> dict:
    """Measure information truncated by the frozen encoder without changing encoding."""
    maximum_length = int(model_cfg["max_length"])
    prefix = str(model_cfg.get("text_prefix", ""))
    batch_size = max(1, min(128, int(model_cfg.get("batch_size", 16)) * 4))
    lengths: list[int] = []
    for start in range(0, len(texts), batch_size):
        prefixed = [prefix + text for text in texts[start : start + batch_size]]
        encoded = tokenizer(
            prefixed,
            add_special_tokens=True,
            padding=False,
            truncation=False,
        )
        lengths.extend(len(ids) for ids in encoded["input_ids"])
    truncated = [length for length in lengths if length > maximum_length]
    return {
        "profile_count": len(lengths),
        "encoder_max_length": maximum_length,
        "maximum_untruncated_token_count": max(lengths, default=0),
        "truncated_profile_count": len(truncated),
        "truncated_profile_fraction": (
            float(len(truncated) / len(lengths)) if lengths else 0.0
        ),
        "tokens_removed_total": int(
            sum(length - maximum_length for length in truncated)
        ),
        "audit_only_encoding_parameters_unchanged": True,
    }


def cache_job(
    *,
    name: str,
    clean_rows: list[dict],
    clean_path: Path,
    matrix_path: Path,
    metadata_path: Path,
    manifest_path: Path,
    identity_base: dict,
    source_paths: list[Path],
    diagnostics: dict,
    precomputed_matrix: np.ndarray | None = None,
    exact_replay: dict | None = None,
) -> dict:
    identity = {
        **identity_base,
        "cache_name": name,
        "source_inputs": common.records_for(source_paths),
        "embedding_source": (
            "frozen_step15_v7_cache_exact_subset"
            if precomputed_matrix is not None
            else "frozen_encoder_recompute"
        ),
        "exact_replay": exact_replay or {},
    }
    existing = common.assert_existing_manifest_identity(manifest_path, identity)
    return {
        "name": name,
        "clean_rows": clean_rows,
        "clean_path": clean_path,
        "matrix_path": matrix_path,
        "metadata_path": metadata_path,
        "manifest_path": manifest_path,
        "identity": identity,
        "source_paths": source_paths,
        "diagnostics": diagnostics,
        "precomputed_matrix": precomputed_matrix,
        "exact_replay": exact_replay or {},
        "existing": existing,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", default=str(common.DEFAULT_POLICY))
    parser.add_argument("--seed", action="append", type=int, dest="seeds")
    parser.add_argument("--track", action="append", choices=["primary", "silver_sensitivity"], dest="tracks")
    parser.add_argument("--device", default=None)
    parser.add_argument("--validate-config-only", action="store_true")
    parser.add_argument("--validate-model-contract-only", action="store_true")
    return parser.parse_args()


def validate_encoder_model_fingerprint(
    model_key: str,
    model_cfg: dict,
    replay: dict,
) -> tuple[Path, dict]:
    model_dir = semantic.resolve_local_model_dir(model_key, model_cfg)
    model_fingerprint = redaction.directory_fingerprint(model_dir)
    frozen_model_fingerprint = replay["metadata"].get("model_directory_fingerprint")
    if not isinstance(frozen_model_fingerprint, dict):
        raise ValueError("Step27 frozen v7 cache has no model-directory fingerprint")
    if model_fingerprint != frozen_model_fingerprint:
        raise ValueError(
            "Step27 current E5 model directory differs from the model that produced "
            "the frozen real cache; synthetic and real embeddings cannot be mixed"
        )
    return model_dir, model_fingerprint


def main() -> None:
    args = parse_args()
    policy_path, policy = common.load_policy(args.policy)
    fields = common.text_fields(policy)
    replay = load_real_replay_contract(policy, fields)
    seeds = args.seeds or common.generation_seeds(policy)
    tracks = args.tracks or ["primary", "silver_sensitivity"]
    semantic_policy_path = verified_contract_input(
        policy,
        "semantic_model_policy",
        "semantic_model_policy_sha256",
    )
    semantic_producer_path = verified_contract_input(
        policy,
        "semantic_encoder_producer",
        "semantic_encoder_producer_sha256",
    )
    if semantic_producer_path.resolve() != Path(semantic.__file__).resolve():
        raise ValueError("Step27 imported semantic encoder differs from the frozen contract")
    semantic_policy = common.load_json(semantic_policy_path)
    model_key = policy.get("encoder", {}).get("model_key", "multilingual_e5_large")
    if model_key != "multilingual_e5_large":
        raise ValueError("Step27 primary encoding contract requires multilingual_e5_large")
    if model_key not in semantic_policy.get("embedding_models", {}):
        raise ValueError(f"Step27 semantic policy does not define {model_key}")
    model_cfg = dict(semantic_policy["embedding_models"][model_key])
    if args.validate_config_only:
        print(
            json.dumps(
                {
                    "status": "pass",
                    "model_key": model_key,
                    "seeds": seeds,
                    "tracks": tracks,
                    "model_directory_checked": False,
                    "numerical_execution_performed": False,
                },
                indent=2,
            )
        )
        return

    model_dir, model_fingerprint = validate_encoder_model_fingerprint(
        model_key,
        model_cfg,
        replay,
    )
    if args.validate_model_contract_only:
        print(
            json.dumps(
                {
                    "status": "pass",
                    "model_key": model_key,
                    "model_directory": str(model_dir),
                    "model_fingerprint": model_fingerprint,
                    "model_loaded": False,
                    "numerical_execution_performed": False,
                },
                indent=2,
            )
        )
        return

    parent_manifest = common.parent_root(policy) / "manifest.json"
    if not parent_manifest.is_file():
        raise FileNotFoundError("Run step27_build_parent_manifest.py first")
    identity_base = {
        "stage": "step27_encode_profiles",
        "policy_sha256": common.sha256_file(policy_path),
        "producer_sha256": common.sha256_file(Path(__file__).resolve()),
        "common_sha256": common.sha256_file(Path(common.__file__).resolve()),
        "shared_dependency_sha256": common.shared_dependency_hashes(),
        "semantic_encoder_producer_sha256": common.sha256_file(Path(semantic.__file__).resolve()),
        "semantic_policy_sha256": common.sha256_file(semantic_policy_path),
        "model_key": model_key,
        "model_repo_id": model_cfg["repo_id"],
        "model_fingerprint": model_fingerprint,
        "identifier_redacted": True,
        "encoder_parameters_updated": False,
    }
    encoder_contract_paths = [
        semantic_policy_path,
        semantic_producer_path,
        replay["v7_policy_path"],
        replay["redaction_producer_path"],
    ]
    jobs = []
    real_rows, real_diagnostics = clean_real_profiles(policy, fields, replay)
    full_index, full_matrix, full_metadata = common.load_normalized_cache(
        replay["metadata_path"], replay["matrix_path"]
    )
    real_uids = [row["seller_uid"] for row in real_rows]
    missing_real_uids = [uid for uid in real_uids if uid not in full_index]
    if missing_real_uids:
        raise ValueError(
            f"Step27 canonical seller is absent from the frozen v7 E5 cache: {missing_real_uids[0]}"
        )
    real_replay_matrix = np.asarray(
        full_matrix[[full_index[uid] for uid in real_uids]], dtype=np.float32
    )
    if full_metadata.get("identifier_redacted") is not True:
        raise ValueError("Step27 frozen v7 cache is not identifier-redacted")
    real_matrix, real_metadata = common.profile_cache_paths(policy, None, "real")
    real_root = real_matrix.parent
    exact_replay = {
        "contract": "step15_v7_identifier_redacted_clean_text_exact_replay",
        "source_metadata_sha256": common.sha256_file(replay["metadata_path"]),
        "source_matrix_sha256": common.sha256_file(replay["matrix_path"]),
        "selected_clean_text_corpus_sha256": real_diagnostics["exact_replay"][
            "selected_clean_text_corpus_sha256"
        ],
        "source_cache_subset_reused_without_reencoding": True,
    }
    jobs.append(
        cache_job(
            name="real_canonical_profiles",
            clean_rows=real_rows,
            clean_path=real_root / "clean_profiles.jsonl",
            matrix_path=real_matrix,
            metadata_path=real_metadata,
            manifest_path=real_root / "manifest.json",
            identity_base=identity_base,
            source_paths=[
                policy_path,
                parent_manifest,
                common.parent_root(policy) / "canonical_pairs.csv",
                common.policy_input(policy, "seller_profiles", "zh_seller_profiles"),
                common.policy_input(policy, "item_identity_signals", "zh_item_identity_signals"),
                replay["v7_policy_path"],
                replay["metadata_path"],
                replay["matrix_path"],
                replay["step24_policy_path"],
                replay["clean_manifest_path"],
                replay["step24_bundle"]["paths"]["sync_manifest"],
                replay["step24_bundle"]["paths"]["pair_feature_summary"],
                replay["step24_bundle"]["paths"]["zh_pair_features"],
                replay["step24_bundle"]["paths"]["model_artifacts"],
                *encoder_contract_paths,
            ],
            diagnostics=real_diagnostics,
            precomputed_matrix=real_replay_matrix,
            exact_replay=exact_replay,
        )
    )
    for seed in seeds:
        generation_manifest = common.seed_root(policy, seed) / "generation_manifest.json"
        if not generation_manifest.is_file():
            raise FileNotFoundError(f"Run Step27 generation first for seed={seed}")
        for track in tracks:
            source_path = common.track_root(policy, seed, track) / "synthetic_profiles.jsonl"
            clean_rows, diagnostics = clean_synthetic_profiles(source_path, fields)
            matrix_path, metadata_path = common.profile_cache_paths(policy, seed, track)
            jobs.append(
                cache_job(
                    name=f"seed_{seed}:{track}",
                    clean_rows=clean_rows,
                    clean_path=matrix_path.parent / "clean_profiles.jsonl",
                    matrix_path=matrix_path,
                    metadata_path=metadata_path,
                    manifest_path=matrix_path.parent / "manifest.json",
                    identity_base=identity_base,
                    source_paths=[
                        policy_path,
                        generation_manifest,
                        source_path,
                        *encoder_contract_paths,
                    ],
                    diagnostics=diagnostics,
                )
            )

    pending = [job for job in jobs if job["existing"] is None]
    cursor = 0
    encode_pending = [job for job in pending if job["precomputed_matrix"] is None]
    if encode_pending:
        torch_module, tokenizer_cls, model_cls, _ = semantic.require_torch_and_transformers()
        device = semantic.choose_device(torch_module, semantic_policy["device_preference"], args.device)
        tokenizer = tokenizer_cls.from_pretrained(
            str(model_dir),
            local_files_only=True,
            trust_remote_code=model_cfg.get("trust_remote_code", False),
        )
        for job in encode_pending:
            job["tokenizer_truncation_audit"] = tokenizer_truncation_audit(
                tokenizer,
                [row["profile_text"] for row in job["clean_rows"]],
                model_cfg,
            )
        all_texts = [row["profile_text"] for job in encode_pending for row in job["clean_rows"]]
        all_embeddings = semantic.encode_texts(
            model_key,
            model_cfg,
            all_texts,
            device,
            torch_module,
            tokenizer_cls,
            model_cls,
        )
        if all_embeddings.shape[0] != len(all_texts):
            raise ValueError("Step27 E5 encoder returned the wrong row count")
    else:
        device = "frozen_step15_v7_cache_exact_subset"
        all_texts = []
        all_embeddings = np.empty((0, 0), dtype=np.float32)

    if pending:
        for job in pending:
            count = len(job["clean_rows"])
            if job["precomputed_matrix"] is not None:
                matrix = np.asarray(job["precomputed_matrix"], dtype=np.float32)
            else:
                matrix = np.asarray(all_embeddings[cursor : cursor + count], dtype=np.float32)
                cursor += count
            norms = np.linalg.norm(matrix, axis=1)
            if matrix.ndim != 2 or not np.all(np.isfinite(matrix)) or np.max(np.abs(norms - 1.0)) > 1e-3:
                raise ValueError(f"Step27 produced an invalid normalized cache: {job['name']}")
            common.write_jsonl_immutable(job["clean_path"], job["clean_rows"])
            common.write_npy_immutable(job["matrix_path"], matrix)
            metadata = {
                "step": "step27_encode_profiles",
                "cache_name": job["name"],
                "model_key": model_key,
                "model_repo_id": model_cfg["repo_id"],
                "model_local_path": model_cfg["local_path"],
                "model_fingerprint": model_fingerprint,
                "seller_uids": [row["seller_uid"] for row in job["clean_rows"]],
                "shape": list(matrix.shape),
                "clean_profiles_path": common.relative(job["clean_path"]),
                "clean_profiles_sha256": common.sha256_file(job["clean_path"]),
                "clean_text_corpus_sha256": common.canonical_hash(
                    [(row["seller_uid"], row["profile_text"]) for row in job["clean_rows"]]
                ),
                "identifier_redacted": True,
                "encoder_parameters_updated": False,
                "embedding_source": job["identity"]["embedding_source"],
                "exact_replay": job["exact_replay"],
                "tokenizer_truncation_audit": job.get(
                    "tokenizer_truncation_audit",
                    {
                        "not_recomputed_for_exact_frozen_cache_subset": True,
                        "audit_only_encoding_parameters_unchanged": True,
                    },
                ),
                "representation_contract": {
                    "semantic_text_preserves_transformed_field_order": True,
                    "residual_lexical_features_use_named_field_groups": True,
                    "semantic_and_residual_order_contracts_are_intentionally_distinct": True,
                },
                "synthetic_train_only": all(
                    common.bool_value(row.get("synthetic_train_only")) for row in job["clean_rows"]
                ),
                "labels_used_for_encoding": False,
                "redaction_diagnostics": job["diagnostics"],
                "device": str(device),
            }
            common.write_json_immutable(job["metadata_path"], metadata)
            common.write_manifest_immutable(
                job["manifest_path"],
                stage="step27_encode_profiles",
                identity=job["identity"],
                inputs=job["source_paths"],
                outputs=[job["clean_path"], job["matrix_path"], job["metadata_path"]],
                extra={"row_count": count, "embedding_dimension": int(matrix.shape[1])},
            )
        if cursor != len(all_texts):
            raise AssertionError("Step27 cache slicing did not consume all embeddings")
    elif not pending:
        device = "existing_immutable_cache"

    persisted_real_index, persisted_real_matrix, persisted_real_metadata = (
        common.load_normalized_cache(real_metadata, real_matrix)
    )
    persisted_real_uids = list(persisted_real_metadata.get("seller_uids", []))
    common.assert_exact_real_embedding_replay(
        real_uids,
        real_replay_matrix,
        persisted_real_uids,
        np.asarray(
            persisted_real_matrix[
                [persisted_real_index[uid] for uid in persisted_real_uids]
            ],
            dtype=np.float32,
        ),
        atol=0.0,
    )
    if persisted_real_metadata.get("embedding_source") != "frozen_step15_v7_cache_exact_subset":
        raise ValueError("Step27 persisted real cache is not an exact frozen-cache subset")

    print(
        json.dumps(
            {
                "status": "pass",
                "device": str(device),
                "cache_count": len(jobs),
                "new_cache_count": len(pending),
                "reused_identical_cache_count": len(jobs) - len(pending),
                "real_seller_count": len(real_rows),
                "synthetic_seller_count": sum(
                    len(job["clean_rows"]) for job in jobs if job["name"] != "real_canonical_profiles"
                ),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
