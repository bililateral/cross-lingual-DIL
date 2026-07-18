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


def clean_real_profiles(policy: dict, fields: list[str]) -> tuple[list[dict], dict]:
    canonical_path = common.parent_root(policy) / "canonical_pairs.csv"
    canonical = common.load_csv(canonical_path)
    seller_uids = sorted(
        {
            row[field]
            for row in canonical
            for field in ("seller_uid_left", "seller_uid_right")
        }
    )
    profiles_path = common.policy_input(policy, "seller_profiles", "zh_seller_profiles")
    signals_path = common.policy_input(policy, "item_identity_signals", "zh_item_identity_signals")
    profiles = common.load_profiles_index(profiles_path)
    literals_by_seller, signal_summary = redaction.signal_literals_by_seller(signals_path)
    rows = []
    counts: Counter[str] = Counter()
    empty_fallback = policy.get("clean_text_contract", {}).get(
        "empty_text_fallback", "[EMPTY_REDACTED_PROFILE]"
    )
    for seller_uid in seller_uids:
        profile = profiles.get(seller_uid)
        if profile is None:
            raise ValueError(f"Step27 canonical seller profile is missing: {seller_uid}")
        cleaned, diagnostics = common.clean_profile_fields(profile, fields, literals_by_seller)
        literals = common.profile_literals(profile, literals_by_seller)
        cleaned, second_pass = common.redact_transformed_fields(cleaned, literals, seller_uid)
        counts.update(diagnostics)
        counts.update({f"second_pass_{key}": value for key, value in second_pass.items()})
        text = common.render_profile_text(cleaned)
        if not text:
            text = empty_fallback
            counts["empty_after_redaction_count"] += 1
        rows.append(
            {
                "seller_uid": seller_uid,
                "data_bucket": "zh_target_strict",
                **cleaned,
                "profile_text": text,
                "identifier_redacted": True,
                "synthetic_train_only": False,
                "source_market_raw": "",
                "source_seller_raw": "",
            }
        )
    return rows, {"redaction": dict(sorted(counts.items())), "signals": signal_summary}


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
) -> dict:
    identity = {**identity_base, "cache_name": name, "source_inputs": common.records_for(source_paths)}
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
        "existing": existing,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", default=str(common.DEFAULT_POLICY))
    parser.add_argument("--seed", action="append", type=int, dest="seeds")
    parser.add_argument("--track", action="append", choices=["primary", "silver_sensitivity"], dest="tracks")
    parser.add_argument("--device", default=None)
    parser.add_argument("--validate-config-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    policy_path, policy = common.load_policy(args.policy)
    fields = common.text_fields(policy)
    seeds = args.seeds or common.generation_seeds(policy)
    tracks = args.tracks or ["primary", "silver_sensitivity"]
    semantic_policy_path = common.policy_input(policy, "semantic_model_policy")
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

    parent_manifest = common.parent_root(policy) / "manifest.json"
    if not parent_manifest.is_file():
        raise FileNotFoundError("Run step27_build_parent_manifest.py first")
    model_dir = semantic.resolve_local_model_dir(model_key, model_cfg)
    model_fingerprint = redaction.directory_fingerprint(model_dir)
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
    jobs = []
    real_rows, real_diagnostics = clean_real_profiles(policy, fields)
    real_matrix, real_metadata = common.profile_cache_paths(policy, None, "real")
    real_root = real_matrix.parent
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
            ],
            diagnostics=real_diagnostics,
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
                    source_paths=[policy_path, generation_manifest, source_path],
                    diagnostics=diagnostics,
                )
            )

    pending = [job for job in jobs if job["existing"] is None]
    if pending:
        torch_module, tokenizer_cls, model_cls, _ = semantic.require_torch_and_transformers()
        device = semantic.choose_device(torch_module, semantic_policy["device_preference"], args.device)
        all_texts = [row["profile_text"] for job in pending for row in job["clean_rows"]]
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
        cursor = 0
        for job in pending:
            count = len(job["clean_rows"])
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
    else:
        device = "existing_immutable_cache"

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
