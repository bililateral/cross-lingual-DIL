#!/usr/bin/env python3
"""Build identifier-redacted BGE/LaBSE/reranker features for Step15-v8."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

import step7_build_semantic_pair_features as semantic
import step15_build_v7_clean_embedding_cache as v7_cache
import step15_v8_common as common


ROOT = Path(__file__).resolve().parent.parent


def cosine_rows(rows: list[dict], seller_uids: list[str], matrix: np.ndarray) -> list[float]:
    index = {seller_uid: position for position, seller_uid in enumerate(seller_uids)}
    result = []
    for row in rows:
        left = np.asarray(matrix[index[row["seller_uid_left"]]], dtype=float)
        right = np.asarray(matrix[index[row["seller_uid_right"]]], dtype=float)
        denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
        result.append(0.0 if denominator <= 1e-12 else float(np.dot(left, right) / denominator))
    return result


def prepare_clean_texts(policy: dict, pool_name: str) -> dict:
    pool = policy["pools"][pool_name]
    profiles_path = common.resolve(pool["seller_profiles"])
    signals_path = common.resolve(pool["item_identity_signals"])
    pair_path = common.resolve(pool["v7_pair_features"])
    profiles_list = semantic.load_jsonl(profiles_path)
    profiles = {str(row["seller_uid"]): row for row in profiles_list}
    if len(profiles) != len(profiles_list):
        raise ValueError(f"Duplicate seller UID in v8 profile input: {pool_name}")
    pair_rows = common.load_csv(pair_path)
    seller_uids = sorted(
        {
            str(row[key])
            for row in pair_rows
            for key in ("seller_uid_left", "seller_uid_right")
            if str(row.get(key, "")).strip()
        }
    )
    missing = [seller_uid for seller_uid in seller_uids if seller_uid not in profiles]
    if missing:
        raise ValueError(f"V8 clean semantic profile missing: {pool_name}:{missing[0]}")
    literals, signal_diagnostics = v7_cache.signal_literals_by_seller(signals_path)
    clean_texts = {}
    diagnostics = Counter()
    for seller_uid in seller_uids:
        source_text = v7_cache.build_content_text(profiles[seller_uid], policy["clean_semantics"])
        seller_literals = list(literals.get(seller_uid, []))
        for alias_field in ("source_seller_raw", "alias_normalized"):
            literal = v7_cache.safe_signal_literal(
                "seller_alias", profiles[seller_uid].get(alias_field, "")
            )
            if literal:
                seller_literals.append(literal)
        seller_literals = sorted(
            set(seller_literals), key=lambda value: (-len(value), value.casefold())
        )
        clean_text, item = v7_cache.redact_identifiers(source_text, seller_literals)
        v7_cache.assert_no_known_identifier_residue(clean_text, seller_literals, seller_uid)
        diagnostics.update(item)
        if item["redaction_pass_count"] > 2:
            diagnostics["fixed_point_extra_pass_seller_count"] += 1
        diagnostics["max_redaction_pass_count"] = max(
            diagnostics["max_redaction_pass_count"], item["redaction_pass_count"]
        )
        if not clean_text:
            clean_text = "content unavailable"
            diagnostics["empty_after_redaction_count"] += 1
        clean_texts[seller_uid] = clean_text
    return {
        "seller_uids": seller_uids,
        "clean_texts": clean_texts,
        "pair_rows": pair_rows,
        "diagnostics": {
            **signal_diagnostics,
            **dict(diagnostics),
            "seller_count": len(seller_uids),
            "pair_count": len(pair_rows),
            "clean_text_corpus_sha256": common.canonical_hash(
                [(seller_uid, clean_texts[seller_uid]) for seller_uid in seller_uids]
            ),
        },
        "input_paths": [profiles_path, signals_path, pair_path],
    }


def load_v7_e5(pool: dict, seller_uids: list[str]) -> tuple[np.ndarray, dict]:
    metadata_path = common.resolve(pool["v7_clean_e5_metadata"])
    matrix_path = common.resolve(pool["v7_clean_e5_matrix"])
    metadata = common.load_json(metadata_path)
    matrix = np.load(matrix_path, mmap_mode="r")
    if metadata.get("identifier_redacted") is not True:
        raise ValueError(f"Frozen v7 E5 cache is not identifier-redacted: {metadata_path}")
    if list(metadata.get("seller_uids", [])) != seller_uids:
        raise ValueError("Frozen v7 E5 seller order differs from the v8 pair universe")
    if list(matrix.shape) != list(metadata.get("shape", [])):
        raise ValueError("Frozen v7 E5 matrix/metadata shape mismatch")
    return matrix, metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", default=str(common.DEFAULT_POLICY))
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--validate-config-only", action="store_true")
    args = parser.parse_args()

    policy_path, policy, v7_policy = common.load_policy(args.policy)
    validation = common.validate_policy_contract(policy, v7_policy)
    if args.validate_config_only:
        print(json.dumps(validation, indent=2))
        return
    run_id = args.run_id or policy["default_run_id"]
    root = common.run_root(policy, run_id)
    final_root = root / policy["clean_semantics"]["output_subdirectory"]
    staging_root = final_root.with_name(f".{final_root.name}.incomplete")
    if final_root.exists() or staging_root.exists():
        raise FileExistsError(
            f"Refusing to overwrite Step15-v8 clean semantics: {final_root} / {staging_root}"
        )
    prepared = {
        pool_name: prepare_clean_texts(policy, pool_name) for pool_name in policy["pools"]
    }
    semantic_policy_path = common.resolve(policy["clean_semantics"]["semantic_model_policy"])
    semantic_policy = common.load_json(semantic_policy_path)
    torch, tokenizer_cls, model_cls, reranker_cls = semantic.require_torch_and_transformers()
    device = semantic.choose_device(torch, semantic_policy["device_preference"], args.device)

    matrices: dict[str, dict[str, np.ndarray]] = {pool_name: {} for pool_name in prepared}
    model_metadata = {}
    for model_key in policy["clean_semantics"]["embedding_model_keys"]:
        model_cfg = semantic_policy["embedding_models"][model_key]
        model_dir = semantic.resolve_local_model_dir(model_key, model_cfg)
        fingerprint = v7_cache.directory_fingerprint(model_dir)
        combined_texts = []
        slices = {}
        start = 0
        for pool_name, item in prepared.items():
            texts = [item["clean_texts"][seller_uid] for seller_uid in item["seller_uids"]]
            combined_texts.extend(texts)
            slices[pool_name] = (start, start + len(texts))
            start += len(texts)
        encoded = semantic.encode_texts(
            model_key,
            model_cfg,
            combined_texts,
            device,
            torch,
            tokenizer_cls,
            model_cls,
        )
        if len(encoded) != len(combined_texts) or not np.all(np.isfinite(encoded)):
            raise ValueError(f"Invalid identifier-redacted embedding matrix: {model_key}")
        for pool_name, (left, right) in slices.items():
            matrices[pool_name][model_key] = np.asarray(encoded[left:right], dtype=np.float32)
        model_metadata[model_key] = {
            "repo_id": model_cfg["repo_id"],
            "local_path": model_cfg["local_path"],
            "directory_fingerprint": fingerprint,
        }

    reranker_key = policy["clean_semantics"]["reranker_model_key"]
    reranker_cfg = semantic_policy["reranker_models"][reranker_key]
    reranker_dir = semantic.resolve_local_model_dir(reranker_key, reranker_cfg)
    model_metadata[reranker_key] = {
        "repo_id": reranker_cfg["repo_id"],
        "local_path": reranker_cfg["local_path"],
        "directory_fingerprint": v7_cache.directory_fingerprint(reranker_dir),
    }
    pair_outputs = {}
    pair_fields = [
        "pair_uid",
        "seller_uid_left",
        "seller_uid_right",
        "embedding_cosine_multilingual_e5_large_identifier_redacted",
        "embedding_cosine_bge_m3_identifier_redacted",
        "embedding_cosine_labse_identifier_redacted",
        "reranker_score_gte_multilingual_reranker_base_identifier_redacted",
        "candidate_rule_count_non_identifier_v8",
        "candidate_rule_hits_non_identifier_v8",
    ]
    allowlist = set(policy["bridge_audit"]["nonidentifier_candidate_rule_allowlist"])
    records = {}
    staging_root.mkdir(parents=True, exist_ok=False)
    for pool_name, item in prepared.items():
        pool_cfg = policy["pools"][pool_name]
        e5_matrix, e5_metadata = load_v7_e5(pool_cfg, item["seller_uids"])
        if e5_metadata.get("clean_text_corpus_sha256") != item["diagnostics"][
            "clean_text_corpus_sha256"
        ]:
            raise ValueError(
                f"V8 redacted text corpus differs from the frozen v7 E5 corpus: {pool_name}"
            )
        e5_scores = cosine_rows(item["pair_rows"], item["seller_uids"], e5_matrix)
        frozen_field = "embedding_cosine_multilingual_e5_large_identifier_redacted"
        maximum_difference = max(
            abs(score - float(row[frozen_field]))
            for score, row in zip(e5_scores, item["pair_rows"], strict=True)
        )
        if maximum_difference > 2e-10:
            raise ValueError(
                f"V8 E5 cosine does not reproduce frozen v7 for {pool_name}: {maximum_difference}"
            )
        bge_scores = cosine_rows(
            item["pair_rows"], item["seller_uids"], matrices[pool_name]["bge_m3"]
        )
        labse_scores = cosine_rows(
            item["pair_rows"], item["seller_uids"], matrices[pool_name]["labse"]
        )
        reversed_pair_rows = [
            {
                **row,
                "seller_uid_left": row["seller_uid_right"],
                "seller_uid_right": row["seller_uid_left"],
            }
            for row in item["pair_rows"]
        ]
        directional_reranker_scores = semantic.reranker_scores(
            item["pair_rows"] + reversed_pair_rows,
            item["clean_texts"],
            reranker_key,
            reranker_cfg,
            device,
            torch,
            tokenizer_cls,
            reranker_cls,
        )
        pair_count = len(item["pair_rows"])
        reranker_scores = [
            0.5 * (forward + reverse)
            for forward, reverse in zip(
                directional_reranker_scores[:pair_count],
                directional_reranker_scores[pair_count:],
                strict=True,
            )
        ]
        candidates = {
            row["pair_uid"]: row
            for row in common.load_csv(common.resolve(pool_cfg["step4_candidates"]))
        }
        if set(candidates) != {row["pair_uid"] for row in item["pair_rows"]}:
            raise ValueError(f"V8 Step4/Step7 pair universes differ: {pool_name}")
        output_rows = []
        for index, row in enumerate(item["pair_rows"]):
            hits = [
                token
                for token in str(candidates[row["pair_uid"]].get("candidate_rule_hits", "")).split("|")
                if token in allowlist
            ]
            output_rows.append(
                {
                    "pair_uid": row["pair_uid"],
                    "seller_uid_left": row["seller_uid_left"],
                    "seller_uid_right": row["seller_uid_right"],
                    frozen_field: f"{e5_scores[index]:.12f}",
                    "embedding_cosine_bge_m3_identifier_redacted": f"{bge_scores[index]:.12f}",
                    "embedding_cosine_labse_identifier_redacted": f"{labse_scores[index]:.12f}",
                    "reranker_score_gte_multilingual_reranker_base_identifier_redacted": f"{reranker_scores[index]:.12f}",
                    "candidate_rule_count_non_identifier_v8": len(set(hits)),
                    "candidate_rule_hits_non_identifier_v8": "|".join(sorted(set(hits))),
                }
            )
        pair_path = staging_root / common.semantic_pair_path(root, pool_name).name
        pair_path.write_bytes(common.render_csv(output_rows, pair_fields))
        pair_outputs[pool_name] = pair_path
        cache_records = {}
        for model_key in policy["clean_semantics"]["embedding_model_keys"]:
            final_metadata, final_matrix = common.semantic_cache_paths(root, pool_name, model_key)
            staged_matrix = staging_root / final_matrix.name
            staged_metadata = staging_root / final_metadata.name
            np.save(staged_matrix, matrices[pool_name][model_key])
            metadata = {
                "model_key": f"{model_key}_identifier_redacted_v8",
                "identifier_redacted": True,
                "seller_uids": item["seller_uids"],
                "shape": list(matrices[pool_name][model_key].shape),
                "matrix_sha256": common.sha256(staged_matrix),
                "clean_text_corpus_sha256": item["diagnostics"]["clean_text_corpus_sha256"],
                "model": model_metadata[model_key],
            }
            staged_metadata.write_text(
                json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
            )
            cache_records[model_key] = {
                "metadata": str(final_metadata.relative_to(ROOT)).replace("\\", "/"),
                "matrix": str(final_matrix.relative_to(ROOT)).replace("\\", "/"),
                "metadata_sha256": common.sha256(staged_metadata),
                "matrix_sha256": common.sha256(staged_matrix),
            }
        records[pool_name] = {
            "pair_count": len(output_rows),
            "seller_count": len(item["seller_uids"]),
            "redaction_diagnostics": item["diagnostics"],
            "frozen_v7_e5_metadata_sha256": common.sha256(
                common.resolve(pool_cfg["v7_clean_e5_metadata"])
            ),
            "frozen_v7_e5_matrix_sha256": common.sha256(
                common.resolve(pool_cfg["v7_clean_e5_matrix"])
            ),
            "frozen_v7_e5_maximum_cosine_difference": maximum_difference,
            "pair_semantics": str(common.semantic_pair_path(root, pool_name).relative_to(ROOT)).replace("\\", "/"),
            "pair_semantics_sha256": common.sha256(pair_path),
            "embedding_caches": cache_records,
        }
    manifest = {
        "step": "step15_build_v8_clean_semantics",
        "version": policy["version"],
        "run_id": run_id,
        "identifier_redacted": True,
        "test_or_valid_statistics_used": False,
        "models": model_metadata,
        "reranker_pair_symmetrization": policy["clean_semantics"][
            "reranker_pair_symmetrization"
        ],
        "pools": records,
        "policy": str(policy_path.relative_to(ROOT)).replace("\\", "/"),
        "policy_sha256": common.sha256(policy_path),
        "semantic_policy_sha256": common.sha256(semantic_policy_path),
        "producer_sha256": common.sha256(Path(__file__).resolve()),
    }
    manifest["manifest_sha256"] = common.canonical_hash(manifest)
    manifest_path = staging_root / "clean_semantics_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    staging_root.replace(final_root)
    print(
        json.dumps(
            {
                "status": "pass",
                "device": str(device),
                "run_id": run_id,
                "manifest": str((final_root / manifest_path.name).relative_to(ROOT)).replace("\\", "/"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
