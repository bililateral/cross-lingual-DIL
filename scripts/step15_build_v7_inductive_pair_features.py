#!/usr/bin/env python3
"""Build isolated Step15-v7 pair features with train-only, OOV-safe references."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from collections import Counter
from pathlib import Path

import numpy as np

import step15_build_v6_inductive_pair_features as v6
import step15_v7_common as common


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_POLICY = ROOT / "schema" / "step15_v7_two_stage_policy.json"
DIAGNOSTIC_FIELDS = [
    "shared_oov_title_count_diagnostic",
    "shared_oov_description_count_diagnostic",
]


def cosine(left: np.ndarray, right: np.ndarray) -> float:
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    return 0.0 if denominator <= 0.0 else float(np.dot(left, right) / denominator)


def resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_hash(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, ensure_ascii=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def reference_signature_state(
    profile: dict,
    reference: dict,
    signature_key: str,
    df_key: str,
    oov_cfg: dict,
) -> dict:
    cfg = reference["boilerplate_config"]
    minimum_length = int(cfg.get("minimum_signature_value_length", 1))
    norms = v6.signature_norms(profile, signature_key, minimum_length)
    seller_count = int(reference["train_seller_count"])
    df_map = reference[df_key]
    df_floor = int(oov_cfg.get("shared_signature_effective_df_floor", 2))
    if df_floor < 2:
        raise ValueError("Shared-signature OOV effective df floor must be at least two")
    boilerplate_threshold = float(cfg.get("boilerplate_seller_share_threshold", 0.01))
    rare_threshold = float(cfg.get("rare_seller_share_threshold", 0.002))
    idf: dict[str, float] = {}
    boilerplate: set[str] = set()
    rare: set[str] = set()
    oov: set[str] = set()
    for norm in norms:
        observed_df = int(df_map.get(norm, 0))
        known = observed_df > 0
        effective_df = max(observed_df, df_floor)
        idf[norm] = math.log((seller_count + 1) / (effective_df + 1)) + 1.0
        if not known:
            oov.add(norm)
            continue
        share = observed_df / max(seller_count, 1)
        if share >= boilerplate_threshold:
            boilerplate.add(norm)
        if share <= rare_threshold:
            rare.add(norm)
    return {
        "norms": norms,
        "idf": idf,
        "boilerplate": boilerplate,
        "rare": rare,
        "oov": oov,
    }


def derive_reference_fields(
    left: dict,
    right: dict,
    reference: dict,
    numeric_paths: dict[str, str],
    oov_cfg: dict,
) -> dict:
    lt = reference_signature_state(left, reference, "signature_titles", "title_df", oov_cfg)
    rt = reference_signature_state(right, reference, "signature_titles", "title_df", oov_cfg)
    ld = reference_signature_state(
        left, reference, "signature_description_segments", "description_df", oov_cfg
    )
    rd = reference_signature_state(
        right, reference, "signature_description_segments", "description_df", oov_cfg
    )
    shared_title = lt["norms"] & rt["norms"]
    shared_description = ld["norms"] & rd["norms"]
    title_idf = [lt["idf"][norm] for norm in sorted(shared_title)]
    description_idf = [ld["idf"][norm] for norm in sorted(shared_description)]
    shared_boilerplate = {
        norm for norm in shared_title if norm in lt["boilerplate"] or norm in rt["boilerplate"]
    } | {
        norm
        for norm in shared_description
        if norm in ld["boilerplate"] or norm in rd["boilerplate"]
    }
    left_norm_count = len(lt["norms"]) + len(ld["norms"])
    right_norm_count = len(rt["norms"]) + len(rd["norms"])
    left_boilerplate_count = len(lt["boilerplate"]) + len(ld["boilerplate"])
    right_boilerplate_count = len(rt["boilerplate"]) + len(rd["boilerplate"])
    left_ratio = left_boilerplate_count / max(left_norm_count, 1) if left_norm_count else 0.0
    right_ratio = right_boilerplate_count / max(right_norm_count, 1) if right_norm_count else 0.0
    result: dict[str, float | str] = {
        "boilerplate_ratio_max": max(left_ratio, right_ratio),
        "boilerplate_ratio_gap_abs": abs(left_ratio - right_ratio),
        "shared_title_idf_sum": sum(title_idf),
        "shared_description_idf_sum": sum(description_idf),
        "shared_title_idf_mean": sum(title_idf) / len(title_idf) if title_idf else 0.0,
        "shared_description_idf_mean": (
            sum(description_idf) / len(description_idf) if description_idf else 0.0
        ),
        "shared_boilerplate_count": len(shared_boilerplate),
        "shared_low_df_sentence_count": len(
            {norm for norm in shared_description if norm in ld["rare"] or norm in rd["rare"]}
        ),
        "shared_rare_ngram_count": len(
            {norm for norm in shared_title if norm in lt["rare"] or norm in rt["rare"]}
        ),
        "shared_oov_title_count_diagnostic": len(
            {norm for norm in shared_title if norm in lt["oov"] or norm in rt["oov"]}
        ),
        "shared_oov_description_count_diagnostic": len(
            {norm for norm in shared_description if norm in ld["oov"] or norm in rd["oov"]}
        ),
    }
    minimum_market_size = int(reference.get("minimum_market_group_size", v6.preview.MIN_MARKET_GROUP_SIZE))
    market_values = reference["market_numeric_values"]
    domain_values = reference["domain_numeric_values"]
    for feature_name, dotted_path in numeric_paths.items():
        left_market = str(left.get("source_market_raw", ""))
        right_market = str(right.get("source_market_raw", ""))
        left_reference = market_values.get(left_market, {}).get(feature_name, [])
        right_reference = market_values.get(right_market, {}).get(feature_name, [])
        if len(left_reference) < minimum_market_size:
            left_reference = domain_values.get(feature_name, [])
        if len(right_reference) < minimum_market_size:
            right_reference = domain_values.get(feature_name, [])
        left_pct = v6.percentile(left_reference, v6.numeric_value(left, dotted_path))
        right_pct = v6.percentile(right_reference, v6.numeric_value(right, dotted_path))
        result[f"{feature_name}_percentile_gap_abs"] = (
            "" if left_pct is None or right_pct is None else abs(left_pct - right_pct)
        )
    return {
        key: (round(float(value), 6) if value != "" else "") for key, value in result.items()
    }


def write_fail_closed(path: Path, payload: bytes, allow_identical_replay: bool) -> str:
    expected = hashlib.sha256(payload).hexdigest()
    if path.exists():
        observed = sha256(path)
        if allow_identical_replay and observed == expected:
            return "identical_replay_noop"
        raise FileExistsError(f"Refusing to overwrite Step15-v7 feature artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_bytes(payload)
    temporary.replace(path)
    return "created"


def eligible(row: dict) -> bool:
    return (
        row.get("review_label") in {"positive", "negative"}
        and row.get("usable_for_supervision") == "1"
        and row.get("usable_for_core_transfer") == "1"
    )


def numeric_feature_stats(rows: list[dict], feature_names: list[str]) -> dict:
    result = {}
    for name in feature_names:
        observed = []
        missing_count = 0
        for row in rows:
            value = str(row.get(name, "")).strip()
            if value == "":
                missing_count += 1
                continue
            observed.append(float(value))
        if not observed:
            raise ValueError(f"Strict-clean feature {name} is entirely missing on combined train")
        imputation = float(statistics.median(observed))
        values = observed + [imputation] * missing_count
        unique = sorted({round(value, 12) for value in values})
        result[name] = {
            "min": min(values),
            "max": max(values),
            "unique_count_after_train_median_imputation": len(unique),
            "missing_count": missing_count,
            "train_median_imputation": imputation,
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", default=str(DEFAULT_POLICY))
    parser.add_argument("--validate-config-only", action="store_true")
    parser.add_argument("--validate-data-only", action="store_true")
    parser.add_argument("--allow-identical-replay", action="store_true")
    args = parser.parse_args()

    policy_path = resolve(args.policy)
    policy = json.loads(policy_path.read_text(encoding="utf-8"))
    cfg = policy["inductive_features"]
    stable_features = list(cfg["stable_strict_clean_features"])
    expected_feature_count = int(cfg["stable_strict_clean_feature_count"])
    removed = set(cfg["removed_from_strict_clean"])
    if len(stable_features) != expected_feature_count or removed & set(stable_features):
        raise ValueError("Step15-v7 stable strict-clean feature contract is inconsistent")
    if args.validate_config_only:
        print(json.dumps({"status": "pass", "feature_count": len(stable_features)}, indent=2))
        return

    assignment_path = resolve(policy["representative_validation"]["split_assignment_output"])
    if not assignment_path.is_file():
        raise FileNotFoundError(
            "Representative validation assignments are required before v7 feature fitting: "
            f"{assignment_path}"
        )
    assignments, _ = v6.load_csv(assignment_path)
    assignment_index = {row["pair_uid"]: row for row in assignments}
    schema_path = resolve(cfg["step7_feature_schema"])
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    numeric_paths = dict(schema["market_relative_numeric_fields"])
    boilerplate_cfg = dict(schema["boilerplate_feature_config"])
    semantic_fields = list(schema["feature_views"]["future_multilingual_semantics"])
    records = {}
    references = {
        "step": "step15_v7_inductive_pair_feature_reference",
        "version": policy["version"],
        "reference_scope": cfg["reference_scope"],
        "oov_policy": cfg["oov_policy"],
        "removed_from_strict_clean": sorted(removed),
        "labels_used_for_reference_values": False,
        "domains": {},
    }
    pending: list[tuple[Path, bytes, str]] = []
    combined_train_rows: list[dict] = []
    for pool_name, pool_cfg in policy["pools"].items():
        labels_path = resolve(pool_cfg["frozen_labels"])
        profiles_path = resolve(pool_cfg["seller_profiles"])
        features_path = resolve(pool_cfg["canonical_pair_features"])
        candidates_path = resolve(pool_cfg["step4_candidates"])
        output_path = resolve(pool_cfg["v7_pair_features"])
        labels, _ = v6.load_csv(labels_path)
        profiles_list = v6.load_jsonl(profiles_path)
        feature_rows, feature_fields = v6.load_csv(features_path)
        candidates, _ = v6.load_csv(candidates_path)
        seller_index, clean_embeddings, clean_embedding_metadata = common.load_embedding_index(
            pool_cfg
        )
        supervised = [row for row in labels if eligible(row)]
        split_by_uid = {}
        for row in supervised:
            if pool_name == "zh_target_strict":
                assignment = assignment_index.get(row["pair_uid"])
                if assignment is None:
                    raise ValueError(f"Missing v7 split assignment for {row['pair_uid']}")
                split_by_uid[row["pair_uid"]] = assignment["v7_split_name"]
            else:
                split_by_uid[row["pair_uid"]] = row["split_name"]
        train_rows = [row for row in supervised if split_by_uid[row["pair_uid"]] == "train"]
        train_sellers = {
            str(row[key])
            for row in train_rows
            for key in ("seller_uid_left", "seller_uid_right")
            if str(row.get(key, "")).strip()
        }
        profiles = {str(row["seller_uid"]): row for row in profiles_list}
        reference = v6.fit_reference(profiles, train_sellers, numeric_paths, boilerplate_cfg)
        reference["minimum_market_group_size"] = v6.preview.MIN_MARKET_GROUP_SIZE
        candidate_index = {str(row["pair_uid"]): row for row in candidates}
        if {str(row["pair_uid"]) for row in feature_rows} != set(candidate_index):
            raise ValueError(f"{pool_name}: canonical Step7 and Step4 pair universes differ")
        output_rows = []
        non_identifier_sources: Counter[str] = Counter()
        for row in feature_rows:
            pair_uid = str(row["pair_uid"])
            left_uid = str(row["seller_uid_left"])
            right_uid = str(row["seller_uid_right"])
            if left_uid not in profiles or right_uid not in profiles:
                raise ValueError(f"{pool_name}: profile missing for {pair_uid}")
            output = dict(row)
            output.update(
                derive_reference_fields(
                    profiles[left_uid], profiles[right_uid], reference, numeric_paths, cfg["oov_policy"]
                )
            )
            candidate = candidate_index[pair_uid]
            non_identifier = str(candidate.get("candidate_rule_count_non_identifier", "")).strip()
            if not non_identifier:
                excluded = {
                    "shared_contact_exact",
                    "shared_pgp_fingerprint",
                    "shared_pgp_fingerprint_via_aux_alias",
                }
                non_identifier = str(
                    sum(
                        1
                        for rule in str(candidate.get("candidate_rule_hits", "")).split("|")
                        if rule and rule not in excluded
                    )
                )
                non_identifier_sources["derived_from_candidate_rule_hits"] += 1
            else:
                non_identifier_sources["materialized_step4_field"] += 1
            output["candidate_rule_count_non_identifier"] = non_identifier
            if left_uid not in seller_index or right_uid not in seller_index:
                raise ValueError(f"{pool_name}: clean E5 cache missing seller for {pair_uid}")
            output[policy["clean_semantic_encoder"]["output_feature"]] = round(
                cosine(
                    np.asarray(clean_embeddings[seller_index[left_uid]], dtype=float),
                    np.asarray(clean_embeddings[seller_index[right_uid]], dtype=float),
                ),
                12,
            )
            output_rows.append(output)
        output_fields = list(feature_fields)
        insertion = (
            output_fields.index("sparse_lexical_similarity_raw")
            if "sparse_lexical_similarity_raw" in output_fields
            else len(output_fields)
        )
        if "candidate_rule_count_non_identifier" not in output_fields:
            output_fields.insert(insertion, "candidate_rule_count_non_identifier")
        clean_semantic_field = policy["clean_semantic_encoder"]["output_feature"]
        if clean_semantic_field not in output_fields:
            output_fields.append(clean_semantic_field)
        for field in DIAGNOSTIC_FIELDS:
            if field not in output_fields:
                output_fields.append(field)
        output_index = {row["pair_uid"]: row for row in output_rows}
        for train_row in train_rows:
            combined_train_rows.append(output_index[train_row["pair_uid"]])
        semantic_before = v6.selected_column_hash(feature_rows, semantic_fields)
        semantic_after = v6.selected_column_hash(output_rows, semantic_fields)
        if semantic_before != semantic_after:
            raise ValueError(f"{pool_name}: semantic values changed in v7 builder")
        payload = v6.render_csv(output_rows, output_fields)
        pending.append((output_path, payload, pool_name))
        references["domains"][pool_name] = reference
        records[pool_name] = {
            "pair_count": len(output_rows),
            "train_pair_count": len(train_rows),
            "train_seller_count": len(train_sellers),
            "output_path": str(output_path.relative_to(ROOT)).replace("\\", "/"),
            "output_sha256": hashlib.sha256(payload).hexdigest(),
            "semantic_values_preserved": True,
            "legacy_profile_text_semantic_features_are_diagnostic_only": True,
            "clean_semantic_feature": clean_semantic_field,
            "clean_embedding_cache_identifier_redacted": clean_embedding_metadata.get(
                "identifier_redacted"
            )
            is True,
            "candidate_rule_count_non_identifier_sources": dict(non_identifier_sources),
            "shared_oov_pair_counts": {
                field: sum(float(row.get(field, 0) or 0) > 0 for row in output_rows)
                for field in DIAGNOSTIC_FIELDS
            },
            "inputs": {
                str(path.relative_to(ROOT)).replace("\\", "/"): sha256(path)
                for path in (
                    labels_path,
                    profiles_path,
                    features_path,
                    candidates_path,
                    resolve(pool_cfg["clean_e5_cache_metadata"]),
                    resolve(pool_cfg["clean_e5_cache_matrix"]),
                )
            },
        }

    feature_stats = numeric_feature_stats(combined_train_rows, stable_features)
    constants = [
        name
        for name, stats in feature_stats.items()
        if stats["unique_count_after_train_median_imputation"] <= 1
    ]
    if constants:
        raise ValueError(f"Configured v7 strict-clean features are constant on combined train: {constants}")
    references["reference_sha256"] = canonical_hash(references)
    reference_path = resolve(cfg["reference_bundle_output"])
    reference_payload = (json.dumps(references, indent=2, ensure_ascii=False) + "\n").encode("utf-8")
    manifest = {
        "step": "step15_build_v7_inductive_pair_features",
        "version": policy["version"],
        "reference_scope": cfg["reference_scope"],
        "oov_policy": cfg["oov_policy"],
        "strict_clean_feature_count": len(stable_features),
        "strict_clean_features": stable_features,
        "removed_constant_or_oov_only_features": sorted(removed),
        "strict_clean_constant_features_on_combined_train": constants,
        "strict_clean_train_stats": feature_stats,
        "current_internal_test_used_to_fit_reference": False,
        "representative_validation_used_to_fit_reference": False,
        "assignment_manifest": policy["representative_validation"]["manifest_output"],
        "assignment_csv_sha256": sha256(assignment_path),
        "reference_bundle": str(reference_path.relative_to(ROOT)).replace("\\", "/"),
        "reference_bundle_sha256": hashlib.sha256(reference_payload).hexdigest(),
        "domains": records,
        "policy": str(policy_path.relative_to(ROOT)).replace("\\", "/"),
        "policy_sha256": sha256(policy_path),
        "producer_sha256": sha256(Path(__file__).resolve()),
    }
    manifest["manifest_sha256"] = canonical_hash(manifest)
    manifest_path = resolve(cfg["manifest_output"])
    manifest_payload = (json.dumps(manifest, indent=2, ensure_ascii=False) + "\n").encode("utf-8")
    if args.validate_data_only:
        print(json.dumps({"status": "pass", "manifest": manifest}, indent=2, ensure_ascii=False))
        return
    publication_root = reference_path.parent
    final_payloads = [
        *[(path, payload, pool) for path, payload, pool in pending],
        (reference_path, reference_payload, "reference_bundle"),
        (manifest_path, manifest_payload, "manifest"),
    ]
    if any(path.parent != publication_root for path, _, _ in final_payloads):
        raise ValueError("Step15-v7 feature outputs must share one publication directory")
    if publication_root.exists():
        identical = all(
            path.is_file() and sha256(path) == hashlib.sha256(payload).hexdigest()
            for path, payload, _ in final_payloads
        )
        if args.allow_identical_replay and identical:
            actions = {name: "identical_replay_noop" for _, _, name in final_payloads}
            print(json.dumps({"status": "pass", "actions": actions, "manifest": manifest}, indent=2))
            return
        raise FileExistsError(f"Refusing to overwrite Step15-v7 feature directory: {publication_root}")
    staging_root = publication_root.with_name(f".{publication_root.name}.incomplete")
    if staging_root.exists():
        raise FileExistsError(f"Incomplete Step15-v7 feature directory exists: {staging_root}")
    actions = {}
    for path, payload, name in final_payloads:
        staged_path = staging_root / path.relative_to(publication_root)
        actions[name] = write_fail_closed(staged_path, payload, False)
    staging_root.replace(publication_root)
    print(json.dumps({"status": "pass", "actions": actions, "manifest": manifest}, indent=2))


if __name__ == "__main__":
    main()
