#!/usr/bin/env python3
"""Recompute Step27 real, synthetic, and duplication pair features from profiles."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import step27_common as common


def profile_index(path: Path) -> dict[str, dict]:
    rows = common.load_jsonl(path)
    result = {row["seller_uid"]: row for row in rows}
    if len(result) != len(rows):
        raise ValueError(f"Step27 clean profile UIDs are duplicated: {path}")
    return result


def feature_row(
    pair: dict,
    *,
    profiles: dict[str, dict],
    embedding_index: dict[str, int],
    embedding_matrix,
    fields: list[str],
    row_kind: str,
) -> dict:
    left_uid = pair["seller_uid_left"]
    right_uid = pair["seller_uid_right"]
    if left_uid not in profiles or right_uid not in profiles:
        raise ValueError(f"Step27 pair profile is missing: {pair['pair_uid']}")
    if left_uid not in embedding_index or right_uid not in embedding_index:
        raise ValueError(f"Step27 pair embedding is missing: {pair['pair_uid']}")
    values = common.recompute_pair_features(
        profiles[left_uid],
        profiles[right_uid],
        embedding_matrix[embedding_index[left_uid]],
        embedding_matrix[embedding_index[right_uid]],
        fields,
    )
    return {
        "pair_uid": pair["pair_uid"],
        "row_kind": row_kind,
        "track": pair.get("track", "real"),
        "parent_pair_uid": pair.get("parent_pair_uid", ""),
        "review_label": pair.get("review_label", ""),
        "seller_uid_left": left_uid,
        "seller_uid_right": right_uid,
        "split_name": pair.get("split_name", ""),
        "component_id": pair.get("component_id", ""),
        "fold": pair.get("fold", ""),
        "evidence_type": pair.get("evidence_type", ""),
        "silver_train_only": pair.get("silver_train_only", ""),
        "training_sample_weight": pair.get("training_sample_weight", "1.000000000000"),
        "seed": pair.get("seed", ""),
        "variant_index": pair.get("variant_index", ""),
        "transform": pair.get("transform", ""),
        "synthetic_train_only": pair.get("synthetic_train_only", "0"),
        "synthetic_split_name": (
            "synthetic_train_only" if row_kind == "synthetic_transformed" else ""
        ),
        **{name: f"{values[name]:.12f}" for name in common.FEATURE_NAMES},
    }


def build_real(policy_path: Path, policy: dict, fields: list[str]) -> tuple[Path, dict]:
    parent_manifest = common.parent_root(policy) / "manifest.json"
    canonical_path = common.parent_root(policy) / "canonical_pairs.csv"
    matrix_path, metadata_path = common.profile_cache_paths(policy, None, "real")
    clean_path = matrix_path.parent / "clean_profiles.jsonl"
    cache_manifest = matrix_path.parent / "manifest.json"
    step24_bundle = common.frozen_step24_bundle(policy)
    step24_policy_path = step24_bundle["paths"]["policy"]
    step24_pair_features = step24_bundle["paths"]["zh_pair_features"]
    inputs = [
        policy_path,
        parent_manifest,
        canonical_path,
        clean_path,
        matrix_path,
        metadata_path,
        cache_manifest,
        step24_policy_path,
        step24_bundle["paths"]["sync_manifest"],
        step24_bundle["paths"]["pair_feature_summary"],
        step24_bundle["paths"]["clean_text_manifest"],
        step24_bundle["paths"]["model_artifacts"],
        step24_pair_features,
    ]
    root = common.output_root(policy) / "pair_features" / "real"
    output_path = root / "real_pair_features.csv"
    split_output_paths = {
        split: root / f"real_pair_features.{split}.csv" for split in ("train", "valid", "test")
    }
    summary_path = root / "summary.json"
    manifest_path = root / "manifest.json"
    identity = {
        "stage": "step27_build_pair_features:real",
        "policy_sha256": common.sha256_file(policy_path),
        "producer_sha256": common.sha256_file(Path(__file__).resolve()),
        "common_sha256": common.sha256_file(Path(common.__file__).resolve()),
        "shared_dependency_sha256": common.shared_dependency_hashes(),
        "inputs": common.records_for(inputs),
        "feature_names": list(common.FEATURE_NAMES),
        "parent_pair_features_copied": False,
    }
    existing = common.assert_existing_manifest_identity(manifest_path, identity)
    if existing is not None:
        return manifest_path, {"status": "identical_replay", "row_count": existing.get("row_count")}
    if root.exists() and any(root.iterdir()):
        raise ValueError(f"Incomplete or foreign Step27 real feature root exists: {root}")

    pairs = common.load_csv(canonical_path)
    profiles = profile_index(clean_path)
    embedding_index, matrix, metadata = common.load_normalized_cache(metadata_path, matrix_path)
    if metadata.get("identifier_redacted") is not True or metadata.get("encoder_parameters_updated") is not False:
        raise ValueError("Step27 real E5 cache violates the frozen redacted encoder contract")
    rows = [
        feature_row(
            pair,
            profiles=profiles,
            embedding_index=embedding_index,
            embedding_matrix=matrix,
            fields=fields,
            row_kind="real_canonical",
        )
        for pair in pairs
    ]
    if len({row["pair_uid"] for row in rows}) != len(rows):
        raise ValueError("Step27 real pair feature UIDs are duplicated")
    train_rows = [row for row in rows if row["split_name"] == "train"]
    step24_reference_rows = common.load_csv(step24_pair_features)
    replay_tolerance = float(
        policy.get("statistics", {}).get(
            "real_pair_feature_replay_absolute_tolerance", 5e-13
        )
    )
    if not 0.0 <= replay_tolerance <= 5e-13:
        raise ValueError("Step27 real pair feature replay tolerance is too permissive")
    common.assert_exact_real_pair_feature_replay(
        step24_reference_rows,
        train_rows,
        ["identifier_redacted_e5_cosine"],
        atol=replay_tolerance,
    )
    common.write_csv_immutable(output_path, rows)
    split_counts = Counter(row["split_name"] for row in rows)
    for split, split_path in split_output_paths.items():
        split_rows = [row for row in rows if row["split_name"] == split]
        if not split_rows:
            raise ValueError(f"Step27 real pair feature split is empty: {split}")
        common.write_csv_immutable(split_path, split_rows)
    summary = {
        "step": "step27_build_pair_features",
        "scope": "real_canonical",
        "row_count": len(rows),
        "split_counts": dict(sorted(split_counts.items())),
        "label_counts": dict(sorted(Counter(row["review_label"] for row in rows).items())),
        "feature_names": list(common.FEATURE_NAMES),
        "feature_count": len(common.FEATURE_NAMES),
        "identifier_features_included": False,
        "parent_pair_features_copied": False,
        "all_features_recomputed_from_clean_profiles": True,
        "step24_real_e5_pair_feature_replay_verified": True,
        "step24_real_e5_pair_feature_replay_tolerance": replay_tolerance,
        "valid_or_test_fitted_statistics_used": False,
        "output": common.relative(output_path),
        "split_outputs": {
            split: common.relative(path) for split, path in split_output_paths.items()
        },
    }
    summary["summary_content_sha256"] = common.canonical_hash(summary)
    common.write_json_immutable(summary_path, summary)
    common.write_manifest_immutable(
        manifest_path,
        stage="step27_build_pair_features:real",
        identity=identity,
        inputs=inputs,
        outputs=[output_path, *split_output_paths.values(), summary_path],
        extra={"row_count": len(rows), "summary_sha256": common.sha256_file(summary_path)},
    )
    return manifest_path, summary


def validate_synthetic_lineage(pair: dict, profiles: dict[str, dict]) -> None:
    for field in ("seller_uid_left", "seller_uid_right"):
        profile = profiles.get(pair[field])
        if profile is None:
            raise ValueError(f"Step27 synthetic pair profile is missing: {pair['pair_uid']}")
        lineage = profile.get("synthetic_lineage", {})
        expected = {
            "parent_pair_uid": pair["parent_pair_uid"],
            "parent_component_id": pair["component_id"],
            "inherited_fold": int(pair["fold"]),
            "inherited_label": pair["review_label"],
            "track": pair["track"],
            "seed": int(pair["seed"]),
            "variant_index": int(pair["variant_index"]),
        }
        if any(lineage.get(key) != value for key, value in expected.items()):
            raise ValueError(f"Step27 synthetic lineage mismatch: {pair['pair_uid']}:{field}")
        if lineage.get("fabricated_identifier_count") != 0 or lineage.get(
            "fabricated_market_or_provenance"
        ) is not False:
            raise ValueError(f"Step27 synthetic lineage reports fabricated evidence: {pair['pair_uid']}")


def build_track(
    policy_path: Path,
    policy: dict,
    fields: list[str],
    seed: int,
    track: str,
) -> tuple[Path, dict]:
    root = common.track_root(policy, seed, track) / "pair_features"
    generation_manifest = common.seed_root(policy, seed) / "generation_manifest.json"
    pairs_path = common.track_root(policy, seed, track) / "synthetic_pairs.csv"
    duplication_path = common.track_root(policy, seed, track) / "equal_weight_duplication_pairs.csv"
    synthetic_matrix, synthetic_metadata = common.profile_cache_paths(policy, seed, track)
    synthetic_clean = synthetic_matrix.parent / "clean_profiles.jsonl"
    synthetic_cache_manifest = synthetic_matrix.parent / "manifest.json"
    real_matrix, real_metadata = common.profile_cache_paths(policy, None, "real")
    real_clean = real_matrix.parent / "clean_profiles.jsonl"
    real_cache_manifest = real_matrix.parent / "manifest.json"
    inputs = [
        policy_path,
        generation_manifest,
        pairs_path,
        duplication_path,
        synthetic_matrix,
        synthetic_metadata,
        synthetic_clean,
        synthetic_cache_manifest,
        real_matrix,
        real_metadata,
        real_clean,
        real_cache_manifest,
    ]
    output_synthetic = root / "synthetic_pair_features.csv"
    output_duplication = root / "equal_weight_duplication_pair_features.csv"
    summary_path = root / "summary.json"
    manifest_path = root / "manifest.json"
    identity = {
        "stage": "step27_build_pair_features:synthetic",
        "seed": seed,
        "track": track,
        "policy_sha256": common.sha256_file(policy_path),
        "producer_sha256": common.sha256_file(Path(__file__).resolve()),
        "common_sha256": common.sha256_file(Path(common.__file__).resolve()),
        "shared_dependency_sha256": common.shared_dependency_hashes(),
        "inputs": common.records_for(inputs),
        "feature_names": list(common.FEATURE_NAMES),
        "parent_pair_features_copied": False,
    }
    existing = common.assert_existing_manifest_identity(manifest_path, identity)
    if existing is not None:
        return manifest_path, {"status": "identical_replay", "row_count": existing.get("row_count")}
    if root.exists() and any(root.iterdir()):
        raise ValueError(f"Incomplete or foreign Step27 feature root exists: {root}")

    synthetic_pairs = common.load_csv(pairs_path)
    duplication_pairs = common.load_csv(duplication_path)
    synthetic_profiles = profile_index(synthetic_clean)
    real_profiles = profile_index(real_clean)
    synthetic_index, synthetic_embeddings, synthetic_meta = common.load_normalized_cache(
        synthetic_metadata, synthetic_matrix
    )
    real_index, real_embeddings, real_meta = common.load_normalized_cache(real_metadata, real_matrix)
    for metadata in (synthetic_meta, real_meta):
        if metadata.get("identifier_redacted") is not True or metadata.get(
            "encoder_parameters_updated"
        ) is not False:
            raise ValueError("Step27 pair feature input violates the frozen redacted encoder contract")
    if synthetic_embeddings.shape[1] != real_embeddings.shape[1]:
        raise ValueError("Step27 synthetic and real E5 cache dimensions disagree")
    for pair in synthetic_pairs:
        validate_synthetic_lineage(pair, synthetic_profiles)
    synthetic_rows = [
        feature_row(
            pair,
            profiles=synthetic_profiles,
            embedding_index=synthetic_index,
            embedding_matrix=synthetic_embeddings,
            fields=fields,
            row_kind="synthetic_transformed",
        )
        for pair in synthetic_pairs
    ]
    duplication_rows = [
        feature_row(
            pair,
            profiles=real_profiles,
            embedding_index=real_index,
            embedding_matrix=real_embeddings,
            fields=fields,
            row_kind="equal_effective_weight_duplication",
        )
        for pair in duplication_pairs
    ]
    if len(synthetic_rows) != len(duplication_rows):
        raise ValueError("Step27 transformed and duplication feature counts disagree")
    if Counter(row["review_label"] for row in synthetic_rows) != Counter(
        row["review_label"] for row in duplication_rows
    ):
        raise ValueError("Step27 transformed and duplication label budgets disagree")
    transformed_weights = sum(float(row["training_sample_weight"]) for row in synthetic_rows)
    duplication_weights = sum(float(row["training_sample_weight"]) for row in duplication_rows)
    if abs(transformed_weights - duplication_weights) > 1e-9:
        raise ValueError("Step27 transformed and duplication effective weights disagree")
    common.write_csv_immutable(output_synthetic, synthetic_rows)
    common.write_csv_immutable(output_duplication, duplication_rows)
    summary = {
        "step": "step27_build_pair_features",
        "scope": "synthetic_train_only",
        "seed": seed,
        "track": track,
        "row_count": len(synthetic_rows),
        "label_counts": dict(sorted(Counter(row["review_label"] for row in synthetic_rows).items())),
        "fold_counts": dict(sorted(Counter(row["fold"] for row in synthetic_rows).items())),
        "feature_names": list(common.FEATURE_NAMES),
        "feature_count": len(common.FEATURE_NAMES),
        "identifier_features_included": False,
        "parent_pair_features_copied": False,
        "all_synthetic_features_recomputed_from_synthetic_profiles": True,
        "equal_effective_weight_duplication_control": True,
        "transformed_effective_weight": transformed_weights,
        "duplication_effective_weight": duplication_weights,
        "valid_or_test_synthetic_count": 0,
        "outputs": {
            "synthetic": common.relative(output_synthetic),
            "duplication": common.relative(output_duplication),
        },
    }
    summary["summary_content_sha256"] = common.canonical_hash(summary)
    common.write_json_immutable(summary_path, summary)
    common.write_manifest_immutable(
        manifest_path,
        stage="step27_build_pair_features:synthetic",
        identity=identity,
        inputs=inputs,
        outputs=[output_synthetic, output_duplication, summary_path],
        extra={"row_count": len(synthetic_rows), "summary_sha256": common.sha256_file(summary_path)},
    )
    return manifest_path, summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", default=str(common.DEFAULT_POLICY))
    parser.add_argument("--seed", action="append", type=int, dest="seeds")
    parser.add_argument("--track", action="append", choices=["primary", "silver_sensitivity"], dest="tracks")
    parser.add_argument("--validate-config-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    policy_path, policy = common.load_policy(args.policy)
    fields = common.text_fields(policy)
    seeds = args.seeds or common.generation_seeds(policy)
    tracks = args.tracks or ["primary", "silver_sensitivity"]
    if args.validate_config_only:
        print(
            json.dumps(
                {
                    "status": "pass",
                    "feature_names": list(common.FEATURE_NAMES),
                    "seeds": seeds,
                    "tracks": tracks,
                    "parent_pair_features_copied": False,
                    "numerical_execution_performed": False,
                },
                indent=2,
            )
        )
        return
    real_manifest, real_summary = build_real(policy_path, policy, fields)
    records = []
    for seed in seeds:
        for track in tracks:
            manifest, summary = build_track(policy_path, policy, fields, seed, track)
            records.append(
                {
                    "seed": seed,
                    "track": track,
                    "manifest": common.relative(manifest),
                    "row_count": summary.get("row_count"),
                    "status": summary.get("status", "pass"),
                }
            )
    print(
        json.dumps(
            {
                "status": "pass",
                "real_manifest": common.relative(real_manifest),
                "real_row_count": real_summary.get("row_count"),
                "synthetic_tracks": records,
                "feature_names": list(common.FEATURE_NAMES),
                "parent_pair_features_copied": False,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
