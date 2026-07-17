#!/usr/bin/env python3
"""Build the three preregistered Step24 train-only pair cosine features."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import step24_build_style_embedding_cache as cache_builder
import step24_common as common


def pair_cosine(row: dict, index: dict[str, int], matrix: np.ndarray) -> float:
    left_uid = row["seller_uid_left"]
    right_uid = row["seller_uid_right"]
    if left_uid not in index or right_uid not in index:
        raise ValueError(f"Step24 pair seller is missing from cache: {row['pair_uid']}")
    return float(np.dot(matrix[index[left_uid]], matrix[index[right_uid]]))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", default=str(common.DEFAULT_POLICY))
    parser.add_argument("--validate-config-only", action="store_true")
    args = parser.parse_args()
    policy_path = common.resolve(args.policy)
    policy = json.loads(policy_path.read_text(encoding="utf-8"))
    common.validate_policy(policy)
    if args.validate_config_only:
        print(
            json.dumps(
                {
                    "status": "pass",
                    "feature_names": list(policy["pair_features"].values())[:3],
                    "numerical_execution_performed": False,
                },
                indent=2,
            )
        )
        return

    output_root = common.resolve(policy["outputs_root"])
    embedding_manifest_path = output_root / policy["outputs"]["embedding_manifest"]
    if not embedding_manifest_path.is_file():
        raise FileNotFoundError(f"Step24 embedding manifest is missing: {embedding_manifest_path}")
    embedding_manifest = json.loads(embedding_manifest_path.read_text(encoding="utf-8"))
    if embedding_manifest.get("encoder_parameters_updated") is not False:
        raise ValueError("Step24 refuses a locally fine-tuned style cache")
    if embedding_manifest.get("valid_test_seller_encoded_count") != 0:
        raise ValueError("Step24 style cache contains valid/test sellers")

    rows_by_pool = common.load_canonical_train_rows(policy)
    feature_cfg = policy["pair_features"]
    feature_names = [
        feature_cfg["identifier_redacted_e5_cosine"],
        feature_cfg["pcm_multilingual_authorship_cosine"],
        feature_cfg["mstyledistance_cosine"],
    ]
    output_paths = {}
    summary_records = {}
    for pool_name, pool_cfg in policy["pools"].items():
        e5_metadata_path = common.resolve(pool_cfg["identifier_redacted_e5_metadata"])
        e5_matrix_path = common.resolve(pool_cfg["identifier_redacted_e5_matrix"])
        e5_index, e5_matrix, e5_metadata = common.load_normalized_cache(
            e5_metadata_path, e5_matrix_path
        )
        if e5_metadata.get("identifier_redacted") is not True:
            raise ValueError(f"Step24 E5 cache is not identifier-redacted: {pool_name}")

        style_caches = {}
        for encoder_key in ("pcm_multilingual_authorship", "mstyledistance"):
            matrix_path, metadata_path = cache_builder.output_paths(
                output_root, encoder_key, pool_name
            )
            index, matrix, metadata = common.load_normalized_cache(metadata_path, matrix_path)
            if metadata.get("encoded_split") != "train" or metadata.get(
                "valid_test_seller_encoded_count"
            ) != 0:
                raise ValueError(f"Step24 style cache split contract failed: {metadata_path}")
            if metadata.get("locally_finetuned") is not False:
                raise ValueError(f"Step24 style encoder was locally fine-tuned: {metadata_path}")
            style_caches[encoder_key] = (index, matrix, metadata_path, matrix_path)

        feature_rows = []
        for row in rows_by_pool[pool_name]:
            values = {
                "identifier_redacted_e5_cosine": pair_cosine(row, e5_index, e5_matrix),
                "pcm_multilingual_authorship_cosine": pair_cosine(
                    row,
                    style_caches["pcm_multilingual_authorship"][0],
                    style_caches["pcm_multilingual_authorship"][1],
                ),
                "mstyledistance_cosine": pair_cosine(
                    row,
                    style_caches["mstyledistance"][0],
                    style_caches["mstyledistance"][1],
                ),
            }
            if any(not np.isfinite(value) or value < -1.0001 or value > 1.0001 for value in values.values()):
                raise ValueError(f"Step24 emitted an invalid cosine: {row['pair_uid']}")
            feature_rows.append(
                {
                    "pair_uid": row["pair_uid"],
                    "pool": pool_name,
                    "domain": row["domain"],
                    "split_name": "train",
                    "seller_uid_left": row["seller_uid_left"],
                    "seller_uid_right": row["seller_uid_right"],
                    **{name: f"{values[name]:.12f}" for name in feature_names},
                }
            )
        if len({row["pair_uid"] for row in feature_rows}) != len(feature_rows):
            raise ValueError(f"Step24 produced duplicate pair features: {pool_name}")
        expected = policy["outputs"][
            "pair_features_en" if pool_name == "en_content_train_pool" else "pair_features_zh"
        ]
        output_path = output_root / expected
        common.write_csv_immutable(output_path, feature_rows)
        output_paths[pool_name] = output_path
        summary_records[pool_name] = {
            "row_count": len(feature_rows),
            "positive_count": sum(
                row["review_label"] == "positive" for row in rows_by_pool[pool_name]
            ),
            "negative_count": sum(
                row["review_label"] == "negative" for row in rows_by_pool[pool_name]
            ),
            "unique_pair_count": len({row["pair_uid"] for row in feature_rows}),
            "unique_seller_count": len(
                {
                    row[field]
                    for row in feature_rows
                    for field in ("seller_uid_left", "seller_uid_right")
                }
            ),
            "output_path": str(output_path.relative_to(common.ROOT)).replace("\\", "/"),
            "output_sha256": common.sha256_file(output_path),
            "e5_metadata_sha256": common.sha256_file(e5_metadata_path),
            "e5_matrix_sha256": common.sha256_file(e5_matrix_path),
            "style_cache_sha256": {
                encoder_key: {
                    "metadata": common.sha256_file(item[2]),
                    "matrix": common.sha256_file(item[3]),
                }
                for encoder_key, item in style_caches.items()
            },
        }

    summary = {
        "step": "step24_build_pair_features",
        "version": policy["version"],
        "feature_names": feature_names,
        "feature_count": len(feature_names),
        "identifier_features_included": False,
        "candidate_rule_features_included": False,
        "random_projection_included": False,
        "synthetic_row_count": 0,
        "valid_test_pair_count": 0,
        "records": summary_records,
        "embedding_manifest_sha256": common.sha256_file(embedding_manifest_path),
        "policy_sha256": common.sha256_file(policy_path),
        "producer_sha256": common.sha256_file(Path(__file__).resolve()),
    }
    summary["summary_sha256"] = common.canonical_hash(summary)
    summary_path = output_root / policy["outputs"]["pair_feature_summary"]
    common.write_json_immutable(summary_path, summary)
    print(
        json.dumps(
            {
                "status": "pass",
                "feature_count": len(feature_names),
                "rows": {key: value["row_count"] for key, value in summary_records.items()},
                "valid_test_pair_count": 0,
                "summary": str(summary_path.relative_to(common.ROOT)).replace("\\", "/"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
