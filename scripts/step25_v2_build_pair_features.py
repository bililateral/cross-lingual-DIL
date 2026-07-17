#!/usr/bin/env python3
"""Build Step25-v2 raw/global/pair-local style features without labels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import step24_common as step24
import step25_common as step25_v1
import step25_v2_common as common


FEATURE_NAMES = [
    "raw_pcm_multilingual_authorship_cosine",
    "raw_mstyledistance_cosine",
    "global_clean_pcm_multilingual_authorship_cosine",
    "global_clean_mstyledistance_cosine",
    "pair_local_clean_pcm_multilingual_authorship_cosine",
    "pair_local_clean_mstyledistance_cosine",
    "pair_local_or_raw_pcm_multilingual_authorship_cosine",
    "pair_local_or_raw_mstyledistance_cosine",
    "pair_local_style_reliable",
    "global_style_reliable",
    "global_and_pair_local_style_reliable",
    "pair_local_maximum_mask_fraction",
    "pair_local_mean_mask_fraction",
    "pair_local_shared_shingle_count_log1p",
]


def load_index(path: Path, key: str) -> dict[str, dict]:
    rows = step24.load_csv(path)
    index = {row[key]: row for row in rows}
    if len(index) != len(rows):
        raise ValueError(f"Step25-v2 duplicate {key}: {path}")
    return index


def cosine(
    pair_uid: str,
    index: dict[str, int],
    matrix: np.ndarray,
) -> float:
    left = common.pair_side_key(pair_uid, "left")
    right = common.pair_side_key(pair_uid, "right")
    if left not in index or right not in index:
        raise ValueError(f"Step25-v2 pair side is missing from embedding cache: {pair_uid}")
    return float(np.dot(np.asarray(matrix[index[left]]), np.asarray(matrix[index[right]])))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", default=str(common.DEFAULT_POLICY))
    parser.add_argument("--validate-config-only", action="store_true")
    args = parser.parse_args()
    policy_path, policy, step24_policy, step25_v1_policy = common.load_policy(args.policy)
    if args.validate_config_only:
        print(
            json.dumps(
                {
                    "status": "pass",
                    "feature_names": FEATURE_NAMES,
                    "missing_pair_local_style_persisted_as": "nan",
                    "fixed_zero_missingness_forbidden": True,
                    "valid_test_pair_count": 0,
                    "numerical_execution_performed": False,
                },
                indent=2,
            )
        )
        return

    output_root = common.resolve(policy["outputs_root"])
    step24_root = common.resolve(policy["inputs"]["step24_outputs_root"])
    step25_v1_root = common.resolve(policy["inputs"]["step25_v1_outputs_root"])
    embedding_manifest_path = output_root / policy["outputs"]["embedding_manifest"]
    embedding_manifest = json.loads(embedding_manifest_path.read_text(encoding="utf-8"))
    if embedding_manifest.get("valid_test_pair_side_count") != 0:
        raise ValueError("Step25-v2 embedding manifest contains valid/test sides")
    rows_by_pool = common.load_rows(policy, step24_policy, step25_v1_policy)
    output_records = {}
    for pool_name, rows in rows_by_pool.items():
        parent_key = "pair_features_en" if pool_name == "en_content_train_pool" else "pair_features_zh"
        raw_path = step24_root / step24_policy["outputs"][parent_key]
        global_path = step25_v1_root / step25_v1_policy["outputs"][parent_key]
        raw_index = load_index(raw_path, "pair_uid")
        global_index = load_index(global_path, "pair_uid")
        text_path = common.pair_text_path(policy, pool_name)
        text_rows = step25_v1.load_jsonl(text_path)
        text_index = {row["pair_uid"]: row for row in text_rows}
        if len(text_index) != len(text_rows):
            raise ValueError(f"Step25-v2 duplicate pair-local text: {pool_name}")
        embedding_indexes = {}
        embedding_matrices = {}
        for encoder_key in policy["frozen_style_encoders"]:
            matrix_path, metadata_path = common.embedding_paths(policy, encoder_key, pool_name)
            index, matrix, metadata = common.load_pair_embedding_cache(
                metadata_path, matrix_path
            )
            if (
                metadata.get("pair_local_decontaminated") is not True
                or metadata.get("valid_test_pair_side_count") != 0
            ):
                raise ValueError(
                    f"Step25-v2 pair-local embedding isolation failed: {encoder_key}:{pool_name}"
                )
            embedding_indexes[encoder_key] = index
            embedding_matrices[encoder_key] = matrix

        pair_rows = []
        local_reliable_count = 0
        global_reliable_count = 0
        intersection_count = 0
        for row in rows:
            pair_uid = row["pair_uid"]
            raw = raw_index.get(pair_uid)
            global_row = global_index.get(pair_uid)
            text = text_index.get(pair_uid)
            if raw is None or global_row is None or text is None:
                raise ValueError(f"Step25-v2 parent feature/text is missing: {pair_uid}")
            raw_pcm = float(raw["pcm_multilingual_authorship_cosine"])
            raw_mstyle = float(raw["mstyledistance_cosine"])
            local_pcm_value = cosine(
                pair_uid,
                embedding_indexes["pcm_multilingual_authorship"],
                embedding_matrices["pcm_multilingual_authorship"],
            )
            local_mstyle_value = cosine(
                pair_uid,
                embedding_indexes["mstyledistance"],
                embedding_matrices["mstyledistance"],
            )
            local_reliable = int(text["pair_reliable"])
            global_reliable = int(global_row["decontaminated_pair_reliable"])
            intersection = int(local_reliable and global_reliable)
            local_reliable_count += local_reliable
            global_reliable_count += global_reliable
            intersection_count += intersection
            local_pcm = local_pcm_value if local_reliable else float("nan")
            local_mstyle = local_mstyle_value if local_reliable else float("nan")
            global_pcm = (
                float(global_row["decontaminated_pcm_multilingual_authorship_cosine"])
                if intersection
                else float("nan")
            )
            global_mstyle = (
                float(global_row["decontaminated_mstyledistance_cosine"])
                if intersection
                else float("nan")
            )
            left_fraction = float(text["left_mask_fraction"])
            right_fraction = float(text["right_mask_fraction"])
            pair_rows.append(
                {
                    "pair_uid": pair_uid,
                    "pool": pool_name,
                    "domain": row["domain"],
                    "split_name": "train",
                    "seller_uid_left": row["seller_uid_left"],
                    "seller_uid_right": row["seller_uid_right"],
                    "component_id": row["step25_component_id"],
                    "raw_pcm_multilingual_authorship_cosine": f"{raw_pcm:.12f}",
                    "raw_mstyledistance_cosine": f"{raw_mstyle:.12f}",
                    "global_clean_pcm_multilingual_authorship_cosine": f"{global_pcm:.12f}",
                    "global_clean_mstyledistance_cosine": f"{global_mstyle:.12f}",
                    "pair_local_clean_pcm_multilingual_authorship_cosine": f"{local_pcm:.12f}",
                    "pair_local_clean_mstyledistance_cosine": f"{local_mstyle:.12f}",
                    "pair_local_or_raw_pcm_multilingual_authorship_cosine": f"{(local_pcm_value if local_reliable else raw_pcm):.12f}",
                    "pair_local_or_raw_mstyledistance_cosine": f"{(local_mstyle_value if local_reliable else raw_mstyle):.12f}",
                    "pair_local_style_reliable": str(local_reliable),
                    "global_style_reliable": str(global_reliable),
                    "global_and_pair_local_style_reliable": str(intersection),
                    "pair_local_maximum_mask_fraction": f"{max(left_fraction, right_fraction):.12f}",
                    "pair_local_mean_mask_fraction": f"{((left_fraction + right_fraction) / 2.0):.12f}",
                    "pair_local_shared_shingle_count_log1p": f"{np.log1p(int(text['shared_shingle_count'])):.12f}",
                }
            )
        output_name = policy["outputs"][parent_key]
        output_path = output_root / output_name
        step24.write_csv_immutable(output_path, pair_rows)
        if len(pair_rows) != len(rows):
            raise ValueError(f"Step25-v2 pair feature row count differs: {pool_name}")
        output_records[pool_name] = {
            "row_count": len(pair_rows),
            "valid_test_pair_count": 0,
            "review_label_columns_written": 0,
            "pair_local_reliable_count": local_reliable_count,
            "pair_local_reliable_fraction": local_reliable_count / max(len(pair_rows), 1),
            "global_reliable_count": global_reliable_count,
            "global_and_pair_local_reliable_count": intersection_count,
            "missing_pair_local_style_row_count": len(pair_rows) - local_reliable_count,
            "missing_pair_local_style_fixed_zero_count": 0,
            "path": str(output_path.relative_to(common.ROOT)).replace("\\", "/"),
            "sha256": step24.sha256_file(output_path),
            "raw_step24_pair_features_sha256": step24.sha256_file(raw_path),
            "global_step25_v1_pair_features_sha256": step24.sha256_file(global_path),
            "pair_local_text_sha256": step24.sha256_file(text_path),
        }
    summary = {
        "step": "step25_v2_build_pair_features",
        "version": policy["version"],
        "status": "pass",
        "boundary": policy["boundary"]["name"],
        "feature_names": FEATURE_NAMES,
        "feature_count": len(FEATURE_NAMES),
        "missing_pair_local_style_persisted_as_nan": True,
        "missing_pair_local_style_encoded_as_fixed_zero": False,
        "labels_evidence_types_or_scores_used_to_build_features": False,
        "valid_test_pair_count": 0,
        "pools": output_records,
        "embedding_manifest_sha256": step24.sha256_file(embedding_manifest_path),
        "policy_sha256": step24.sha256_file(policy_path),
        "producer_sha256": step24.sha256_file(Path(__file__).resolve()),
    }
    summary["summary_sha256"] = step24.canonical_hash(summary)
    summary_path = output_root / policy["outputs"]["pair_feature_summary"]
    step24.write_json_immutable(summary_path, summary)
    print(
        json.dumps(
            {
                "status": "pass",
                "feature_count": len(FEATURE_NAMES),
                "pools": output_records,
                "summary": str(summary_path.relative_to(common.ROOT)).replace("\\", "/"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
