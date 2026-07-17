#!/usr/bin/env python3
"""Build fixed raw/decontaminated Step25 pair features on canonical train only."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import step24_common as step24
import step25_common as common


FEATURE_NAMES = [
    "identifier_redacted_e5_cosine",
    "raw_pcm_multilingual_authorship_cosine",
    "raw_mstyledistance_cosine",
    "decontaminated_pcm_multilingual_authorship_cosine",
    "decontaminated_mstyledistance_cosine",
    "pcm_raw_minus_decontaminated",
    "mstyledistance_raw_minus_decontaminated",
    "pair_maximum_boilerplate_fraction",
    "pair_mean_boilerplate_fraction",
    "decontaminated_pair_reliable",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", default=str(common.DEFAULT_POLICY))
    parser.add_argument("--validate-config-only", action="store_true")
    args = parser.parse_args()
    policy_path, policy, step24_policy = common.load_policy(args.policy)
    if args.validate_config_only:
        print(
            json.dumps(
                {
                    "status": "pass",
                    "feature_names": FEATURE_NAMES,
                    "pair_split": "train",
                    "valid_test_pair_count": 0,
                    "numerical_execution_performed": False,
                },
                indent=2,
            )
        )
        return

    output_root = common.resolve(policy["outputs_root"])
    step24_root = common.resolve(policy["inputs"]["step24_outputs_root"])
    rows_by_pool = common.load_rows(policy, step24_policy)
    output_records = {}
    output_names = {
        "en_content_train_pool": policy["outputs"]["pair_features_en"],
        "zh_target_strict": policy["outputs"]["pair_features_zh"],
    }
    for pool_name, rows in rows_by_pool.items():
        parent_output_key = (
            "pair_features_en" if pool_name == "en_content_train_pool" else "pair_features_zh"
        )
        raw_path = step24_root / step24_policy["outputs"][parent_output_key]
        raw_rows = step24.load_csv(raw_path)
        raw_index = {row["pair_uid"]: row for row in raw_rows}
        if len(raw_index) != len(raw_rows):
            raise ValueError(f"Step25 duplicate Step24 pair feature: {pool_name}")

        _, text_path, summary_path = common.template_output_paths(output_root, pool_name)
        text_records = common.load_jsonl(text_path)
        text_index = {row["seller_uid"]: row for row in text_records}
        if len(text_index) != len(text_records):
            raise ValueError(f"Step25 duplicate decontaminated seller: {pool_name}")
        reliable = {
            seller: bool(int(item["decontaminated_text_reliable"]))
            for seller, item in text_index.items()
        }
        embedding_indices = {}
        embedding_matrices = {}
        for encoder_key in policy["frozen_style_encoders"]:
            matrix_path, metadata_path = common.embedding_output_paths(
                output_root, encoder_key, pool_name
            )
            index, matrix, metadata = step24.load_normalized_cache(metadata_path, matrix_path)
            if metadata.get("template_decontaminated") is not True or metadata.get(
                "valid_test_seller_encoded_count"
            ) != 0:
                raise ValueError(f"Step25 embedding isolation failed: {encoder_key}:{pool_name}")
            embedding_indices[encoder_key] = index
            embedding_matrices[encoder_key] = matrix

        pair_rows = []
        for row in rows:
            raw = raw_index.get(row["pair_uid"])
            if raw is None:
                raise ValueError(f"Step25 missing raw Step24 pair feature: {row['pair_uid']}")
            left = row["seller_uid_left"]
            right = row["seller_uid_right"]
            if left not in text_index or right not in text_index:
                raise ValueError(f"Step25 missing text diagnostics: {row['pair_uid']}")
            raw_pcm = float(raw["pcm_multilingual_authorship_cosine"])
            raw_mstyle = float(raw["mstyledistance_cosine"])
            clean_pcm = common.pair_cosine(
                row,
                embedding_indices["pcm_multilingual_authorship"],
                embedding_matrices["pcm_multilingual_authorship"],
                reliable,
                float(policy["template_decontamination"]["insufficient_text_pair_cosine"]),
            )
            clean_mstyle = common.pair_cosine(
                row,
                embedding_indices["mstyledistance"],
                embedding_matrices["mstyledistance"],
                reliable,
                float(policy["template_decontamination"]["insufficient_text_pair_cosine"]),
            )
            left_fraction = float(text_index[left]["boilerplate_fraction"])
            right_fraction = float(text_index[right]["boilerplate_fraction"])
            pair_reliable = int(reliable[left] and reliable[right])
            pair_rows.append(
                {
                    "pair_uid": row["pair_uid"],
                    "pool": pool_name,
                    "domain": row["domain"],
                    "split_name": "train",
                    "seller_uid_left": left,
                    "seller_uid_right": right,
                    "component_id": row["step25_component_id"],
                    "identifier_redacted_e5_cosine": f"{float(raw['identifier_redacted_e5_cosine']):.12f}",
                    "raw_pcm_multilingual_authorship_cosine": f"{raw_pcm:.12f}",
                    "raw_mstyledistance_cosine": f"{raw_mstyle:.12f}",
                    "decontaminated_pcm_multilingual_authorship_cosine": f"{clean_pcm:.12f}",
                    "decontaminated_mstyledistance_cosine": f"{clean_mstyle:.12f}",
                    "pcm_raw_minus_decontaminated": f"{raw_pcm - clean_pcm:.12f}",
                    "mstyledistance_raw_minus_decontaminated": f"{raw_mstyle - clean_mstyle:.12f}",
                    "pair_maximum_boilerplate_fraction": f"{max(left_fraction, right_fraction):.12f}",
                    "pair_mean_boilerplate_fraction": f"{(left_fraction + right_fraction) / 2.0:.12f}",
                    "decontaminated_pair_reliable": str(pair_reliable),
                }
            )
        path = output_root / output_names[pool_name]
        step24.write_csv_immutable(path, pair_rows)
        matrix = np.asarray(
            [[float(row[name]) for name in FEATURE_NAMES] for row in pair_rows], dtype=float
        )
        if not np.all(np.isfinite(matrix)):
            raise ValueError(f"Step25 non-finite pair feature: {pool_name}")
        output_records[pool_name] = {
            "row_count": len(pair_rows),
            "positive_or_negative_label_columns_written": 0,
            "valid_test_pair_count": 0,
            "reliable_pair_count": int(
                sum(int(row["decontaminated_pair_reliable"]) for row in pair_rows)
            ),
            "reliable_pair_fraction": float(
                np.mean(matrix[:, FEATURE_NAMES.index("decontaminated_pair_reliable")])
            ),
            "feature_minimums": {
                name: float(np.min(matrix[:, index])) for index, name in enumerate(FEATURE_NAMES)
            },
            "feature_maximums": {
                name: float(np.max(matrix[:, index])) for index, name in enumerate(FEATURE_NAMES)
            },
            "path": str(path.relative_to(common.ROOT)).replace("\\", "/"),
            "sha256": step24.sha256_file(path),
            "template_summary_sha256": step24.sha256_file(summary_path),
            "raw_step24_pair_features_sha256": step24.sha256_file(raw_path),
        }
    summary = {
        "step": "step25_build_pair_features",
        "version": policy["version"],
        "feature_names": FEATURE_NAMES,
        "feature_count": len(FEATURE_NAMES),
        "identifiers_used_as_clean_features": False,
        "candidate_rule_features_used": False,
        "review_labels_written": False,
        "valid_test_pair_count": 0,
        "records": output_records,
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
                "valid_test_pair_count": 0,
                "summary": str(summary_path.relative_to(common.ROOT)).replace("\\", "/"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
