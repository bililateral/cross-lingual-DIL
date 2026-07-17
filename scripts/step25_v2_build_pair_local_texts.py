#!/usr/bin/env python3
"""Build label-free pair-local copied-span removals on canonical train pairs."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import step24_common as step24
import step25_common as step25_v1
import step25_v2_common as common


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
                    "detector": "pair_local_symmetric_character_shingle",
                    "character_shingle_length": policy["pair_local_copy_detector"][
                        "character_shingle_length"
                    ],
                    "valid_test_rows_read": 0,
                    "labels_or_scores_read_by_detector": False,
                    "numerical_embedding_execution_performed": False,
                },
                indent=2,
            )
        )
        return

    rows_by_pool = common.load_rows(policy, step24_policy, step25_v1_policy)
    detector_cfg = policy["pair_local_copy_detector"]
    pool_summaries = {}
    for pool_name, rows in rows_by_pool.items():
        text_index, replay = common.replay_train_text_index(pool_name, rows, step24_policy)
        records = []
        counts = Counter()
        total_left_masked = 0
        total_right_masked = 0
        for row in rows:
            result = common.detect_pair_local_copy(
                text_index[row["seller_uid_left"]],
                text_index[row["seller_uid_right"]],
                detector_cfg,
            )
            counts["pair_count"] += 1
            counts["pair_with_shared_shingle"] += int(result["shared_shingle_count"] > 0)
            counts["pair_with_masked_span"] += int(
                result["left_masked_span_count"] > 0
                and result["right_masked_span_count"] > 0
            )
            counts["pair_reliable"] += int(result["pair_reliable"])
            total_left_masked += int(result["left_masked_character_count"])
            total_right_masked += int(result["right_masked_character_count"])
            records.append(
                {
                    "pair_uid": row["pair_uid"],
                    "pool": pool_name,
                    "domain": row["domain"],
                    "split_name": "train",
                    "seller_uid_left": row["seller_uid_left"],
                    "seller_uid_right": row["seller_uid_right"],
                    "component_id": row["step25_component_id"],
                    **result,
                    "labels_evidence_types_or_scores_read_by_detector": False,
                }
            )
        output_path = common.pair_text_path(policy, pool_name)
        step25_v1.write_jsonl_immutable(output_path, records)
        pool_summaries[pool_name] = {
            **replay,
            "output_path": str(output_path.relative_to(common.ROOT)).replace("\\", "/"),
            "output_sha256": step24.sha256_file(output_path),
            "pair_count": counts["pair_count"],
            "pair_with_shared_shingle_count": counts["pair_with_shared_shingle"],
            "pair_with_masked_span_count": counts["pair_with_masked_span"],
            "pair_reliable_count": counts["pair_reliable"],
            "pair_reliable_fraction": counts["pair_reliable"] / max(counts["pair_count"], 1),
            "left_masked_character_count": total_left_masked,
            "right_masked_character_count": total_right_masked,
            "persisted_raw_shared_span_count": 0,
            "review_label_evidence_type_or_model_score_read": False,
        }
    summary = {
        "step": "step25_v2_build_pair_local_texts",
        "version": policy["version"],
        "status": "pass",
        "boundary": policy["boundary"]["name"],
        "detector_scope": "pair_local_only",
        "hypothesis_informed_retrospective": True,
        "d1_candidate_eligible": False,
        "publication_promotion_eligible": False,
        "valid_test_rows_read": 0,
        "labels_evidence_types_or_scores_read_by_detector": False,
        "detector_configuration": detector_cfg,
        "pools": pool_summaries,
        "policy_sha256": step24.sha256_file(policy_path),
        "step24_policy_sha256": step24.sha256_file(
            common.resolve(policy["inputs"]["step24_policy"])
        ),
        "step25_v1_policy_sha256": step24.sha256_file(
            common.resolve(policy["inputs"]["step25_v1_policy"])
        ),
        "producer_sha256": step24.sha256_file(Path(__file__).resolve()),
    }
    summary["summary_sha256"] = step24.canonical_hash(summary)
    summary_path = common.resolve(policy["outputs_root"]) / policy["outputs"][
        "detector_summary"
    ]
    step24.write_json_immutable(summary_path, summary)
    print(
        json.dumps(
            {
                "status": "pass",
                "pools": {
                    key: {
                        "pair_count": value["pair_count"],
                        "pair_with_masked_span_count": value[
                            "pair_with_masked_span_count"
                        ],
                        "pair_reliable_fraction": value["pair_reliable_fraction"],
                    }
                    for key, value in pool_summaries.items()
                },
                "summary": str(summary_path.relative_to(common.ROOT)).replace("\\", "/"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
