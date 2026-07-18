#!/usr/bin/env python3
"""Join frozen Step24/25-v1/v2 train-only features for Step25-v3."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import step24_common as step24
import step25_common as step25_v1
import step25_v3_common as common


BASE_FIELDS = [
    "pair_uid",
    "pool",
    "domain",
    "split_name",
    "seller_uid_left",
    "seller_uid_right",
    "component_id",
]


def index_unique(rows: list[dict], name: str) -> dict[str, dict]:
    index = {row["pair_uid"]: row for row in rows}
    if len(index) != len(rows):
        raise ValueError(f"Step25-v3 duplicate pair_uid in {name}")
    return index


def load_jsonl_index(path: Path) -> dict[str, dict]:
    return index_unique(step25_v1.load_jsonl(path), str(path))


def assert_parent_alignment(
    pair_uid: str, v1: dict, v2: dict, text: dict, pool_name: str
) -> None:
    checks = {
        "pool": pool_name,
        "split_name": "train",
        "seller_uid_left": v1["seller_uid_left"],
        "seller_uid_right": v1["seller_uid_right"],
    }
    for key, expected in checks.items():
        if str(v2.get(key)) != str(expected) or str(text.get(key)) != str(expected):
            raise ValueError(f"Step25-v3 parent alignment mismatch: {pair_uid}:{key}")
    if str(v1["component_id"]) != str(v2["component_id"]) or str(v1["component_id"]) != str(
        text["component_id"]
    ):
        raise ValueError(f"Step25-v3 component mismatch: {pair_uid}")


def build_pool(
    policy: dict,
    step25_v1_policy: dict,
    step25_v2_policy: dict,
    pool_name: str,
) -> tuple[list[dict], dict]:
    v1_root = common.resolve(policy["inputs"]["step25_v1_outputs_root"])
    v2_root = common.resolve(policy["inputs"]["step25_v2_outputs_root"])
    v1_key = "pair_features_en" if pool_name == "en_content_train_pool" else "pair_features_zh"
    v2_key = v1_key
    text_key = "pair_local_texts_en" if pool_name == "en_content_train_pool" else "pair_local_texts_zh"
    v1_path = v1_root / step25_v1_policy["outputs"][v1_key]
    v2_path = v2_root / step25_v2_policy["outputs"][v2_key]
    text_path = v2_root / step25_v2_policy["outputs"][text_key]
    for path in (v1_path, v2_path, text_path):
        if not path.is_file():
            raise FileNotFoundError(f"Step25-v3 parent feature is missing: {path}")
    v1_rows = step24.load_csv(v1_path)
    v2_rows = step24.load_csv(v2_path)
    v1_index = index_unique(v1_rows, str(v1_path))
    v2_index = index_unique(v2_rows, str(v2_path))
    text_index = load_jsonl_index(text_path)
    if set(v1_index) != set(v2_index) or set(v1_index) != set(text_index):
        raise ValueError(f"Step25-v3 parent pair sets disagree for {pool_name}")
    output = []
    fallback_count = 0
    masked_pair_count = 0
    for pair_uid in sorted(v1_index):
        v1 = v1_index[pair_uid]
        v2 = v2_index[pair_uid]
        text = text_index[pair_uid]
        assert_parent_alignment(pair_uid, v1, v2, text, pool_name)
        raw_pcm = float(v2["raw_pcm_multilingual_authorship_cosine"])
        raw_mstyle = float(v2["raw_mstyledistance_cosine"])
        local_pcm = float(v2["pair_local_or_raw_pcm_multilingual_authorship_cosine"])
        local_mstyle = float(v2["pair_local_or_raw_mstyledistance_cosine"])
        local_reliable = int(v2["pair_local_style_reliable"])
        if local_reliable not in {0, 1}:
            raise ValueError(f"Step25-v3 local reliability is not binary: {pair_uid}")
        if not local_reliable:
            fallback_count += 1
            if abs(local_pcm - raw_pcm) > 1e-10 or abs(local_mstyle - raw_mstyle) > 1e-10:
                raise ValueError(f"Step25-v3 unreliable local style did not fall back to raw: {pair_uid}")
        pcm_delta = raw_pcm - local_pcm
        mstyle_delta = raw_mstyle - local_mstyle
        if not local_reliable and (abs(pcm_delta) > 1e-10 or abs(mstyle_delta) > 1e-10):
            raise ValueError(f"Step25-v3 unreliable local residual is nonzero: {pair_uid}")
        masked_span_count = int(text["left_masked_span_count"]) + int(
            text["right_masked_span_count"]
        )
        masked_pair_count += int(masked_span_count > 0)
        output.append(
            {
                "pair_uid": pair_uid,
                "pool": pool_name,
                "domain": v1["domain"],
                "split_name": "train",
                "seller_uid_left": v1["seller_uid_left"],
                "seller_uid_right": v1["seller_uid_right"],
                "component_id": v1["component_id"],
                "identifier_redacted_e5_cosine": f"{float(v1['identifier_redacted_e5_cosine']):.12f}",
                "raw_pcm_multilingual_authorship_cosine": f"{raw_pcm:.12f}",
                "raw_mstyledistance_cosine": f"{raw_mstyle:.12f}",
                "pair_local_or_raw_pcm_multilingual_authorship_cosine": f"{local_pcm:.12f}",
                "pair_local_or_raw_mstyledistance_cosine": f"{local_mstyle:.12f}",
                "pcm_raw_minus_pair_local_or_raw": f"{pcm_delta:.12f}",
                "mstyledistance_raw_minus_pair_local_or_raw": f"{mstyle_delta:.12f}",
                "pair_local_maximum_mask_fraction": f"{float(v2['pair_local_maximum_mask_fraction']):.12f}",
                "pair_local_mean_mask_fraction": f"{float(v2['pair_local_mean_mask_fraction']):.12f}",
                "pair_local_shared_shingle_count_log1p": f"{float(v2['pair_local_shared_shingle_count_log1p']):.12f}",
                "pair_local_masked_span_count_log1p": f"{math.log1p(masked_span_count):.12f}",
                "pair_local_style_reliable": str(local_reliable),
                "global_pair_maximum_boilerplate_fraction": f"{float(v1['pair_maximum_boilerplate_fraction']):.12f}",
                "global_pair_mean_boilerplate_fraction": f"{float(v1['pair_mean_boilerplate_fraction']):.12f}",
                "global_style_reliable": str(int(v1["decontaminated_pair_reliable"])),
            }
        )
    return output, {
        "pool": pool_name,
        "row_count": len(output),
        "valid_or_test_row_count": 0,
        "pair_local_raw_fallback_count": fallback_count,
        "pair_with_masked_span_count": masked_pair_count,
        "v1_pair_features_path": str(v1_path.relative_to(common.ROOT)).replace("\\", "/"),
        "v1_pair_features_sha256": step24.sha256_file(v1_path),
        "v2_pair_features_path": str(v2_path.relative_to(common.ROOT)).replace("\\", "/"),
        "v2_pair_features_sha256": step24.sha256_file(v2_path),
        "v2_pair_local_texts_path": str(text_path.relative_to(common.ROOT)).replace("\\", "/"),
        "v2_pair_local_texts_sha256": step24.sha256_file(text_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", default=str(common.DEFAULT_POLICY))
    parser.add_argument("--validate-config-only", action="store_true")
    args = parser.parse_args()
    (
        policy_path,
        policy,
        _v7_policy,
        _step24_policy,
        step25_v1_policy,
        step25_v2_policy,
    ) = common.load_policy(args.policy)
    if args.validate_config_only:
        print(
            json.dumps(
                {
                    "status": "pass",
                    "output_root": policy["outputs_root"],
                    "valid_or_test_read": False,
                    "numerical_execution_performed": False,
                },
                indent=2,
            )
        )
        return
    parents = common.require_parent_manifests(policy)
    root = common.resolve(policy["outputs_root"])
    pool_records = {}
    feature_names = sorted(
        {
            name
            for names in common.feature_groups(policy).values()
            for name in names
        }
    )
    for pool_name, output_key in (
        ("en_content_train_pool", "pair_features_en"),
        ("zh_target_strict", "pair_features_zh"),
    ):
        rows, record = build_pool(policy, step25_v1_policy, step25_v2_policy, pool_name)
        output_path = root / policy["outputs"][output_key]
        step24.write_csv_immutable(output_path, rows)
        record.update(
            {
                "output_path": str(output_path.relative_to(common.ROOT)).replace("\\", "/"),
                "output_sha256": step24.sha256_file(output_path),
            }
        )
        pool_records[pool_name] = record
    summary = {
        "step": "step25_v3_build_dual_channel_features",
        "version": policy["version"],
        "status": "pass",
        "boundary": policy["boundary"]["name"],
        "valid_or_test_rows_read": 0,
        "labels_or_evidence_types_used_as_features": False,
        "unreliable_pair_local_clean_value": "raw_style_fallback",
        "missing_clean_style_encoded_as_zero": False,
        "feature_names": feature_names,
        "feature_count": len(feature_names),
        "parents": parents,
        "pools": pool_records,
        "policy_path": str(policy_path.relative_to(common.ROOT)).replace("\\", "/"),
        "policy_sha256": step24.sha256_file(policy_path),
        "producer_sha256": step24.sha256_file(Path(__file__).resolve()),
    }
    summary["summary_sha256"] = step24.canonical_hash(summary)
    summary_path = root / policy["outputs"]["pair_feature_summary"]
    step24.write_json_immutable(summary_path, summary)
    print(
        json.dumps(
            {
                "status": "pass",
                "output_root": policy["outputs_root"],
                "pool_counts": {key: value["row_count"] for key, value in pool_records.items()},
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
