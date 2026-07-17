#!/usr/bin/env python3
"""Build label-free, seller-component-cross-fitted boilerplate catalogs and texts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import step25_common as common
import step24_common as step24
import step24_build_style_embedding_cache as step24_cache


def selected_sellers_and_components(
    policy: dict,
    step24_policy: dict,
    pool_name: str,
) -> tuple[list[str], dict[str, str], dict]:
    step24_root = common.resolve(policy["inputs"]["step24_outputs_root"])
    raw_matrix_path, raw_metadata_path = step24_cache.output_paths(
        step24_root, "pcm_multilingual_authorship", pool_name
    )
    if not raw_matrix_path.is_file() or not raw_metadata_path.is_file():
        raise FileNotFoundError(
            "Step25 requires the frozen Step24 raw style cache on Linux before "
            f"template removal: {raw_metadata_path}"
        )
    metadata = json.loads(raw_metadata_path.read_text(encoding="utf-8"))
    seller_uids = list(metadata.get("seller_uids", []))
    if not seller_uids or len(set(seller_uids)) != len(seller_uids):
        raise ValueError(f"Step25 invalid Step24 seller index: {pool_name}")
    if metadata.get("encoded_split") != "train" or metadata.get(
        "valid_test_seller_encoded_count"
    ) != 0:
        raise ValueError(f"Step25 raw style cache is not train-only: {pool_name}")
    assignment_rows = step24.load_csv(common.resolve(policy["inputs"]["component_assignments"]))
    selected = set(seller_uids)
    component_by_seller: dict[str, str] = {}
    for row in assignment_rows:
        if row.get("dataset") != pool_name or row.get("split_name") != "train":
            continue
        if row.get("cross_split_component_leakage") == "1" or row.get(
            "cross_split_seller_leakage"
        ) == "1":
            raise ValueError(f"Step25 refuses a leaking component assignment: {row['pair_uid']}")
        component = row["recomputed_component_id"]
        for field in ("seller_uid_left", "seller_uid_right"):
            seller = row[field]
            if seller not in selected:
                continue
            previous = component_by_seller.setdefault(seller, component)
            if previous != component:
                raise ValueError(f"Step25 seller spans components: {seller}")
    missing = sorted(selected - set(component_by_seller))
    if missing:
        raise ValueError(f"Step25 selected seller lacks a component: {pool_name}:{missing[0]}")
    return seller_uids, component_by_seller, metadata


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
                    "policy": str(policy_path.relative_to(common.ROOT)).replace("\\", "/"),
                    "template_labels_read": False,
                    "component_cross_fitted": True,
                    "model_directory_checked": False,
                    "numerical_embedding_execution_performed": False,
                },
                indent=2,
            )
        )
        return

    output_root = common.resolve(policy["outputs_root"])
    cfg = policy["template_decontamination"]
    component_assignments_path = common.resolve(policy["inputs"]["component_assignments"])
    component_assignments_sha256 = step24.sha256_file(component_assignments_path)
    records = {}
    for pool_name, pool_cfg in step24_policy["pools"].items():
        seller_uids, component_by_seller, raw_metadata = selected_sellers_and_components(
            policy, step24_policy, pool_name
        )
        texts, diagnostics = step24.replay_v7_clean_texts(
            pool_cfg, policy["clean_text_contract"], seller_uids
        )
        if diagnostics["clean_text_corpus_sha256"] != raw_metadata.get(
            "clean_text_corpus_sha256"
        ):
            raise ValueError(
                f"Step25 clean-text replay differs from the frozen Step24 corpus: {pool_name}"
            )
        text_records, catalog_records, summary = common.decontaminate_corpus(
            seller_uids, texts, component_by_seller, cfg
        )
        catalog_path, text_path, summary_path = common.template_output_paths(
            output_root, pool_name
        )
        common.write_csv_immutable_allow_empty(
            catalog_path,
            catalog_records,
            [
                "shingle_sha256",
                "seller_document_frequency",
                "component_document_frequency",
                "character_length",
            ],
        )
        common.write_jsonl_immutable(text_path, text_records)
        summary_payload = {
            "step": "step25_template_decontamination_pool",
            "version": policy["version"],
            "pool": pool_name,
            "domain": pool_cfg["domain"],
            "encoded_split": "train",
            "valid_test_seller_count": 0,
            "review_label_read_by_template_detector": False,
            "evidence_type_read_by_template_detector": False,
            "model_score_read_by_template_detector": False,
            "raw_ngram_text_persisted": False,
            "source_clean_text_corpus_sha256": diagnostics["clean_text_corpus_sha256"],
            "component_assignments_sha256": component_assignments_sha256,
            "raw_step24_metadata_sha256": step24.sha256_file(
                step24_cache.output_paths(
                    common.resolve(policy["inputs"]["step24_outputs_root"]),
                    "pcm_multilingual_authorship",
                    pool_name,
                )[1]
            ),
            "catalog_sha256": step24.sha256_file(catalog_path),
            "decontaminated_texts_sha256": step24.sha256_file(text_path),
            "detector_configuration": cfg,
            **summary,
            "policy_sha256": step24.sha256_file(policy_path),
            "producer_sha256": step24.sha256_file(Path(__file__).resolve()),
        }
        summary_payload["summary_sha256"] = step24.canonical_hash(summary_payload)
        step24.write_json_immutable(summary_path, summary_payload)
        records[pool_name] = {
            "catalog": str(catalog_path.relative_to(common.ROOT)).replace("\\", "/"),
            "decontaminated_texts": str(text_path.relative_to(common.ROOT)).replace(
                "\\", "/"
            ),
            "summary": str(summary_path.relative_to(common.ROOT)).replace("\\", "/"),
            "catalog_sha256": step24.sha256_file(catalog_path),
            "decontaminated_texts_sha256": step24.sha256_file(text_path),
            "summary_file_sha256": step24.sha256_file(summary_path),
            "seller_count": summary["seller_count"],
            "component_count": summary["component_count"],
            "reliable_seller_fraction": summary["reliable_seller_fraction"],
        }
    manifest = {
        "step": "step25_build_template_decontamination",
        "version": policy["version"],
        "boundary": "d0_current_canonical_train",
        "publication_promotion_allowed": False,
        "template_detector_read_labels_evidence_types_or_scores": False,
        "valid_test_text_or_seller_count": 0,
        "component_cross_fitted": True,
        "component_assignments_path": str(
            component_assignments_path.relative_to(common.ROOT)
        ).replace("\\", "/"),
        "component_assignments_sha256": component_assignments_sha256,
        "records": records,
        "policy_sha256": step24.sha256_file(policy_path),
        "producer_sha256": step24.sha256_file(Path(__file__).resolve()),
    }
    manifest["manifest_sha256"] = step24.canonical_hash(manifest)
    manifest_path = output_root / policy["outputs"]["template_manifest"]
    step24.write_json_immutable(manifest_path, manifest)
    print(
        json.dumps(
            {
                "status": "pass",
                "component_cross_fitted": True,
                "template_detector_read_labels": False,
                "records": records,
                "manifest": str(manifest_path.relative_to(common.ROOT)).replace("\\", "/"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
