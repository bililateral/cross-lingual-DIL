#!/usr/bin/env python3
"""Prepare the label-free V9.4.1 public projection used by M0 and M3.

The current entry point provides two physically separate no-output smoke views
of one complete world.  ``base`` reconstructs the English reference and checks
legacy18/text inputs but cannot open identity33.  ``identity`` checks identity33
alignment but cannot open text or seller profiles.  Neither view opens any
supervision.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from collections import Counter, defaultdict
from itertools import zip_longest
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

import numpy as np

import step28_v13_v1_13_v9_4_1_model_experiment_common_v1 as common


SPLITS = ("train", "development", "audit_a", "audit_b")
BASE_PUBLIC_ROLES = (
    "worlds.jsonl",
    "sellers.jsonl",
    "redacted_items.jsonl",
    "model_seller_profiles.jsonl",
    "complete_model_pair_endpoints.csv",
)
IDENTITY_PUBLIC_ROLES = (
    "worlds.jsonl",
    "complete_model_pair_endpoints.csv",
    "identity33_all_pairs.csv",
)


def iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise common.ModelExperimentContractError(
                    f"Invalid public JSONL at {path}:{line_number}"
                ) from exc
            if not isinstance(row, dict):
                raise common.ModelExperimentContractError(
                    f"Non-object public JSONL row at {path}:{line_number}"
                )
            yield row


def iter_csv(path: Path) -> Iterator[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        yield from csv.DictReader(handle)


def _public_file_map(root_manifest: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    rows = root_manifest.get("public_files")
    if not isinstance(rows, list):
        raise common.ModelExperimentContractError("Root public-file registry is absent")
    output = {str(row["path"]): row for row in rows}
    if len(output) != len(rows):
        raise common.ModelExperimentContractError("Root public-file registry duplicates paths")
    return output


def verify_split_public_inputs(
    policy: Mapping[str, Any], split: str, roles: Sequence[str]
) -> dict[str, Path]:
    if split not in SPLITS:
        raise common.ModelExperimentContractError(f"Unknown formal split: {split}")
    qualification = policy["dataset_qualification"]
    root_manifest = common.load_json(
        common.resolve(qualification["root_manifest"]["path"])
    )
    public_files = _public_file_map(root_manifest)
    root = common.resolve(qualification["root"])
    output = {}
    allowed = set(policy["public_projection"]["allowed_observed_files"])
    if tuple(roles) not in {BASE_PUBLIC_ROLES, IDENTITY_PUBLIC_ROLES}:
        raise common.ModelExperimentContractError("Unknown public projection view")
    if not roles or len(set(roles)) != len(roles) or not set(roles).issubset(allowed):
        raise common.ModelExperimentContractError("Public projection role request drift")
    forbidden = (
        {"identity33_all_pairs.csv"}
        if tuple(roles) == BASE_PUBLIC_ROLES
        else {"redacted_items.jsonl", "model_seller_profiles.jsonl", "sellers.jsonl"}
    )
    if set(roles) & forbidden:
        raise common.ModelExperimentContractError("Public projection view isolation drift")
    for filename in roles:
        relative = f"{split}/observed/{filename}"
        if relative not in public_files:
            raise common.ModelExperimentContractError(
                f"Formal root does not register public input: {relative}"
            )
        spec = public_files[relative]
        output[filename] = common.verify_file_pin(
            {**spec, "path": str(root / relative)},
            label=f"formal public input {relative}",
        )
    return output


def reconstruct_frozen_english_reference(
    policy: Mapping[str, Any],
) -> dict[str, Any]:
    """Replay Step7's exact label-free reference; this may take a few minutes."""

    scripts = str(common.ROOT / "scripts")
    if scripts not in sys.path:
        sys.path.insert(0, scripts)
    import step7_v4_common as step7_common
    import step7_v4_prepare_source_data as step7_prepare

    step7_policy = step7_common.load_policy()
    _parent, public, _pairs, _safe = step7_prepare.replay_parent_public(step7_policy)
    reference = public["reference"]
    expected = policy["frozen_english_reference"]
    if (
        int(reference.get("train_seller_count", -1)) != expected["fit_seller_count"]
        or reference.get("train_seller_uid_sha256") != expected["seller_uid_sha256"]
        or common.canonical_sha256(reference) != expected["feature_reference_sha256"]
    ):
        raise common.ModelExperimentContractError("Frozen English reference replay drift")
    return reference


def _split_concat_and_normalize(value: str) -> list[str]:
    scripts = str(common.ROOT / "scripts")
    if scripts not in sys.path:
        sys.path.insert(0, scripts)
    import step7_v3_1_source_data as source

    return sorted(
        {
            normalized
            for segment in source.split_concat(value)
            if (normalized := source.normalize_signature(segment))
        }
    )


def project_model_profile(row: Mapping[str, Any]) -> dict[str, Any]:
    required = [
        "seller_uid",
        "category_concat_top",
        "signature_title_concat",
        "title_concat_top",
        "signature_description_concat",
        "description_concat_top",
        "item_count",
        "title_length_stats",
        "description_length_stats",
        "style_stats",
    ]
    if list(row) != required:
        raise common.ModelExperimentContractError("Model seller-profile schema/order drift")
    style = row["style_stats"]
    numeric = {
        "item_count": float(row["item_count"]),
        "title_length_median": float(row["title_length_stats"]["median"]),
        "description_length_median": float(
            row["description_length_stats"]["median"]
        ),
        "digit_ratio_mean": float(style["digit_ratio_mean"]),
        "punct_ratio_mean": float(style["punct_ratio_mean"]),
        "repeated_title_share": float(style["repeated_title_share"]),
        "repeated_description_share": float(style["repeated_description_share"]),
        "max_category_share": float(style["max_category_share"]),
    }
    if not all(math.isfinite(value) for value in numeric.values()):
        raise common.ModelExperimentContractError("Model seller profile is non-finite")
    return {
        "seller_uid": str(row["seller_uid"]),
        "clean_categories": _split_concat_and_normalize(row["category_concat_top"]),
        "clean_titles": sorted(
            set(_split_concat_and_normalize(row["signature_title_concat"]))
            | set(_split_concat_and_normalize(row["title_concat_top"]))
        ),
        "clean_descriptions": sorted(
            set(_split_concat_and_normalize(row["signature_description_concat"]))
            | set(_split_concat_and_normalize(row["description_concat_top"]))
        ),
        "numeric_profile": numeric,
    }


def legacy18_row(
    pair: Mapping[str, str],
    seller_records: Mapping[str, Mapping[str, Any]],
    reference: Mapping[str, Any],
    feature_names: Sequence[str],
) -> np.ndarray:
    scripts = str(common.ROOT / "scripts")
    if scripts not in sys.path:
        sys.path.insert(0, scripts)
    import step7_v3_1_source_data as source

    left = seller_records[pair["seller_uid_left"]]
    right = seller_records[pair["seller_uid_right"]]
    left_categories = set(left["clean_categories"])
    right_categories = set(right["clean_categories"])
    shared_categories = left_categories & right_categories
    shared_titles = set(left["clean_titles"]) & set(right["clean_titles"])
    shared_descriptions = set(left["clean_descriptions"]) & set(
        right["clean_descriptions"]
    )
    title_sum, title_mean = source.shared_idf(
        shared_titles, reference["title_df"], int(reference["train_seller_count"])
    )
    description_sum, description_mean = source.shared_idf(
        shared_descriptions,
        reference["description_df"],
        int(reference["train_seller_count"]),
    )
    values = {
        "clean_category_jaccard": source.jaccard(
            left_categories, right_categories
        ),
        "clean_shared_title_bool": int(bool(shared_titles)),
        "clean_shared_description_bool": int(bool(shared_descriptions)),
        "clean_shared_title_count_capped": min(len(shared_titles), 5),
        "clean_shared_description_count_capped": min(len(shared_descriptions), 5),
        "clean_shared_category_count_capped": min(len(shared_categories), 5),
        "clean_shared_title_idf_sum": title_sum,
        "clean_shared_description_idf_sum": description_sum,
        "clean_shared_title_idf_mean": title_mean,
        "clean_shared_description_idf_mean": description_mean,
    }
    for name in source.NUMERIC_PROFILE_FIELDS:
        left_percentile = source.empirical_percentile(
            reference["numeric_references"][name], left["numeric_profile"][name]
        )
        right_percentile = source.empirical_percentile(
            reference["numeric_references"][name], right["numeric_profile"][name]
        )
        values[f"{name}_train_percentile_gap_abs"] = abs(
            left_percentile - right_percentile
        )
    if list(values) != list(feature_names):
        raise common.ModelExperimentContractError("legacy18 feature order drift")
    result = np.asarray([values[name] for name in feature_names], dtype="<f8")
    if not np.isfinite(result).all():
        raise common.ModelExperimentContractError("legacy18 row is non-finite")
    return result


def load_worlds(path: Path, split: str) -> list[dict[str, Any]]:
    worlds = list(iter_jsonl(path))
    if len(worlds) != 500:
        raise common.ModelExperimentContractError("Formal world count drift")
    for ordinal, row in enumerate(worlds):
        if (
            list(row)
            != [
                "world_uid",
                "split",
                "world_ordinal",
                "seller_count",
                "item_count",
                "pair_count",
            ]
            or row["split"] != split
            or int(row["world_ordinal"]) != ordinal
            or int(row["seller_count"]) != 28
            or int(row["item_count"]) != 99
            or int(row["pair_count"]) != 378
        ):
            raise common.ModelExperimentContractError(
                f"Formal world row drift at {split}:{ordinal}"
            )
    return worlds


def _rows_for_world(
    rows: Iterable[Mapping[str, Any]], world_uid: str
) -> list[Mapping[str, Any]]:
    return [row for row in rows if row.get("world_uid") == world_uid]


def smoke_world_base(
    policy: Mapping[str, Any],
    split: str,
    world_ordinal: int,
    reference: Mapping[str, Any],
) -> dict[str, Any]:
    paths = verify_split_public_inputs(policy, split, BASE_PUBLIC_ROLES)
    worlds = load_worlds(paths["worlds.jsonl"], split)
    if not 0 <= world_ordinal < len(worlds):
        raise common.ModelExperimentContractError("Smoke world ordinal is out of range")
    world_uid = worlds[world_ordinal]["world_uid"]
    sellers = _rows_for_world(iter_jsonl(paths["sellers.jsonl"]), world_uid)
    if len(sellers) != 28 or len({row["seller_uid"] for row in sellers}) != 28:
        raise common.ModelExperimentContractError("Smoke seller universe drift")
    seller_uids = {str(row["seller_uid"]) for row in sellers}

    profiles = {
        str(row["seller_uid"]): project_model_profile(row)
        for row in iter_jsonl(paths["model_seller_profiles.jsonl"])
        if row.get("seller_uid") in seller_uids
    }
    if set(profiles) != seller_uids:
        raise common.ModelExperimentContractError("Smoke profile universe drift")

    pairs = _rows_for_world(iter_csv(paths["complete_model_pair_endpoints.csv"]), world_uid)
    if len(pairs) != 378:
        raise common.ModelExperimentContractError("Smoke pair count drift")
    expected_edges = {
        common.canonical_pair_endpoints(left, right)
        for index, left in enumerate(sorted(seller_uids))
        for right in sorted(seller_uids)[index + 1 :]
    }
    observed_edges = {
        common.canonical_pair_endpoints(row["seller_uid_left"], row["seller_uid_right"])
        for row in pairs
    }
    if observed_edges != expected_edges:
        raise common.ModelExperimentContractError("Smoke K28 pair universe drift")

    names = policy["feature_contract"]["legacy18"]
    matrix = np.vstack(
        [legacy18_row(pair, profiles, reference, names) for pair in pairs]
    ).astype("<f8", copy=False)

    items = _rows_for_world(iter_jsonl(paths["redacted_items.jsonl"]), world_uid)
    if len(items) != 99 or any(row["seller_uid"] not in seller_uids for row in items):
        raise common.ModelExperimentContractError("Smoke redacted-item universe drift")
    unique_texts = {
        str(row[field])
        for row in items
        for field in ("title", "description")
        if str(row[field]).strip()
    }
    return {
        "step": "step28_v13_v1_13_v9_4_1_base_projection_smoke_v1",
        "status": "PASSED_LABEL_FREE_BASE_VIEW_ONE_WORLD_NO_OUTPUT",
        "split": split,
        "world_ordinal": world_ordinal,
        "pair_count": len(pairs),
        "legacy18_shape": list(matrix.shape),
        "legacy18_value_sha256": common.matrix_value_sha256(matrix),
        "redacted_item_count": len(items),
        "unique_nonempty_text_count": len(unique_texts),
        "labels_or_qrels_or_controller_read": False,
        "audit_truth_read": False,
        "output_files_written": 0,
        "identity33_files_opened": 0,
    }


def smoke_world_identity(
    policy: Mapping[str, Any], split: str, world_ordinal: int
) -> dict[str, Any]:
    paths = verify_split_public_inputs(policy, split, IDENTITY_PUBLIC_ROLES)
    worlds = load_worlds(paths["worlds.jsonl"], split)
    if not 0 <= world_ordinal < len(worlds):
        raise common.ModelExperimentContractError("Smoke world ordinal is out of range")
    world_uid = worlds[world_ordinal]["world_uid"]
    pairs = _rows_for_world(iter_csv(paths["complete_model_pair_endpoints.csv"]), world_uid)
    identity_rows = _rows_for_world(
        iter_csv(paths["identity33_all_pairs.csv"]), world_uid
    )
    if len(pairs) != 378 or len(identity_rows) != 378:
        raise common.ModelExperimentContractError("Identity-view pair count drift")
    identity_names = list(identity_rows[0])[2:]
    common.validate_identity33_column_names(policy, identity_names)
    for pair, identity in zip(pairs, identity_rows, strict=True):
        if (
            identity["canonical_pair_uid"] != pair["canonical_pair_uid"]
            or identity["world_uid"] != pair["world_uid"]
        ):
            raise common.ModelExperimentContractError("identity33/pair row alignment drift")
    identity_matrix = np.asarray(
        [[float(row[name]) for name in identity_names] for row in identity_rows],
        dtype="<f8",
    )
    active = common.active_mask(identity_matrix)
    return {
        "step": "step28_v13_v1_13_v9_4_1_identity_projection_smoke_v1",
        "status": "PASSED_LABEL_FREE_IDENTITY_VIEW_ONE_WORLD_NO_OUTPUT",
        "split": split,
        "world_ordinal": world_ordinal,
        "pair_count": len(pairs),
        "identity33_shape": list(identity_matrix.shape),
        "identity33_value_sha256": common.matrix_value_sha256(identity_matrix),
        "identity33_active_pair_count": int(active.sum()),
        "labels_or_qrels_or_controller_read": False,
        "audit_truth_read": False,
        "output_files_written": 0,
        "text_or_seller_profile_files_opened": 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--view", choices=("base", "identity"), required=True)
    parser.add_argument("--split", choices=SPLITS, default="train")
    parser.add_argument("--world-ordinal", type=int, default=0)
    args = parser.parse_args()
    policy = common.load_policy()
    if args.view == "base":
        common.validate_frozen_model_payloads(policy)
        reference = reconstruct_frozen_english_reference(policy)
        result = smoke_world_base(policy, args.split, args.world_ordinal, reference)
    else:
        result = smoke_world_identity(policy, args.split, args.world_ordinal)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
