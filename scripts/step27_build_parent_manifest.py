#!/usr/bin/env python3
"""Freeze Step27 canonical pairs, four-fold components, and matched parent pairs."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

import step27_common as common
import step15_build_v7_clean_embedding_cache as redaction


def parent_record(
    row: dict,
    *,
    track: str,
    role: str,
    matched_index: int,
    fold: int,
    shape: dict,
) -> dict:
    return {
        "track": track,
        "matched_set_id": f"{track}_matched_{matched_index:03d}",
        "parent_role": role,
        "parent_pair_uid": row["pair_uid"],
        "review_label": row["review_label"],
        "seller_uid_left": row["seller_uid_left"],
        "seller_uid_right": row["seller_uid_right"],
        "split_name": row["split_name"],
        "component_id": row["component_id"],
        "fold": str(fold),
        "evidence_type": row.get("evidence_type", ""),
        "review_stratum": row.get("review_stratum", ""),
        "label_tier": row.get("label_tier", ""),
        "silver_train_only": row.get("silver_train_only", ""),
        "parent_training_sample_weight": f"{float(row.get('training_sample_weight') or 1.0):.12f}",
        "usable_for_supervision": row.get("usable_for_supervision", ""),
        "usable_for_core_transfer": row.get("usable_for_core_transfer", ""),
        "synthetic_train_only": "0",
        "match_stratum_family": shape["stratum_family"],
        "match_clean_text_length_bin": str(shape["text_length_bin"]),
        "match_clean_segment_count_bin": str(shape["segment_count_bin"]),
        "match_clean_field_missingness_pattern": shape["missingness_pattern"],
    }


def stratum_family(row: dict) -> str:
    value = f"{row.get('review_stratum', '')}|{row.get('evidence_type', '')}".casefold()
    if any(token in value for token in ("identifier", "contact", "pgp", "wallet")):
        return "identifier_or_contact"
    if any(token in value for token in ("clone", "template")):
        return "template_clone"
    if "structural" in value or "component" in value:
        return "structural_or_component"
    if "semantic" in value:
        return "semantic"
    return "other"


def pair_shape(
    row: dict,
    profiles: dict[str, dict],
    signal_literals: dict[str, list[str]],
    fields: list[str],
) -> dict:
    cleaned = []
    for seller_uid in (row["seller_uid_left"], row["seller_uid_right"]):
        profile = profiles.get(seller_uid)
        if profile is None:
            raise ValueError(f"Step27 matching profile is missing: {seller_uid}")
        clean_fields, _ = common.clean_profile_fields(profile, fields, signal_literals)
        cleaned.append(clean_fields)
    total_length = sum(len(common.render_profile_text(value)) for value in cleaned)
    total_segments = sum(
        len(common.split_segments(value.get(field, "")))
        for value in cleaned
        for field in fields
    )
    pattern = "".join(
        "1" if str(value.get(field, "")).strip() else "0"
        for value in cleaned
        for field in fields
    )
    return {
        "stratum_family": stratum_family(row),
        "text_length_bin": int(math.log2(max(total_length, 1))),
        "segment_count_bin": min(total_segments // 4, 20),
        "missingness_pattern": pattern,
    }


def match_negatives(
    positives: list[dict],
    negatives: list[dict],
    shapes: dict[str, dict],
    seed: int,
    namespace: str,
    folds: dict[str, int],
    positive_components: set[str],
) -> list[dict]:
    selected = []
    used_pairs = set()
    used_components = set()
    for positive in positives:
        target = shapes[positive["pair_uid"]]

        def cost(row: dict) -> tuple:
            shape = shapes[row["pair_uid"]]
            family_penalty = 0 if shape["stratum_family"] == target["stratum_family"] else 1
            missingness_distance = sum(
                left != right
                for left, right in zip(
                    shape["missingness_pattern"], target["missingness_pattern"], strict=True
                )
            )
            tie = hashlib.sha256(
                f"{seed}|{namespace}|{positive['pair_uid']}|{row['pair_uid']}".encode("utf-8")
            ).hexdigest()
            return (
                row["component_id"] in positive_components,
                row["component_id"] in used_components,
                family_penalty,
                abs(shape["text_length_bin"] - target["text_length_bin"]),
                abs(shape["segment_count_bin"] - target["segment_count_bin"]),
                missingness_distance,
                tie,
            )

        positive_fold = folds[positive["component_id"]]
        available = [
            row
            for row in negatives
            if row["pair_uid"] not in used_pairs
            and folds[row["component_id"]] == positive_fold
        ]
        if not available:
            raise ValueError(
                "Step27 exhausted same-fold reviewed negative parents during matched selection: "
                f"positive={positive['pair_uid']} fold={positive_fold}"
            )
        chosen = min(available, key=cost)
        selected.append(chosen)
        used_pairs.add(chosen["pair_uid"])
        used_components.add(chosen["component_id"])
    return selected


def select_matched_track(
    rows: list[dict],
    *,
    track: str,
    positive_cap: int,
    negative_cap: int,
    seed: int,
    folds: dict[str, int],
    silver: bool,
    positive_evidence_allowlist: set[str] | None,
    positive_review_stratum_allowlist: set[str] | None,
    shapes: dict[str, dict],
) -> list[dict]:
    candidates = [
        row
        for row in rows
        if row["split_name"] == "train"
        and common.bool_value(row.get("silver_train_only")) is silver
    ]
    positives = [row for row in candidates if row["review_label"] == "positive"]
    if positive_evidence_allowlist is not None:
        positives = [
            row for row in positives if row.get("evidence_type") in positive_evidence_allowlist
        ]
    if positive_review_stratum_allowlist is not None:
        positives = [
            row
            for row in positives
            if row.get("review_stratum") in positive_review_stratum_allowlist
        ]
    negatives = [row for row in candidates if row["review_label"] == "negative"]
    positive_selected = common.balanced_parent_select(
        positives, positive_cap, seed, f"{track}:positive"
    )
    positive_components = {row["component_id"] for row in positive_selected}
    positive_selected = positive_selected[: min(positive_cap, negative_cap)]
    negative_selected = match_negatives(
        positive_selected,
        negatives,
        shapes,
        seed,
        f"{track}:negative_match",
        folds,
        positive_components,
    )
    matched_count = min(len(positive_selected), len(negative_selected), positive_cap, negative_cap)
    minimum = 10 if track == "primary" else 1
    if matched_count < minimum:
        raise ValueError(
            f"Step27 {track} has only {matched_count} matched positive/negative parents; "
            f"minimum={minimum}"
        )
    output: list[dict] = []
    for index in range(matched_count):
        positive = positive_selected[index]
        negative = negative_selected[index]
        if folds[positive["component_id"]] != folds[negative["component_id"]]:
            raise AssertionError("Step27 matched positive/negative parents cross OOF folds")
        relation = (
            "same_component_fallback"
            if positive["component_id"] == negative["component_id"]
            else "distinct_component_preferred"
        )
        for row, role in ((positive, "positive"), (negative, "negative")):
            component_id = row["component_id"]
            if component_id not in folds:
                raise ValueError(f"Step27 parent component has no fixed fold: {row['pair_uid']}")
            record = parent_record(
                row,
                track=track,
                role=role,
                matched_index=index,
                fold=folds[component_id],
                shape=shapes[row["pair_uid"]],
            )
            record["matched_component_relation"] = relation
            output.append(record)
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", default=str(common.DEFAULT_POLICY))
    parser.add_argument("--validate-config-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    policy_path, policy = common.load_policy(args.policy)
    fold_count, fold_seed = common.fold_config(policy)
    limits = common.generation_limits(policy)
    common.transform_schedule(policy, limits["primary_variants"])
    if args.validate_config_only:
        print(
            json.dumps(
                {
                    "status": "pass",
                    "fold_count": fold_count,
                    "primary_child_cap_per_seed": limits["primary_child_cap"],
                    "silver_sensitivity_isolated": True,
                    "numerical_execution_performed": False,
                },
                indent=2,
            )
        )
        return

    input_paths = [
        common.policy_input(policy, "frozen_labels", "zh_frozen_labels"),
        common.policy_input(policy, "evidence_labels", "zh_evidence_labels"),
        common.policy_input(policy, "component_assignments"),
        common.policy_input(policy, "seller_profiles", "zh_seller_profiles"),
        common.policy_input(policy, "item_identity_signals", "zh_item_identity_signals"),
    ]
    pinned_hashes = {
        "zh_frozen_labels_sha256": input_paths[0],
        "zh_evidence_labels_sha256": input_paths[1],
        "seller_component_assignments_sha256": input_paths[2],
    }
    for policy_key, path in pinned_hashes.items():
        expected_sha256 = str(policy.get("inputs", {}).get(policy_key, ""))
        if not expected_sha256 or common.sha256_file(path) != expected_sha256:
            raise ValueError(f"Step27 pinned input hash changed: {policy_key} -> {path}")
    producer_path = Path(__file__).resolve()
    common_path = Path(common.__file__).resolve()
    identity = {
        "stage": "step27_parent_manifest",
        "policy_sha256": common.sha256_file(policy_path),
        "producer_sha256": common.sha256_file(producer_path),
        "common_sha256": common.sha256_file(common_path),
        "shared_dependency_sha256": common.shared_dependency_hashes(),
        "inputs": common.records_for(input_paths),
        "fold_count": fold_count,
        "fold_seed": fold_seed,
        "limits": limits,
    }
    root = common.parent_root(policy)
    manifest_path = root / "manifest.json"
    existing = common.assert_existing_manifest_identity(manifest_path, identity)
    if existing is not None:
        print(json.dumps({"status": "identical_replay", "manifest": common.relative(manifest_path)}, indent=2))
        return
    if root.exists() and any(root.iterdir()):
        raise ValueError(f"Incomplete or foreign Step27 parent root exists: {root}")

    canonical = common.canonical_rows(policy, {"train", "valid", "test"})
    expected_boundary = policy["canonical_chinese_boundary"]["split_counts"]
    for split_name, expected in expected_boundary.items():
        split_rows = [row for row in canonical if row["split_name"] == split_name]
        observed = {
            "rows": len(split_rows),
            "positive": sum(row["review_label"] == "positive" for row in split_rows),
            "negative": sum(row["review_label"] == "negative" for row in split_rows),
        }
        if observed != expected:
            raise ValueError(
                f"Step27 canonical Chinese boundary changed for {split_name}: "
                f"expected={expected} observed={observed}"
            )
    folds = common.build_fixed_component_folds(canonical, fold_count, fold_seed)
    train_rows = [row for row in canonical if row["split_name"] == "train"]
    profiles = common.load_profiles_index(
        common.policy_input(policy, "seller_profiles", "zh_seller_profiles")
    )
    signal_literals, _ = redaction.signal_literals_by_seller(
        common.policy_input(policy, "item_identity_signals", "zh_item_identity_signals")
    )
    fields = common.text_fields(policy)
    shapes = {
        row["pair_uid"]: pair_shape(row, profiles, signal_literals, fields)
        for row in train_rows
    }

    primary = select_matched_track(
        train_rows,
        track="primary",
        positive_cap=limits["primary_positive_parents"],
        negative_cap=limits["primary_negative_parents"],
        seed=fold_seed,
        folds=folds,
        silver=False,
        positive_evidence_allowlist=None,
        positive_review_stratum_allowlist=None,
        shapes=shapes,
    )
    if len(primary) != 2 * limits["primary_positive_parents"]:
        raise ValueError(
            "Step27 primary parent cohort does not match the preregistered exact count: "
            f"expected={2 * limits['primary_positive_parents']} observed={len(primary)}"
        )
    silver = select_matched_track(
        train_rows,
        track="silver_sensitivity",
        positive_cap=limits["silver_positive_parents"],
        negative_cap=limits["silver_negative_parents"],
        seed=fold_seed + 1,
        folds=folds,
        silver=True,
        positive_evidence_allowlist={
            "same_controller_direct_identifier",
            "same_controller_component_anchor",
        },
        positive_review_stratum_allowlist={"silver_direct_or_contact"},
        shapes=shapes,
    )
    common.ensure_track_isolation(primary, silver)
    if any(row["split_name"] != "train" for row in primary + silver):
        raise ValueError("Step27 selected a non-train synthetic parent")

    canonical_rows = []
    for row in canonical:
        canonical_rows.append(
            {
                "pair_uid": row["pair_uid"],
                "split_name": row["split_name"],
                "review_label": row["review_label"],
                "seller_uid_left": row["seller_uid_left"],
                "seller_uid_right": row["seller_uid_right"],
                "component_id": row["component_id"],
                "fold": str(folds[row["component_id"]]) if row["split_name"] == "train" else "",
                "evidence_type": row.get("evidence_type", ""),
                "label_tier": row.get("label_tier", ""),
                "silver_train_only": row.get("silver_train_only", ""),
                "training_sample_weight": f"{float(row.get('training_sample_weight') or 1.0):.12f}",
            }
        )
    fold_rows = []
    for component_id, fold in sorted(folds.items()):
        component_rows = [row for row in train_rows if row["component_id"] == component_id]
        fold_rows.append(
            {
                "component_id": component_id,
                "fold": str(fold),
                "pair_count": str(len(component_rows)),
                "positive_count": str(sum(row["review_label"] == "positive" for row in component_rows)),
                "negative_count": str(sum(row["review_label"] == "negative" for row in component_rows)),
                "assignment_seed": str(fold_seed),
            }
        )
    canonical_path = root / "canonical_pairs.csv"
    folds_path = root / "fixed_four_fold_components.csv"
    primary_path = root / "primary_matched_parents.csv"
    silver_path = root / "silver_sensitivity_matched_parents.csv"
    summary_path = root / "summary.json"
    common.write_csv_immutable(canonical_path, canonical_rows)
    common.write_csv_immutable(folds_path, fold_rows)
    common.write_csv_immutable(primary_path, primary)
    common.write_csv_immutable(silver_path, silver)

    summary = {
        "step": "step27_parent_manifest",
        "status": "frozen_train_only_parent_manifest",
        "canonical_split_counts": dict(sorted(Counter(row["split_name"] for row in canonical).items())),
        "canonical_label_counts": dict(sorted(Counter(row["review_label"] for row in canonical).items())),
        "train_component_count": len(folds),
        "fold_count": fold_count,
        "fold_seed": fold_seed,
        "fold_pair_counts": dict(
            sorted(Counter(row["fold"] for row in canonical_rows if row["split_name"] == "train").items())
        ),
        "primary_parent_count": len(primary),
        "primary_matched_set_count": len(primary) // 2,
        "primary_parent_label_counts": dict(sorted(Counter(row["review_label"] for row in primary).items())),
        "primary_match_component_relations": dict(
            sorted(Counter(row["matched_component_relation"] for row in primary[::2]).items())
        ),
        "silver_sensitivity_parent_count": len(silver),
        "silver_sensitivity_matched_set_count": len(silver) // 2,
        "silver_match_component_relations": dict(
            sorted(Counter(row["matched_component_relation"] for row in silver[::2]).items())
        ),
        "primary_and_silver_physically_separate": True,
        "valid_or_test_parent_count": 0,
        "children_must_inherit_parent_fold_component_label": True,
        "outputs": {
            "canonical_pairs": common.relative(canonical_path),
            "fixed_folds": common.relative(folds_path),
            "primary_parents": common.relative(primary_path),
            "silver_sensitivity_parents": common.relative(silver_path),
        },
    }
    summary["summary_content_sha256"] = common.canonical_hash(summary)
    common.write_json_immutable(summary_path, summary)
    outputs = [canonical_path, folds_path, primary_path, silver_path, summary_path]
    common.write_manifest_immutable(
        manifest_path,
        stage="step27_parent_manifest",
        identity=identity,
        inputs=input_paths,
        outputs=outputs,
        extra={"summary_sha256": common.sha256_file(summary_path)},
    )
    print(json.dumps({"status": "pass", **summary, "manifest": common.relative(manifest_path)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
