#!/usr/bin/env python3
"""Audit every formal Step28-v13 row without opening Audit labels.

The hard scan checks all four splits for residual planned identity surfaces
and internal generator markers in model-visible redacted text and seller
profiles.  Label-aware shortcut diagnostics are restricted to train and
development C40 rows.  Audit A/B labels, qrels, controller membership and
identity assets are never opened.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import stat
import uuid
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import step28_v13_common as common
import step28_v13_review_smoke_rows_for_shortcuts as review


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = (
    ROOT
    / "reports"
    / "step28_synthetic_chinese_dataset"
    / "v13_training_ready_v1_2_order_repair_20260731"
)
DEFAULT_OUTPUT = (
    ROOT
    / "reports"
    / "step28_synthetic_chinese_dataset"
    / "post_release_audits"
    / "formal_order_repair_row_shortcut_audit_v1_20260731.json"
)
ALL_SPLITS = ("train", "development", "audit_a", "audit_b")
SUPERVISED_SPLITS = ("train", "development")
REPORT_VERSION = (
    "2026-07-31-step28-v13-formal-order-repair-row-shortcut-audit-v4"
)
EXPECTED_RUN_ID = "v13_training_ready_v1_2_order_repair_20260731"
EXPECTED_RELEASE_STATUS = "PASS_DATASET_ONLY_READY_FOR_M0_M1_M2"
REDACTED_ITEM_KEYS = {
    "world_uid",
    "seller_uid",
    "item_uid",
    "title",
    "description",
}
INTERNAL_MARKER = re.compile(
    r"(?i)(?:\b(?:ctl|ias|id)_[0-9a-f]{16,}\b|"
    r"same_controller|different_controller|positive_target|"
    r"negative_flag|mechanism_slot|controller_uid|identity_asset_uid|"
    r"identity_uid|\blabel\s*[=:])"
)


def _has_reparse_attribute(path: Path) -> bool:
    attributes = getattr(os.lstat(path), "st_file_attributes", 0)
    marker = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
    return bool(marker and attributes & marker)


def require_plain_file(path: Path, *, label: str) -> None:
    if (
        path.is_symlink()
        or not path.is_file()
        or _has_reparse_attribute(path)
    ):
        raise common.ContractError(
            f"{label} is not a plain non-reparse regular file: {path}"
        )


def flatten_strings(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, Mapping):
        output: list[str] = []
        for nested in value.values():
            output.extend(flatten_strings(nested))
        return output
    if isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray)
    ):
        output = []
        for nested in value:
            output.extend(flatten_strings(nested))
        return output
    return []


def scan_redaction_rows(
    *,
    items: Sequence[Mapping[str, Any]],
    profiles: Sequence[Mapping[str, Any]],
    slots: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    item_by_uid: dict[str, Mapping[str, Any]] = {}
    for row in items:
        if set(row) != REDACTED_ITEM_KEYS:
            raise common.ContractError("Redacted item schema drift")
        item_uid = str(row["item_uid"])
        if item_uid in item_by_uid:
            raise common.ContractError("Duplicate redacted item UID")
        item_by_uid[item_uid] = row

    profile_text_by_seller: dict[str, str] = {}
    for row in profiles:
        seller_uid = str(row.get("seller_uid", ""))
        if not seller_uid or seller_uid in profile_text_by_seller:
            raise common.ContractError(
                "Seller profile UID is empty or duplicated"
            )
        profile_text_by_seller[seller_uid] = "\n".join(
            flatten_strings(row)
        )

    forbidden_marker_rows: list[dict[str, str]] = []
    for row in items:
        text = f"{row['title']}\n{row['description']}"
        if INTERNAL_MARKER.search(text):
            forbidden_marker_rows.append(
                {
                    "item_uid": str(row["item_uid"]),
                    "seller_uid": str(row["seller_uid"]),
                }
            )
    for row in profiles:
        text = profile_text_by_seller[str(row["seller_uid"])]
        if INTERNAL_MARKER.search(text):
            forbidden_marker_rows.append(
                {
                    "item_uid": "",
                    "seller_uid": str(row["seller_uid"]),
                }
            )

    raw_surface_residuals: list[dict[str, str]] = []
    canonical_value_residuals: list[dict[str, str]] = []
    seen_slot_uids: set[str] = set()
    for slot in slots:
        slot_uid = str(slot.get("slot_uid", ""))
        if not slot_uid or slot_uid in seen_slot_uids:
            raise common.ContractError(
                "Identity slot UID is empty or duplicated"
            )
        seen_slot_uids.add(slot_uid)
        item_uid = str(slot.get("item_uid", ""))
        seller_uid = str(slot.get("seller_uid", ""))
        field_name = str(slot.get("field_name", ""))
        if item_uid not in item_by_uid or field_name not in {
            "title",
            "description",
        }:
            raise common.ContractError(
                "Identity slot does not resolve to a redacted field"
            )
        item = item_by_uid[item_uid]
        if str(item["seller_uid"]) != seller_uid:
            raise common.ContractError(
                "Identity slot seller does not match redacted item"
            )
        raw_surface = str(slot.get("raw_surface", ""))
        canonical = str(slot.get("downstream_canonical_value", ""))
        if not raw_surface or not canonical:
            raise common.ContractError(
                "Identity slot surface or canonical value is empty"
            )
        visible_field = str(item[field_name])
        visible_profile = profile_text_by_seller.get(seller_uid)
        if visible_profile is None:
            raise common.ContractError(
                "Identity slot seller lacks a visible profile"
            )
        if raw_surface in visible_field or raw_surface in visible_profile:
            raw_surface_residuals.append(
                {
                    "slot_uid": slot_uid,
                    "item_uid": item_uid,
                    "seller_uid": seller_uid,
                    "field_name": field_name,
                }
            )
        if canonical in visible_field or canonical in visible_profile:
            canonical_value_residuals.append(
                {
                    "slot_uid": slot_uid,
                    "item_uid": item_uid,
                    "seller_uid": seller_uid,
                    "field_name": field_name,
                }
            )

    hard_pass = not (
        forbidden_marker_rows
        or raw_surface_residuals
        or canonical_value_residuals
    )
    return {
        "redacted_item_count": len(items),
        "seller_profile_count": len(profiles),
        "planned_identity_slot_count": len(slots),
        "forbidden_internal_marker_count": len(forbidden_marker_rows),
        "raw_surface_residual_count": len(raw_surface_residuals),
        "canonical_value_residual_count": len(
            canonical_value_residuals
        ),
        "hard_leakage_gate_pass": hard_pass,
        "offenders_truncated": {
            "forbidden_internal_markers": forbidden_marker_rows[:20],
            "raw_surface_residuals": raw_surface_residuals[:20],
            "canonical_value_residuals": canonical_value_residuals[:20],
        },
    }


def candidate_order_contract(
    rows: Sequence[Mapping[str, Any]],
    *,
    candidate_key_hex: str,
) -> dict[str, Any]:
    rows_by_world: dict[str, list[str]] = {}
    for row in rows:
        if set(row) != {
            "canonical_pair_uid",
            "world_uid",
            "seller_uid_left",
            "seller_uid_right",
        }:
            raise common.ContractError("Candidate pair schema drift")
        world_uid = str(row["world_uid"])
        rows_by_world.setdefault(world_uid, []).append(
            str(row["canonical_pair_uid"])
        )
    mismatches: list[dict[str, Any]] = []
    for world_uid, observed in rows_by_world.items():
        if len(observed) != 40 or len(set(observed)) != 40:
            raise common.ContractError(
                "Candidate order audit requires 40 unique rows per world"
            )
        expected = sorted(
            observed,
            key=lambda pair_uid: (
                common.hmac_digest(
                    candidate_key_hex,
                    world_uid,
                    "selected_global_rank",
                    pair_uid,
                ),
                pair_uid.encode("utf-8"),
            ),
        )
        if observed != expected:
            mismatches.append(
                {
                    "world_uid": world_uid,
                    "mismatched_positions": sum(
                        left != right
                        for left, right in zip(
                            observed, expected, strict=True
                        )
                    ),
                }
            )
    return {
        "world_count": len(rows_by_world),
        "row_count": len(rows),
        "expected_order": (
            "HMAC-SHA256(candidate_key, world_uid, "
            "selected_global_rank, canonical_pair_uid), then pair UID UTF-8"
        ),
        "mismatched_world_count": len(mismatches),
        "contract_exact": not mismatches,
        "offenders_truncated": mismatches[:20],
    }


def scan_split(
    dataset: Path,
    split: str,
    *,
    candidate_key_hex: str,
) -> dict[str, Any]:
    root = dataset / split
    paths = {
        "redacted_items": root / "observed" / "redacted_items.jsonl",
        "seller_profiles": root / "observed" / "seller_profiles.jsonl",
        "identity_slots": (
            root
            / "private_audit"
            / "renderer_identity_slots.audit.jsonl"
        ),
        "split_manifest": root / "split_manifest.json",
        "candidate_pairs": root / "observed" / "candidate_pairs.csv",
    }
    for name, path in paths.items():
        require_plain_file(path, label=f"{split} {name}")
    manifest = common.load_json(paths["split_manifest"])
    if (
        manifest.get("status") != "PASS_SPLIT_DATASET_READY"
        or manifest.get("split") != split
    ):
        raise common.ContractError(
            f"Formal split manifest is not ready: {split}"
        )
    result = scan_redaction_rows(
        items=review.read_jsonl(paths["redacted_items"]),
        profiles=review.read_jsonl(paths["seller_profiles"]),
        slots=review.read_jsonl(paths["identity_slots"]),
    )
    result["candidate_order_contract"] = candidate_order_contract(
        review.read_csv(paths["candidate_pairs"]),
        candidate_key_hex=candidate_key_hex,
    )
    result["hard_leakage_gate_pass"] = bool(
        result["hard_leakage_gate_pass"]
        and result["candidate_order_contract"]["contract_exact"]
    )
    result["input_sha256"] = {
        name: common.sha256_file(path) for name, path in paths.items()
    }
    return result


def supervised_c40_diagnostics(dataset: Path) -> dict[str, Any]:
    template = common.load_json(
        ROOT / "schema" / "step28_v13_synthetic_text_templates.json"
    )
    style_to_base = review.reachable_style_to_base(template)
    output: dict[str, Any] = {}
    for split in SUPERVISED_SPLITS:
        data = review.load_split(
            dataset,
            split,
            style_to_base,
            oracle_directory_name="private_oracle",
            audit_directory_name="private_audit",
        )
        rows = review.build_pair_rows({split: data}, "c40")
        output[split] = {
            "split_row_counts": data["row_counts"],
            "c40": review.audit_universe(rows),
        }
    return output


def build_report(dataset: Path) -> dict[str, Any]:
    dataset = dataset.resolve()
    release_path = dataset / "release_manifest.json"
    require_plain_file(release_path, label="Formal release manifest")
    release = common.load_json(release_path)
    if release.get("status") != EXPECTED_RELEASE_STATUS:
        raise common.ContractError(
            "Formal release manifest does not grant dataset readiness"
        )
    if (
        release.get("run_id") != EXPECTED_RUN_ID
        or release.get(
            "all_candidate_output_order_replays_exact"
        )
        is not True
        or release.get(
            "parent_order_only_repair_equivalence_exact"
        )
        is not True
        or release.get("repair_equivalence_report", {}).get("status")
        != "PASS_C40_OUTPUT_ORDER_ONLY_REPAIR_EQUIVALENCE"
    ):
        raise common.ContractError(
            "Formal release lacks the frozen order-repair evidence"
        )
    base_policy = common.load_json(
        ROOT / "schema" / "step28_v13_synthetic_chinese_dataset_policy.json"
    )
    candidate_key_hex = str(
        base_policy["randomness"]["formal"]["candidate_key_hex"]
    )
    row_scans = {
        split: scan_split(
            dataset,
            split,
            candidate_key_hex=candidate_key_hex,
        )
        for split in ALL_SPLITS
    }
    hard_pass = all(
        scan["hard_leakage_gate_pass"] for scan in row_scans.values()
    )
    report: dict[str, Any] = {
        "version": REPORT_VERSION,
        "status": (
            "PASS_POST_RELEASE_ROW_LEVEL_HARD_LEAKAGE_SCAN_"
            "VISIBLE_PROXY_DIAGNOSTIC_REPORTED"
            if hard_pass
            else "FAIL_POST_RELEASE_ROW_LEVEL_HARD_LEAKAGE_SCAN"
        ),
        "explicit_boundary": {
            "all_four_split_redacted_rows_opened": True,
            "all_four_split_identity_slot_audits_opened": True,
            "supervised_controller_membership_opened": list(
                SUPERVISED_SPLITS
            ),
            "audit_a_b_controller_membership_opened": False,
            "audit_a_b_labels_opened": False,
            "audit_a_b_qrels_opened": False,
            "audit_a_b_identity_assets_opened": False,
            "formal_dataset_bytes_modified": False,
            "unknown_shortcuts_excluded": False,
            "claim": (
                "The hard scan excludes only residual planned identity "
                "surfaces, canonical values, registered internal markers "
                "and violations of the frozen independent C40 output-order "
                "contract. Train/development visible-proxy models are a "
                "post-release diagnostic, not a new confirmatory gate."
            ),
        },
        "formal_release": {
            "path": release_path.relative_to(ROOT).as_posix(),
            "sha256": common.sha256_file(release_path),
            "status": release["status"],
            "canonical_self_hash": release["canonical_self_hash"],
        },
        "row_level_hard_leakage_scan": row_scans,
        "supervised_c40_visible_proxy_diagnostics": (
            supervised_c40_diagnostics(dataset)
        ),
    }
    report["canonical_self_hash"] = common.canonical_sha256(report)
    return report


def require_output_outside_dataset(dataset: Path, output: Path) -> None:
    dataset = dataset.resolve()
    output = output.resolve()
    try:
        output.relative_to(dataset)
    except ValueError:
        return
    raise common.ContractError(
        "Post-release audit output must remain outside formal dataset root"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset = args.dataset.resolve()
    output = args.output.resolve()
    require_output_outside_dataset(dataset, output)
    if output.exists():
        raise FileExistsError(
            f"Refusing to overwrite post-release audit: {output}"
        )
    report = build_report(dataset)
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = output.parent / f".row-audit-{uuid.uuid4().hex[:10]}.tmp"
    common.write_json(stage, report)
    if common.load_json(stage) != report:
        raise common.ContractError("Post-release audit write drift")
    common.atomic_rename_no_replace(stage, output)
    print(
        json.dumps(
            {
                "status": report["status"],
                "output": output.relative_to(ROOT).as_posix(),
                "canonical_self_hash": report["canonical_self_hash"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    if not report["status"].startswith("PASS_"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
