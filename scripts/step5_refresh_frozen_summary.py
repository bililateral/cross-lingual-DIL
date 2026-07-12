#!/usr/bin/env python3
"""Refresh Step 5 summary metadata from the active frozen label files.

This utility never rebuilds or rewrites frozen labels. It exists for later
train-only expansions whose canonical label CSV is intentionally updated after
the original Step 5 freeze.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import step5_freeze_silver_labels as step5


ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Refresh Step 5 summary without modifying frozen labels.")
    parser.add_argument("--reason", required=True)
    parser.add_argument("--boundary-id", required=True)
    parser.add_argument("--summary", type=Path, default=step5.SUMMARY_PATH)
    args = parser.parse_args()

    policy = step5.load_json(step5.POLICY_PATH)
    prior = step5.load_json(args.summary) if args.summary.exists() else {}
    summary = {
        "schema_path": str(step5.POLICY_PATH.relative_to(ROOT)),
        "input_dependencies": policy["input_dependencies"],
        "output_files": {
            "frozen_labels": {pool: str(path.relative_to(ROOT)) for pool, path in step5.OUTPUT_PATHS.items()},
            "summary": str(args.summary.relative_to(ROOT) if args.summary.is_absolute() else args.summary),
        },
        "pool_summaries": {},
        "acceptance_checks": {},
    }

    all_rows: list[dict] = []
    coverage_warnings: list[str] = []
    coverage_errors: list[str] = []
    hashes: dict[str, str] = {}
    for pool, path in step5.OUTPUT_PATHS.items():
        if not path.exists():
            continue
        rows = step5.load_csv(path)
        pool_summary = step5.summarize_pool(rows, policy, 0)
        coverage_results, warnings, errors = step5.evaluate_coverage_requirements(
            pool,
            rows,
            policy,
        )
        pool_summary["coverage_requirement_results"] = coverage_results
        summary["pool_summaries"][pool] = pool_summary
        coverage_warnings.extend(warnings)
        coverage_errors.extend(errors)
        all_rows.extend(rows)
        hashes[pool] = sha256(path)

    non_identifier_strata = set(policy["non_identifier_positive_strata"])
    positive_rows = [
        row
        for row in all_rows
        if step5.as_int(row.get("usable_for_supervision")) == 1 and row.get("review_label") == "positive"
    ]
    non_identifier_positive_count = sum(
        1 for row in positive_rows if row.get("review_stratum") in non_identifier_strata
    )
    non_identifier_positive_share = (
        round(non_identifier_positive_count / len(positive_rows), 6) if positive_rows else None
    )
    summary["acceptance_checks"] = {
        "no_same_alias_in_supervision": not any(
            row.get("candidate_scope") == "same_alias_identity_continuity"
            and step5.as_int(row.get("usable_for_supervision")) == 1
            for row in all_rows
        ),
        "no_soft_same_alias_in_supervision": not any(
            step5.as_int(row.get("soft_same_alias_continuity_bool")) == 1
            and step5.as_int(row.get("usable_for_supervision")) == 1
            for row in all_rows
        ),
        "all_frozen_rows_have_reviewer_id": all(bool(row.get("reviewer_id")) for row in all_rows),
        "all_supervision_rows_have_split_name": all(
            bool(row.get("split_name"))
            for row in all_rows
            if step5.as_int(row.get("usable_for_supervision")) == 1
        ),
        "no_seller_overlap_across_supervision_splits": all(
            overlap == 0
            for pool, pool_summary in summary["pool_summaries"].items()
            if pool != "zh_target_aux"
            for overlap in pool_summary.get("split_seller_overlap_counts", {}).values()
        ),
        "no_normalized_alias_overlap_across_supervision_splits": all(
            overlap == 0
            for pool, pool_summary in summary["pool_summaries"].items()
            if pool != "zh_target_aux"
            for overlap in pool_summary.get("split_alias_overlap_counts", {}).values()
        ),
        "global_positive_supervision_count": len(positive_rows),
        "global_non_identifier_positive_count": non_identifier_positive_count,
        "global_non_identifier_positive_share": non_identifier_positive_share,
        "non_identifier_positive_share_pass": (
            True if non_identifier_positive_share is None else non_identifier_positive_share >= 0.3
        ),
        "coverage_requirement_warning_count": len(coverage_warnings),
        "coverage_requirement_error_count": len(coverage_errors),
        "coverage_requirements_pass": len(coverage_errors) == 0,
        "coverage_requirement_warnings": coverage_warnings,
        "coverage_requirement_errors": coverage_errors,
    }

    for key, value in prior.items():
        if key not in summary:
            summary[key] = value
    summary["summary_refresh"] = {
        "boundary_id": args.boundary_id,
        "reason": args.reason,
        "frozen_labels_modified": False,
        "frozen_label_sha256": hashes,
    }

    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if coverage_errors:
        raise SystemExit("Refreshed Step 5 summary has coverage errors:\n- " + "\n- ".join(coverage_errors))
    print(json.dumps(summary["summary_refresh"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
