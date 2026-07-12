#!/usr/bin/env python3
"""Build a hash-verified explicit allow-list manifest from Step11 summaries."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from immutable_artifact_io import csv_bytes, json_bytes, write_immutable_bundle


ROOT = Path(__file__).resolve().parent.parent
MANIFEST_VERSION = "step11_explicit_summary_manifest_v2_hash_closed"
PUBLICATION_REQUIRED_ACCEPTANCE_CHECKS = {
    "pair_rows_scored",
    "feature_names_resolved_for_scorer",
    "all_pair_rows_core_transfer_eligible",
    "graph_primary_threshold_present",
    "cluster_files_emitted_for_all_thresholds",
    "posthoc_frozen_label_audit_nonempty",
    "graph_primary_threshold_not_above_score_ceiling",
    "graph_primary_threshold_has_candidate_edges",
    "graph_primary_threshold_has_post_filter_edges",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--summary", action="append", required=True, dest="summaries")
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument(
        "--validation-mode",
        choices=("clean_topology", "identifier_assisted_operational"),
        help="Require every summary runtime policy to use this graph-validation mode.",
    )
    parser.add_argument(
        "--expected-scorer-token",
        action="append",
        default=[],
        help="Repeat to require an exact scorer-token roster.",
    )
    parser.add_argument("--allow-diagnostic", action="store_true")
    parser.add_argument(
        "--publication-v6",
        action="store_true",
        help=(
            "Enable fail-closed publication-v6 contracts: validation mode and exact unique "
            "scorer roster are mandatory, and every summary must prove frozen-input verification."
        ),
    )
    return parser.parse_args()


def resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def flatten_output_paths(value: object) -> list[str]:
    if isinstance(value, str):
        return [value] if value else []
    if isinstance(value, dict):
        paths: list[str] = []
        for nested in value.values():
            paths.extend(flatten_output_paths(nested))
        return paths
    if isinstance(value, list):
        paths: list[str] = []
        for nested in value:
            paths.extend(flatten_output_paths(nested))
        return paths
    return []


def validate_publication_args(args: argparse.Namespace) -> None:
    if args.publication_v6 and not args.validation_mode:
        raise SystemExit("--publication-v6 requires --validation-mode")
    if args.publication_v6 and not args.expected_scorer_token:
        raise SystemExit("--publication-v6 requires one-or-more --expected-scorer-token values")
    if len(args.expected_scorer_token) != len(set(args.expected_scorer_token)):
        raise SystemExit("Duplicate --expected-scorer-token values are not allowed")


def validate_publication_summary(
    summary: dict,
    runtime_policy: dict,
    *,
    summary_path: Path,
    runtime_policy_path: Path,
) -> None:
    publication = (
        (runtime_policy.get("scorer_selection", {}) or {}).get(
            "publication_validation", {}
        )
        or {}
    )
    if publication.get("selection_mode") != "explicit_allowlist_only":
        raise ValueError(
            f"Step11 runtime policy is not publication explicit-only: {runtime_policy_path}"
        )
    if bool(publication.get("auto_selector_allowed", True)):
        raise ValueError(
            f"Step11 runtime policy still allows auto selection: {runtime_policy_path}"
        )
    frozen_verification = summary.get("frozen_input_verification", {}) or {}
    if not bool(frozen_verification.get("enabled", False)) or int(
        frozen_verification.get("verified_file_count", 0) or 0
    ) <= 0:
        raise ValueError(
            f"Step11 summary has no successful frozen-input verification: {summary_path}"
        )
    acceptance = summary.get("acceptance_checks", {}) or {}
    failed_checks = sorted(
        check
        for check in PUBLICATION_REQUIRED_ACCEPTANCE_CHECKS
        if not bool(acceptance.get(check, False))
    )
    if failed_checks:
        raise ValueError(
            "Step11 publication summary has failed acceptance checks "
            f"{failed_checks}: {summary_path}"
        )
    posthoc = summary.get("posthoc_frozen_label_audit", {}) or {}
    if bool(posthoc.get("used_by_model_features", True)) or bool(
        posthoc.get("used_by_graph_filter_decisions", True)
    ):
        raise ValueError(
            f"Step11 publication summary violates posthoc-label isolation: {summary_path}"
        )
    if int(posthoc.get("auditable_reviewed_label_count", 0) or 0) <= 0:
        raise ValueError(
            f"Step11 publication summary has an empty frozen-label audit: {summary_path}"
        )


def main() -> None:
    args = parse_args()
    validate_publication_args(args)
    summary_paths = [resolve(value) for value in args.summaries]
    if len(summary_paths) != len(set(summary_paths)):
        raise SystemExit("Duplicate --summary paths are not allowed")
    records = []
    for summary_path in summary_paths:
        if not summary_path.exists():
            raise FileNotFoundError(summary_path)
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        selected = summary.get("selected_scorer", {}) or {}
        scorer_token = str(selected.get("scorer_token", "")).strip()
        if not scorer_token:
            raise ValueError(f"Step11 summary has no selected_scorer.scorer_token: {summary_path}")
        runtime_policy_path = resolve(str(summary.get("policy_path", "")))
        if not runtime_policy_path.exists():
            raise FileNotFoundError(
                f"Step11 summary runtime policy is missing: {runtime_policy_path}"
            )
        runtime_policy = json.loads(runtime_policy_path.read_text(encoding="utf-8"))
        gate = runtime_policy.get("step15_v6_validation_gate", {}) or {}
        publication = (
            (runtime_policy.get("scorer_selection", {}) or {}).get("publication_validation", {})
            or {}
        )
        if args.publication_v6:
            validate_publication_summary(
                summary,
                runtime_policy,
                summary_path=summary_path,
                runtime_policy_path=runtime_policy_path,
            )
        mode = str(gate.get("graph_validation_mode") or publication.get("graph_validation_mode") or "")
        if args.validation_mode and mode != args.validation_mode:
            raise ValueError(
                f"Step11 validation mode mismatch for {summary_path}: "
                f"expected={args.validation_mode!r} actual={mode!r}"
            )
        if args.publication_v6 and mode == "clean_topology":
            expected_scope = (
                "identifier_free_scoring_and_graph_filter_conditional_on_fixed_candidate_universe"
            )
            if str(gate.get("scientific_scope", "")) != expected_scope:
                raise ValueError(
                    f"Step11 clean publication scope is not explicit: {runtime_policy_path}"
                )
        diagnostic_override = bool(gate.get("diagnostic_override_used", False))
        if diagnostic_override and not args.allow_diagnostic:
            raise ValueError(f"Refusing diagnostic non-promoted Step11 summary: {summary_path}")
        output_values = flatten_output_paths(summary.get("output_paths", {}))
        if str(summary.get("output_paths", {}).get("summary", "")) != str(
            summary_path.relative_to(ROOT)
        ):
            raise ValueError(f"Step11 summary output_paths.summary is not self-consistent: {summary_path}")
        output_records = []
        for value in sorted(set(output_values)):
            path = resolve(value)
            if not path.exists():
                raise FileNotFoundError(f"Step11 summary references a missing output: {path}")
            output_records.append(
                {
                    "path": str(path.relative_to(ROOT)),
                    "sha256": sha256(path),
                    "size_bytes": path.stat().st_size,
                }
            )
        records.append(
            {
                "run_id": args.run_id,
                "summary_path": str(summary_path.relative_to(ROOT)),
                "summary_sha256": sha256(summary_path),
                "scoring_experiment_name": selected.get("source_experiment_name", ""),
                "scorer_token": scorer_token,
                "scorer_family": selected.get("scorer_family", ""),
                "selection_mode": selected.get("selection_mode", ""),
                "graph_validation_mode": mode,
                "runtime_policy_path": str(runtime_policy_path.relative_to(ROOT)),
                "runtime_policy_sha256": sha256(runtime_policy_path),
                "diagnostic_override_used": diagnostic_override,
                "output_paths_json": json.dumps(summary.get("output_paths", {}), sort_keys=True),
                "output_file_records_json": json.dumps(output_records, sort_keys=True),
            }
        )
    actual_tokens = {record["scorer_token"] for record in records}
    actual_token_list = [record["scorer_token"] for record in records]
    if len(actual_token_list) != len(set(actual_token_list)):
        duplicates = sorted(
            token for token in set(actual_token_list) if actual_token_list.count(token) > 1
        )
        raise ValueError(f"Duplicate scorer tokens are not allowed: {duplicates}")
    expected_tokens = set(args.expected_scorer_token)
    if expected_tokens and actual_tokens != expected_tokens:
        raise ValueError(
            "Step11 explicit roster mismatch: "
            f"missing={sorted(expected_tokens - actual_tokens)} "
            f"extra={sorted(actual_tokens - expected_tokens)}"
        )
    output_json = resolve(args.output_json)
    output_csv = resolve(args.output_csv)
    manifest_csv = csv_bytes(records, list(records[0]), encoding="utf-8-sig")
    payload = {
        "manifest_version": MANIFEST_VERSION,
        "run_id": args.run_id,
        "selection_mode": "explicit_allowlist_only",
        "summary_count": len(records),
        "graph_validation_mode": args.validation_mode,
        "publication_v6": bool(args.publication_v6),
        "expected_scorer_tokens": sorted(expected_tokens),
        "summaries": records,
        "manifest_csv_path": str(output_csv.relative_to(ROOT)),
        "manifest_csv_sha256": hashlib.sha256(manifest_csv).hexdigest(),
        "manifest_csv_row_count": len(records),
        "rule": "Downstream audit must read only these summaries and each summary's output_paths; reports-directory globbing is prohibited.",
    }
    canonical = json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    payload["manifest_sha256"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    write_immutable_bundle(
        [
            (output_csv, manifest_csv),
            (output_json, json_bytes(payload, ensure_ascii=False, indent=2)),
        ]
    )
    print(json.dumps({"manifest": str(output_json.relative_to(ROOT)), "summary_count": len(records)}, indent=2))


if __name__ == "__main__":
    main()
