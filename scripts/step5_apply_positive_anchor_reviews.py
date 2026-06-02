from __future__ import annotations

import argparse
import csv
import json
import shutil
from collections import Counter
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
POLICY_PATH = ROOT / "schema" / "step5_positive_anchor_expansion_policy.json"
VALID_LABELS = {"positive", "negative", "uncertain"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Apply reviewed Step 5 positive-anchor expansion labels to Step 4 candidates and the active Step 5 queue."
    )
    parser.add_argument(
        "--policy-path",
        default=str(POLICY_PATH),
        help="Path to the Step 5 positive-anchor expansion policy JSON.",
    )
    parser.add_argument(
        "--require-complete",
        action="store_true",
        help="Fail unless every targeted queue row has a valid final review label.",
    )
    parser.add_argument(
        "--allow-overwrite",
        action="store_true",
        help="Allow replacing an existing active-queue label for the same pair_uid.",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_csv(path: Path) -> tuple[list[dict], list[str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader), list(reader.fieldnames or [])


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def backup(path: Path, suffix: str) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    backup_path = path.with_name(f"{path.stem}.codexbak.{suffix}.{timestamp}{path.suffix}")
    shutil.copy2(path, backup_path)
    return backup_path


def normalize_text(value: object) -> str:
    return str(value or "").strip()


def normalize_label(value: object) -> str:
    return normalize_text(value).lower()


def is_complete(row: dict) -> bool:
    label = normalize_label(row.get("review_label"))
    status = normalize_text(row.get("review_status")).lower()
    reviewer_id = normalize_text(row.get("reviewer_id"))
    return label in VALID_LABELS and status not in {"", "pending"} and reviewer_id != ""


def max_rank(rows: list[dict]) -> int:
    ranks = []
    for row in rows:
        try:
            ranks.append(int(row.get("balanced_review_rank", "") or 0))
        except ValueError:
            continue
    return max(ranks, default=0)


def main() -> None:
    args = parse_args()
    policy_path = Path(args.policy_path)
    if not policy_path.is_absolute():
        policy_path = ROOT / policy_path
    policy = load_json(policy_path)
    inputs = policy["inputs"]

    queue_path = ROOT / policy["outputs"]["targeted_review_queue"]
    summary_path = ROOT / policy["outputs"]["application_summary"]
    step4_path = ROOT / inputs["step4_candidates"]
    active_path = ROOT / inputs["active_review_queue"]
    step4_schema = load_json(ROOT / inputs["step4_schema"])

    queue_rows, _ = load_csv(queue_path)
    candidate_rows, candidate_fieldnames = load_csv(step4_path)
    active_rows, active_fieldnames = load_csv(active_path)

    incomplete = [row for row in queue_rows if not is_complete(row)]
    if args.require_complete and incomplete:
        raise SystemExit(f"Positive-anchor review incomplete: {len(incomplete)} rows lack a valid final label.")
    reviewed_rows = [row for row in queue_rows if is_complete(row)]
    if not reviewed_rows:
        raise SystemExit("No reviewed positive-anchor rows found to apply.")

    candidate_fieldnames = candidate_fieldnames or step4_schema["candidate_output_fields"]
    candidate_index = {row["pair_uid"]: row for row in candidate_rows}
    active_index = {row["pair_uid"]: row for row in active_rows}
    next_rank = max_rank(active_rows)

    step4_backup = backup(step4_path, "step5_positive_anchor_apply")
    active_backup = backup(active_path, "step5_positive_anchor_apply")

    label_counts = Counter()
    bucket_counts = Counter()
    appended_candidate_rows = []
    appended_active_rows = []
    updated_active_rows = []
    skipped_rows = []

    for queue_row in reviewed_rows:
        pair_uid = queue_row["pair_uid"]
        label = normalize_label(queue_row.get("review_label"))
        reviewer_id = normalize_text(queue_row.get("reviewer_id"))
        notes = normalize_text(queue_row.get("review_notes"))

        if pair_uid not in candidate_index:
            candidate_row = {field: queue_row.get(field, "") for field in candidate_fieldnames}
            candidate_row["review_status"] = "pending"
            candidate_row["review_label"] = ""
            candidate_row["reviewer_id"] = ""
            candidate_row["review_notes"] = ""
            candidate_rows.append(candidate_row)
            candidate_index[pair_uid] = candidate_row
            appended_candidate_rows.append(pair_uid)
        elif queue_row.get("target_bucket") == "positive_component_transitive_closure":
            candidate_index[pair_uid]["candidate_scope"] = queue_row.get(
                "candidate_scope",
                "positive_component_closure_audit",
            )

        active_row = active_index.get(pair_uid)
        if active_row is None:
            next_rank += 1
            active_row = {field: queue_row.get(field, "") for field in active_fieldnames}
            active_row["balanced_review_rank"] = str(next_rank)
            active_rows.append(active_row)
            active_index[pair_uid] = active_row
            appended_active_rows.append(pair_uid)
        else:
            old_label = normalize_label(active_row.get("review_label"))
            if old_label and not args.allow_overwrite:
                skipped_rows.append(
                    {
                        "pair_uid": pair_uid,
                        "reason": "active_queue_already_labeled",
                        "active_review_label": active_row.get("review_label", ""),
                        "positive_anchor_review_label": label,
                    }
                )
                continue
            updated_active_rows.append(pair_uid)

        if queue_row.get("target_bucket") == "positive_component_transitive_closure":
            active_row["candidate_scope"] = queue_row.get("candidate_scope", "positive_component_closure_audit")
        active_row["review_status"] = "reviewed"
        active_row["review_label"] = label
        active_row["reviewer_id"] = reviewer_id
        active_row["review_notes"] = notes
        label_counts[label] += 1
        bucket_counts[queue_row.get("target_bucket", "")] += 1

    write_csv(step4_path, candidate_rows, candidate_fieldnames)
    write_csv(active_path, active_rows, active_fieldnames)

    summary = {
        "policy_path": str(policy_path.relative_to(ROOT)),
        "positive_anchor_queue": str(queue_path.relative_to(ROOT)),
        "step4_candidates": str(step4_path.relative_to(ROOT)),
        "active_review_queue": str(active_path.relative_to(ROOT)),
        "step4_backup_path": str(step4_backup.relative_to(ROOT)),
        "active_backup_path": str(active_backup.relative_to(ROOT)),
        "require_complete": bool(args.require_complete),
        "allow_overwrite": bool(args.allow_overwrite),
        "queue_row_count": len(queue_rows),
        "reviewed_row_count": len(reviewed_rows),
        "changed_label_count": sum(label_counts.values()),
        "appended_candidate_count": len(appended_candidate_rows),
        "appended_active_count": len(appended_active_rows),
        "updated_active_count": len(updated_active_rows),
        "skipped_row_count": len(skipped_rows),
        "label_counts": dict(label_counts),
        "target_bucket_counts": dict(bucket_counts),
        "appended_candidate_rows": appended_candidate_rows,
        "appended_active_rows": appended_active_rows,
        "updated_active_rows": updated_active_rows,
        "skipped_rows": skipped_rows,
        "next_required_steps": [
            "Run scripts/step5_freeze_silver_labels.py.",
            "Because Step 4 may have net-new rows, rerun scripts/step7_run_default_pipeline.py on the Linux runtime without --skip-preview or --skip-semantic.",
        ],
    }
    write_json(summary_path, summary)
    print(f"Backed up Step 4 candidates to: {step4_backup}")
    print(f"Backed up active review queue to: {active_backup}")
    print(f"Wrote application summary: {summary_path}")
    print(
        "changed_label_count="
        f"{summary['changed_label_count']} appended_candidate_count={len(appended_candidate_rows)} "
        f"appended_active_count={len(appended_active_rows)} label_counts={dict(label_counts)}"
    )


if __name__ == "__main__":
    main()
