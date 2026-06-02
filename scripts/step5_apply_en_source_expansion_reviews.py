from __future__ import annotations

import argparse
import csv
import json
import shutil
from collections import Counter
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
POLICY_PATH = ROOT / "schema" / "step5_en_source_expansion_policy.json"
VALID_LABELS = {"positive", "negative", "uncertain"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Apply reviewed English source-domain expansion labels to the active Step 5 English queue."
    )
    parser.add_argument(
        "--policy-path",
        default=str(POLICY_PATH),
        help="Path to the English source expansion policy JSON.",
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


def main() -> None:
    args = parse_args()
    policy_path = Path(args.policy_path)
    if not policy_path.is_absolute():
        policy_path = ROOT / policy_path
    policy = load_json(policy_path)
    inputs = policy["inputs"]

    queue_path = ROOT / policy["outputs"]["targeted_review_queue"]
    summary_path = ROOT / policy["outputs"]["application_summary"]
    active_path = ROOT / inputs["active_review_queue"]

    queue_rows, _queue_fields = load_csv(queue_path)
    active_rows, active_fieldnames = load_csv(active_path)
    active_index = {row["pair_uid"]: row for row in active_rows}

    incomplete = [row for row in queue_rows if not is_complete(row)]
    if args.require_complete and incomplete:
        raise SystemExit(f"English source expansion review incomplete: {len(incomplete)} rows lack a valid final label.")
    reviewed_rows = [row for row in queue_rows if is_complete(row)]
    if not reviewed_rows:
        raise SystemExit("No reviewed English source expansion rows found to apply.")

    active_backup = backup(active_path, "step5_en_source_expansion_apply")
    label_counts = Counter()
    bucket_counts = Counter()
    updated_active_rows = []
    skipped_rows = []

    for queue_row in reviewed_rows:
        pair_uid = queue_row["pair_uid"]
        active_row = active_index.get(pair_uid)
        if active_row is None:
            skipped_rows.append({"pair_uid": pair_uid, "reason": "missing_active_queue_row"})
            continue

        old_label = normalize_label(active_row.get("review_label"))
        if old_label and not args.allow_overwrite:
            skipped_rows.append(
                {
                    "pair_uid": pair_uid,
                    "reason": "active_queue_already_labeled",
                    "active_review_label": active_row.get("review_label", ""),
                    "expansion_review_label": queue_row.get("review_label", ""),
                }
            )
            continue

        label = normalize_label(queue_row.get("review_label"))
        active_row["review_status"] = "reviewed"
        active_row["review_label"] = label
        active_row["reviewer_id"] = normalize_text(queue_row.get("reviewer_id"))
        active_row["review_notes"] = normalize_text(queue_row.get("review_notes"))
        label_counts[label] += 1
        bucket_counts[queue_row.get("target_bucket", "")] += 1
        updated_active_rows.append(pair_uid)

    write_csv(active_path, active_rows, active_fieldnames)

    summary = {
        "policy_path": str(policy_path.relative_to(ROOT)),
        "english_source_expansion_queue": str(queue_path.relative_to(ROOT)),
        "active_review_queue": str(active_path.relative_to(ROOT)),
        "active_backup_path": str(active_backup.relative_to(ROOT)),
        "require_complete": bool(args.require_complete),
        "allow_overwrite": bool(args.allow_overwrite),
        "queue_row_count": len(queue_rows),
        "reviewed_row_count": len(reviewed_rows),
        "changed_label_count": sum(label_counts.values()),
        "updated_active_count": len(updated_active_rows),
        "skipped_row_count": len(skipped_rows),
        "label_counts": dict(label_counts),
        "target_bucket_counts": dict(bucket_counts),
        "updated_active_rows": updated_active_rows,
        "skipped_rows": skipped_rows,
        "next_required_steps": [
            "Run scripts/step5_freeze_silver_labels.py.",
            "Audit reports/step5_frozen_silver_summary.json for English train/valid/test size and seller overlap.",
            "Rerun Step 7 semantic extraction/training on Linux before treating source-domain metrics as current.",
        ],
    }
    write_json(summary_path, summary)
    print(f"Backed up active review queue to: {active_backup}")
    print(f"Wrote application summary: {summary_path}")
    print(f"changed_label_count={summary['changed_label_count']} label_counts={dict(label_counts)}")


if __name__ == "__main__":
    main()
