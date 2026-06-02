from __future__ import annotations

import argparse
import csv
import json
import shutil
from collections import Counter
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
POLICY_PATH = ROOT / "schema" / "step5_en_item_identity_expansion_policy.json"
VALID_LABELS = {"positive", "negative", "uncertain"}
STEP4_REVIEW_FIELDS = {"review_status", "review_label", "reviewer_id", "review_notes"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Apply reviewed item-identity expansion rows to Step 4/5 artifacts.")
    parser.add_argument("--policy-path", default=str(POLICY_PATH), help="Path to item-identity expansion policy JSON.")
    parser.add_argument("--require-complete", action="store_true", help="Fail if any queue row is not fully reviewed.")
    parser.add_argument("--allow-overwrite", action="store_true", help="Allow replacing an existing active Step 5 label.")
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
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


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
    return label in VALID_LABELS and status not in {"", "pending"} and bool(reviewer_id)


def project_active_row(row: dict, fieldnames: list[str]) -> dict:
    return {field: row.get(field, "") for field in fieldnames}


def project_candidate_row(row: dict, fieldnames: list[str]) -> dict:
    projected = {field: row.get(field, "") for field in fieldnames}
    for field in STEP4_REVIEW_FIELDS:
        if field in projected:
            projected[field] = "" if field != "review_status" else "pending"
    return projected


def merge_candidate_row(existing: dict, incoming: dict) -> dict:
    merged = dict(existing)
    for field, value in incoming.items():
        if field in STEP4_REVIEW_FIELDS:
            continue
        if value not in {"", None}:
            merged[field] = value
    return merged


def main() -> None:
    args = parse_args()
    policy_path = Path(args.policy_path)
    if not policy_path.is_absolute():
        policy_path = ROOT / policy_path
    policy = load_json(policy_path)
    inputs = policy["inputs"]
    outputs = policy["outputs"]

    queue_path = ROOT / outputs["targeted_review_queue"]
    active_path = ROOT / inputs["active_review_queue"]
    candidate_path = ROOT / inputs["step4_candidates"]
    summary_path = ROOT / outputs["application_summary"]

    queue_rows, _queue_fields = load_csv(queue_path)
    active_rows, active_fields = load_csv(active_path)
    candidate_rows, candidate_fields = load_csv(candidate_path)

    incomplete = [row for row in queue_rows if not is_complete(row)]
    if args.require_complete and incomplete:
        raise SystemExit(f"Item-identity expansion review incomplete: {len(incomplete)} rows lack a valid final label.")

    reviewed_rows = [row for row in queue_rows if is_complete(row)]
    if not reviewed_rows:
        raise SystemExit("No reviewed item-identity rows found to apply.")

    active_backup = backup(active_path, "step5_item_identity_expansion_apply")
    candidate_backup = backup(candidate_path, "step5_item_identity_expansion_apply")

    active_index = {row["pair_uid"]: idx for idx, row in enumerate(active_rows)}
    candidate_index = {row["pair_uid"]: idx for idx, row in enumerate(candidate_rows)}

    label_counts = Counter()
    action_counts = Counter()
    skipped_rows = []
    appended_active = []
    updated_active = []
    appended_candidates = []
    updated_candidates = []

    for queue_row in reviewed_rows:
        pair_uid = queue_row["pair_uid"]
        label = normalize_label(queue_row.get("review_label"))
        label_counts[label] += 1

        incoming_candidate = project_candidate_row(queue_row, candidate_fields)
        candidate_idx = candidate_index.get(pair_uid)
        if candidate_idx is None:
            candidate_rows.append(incoming_candidate)
            candidate_index[pair_uid] = len(candidate_rows) - 1
            appended_candidates.append(pair_uid)
            action_counts["append_step4_candidate"] += 1
        else:
            candidate_rows[candidate_idx] = merge_candidate_row(candidate_rows[candidate_idx], incoming_candidate)
            updated_candidates.append(pair_uid)
            action_counts["update_step4_candidate"] += 1

        active_idx = active_index.get(pair_uid)
        incoming_active = project_active_row(queue_row, active_fields)
        if active_idx is None:
            active_rows.append(incoming_active)
            active_index[pair_uid] = len(active_rows) - 1
            appended_active.append(pair_uid)
            action_counts["append_step5_active_row"] += 1
            continue

        active_row = active_rows[active_idx]
        old_label = normalize_label(active_row.get("review_label"))
        if old_label and not args.allow_overwrite:
            skipped_rows.append(
                {
                    "pair_uid": pair_uid,
                    "reason": "active_queue_already_labeled",
                    "active_review_label": active_row.get("review_label", ""),
                    "item_identity_review_label": queue_row.get("review_label", ""),
                }
            )
            action_counts["skip_active_already_labeled"] += 1
            continue

        for field in active_fields:
            value = incoming_active.get(field, "")
            if field in {"review_status", "review_label", "reviewer_id", "review_notes", "review_stratum", "review_priority"}:
                active_row[field] = value
            elif value not in {"", None}:
                active_row[field] = value
        updated_active.append(pair_uid)
        action_counts["update_step5_active_row"] += 1

    write_csv(candidate_path, candidate_rows, candidate_fields)
    write_csv(active_path, active_rows, active_fields)

    summary = {
        "policy_path": str(policy_path.relative_to(ROOT)),
        "queue_path": str(queue_path.relative_to(ROOT)),
        "active_review_queue": str(active_path.relative_to(ROOT)),
        "step4_candidates": str(candidate_path.relative_to(ROOT)),
        "active_backup_path": str(active_backup.relative_to(ROOT)),
        "candidate_backup_path": str(candidate_backup.relative_to(ROOT)),
        "require_complete": bool(args.require_complete),
        "allow_overwrite": bool(args.allow_overwrite),
        "queue_row_count": len(queue_rows),
        "reviewed_row_count": len(reviewed_rows),
        "label_counts": dict(label_counts),
        "action_counts": dict(action_counts),
        "appended_active_count": len(appended_active),
        "updated_active_count": len(updated_active),
        "appended_candidate_count": len(appended_candidates),
        "updated_candidate_count": len(updated_candidates),
        "skipped_row_count": len(skipped_rows),
        "appended_active_rows": appended_active,
        "updated_active_rows": updated_active,
        "appended_candidate_rows": appended_candidates,
        "updated_candidate_rows": updated_candidates,
        "skipped_rows": skipped_rows,
        "next_required_steps": [
            "Run scripts/step5_freeze_silver_labels.py.",
            "Audit reports/step5_frozen_silver_summary.json for English split size and seller/alias overlap.",
            "Rerun Linux Step 7 before treating source-domain metrics as current.",
        ],
    }
    write_json(summary_path, summary)
    print(f"Backed up active queue to: {active_backup}")
    print(f"Backed up Step 4 candidates to: {candidate_backup}")
    print(f"Applied reviewed rows: labels={dict(label_counts)} actions={dict(action_counts)}")
    print(f"Wrote application summary: {summary_path}")


if __name__ == "__main__":
    main()
