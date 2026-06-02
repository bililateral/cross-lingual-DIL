from __future__ import annotations

import argparse
import csv
import json
import shutil
from collections import Counter
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
POLICY_PATH = ROOT / "schema" / "step5_boundary_expansion_policy.json"
VALID_LABELS = {"positive", "negative", "uncertain"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Copy human-reviewed Step 5 boundary expansion labels back into the active review queue."
    )
    parser.add_argument(
        "--policy-path",
        default=str(POLICY_PATH),
        help="Path to the Step 5 boundary expansion policy JSON.",
    )
    parser.add_argument(
        "--queue-path",
        help="Optional reviewed boundary expansion queue CSV. Defaults to policy outputs.targeted_review_queue.",
    )
    parser.add_argument(
        "--require-complete",
        action="store_true",
        help="Fail unless every row in the boundary expansion queue has a valid final label.",
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


def build_backup(path: Path, suffix: str) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    backup_path = path.with_name(f"{path.stem}.codexbak.{suffix}.{timestamp}{path.suffix}")
    shutil.copy2(path, backup_path)
    return backup_path


def normalize_text(value: object) -> str:
    return str(value or "").strip()


def normalize_label(value: object) -> str:
    return normalize_text(value).lower()


def is_complete_review(row: dict) -> bool:
    label = normalize_label(row.get("review_label"))
    reviewer_id = normalize_text(row.get("reviewer_id"))
    status = normalize_text(row.get("review_status")).lower()
    return label in VALID_LABELS and reviewer_id != "" and status not in {"", "pending"}


def main() -> None:
    args = parse_args()
    policy_path = Path(args.policy_path)
    if not policy_path.is_absolute():
        policy_path = ROOT / policy_path
    policy = load_json(policy_path)

    queue_path = Path(args.queue_path or policy["outputs"]["targeted_review_queue"])
    if not queue_path.is_absolute():
        queue_path = ROOT / queue_path
    active_queue_path = ROOT / policy["inputs"]["active_review_queue"]
    output_summary_path = ROOT / policy["outputs"]["application_summary"]

    boundary_rows, _ = load_csv(queue_path)
    active_rows, active_fieldnames = load_csv(active_queue_path)
    active_index = {row["pair_uid"]: row for row in active_rows}

    pending_rows = [row for row in boundary_rows if not is_complete_review(row)]
    if args.require_complete and pending_rows:
        raise SystemExit(
            "Boundary expansion review is incomplete: "
            f"{len(pending_rows)} rows lack a valid label/reviewer_id. "
            "Finish manual review or rerun without --require-complete to apply reviewed rows only."
        )

    reviewed_rows = [row for row in boundary_rows if is_complete_review(row)]
    if not reviewed_rows:
        raise SystemExit("No reviewed boundary expansion rows found to apply.")

    active_backup = build_backup(active_queue_path, "step5_boundary_expansion_apply")
    changed_rows = []
    label_counts = Counter()
    bucket_counts = Counter()
    skipped_rows = []

    for boundary_row in reviewed_rows:
        pair_uid = boundary_row["pair_uid"]
        active_row = active_index.get(pair_uid)
        if active_row is None:
            raise SystemExit(f"Boundary expansion pair_uid not found in active queue: {pair_uid}")

        old_label = normalize_label(active_row.get("review_label"))
        if old_label and not args.allow_overwrite:
            skipped_rows.append(
                {
                    "pair_uid": pair_uid,
                    "reason": "active_queue_already_labeled",
                    "active_review_label": active_row.get("review_label", ""),
                    "boundary_review_label": boundary_row.get("review_label", ""),
                }
            )
            continue

        previous = {
            "review_status": active_row.get("review_status", ""),
            "review_label": active_row.get("review_label", ""),
            "reviewer_id": active_row.get("reviewer_id", ""),
            "review_notes": active_row.get("review_notes", ""),
        }
        active_row["review_status"] = "reviewed"
        active_row["review_label"] = normalize_label(boundary_row.get("review_label"))
        active_row["reviewer_id"] = normalize_text(boundary_row.get("reviewer_id"))
        active_row["review_notes"] = normalize_text(boundary_row.get("review_notes"))

        label_counts[active_row["review_label"]] += 1
        bucket_counts[boundary_row.get("target_bucket", "")] += 1
        changed_rows.append(
            {
                "pair_uid": pair_uid,
                "target_bucket": boundary_row.get("target_bucket", ""),
                "review_stratum": active_row.get("review_stratum", ""),
                "balanced_review_rank": active_row.get("balanced_review_rank", ""),
                "previous": previous,
                "updated": {
                    "review_status": active_row["review_status"],
                    "review_label": active_row["review_label"],
                    "reviewer_id": active_row["reviewer_id"],
                    "review_notes": active_row["review_notes"],
                },
                "source_step11_prob_positive": boundary_row.get("source_step11_prob_positive", ""),
                "source_step11_graph_filter_retained_bool": boundary_row.get(
                    "source_step11_graph_filter_retained_bool", ""
                ),
                "style_gap_score": boundary_row.get("style_gap_score", ""),
            }
        )

    write_csv(active_queue_path, active_rows, active_fieldnames)

    summary = {
        "policy_path": str(policy_path.relative_to(ROOT)),
        "boundary_expansion_queue": str(queue_path.relative_to(ROOT)),
        "active_review_queue": str(active_queue_path.relative_to(ROOT)),
        "active_backup_path": str(active_backup.relative_to(ROOT)),
        "require_complete": bool(args.require_complete),
        "allow_overwrite": bool(args.allow_overwrite),
        "boundary_queue_row_count": len(boundary_rows),
        "reviewed_boundary_row_count": len(reviewed_rows),
        "pending_boundary_row_count": len(pending_rows),
        "changed_row_count": len(changed_rows),
        "skipped_row_count": len(skipped_rows),
        "label_counts": dict(label_counts),
        "target_bucket_counts": dict(bucket_counts),
        "changed_rows": changed_rows,
        "skipped_rows": skipped_rows,
        "next_required_step": "Run scripts/step5_freeze_silver_labels.py before any Step 7, Step 9, or Step 11 rerun.",
    }
    write_json(output_summary_path, summary)

    print(f"Backed up active review queue to: {active_backup}")
    print(f"Updated active review queue: {active_queue_path}")
    print(f"Wrote boundary expansion apply summary: {output_summary_path}")
    print(f"changed_row_count={len(changed_rows)} label_counts={dict(label_counts)} skipped={len(skipped_rows)}")


if __name__ == "__main__":
    main()
