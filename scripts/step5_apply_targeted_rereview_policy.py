from __future__ import annotations

import argparse
import csv
import json
import shutil
from collections import Counter
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_POLICY_PATH = ROOT / "schema" / "step5_v3_targeted_cleanup_policy.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Apply a targeted Step 5 rereview policy to both the active review queue and the targeted rereview queue."
    )
    parser.add_argument(
        "--policy-path",
        default=str(DEFAULT_POLICY_PATH),
        help="Path to the targeted rereview cleanup policy JSON.",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_csv(path: Path) -> tuple[list[dict], list[str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        return rows, list(reader.fieldnames or [])


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_backup(path: Path, suffix: str) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    backup_path = path.with_name(f"{path.stem}.codexbak.{suffix}.{timestamp}{path.suffix}")
    shutil.copy2(path, backup_path)
    return backup_path


def build_decision_map(policy: dict, rereview_rows: list[dict]) -> dict[str, dict]:
    bucket_defaults = policy.get("bucket_defaults", {})
    pair_overrides = policy.get("pair_overrides", {})
    decisions: dict[str, dict] = {}
    for row in rereview_rows:
        pair_uid = row["pair_uid"]
        bucket = row.get("target_bucket", "")
        decision = dict(bucket_defaults.get(bucket, {}))
        decision.update(pair_overrides.get(pair_uid, {}))
        if "new_review_label" not in decision or "review_notes" not in decision:
            raise SystemExit(f"Missing review decision for pair_uid={pair_uid}")
        decisions[pair_uid] = decision
    return decisions


def apply_decision(row: dict, reviewer_id: str, decision: dict) -> None:
    row["review_status"] = "reviewed"
    row["review_label"] = decision["new_review_label"]
    row["reviewer_id"] = reviewer_id
    row["review_notes"] = decision["review_notes"]


def main() -> None:
    args = parse_args()
    policy_path = Path(args.policy_path)
    if not policy_path.is_absolute():
        policy_path = ROOT / policy_path
    policy = load_json(policy_path)

    active_queue_path = ROOT / policy["inputs"]["active_review_queue"]
    rereview_queue_path = ROOT / policy["inputs"]["targeted_rereview_queue"]
    output_summary_path = ROOT / policy["outputs"]["summary"]
    reviewer_id = policy["reviewer_id"]
    cleanup_version = policy["cleanup_version"]

    active_rows, active_fieldnames = load_csv(active_queue_path)
    rereview_rows, rereview_fieldnames = load_csv(rereview_queue_path)

    decision_map = build_decision_map(policy, rereview_rows)
    active_index = {row["pair_uid"]: row for row in active_rows}
    rereview_index = {row["pair_uid"]: row for row in rereview_rows}

    active_backup = build_backup(active_queue_path, cleanup_version)
    rereview_backup = build_backup(rereview_queue_path, cleanup_version)

    changed_rows = []
    label_counts = Counter()

    for pair_uid, decision in decision_map.items():
        active_row = active_index.get(pair_uid)
        rereview_row = rereview_index.get(pair_uid)
        if active_row is None:
            raise SystemExit(f"pair_uid not found in active review queue: {pair_uid}")
        if rereview_row is None:
            raise SystemExit(f"pair_uid not found in targeted rereview queue: {pair_uid}")

        previous_active = {
            "review_status": active_row.get("review_status", ""),
            "review_label": active_row.get("review_label", ""),
            "reviewer_id": active_row.get("reviewer_id", ""),
            "review_notes": active_row.get("review_notes", ""),
        }
        previous_rereview = {
            "review_status": rereview_row.get("review_status", ""),
            "review_label": rereview_row.get("review_label", ""),
            "reviewer_id": rereview_row.get("reviewer_id", ""),
            "review_notes": rereview_row.get("review_notes", ""),
        }

        apply_decision(active_row, reviewer_id, decision)
        apply_decision(rereview_row, reviewer_id, decision)

        label_counts[decision["new_review_label"]] += 1
        changed_rows.append(
            {
                "pair_uid": pair_uid,
                "target_bucket": rereview_row.get("target_bucket", ""),
                "review_stratum": active_row.get("review_stratum", ""),
                "balanced_review_rank": active_row.get("balanced_review_rank", ""),
                "previous_active": previous_active,
                "previous_rereview": previous_rereview,
                "updated": {
                    "review_status": active_row["review_status"],
                    "review_label": active_row["review_label"],
                    "reviewer_id": active_row["reviewer_id"],
                    "review_notes": active_row["review_notes"],
                },
                "target_action": rereview_row.get("target_action", ""),
                "source_step11_prob_positive": rereview_row.get("source_step11_prob_positive", ""),
                "source_step11_cluster_rank": rereview_row.get("source_step11_cluster_rank", ""),
            }
        )

    write_csv(active_queue_path, active_rows, active_fieldnames)
    write_csv(rereview_queue_path, rereview_rows, rereview_fieldnames)

    summary = {
        "cleanup_version": cleanup_version,
        "policy_path": str(policy_path.relative_to(ROOT)),
        "active_review_queue": policy["inputs"]["active_review_queue"],
        "targeted_rereview_queue": policy["inputs"]["targeted_rereview_queue"],
        "active_backup_path": str(active_backup.relative_to(ROOT)),
        "targeted_rereview_backup_path": str(rereview_backup.relative_to(ROOT)),
        "reviewer_id": reviewer_id,
        "changed_row_count": len(changed_rows),
        "label_counts": dict(label_counts),
        "changed_rows": changed_rows,
    }

    output_summary_path.parent.mkdir(parents=True, exist_ok=True)
    output_summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Backed up active review queue to: {active_backup}")
    print(f"Backed up targeted rereview queue to: {rereview_backup}")
    print(f"Updated active review queue: {active_queue_path}")
    print(f"Updated targeted rereview queue: {rereview_queue_path}")
    print(f"Wrote cleanup summary: {output_summary_path}")
    print(f"changed_row_count={len(changed_rows)}")
    print(f"label_counts={dict(label_counts)}")


if __name__ == "__main__":
    main()
