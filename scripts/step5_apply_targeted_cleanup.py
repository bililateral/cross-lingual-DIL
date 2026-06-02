from __future__ import annotations

import csv
import json
import shutil
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
POLICY_PATH = ROOT / "schema" / "step5_targeted_cleanup_policy.json"


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
            writer.writerow(row)


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def build_backup(path: Path, tag: str) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    backup_path = path.with_name(f"{path.stem}.codexbak.{tag}.{timestamp}{path.suffix}")
    shutil.copy2(path, backup_path)
    return backup_path


def main() -> None:
    policy = load_json(POLICY_PATH)
    queue_path = ROOT / policy["inputs"]["active_review_queue"]
    rereview_queue_path = ROOT / policy["inputs"]["targeted_rereview_queue"]
    output_summary_path = ROOT / policy["outputs"]["summary"]

    queue_rows, fieldnames = load_csv(queue_path)
    rereview_rows, _ = load_csv(rereview_queue_path)
    queue_index = {row["pair_uid"]: row for row in queue_rows}
    rereview_index = {row["pair_uid"]: row for row in rereview_rows}

    backup_path = build_backup(queue_path, "step5_v2_cleanup")
    changed_rows = []

    for decision in policy["decisions"]:
        pair_uid = decision["pair_uid"]
        queue_row = queue_index.get(pair_uid)
        if queue_row is None:
            raise SystemExit(f"Targeted cleanup pair_uid not found in active review queue: {pair_uid}")
        rereview_row = rereview_index.get(pair_uid)
        if rereview_row is None:
            raise SystemExit(f"Targeted cleanup pair_uid not found in targeted rereview queue: {pair_uid}")

        previous = {
            "review_status": queue_row.get("review_status", ""),
            "review_label": queue_row.get("review_label", ""),
            "reviewer_id": queue_row.get("reviewer_id", ""),
            "review_notes": queue_row.get("review_notes", ""),
        }

        queue_row["review_status"] = "reviewed"
        queue_row["review_label"] = decision["new_review_label"]
        queue_row["reviewer_id"] = policy["reviewer_id"]
        queue_row["review_notes"] = decision["review_notes"]

        changed_rows.append(
            {
                "pair_uid": pair_uid,
                "review_stratum": queue_row.get("review_stratum", ""),
                "balanced_review_rank": queue_row.get("balanced_review_rank", ""),
                "previous": previous,
                "updated": {
                    "review_status": queue_row["review_status"],
                    "review_label": queue_row["review_label"],
                    "reviewer_id": queue_row["reviewer_id"],
                    "review_notes": queue_row["review_notes"],
                },
                "target_action": rereview_row.get("target_action", ""),
                "source_step11_prob_positive": rereview_row.get("source_step11_prob_positive", ""),
                "source_step11_cluster_rank": rereview_row.get("source_step11_cluster_rank", ""),
            }
        )

    write_csv(queue_path, queue_rows, fieldnames)

    summary = {
        "cleanup_version": policy["cleanup_version"],
        "active_review_queue": policy["inputs"]["active_review_queue"],
        "targeted_rereview_queue": policy["inputs"]["targeted_rereview_queue"],
        "backup_path": str(backup_path.relative_to(ROOT)),
        "reviewer_id": policy["reviewer_id"],
        "changed_row_count": len(changed_rows),
        "changed_rows": changed_rows,
    }
    write_json(output_summary_path, summary)

    print(f"Backed up active review queue to: {backup_path}")
    print(f"Updated active review queue: {queue_path}")
    print(f"Wrote cleanup summary: {output_summary_path}")
    print(f"changed_row_count={len(changed_rows)}")


if __name__ == "__main__":
    main()
