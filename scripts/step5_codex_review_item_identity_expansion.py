from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
POLICY_PATH = ROOT / "schema" / "step5_en_item_identity_expansion_policy.json"
REVIEWER_ID = "codex_item_identity_expansion_review_20260422"

GENERIC_TOKEN_RE = re.compile(
    r"^(?:app|application|before|contact|download|joinchat|mail|messenger|messgenger|private|store|support|"
    r"telegram|wickr|w1ckr|wechat|whatsapp)$",
    re.I,
)
PUBLIC_MEDIA_CONTEXT_RE = re.compile(r"(youtube\.com/watch|youtu\.be/|vimeo\.com|clear\s*net)", re.I)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Conservatively review a Step 5 item-identity expansion queue.")
    parser.add_argument("--policy-path", default=str(POLICY_PATH), help="Path to item-identity expansion policy JSON.")
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


def normalize_text(value: object) -> str:
    return str(value or "").strip()


def to_int(value: object) -> int:
    try:
        return int(round(float(value)))
    except (TypeError, ValueError):
        return 0


def anchor_value(anchor_token: str) -> str:
    if ":" not in anchor_token:
        return anchor_token.strip().lower()
    return anchor_token.split(":", 1)[1].strip().lower()


def combined_context(row: dict) -> str:
    return " ".join(
        normalize_text(row.get(field))
        for field in ("left_item_contexts", "right_item_contexts", "left_preview", "right_preview")
    )


def review_row(row: dict) -> tuple[str, str, str]:
    anchor_type = normalize_text(row.get("anchor_type")).lower()
    token = anchor_value(normalize_text(row.get("anchor_token")))
    context = combined_context(row)

    if to_int(row.get("product_data_risk_any")) > 0:
        return (
            "uncertain",
            "product_or_victim_data_risk",
            "Shared token appears near product/victim/sample-data cues; not safe as identity supervision.",
        )

    if to_int(row.get("both_seller_facing_context")) <= 0:
        return (
            "uncertain",
            "weak_or_one_sided_seller_contact_context",
            "Shared token is not seller-facing on both sides.",
        )

    if GENERIC_TOKEN_RE.match(token):
        return (
            "negative",
            "generic_parser_token",
            "Parser captured a generic platform/action word rather than a seller identifier.",
        )

    if anchor_type in {"external_url"}:
        return (
            "uncertain",
            "support_only_external_url",
            "External URLs are support-only and cannot create a direct identity label.",
        )

    if anchor_type in {"wechat", "telegram"} and PUBLIC_MEDIA_CONTEXT_RE.search(context):
        return (
            "negative",
            "public_media_id_not_seller_identity",
            "The token is from a public media URL, not a seller-facing contact handle.",
        )

    if anchor_type in {
        "pgp_public_key",
        "pgp_fingerprint",
        "crypto_wallet",
        "telegram",
        "wickr",
        "jabber",
        "wechat",
        "qq",
        "bat",
        "phone",
        "email",
    }:
        return (
            "positive",
            "shared_seller_facing_direct_identifier",
            "Both sides expose the same low-frequency seller-facing direct identifier with no product-data risk flag.",
        )

    return (
        "uncertain",
        "unsupported_anchor_type",
        f"Anchor type {anchor_type!r} is not part of the positive-review rubric.",
    )


def main() -> None:
    args = parse_args()
    policy_path = Path(args.policy_path)
    if not policy_path.is_absolute():
        policy_path = ROOT / policy_path
    policy = load_json(policy_path)
    outputs = policy["outputs"]
    queue_path = ROOT / outputs["targeted_review_queue"]
    summary_path = ROOT / outputs["codex_review_summary"]

    rows, fieldnames = load_csv(queue_path)
    reviewed_rows = []
    label_counts = Counter()
    rule_counts = Counter()
    anchor_label_counts: dict[str, Counter] = {}
    now = datetime.now().isoformat(timespec="seconds")

    for row in rows:
        label, rule_id, notes = review_row(row)
        row["review_status"] = "reviewed"
        row["review_label"] = label
        row["reviewer_id"] = REVIEWER_ID
        row["review_notes"] = f"{rule_id}: {notes}"
        label_counts[label] += 1
        rule_counts[rule_id] += 1
        anchor_label_counts.setdefault(row.get("anchor_type", ""), Counter())[label] += 1
        reviewed_rows.append(
            {
                "item_identity_queue_rank": row.get("item_identity_queue_rank", ""),
                "pair_uid": row.get("pair_uid", ""),
                "anchor_token": row.get("anchor_token", ""),
                "anchor_type": row.get("anchor_type", ""),
                "target_action": row.get("target_action", ""),
                "review_label": label,
                "rule_id": rule_id,
                "source_seller_raw_left": row.get("source_seller_raw_left", ""),
                "source_seller_raw_right": row.get("source_seller_raw_right", ""),
            }
        )

    write_csv(queue_path, rows, fieldnames)
    write_json(
        summary_path,
        {
            "reviewer_id": REVIEWER_ID,
            "review_timestamp_local": now,
            "policy_path": str(policy_path.relative_to(ROOT)),
            "queue_path": outputs["targeted_review_queue"],
            "reviewed_row_count": len(rows),
            "label_counts": dict(label_counts),
            "rule_counts": dict(rule_counts),
            "anchor_label_counts": {key: dict(value) for key, value in sorted(anchor_label_counts.items())},
            "reviewed_rows": reviewed_rows,
            "rubric": policy.get("review_guidelines", {}),
            "hard_rules": policy.get("hard_rules", []),
        },
    )
    print(f"Reviewed {len(rows)} rows: {dict(label_counts)}")
    print(f"Wrote review summary: {summary_path}")


if __name__ == "__main__":
    main()
