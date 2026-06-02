from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
POLICY_PATH = ROOT / "schema" / "step5_paper_targeted_expansion_policy.json"
REVIEWER_ID = "codex_paper_targeted_conservative_review_20260422"

SELLER_CONTACT_RE = re.compile(
    r"(telegram|\btg\b|电报|飞机|纸飞机|wickr|wechat|微信|vx|wx|jabber|xmpp|qq|联系|联系方式|客服|咨询|合作|拍前|找我|加我)",
    re.I,
)
DIRECT_CONTACT_PREFIX_RE = re.compile(r"\b(telegram|wickr|wechat|jabber|qq|phone|crypto_wallet):", re.I)
PRODUCT_DATA_RE = re.compile(
    r"(数据格式|脱库|泄露|客户数据|银行卡数据|股民数据|彩票数据|邮箱|手机号|身份证|预留手机|测试数据|mega|网盘|源码|教程|自动发货|模板)",
    re.I,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Apply a conservative Codex review pass to the paper-targeted Step 5 expansion queue."
    )
    parser.add_argument(
        "--policy-path",
        default=str(POLICY_PATH),
        help="Path to the paper-targeted expansion policy JSON.",
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


def normalize_text(value: object) -> str:
    return str(value or "").strip()


def row_context(row: dict) -> str:
    return " ".join(
        normalize_text(row.get(key))
        for key in (
            "shared_contact_values",
            "shared_title_values",
            "shared_description_values",
            "left_preview",
            "right_preview",
        )
    )


def review_row(row: dict) -> tuple[str, str, str]:
    shared_contact = normalize_text(row.get("shared_contact_values"))
    context = row_context(row)
    touches_proof = normalize_text(row.get("touches_strict_direct_proof_seller")) in {"1", "true", "True"}

    if shared_contact and DIRECT_CONTACT_PREFIX_RE.search(shared_contact) and SELLER_CONTACT_RE.search(context):
        return (
            "positive",
            "seller_facing_shared_direct_contact",
            "codex_paper_review: positive; shared seller-facing direct contact is present with surrounding contact context.",
        )

    if touches_proof:
        return (
            "uncertain",
            "proof_anchor_neighbor_no_independent_direct_identity",
            "codex_paper_review: uncertain; candidate touches a strict direct-contact proof seller, but this edge lacks its own shared seller-facing identifier and cannot be used as independent paper-positive evidence.",
        )

    if PRODUCT_DATA_RE.search(context):
        return (
            "uncertain",
            "product_or_template_overlap_no_identity_anchor",
            "codex_paper_review: uncertain; high model score is driven by product/template/data overlap without a seller-specific identity anchor.",
        )

    return (
        "uncertain",
        "model_high_score_without_identity_anchor",
        "codex_paper_review: uncertain; model score alone is not sufficient for a defensible Step 5 positive label.",
    )


def main() -> None:
    args = parse_args()
    policy_path = Path(args.policy_path)
    if not policy_path.is_absolute():
        policy_path = ROOT / policy_path
    policy = load_json(policy_path)
    queue_path = ROOT / policy["outputs"]["targeted_review_queue"]
    rows, fieldnames = load_csv(queue_path)

    label_counts = Counter()
    rule_counts = Counter()
    bucket_label_counts: dict[str, Counter] = {}
    reviewed_rows = []

    for row in rows:
        label, rule_id, notes = review_row(row)
        row["review_status"] = "reviewed"
        row["review_label"] = label
        row["reviewer_id"] = REVIEWER_ID
        row["review_notes"] = notes
        label_counts[label] += 1
        rule_counts[rule_id] += 1
        bucket = row.get("target_bucket", "")
        bucket_label_counts.setdefault(bucket, Counter())[label] += 1
        reviewed_rows.append(
            {
                "paper_queue_rank": row.get("paper_queue_rank", ""),
                "pair_uid": row.get("pair_uid", ""),
                "target_bucket": bucket,
                "review_label": label,
                "rule_id": rule_id,
                "lr_l2_prob_min": row.get("lr_l2_prob_min", ""),
                "identifier_prob_max": row.get("identifier_prob_max", ""),
                "touches_strict_direct_proof_seller": row.get("touches_strict_direct_proof_seller", ""),
                "shared_contact_values": row.get("shared_contact_values", ""),
            }
        )

    write_csv(queue_path, rows, fieldnames)

    summary_path = queue_path.with_name("step5_paper_targeted_expansion_codex_review_summary.20260422.json")
    summary = {
        "reviewer_id": REVIEWER_ID,
        "review_timestamp_local": datetime.now().replace(microsecond=0).isoformat(),
        "policy_path": str(policy_path.relative_to(ROOT)),
        "queue_path": str(queue_path.relative_to(ROOT)),
        "reviewed_row_count": len(rows),
        "label_counts": dict(label_counts),
        "rule_counts": dict(rule_counts),
        "bucket_label_counts": {
            bucket: dict(counts)
            for bucket, counts in bucket_label_counts.items()
        },
        "application_recommendation": (
            "Do not apply this queue to Step 5 supervision yet: no high-confidence new positive or negative labels were found. "
            "The reviewed rows are useful as audit evidence of direct-identity scarcity."
        ),
        "reviewed_rows": reviewed_rows,
    }
    write_json(summary_path, summary)

    print(f"Reviewed paper-targeted queue: {queue_path}")
    print(f"Wrote review summary: {summary_path}")
    print(f"label_counts={dict(label_counts)}")


if __name__ == "__main__":
    main()
