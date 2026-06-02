from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
POLICY_PATH = ROOT / "schema" / "step5_positive_anchor_expansion_policy.json"
REVIEWER_ID = "codex_conservative_positive_anchor_review_20260421"
SELLER_CONTACT_CUE_RE = re.compile(
    r"(telegram|\btg\b|电报|飞机|纸飞机|wickr|微信|wechat|vx|wx|jabber|xmpp|qq|联系|联系方式|客服|咨询|合作|拍前|找我|加我|留下)",
    re.I,
)
SELLER_URL_CUE_RE = re.compile(r"(官网|网址|网站|客服|联系|入口|最新地址|备用|注册|平台|后台)", re.I)
SELLER_OWNED_URL_CUE_RE = re.compile(r"(本人网站|我的网站|我方网站|自营网站|店铺官网|客服网址|联系网址|官方网址|卖家网站)", re.I)
URL_PRODUCT_CONTEXT_RE = re.compile(
    r"(数据|脱库|可验证|反向验证|注册页|账号|帐号|会员|娱乐城|博彩|彩票|网贷|贷款|股票|配资|客户|数据库|手机号|电话号码|电子邮箱|email|qq|reg_ip|格式)",
    re.I,
)
WALLET_CUE_RE = re.compile(r"(钱包|地址|收款|付款|转账|btc|bitcoin|usdt|eth|tron|trc20|erc20)", re.I)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Apply a conservative Codex review pass to the Step 5 positive-anchor expansion queue."
    )
    parser.add_argument(
        "--policy-path",
        default=str(POLICY_PATH),
        help="Path to the Step 5 positive-anchor expansion policy JSON.",
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


def to_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def victim_data_context(row: dict) -> bool:
    text = " ".join(
        normalize_text(row.get(key))
        for key in (
            "left_anchor_context",
            "right_anchor_context",
            "left_preview",
            "right_preview",
            "shared_title_values",
            "shared_description_values",
        )
    )
    return bool(
        re.search(
            r"(通讯录|邮箱库|数据字段|手机号|身份证|医生资源|店长联系方式|泄露|脱裤|女性数据|快递数据|email spoof|电子邮件欺骗|不要让@)",
            text,
            re.I,
        )
    )


def row_context(row: dict) -> str:
    return " ".join(
        normalize_text(row.get(key))
        for key in (
            "left_anchor_context",
            "right_anchor_context",
            "left_preview",
            "right_preview",
            "shared_title_values",
            "shared_description_values",
        )
    )


def anchor_type_and_value(row: dict) -> tuple[str, str]:
    anchor_token = normalize_text(row.get("anchor_token"))
    if ":" not in anchor_token:
        return anchor_token.lower(), ""
    anchor_type, anchor_value = anchor_token.split(":", 1)
    return anchor_type.lower(), anchor_value


def review_row(row: dict) -> tuple[str, str, str]:
    bucket = normalize_text(row.get("target_bucket"))
    anchor_token = normalize_text(row.get("anchor_token"))
    anchor_type, _anchor_value = anchor_type_and_value(row)
    evidence = normalize_text(row.get("anchor_evidence_level"))
    lexical = to_float(row.get("lexical_similarity"))
    structural = to_float(row.get("structural_support_score"))
    has_text_clone = normalize_text(row.get("shared_title_values")) or normalize_text(row.get("shared_description_values"))
    context = row_context(row)
    has_support = bool(has_text_clone) or lexical >= 0.05 or structural >= 0.2
    strong_support = bool(has_text_clone) or lexical >= 0.14 or structural >= 0.5

    if bucket == "positive_component_transitive_closure":
        return (
            "positive",
            "clean_positive_component_closure",
            "codex_review: positive; contradiction-free reviewed positive component closure. Treat as closure-derived support, not an independent discovery edge.",
        )

    if bucket in {"supplemental_shared_seller_contact", "rare_identity_with_text_overlap"}:
        if victim_data_context(row) and not SELLER_CONTACT_CUE_RE.search(context):
            return (
                "uncertain",
                "identity_anchor_victim_data_context",
                "codex_review: uncertain; the shared token may come from product/victim-data text rather than seller-facing contact evidence.",
            )
        if anchor_type in {"telegram", "qq", "wickr", "wechat", "jabber", "phone"}:
            if ("strong_contact" in evidence or "cue_at_handle" in evidence) and SELLER_CONTACT_CUE_RE.search(context):
                if has_support:
                    return (
                        "positive",
                        "seller_facing_shared_contact",
                        "codex_review: positive; low-frequency shared seller-facing contact identifier appears on both profiles with enough surrounding profile support.",
                    )
        if anchor_type == "crypto_wallet":
            if WALLET_CUE_RE.search(context) and strong_support:
                return (
                    "positive",
                    "seller_facing_shared_wallet",
                    "codex_review: positive; rare shared wallet/payment identifier appears with supporting seller-profile overlap.",
                )
        if anchor_type == "email":
            if SELLER_CONTACT_CUE_RE.search(context) and not victim_data_context(row) and strong_support:
                return (
                    "positive",
                    "seller_facing_shared_email",
                    "codex_review: positive; shared email is presented as seller-facing contact with enough surrounding profile support.",
                )
            return (
                "uncertain",
                "email_identity_ambiguous",
                "codex_review: uncertain; shared email may be copied customer/victim/product data rather than a seller identity anchor.",
            )
        return (
            "uncertain",
            "identity_anchor_insufficient_context",
            "codex_review: uncertain; shared contact candidate is plausible but not strong enough for a positive label under the conservative rubric.",
        )

    if bucket == "rare_external_url_with_text_overlap":
        if victim_data_context(row) or URL_PRODUCT_CONTEXT_RE.search(context):
            return (
                "uncertain",
                "external_url_victim_or_product_context",
                "codex_review: uncertain; shared URL appears compatible with product/data content rather than seller-operated identity infrastructure.",
            )
        if SELLER_OWNED_URL_CUE_RE.search(context) and strong_support and not re.search(r"(youtube|mega\\.nz|nordvpn|expressvpn|viewtopic)", anchor_token, re.I):
            return (
                "positive",
                "seller_operated_shared_url",
                "codex_review: positive; rare shared URL is presented as seller-operated infrastructure with supporting profile overlap.",
            )
        return (
            "uncertain",
            "external_url_not_identity_definitive",
            "codex_review: uncertain; rare shared URL plus text support is not enough to prove same controller without clearer seller-operated identity context.",
        )

    if bucket == "cross_market_unique_alias_closure":
        if strong_support and not victim_data_context(row):
            return (
                "positive",
                "cross_market_unique_alias_with_content_closure",
                "codex_review: positive; uncommon cross-market alias has additional content/contact closure beyond alias reuse.",
            )
        return (
            "uncertain",
            "cross_market_alias_insufficient_support",
            "codex_review: uncertain; same/rare alias without strong independent closure is not enough for positive supervision.",
        )

    if bucket == "direct_contact_existing_pending":
        lowered = anchor_token.lower()
        if "telegram:https" in lowered:
            return (
                "negative",
                "known_parser_noise_contact",
                "codex_review: negative; shared contact is known parser noise rather than a seller identifier.",
            )
        if lowered.startswith("email:") or " email:" in lowered:
            return (
                "uncertain",
                "email_inside_data_product",
                "codex_review: uncertain; shared email evidence appears compatible with leaked/sample data rows and lacks enough seller-specific identity closure.",
            )
        if (
            (lowered.startswith("telegram:") or lowered.startswith("qq:") or lowered.startswith("wechat:") or lowered.startswith("wickr:"))
            and (lexical >= 0.1 or structural >= 0.35 or has_text_clone)
        ):
            return (
                "positive",
                "existing_direct_contact_with_support",
                "codex_review: positive; existing direct contact anchor has supporting text or structural evidence.",
            )
        return (
            "uncertain",
            "existing_direct_contact_without_support",
            "codex_review: uncertain; direct contact exists but the row lacks enough surrounding support for a defensible positive label.",
        )

    return (
        "uncertain",
        "unknown_positive_anchor_bucket",
        "codex_review: uncertain; unknown positive-anchor bucket.",
    )


def main() -> None:
    args = parse_args()
    policy_path = Path(args.policy_path)
    if not policy_path.is_absolute():
        policy_path = ROOT / policy_path
    policy = load_json(policy_path)
    queue_path = ROOT / policy["outputs"]["targeted_review_queue"]
    summary_path = ROOT / policy["outputs"]["codex_review_summary"]

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
        bucket_label_counts.setdefault(row.get("target_bucket", ""), Counter())[label] += 1
        reviewed_rows.append(
            {
                "positive_anchor_queue_rank": row.get("positive_anchor_queue_rank", ""),
                "pair_uid": row["pair_uid"],
                "target_bucket": row.get("target_bucket", ""),
                "review_label": label,
                "rule_id": rule_id,
                "anchor_token": row.get("anchor_token", ""),
                "anchor_evidence_level": row.get("anchor_evidence_level", ""),
                "source_existing_active_bool": row.get("source_existing_active_bool", ""),
                "source_existing_step4_bool": row.get("source_existing_step4_bool", ""),
            }
        )

    write_csv(queue_path, rows, fieldnames)
    summary = {
        "reviewer_id": REVIEWER_ID,
        "review_timestamp_local": datetime.now().replace(microsecond=0).isoformat(),
        "policy_path": str(policy_path.relative_to(ROOT)),
        "queue_path": str(queue_path.relative_to(ROOT)),
        "reviewed_row_count": len(rows),
        "label_counts": dict(label_counts),
        "rule_counts": dict(rule_counts),
        "bucket_label_counts": {bucket: dict(counter) for bucket, counter in bucket_label_counts.items()},
        "rubric": {
            "positive": "Seller-specific shared contact or contradiction-free reviewed positive-component closure.",
            "uncertain": "Plausible but insufficient seller-specific closure, especially victim-data contact ambiguity.",
            "negative": "Clear parser noise or controller-distinct evidence only.",
        },
        "reviewed_rows": reviewed_rows,
    }
    write_json(summary_path, summary)
    print(f"Reviewed positive-anchor queue: {queue_path}")
    print(f"Wrote review summary: {summary_path}")
    print(f"label_counts={dict(label_counts)}")


if __name__ == "__main__":
    main()
