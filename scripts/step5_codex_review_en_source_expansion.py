from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
POLICY_PATH = ROOT / "schema" / "step5_en_source_expansion_policy.json"
REVIEWER_ID = "codex_en_source_expansion_review_20260421"

TRUST_SUFFIX_RE = re.compile(r"\s*\(\d+%\)\s*$", re.I)
NON_ALIAS_CHARS_RE = re.compile(r"[^0-9a-z]+", re.I)
CONTACT_SPLIT_RE = re.compile(r"\s*\|\|\s*")
SELLER_CONTACT_CUE_RE = re.compile(
    r"(contact\s+(?:me|us)|reach\s+(?:me|us)|message\s+(?:me|us)|dm\b|wickr|telegram|whatsapp|call/text|"
    r"text\s+(?:me|us)|support|before\s+placing|after\s+placing|fast\s+response|snapchat|jabber|xmpp)",
    re.I,
)
PRODUCT_OR_VICTIM_DATA_RE = re.compile(
    r"(email\s*:\s*password|username\s*:|password\s*:|format\s*:|cc\s+number|expiry|cvv|full\s+name|"
    r"\bdob\b|\bssn\b|\bmmn\b|available\s+credit|combo\s+list|account\s+with\s+\$|login\s+password|"
    r"useragent|cookie|fullz|dump|database|credential|sample\s+data|leaked|victim|passport\s+template|"
    r"statement\s+template|utility\s+statement|id\s+card\s+psd)",
    re.I,
)
PUBLIC_TEMPLATE_RE = re.compile(
    r"(no\s+item\s+name|no\s+description\s+specified|listing\s+title\s+corrupted|custom\s+listing|"
    r"is\s+used\s+to\s+treat|why\s+is\s+this\s+medication\s+prescribed|take\s+.+\s+exactly\s+as\s+directed|"
    r"cas\s+no\.?|molecular\s+formula|synonyms|threshold\s+\d|common\s+\d|strong\s+\d|heavy\s+\d|"
    r"this\s+book|pages\s*:\s*\d+|serial\s+key|private\s+message|medical\s+facts|dispose\s+of\s+your\s+medication)",
    re.I,
)
SELLER_SPECIFIC_OPERATION_RE = re.compile(
    r"(refund|blacklist|finalize|tracking|stealth|escrow|ordered\s+before|delivery\s+time|ship(?:ped|ping)?\s+"
    r"(?:from|to)|we\s+(?:ship|offer|sell|accept|provide|recommend)|our\s+(?:shop|store|products?|quality|customers?)|"
    r"welcome\s+valued\s+clients|customer\s+safety|satisfaction\s+is)",
    re.I,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Conservatively review the English source-domain expansion queue."
    )
    parser.add_argument(
        "--policy-path",
        default=str(POLICY_PATH),
        help="Path to the English source expansion policy JSON.",
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


def normalize_alias(value: object) -> str:
    text = normalize_text(value).casefold()
    text = TRUST_SUFFIX_RE.sub("", text)
    text = NON_ALIAS_CHARS_RE.sub("", text)
    return text


def is_soft_same_alias(row: dict) -> bool:
    left = normalize_alias(row.get("source_seller_raw_left"))
    right = normalize_alias(row.get("source_seller_raw_right"))
    return bool(left and right and left == right)


def bounded_edit_distance(left: str, right: str, limit: int = 2) -> int:
    if abs(len(left) - len(right)) > limit:
        return limit + 1
    previous = list(range(len(right) + 1))
    for i, left_ch in enumerate(left, start=1):
        current = [i]
        best = current[0]
        for j, right_ch in enumerate(right, start=1):
            cost = 0 if left_ch == right_ch else 1
            value = min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + cost)
            current.append(value)
            best = min(best, value)
        if best > limit:
            return limit + 1
        previous = current
    return previous[-1]


def longest_common_substring_len(left: str, right: str) -> int:
    if not left or not right:
        return 0
    previous = [0] * (len(right) + 1)
    best = 0
    for left_ch in left:
        current = [0]
        for j, right_ch in enumerate(right, start=1):
            if left_ch == right_ch:
                value = previous[j - 1] + 1
                best = max(best, value)
            else:
                value = 0
            current.append(value)
        previous = current
    return best


def alias_related(row: dict) -> bool:
    left = normalize_alias(row.get("source_seller_raw_left"))
    right = normalize_alias(row.get("source_seller_raw_right"))
    if not left or not right:
        return False
    if left == right:
        return True
    if min(len(left), len(right)) >= 5 and (left in right or right in left):
        return True
    if min(len(left), len(right)) >= 4 and bounded_edit_distance(left, right, 1) <= 1:
        return True
    if min(len(left), len(right)) >= 6 and bounded_edit_distance(left, right, 2) <= 2:
        return True
    shared_len = longest_common_substring_len(left, right)
    if shared_len >= 6 and shared_len >= int(round(0.6 * min(len(left), len(right)))):
        return True
    return False


def to_int(value: object) -> int:
    try:
        return int(round(float(value)))
    except (TypeError, ValueError):
        return 0


def to_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def row_context(row: dict) -> str:
    return " ".join(
        normalize_text(row.get(field))
        for field in (
            "source_seller_raw_left",
            "source_seller_raw_right",
            "shared_contact_values",
            "shared_title_values",
            "shared_description_values",
            "left_preview",
            "right_preview",
        )
    )


def evidence_context(row: dict) -> str:
    return " ".join(
        normalize_text(row.get(field))
        for field in (
            "source_seller_raw_left",
            "source_seller_raw_right",
            "shared_title_values",
            "shared_description_values",
            "left_preview",
            "right_preview",
        )
    )


def contact_tokens(row: dict) -> list[tuple[str, str]]:
    tokens = []
    for token in CONTACT_SPLIT_RE.split(normalize_text(row.get("shared_contact_values"))):
        if ":" not in token:
            continue
        token_type, token_value = token.split(":", 1)
        token_type = token_type.strip().lower()
        token_value = token_value.strip().lower().lstrip("@")
        if token_type and token_value:
            tokens.append((token_type, token_value))
    return tokens


def token_appears_in_context(token_type: str, token_value: str, context: str) -> bool:
    lowered = context.lower()
    if token_type == "phone":
        digits = re.sub(r"\D", "", token_value)
        context_digits = re.sub(r"\D", "", lowered)
        return bool(digits and digits in context_digits)
    return token_value in lowered


def has_seller_facing_contact(row: dict) -> bool:
    context = evidence_context(row)
    if not SELLER_CONTACT_CUE_RE.search(context):
        return False
    return any(token_appears_in_context(token_type, token_value, context) for token_type, token_value in contact_tokens(row))


def has_product_or_victim_data_context(row: dict) -> bool:
    return bool(PRODUCT_OR_VICTIM_DATA_RE.search(row_context(row)))


def has_public_template_context(row: dict) -> bool:
    return bool(PUBLIC_TEMPLATE_RE.search(row_context(row)))


def has_seller_specific_operation(row: dict) -> bool:
    return bool(SELLER_SPECIFIC_OPERATION_RE.search(row_context(row)))


def clone_strength(row: dict) -> int:
    return to_int(row.get("shared_title_count")) + to_int(row.get("shared_description_count")) * 2


def has_support(row: dict) -> bool:
    return (
        clone_strength(row) > 0
        or to_float(row.get("lexical_similarity")) >= 0.18
        or to_float(row.get("structural_support_score")) >= 0.55
    )


def review_row(row: dict) -> tuple[str, str, str]:
    bucket = normalize_text(row.get("target_bucket"))
    context = row_context(row)
    shared_contact_count = to_int(row.get("shared_contact_count"))
    shared_pgp_count = to_int(row.get("shared_pgp_fingerprint_count"))
    shared_title_count = to_int(row.get("shared_title_count"))
    shared_description_count = to_int(row.get("shared_description_count"))

    if is_soft_same_alias(row):
        return (
            "uncertain",
            "soft_same_alias_excluded_from_primary_expansion",
            "codex_review: uncertain; same/soft-same alias continuity is excluded from primary English supervision expansion.",
        )

    if shared_pgp_count > 0:
        return (
            "positive",
            "shared_pgp_fingerprint_positive",
            "codex_review: positive; shared PGP fingerprint is strong seller identity evidence.",
        )

    if bucket in {"seller_facing_identifier_plus_text", "seller_facing_identifier_primary"}:
        if shared_contact_count <= 0:
            return (
                "uncertain",
                "identifier_bucket_without_contact",
                "codex_review: uncertain; identifier bucket lacks a retained shared contact or PGP count.",
            )
        tokens = contact_tokens(row)
        has_email = any(token_type == "email" for token_type, _token_value in tokens)
        has_im_or_phone = any(token_type in {"telegram", "wickr", "wechat", "qq", "phone", "jabber"} for token_type, _ in tokens)
        seller_contact = has_seller_facing_contact(row)
        product_context = has_product_or_victim_data_context(row)

        if product_context and not seller_contact:
            return (
                "uncertain",
                "contact_inside_product_or_victim_data",
                "codex_review: uncertain; shared contact-like token appears in product credentials, leaked data, or sample account context.",
            )
        if has_im_or_phone and seller_contact and has_support(row):
            return (
                "positive",
                "seller_facing_im_or_phone_positive",
                "codex_review: positive; shared seller-facing IM/phone contact has supporting profile evidence.",
            )
        if has_email and seller_contact and has_support(row) and not product_context:
            return (
                "positive",
                "seller_facing_email_positive",
                "codex_review: positive; shared email is presented as seller contact and has supporting profile evidence.",
            )
        if has_email and product_context:
            return (
                "uncertain",
                "email_product_data_ambiguous",
                "codex_review: uncertain; shared email may be sample/customer/victim data rather than seller identity.",
            )
        return (
            "uncertain",
            "contact_not_seller_facing_enough",
            "codex_review: uncertain; shared contact exists but seller-facing context or support is insufficient for a positive label.",
        )

    if bucket == "strong_text_clone_positive_probe":
        strong_clone = shared_title_count >= 2 and shared_description_count >= 1
        very_strong_clone = shared_title_count >= 3 and shared_description_count >= 2
        public_template = has_public_template_context(row) or has_product_or_victim_data_context(row)
        seller_specific = has_seller_specific_operation(row)

        if strong_clone and alias_related(row) and not public_template:
            return (
                "positive",
                "alias_related_text_clone_positive",
                "codex_review: positive; near-alias relation plus exact title/description clone evidence supports same controller.",
            )
        if very_strong_clone and seller_specific and not public_template:
            return (
                "positive",
                "seller_specific_operational_text_clone_positive",
                "codex_review: positive; repeated exact catalog text includes seller-specific operational language, not merely public boilerplate.",
            )
        if public_template and not seller_specific:
            return (
                "uncertain",
                "public_or_product_template_clone",
                "codex_review: uncertain; clone evidence is compatible with public product/template reuse and lacks identity closure.",
            )
        return (
            "uncertain",
            "text_clone_without_identity_closure",
            "codex_review: uncertain; strong text overlap exists, but it is not enough for primary positive supervision without clearer identity closure.",
        )

    if bucket == "english_hard_negative_template_probe":
        if shared_contact_count > 0 or shared_pgp_count > 0:
            return (
                "uncertain",
                "hard_negative_has_identifier_anchor",
                "codex_review: uncertain; hard-negative probe unexpectedly contains direct identifier evidence.",
            )
        if re.search(r"(no\s+item\s+name|no\s+description\s+specified|listing\s+title\s+corrupted)", context, re.I):
            return (
                "negative",
                "generic_placeholder_hard_negative",
                "codex_review: negative; high similarity is caused by generic placeholder text rather than seller-specific evidence.",
            )
        if has_public_template_context(row) or has_product_or_victim_data_context(row):
            return (
                "negative",
                "public_template_hard_negative",
                "codex_review: negative; high similarity is explained by public/product boilerplate or leaked-data field templates without identity closure.",
            )
        if not alias_related(row):
            return (
                "negative",
                "same_topic_no_identity_anchor_hard_negative",
                "codex_review: negative; same-topic high-semantic overlap lacks contact, PGP, clone, alias, or other seller-specific identity closure.",
            )
        return (
            "uncertain",
            "alias_related_hard_negative_uncertain",
            "codex_review: uncertain; alias relation prevents using this row as a clean negative.",
        )

    return (
        "uncertain",
        "unknown_en_source_expansion_bucket",
        "codex_review: uncertain; unknown English source expansion bucket.",
    )


def main() -> None:
    args = parse_args()
    policy_path = Path(args.policy_path)
    if not policy_path.is_absolute():
        policy_path = ROOT / policy_path
    policy = load_json(policy_path)
    queue_path = ROOT / policy["outputs"]["targeted_review_queue"]
    summary_path = ROOT / policy["outputs"]["codex_review_summary"]

    reviewer_id = normalize_text(policy.get("reviewer_id")) or REVIEWER_ID
    rows, fieldnames = load_csv(queue_path)
    label_counts = Counter()
    rule_counts = Counter()
    bucket_label_counts: dict[str, Counter] = {}
    reviewed_rows = []

    for row in rows:
        label, rule_id, notes = review_row(row)
        row["review_status"] = "reviewed"
        row["review_label"] = label
        row["reviewer_id"] = reviewer_id
        row["review_notes"] = notes
        label_counts[label] += 1
        rule_counts[rule_id] += 1
        bucket_label_counts.setdefault(row.get("target_bucket", ""), Counter())[label] += 1
        reviewed_rows.append(
            {
                "en_source_expansion_rank": row.get("en_source_expansion_rank", ""),
                "pair_uid": row.get("pair_uid", ""),
                "target_bucket": row.get("target_bucket", ""),
                "review_label": label,
                "rule_id": rule_id,
                "source_seller_raw_left": row.get("source_seller_raw_left", ""),
                "source_seller_raw_right": row.get("source_seller_raw_right", ""),
            }
        )

    write_csv(queue_path, rows, fieldnames)
    summary = {
        "reviewer_id": reviewer_id,
        "review_timestamp_local": datetime.now().replace(microsecond=0).isoformat(),
        "policy_path": str(policy_path.relative_to(ROOT)),
        "queue_path": str(queue_path.relative_to(ROOT)),
        "reviewed_row_count": len(rows),
        "label_counts": dict(label_counts),
        "rule_counts": dict(rule_counts),
        "bucket_label_counts": {bucket: dict(counter) for bucket, counter in bucket_label_counts.items()},
        "rubric": policy["review_rubric"],
        "reviewed_rows": reviewed_rows,
    }
    write_json(summary_path, summary)
    print(f"Reviewed English source expansion queue: {queue_path}")
    print(f"label_counts={dict(label_counts)} rule_counts={dict(rule_counts)}")


if __name__ == "__main__":
    main()
