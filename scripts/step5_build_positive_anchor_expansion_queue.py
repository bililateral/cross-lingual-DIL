from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path
from urllib.parse import urlparse

import step4_build_silver_candidates as step4
import step5_build_review_strata as step5_strata


ROOT = Path(__file__).resolve().parent.parent
POLICY_PATH = ROOT / "schema" / "step5_positive_anchor_expansion_policy.json"

TARGET_EXTRA_FIELDS = [
    "positive_anchor_queue_rank",
    "target_bucket",
    "target_reason",
    "target_action",
    "suggested_label",
    "suggested_confidence",
    "anchor_token",
    "anchor_type",
    "anchor_frequency",
    "anchor_evidence_level",
    "left_anchor_context",
    "right_anchor_context",
    "source_existing_active_bool",
    "source_existing_step4_bool",
    "source_positive_component_id",
    "source_positive_component_size",
    "closure_source_pair_uids",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a Step 5 positive-anchor expansion queue from direct seller-contact and clean closure evidence."
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
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def normalize_text(value: object) -> str:
    return str(value or "").strip()


def normalize_label(value: object) -> str:
    return normalize_text(value).lower()


def to_float(value: object, default: float = 0.0) -> float:
    if value in {"", None}:
        return default
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    if math.isnan(result):
        return default
    return result


def to_int(value: object) -> int:
    return int(round(to_float(value, 0.0)))


def is_pending(row: dict) -> bool:
    return normalize_text(row.get("review_status")).lower() in {"", "pending"} and normalize_label(row.get("review_label")) == ""


def ordered_profiles(left, right):
    return (left, right) if left.seller_uid <= right.seller_uid else (right, left)


def lexical_similarity(left, right) -> float:
    if not left.retrieval_norm or not right.retrieval_norm:
        return 0.0
    left_weights = left.retrieval_weights
    right_weights = right.retrieval_weights
    if len(left_weights) > len(right_weights):
        left_weights, right_weights = right_weights, left_weights
    dot = sum(weight * right_weights.get(term, 0.0) for term, weight in left_weights.items())
    return round(dot / (left.retrieval_norm * right.retrieval_norm), 6) if dot > 0 else 0.0


def shared_values(left_map: dict, right_map: dict, limit: int = 5, width: int = 180) -> list[str]:
    values = []
    for norm in sorted(set(left_map) & set(right_map)):
        raw = left_map.get(norm) or right_map.get(norm) or norm
        values.append(step4.capped_text(raw, width))
        if len(values) >= limit:
            break
    return values


def make_candidate_row(left, right, *, shared_contacts: list[str], extra_rule_hits: list[str]) -> dict:
    left, right = ordered_profiles(left, right)
    data_bucket = left.data_bucket
    candidate_language = "en" if data_bucket == "en_content_train_pool" else "zh"
    support_score, item_ratio, price_ratio, style_l1 = step4.structural_support(left, right)
    lexical = lexical_similarity(left, right)
    shared_titles = shared_values(left.title_norm_to_raw, right.title_norm_to_raw)
    shared_descs = shared_values(left.desc_norm_to_raw, right.desc_norm_to_raw, limit=3)
    shared_categories = [
        left.category_norm_to_raw.get(norm) or right.category_norm_to_raw.get(norm) or norm
        for norm in sorted(left.category_norms & right.category_norms)[:5]
    ]
    rule_hits = set(extra_rule_hits)
    if shared_contacts:
        rule_hits.add("shared_contact_exact")
    if shared_titles:
        rule_hits.add("shared_title_clone")
    if shared_descs:
        rule_hits.add("shared_description_clone")
    if lexical >= 0.14:
        rule_hits.add("profile_lexical_neighbor")
    if support_score >= 0.5:
        rule_hits.add("structural_support")

    rank_score = 0.0
    rank_score += 6.0 * (1 if shared_contacts else 0)
    rank_score += 5.0 * (1 if shared_descs else 0)
    rank_score += min(len(shared_titles), 3) * 2.0
    rank_score += lexical * 6.0
    rank_score += support_score * 2.0

    return {
        "pair_uid": step4.pair_uid(left, right),
        "candidate_language": candidate_language,
        "data_bucket": data_bucket,
        "candidate_scope": step4.candidate_scope(left.alias_normalized, right.alias_normalized),
        "seller_uid_left": left.seller_uid,
        "seller_uid_right": right.seller_uid,
        "source_market_raw_left": left.source_market_raw,
        "source_market_raw_right": right.source_market_raw,
        "source_seller_raw_left": left.source_seller_raw,
        "source_seller_raw_right": right.source_seller_raw,
        "alias_normalized_left": left.alias_normalized,
        "alias_normalized_right": right.alias_normalized,
        "alias_relation": step4.alias_relation(left.alias_normalized, right.alias_normalized),
        "same_market_raw": str(left.source_market_raw == right.source_market_raw).lower(),
        "item_count_left": left.item_count,
        "item_count_right": right.item_count,
        "shared_contact_count": len(shared_contacts),
        "shared_contact_types": "|".join(sorted({value.split(":", 1)[0] for value in shared_contacts})),
        "shared_contact_values": " || ".join(shared_contacts[:5]),
        "shared_title_count": len(shared_titles),
        "shared_title_values": " || ".join(shared_titles),
        "shared_description_count": len(shared_descs),
        "shared_description_values": " || ".join(shared_descs),
        "shared_category_count": len(shared_categories),
        "shared_category_values": " || ".join(shared_categories),
        "lexical_similarity": round(lexical, 6),
        "structural_support_score": round(support_score, 6),
        "item_count_ratio": "" if item_ratio is None else round(item_ratio, 6),
        "price_median_ratio": "" if price_ratio is None else round(price_ratio, 6),
        "style_distance_l1": round(style_l1, 6),
        "shared_pgp_fingerprint_count": 0,
        "shared_pgp_fingerprint_values": "",
        "pgp_alias_hit_count_left": len(left.pgp_fingerprints),
        "pgp_alias_hit_count_right": len(right.pgp_fingerprints),
        "candidate_rule_hits": "|".join(sorted(rule_hits)),
        "candidate_rule_count": len(rule_hits),
        "candidate_rank_score": round(rank_score, 6),
        "review_priority": "high",
        "left_preview": left.preview,
        "right_preview": right.preview,
        "review_status": "pending",
        "review_label": "",
        "reviewer_id": "",
        "review_notes": "",
    }


def queue_row_from_candidate(candidate_row: dict, active_row: dict | None, *, rank: int, bucket_cfg: dict, extras: dict) -> dict:
    base = dict(candidate_row)
    if active_row:
        base.update(active_row)
    if bucket_cfg["bucket_id"] == "positive_component_transitive_closure":
        base["candidate_scope"] = "positive_component_closure_audit"
    base["review_stratum"] = active_row.get("review_stratum") if active_row else step5_strata.classify_review_stratum(candidate_row)
    base["balanced_review_rank"] = active_row.get("balanced_review_rank") if active_row else extras["balanced_review_rank"]
    base["review_priority"] = bucket_cfg["review_priority"]
    base["candidate_rule_hits"] = candidate_row.get("candidate_rule_hits", base.get("candidate_rule_hits", ""))
    base["candidate_rank_score"] = candidate_row.get("candidate_rank_score", base.get("candidate_rank_score", ""))
    base["review_status"] = "pending"
    base["review_label"] = ""
    base["reviewer_id"] = ""
    base["review_notes"] = bucket_cfg["review_notes"]
    base.update(
        {
            "positive_anchor_queue_rank": rank,
            "target_bucket": bucket_cfg["bucket_id"],
            "target_reason": bucket_cfg["target_reason"],
            "target_action": "review_existing_pending_queue_row" if active_row else "append_candidate_then_review",
            "suggested_label": bucket_cfg["suggested_label"],
            "suggested_confidence": bucket_cfg["suggested_confidence"],
        }
    )
    base.update(extras)
    return base


def profile_text(profile: dict) -> str:
    fields = (
        "signature_title_concat",
        "title_concat_top",
        "signature_description_concat",
        "description_concat_top",
        "profile_text",
    )
    return "\n".join(str(profile.get(field, "") or "") for field in fields)


PGP_BLOCK_RE = re.compile(r"-----BEGIN PGP PUBLIC KEY BLOCK-----(.*?)-----END PGP PUBLIC KEY BLOCK-----", re.I | re.S)


def normalize_url_token(raw_value: str, excluded_hosts: set[str]) -> tuple[str, str] | None:
    raw = normalize_text(raw_value).strip("[](){}<>,;:'\"，。；、")
    if not raw:
        return None
    parse_value = raw if re.match(r"https?://", raw, re.I) else f"http://{raw}"
    parsed = urlparse(parse_value)
    host = parsed.netloc.lower()
    if not host:
        return None
    host = host.split("@")[-1].split(":")[0]
    host = host.strip(".")
    if not host or "." not in host:
        return None
    if host in excluded_hosts or any(host.endswith("." + excluded) for excluded in excluded_hosts):
        return None

    path = re.sub(r"/+", "/", parsed.path or "").strip("/")
    if host.endswith(".onion") and re.match(r"(viewtopic|index|forum|thread)", path, re.I):
        return None
    if path:
        segments = [segment for segment in path.split("/") if segment][:2]
        normalized = host + "/" + "/".join(segments)
    else:
        normalized = host
    normalized = normalized.lower().rstrip("/")
    if len(normalized) < 6 or len(normalized) > 140:
        return None
    return normalized, host


def normalize_identity_token(contact_type: str, raw_value: str, excluded_tokens: set[str], excluded_hosts: set[str]) -> str:
    raw = normalize_text(raw_value).strip("[](){}<>,;:'\"，。；、")
    if not raw:
        return ""
    lowered = raw.lower().strip("._-")

    if contact_type == "email" or contact_type == "jabber":
        match = re.fullmatch(r"[a-z0-9._%+\-]+@[a-z0-9.\-]+\.[a-z]{2,24}", lowered)
        if not match:
            return ""
        local = lowered.split("@", 1)[0]
        if local.startswith(".") or local.endswith(".") or ".." in local:
            return ""
        return lowered

    if contact_type in {"telegram", "wickr", "wechat"}:
        lowered = lowered.lstrip("@")
        lowered = re.sub(r"[^a-z0-9_.\-]", "", lowered).strip("._-")
        if len(lowered) < 5 or len(lowered) > 32:
            return ""
        if lowered in excluded_tokens or lowered.isdigit() or not re.search(r"[a-z]", lowered):
            return ""
        return lowered

    if contact_type == "qq":
        digits = re.sub(r"\D", "", raw)
        if re.fullmatch(r"[1-9]\d{4,11}", digits or ""):
            return digits
        return ""

    if contact_type == "phone":
        digits = re.sub(r"\D", "", raw)
        if 7 <= len(digits) <= 15 and len(set(digits)) >= 3:
            return digits
        return ""

    if contact_type == "crypto_wallet":
        wallet = re.sub(r"[^a-zA-Z0-9]", "", raw)
        if re.fullmatch(r"bc1[ac-hj-np-z02-9]{11,71}", wallet, re.I):
            return wallet.lower()
        if re.fullmatch(r"[13][a-km-zA-HJ-NP-Z1-9]{25,34}", wallet):
            return wallet
        if re.fullmatch(r"0x[a-fA-F0-9]{40}", wallet):
            return wallet.lower()
        if re.fullmatch(r"T[A-HJ-NP-Za-km-z1-9]{33}", wallet):
            return wallet
        return ""

    if contact_type == "external_url":
        normalized = normalize_url_token(raw, excluded_hosts)
        return normalized[0] if normalized else ""

    return ""


def candidate_has_pair_support(candidate_row: dict, *, external_url: bool = False) -> bool:
    shared_text = to_int(candidate_row.get("shared_title_count")) > 0 or to_int(candidate_row.get("shared_description_count")) > 0
    lexical = to_float(candidate_row.get("lexical_similarity"))
    structural = to_float(candidate_row.get("structural_support_score"))
    if external_url:
        return shared_text or lexical >= 0.14 or structural >= 0.5
    return shared_text or lexical >= 0.05 or structural >= 0.2


def build_supplemental_contact_index(raw_profiles: list[dict], policy: dict) -> tuple[dict[tuple[str, str], list[dict]], Counter]:
    cfg = policy["selection"]["supplemental_contact_extraction"]
    excluded = {str(item).lower() for item in cfg.get("exclude_tokens", [])}
    excluded_hosts = {str(item).lower() for item in cfg.get("external_url_excluded_hosts", [])}
    false_context_patterns = [
        re.compile(str(pattern), re.I) for pattern in cfg.get("false_positive_context_patterns", [])
    ]
    contact_cue_re = re.compile(
        r"(telegram|\btg\b|电报|飞机|纸飞机|wickr|wechat|微信|vx|wx|jabber|xmpp|qq|联系|联系方式|加|客服|咨询|合作|拍前|找我|留下)",
        re.I,
    )
    patterns = [
        ("email", "plain_email", re.compile(r"([a-z0-9._%+\-]+@[a-z0-9.\-]+\.[a-z]{2,24})", re.I)),
        ("telegram", "strong_contact_cue", re.compile(r"(?:telegram|\btg\b|电报|飞机|纸飞机|tele)\s*[:：@]?\s*([a-zA-Z][a-zA-Z0-9_]{4,31})", re.I)),
        ("telegram", "strong_contact_url", re.compile(r"(?:t\.me|telegram\.me)\s*/\s*([a-zA-Z][a-zA-Z0-9_]{4,31})", re.I)),
        ("telegram", "cue_at_handle", re.compile(r"(?<![a-zA-Z0-9._%+\-])@([a-zA-Z][a-zA-Z0-9_]{4,31})(?![a-zA-Z0-9._%+\-]*\.[a-zA-Z]{2,})")),
        ("wickr", "strong_contact_cue", re.compile(r"(?:wickr|wickrme)\s*[:：@]?\s*([a-zA-Z][a-zA-Z0-9_.\-]{4,31})", re.I)),
        ("wechat", "strong_contact_cue", re.compile(r"(?:微信|wechat|weixin|\bvx\b|\bwx\b)\s*[:：号@]?\s*([a-zA-Z][a-zA-Z0-9_\-]{4,31})", re.I)),
        ("jabber", "strong_contact_cue", re.compile(r"(?:jabber|xmpp)\s*[:：]?\s*([a-z0-9._%+\-]+@[a-z0-9.\-]+\.[a-z]{2,24})", re.I)),
        ("qq", "strong_contact_cue", re.compile(r"(?:qq|QQ|企鹅)\s*[:：号]?\s*([1-9]\d{4,11})")),
        ("phone", "strong_contact_cue", re.compile(r"(?:whatsapp|phone|tel|电话|手机|联系|联系方式)[^0-9A-Za-z+]{0,8}([+]?\d[\d\-\s()]{7,}\d)", re.I)),
        ("crypto_wallet", "btc_wallet", re.compile(r"\b(bc1[ac-hj-np-z02-9]{11,71}|[13][a-km-zA-HJ-NP-Z1-9]{25,34})\b")),
        ("crypto_wallet", "eth_wallet", re.compile(r"\b(0x[a-fA-F0-9]{40})\b")),
        ("crypto_wallet", "tron_wallet", re.compile(r"\b(T[A-HJ-NP-Za-km-z1-9]{33})\b")),
        ("external_url", "plain_url", re.compile(r"\b((?:https?://|www\.)[^\s\]）)>,，。；;'\"`]{6,160})", re.I)),
        ("external_url", "bare_domain", re.compile(r"(?<![@\w.-])([a-z0-9][a-z0-9\-]{1,40}\.(?:com|net|org|vip|cn|cc|top|xyz|onion)(?:/[^\s\]）)>,，。；;'\"`]{1,120})?)", re.I)),
    ]
    enabled_types = set(cfg["contact_types"])
    rows_by_token: dict[tuple[str, str], list[dict]] = defaultdict(list)
    rejected = Counter()

    for profile in raw_profiles:
        text = profile_text(profile)
        text_without_pgp = PGP_BLOCK_RE.sub(" ", text)
        seen: set[tuple[str, str]] = set()
        for contact_type, evidence_level, pattern in patterns:
            if contact_type not in enabled_types:
                continue
            search_text = text if contact_type == "email" else text_without_pgp
            for match in pattern.finditer(search_text):
                raw_token = match.group(1)
                context = search_text[max(0, match.start() - 60) : match.end() + 60].replace("\n", " ")
                token = normalize_identity_token(contact_type, raw_token, excluded, excluded_hosts)
                if not token or token in excluded:
                    rejected["excluded_token"] += 1
                    continue
                if evidence_level == "cue_at_handle" and cfg.get("require_contact_cue_for_at_handle", True):
                    if not contact_cue_re.search(context):
                        rejected["at_handle_without_contact_cue"] += 1
                        continue
                if any(pattern.search(context) for pattern in false_context_patterns):
                    rejected["false_positive_context"] += 1
                    continue
                if contact_type == "external_url" and re.search(r"(youtube|mega\.nz|nordvpn|expressvpn|viewtopic\.php)", token, re.I):
                    rejected["external_url_public_or_forum_host"] += 1
                    continue
                key = (contact_type, token)
                if key in seen:
                    continue
                seen.add(key)
                rows_by_token[key].append(
                    {
                        "seller_uid": profile["seller_uid"],
                        "contact_type": contact_type,
                        "token": token,
                        "evidence_level": evidence_level,
                        "context": step4.capped_text(context, 240),
                    }
                )

    return rows_by_token, rejected


def clean_positive_components(frozen_rows: list[dict], exclude_conflicted: bool) -> dict[str, dict]:
    positive_rows = [
        row
        for row in frozen_rows
        if row.get("review_label") == "positive" and row.get("usable_for_supervision") == "1"
    ]
    parent: dict[str, str] = {}

    def find(value: str) -> str:
        parent.setdefault(value, value)
        if parent[value] != value:
            parent[value] = find(parent[value])
        return parent[value]

    def union(left: str, right: str) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    for row in positive_rows:
        union(row["seller_uid_left"], row["seller_uid_right"])

    components: dict[str, dict] = defaultdict(lambda: {"sellers": set(), "positive_rows": [], "conflict_rows": []})
    for row in positive_rows:
        component_id = find(row["seller_uid_left"])
        components[component_id]["sellers"].update([row["seller_uid_left"], row["seller_uid_right"]])
        components[component_id]["positive_rows"].append(row)

    seller_to_component = {
        seller_uid: component_id
        for component_id, component in components.items()
        for seller_uid in component["sellers"]
    }
    for row in frozen_rows:
        if row.get("review_label") not in {"negative", "uncertain"}:
            continue
        left_component = seller_to_component.get(row.get("seller_uid_left", ""))
        if left_component and left_component == seller_to_component.get(row.get("seller_uid_right", "")):
            components[left_component]["conflict_rows"].append(row)

    if not exclude_conflicted:
        return components
    return {
        component_id: component
        for component_id, component in components.items()
        if not component["conflict_rows"]
    }


def bucket_config(policy: dict, bucket_id: str) -> dict:
    for bucket in policy["selection"]["buckets"]:
        if bucket["bucket_id"] == bucket_id:
            return bucket
    raise KeyError(bucket_id)


def main() -> None:
    args = parse_args()
    policy_path = Path(args.policy_path)
    if not policy_path.is_absolute():
        policy_path = ROOT / policy_path
    policy = load_json(policy_path)
    inputs = policy["inputs"]

    step4_schema = load_json(ROOT / inputs["step4_schema"])
    raw_profiles = step4.load_jsonl(ROOT / inputs["seller_profiles"])
    pgp_alias_map = step4.load_pgp_alias_map()
    profiles = step4.build_seller_profiles(
        raw_profiles,
        "zh_target_strict",
        "zh",
        set(step4_schema["filtering_policy"]["contact_noise_stopwords"]),
        step4_schema["filtering_policy"]["content_minimums"],
        pgp_alias_map,
    )
    step4.compute_retrieval_weights(profiles, step4_schema["retrieval_policy"]["zh_target_strict"])
    profile_by_uid = {profile.seller_uid: profile for profile in profiles}
    candidate_rows, candidate_fieldnames = load_csv(ROOT / inputs["step4_candidates"])
    active_rows, active_fieldnames = load_csv(ROOT / inputs["active_review_queue"])
    frozen_rows, _ = load_csv(ROOT / inputs["active_frozen_labels"])

    candidate_index = {row["pair_uid"]: row for row in candidate_rows}
    active_index = {row["pair_uid"]: row for row in active_rows}
    frozen_pair_uids = {row["pair_uid"] for row in frozen_rows}
    reviewed_pair_uids = {
        row["pair_uid"]
        for row in active_rows
        if normalize_label(row.get("review_label")) in {"positive", "negative", "uncertain"}
    }
    max_active_rank = max((to_int(row.get("balanced_review_rank")) for row in active_rows), default=0)
    selected: list[dict] = []
    selected_uids: set[str] = set()
    appended_rank = max_active_rank

    def next_new_rank() -> int:
        nonlocal appended_rank
        appended_rank += 1
        return appended_rank

    direct_bucket = bucket_config(policy, "direct_contact_existing_pending")
    for active_row in active_rows:
        pair_uid = active_row["pair_uid"]
        if pair_uid in frozen_pair_uids or pair_uid in selected_uids or not is_pending(active_row):
            continue
        has_direct = normalize_text(active_row.get("shared_contact_values")) or normalize_text(
            active_row.get("shared_pgp_fingerprint_values")
        )
        if not has_direct:
            continue
        candidate_row = candidate_index[pair_uid]
        selected_uids.add(pair_uid)
        selected.append(
            queue_row_from_candidate(
                candidate_row,
                active_row,
                rank=0,
                bucket_cfg=direct_bucket,
                extras={
                    "balanced_review_rank": active_row["balanced_review_rank"],
                    "anchor_token": active_row.get("shared_contact_values") or active_row.get("shared_pgp_fingerprint_values"),
                    "anchor_type": candidate_row.get("shared_contact_types", ""),
                    "anchor_frequency": "",
                    "anchor_evidence_level": "existing_step4_direct_identifier",
                    "left_anchor_context": "",
                    "right_anchor_context": "",
                    "source_existing_active_bool": 1,
                    "source_existing_step4_bool": 1,
                    "source_positive_component_id": "",
                    "source_positive_component_size": "",
                    "closure_source_pair_uids": "",
                },
            )
        )

    supplemental_bucket = bucket_config(policy, "supplemental_shared_seller_contact")
    rare_identity_bucket = bucket_config(policy, "rare_identity_with_text_overlap")
    external_url_bucket = bucket_config(policy, "rare_external_url_with_text_overlap")
    contact_index, rejected_contacts = build_supplemental_contact_index(raw_profiles, policy)
    contact_cfg = policy["selection"]["supplemental_contact_extraction"]
    min_freq = int(contact_cfg["min_token_seller_frequency"])
    default_max_freq = int(contact_cfg["max_token_seller_frequency"])
    max_freq_by_type = {
        str(key): int(value)
        for key, value in contact_cfg.get("max_token_seller_frequency_by_type", {}).items()
    }
    require_pair_support = bool(contact_cfg.get("require_pair_text_support", True))
    external_requires_support = bool(contact_cfg.get("external_url_requires_text_support", True))
    supplemental_candidates = []
    supplemental_filter_counts = Counter()
    for (contact_type, token), evidence_rows in contact_index.items():
        seller_rows = {}
        for evidence in evidence_rows:
            seller_rows.setdefault(evidence["seller_uid"], evidence)
        frequency = len(seller_rows)
        max_freq = max_freq_by_type.get(contact_type, default_max_freq)
        if frequency < min_freq or frequency > max_freq:
            supplemental_filter_counts["token_frequency_outside_bounds"] += 1
            continue
        sorted_sellers = sorted(seller_rows)
        for left_uid, right_uid in combinations(sorted_sellers, 2):
            pair_uid = "||".join(sorted([left_uid, right_uid]))
            if pair_uid in selected_uids or pair_uid in frozen_pair_uids or pair_uid in reviewed_pair_uids:
                supplemental_filter_counts["already_selected_or_reviewed"] += 1
                continue
            if pair_uid in active_index and not is_pending(active_index[pair_uid]):
                supplemental_filter_counts["active_row_not_pending"] += 1
                continue
            left_profile = profile_by_uid[left_uid]
            right_profile = profile_by_uid[right_uid]
            candidate_row = candidate_index.get(pair_uid) or make_candidate_row(
                left_profile,
                right_profile,
                shared_contacts=[f"{contact_type}:{token}"],
                extra_rule_hits=["supplemental_contact_exact"],
            )
            is_external_url = contact_type == "external_url"
            if (is_external_url and external_requires_support) or (not is_external_url and require_pair_support):
                if not candidate_has_pair_support(candidate_row, external_url=is_external_url):
                    supplemental_filter_counts["pair_lacks_text_or_structural_support"] += 1
                    continue
            left_evidence = seller_rows[left_uid]
            right_evidence = seller_rows[right_uid]
            evidence_levels = sorted({left_evidence["evidence_level"], right_evidence["evidence_level"]})
            if contact_type == "external_url":
                bucket = external_url_bucket
            elif contact_type in {"telegram", "qq"}:
                bucket = supplemental_bucket
            else:
                bucket = rare_identity_bucket
            supplemental_candidates.append(
                (
                    -to_float(candidate_row.get("candidate_rank_score")),
                    frequency,
                    pair_uid,
                    candidate_row,
                    active_index.get(pair_uid),
                    bucket,
                    {
                        "balanced_review_rank": active_index[pair_uid]["balanced_review_rank"] if pair_uid in active_index else next_new_rank(),
                        "anchor_token": f"{contact_type}:{token}",
                        "anchor_type": contact_type,
                        "anchor_frequency": frequency,
                        "anchor_evidence_level": "|".join(evidence_levels),
                        "left_anchor_context": left_evidence["context"],
                        "right_anchor_context": right_evidence["context"],
                        "source_existing_active_bool": int(pair_uid in active_index),
                        "source_existing_step4_bool": int(pair_uid in candidate_index),
                        "source_positive_component_id": "",
                        "source_positive_component_size": "",
                        "closure_source_pair_uids": "",
                    },
                )
            )

    for _sort_score, _freq, pair_uid, candidate_row, active_row, bucket, extras in sorted(
        supplemental_candidates,
        key=lambda item: (item[0], item[1], item[2]),
    ):
        if pair_uid in selected_uids:
            continue
        selected_uids.add(pair_uid)
        selected.append(
            queue_row_from_candidate(
                candidate_row,
                active_row,
                rank=0,
                bucket_cfg=bucket,
                extras=extras,
            )
        )

    alias_bucket = bucket_config(policy, "cross_market_unique_alias_closure")
    alias_to_profiles: dict[str, list] = defaultdict(list)
    for profile in profiles:
        alias = normalize_text(profile.alias_normalized).lower()
        if not alias or alias.startswith("/shop/") or re.fullmatch(r"\d+", alias) or len(alias) < 4:
            continue
        alias_to_profiles[alias].append(profile)
    for alias, alias_profiles in sorted(alias_to_profiles.items()):
        if len(alias_profiles) < 2 or len(alias_profiles) > 6:
            continue
        markets = {profile.source_market_raw for profile in alias_profiles}
        if len(markets) < 2:
            continue
        for left_profile, right_profile in combinations(sorted(alias_profiles, key=lambda item: item.seller_uid), 2):
            if left_profile.source_market_raw == right_profile.source_market_raw:
                continue
            pair_uid = step4.pair_uid(left_profile, right_profile)
            if pair_uid in selected_uids or pair_uid in frozen_pair_uids or pair_uid in reviewed_pair_uids:
                continue
            if pair_uid in active_index and not is_pending(active_index[pair_uid]):
                continue
            candidate_row = candidate_index.get(pair_uid) or make_candidate_row(
                left_profile,
                right_profile,
                shared_contacts=[],
                extra_rule_hits=["cross_market_unique_alias_closure"],
            )
            shared_direct_contact = False
            for contact_type in left_profile.contact_values_by_type:
                if left_profile.contact_values_by_type[contact_type] & right_profile.contact_values_by_type[contact_type]:
                    shared_direct_contact = True
                    break
            shared_text = to_int(candidate_row.get("shared_title_count")) > 0 or to_int(candidate_row.get("shared_description_count")) > 0
            if not shared_direct_contact and not shared_text and to_float(candidate_row.get("lexical_similarity")) < 0.2:
                continue
            candidate_row = dict(candidate_row)
            candidate_row["candidate_scope"] = "sockpuppet_primary"
            selected_uids.add(pair_uid)
            selected.append(
                queue_row_from_candidate(
                    candidate_row,
                    active_index.get(pair_uid),
                    rank=0,
                    bucket_cfg=alias_bucket,
                    extras={
                        "balanced_review_rank": active_index[pair_uid]["balanced_review_rank"] if pair_uid in active_index else next_new_rank(),
                        "anchor_token": f"alias:{alias}",
                        "anchor_type": "cross_market_unique_alias",
                        "anchor_frequency": len(alias_profiles),
                        "anchor_evidence_level": "cross_market_unique_alias_plus_contact_or_content",
                        "left_anchor_context": left_profile.preview,
                        "right_anchor_context": right_profile.preview,
                        "source_existing_active_bool": int(pair_uid in active_index),
                        "source_existing_step4_bool": int(pair_uid in candidate_index),
                        "source_positive_component_id": "",
                        "source_positive_component_size": "",
                        "closure_source_pair_uids": "",
                    },
                )
            )

    closure_bucket = bucket_config(policy, "positive_component_transitive_closure")
    closure_cfg = policy["selection"]["positive_component_closure"]
    if closure_cfg.get("enabled", True):
        positive_edge_uids = {
            row["pair_uid"]
            for row in frozen_rows
            if row.get("review_label") == "positive" and row.get("usable_for_supervision") == "1"
        }
        components = clean_positive_components(
            frozen_rows,
            bool(closure_cfg.get("exclude_components_with_reviewed_negative_or_uncertain_internal_edges", True)),
        )
        for component_id, component in sorted(components.items()):
            edge_count = 0
            positive_sources = sorted(row["pair_uid"] for row in component["positive_rows"])
            for left_uid, right_uid in combinations(sorted(component["sellers"]), 2):
                pair_uid = "||".join(sorted([left_uid, right_uid]))
                if pair_uid in positive_edge_uids or pair_uid in selected_uids or pair_uid in reviewed_pair_uids:
                    continue
                if pair_uid in active_index and not is_pending(active_index[pair_uid]):
                    continue
                if edge_count >= int(closure_cfg.get("max_new_edges_per_component", 5)):
                    break
                candidate_row = candidate_index.get(pair_uid) or make_candidate_row(
                    profile_by_uid[left_uid],
                    profile_by_uid[right_uid],
                    shared_contacts=[],
                    extra_rule_hits=["positive_component_transitive_closure"],
                )
                candidate_row = dict(candidate_row)
                candidate_row["candidate_scope"] = str(
                    closure_cfg.get("audit_only_candidate_scope", "positive_component_closure_audit")
                )
                extras = {
                    "balanced_review_rank": active_index[pair_uid]["balanced_review_rank"] if pair_uid in active_index else next_new_rank(),
                    "anchor_token": "",
                    "anchor_type": "positive_component",
                    "anchor_frequency": "",
                    "anchor_evidence_level": "clean_reviewed_positive_component_closure",
                    "left_anchor_context": "",
                    "right_anchor_context": "",
                    "source_existing_active_bool": int(pair_uid in active_index),
                    "source_existing_step4_bool": int(pair_uid in candidate_index),
                    "source_positive_component_id": component_id,
                    "source_positive_component_size": len(component["sellers"]),
                    "closure_source_pair_uids": " || ".join(positive_sources[:8]),
                }
                selected_uids.add(pair_uid)
                selected.append(
                    queue_row_from_candidate(
                        candidate_row,
                        active_index.get(pair_uid),
                        rank=0,
                        bucket_cfg=closure_bucket,
                        extras=extras,
                    )
                )
                edge_count += 1

    selected.sort(
        key=lambda row: (
            {
                "supplemental_shared_seller_contact": 0,
                "rare_identity_with_text_overlap": 1,
                "cross_market_unique_alias_closure": 2,
                "rare_external_url_with_text_overlap": 3,
                "positive_component_transitive_closure": 4,
                "direct_contact_existing_pending": 5,
            }.get(
                row["target_bucket"],
                9,
            ),
            -to_float(row.get("candidate_rank_score")),
            to_int(row.get("balanced_review_rank")),
            row["pair_uid"],
        )
    )
    for idx, row in enumerate(selected, start=1):
        row["positive_anchor_queue_rank"] = idx

    output_fieldnames = []
    for key in active_fieldnames + TARGET_EXTRA_FIELDS + candidate_fieldnames:
        if key not in output_fieldnames:
            output_fieldnames.append(key)

    output_queue_path = ROOT / policy["outputs"]["targeted_review_queue"]
    output_summary_path = ROOT / policy["outputs"]["summary"]
    write_csv(output_queue_path, selected, output_fieldnames)

    summary = {
        "policy_path": str(policy_path.relative_to(ROOT)),
        "queue_version": policy["queue_version"],
        "targeted_review_queue": str(output_queue_path.relative_to(ROOT)),
        "selected_row_count": len(selected),
        "selected_bucket_counts": dict(Counter(row["target_bucket"] for row in selected)),
        "selected_action_counts": dict(Counter(row["target_action"] for row in selected)),
        "selected_review_stratum_counts": dict(Counter(row["review_stratum"] for row in selected)),
        "source_existing_active_counts": dict(Counter(str(row["source_existing_active_bool"]) for row in selected)),
        "source_existing_step4_counts": dict(Counter(str(row["source_existing_step4_bool"]) for row in selected)),
        "supplemental_contact_token_count": len(contact_index),
        "supplemental_contact_rejected_counts": dict(rejected_contacts),
        "supplemental_contact_filter_counts": dict(supplemental_filter_counts),
        "active_pending_direct_identifier_rows": sum(1 for row in selected if row["target_bucket"] == "direct_contact_existing_pending"),
        "closure_rows": sum(1 for row in selected if row["target_bucket"] == "positive_component_transitive_closure"),
        "target_queue_size": policy["selection"].get("target_queue_size", {}),
        "positive_split_targets": policy["selection"].get("positive_split_targets", {}),
        "current_positive_supervision_by_split_before_review": dict(
            Counter(
                row.get("split_name", "")
                for row in frozen_rows
                if row.get("review_label") == "positive" and row.get("usable_for_supervision") == "1"
            )
        ),
        "acceptance_checks": {
            "no_duplicate_pair_uid": len(selected_uids) == len(selected),
            "all_selected_rows_pending_or_new": all(
                row["source_existing_active_bool"] == 0 or is_pending(active_index[row["pair_uid"]])
                for row in selected
            ),
            "all_new_rows_have_candidate_fields": all(
                all(normalize_text(row.get(field)) != "" for field in ("seller_uid_left", "seller_uid_right", "candidate_scope"))
                for row in selected
                if row["source_existing_active_bool"] == 0
            ),
        },
        "top_rows": [
            {
                "positive_anchor_queue_rank": row["positive_anchor_queue_rank"],
                "target_bucket": row["target_bucket"],
                "pair_uid": row["pair_uid"],
                "anchor_token": row.get("anchor_token", ""),
                "candidate_rank_score": row.get("candidate_rank_score", ""),
                "review_stratum": row.get("review_stratum", ""),
            }
            for row in selected[:20]
        ],
    }
    write_json(output_summary_path, summary)

    print(f"Wrote positive-anchor review queue: {output_queue_path}")
    print(f"Wrote summary: {output_summary_path}")
    print(f"selected_row_count={len(selected)} bucket_counts={summary['selected_bucket_counts']}")


if __name__ == "__main__":
    main()
