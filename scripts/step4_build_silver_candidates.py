from __future__ import annotations

import csv
import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from itertools import combinations
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCHEMA_PATH = ROOT / "schema" / "step4_silver_candidate_schema.json"
STEP2_PGP_PATH = ROOT / "reports" / "step2_aux_pgp_evidence_manifest.csv"
PROFILE_PATHS = {
    "en_content_train_pool": ROOT / "reports" / "step3_seller_profiles.en_content_train_pool.jsonl",
    "zh_target_strict": ROOT / "reports" / "step3_seller_profiles.zh_target_strict.jsonl",
    "zh_target_aux": ROOT / "reports" / "step3_seller_profiles.zh_target_aux.jsonl",
}
SUMMARY_PATH = ROOT / "reports" / "step4_candidate_summary.json"
PAIR_OUTPUTS = {
    "en_content_train_pool": ROOT / "reports" / "step4_en_silver_candidate_pairs.csv",
    "zh_target_strict": ROOT / "reports" / "step4_zh_target_strict_silver_candidate_pairs.csv",
    "zh_target_aux": ROOT / "reports" / "step4_zh_target_aux_silver_candidate_pairs.csv",
}
QUEUE_OUTPUTS = {
    "en_content_train_pool": ROOT / "reports" / "step4_en_manual_review_queue.csv",
    "zh_target_strict": ROOT / "reports" / "step4_zh_target_strict_manual_review_queue.csv",
    "zh_target_aux": ROOT / "reports" / "step4_zh_target_aux_manual_review_queue.csv",
}

EN_STOPWORDS = {
    "about", "after", "again", "also", "and", "any", "are", "back", "been", "before", "between",
    "but", "can", "custom", "description", "details", "direct", "from", "free", "fresh", "get",
    "has", "have", "high", "into", "item", "its", "just", "like", "listing", "more", "most",
    "new", "not", "only", "order", "our", "out", "please", "product", "quality", "sale", "same",
    "ship", "shipping", "specified", "stealth", "test", "that", "the", "their", "this", "title",
    "too", "use", "very", "was", "with", "you", "your"
}

GENERIC_TITLE_VALUES = {
    "adderall", "cocaine", "custom", "custom listing", "custom order", "listing title corrupted add new title",
    "mdma", "no item name", "oxycodone", "oxycodone 30mg", "percocet", "private listing", "subutex", "test", "xanax"
}

GENERIC_DESCRIPTION_VALUES = {
    "custom", "custom listing", "custom order", "no description specified", "test", "无详情描述"
}

CONTACT_TYPE_ORDER = ("email", "telegram", "wickr", "wechat", "qq", "phone")
REVIEW_PRIORITY_ORDER = {"high": 0, "medium": 1, "low": 2}
CANDIDATE_SCOPE_ORDER = {"sockpuppet_primary": 0, "same_alias_identity_continuity": 1}


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip())


def is_cjk_char(ch: str) -> bool:
    code = ord(ch)
    return (
        0x4E00 <= code <= 0x9FFF
        or 0x3400 <= code <= 0x4DBF
        or 0x20000 <= code <= 0x2A6DF
        or 0x2A700 <= code <= 0x2B73F
        or 0x2B740 <= code <= 0x2B81F
        or 0x2B820 <= code <= 0x2CEAF
        or 0xF900 <= code <= 0xFAFF
    )


def contains_cjk(text: str) -> bool:
    return any(is_cjk_char(ch) for ch in text)


def count_cjk(text: str) -> int:
    return sum(1 for ch in text if is_cjk_char(ch))


def normalize_duplicate_text(text: str) -> str:
    text = normalize_space(text).lower()
    if not text:
        return ""
    if contains_cjk(text):
        return "".join(ch for ch in text if is_cjk_char(ch) or ch.isalnum())
    tokens = re.findall(r"[a-z0-9]+", text)
    return " ".join(tokens)


def normalize_category(text: str) -> str:
    return normalize_duplicate_text(text)


def safe_median(stats: dict) -> float | None:
    if not stats:
        return None
    value = stats.get("median")
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def normalize_contact_value(contact_type: str, value: str, stopwords: set[str]) -> str:
    raw = normalize_space(value).strip("[](){}<>,;:'\"")
    if not raw:
        return ""

    if contact_type == "email":
        matches = re.findall(r"[a-z0-9._%+\-]+@[a-z0-9.\-]+\.[a-z]{2,24}", raw.lower())
        valid_matches = []
        for match in matches:
            local_part = match.split("@", 1)[0]
            if local_part.startswith(".") or local_part.endswith(".") or ".." in local_part:
                continue
            valid_matches.append(match)
        unique_matches = sorted(set(valid_matches))
        if len(unique_matches) == 1:
            return unique_matches[0]
        return ""

    if contact_type in {"telegram", "wickr", "wechat"}:
        raw = raw.lower().lstrip("@")
        raw = re.sub(r"[^a-z0-9_.\-]", "", raw)
        raw = raw.strip("._-")
        if len(raw) < 5 or len(raw) > 32:
            return ""
        if raw in stopwords or raw.isdigit():
            return ""
        if not re.search(r"[a-z]", raw):
            return ""
        return raw

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

    return ""


def english_word_tokens(text: str) -> list[str]:
    tokens = re.findall(r"[a-z][a-z0-9]{2,23}", text.lower())
    return [token for token in tokens if token not in EN_STOPWORDS]


def chinese_ngrams(text: str) -> Counter:
    grams: Counter = Counter()
    for seq in re.findall(r"[\u3400-\u9fff]{2,}", text):
        for n in (2, 3):
            if len(seq) < n:
                continue
            for idx in range(len(seq) - n + 1):
                grams[seq[idx : idx + n]] += 1
    return grams


def ratio_or_zero(a: float | None, b: float | None) -> float | None:
    if a is None or b is None or a <= 0 or b <= 0:
        return None
    lo = min(a, b)
    hi = max(a, b)
    return lo / hi


def jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    union = a | b
    if not union:
        return 0.0
    return len(a & b) / len(union)


def capped_text(value: str, width: int = 160) -> str:
    value = normalize_space(value)
    if len(value) <= width:
        return value
    return value[: width - 3] + "..."


def is_contentful_title(norm_value: str, language: str, min_config: dict) -> bool:
    if not norm_value or norm_value in GENERIC_TITLE_VALUES:
        return False
    if language == "en":
        tokens = norm_value.split()
        return (
            len("".join(tokens)) >= int(min_config["en_title_min_chars"])
            or len(tokens) >= int(min_config["en_title_min_tokens"])
        )
    return count_cjk(norm_value) >= int(min_config["zh_title_min_cjk_chars"])


def is_contentful_description(norm_value: str, language: str, min_config: dict) -> bool:
    if not norm_value or norm_value in GENERIC_DESCRIPTION_VALUES:
        return False
    if language == "en":
        return len(norm_value.replace(" ", "")) >= int(min_config["en_description_min_chars"])
    return count_cjk(norm_value) >= int(min_config["zh_description_min_cjk_chars"])


def build_preview(profile: dict) -> str:
    title_values = [item["value"] for item in profile.get("signature_titles", [])[:1] if item.get("value")]
    if len(title_values) < 2:
        title_values.extend(item["value"] for item in profile.get("top_titles", [])[:2] if item.get("value"))
    title_values = title_values[:2]

    desc_values = [item["value"] for item in profile.get("signature_description_segments", [])[:1] if item.get("value")]
    if len(desc_values) < 1:
        desc_values.extend(item["value"] for item in profile.get("top_description_snippets", [])[:1] if item.get("value"))
    desc_values = desc_values[:1]

    category_values = [item["value"] for item in profile.get("top_categories", [])[:2] if item.get("value")]
    parts = []
    if title_values:
        parts.append("T: " + " || ".join(capped_text(value, 100) for value in title_values))
    if desc_values:
        parts.append("D: " + " || ".join(capped_text(value, 160) for value in desc_values))
    if category_values:
        parts.append("C: " + " || ".join(capped_text(value, 80) for value in category_values))
    return " | ".join(parts)


def build_retrieval_terms(profile: dict, language: str) -> Counter:
    terms: Counter = Counter()

    for item in profile.get("top_categories", [])[:8]:
        value = normalize_space(item.get("value", ""))
        if not value:
            continue
        if language == "en":
            for token in english_word_tokens(value):
                terms[f"cat:{token}"] += 3
        else:
            for token, count in chinese_ngrams(value).items():
                terms[f"cat:{token}"] += 3 * count

    for item in profile.get("signature_titles", [])[:6]:
        value = normalize_space(item.get("value", ""))
        if not value:
            continue
        if language == "en":
            words = english_word_tokens(value)
            for token in words:
                terms[f"sigtitle:{token}"] += 5
            for left, right in zip(words, words[1:]):
                terms[f"sigtitle2:{left}_{right}"] += 6
        else:
            for token, count in chinese_ngrams(value).items():
                terms[f"sigtitle:{token}"] += 5 * count
            for token in re.findall(r"[a-z0-9]{3,24}", value.lower()):
                terms[f"sigtitlelatin:{token}"] += 3

    for item in profile.get("top_titles", [])[:10]:
        value = normalize_space(item.get("value", ""))
        if not value:
            continue
        if language == "en":
            words = english_word_tokens(value)
            for token in words:
                terms[f"title:{token}"] += 3
            for left, right in zip(words, words[1:]):
                terms[f"title2:{left}_{right}"] += 4
        else:
            for token, count in chinese_ngrams(value).items():
                terms[f"title:{token}"] += 3 * count
            for token in re.findall(r"[a-z0-9]{3,24}", value.lower()):
                terms[f"titlelatin:{token}"] += 2

    for item in profile.get("signature_description_segments", [])[:8]:
        value = normalize_space(item.get("value", ""))
        if not value:
            continue
        if language == "en":
            words = english_word_tokens(value)
            for token in words:
                terms[f"sigdesc:{token}"] += 3
            for left, right in zip(words, words[1:]):
                terms[f"sigdesc2:{left}_{right}"] += 3
        else:
            for token, count in chinese_ngrams(value).items():
                terms[f"sigdesc:{token}"] += 3 * count
            for token in re.findall(r"[a-z0-9]{3,24}", value.lower()):
                terms[f"sigdesclatin:{token}"] += 2

    for item in profile.get("top_description_snippets", [])[:8]:
        value = normalize_space(item.get("value", ""))
        if not value:
            continue
        if language == "en":
            words = english_word_tokens(value)
            for token in words:
                terms[f"desc:{token}"] += 1
            for left, right in zip(words, words[1:]):
                terms[f"desc2:{left}_{right}"] += 1
        else:
            for token, count in chinese_ngrams(value).items():
                terms[f"desc:{token}"] += count
            for token in re.findall(r"[a-z0-9]{3,24}", value.lower()):
                terms[f"desclatin:{token}"] += 1

    return terms


@dataclass
class SellerProfile:
    seller_uid: str
    data_bucket: str
    language: str
    source_market_raw: str
    source_seller_raw: str
    alias_normalized: str
    item_count: int
    title_norm_to_raw: dict[str, str]
    desc_norm_to_raw: dict[str, str]
    category_norm_to_raw: dict[str, str]
    contact_values_by_type: dict[str, set[str]]
    category_norms: set[str]
    median_price: float | None
    style_vector: list[float]
    preview: str
    pgp_fingerprints: set[str]
    retrieval_terms: Counter
    retrieval_weights: dict[str, float] = field(default_factory=dict)
    retrieval_norm: float = 0.0


def load_pgp_alias_map() -> dict[str, set[str]]:
    alias_map: dict[str, set[str]] = defaultdict(set)
    with STEP2_PGP_PATH.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            alias = normalize_space(row.get("alias_normalized", "")).lower()
            fingerprint = normalize_space(row.get("fingerprint_raw", "") or row.get("fingerprint_short_raw", ""))
            if alias and fingerprint:
                alias_map[alias].add(fingerprint)
    return alias_map


def build_seller_profiles(
    rows: list[dict],
    data_bucket: str,
    language: str,
    stopwords: set[str],
    min_config: dict,
    pgp_alias_map: dict[str, set[str]],
) -> list[SellerProfile]:
    profiles: list[SellerProfile] = []
    for row in rows:
        title_norm_to_raw: dict[str, str] = {}
        for item in row.get("signature_titles", []):
            raw = normalize_space(item.get("value", ""))
            norm = normalize_duplicate_text(raw)
            if norm and is_contentful_title(norm, language, min_config):
                title_norm_to_raw.setdefault(norm, raw)
        for item in row.get("top_titles", []):
            raw = normalize_space(item.get("value", ""))
            norm = normalize_duplicate_text(raw)
            if norm and is_contentful_title(norm, language, min_config):
                title_norm_to_raw.setdefault(norm, raw)

        desc_norm_to_raw: dict[str, str] = {}
        for item in row.get("signature_description_segments", []):
            raw = normalize_space(item.get("value", ""))
            norm = normalize_duplicate_text(raw)
            if norm and is_contentful_description(norm, language, min_config):
                desc_norm_to_raw.setdefault(norm, raw)
        for item in row.get("top_description_snippets", []):
            raw = normalize_space(item.get("value", ""))
            norm = normalize_duplicate_text(raw)
            if norm and is_contentful_description(norm, language, min_config):
                desc_norm_to_raw.setdefault(norm, raw)

        category_norm_to_raw: dict[str, str] = {}
        for item in row.get("top_categories", [])[:10]:
            raw = normalize_space(item.get("value", ""))
            norm = normalize_category(raw)
            if norm:
                category_norm_to_raw.setdefault(norm, raw)

        contact_values_by_type: dict[str, set[str]] = {contact_type: set() for contact_type in CONTACT_TYPE_ORDER}
        for contact_type in CONTACT_TYPE_ORDER:
            for item in row.get("contact_signals", {}).get(contact_type, []):
                value = normalize_contact_value(contact_type, item.get("value", ""), stopwords)
                if value:
                    contact_values_by_type[contact_type].add(value)

        style_stats = row.get("style_stats", {}) or {}
        style_vector = [
            float(style_stats.get("digit_ratio_mean", 0.0) or 0.0),
            float(style_stats.get("punct_ratio_mean", 0.0) or 0.0),
            float(style_stats.get("uppercase_ratio_mean", 0.0) or 0.0),
            float(style_stats.get("repeated_title_share", 0.0) or 0.0),
            float(style_stats.get("repeated_description_share", 0.0) or 0.0),
            float(style_stats.get("max_category_share", 0.0) or 0.0),
        ]

        alias_normalized = normalize_space(row.get("alias_normalized", "")).lower()
        profiles.append(
            SellerProfile(
                seller_uid=row["seller_uid"],
                data_bucket=data_bucket,
                language=language,
                source_market_raw=row.get("source_market_raw", ""),
                source_seller_raw=row.get("source_seller_raw", ""),
                alias_normalized=alias_normalized,
                item_count=int(row.get("item_count", 0) or 0),
                title_norm_to_raw=title_norm_to_raw,
                desc_norm_to_raw=desc_norm_to_raw,
                category_norm_to_raw=category_norm_to_raw,
                contact_values_by_type=contact_values_by_type,
                category_norms=set(category_norm_to_raw.keys()),
                median_price=safe_median(row.get("price_numeric_approx_stats", {}) or {}),
                style_vector=style_vector,
                preview=build_preview(row),
                pgp_fingerprints=set(pgp_alias_map.get(alias_normalized, set())),
                retrieval_terms=build_retrieval_terms(row, language),
            )
        )
    return profiles


def build_duplicate_index(profiles: list[SellerProfile], getter) -> dict[str, list[int]]:
    index: dict[str, list[int]] = defaultdict(list)
    for seller_idx, profile in enumerate(profiles):
        for value in getter(profile):
            index[value].append(seller_idx)
    return index


def build_contact_index(profiles: list[SellerProfile]) -> dict[tuple[str, str], list[int]]:
    index: dict[tuple[str, str], list[int]] = defaultdict(list)
    for seller_idx, profile in enumerate(profiles):
        for contact_type, values in profile.contact_values_by_type.items():
            for value in values:
                index[(contact_type, value)].append(seller_idx)
    return index


def compute_retrieval_weights(profiles: list[SellerProfile], config: dict) -> None:
    df_counter: Counter = Counter()
    for profile in profiles:
        df_counter.update(profile.retrieval_terms.keys())

    profile_count = max(len(profiles), 1)
    max_df = min(int(math.ceil(profile_count * float(config["max_df_ratio"]))), int(config["max_df_cap"]))
    min_df = int(config["min_df"])

    for profile in profiles:
        weights: dict[str, float] = {}
        for term, tf_value in profile.retrieval_terms.items():
            df_value = df_counter.get(term, 0)
            if df_value < min_df or df_value > max_df:
                continue
            idf = math.log((profile_count + 1) / (df_value + 1)) + 1.0
            weights[term] = (1.0 + math.log1p(float(tf_value))) * idf

        top_terms = sorted(weights.items(), key=lambda item: (-item[1], item[0]))[: int(config["top_terms_per_profile"])]
        profile.retrieval_weights = dict(top_terms)
        profile.retrieval_norm = math.sqrt(sum(weight * weight for weight in profile.retrieval_weights.values()))


def lexical_neighbor_pairs(profiles: list[SellerProfile], config: dict) -> dict[tuple[int, int], float]:
    inverted: dict[str, list[tuple[int, float]]] = defaultdict(list)
    for idx, profile in enumerate(profiles):
        for term, weight in profile.retrieval_weights.items():
            inverted[term].append((idx, weight))

    min_cosine = float(config["min_cosine_similarity"])
    top_neighbors = int(config["top_neighbors_per_profile"])
    pair_scores: dict[tuple[int, int], float] = {}

    for idx, profile in enumerate(profiles):
        if not profile.retrieval_weights or not profile.retrieval_norm:
            continue
        candidate_dots: dict[int, float] = defaultdict(float)
        for term, weight in profile.retrieval_weights.items():
            for other_idx, other_weight in inverted.get(term, []):
                if other_idx <= idx:
                    continue
                candidate_dots[other_idx] += weight * other_weight

        scored = []
        for other_idx, dot in candidate_dots.items():
            other_norm = profiles[other_idx].retrieval_norm
            if not other_norm:
                continue
            cosine = dot / (profile.retrieval_norm * other_norm)
            if cosine >= min_cosine:
                scored.append((other_idx, cosine))

        for other_idx, cosine in sorted(scored, key=lambda item: (-item[1], profiles[item[0]].seller_uid))[:top_neighbors]:
            pair_scores[(idx, other_idx)] = max(pair_scores.get((idx, other_idx), 0.0), cosine)

    return pair_scores


def style_distance(profile_left: SellerProfile, profile_right: SellerProfile) -> float:
    return sum(abs(left - right) for left, right in zip(profile_left.style_vector, profile_right.style_vector))


def structural_support(profile_left: SellerProfile, profile_right: SellerProfile) -> tuple[float, float | None, float | None, float]:
    category_overlap = jaccard(profile_left.category_norms, profile_right.category_norms)
    item_ratio = ratio_or_zero(float(profile_left.item_count), float(profile_right.item_count))
    price_ratio = ratio_or_zero(profile_left.median_price, profile_right.median_price)
    style_l1 = style_distance(profile_left, profile_right)
    style_similarity = max(0.0, 1.0 - min(style_l1 / 2.0, 1.0))

    parts = [category_overlap, item_ratio or 0.0, style_similarity]
    if price_ratio is not None:
        parts.append(price_ratio)
    score = sum(parts) / len(parts)
    return round(score, 6), item_ratio, price_ratio, round(style_l1, 6)


def add_rule_hit(store: dict, rule_name: str, value: str) -> None:
    bucket = store.setdefault(rule_name, [])
    if value and value not in bucket:
        bucket.append(value)


def candidate_scope(alias_left: str, alias_right: str) -> str:
    if alias_left and alias_right and alias_left == alias_right:
        return "same_alias_identity_continuity"
    return "sockpuppet_primary"


def alias_relation(alias_left: str, alias_right: str) -> str:
    if alias_left and alias_right and alias_left == alias_right:
        return "same_alias"
    if not alias_left or not alias_right:
        return "alias_missing"
    return "different_alias"


def pair_uid(profile_left: SellerProfile, profile_right: SellerProfile) -> str:
    ordered = sorted([profile_left.seller_uid, profile_right.seller_uid])
    return ordered[0] + "||" + ordered[1]


def build_candidates_for_pool(
    profiles: list[SellerProfile],
    pool_config: dict,
    language: str,
    duplicate_limits: dict,
) -> list[dict]:
    pair_evidence: dict[tuple[int, int], dict[str, list[str] | float]] = {}

    def ensure_pair(left_idx: int, right_idx: int) -> dict:
        ordered = (left_idx, right_idx) if left_idx < right_idx else (right_idx, left_idx)
        return pair_evidence.setdefault(ordered, {})

    title_index = build_duplicate_index(profiles, lambda profile: profile.title_norm_to_raw.keys())
    desc_index = build_duplicate_index(profiles, lambda profile: profile.desc_norm_to_raw.keys())
    contact_index = build_contact_index(profiles)
    fp_index = build_duplicate_index(profiles, lambda profile: profile.pgp_fingerprints)

    for norm_value, seller_indexes in title_index.items():
        seller_indexes = sorted(set(seller_indexes))
        if len(seller_indexes) < 2 or len(seller_indexes) > int(duplicate_limits["shared_title_clone"]):
            continue
        for left_idx, right_idx in combinations(seller_indexes, 2):
            evidence = ensure_pair(left_idx, right_idx)
            raw_value = profiles[left_idx].title_norm_to_raw.get(norm_value) or profiles[right_idx].title_norm_to_raw.get(norm_value) or norm_value
            add_rule_hit(evidence, "shared_title_clone", raw_value)

    for norm_value, seller_indexes in desc_index.items():
        seller_indexes = sorted(set(seller_indexes))
        if len(seller_indexes) < 2 or len(seller_indexes) > int(duplicate_limits["shared_description_clone"]):
            continue
        for left_idx, right_idx in combinations(seller_indexes, 2):
            evidence = ensure_pair(left_idx, right_idx)
            raw_value = profiles[left_idx].desc_norm_to_raw.get(norm_value) or profiles[right_idx].desc_norm_to_raw.get(norm_value) or norm_value
            add_rule_hit(evidence, "shared_description_clone", raw_value)

    for (contact_type, value), seller_indexes in contact_index.items():
        seller_indexes = sorted(set(seller_indexes))
        if len(seller_indexes) < 2 or len(seller_indexes) > int(duplicate_limits["shared_contact_exact"]):
            continue
        for left_idx, right_idx in combinations(seller_indexes, 2):
            evidence = ensure_pair(left_idx, right_idx)
            add_rule_hit(evidence, "shared_contact_exact", f"{contact_type}:{value}")

    for fingerprint, seller_indexes in fp_index.items():
        seller_indexes = sorted(set(seller_indexes))
        if len(seller_indexes) < 2 or len(seller_indexes) > int(duplicate_limits["shared_pgp_fingerprint_via_aux_alias"]):
            continue
        for left_idx, right_idx in combinations(seller_indexes, 2):
            evidence = ensure_pair(left_idx, right_idx)
            add_rule_hit(evidence, "shared_pgp_fingerprint_via_aux_alias", fingerprint)

    lexical_pairs = lexical_neighbor_pairs(profiles, pool_config)
    strong_cosine = float(pool_config["strong_cosine_similarity"])
    structural_threshold = 0.55 if language == "en" else 0.5
    for (left_idx, right_idx), cosine in lexical_pairs.items():
        left_profile = profiles[left_idx]
        right_profile = profiles[right_idx]
        support_score, _, _, _ = structural_support(left_profile, right_profile)
        has_precise_rule = (left_idx, right_idx) in pair_evidence
        if cosine >= strong_cosine or support_score >= structural_threshold or has_precise_rule:
            evidence = ensure_pair(left_idx, right_idx)
            evidence["profile_lexical_neighbor"] = [f"{cosine:.6f}"]

    rows: list[dict] = []
    for (left_idx, right_idx), evidence in pair_evidence.items():
        left_profile = profiles[left_idx]
        right_profile = profiles[right_idx]
        support_score, item_ratio, price_ratio, style_l1 = structural_support(left_profile, right_profile)

        if support_score >= (0.6 if language == "en" else 0.55):
            add_rule_hit(evidence, "structural_support", f"{support_score:.6f}")

        shared_contacts = evidence.get("shared_contact_exact", [])
        shared_titles = evidence.get("shared_title_clone", [])
        shared_descs = evidence.get("shared_description_clone", [])
        shared_fps = evidence.get("shared_pgp_fingerprint_via_aux_alias", [])
        lexical_similarity = 0.0
        if evidence.get("profile_lexical_neighbor"):
            lexical_similarity = max(float(value) for value in evidence["profile_lexical_neighbor"])

        scope = candidate_scope(left_profile.alias_normalized, right_profile.alias_normalized)
        alias_state = alias_relation(left_profile.alias_normalized, right_profile.alias_normalized)
        shared_categories = sorted(
            left_profile.category_norms & right_profile.category_norms,
            key=lambda norm: left_profile.category_norm_to_raw.get(norm) or right_profile.category_norm_to_raw.get(norm) or norm,
        )
        shared_category_values = [
            left_profile.category_norm_to_raw.get(norm) or right_profile.category_norm_to_raw.get(norm) or norm
            for norm in shared_categories[:5]
        ]

        rule_hits = sorted(evidence.keys())
        rank_score = 0.0
        rank_score += 6.0 * (1 if shared_contacts else 0)
        rank_score += 5.0 * (1 if shared_descs else 0)
        rank_score += 4.5 * (1 if shared_fps else 0)
        rank_score += min(len(shared_titles), 3) * 2.0
        rank_score += lexical_similarity * 6.0
        rank_score += support_score * 2.0
        if scope == "same_alias_identity_continuity":
            rank_score -= 0.5

        if shared_contacts or shared_descs or shared_fps:
            review_priority = "high"
        elif shared_titles and (lexical_similarity >= 0.16 or support_score >= 0.6):
            review_priority = "high"
        elif lexical_similarity >= 0.22 or (shared_titles and support_score >= 0.5):
            review_priority = "medium"
        else:
            review_priority = "low"

        rows.append(
            {
                "pair_uid": pair_uid(left_profile, right_profile),
                "candidate_language": language,
                "data_bucket": left_profile.data_bucket,
                "candidate_scope": scope,
                "seller_uid_left": left_profile.seller_uid,
                "seller_uid_right": right_profile.seller_uid,
                "source_market_raw_left": left_profile.source_market_raw,
                "source_market_raw_right": right_profile.source_market_raw,
                "source_seller_raw_left": left_profile.source_seller_raw,
                "source_seller_raw_right": right_profile.source_seller_raw,
                "alias_normalized_left": left_profile.alias_normalized,
                "alias_normalized_right": right_profile.alias_normalized,
                "alias_relation": alias_state,
                "same_market_raw": str(left_profile.source_market_raw == right_profile.source_market_raw).lower(),
                "item_count_left": left_profile.item_count,
                "item_count_right": right_profile.item_count,
                "shared_contact_count": len(shared_contacts),
                "shared_contact_types": "|".join(sorted({value.split(":", 1)[0] for value in shared_contacts})),
                "shared_contact_values": " || ".join(shared_contacts[:5]),
                "shared_title_count": len(shared_titles),
                "shared_title_values": " || ".join(shared_titles[:5]),
                "shared_description_count": len(shared_descs),
                "shared_description_values": " || ".join(capped_text(value, 180) for value in shared_descs[:3]),
                "shared_category_count": len(shared_category_values),
                "shared_category_values": " || ".join(shared_category_values),
                "lexical_similarity": round(lexical_similarity, 6),
                "structural_support_score": round(support_score, 6),
                "item_count_ratio": "" if item_ratio is None else round(item_ratio, 6),
                "price_median_ratio": "" if price_ratio is None else round(price_ratio, 6),
                "style_distance_l1": round(style_l1, 6),
                "shared_pgp_fingerprint_count": len(shared_fps),
                "shared_pgp_fingerprint_values": " || ".join(shared_fps[:5]),
                "pgp_alias_hit_count_left": len(left_profile.pgp_fingerprints),
                "pgp_alias_hit_count_right": len(right_profile.pgp_fingerprints),
                "candidate_rule_hits": "|".join(rule_hits),
                "candidate_rule_count": len(rule_hits),
                "candidate_rank_score": round(rank_score, 6),
                "review_priority": review_priority,
                "left_preview": left_profile.preview,
                "right_preview": right_profile.preview,
                "review_status": "pending",
                "review_label": "",
                "reviewer_id": "",
                "review_notes": "",
            }
        )

    rows.sort(
        key=lambda row: (
            CANDIDATE_SCOPE_ORDER.get(row["candidate_scope"], 9),
            REVIEW_PRIORITY_ORDER.get(row["review_priority"], 9),
            -float(row["candidate_rank_score"]),
            -int(row["candidate_rule_count"]),
            row["pair_uid"],
        )
    )
    return rows


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def build_review_queue(rows: list[dict]) -> list[dict]:
    queue_rows = []
    for idx, row in enumerate(rows, start=1):
        queue_rows.append(
            {
                "review_rank": idx,
                "pair_uid": row["pair_uid"],
                "candidate_scope": row["candidate_scope"],
                "review_priority": row["review_priority"],
                "candidate_rule_hits": row["candidate_rule_hits"],
                "candidate_rank_score": row["candidate_rank_score"],
                "alias_relation": row["alias_relation"],
                "same_market_raw": row["same_market_raw"],
                "source_market_raw_left": row["source_market_raw_left"],
                "source_market_raw_right": row["source_market_raw_right"],
                "source_seller_raw_left": row["source_seller_raw_left"],
                "source_seller_raw_right": row["source_seller_raw_right"],
                "shared_contact_values": row["shared_contact_values"],
                "shared_title_values": row["shared_title_values"],
                "shared_description_values": row["shared_description_values"],
                "shared_category_values": row["shared_category_values"],
                "shared_pgp_fingerprint_values": row["shared_pgp_fingerprint_values"],
                "lexical_similarity": row["lexical_similarity"],
                "structural_support_score": row["structural_support_score"],
                "left_preview": row["left_preview"],
                "right_preview": row["right_preview"],
                "review_status": row["review_status"],
                "review_label": row["review_label"],
                "reviewer_id": row["reviewer_id"],
                "review_notes": row["review_notes"],
            }
        )
    return queue_rows


def summarize_pool(rows: list[dict], seller_count: int) -> dict:
    scope_counts = Counter(row["candidate_scope"] for row in rows)
    priority_counts = Counter(row["review_priority"] for row in rows)
    alias_relation_counts = Counter(row["alias_relation"] for row in rows)
    rule_counts = Counter()
    for row in rows:
        for rule_name in filter(None, row["candidate_rule_hits"].split("|")):
            rule_counts[rule_name] += 1
    return {
        "seller_count": seller_count,
        "candidate_pair_count": len(rows),
        "candidate_scope_counts": dict(scope_counts),
        "review_priority_counts": dict(priority_counts),
        "alias_relation_counts": dict(alias_relation_counts),
        "rule_pair_counts": dict(rule_counts),
        "top_ranked_pairs": [
            {
                "pair_uid": row["pair_uid"],
                "candidate_scope": row["candidate_scope"],
                "review_priority": row["review_priority"],
                "candidate_rule_hits": row["candidate_rule_hits"],
                "candidate_rank_score": row["candidate_rank_score"],
            }
            for row in rows[:10]
        ],
    }


def main() -> None:
    schema = load_json(SCHEMA_PATH)
    retrieval_policy = schema["retrieval_policy"]
    filtering_policy = schema["filtering_policy"]
    min_config = filtering_policy["content_minimums"]
    duplicate_limits = filtering_policy["duplicate_cluster_limits"]
    stopwords = {value.lower() for value in filtering_policy["contact_noise_stopwords"]}

    pgp_alias_map = load_pgp_alias_map()

    all_rows_by_pool: dict[str, list[dict]] = {}
    summary = {
        "schema_path": str(SCHEMA_PATH.relative_to(ROOT)),
        "input_dependencies": schema["input_dependencies"],
        "output_files": {
            "candidate_tables": {pool: str(path.relative_to(ROOT)) for pool, path in PAIR_OUTPUTS.items()},
            "manual_review_queues": {pool: str(path.relative_to(ROOT)) for pool, path in QUEUE_OUTPUTS.items()},
        },
        "pool_summaries": {},
        "acceptance_checks": {},
        "residual_risks": [
            "Step 4 candidates remain review candidates only and must not be promoted to final labels without human adjudication.",
            "Lexical-neighbor recall is sparse and surface-form based; it is designed for candidate generation, not final pair classification.",
            "Contact evidence is filtered more strictly than Step 3 to improve precision and may under-recall weak or obfuscated handles."
        ],
    }

    fieldnames = schema["candidate_output_fields"]
    fallback_queue_fields = [
        "review_rank",
        "pair_uid",
        "candidate_scope",
        "review_priority",
        "candidate_rule_hits",
        "candidate_rank_score",
        "alias_relation",
        "same_market_raw",
        "source_market_raw_left",
        "source_market_raw_right",
        "source_seller_raw_left",
        "source_seller_raw_right",
        "shared_contact_values",
        "shared_title_values",
        "shared_description_values",
        "shared_category_values",
        "shared_pgp_fingerprint_values",
        "lexical_similarity",
        "structural_support_score",
        "left_preview",
        "right_preview",
        "review_status",
        "review_label",
        "reviewer_id",
        "review_notes",
    ]

    for pool in ("en_content_train_pool", "zh_target_strict", "zh_target_aux"):
        rows = load_jsonl(PROFILE_PATHS[pool])
        language = "en" if pool == "en_content_train_pool" else "zh"
        profiles = build_seller_profiles(
            rows=rows,
            data_bucket=pool,
            language=language,
            stopwords=stopwords,
            min_config=min_config,
            pgp_alias_map=pgp_alias_map,
        )
        compute_retrieval_weights(profiles, retrieval_policy[pool])
        candidate_rows = build_candidates_for_pool(
            profiles=profiles,
            pool_config=retrieval_policy[pool],
            language=language,
            duplicate_limits=duplicate_limits,
        )
        write_csv(PAIR_OUTPUTS[pool], candidate_rows, fieldnames)

        review_rows = build_review_queue(candidate_rows)
        queue_fields = list(review_rows[0].keys()) if review_rows else fallback_queue_fields
        write_csv(QUEUE_OUTPUTS[pool], review_rows, queue_fields)

        all_rows_by_pool[pool] = candidate_rows
        summary["pool_summaries"][pool] = summarize_pool(candidate_rows, len(profiles))

    summary["acceptance_checks"] = {
        "all_candidate_rows_have_rule_hits": all(
            bool(row["candidate_rule_hits"])
            for rows in all_rows_by_pool.values()
            for row in rows
        ),
        "all_candidate_rows_have_pending_review_state": all(
            row["review_status"] == "pending" and row["review_label"] == ""
            for rows in all_rows_by_pool.values()
            for row in rows
        ),
        "no_candidate_auto_labeled": all(
            row["review_label"] == ""
            for rows in all_rows_by_pool.values()
            for row in rows
        ),
        "zh_target_aux_kept_separate": True,
    }

    with SUMMARY_PATH.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
