from __future__ import annotations

import csv
import json
import math
import re
from bisect import bisect_right
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
SCHEMA_PATH = ROOT / "schema" / "step7_transfer_safe_pair_feature_schema.json"
PROFILE_PATHS = {
    "en_content_train_pool": ROOT / "reports" / "step3_seller_profiles.en_content_train_pool.jsonl",
    "zh_target_strict": ROOT / "reports" / "step3_seller_profiles.zh_target_strict.jsonl",
    "zh_target_aux": ROOT / "reports" / "step3_seller_profiles.zh_target_aux.jsonl",
}
PAIR_PATHS = {
    "en_content_train_pool": ROOT / "reports" / "step4_en_silver_candidate_pairs.csv",
    "zh_target_strict": ROOT / "reports" / "step4_zh_target_strict_silver_candidate_pairs.csv",
    "zh_target_aux": ROOT / "reports" / "step4_zh_target_aux_silver_candidate_pairs.csv",
}
OUTPUT_PATHS = {
    "en_content_train_pool": ROOT / "reports" / "step7_pair_feature_preview.en_content_train_pool.csv",
    "zh_target_strict": ROOT / "reports" / "step7_pair_feature_preview.zh_target_strict.csv",
    "zh_target_aux": ROOT / "reports" / "step7_pair_feature_preview.zh_target_aux.csv",
}
SUMMARY_PATH = ROOT / "reports" / "step7_pair_feature_preview_summary.json"
MIN_MARKET_GROUP_SIZE = 20
COUNT_CAP = 5
CJK_RE = re.compile(r"[\u3400-\u9fff]")


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def load_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def dotted_get(row: dict, dotted_path: str):
    value = row
    for part in dotted_path.split("."):
        if not isinstance(value, dict):
            return None
        value = value.get(part)
    return value


def to_float(value) -> float | None:
    if value in {"", None}:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def to_int(value) -> int:
    if value in {"", None}:
        return 0
    return int(value)


def to_bool_int(raw: str) -> int:
    return 1 if str(raw).strip().lower() == "true" else 0


def normalize_category(value: str) -> str:
    text = str(value or "").strip().casefold()
    if not text:
        return ""
    if CJK_RE.search(text):
        return re.sub(r"[^0-9a-z\u3400-\u9fff]+", "", text)
    return " ".join(re.findall(r"[a-z0-9]+", text))


def normalize_signature_value(value: str) -> str:
    text = str(value or "").strip().casefold()
    if not text:
        return ""
    if CJK_RE.search(text):
        return re.sub(r"[^0-9a-z\u3400-\u9fff]+", "", text)
    return " ".join(re.findall(r"[a-z0-9]+", text))


def signature_feature_state(items: list[dict], seller_count: int, cfg: dict) -> dict:
    boilerplate_threshold = float(cfg.get("boilerplate_seller_share_threshold", 0.01))
    rare_threshold = float(cfg.get("rare_seller_share_threshold", 0.002))
    minimum_length = int(cfg.get("minimum_signature_value_length", 1))
    idf_by_norm: dict[str, float] = {}
    boilerplate_norms: set[str] = set()
    rare_norms: set[str] = set()

    for item in items or []:
        value = str(item.get("value", "") or "").strip()
        norm = normalize_signature_value(value)
        if not norm or len(norm) < minimum_length:
            continue
        seller_df = max(int(item.get("seller_df", 1) or 1), 1)
        seller_share = seller_df / max(seller_count, 1)
        idf = math.log((seller_count + 1) / (seller_df + 1)) + 1.0
        previous = idf_by_norm.get(norm)
        if previous is None or idf > previous:
            idf_by_norm[norm] = round(idf, 6)
        if seller_share >= boilerplate_threshold:
            boilerplate_norms.add(norm)
        if seller_share <= rare_threshold:
            rare_norms.add(norm)

    norms = set(idf_by_norm.keys())
    return {
        "norms": norms,
        "idf_by_norm": idf_by_norm,
        "boilerplate_norms": boilerplate_norms,
        "rare_norms": rare_norms,
        "boilerplate_ratio": round(len(boilerplate_norms) / max(len(norms), 1), 6) if norms else 0.0,
    }


def jaccard(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    union = left | right
    if not union:
        return 0.0
    return round(len(left & right) / len(union), 6)


def percentile_rank(sorted_values: list[float], value: float) -> float | None:
    if not sorted_values:
        return None
    if len(sorted_values) == 1:
        return 0.5
    position = bisect_right(sorted_values, value) - 1
    position = max(0, min(position, len(sorted_values) - 1))
    return round(position / (len(sorted_values) - 1), 6)


def classify_review_stratum(row: dict) -> str:
    if row.get("candidate_scope") == "same_alias_identity_continuity":
        return "same_alias_continuity"

    has_identifier = to_int(row.get("shared_contact_count")) > 0 or to_int(row.get("shared_pgp_fingerprint_count")) > 0
    has_clone = to_int(row.get("shared_description_count")) > 0 or to_int(row.get("shared_title_count")) > 0
    has_semantic = to_float(row.get("lexical_similarity")) is not None and float(row["lexical_similarity"]) > 0
    has_structural = to_float(row.get("structural_support_score")) is not None and float(row["structural_support_score"]) >= 0.5

    if has_identifier and has_clone:
        return "identifier_plus_text"
    if has_clone:
        return "text_clone_primary"
    if has_identifier:
        return "identifier_primary"
    if has_semantic and has_structural:
        return "semantic_structural"
    return "semantic_only"


def prepare_profiles(rows: list[dict], numeric_paths: dict[str, str], en_only_paths: dict[str, str]) -> tuple[dict[str, dict], dict]:
    profile_index: dict[str, dict] = {}
    group_values: dict[tuple[str, str], dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    boilerplate_cfg = load_json(SCHEMA_PATH).get("boilerplate_feature_config", {})
    seller_count = len(rows)

    for row in rows:
        seller_uid = row["seller_uid"]
        market = row["source_market_raw"]
        bucket = row["data_bucket"]
        record = {
            "seller_uid": seller_uid,
            "data_bucket": bucket,
            "source_market_raw": market,
            "source_dataset": row["source_dataset"],
            "category_norms": {
                normalize_category(item.get("value", ""))
                for item in row.get("top_categories", [])
                if normalize_category(item.get("value", ""))
            },
            "title_signature_state": signature_feature_state(
                row.get("signature_titles", []),
                seller_count,
                boilerplate_cfg,
            ),
            "description_signature_state": signature_feature_state(
                row.get("signature_description_segments", []),
                seller_count,
                boilerplate_cfg,
            ),
            "numeric": {},
        }
        title_state = record["title_signature_state"]
        description_state = record["description_signature_state"]
        signature_item_count = len(title_state["norms"]) + len(description_state["norms"])
        boilerplate_item_count = len(title_state["boilerplate_norms"]) + len(description_state["boilerplate_norms"])
        record["boilerplate_ratio"] = round(boilerplate_item_count / max(signature_item_count, 1), 6) if signature_item_count else 0.0
        for feature_name, dotted_path in numeric_paths.items():
            value = to_float(dotted_get(row, dotted_path))
            record["numeric"][feature_name] = value
            if value is not None:
                group_values[(bucket, market)][feature_name].append(value)
                group_values[(bucket, "__bucket__")][feature_name].append(value)
        for feature_name, dotted_path in en_only_paths.items():
            value = to_float(dotted_get(row, dotted_path))
            record["numeric"][feature_name] = value
            if value is not None:
                group_values[(bucket, market)][feature_name].append(value)
                group_values[(bucket, "__bucket__")][feature_name].append(value)
        profile_index[seller_uid] = record

    prepared_groups: dict[tuple[str, str], dict[str, list[float]]] = {}
    for key, feature_map in group_values.items():
        prepared_groups[key] = {
            feature_name: sorted(values)
            for feature_name, values in feature_map.items()
            if values
        }
    return profile_index, prepared_groups


def percentile_for_profile(
    profile: dict,
    groups: dict[tuple[str, str], dict[str, list[float]]],
    feature_name: str,
) -> tuple[float | None, int]:
    value = profile["numeric"].get(feature_name)
    if value is None:
        return None, 0
    bucket = profile["data_bucket"]
    market = profile["source_market_raw"]
    market_values = groups.get((bucket, market), {}).get(feature_name, [])
    if len(market_values) >= MIN_MARKET_GROUP_SIZE:
        return percentile_rank(market_values, value), len(market_values)
    bucket_values = groups.get((bucket, "__bucket__"), {}).get(feature_name, [])
    return percentile_rank(bucket_values, value), len(bucket_values)


def build_pair_rows(
    candidate_rows: list[dict],
    profile_index: dict[str, dict],
    groups: dict[tuple[str, str], dict[str, list[float]]],
    schema: dict,
) -> list[dict]:
    output_rows = []
    numeric_features = list(schema["market_relative_numeric_fields"].keys())
    en_only_features = list(schema["en_only_auxiliary_fields"].keys())

    for row in candidate_rows:
        left = profile_index[row["seller_uid_left"]]
        right = profile_index[row["seller_uid_right"]]
        review_stratum = classify_review_stratum(row)
        shared_title_signature_norms = left["title_signature_state"]["norms"] & right["title_signature_state"]["norms"]
        shared_description_signature_norms = (
            left["description_signature_state"]["norms"] & right["description_signature_state"]["norms"]
        )
        shared_title_idf_values = [
            max(
                left["title_signature_state"]["idf_by_norm"].get(norm, 0.0),
                right["title_signature_state"]["idf_by_norm"].get(norm, 0.0),
            )
            for norm in sorted(shared_title_signature_norms)
        ]
        shared_description_idf_values = [
            max(
                left["description_signature_state"]["idf_by_norm"].get(norm, 0.0),
                right["description_signature_state"]["idf_by_norm"].get(norm, 0.0),
            )
            for norm in sorted(shared_description_signature_norms)
        ]
        shared_boilerplate_norms = {
            norm
            for norm in shared_title_signature_norms
            if norm in left["title_signature_state"]["boilerplate_norms"]
            or norm in right["title_signature_state"]["boilerplate_norms"]
        } | {
            norm
            for norm in shared_description_signature_norms
            if norm in left["description_signature_state"]["boilerplate_norms"]
            or norm in right["description_signature_state"]["boilerplate_norms"]
        }
        shared_low_df_sentence_count = len(
            {
                norm
                for norm in shared_description_signature_norms
                if norm in left["description_signature_state"]["rare_norms"]
                or norm in right["description_signature_state"]["rare_norms"]
            }
        )
        shared_rare_ngram_count = len(
            {
                norm
                for norm in shared_title_signature_norms
                if norm in left["title_signature_state"]["rare_norms"]
                or norm in right["title_signature_state"]["rare_norms"]
            }
        )
        output = {
            "pair_uid": row["pair_uid"],
            "data_bucket": row["data_bucket"],
            "candidate_language": row["candidate_language"],
            "candidate_scope": row["candidate_scope"],
            "review_stratum": review_stratum,
            "core_transfer_eligible": int(row["candidate_scope"] == "sockpuppet_primary"),
            "seller_uid_left": row["seller_uid_left"],
            "seller_uid_right": row["seller_uid_right"],
            "source_market_raw_left": row["source_market_raw_left"],
            "source_market_raw_right": row["source_market_raw_right"],
            "same_market_raw_bool": to_bool_int(row["same_market_raw"]),
            "same_source_dataset_bool": int(left["source_dataset"] == right["source_dataset"]),
            "review_priority": row["review_priority"],
            "review_status": row["review_status"],
            "review_label": row["review_label"],
            "profile_category_jaccard": jaccard(left["category_norms"], right["category_norms"]),
            "has_shared_title_clone": int(to_int(row["shared_title_count"]) > 0),
            "has_shared_description_clone": int(to_int(row["shared_description_count"]) > 0),
            "has_shared_contact_exact": int(to_int(row["shared_contact_count"]) > 0),
            "has_shared_pgp_fingerprint": int(to_int(row["shared_pgp_fingerprint_count"]) > 0),
            "shared_title_count_capped": min(to_int(row["shared_title_count"]), COUNT_CAP),
            "shared_description_count_capped": min(to_int(row["shared_description_count"]), COUNT_CAP),
            "shared_category_count_capped": min(to_int(row["shared_category_count"]), COUNT_CAP),
            "shared_contact_count_capped": min(to_int(row["shared_contact_count"]), COUNT_CAP),
            "shared_pgp_fingerprint_count_capped": min(to_int(row["shared_pgp_fingerprint_count"]), COUNT_CAP),
            "boilerplate_ratio_left": left["boilerplate_ratio"],
            "boilerplate_ratio_right": right["boilerplate_ratio"],
            "boilerplate_ratio_max": round(max(left["boilerplate_ratio"], right["boilerplate_ratio"]), 6),
            "boilerplate_ratio_gap_abs": round(abs(left["boilerplate_ratio"] - right["boilerplate_ratio"]), 6),
            "shared_title_idf_sum": round(sum(shared_title_idf_values), 6),
            "shared_description_idf_sum": round(sum(shared_description_idf_values), 6),
            "shared_title_idf_mean": round(sum(shared_title_idf_values) / max(len(shared_title_idf_values), 1), 6)
            if shared_title_idf_values
            else 0.0,
            "shared_description_idf_mean": round(
                sum(shared_description_idf_values) / max(len(shared_description_idf_values), 1), 6
            )
            if shared_description_idf_values
            else 0.0,
            "shared_boilerplate_count": len(shared_boilerplate_norms),
            "shared_low_df_sentence_count": shared_low_df_sentence_count,
            "shared_rare_ngram_count": shared_rare_ngram_count,
            "candidate_rule_count_raw": to_int(row["candidate_rule_count"]),
            "sparse_lexical_similarity_raw": round(to_float(row["lexical_similarity"]) or 0.0, 6),
            "structural_support_score_raw": round(to_float(row["structural_support_score"]) or 0.0, 6),
            "left_market_profile_size": len(groups.get((left["data_bucket"], left["source_market_raw"]), {}).get("item_count", [])),
            "right_market_profile_size": len(groups.get((right["data_bucket"], right["source_market_raw"]), {}).get("item_count", [])),
        }

        for feature_name in numeric_features + en_only_features:
            left_value = left["numeric"].get(feature_name)
            right_value = right["numeric"].get(feature_name)
            left_pct, left_n = percentile_for_profile(left, groups, feature_name)
            right_pct, right_n = percentile_for_profile(right, groups, feature_name)
            if feature_name not in en_only_features:
                output.setdefault("left_market_profile_size", left_n or output["left_market_profile_size"])
                output.setdefault("right_market_profile_size", right_n or output["right_market_profile_size"])
            gap = None
            if left_pct is not None and right_pct is not None:
                gap = round(abs(left_pct - right_pct), 6)
            output[f"{feature_name}_percentile_gap_abs"] = "" if gap is None else gap
            raw_gap = None
            if left_value is not None and right_value is not None:
                raw_gap = round(abs(left_value - right_value), 6)
            output[f"{feature_name}_raw_gap_abs"] = "" if raw_gap is None else raw_gap

        output_rows.append(output)

    return output_rows


def summarize_pool(rows: list[dict], output_fields: list[str]) -> dict:
    stratum_counts = Counter(row["review_stratum"] for row in rows)
    label_counts = Counter(row["review_label"] or "__blank__" for row in rows)
    percentile_missing_rates = {}
    raw_gap_missing_rates = {}
    numeric_gap_fields = [field for field in output_fields if field.endswith("_percentile_gap_abs")]
    for field in numeric_gap_fields:
        missing = sum(1 for row in rows if row[field] in {"", None})
        percentile_missing_rates[field] = round(missing / max(len(rows), 1), 6)
    raw_gap_fields = [field for field in output_fields if field.endswith("_raw_gap_abs")]
    for field in raw_gap_fields:
        missing = sum(1 for row in rows if row[field] in {"", None})
        raw_gap_missing_rates[field] = round(missing / max(len(rows), 1), 6)
    return {
        "row_count": len(rows),
        "review_stratum_counts": dict(stratum_counts),
        "label_state_counts": dict(label_counts),
        "core_transfer_eligible_count": sum(int(row["core_transfer_eligible"]) for row in rows),
        "missing_rate_by_percentile_gap_feature": percentile_missing_rates,
        "missing_rate_by_raw_gap_feature": raw_gap_missing_rates,
        "top_rows": [
            {
                "pair_uid": row["pair_uid"],
                "review_stratum": row["review_stratum"],
                "review_priority": row["review_priority"],
                "shared_description_count_capped": row["shared_description_count_capped"],
                "shared_boilerplate_count": row["shared_boilerplate_count"],
                "shared_description_idf_sum": row["shared_description_idf_sum"],
                "shared_contact_count_capped": row["shared_contact_count_capped"],
            }
            for row in rows[:10]
        ],
    }


def validate_schema_output_fields(schema: dict) -> None:
    numeric_features = list(schema["market_relative_numeric_fields"].keys())
    en_only_features = list(schema["en_only_auxiliary_fields"].keys())
    expected_fields = set()
    for feature_name in numeric_features + en_only_features:
        expected_fields.add(f"{feature_name}_percentile_gap_abs")
        expected_fields.add(f"{feature_name}_raw_gap_abs")
    output_fields = set(schema["pair_output_fields"])
    missing = sorted(expected_fields - output_fields)
    if missing:
        raise SystemExit(
            "step7_build_pair_feature_preview.py is out of sync with "
            "schema/step7_transfer_safe_pair_feature_schema.json. "
            "Missing pair_output_fields entries for: " + ", ".join(missing)
        )


def main() -> None:
    schema = load_json(SCHEMA_PATH)
    validate_schema_output_fields(schema)
    profile_maps = {}
    group_maps = {}

    for pool, path in PROFILE_PATHS.items():
        profiles = load_jsonl(path)
        profile_index, groups = prepare_profiles(
            profiles,
            schema["market_relative_numeric_fields"],
            schema["en_only_auxiliary_fields"],
        )
        profile_maps[pool] = profile_index
        group_maps[pool] = groups

    output_fields = schema["pair_output_fields"]
    summary = {
        "schema_path": str(SCHEMA_PATH.relative_to(ROOT)),
        "input_dependencies": schema["input_dependencies"],
        "output_files": {pool: str(path.relative_to(ROOT)) for pool, path in OUTPUT_PATHS.items()},
        "feature_views": schema["feature_views"],
        "pool_summaries": {},
        "acceptance_checks": {},
    }

    all_rows: list[dict] = []
    for pool, pair_path in PAIR_PATHS.items():
        candidate_rows = load_csv(pair_path)
        output_rows = build_pair_rows(candidate_rows, profile_maps[pool], group_maps[pool], schema)
        write_csv(OUTPUT_PATHS[pool], output_rows, output_fields)
        summary["pool_summaries"][pool] = summarize_pool(output_rows, output_fields)
        all_rows.extend(output_rows)

    summary["acceptance_checks"] = {
        "same_alias_not_marked_core_transfer": not any(
            row["core_transfer_eligible"] == 1 and row["candidate_scope"] == "same_alias_identity_continuity"
            for row in all_rows
        ),
        "all_rows_have_review_stratum": all(bool(row["review_stratum"]) for row in all_rows),
        "all_rows_have_pair_uid": all(bool(row["pair_uid"]) for row in all_rows),
    }

    with SUMMARY_PATH.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)

    print(f"Wrote {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
