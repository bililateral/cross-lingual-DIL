#!/usr/bin/env python3
"""Build Step15-v6 pair features with train-only corpus reference statistics.

The canonical Step7 pair files contain semantic scores that are expensive to
recompute and corpus-relative fields historically calculated over a complete
language pool.  This builder preserves every canonical semantic/pair value,
recomputes only corpus-relative IDF/boilerplate/percentile fields from sellers
appearing in the frozen training split, and writes an isolated v6 feature set.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from bisect import bisect_right
from collections import Counter, defaultdict
from pathlib import Path

import step7_build_pair_feature_preview as preview


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_POLICY = ROOT / "schema" / "step15_v6_paper_hardening_policy.json"
REFERENCE_FIELDS = [
    "boilerplate_ratio_max",
    "boilerplate_ratio_gap_abs",
    "shared_title_idf_sum",
    "shared_description_idf_sum",
    "shared_title_idf_mean",
    "shared_description_idf_mean",
    "shared_boilerplate_count",
    "shared_low_df_sentence_count",
    "shared_rare_ngram_count",
    "item_count_percentile_gap_abs",
    "price_median_percentile_gap_abs",
    "title_length_median_percentile_gap_abs",
    "description_length_median_percentile_gap_abs",
    "digit_ratio_mean_percentile_gap_abs",
    "punct_ratio_mean_percentile_gap_abs",
    "repeated_title_share_percentile_gap_abs",
    "repeated_description_share_percentile_gap_abs",
    "max_category_share_percentile_gap_abs",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", default=str(DEFAULT_POLICY))
    parser.add_argument("--validate-config-only", action="store_true")
    parser.add_argument(
        "--validate-data-only",
        action="store_true",
        help="Build and hash every row in memory, validate lineage, and exit without writing outputs.",
    )
    parser.add_argument(
        "--allow-identical-replay",
        action="store_true",
        help="Permit a no-op rerun only when every existing output byte hash is identical.",
    )
    return parser.parse_args()


def resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def load_csv(path: Path) -> tuple[list[dict], list[str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader), list(reader.fieldnames or [])


def load_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_hash(value: object) -> str:
    payload = json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def selected_column_hash(rows: list[dict], fields: list[str]) -> str:
    return canonical_hash(
        [
            [row["pair_uid"], *[str(row.get(field, "")) for field in fields]]
            for row in sorted(rows, key=lambda item: str(item["pair_uid"]))
        ]
    )


def numeric_value(profile: dict, dotted_path: str) -> float | None:
    value: object = profile
    for part in dotted_path.split("."):
        if not isinstance(value, dict):
            return None
        value = value.get(part)
    try:
        return None if value in {None, ""} else float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def signature_norms(profile: dict, key: str, minimum_length: int) -> set[str]:
    values = set()
    for item in profile.get(key, []) or []:
        norm = preview.normalize_signature_value(item.get("value", ""))
        if norm and len(norm) >= minimum_length:
            values.add(norm)
    return values


def percentile(sorted_values: list[float], value: float | None) -> float | None:
    if value is None or not sorted_values:
        return None
    if len(sorted_values) == 1:
        return 0.5
    position = bisect_right(sorted_values, value) - 1
    position = max(0, min(position, len(sorted_values) - 1))
    return position / (len(sorted_values) - 1)


def fit_reference(
    profiles: dict[str, dict],
    train_sellers: set[str],
    numeric_paths: dict[str, str],
    boilerplate_cfg: dict,
) -> dict:
    missing = sorted(train_sellers - set(profiles))
    if missing:
        raise ValueError(f"Training sellers missing from Step3 profiles; first={missing[0]}")
    if not train_sellers:
        raise ValueError("Cannot fit a v6 corpus reference without training sellers")
    minimum_length = int(boilerplate_cfg.get("minimum_signature_value_length", 1))
    title_df: Counter[str] = Counter()
    description_df: Counter[str] = Counter()
    market_values: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    domain_values: dict[str, list[float]] = defaultdict(list)
    for seller_uid in sorted(train_sellers):
        profile = profiles[seller_uid]
        title_df.update(signature_norms(profile, "signature_titles", minimum_length))
        description_df.update(signature_norms(profile, "signature_description_segments", minimum_length))
        market = str(profile.get("source_market_raw", ""))
        for feature_name, dotted_path in numeric_paths.items():
            value = numeric_value(profile, dotted_path)
            if value is not None:
                market_values[market][feature_name].append(value)
                domain_values[feature_name].append(value)
    for feature_map in market_values.values():
        for values in feature_map.values():
            values.sort()
    for values in domain_values.values():
        values.sort()
    return {
        "reference_scope": "frozen_train_sellers_only",
        "train_seller_count": len(train_sellers),
        "train_seller_uid_sha256": canonical_hash(sorted(train_sellers)),
        "title_df": dict(sorted(title_df.items())),
        "description_df": dict(sorted(description_df.items())),
        "market_numeric_values": {
            market: {name: values for name, values in sorted(feature_map.items())}
            for market, feature_map in sorted(market_values.items())
        },
        "domain_numeric_values": dict(sorted(domain_values.items())),
        "boilerplate_config": boilerplate_cfg,
    }


def reference_signature_state(
    profile: dict,
    reference: dict,
    signature_key: str,
    df_key: str,
) -> dict:
    cfg = reference["boilerplate_config"]
    minimum_length = int(cfg.get("minimum_signature_value_length", 1))
    norms = signature_norms(profile, signature_key, minimum_length)
    seller_count = int(reference["train_seller_count"])
    df_map = reference[df_key]
    boilerplate_threshold = float(cfg.get("boilerplate_seller_share_threshold", 0.01))
    rare_threshold = float(cfg.get("rare_seller_share_threshold", 0.002))
    idf = {}
    boilerplate = set()
    rare = set()
    for norm in norms:
        df = int(df_map.get(norm, 0))
        share = df / max(seller_count, 1)
        idf[norm] = math.log((seller_count + 1) / (df + 1)) + 1.0
        if share >= boilerplate_threshold:
            boilerplate.add(norm)
        if share <= rare_threshold:
            rare.add(norm)
    return {"norms": norms, "idf": idf, "boilerplate": boilerplate, "rare": rare}


def derive_reference_fields(left: dict, right: dict, reference: dict, numeric_paths: dict[str, str]) -> dict:
    lt = reference_signature_state(left, reference, "signature_titles", "title_df")
    rt = reference_signature_state(right, reference, "signature_titles", "title_df")
    ld = reference_signature_state(left, reference, "signature_description_segments", "description_df")
    rd = reference_signature_state(right, reference, "signature_description_segments", "description_df")
    shared_title = lt["norms"] & rt["norms"]
    shared_description = ld["norms"] & rd["norms"]
    title_idf = [max(lt["idf"][norm], rt["idf"][norm]) for norm in sorted(shared_title)]
    description_idf = [max(ld["idf"][norm], rd["idf"][norm]) for norm in sorted(shared_description)]
    shared_boilerplate = {
        norm for norm in shared_title if norm in lt["boilerplate"] or norm in rt["boilerplate"]
    } | {
        norm for norm in shared_description if norm in ld["boilerplate"] or norm in rd["boilerplate"]
    }
    left_norm_count = len(lt["norms"]) + len(ld["norms"])
    right_norm_count = len(rt["norms"]) + len(rd["norms"])
    left_boilerplate_count = len(lt["boilerplate"]) + len(ld["boilerplate"])
    right_boilerplate_count = len(rt["boilerplate"]) + len(rd["boilerplate"])
    left_ratio = left_boilerplate_count / max(left_norm_count, 1) if left_norm_count else 0.0
    right_ratio = right_boilerplate_count / max(right_norm_count, 1) if right_norm_count else 0.0
    result = {
        "boilerplate_ratio_max": max(left_ratio, right_ratio),
        "boilerplate_ratio_gap_abs": abs(left_ratio - right_ratio),
        "shared_title_idf_sum": sum(title_idf),
        "shared_description_idf_sum": sum(description_idf),
        "shared_title_idf_mean": sum(title_idf) / len(title_idf) if title_idf else 0.0,
        "shared_description_idf_mean": sum(description_idf) / len(description_idf) if description_idf else 0.0,
        "shared_boilerplate_count": len(shared_boilerplate),
        "shared_low_df_sentence_count": len(
            {norm for norm in shared_description if norm in ld["rare"] or norm in rd["rare"]}
        ),
        "shared_rare_ngram_count": len(
            {norm for norm in shared_title if norm in lt["rare"] or norm in rt["rare"]}
        ),
    }
    minimum_market_size = int(reference.get("minimum_market_group_size", preview.MIN_MARKET_GROUP_SIZE))
    market_values = reference["market_numeric_values"]
    domain_values = reference["domain_numeric_values"]
    for feature_name, dotted_path in numeric_paths.items():
        left_market = str(left.get("source_market_raw", ""))
        right_market = str(right.get("source_market_raw", ""))
        left_reference = market_values.get(left_market, {}).get(feature_name, [])
        right_reference = market_values.get(right_market, {}).get(feature_name, [])
        if len(left_reference) < minimum_market_size:
            left_reference = domain_values.get(feature_name, [])
        if len(right_reference) < minimum_market_size:
            right_reference = domain_values.get(feature_name, [])
        left_pct = percentile(left_reference, numeric_value(left, dotted_path))
        right_pct = percentile(right_reference, numeric_value(right, dotted_path))
        result[f"{feature_name}_percentile_gap_abs"] = (
            "" if left_pct is None or right_pct is None else abs(left_pct - right_pct)
        )
    return {
        key: (round(float(value), 6) if value != "" else "")
        for key, value in result.items()
    }


def render_csv(rows: list[dict], fields: list[str]) -> bytes:
    import io

    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return ("\ufeff" + buffer.getvalue()).encode("utf-8")


def write_fail_closed(path: Path, payload: bytes, allow_identical_replay: bool) -> str:
    expected_hash = hashlib.sha256(payload).hexdigest()
    if path.exists():
        observed_hash = sha256(path)
        if allow_identical_replay and observed_hash == expected_hash:
            return "identical_replay_noop"
        raise FileExistsError(
            f"Refusing to overwrite v6 feature output: {path}; "
            "use a new versioned path, or --allow-identical-replay for byte-identical output"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_bytes(payload)
    temporary.replace(path)
    return "created"


def main() -> None:
    args = parse_args()
    policy_path = resolve(args.policy)
    policy = json.loads(policy_path.read_text(encoding="utf-8"))
    cfg = policy.get("inductive_feature_lineage", {})
    domains = cfg.get("domains", {})
    if not cfg.get("enabled") or set(domains) != {"en_content_train_pool", "zh_target_strict"}:
        raise ValueError("Step15-v6 inductive feature lineage must define exactly EN and ZH domains")
    if cfg.get("reference_scope") != "frozen_train_sellers_only":
        raise ValueError("Step15-v6 corpus reference scope is not train-only")
    if args.validate_config_only:
        print(json.dumps({"status": "pass", "policy": str(policy_path.relative_to(ROOT)), "domains": sorted(domains)}, indent=2))
        return

    schema_path = resolve(cfg["step7_feature_schema"])
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    numeric_paths = dict(schema["market_relative_numeric_fields"])
    boilerplate_cfg = dict(schema["boilerplate_feature_config"])
    semantic_fields = list(schema["feature_views"]["future_multilingual_semantics"])
    records = {}
    reference_payload = {
        "step": "step15_v6_inductive_pair_feature_reference",
        "version": cfg["version"],
        "reference_scope": cfg["reference_scope"],
        "labels_used_for_values": False,
        "split_membership_used_only_to_select_reference_sellers": True,
        "reference_fields": REFERENCE_FIELDS,
        "domains": {},
    }
    pending_outputs: list[tuple[Path, bytes, str]] = []
    for domain, domain_cfg in sorted(domains.items()):
        labels_path = resolve(domain_cfg["frozen_labels"])
        profiles_path = resolve(domain_cfg["seller_profiles"])
        features_path = resolve(domain_cfg["canonical_pair_features"])
        candidates_path = resolve(domain_cfg["step4_candidates"])
        output_path = resolve(domain_cfg["output_pair_features"])
        labels, _ = load_csv(labels_path)
        profiles_list = load_jsonl(profiles_path)
        feature_rows, feature_fields = load_csv(features_path)
        candidates, _ = load_csv(candidates_path)
        train_split = str(domain_cfg.get("train_split", "train"))
        train_rows = [row for row in labels if row.get("split_name") == train_split]
        train_sellers = {
            str(row[key])
            for row in train_rows
            for key in ("seller_uid_left", "seller_uid_right")
            if str(row.get(key, "")).strip()
        }
        profiles = {str(row["seller_uid"]): row for row in profiles_list}
        reference = fit_reference(profiles, train_sellers, numeric_paths, boilerplate_cfg)
        reference["minimum_market_group_size"] = preview.MIN_MARKET_GROUP_SIZE
        candidate_index = {str(row["pair_uid"]): row for row in candidates}
        if {str(row["pair_uid"]) for row in feature_rows} != set(candidate_index):
            raise ValueError(f"{domain}: canonical Step7 and Step4 pair universes differ")
        output_rows = []
        non_identifier_source_counts: Counter[str] = Counter()
        for row in feature_rows:
            pair_uid = str(row["pair_uid"])
            left_uid = str(row["seller_uid_left"])
            right_uid = str(row["seller_uid_right"])
            if left_uid not in profiles or right_uid not in profiles:
                raise ValueError(f"{domain}: missing profile for pair {pair_uid}")
            derived = derive_reference_fields(profiles[left_uid], profiles[right_uid], reference, numeric_paths)
            output = dict(row)
            output.update(derived)
            candidate = candidate_index[pair_uid]
            non_identifier = str(candidate.get("candidate_rule_count_non_identifier", "")).strip()
            if not non_identifier:
                excluded = {
                    "shared_contact_exact",
                    "shared_pgp_fingerprint",
                    "shared_pgp_fingerprint_via_aux_alias",
                }
                non_identifier = str(sum(
                    1 for rule in str(candidate.get("candidate_rule_hits", "")).split("|")
                    if rule and rule not in excluded
                ))
                non_identifier_source_counts["derived_from_frozen_candidate_rule_hits"] += 1
            else:
                non_identifier_source_counts["materialized_step4_field"] += 1
            output["candidate_rule_count_non_identifier"] = non_identifier
            output_rows.append(output)
        output_fields = list(feature_fields)
        if "candidate_rule_count_non_identifier" not in output_fields:
            insertion = output_fields.index("sparse_lexical_similarity_raw") if "sparse_lexical_similarity_raw" in output_fields else len(output_fields)
            output_fields.insert(insertion, "candidate_rule_count_non_identifier")
        output_payload = render_csv(output_rows, output_fields)
        semantic_hash_before = selected_column_hash(feature_rows, semantic_fields)
        semantic_hash_after = selected_column_hash(output_rows, semantic_fields)
        if semantic_hash_before != semantic_hash_after:
            raise ValueError(f"{domain}: semantic scores changed while building v6 features")
        pending_outputs.append((output_path, output_payload, domain))
        reference_payload["domains"][domain] = reference
        records[domain] = {
            "pair_count": len(output_rows),
            "train_pair_count": len(train_rows),
            "train_seller_count": len(train_sellers),
            "pair_uid_sha256": canonical_hash(sorted(str(row["pair_uid"]) for row in output_rows)),
            "canonical_semantic_values_preserved": True,
            "candidate_rule_count_non_identifier_source_counts": dict(non_identifier_source_counts),
            "semantic_sha256_before": semantic_hash_before,
            "semantic_sha256_after": semantic_hash_after,
            "output_path": str(output_path.relative_to(ROOT)),
            "output_sha256": hashlib.sha256(output_payload).hexdigest(),
            "inputs": {
                str(path.relative_to(ROOT)): sha256(path)
                for path in (labels_path, profiles_path, features_path, candidates_path)
            },
        }
    reference_payload["reference_sha256"] = canonical_hash(reference_payload)
    reference_path = resolve(cfg["reference_bundle_output"])
    reference_bytes = (json.dumps(reference_payload, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    manifest = {
        "step": "step15_build_v6_inductive_pair_features",
        "version": cfg["version"],
        "policy": str(policy_path.relative_to(ROOT)),
        "policy_sha256": sha256(policy_path),
        "producer": str(Path(__file__).resolve().relative_to(ROOT)),
        "producer_sha256": sha256(Path(__file__).resolve()),
        "reference_scope": cfg["reference_scope"],
        "transductive_valid_or_test_covariates_used_for_reference": False,
        "candidate_pair_universe_changed": False,
        "reference_bundle": str(reference_path.relative_to(ROOT)),
        "reference_bundle_sha256": hashlib.sha256(reference_bytes).hexdigest(),
        "domains": records,
    }
    manifest["manifest_sha256"] = canonical_hash(manifest)
    manifest_path = resolve(cfg["manifest_output"])
    manifest_bytes = (json.dumps(manifest, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    if args.validate_data_only:
        print(json.dumps({"status": "pass", "mode": "validate_data_only", "manifest": manifest}, indent=2))
        return
    actions = {}
    for output_path, payload, domain in pending_outputs:
        actions[domain] = write_fail_closed(output_path, payload, args.allow_identical_replay)
    actions["reference_bundle"] = write_fail_closed(reference_path, reference_bytes, args.allow_identical_replay)
    actions["manifest"] = write_fail_closed(manifest_path, manifest_bytes, args.allow_identical_replay)
    print(json.dumps({"manifest": str(manifest_path.relative_to(ROOT)), "actions": actions, "domains": records}, indent=2))


if __name__ == "__main__":
    main()
