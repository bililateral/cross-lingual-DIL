from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LABELS = ROOT / "reports" / "step5_zh_target_strict_frozen_silver_labels.csv"
EVIDENCE = ROOT / "reports" / "step15_evidence_type_labels.zh_target_strict.csv"
SIGNALS = ROOT / "reports" / "step3_item_identity_signals.zh_target_strict.csv"
OUTPUT_CSV = ROOT / "reports" / "step16f_valid_test_positive_reaudit.csv"
OUTPUT_JSON = ROOT / "reports" / "step16f_valid_test_positive_reaudit_summary.json"
OUTPUT_MD = ROOT / "docs" / "STEP16F_VALID_TEST_POSITIVE_REAUDIT.md"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_md(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def nested_counts(rows: list[dict[str, object]], outer: str, inner: str) -> dict[str, dict[str, int]]:
    counts: dict[str, Counter] = defaultdict(Counter)
    for row in rows:
        counts[str(row[outer])][str(row[inner])] += 1
    return {key: dict(counter) for key, counter in sorted(counts.items())}


def table_from_counter(title: str, counts: dict[str, int]) -> str:
    lines = [f"## {title}", "", "| Value | Count |", "|---|---:|"]
    for key, value in sorted(counts.items(), key=lambda item: (-item[1], item[0])):
        lines.append(f"| `{key}` | {value} |")
    return "\n".join(lines)


def table_from_nested_counter(title: str, counts: dict[str, dict[str, int]]) -> str:
    columns = sorted({inner for inner_counts in counts.values() for inner in inner_counts})
    lines = [f"## {title}", "", "| Split | " + " | ".join(f"`{column}`" for column in columns) + " |"]
    lines.append("|---|" + "|".join("---:" for _ in columns) + "|")
    for split, inner_counts in sorted(counts.items()):
        lines.append("| `" + split + "` | " + " | ".join(str(inner_counts.get(column, 0)) for column in columns) + " |")
    return "\n".join(lines)


def as_float(value: str, default: float = 0.0) -> float:
    try:
        return float(value or default)
    except (TypeError, ValueError):
        return default


def as_int(value: str, default: int = 0) -> int:
    try:
        return int(round(float(value or default)))
    except (TypeError, ValueError):
        return default


def split_contacts(value: str) -> list[str]:
    return [part.strip().lower() for part in str(value or "").split("||") if part.strip()]


def token_type(token: str) -> str:
    return token.split(":", 1)[0] if ":" in token else "unknown"


def build_signal_index(signal_rows: list[dict[str, str]]) -> dict[tuple[str, str], list[dict[str, str]]]:
    index: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in signal_rows:
        token = f"{row.get('contact_type', '')}:{row.get('normalized_value', '').lower()}"
        index[(row.get("seller_uid", ""), token)].append(row)
    return index


def contact_context_summary(row: dict[str, str], signal_index: dict[tuple[str, str], list[dict[str, str]]]) -> dict[str, object]:
    contacts = split_contacts(row.get("shared_contact_values", ""))
    summaries = []
    totals = Counter()
    for token in contacts:
        signals = signal_index.get((row.get("seller_uid_left", ""), token), []) + signal_index.get(
            (row.get("seller_uid_right", ""), token), []
        )
        seller_facing = sum(1 for signal in signals if signal.get("seller_facing_context") == "1")
        product_data_risk = sum(1 for signal in signals if signal.get("product_data_risk_context") == "1")
        direct_eligible = sum(1 for signal in signals if signal.get("direct_identity_eligible") == "1")
        summaries.append(
            {
                "token": token,
                "type": token_type(token),
                "seller_facing": seller_facing,
                "product_data_risk": product_data_risk,
                "direct_eligible": direct_eligible,
                "context_preview": " || ".join(signal.get("context", "")[:120] for signal in signals[:2]),
            }
        )
        totals["seller_facing"] += seller_facing
        totals["product_data_risk"] += product_data_risk
        totals["direct_eligible"] += direct_eligible
        totals[f"type::{token_type(token)}"] += 1
    return {"contacts": contacts, "summaries": summaries, "totals": dict(totals)}


def classify_positive(row: dict[str, str], evidence_type: str, contact_summary: dict[str, object]) -> dict[str, object]:
    notes = row.get("review_notes", "").lower()
    title_count = as_int(row.get("shared_title_count"))
    desc_count = as_int(row.get("shared_description_count"))
    category_count = as_int(row.get("shared_category_count"))
    lexical = as_float(row.get("lexical_similarity"))
    structural = as_float(row.get("structural_support_score"))
    contacts = contact_summary["contacts"]
    totals = Counter(contact_summary["totals"])
    contact_types = {token_type(token) for token in contacts}
    text_overlap = title_count + desc_count
    strong_note = any(
        token in notes
        for token in [
            "exact",
            "multiple",
            "long",
            "clone",
            "cloned",
            "duplicated",
            "repeated",
            "repeats",
            "rare",
            "distinctive",
            "ordered",
            "same detailed",
            "连续重合",
            "高度一致",
            "重复",
            "一致",
            "超出",
        ]
    )
    component_note = any(token in notes for token in ["component", "cluster", "links into", "extends", "reinforces"])
    low_frequency_contact_note = "low-frequency shared seller-facing contact" in notes

    reasons: list[str] = []
    risk_flags: list[str] = []

    if evidence_type == "same_controller_direct_identifier":
        if contacts and contact_types <= {"email"} and totals["seller_facing"] == 0 and totals["direct_eligible"] == 0:
            reasons.append("shared contact values are email fields from product/user records, not seller-facing contacts")
            risk_flags.append("product_data_email_not_seller_identity")
            return {
                "paper_evidence_tier": "soft_product_data_clone_not_direct_identity",
                "recommended_use": "secondary_or_sensitivity_only",
                "confidence": "medium",
                "needs_manual_recheck": True,
                "risk_flags": "|".join(risk_flags),
                "rationale": "; ".join(reasons),
            }
        if contacts and contact_types - {"email"} and totals["seller_facing"] > 0:
            reasons.append("shared seller-facing handle/contact appears in item/profile context")
            if totals["product_data_risk"] > 0:
                risk_flags.append("contact_context_also_mentions_data_product")
            return {
                "paper_evidence_tier": "gold_direct_seller_contact",
                "recommended_use": "primary_gold_benchmark",
                "confidence": "high",
                "needs_manual_recheck": bool(totals["direct_eligible"] == 0 and totals["product_data_risk"] > 0),
                "risk_flags": "|".join(risk_flags),
                "rationale": "; ".join(reasons),
            }
        if contacts and totals["seller_facing"] > 0:
            reasons.append("shared contact has seller-facing context but weaker contact type support")
            return {
                "paper_evidence_tier": "gold_direct_seller_contact_weaker_type",
                "recommended_use": "primary_gold_benchmark_with_contact_slice",
                "confidence": "medium_high",
                "needs_manual_recheck": False,
                "risk_flags": "",
                "rationale": "; ".join(reasons),
            }

    if evidence_type == "same_controller_component_anchor":
        reasons.append("positive is supported through reviewed same-controller component closure")
        return {
            "paper_evidence_tier": "gold_component_anchor",
            "recommended_use": "primary_gold_benchmark_component_slice",
            "confidence": "medium_high",
            "needs_manual_recheck": False,
            "risk_flags": "",
            "rationale": "; ".join(reasons),
        }

    if low_frequency_contact_note:
        reasons.append("review note records low-frequency shared seller-facing contact support, although pair feature lacks direct contact")
        return {
            "paper_evidence_tier": "component_or_contact_supported_soft_positive",
            "recommended_use": "primary_gold_benchmark_with_soft_slice",
            "confidence": "medium_high",
            "needs_manual_recheck": True,
            "risk_flags": "direct_contact_not_in_pair_feature",
            "rationale": "; ".join(reasons),
        }

    if text_overlap >= 4 and (strong_note or lexical >= 0.45 or structural >= 0.35):
        reasons.append("multiple shared titles/descriptions or long clone-like text support same-controller label")
        return {
            "paper_evidence_tier": "strong_soft_structural_clone",
            "recommended_use": "primary_gold_benchmark_soft_slice",
            "confidence": "medium_high",
            "needs_manual_recheck": False,
            "risk_flags": "not_direct_identity",
            "rationale": "; ".join(reasons),
        }

    if text_overlap >= 2 and (strong_note or structural >= 0.30 or lexical >= 0.50):
        reasons.append("repeated text/structure gives moderate soft support but no direct identity anchor")
        return {
            "paper_evidence_tier": "moderate_soft_structural_positive",
            "recommended_use": "secondary_or_slice_reported_gold",
            "confidence": "medium",
            "needs_manual_recheck": True,
            "risk_flags": "not_direct_identity",
            "rationale": "; ".join(reasons),
        }

    if component_note or category_count > 0 or structural >= 0.45:
        reasons.append("component/category/structural continuity supports label but direct text/contact evidence is limited")
        return {
            "paper_evidence_tier": "weak_component_or_semantic_positive",
            "recommended_use": "sensitivity_only_or_reaudit",
            "confidence": "low_medium",
            "needs_manual_recheck": True,
            "risk_flags": "weak_soft_positive",
            "rationale": "; ".join(reasons),
        }

    reasons.append("positive relies on weak semantic/structural notes with little direct pair evidence")
    return {
        "paper_evidence_tier": "weak_soft_positive_needs_reaudit",
        "recommended_use": "sensitivity_only_or_exclude_from_primary",
        "confidence": "low",
        "needs_manual_recheck": True,
        "risk_flags": "weak_soft_positive",
        "rationale": "; ".join(reasons),
    }


def main() -> None:
    label_rows = read_csv(LABELS)
    evidence_rows = {row["pair_uid"]: row for row in read_csv(EVIDENCE)}
    signal_index = build_signal_index(read_csv(SIGNALS))

    audit_rows: list[dict[str, object]] = []
    for row in label_rows:
        if row.get("split_name") not in {"valid", "test"} or row.get("review_label") != "positive":
            continue
        evidence_type = evidence_rows.get(row["pair_uid"], {}).get("evidence_type", "missing")
        contact_summary = contact_context_summary(row, signal_index)
        classification = classify_positive(row, evidence_type, contact_summary)
        audit_rows.append(
            {
                "pair_uid": row["pair_uid"],
                "split_name": row.get("split_name", ""),
                "source_market_raw_left": row.get("source_market_raw_left", ""),
                "source_market_raw_right": row.get("source_market_raw_right", ""),
                "source_seller_raw_left": row.get("source_seller_raw_left", ""),
                "source_seller_raw_right": row.get("source_seller_raw_right", ""),
                "review_stratum": row.get("review_stratum", ""),
                "evidence_type": evidence_type,
                "paper_evidence_tier": classification["paper_evidence_tier"],
                "recommended_use": classification["recommended_use"],
                "confidence": classification["confidence"],
                "needs_manual_recheck": str(classification["needs_manual_recheck"]).lower(),
                "risk_flags": classification["risk_flags"],
                "rationale": classification["rationale"],
                "shared_contact_count": row.get("shared_contact_count", ""),
                "shared_contact_values": row.get("shared_contact_values", ""),
                "contact_context_totals": json.dumps(contact_summary["totals"], ensure_ascii=False, sort_keys=True),
                "shared_title_count": row.get("shared_title_count", ""),
                "shared_description_count": row.get("shared_description_count", ""),
                "shared_category_count": row.get("shared_category_count", ""),
                "lexical_similarity": row.get("lexical_similarity", ""),
                "structural_support_score": row.get("structural_support_score", ""),
                "reviewer_id": row.get("reviewer_id", ""),
                "review_notes": row.get("review_notes", ""),
            }
        )

    fieldnames = list(audit_rows[0].keys()) if audit_rows else []
    write_csv(OUTPUT_CSV, audit_rows, fieldnames)

    tier_counts = dict(Counter(row["paper_evidence_tier"] for row in audit_rows))
    recommended_counts = dict(Counter(row["recommended_use"] for row in audit_rows))
    confidence_counts = dict(Counter(row["confidence"] for row in audit_rows))
    split_counts = dict(Counter(row["split_name"] for row in audit_rows))
    by_split_tier = nested_counts(audit_rows, "split_name", "paper_evidence_tier")
    by_split_recommended = nested_counts(audit_rows, "split_name", "recommended_use")
    by_split_confidence = nested_counts(audit_rows, "split_name", "confidence")
    primary_direct_or_component_tiers = {
        "gold_direct_seller_contact",
        "gold_direct_seller_contact_weaker_type",
        "gold_component_anchor",
    }
    primary_soft_tiers = {
        "strong_soft_structural_clone",
        "component_or_contact_supported_soft_positive",
    }
    paper_bucket_counts = {
        "direct_or_component_primary": sum(
            1 for row in audit_rows if row["paper_evidence_tier"] in primary_direct_or_component_tiers
        ),
        "soft_primary_or_slice": sum(1 for row in audit_rows if row["paper_evidence_tier"] in primary_soft_tiers),
        "secondary_or_sensitivity_only": sum(
            1
            for row in audit_rows
            if row["paper_evidence_tier"] not in primary_direct_or_component_tiers
            and row["paper_evidence_tier"] not in primary_soft_tiers
        ),
    }

    summary = {
        "step": "step16f_valid_test_positive_reaudit",
        "scope": "zh_target_strict valid/test positives only",
        "input_labels": str(LABELS.relative_to(ROOT)),
        "input_evidence": str(EVIDENCE.relative_to(ROOT)),
        "input_signals": str(SIGNALS.relative_to(ROOT)),
        "row_count": len(audit_rows),
        "split_counts": split_counts,
        "paper_evidence_tier_counts": tier_counts,
        "recommended_use_counts": recommended_counts,
        "confidence_counts": confidence_counts,
        "paper_bucket_counts": paper_bucket_counts,
        "by_split_paper_evidence_tier_counts": by_split_tier,
        "by_split_recommended_use_counts": by_split_recommended,
        "by_split_confidence_counts": by_split_confidence,
        "needs_manual_recheck_count": sum(1 for row in audit_rows if row["needs_manual_recheck"] == "true"),
        "risk_flag_counts": dict(
            Counter(flag for row in audit_rows for flag in str(row["risk_flags"]).split("|") if flag)
        ),
        "outputs": {
            "csv": str(OUTPUT_CSV.relative_to(ROOT)),
            "json": str(OUTPUT_JSON.relative_to(ROOT)),
            "md": str(OUTPUT_MD.relative_to(ROOT)),
        },
    }
    write_json(OUTPUT_JSON, summary)

    md = f"""# Step16F Valid/Test Positive Reaudit

Date: 2026-07-09

## Scope

This audit rechecks only the current `zh_target_strict` validation and test positive rows after the Step16C/E refreeze.
It does not modify Step5 labels. Its purpose is to stratify positive labels by evidence strength for paper reporting.

## Summary

- Audited positive rows: `{summary['row_count']}`
- Split counts: `{json.dumps(summary['split_counts'], ensure_ascii=False)}`
- Paper bucket counts: `{json.dumps(summary['paper_bucket_counts'], ensure_ascii=False)}`
- Rows needing manual recheck before a strongest paper claim: `{summary['needs_manual_recheck_count']}`

{table_from_counter("Evidence-Tier Counts", tier_counts)}

{table_from_nested_counter("Evidence-Tier Counts by Split", by_split_tier)}

{table_from_counter("Recommended-Use Counts", recommended_counts)}

{table_from_nested_counter("Recommended-Use Counts by Split", by_split_recommended)}

{table_from_counter("Risk-Flag Counts", summary['risk_flag_counts'])}

## Interpretation

The current validation/test positives are usable for continued experiments, but they should not be reported as one undifferentiated gold class.
The paper should report at least these positive slices:

1. `gold_direct_seller_contact`
2. `gold_component_anchor`
3. `strong_soft_structural_clone`
4. softer or risk-flagged positives used only in secondary/sensitivity analysis

Rows flagged as `product_data_email_not_seller_identity` should not be described as direct identity-anchor positives. They can remain as clone/soft positives only if the cloned listing evidence is accepted by the annotation protocol.

The strictest primary-positive subset is `direct_or_component_primary`.
The broader internal benchmark can additionally include `soft_primary_or_slice`, but this must be stated explicitly because these rows are not direct seller-identity anchors.
Rows in `secondary_or_sensitivity_only` should be used for sensitivity analysis or manual follow-up, not for the strongest paper claim.

## Outputs

- CSV: `{summary['outputs']['csv']}`
- JSON summary: `{summary['outputs']['json']}`
"""
    write_md(OUTPUT_MD, md)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
