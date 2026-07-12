#!/usr/bin/env python3
"""Aggregate Step16H reviews, adjudicate conflicts, and freeze auditable decisions."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import subprocess
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_POLICY = ROOT / "schema" / "step16h_blind_positive_reaudit_policy.json"
REVIEW_FIELDS = {
    "review_index",
    "independent_decision",
    "seller_facing_direct_evidence",
    "alternative_explanation_ruled_out",
    "review_confidence",
    "review_rationale",
}


def resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def load_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, capture_output=True, text=True
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip() or None


def prepared_queue_hash(rows: list[dict]) -> str:
    """Recreate the prepared blank queue hash while ignoring completed review responses."""
    if not rows:
        raise ValueError("Cannot hash an empty Step16H reviewer queue")
    fields = list(rows[0])
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fields, lineterminator="\r\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({field: "" if field in REVIEW_FIELDS - {"review_index"} else row.get(field, "") for field in fields})
    return hashlib.sha256(buffer.getvalue().encode("utf-8-sig")).hexdigest()


def cohen_kappa(left: list[str], right: list[str]) -> float | None:
    if not left:
        return None
    observed = sum(a == b for a, b in zip(left, right, strict=True)) / len(left)
    left_counts = Counter(left)
    right_counts = Counter(right)
    expected = sum(
        (left_counts[label] / len(left)) * (right_counts[label] / len(right))
        for label in set(left) | set(right)
    )
    return None if expected >= 1.0 else (observed - expected) / (1.0 - expected)


def krippendorff_alpha_nominal(left: list[str], right: list[str]) -> float | None:
    if not left:
        return None
    observed_disagreement = sum(a != b for a, b in zip(left, right, strict=True)) / len(left)
    pooled = Counter([*left, *right])
    total = 2 * len(left)
    if total <= 1:
        return None
    expected_disagreement = sum(
        count * (total - count) for count in pooled.values()
    ) / (total * (total - 1))
    return None if expected_disagreement <= 0.0 else 1.0 - observed_disagreement / expected_disagreement


def agreement_stats(left: list[str], right: list[str]) -> dict:
    conflicts = sum(a != b for a, b in zip(left, right, strict=True))
    return {
        "row_count": len(left),
        "exact_agreement_count": len(left) - conflicts,
        "disagreement_count": conflicts,
        "exact_agreement_rate": round((len(left) - conflicts) / max(len(left), 1), 8),
        "conflict_rate": round(conflicts / max(len(left), 1), 8),
        "cohen_kappa": cohen_kappa(left, right),
        "krippendorff_alpha_nominal": krippendorff_alpha_nominal(left, right),
        "reviewer_a_decision_counts": dict(Counter(left)),
        "reviewer_b_decision_counts": dict(Counter(right)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", default=str(DEFAULT_POLICY))
    args = parser.parse_args()
    policy_path = resolve(args.policy)
    policy = json.loads(policy_path.read_text(encoding="utf-8"))
    outputs = policy["outputs"]
    key_field = str(policy.get("review_key_field", "pair_uid"))
    queue_paths = {
        "reviewer_a": resolve(outputs["reviewer_a_queue"]),
        "reviewer_b": resolve(outputs["reviewer_b_queue"]),
    }
    rows_a = load_csv(queue_paths["reviewer_a"])
    rows_b = load_csv(queue_paths["reviewer_b"])
    expected_count = int(policy["expected_row_count"])
    if len(rows_a) != expected_count or len(rows_b) != expected_count:
        raise ValueError("Step16H reviewer queue row count does not match policy")
    index_a = {row[key_field]: row for row in rows_a}
    index_b = {row[key_field]: row for row in rows_b}
    if len(index_a) != len(rows_a) or len(index_b) != len(rows_b) or set(index_a) != set(index_b):
        raise ValueError("Step16H reviewer queues do not contain the same unique review universe")

    prepared_manifest_path = resolve(outputs["manifest"])
    prepared_manifest = json.loads(prepared_manifest_path.read_text(encoding="utf-8"))
    manifest_queue_index = {
        name: payload for name, payload in (prepared_manifest.get("queues") or {}).items()
    }
    for reviewer, rows in (("reviewer_a", rows_a), ("reviewer_b", rows_b)):
        if prepared_queue_hash(rows) != (manifest_queue_index.get(reviewer) or {}).get("sha256"):
            raise ValueError(f"Step16H {reviewer} evidence/order differs from the prepared blind queue")

    evidence_fields = [field for field in rows_a[0] if field not in REVIEW_FIELDS and field != key_field]
    for review_key in index_a:
        for field in evidence_fields:
            if index_a[review_key].get(field, "") != index_b[review_key].get(field, ""):
                raise ValueError(f"Step16H reviewer evidence mismatch for {review_key}/{field}")

    allowed = set(policy["allowed_decisions"])
    incomplete = []
    required_review_fields = (
        "review_rationale",
        "seller_facing_direct_evidence",
        "alternative_explanation_ruled_out",
        "review_confidence",
    )
    for reviewer, index in (("reviewer_a", index_a), ("reviewer_b", index_b)):
        for review_key, row in index.items():
            if row.get("independent_decision") not in allowed or any(
                not row.get(field, "").strip() for field in required_review_fields
            ):
                incomplete.append(f"{reviewer}:{review_key}")
    if incomplete:
        raise SystemExit(f"Step16H reviews are incomplete: {len(incomplete)}; first={incomplete[0]}")

    review_order = sorted(index_a)
    decisions_a = [index_a[key]["independent_decision"] for key in review_order]
    decisions_b = [index_b[key]["independent_decision"] for key in review_order]
    adjudication_path = resolve(outputs["adjudication_queue"])
    existing_adjudication = {
        row[key_field]: row for row in load_csv(adjudication_path)
    } if adjudication_path.exists() else {}
    disagreements = []
    for review_key in review_order:
        left = index_a[review_key]
        right = index_b[review_key]
        if left["independent_decision"] == right["independent_decision"]:
            continue
        prior = existing_adjudication.get(review_key, {})
        if prior and (
            prior.get("reviewer_a_decision") != left["independent_decision"]
            or prior.get("reviewer_b_decision") != right["independent_decision"]
        ):
            prior = {}
        disagreements.append(
            {
                key_field: review_key,
                **{field: left.get(field, "") for field in evidence_fields},
                "reviewer_a_decision": left["independent_decision"],
                "reviewer_a_rationale": left["review_rationale"],
                "reviewer_b_decision": right["independent_decision"],
                "reviewer_b_rationale": right["review_rationale"],
                "adjudicated_decision": prior.get("adjudicated_decision", ""),
                "adjudicator_id": prior.get("adjudicator_id", ""),
                "adjudication_rationale": prior.get("adjudication_rationale", ""),
            }
        )
    adjudication_fields = [
        key_field,
        *evidence_fields,
        "reviewer_a_decision",
        "reviewer_a_rationale",
        "reviewer_b_decision",
        "reviewer_b_rationale",
        "adjudicated_decision",
        "adjudicator_id",
        "adjudication_rationale",
    ]
    write_csv(adjudication_path, disagreements, adjudication_fields)
    adjudication_index = {row[key_field]: row for row in disagreements}
    adjudication_complete = all(
        row.get("adjudicated_decision") in allowed
        and row.get("adjudicator_id", "").strip()
        and row.get("adjudicator_id") not in set(policy.get("reviewers", []))
        and row.get("adjudication_rationale", "").strip()
        for row in disagreements
    )

    labels = load_csv(resolve(policy["inputs"]["frozen_labels"]))
    label_index = {row["pair_uid"]: row for row in labels}
    mapping: dict[str, dict] = {}
    blind_mapping_value = outputs.get("blind_mapping")
    if blind_mapping_value:
        mapping_path = resolve(blind_mapping_value)
        expected_mapping_sha = str(prepared_manifest.get("blind_mapping_sha256", "")).strip()
        if expected_mapping_sha and sha256(mapping_path) != expected_mapping_sha:
            raise ValueError("Step16H blind mapping hash disagrees with the prepared manifest")
        mapping = {row[key_field]: row for row in load_csv(mapping_path)}
    else:
        mapping = {
            key: {
                key_field: key,
                "pair_uid": key,
                "split_name": label_index[key]["split_name"],
                "reference_subset": (
                    "positive_candidate"
                    if label_index[key]["review_label"] == "positive"
                    else "negative_control"
                ),
            }
            for key in review_order
        }
    if set(mapping) != set(review_order):
        raise ValueError("Step16H blind mapping does not match the review universe")

    final_rows = []
    for review_key in review_order:
        left = index_a[review_key]["independent_decision"]
        right = index_b[review_key]["independent_decision"]
        adjudication = adjudication_index.get(review_key, {})
        final_decision = left if left == right else adjudication.get("adjudicated_decision", "")
        final_rows.append(
            {
                key_field: review_key,
                "pair_uid": mapping[review_key]["pair_uid"],
                "split_name": mapping[review_key]["split_name"],
                "reference_subset": mapping[review_key]["reference_subset"],
                "reviewer_a_decision": left,
                "reviewer_b_decision": right,
                "adjudicated_decision": adjudication.get("adjudicated_decision", ""),
                "adjudicator_id": adjudication.get("adjudicator_id", ""),
                "final_decision": final_decision,
                "final_tier_role": {
                    "strict_same_controller": "strict_gold_candidate",
                    "soft_same_controller": "sensitivity_only_not_strict_gold",
                    "different_controller": "reviewed_negative_candidate",
                    "uncertain": "exclude_from_binary_evaluation",
                }.get(final_decision, "pending_adjudication"),
            }
        )
    final_path = resolve(outputs["final_pair_decisions"])
    if adjudication_complete:
        write_csv(final_path, final_rows, list(final_rows[0]))

    subset_stats = {}
    for subset_name in ("positive_candidate", "negative_control"):
        keys = [key for key in review_order if mapping[key]["reference_subset"] == subset_name]
        subset_stats[subset_name] = agreement_stats(
            [index_a[key]["independent_decision"] for key in keys],
            [index_b[key]["independent_decision"] for key in keys],
        )
    split_stats = {}
    for split_name in sorted({mapping[key]["split_name"] for key in review_order}):
        keys = [key for key in review_order if mapping[key]["split_name"] == split_name]
        split_stats[split_name] = agreement_stats(
            [index_a[key]["independent_decision"] for key in keys],
            [index_b[key]["independent_decision"] for key in keys],
        )
    subset_split_stats = {}
    for subset_name in ("positive_candidate", "negative_control"):
        subset_split_stats[subset_name] = {}
        for split_name in sorted({mapping[key]["split_name"] for key in review_order}):
            keys = [
                key
                for key in review_order
                if mapping[key]["reference_subset"] == subset_name
                and mapping[key]["split_name"] == split_name
            ]
            subset_split_stats[subset_name][split_name] = agreement_stats(
                [index_a[key]["independent_decision"] for key in keys],
                [index_b[key]["independent_decision"] for key in keys],
            )
    final_reference_summary = {}
    if adjudication_complete:
        for subset_name in ("positive_candidate", "negative_control"):
            decisions = [
                row["final_decision"] for row in final_rows if row["reference_subset"] == subset_name
            ]
            final_reference_summary[subset_name] = {
                "decision_counts": dict(Counter(decisions)),
                "strict_same_controller_rate": round(
                    decisions.count("strict_same_controller") / max(len(decisions), 1), 8
                ),
                "strict_or_soft_same_controller_rate": round(
                    sum(value in {"strict_same_controller", "soft_same_controller"} for value in decisions)
                    / max(len(decisions), 1),
                    8,
                ),
            }

    summary = {
        "step": "step16h_aggregate_blind_positive_reaudit",
        "policy_version": policy["version"],
        "reviewer_execution_type": policy.get("reviewer_execution_type"),
        **agreement_stats(decisions_a, decisions_b),
        "agreement_by_reference_subset": subset_stats,
        "agreement_by_split": split_stats,
        "agreement_by_reference_subset_and_split": subset_split_stats,
        "adjudication_complete": adjudication_complete,
        "completion_status": "complete" if adjudication_complete else "pending_adjudication",
        "final_reference_subset_summary": final_reference_summary,
        "final_decision_counts": dict(Counter(row["final_decision"] for row in final_rows)),
        "step5_labels_modified": False,
        "step16f_tiers_modified": False,
        "adjudication_required": bool(disagreements) and not adjudication_complete,
        "adjudication_queue": str(adjudication_path.relative_to(ROOT)),
        "final_pair_decisions": str(final_path.relative_to(ROOT)),
        "scientific_guard": (
            "Independent AI-assisted review is a sensitivity/data-quality audit. It is not human "
            "annotation, does not refreeze Step5/Step16F, and cannot be used to retune v6."
        ),
    }
    summary_path = resolve(outputs["agreement_summary"])
    write_json(summary_path, summary)

    completion_path = resolve(outputs["completion_manifest"])
    if not adjudication_complete:
        if completion_path.exists():
            raise ValueError(
                "Step16H found a stale completion manifest while adjudication is incomplete; "
                f"remove the invalid stale file before rerunning: {completion_path}"
            )
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        return

    completion_paths = [
        policy_path,
        prepared_manifest_path,
        queue_paths["reviewer_a"],
        queue_paths["reviewer_b"],
        adjudication_path,
        final_path,
        summary_path,
        Path(__file__).resolve(),
    ]
    if blind_mapping_value:
        completion_paths.append(resolve(blind_mapping_value))
    completion_core = {
        "step": "step16h_completion_manifest",
        "policy_version": policy["version"],
        "git_commit": git_commit(),
        "files": [
            {
                "path": str(path.relative_to(ROOT)),
                "sha256": sha256(path),
                "size_bytes": path.stat().st_size,
            }
            for path in sorted(set(completion_paths), key=lambda value: str(value))
        ],
    }
    completion_core["manifest_sha256"] = hashlib.sha256(
        json.dumps(completion_core, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
    ).hexdigest()
    write_json(completion_path, completion_core)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
