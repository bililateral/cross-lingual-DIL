#!/usr/bin/env python3
"""Reconcile two blind Step16-v8 identity-control reviews and prepare adjudication."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent

SOURCE_FIELDS = [
    "candidate_uid",
    "seller_uid_left",
    "seller_uid_right",
    "platform_vendor_id",
    "strict_profile_seller_uid",
    "aux_profile_seller_uid",
    "strict_source_dataset",
    "strict_source_market",
    "aux_source_dataset",
    "aux_source_market",
    "strict_profile_item_count",
    "aux_profile_item_count",
    "exact_shared_title_count",
    "exact_shared_titles",
    "exact_shared_description_count",
    "exact_shared_descriptions",
    "strict_title_preview",
    "strict_description_preview",
    "aux_title_preview",
    "aux_description_preview",
    "cohort_a_item_count",
    "cohort_b_item_count",
    "cohort_a_item_preview",
    "cohort_b_item_preview",
    "same_vendor_path_evidence",
    "strict_profile_jsonl_line_number",
    "aux_profile_jsonl_line_number",
    "strict_profile_record_sha256",
    "aux_profile_record_sha256",
    "source_evidence_sha256",
]
ANSWER_FIELDS = [
    "review_label",
    "evidence_type",
    "review_confidence",
    "review_reason",
    "reviewer_id",
]
PACKET_FIELDS = SOURCE_FIELDS + ANSWER_FIELDS


def resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def relative(path: Path) -> str:
    return str(path.resolve().relative_to(ROOT.resolve())).replace("\\", "/")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_hash(value: object) -> str:
    payload = json.dumps(
        value, sort_keys=True, ensure_ascii=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def load_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def render_csv(rows: list[dict]) -> bytes:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(
        buffer, fieldnames=PACKET_FIELDS, extrasaction="raise", lineterminator="\n"
    )
    writer.writeheader()
    writer.writerows(rows)
    return ("\ufeff" + buffer.getvalue()).encode("utf-8")


def validate_summary(path: Path) -> dict:
    summary = json.loads(path.read_text(encoding="utf-8"))
    expected = summary.get("summary_self_sha256")
    unsigned = dict(summary)
    unsigned.pop("summary_self_sha256", None)
    if expected != canonical_hash(unsigned):
        raise ValueError(f"Identity-control summary self-hash is invalid: {path}")
    producer_record = summary["provenance"]
    producer_path = resolve(producer_record["producer"])
    if sha256(producer_path) != producer_record["producer_sha256"]:
        raise ValueError("Identity-control queue producer changed after generation")
    return summary


def validate_source_hash(row: dict) -> None:
    source = {field: row[field] for field in SOURCE_FIELDS if field != "source_evidence_sha256"}
    if canonical_hash(source) != row["source_evidence_sha256"]:
        raise ValueError(f"Source-evidence hash mismatch: {row['candidate_uid']}")


def load_completed(
    path: Path,
    template_index: dict[str, dict],
    role: str,
) -> dict[str, dict]:
    rows = load_csv(path)
    if not rows or list(rows[0]) != PACKET_FIELDS:
        raise ValueError(f"Reviewer {role} packet schema is invalid: {path}")
    index = {}
    reviewer_ids = set()
    for row in rows:
        uid = row["candidate_uid"]
        if not uid or uid in index or uid not in template_index:
            raise ValueError(f"Reviewer {role} has an unknown or duplicate UID: {uid}")
        for field in SOURCE_FIELDS:
            if row[field] != template_index[uid][field]:
                raise ValueError(f"Reviewer {role} changed {field}: {uid}")
        validate_source_hash(row)
        label = row["review_label"].strip().lower()
        evidence = row["evidence_type"].strip()
        confidence = row["review_confidence"].strip().lower()
        reviewer_id = row["reviewer_id"].strip()
        if label not in {"positive", "negative", "uncertain"}:
            raise ValueError(f"Reviewer {role} label is invalid: {uid}:{label}")
        allowed_evidence = {
            "positive": {
                "same_controller_direct_identifier",
                "same_controller_component_anchor",
            },
            "negative": {"ordinary_negative"},
            "uncertain": {"uncertain_insufficient_evidence"},
        }
        if evidence not in allowed_evidence[label]:
            raise ValueError(f"Reviewer {role} evidence is invalid: {uid}:{evidence}")
        if confidence not in {"high", "medium", "low"}:
            raise ValueError(f"Reviewer {role} confidence is invalid: {uid}:{confidence}")
        if not reviewer_id or not row["review_reason"].strip():
            raise ValueError(f"Reviewer {role} lacks id or reason: {uid}")
        reviewer_ids.add(reviewer_id.casefold())
        index[uid] = row
    if set(index) != set(template_index):
        raise ValueError(f"Reviewer {role} did not complete the exact formal universe")
    if len(reviewer_ids) != 1:
        raise ValueError(f"Reviewer {role} uses multiple ids: {sorted(reviewer_ids)}")
    return index


def decision(row: dict) -> tuple[str, str, str]:
    return (
        row["review_label"].strip().lower(),
        row["evidence_type"].strip(),
        row["review_confidence"].strip().lower(),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", required=True)
    parser.add_argument("--reviewer-a", required=True)
    parser.add_argument("--reviewer-b", required=True)
    args = parser.parse_args()

    summary_path = resolve(args.summary)
    summary = validate_summary(summary_path)
    root = summary_path.parent
    artifacts = summary["artifacts"]
    master_path = root / artifacts["candidate_master"]["filename"]
    template_path = root / artifacts["reviewer_a"]["filename"]
    adjudicator_source_path = root / artifacts["adjudicator"]["filename"]
    for key, path in (
        ("candidate_master", master_path),
        ("reviewer_a", template_path),
        ("adjudicator", adjudicator_source_path),
    ):
        if sha256(path) != artifacts[key]["sha256"]:
            raise ValueError(f"Immutable identity-control artifact changed: {path}")
    templates = load_csv(template_path)
    if not templates or list(templates[0]) != PACKET_FIELDS:
        raise ValueError("Identity-control blind template schema changed")
    template_index = {row["candidate_uid"]: row for row in templates}
    if len(template_index) != len(templates):
        raise ValueError("Identity-control blind template duplicates candidates")
    for row in templates:
        validate_source_hash(row)
        if any(row[field].strip() for field in ANSWER_FIELDS):
            raise ValueError("Identity-control blind template contains reviewer answers")

    master_rows = load_csv(master_path)
    master_index = {row["candidate_uid"]: row for row in master_rows}
    if set(master_index) != set(template_index):
        raise ValueError("Identity-control master/template candidate universes differ")
    reviewer_a_path = resolve(args.reviewer_a)
    reviewer_b_path = resolve(args.reviewer_b)
    reviewer_a = load_completed(reviewer_a_path, template_index, "a")
    reviewer_b = load_completed(reviewer_b_path, template_index, "b")
    ids_a = {row["reviewer_id"].casefold() for row in reviewer_a.values()}
    ids_b = {row["reviewer_id"].casefold() for row in reviewer_b.values()}
    if ids_a & ids_b:
        raise ValueError("The two identity-control reviewers must be distinct")

    disagreement_uids = set()
    invalid_kind_decisions = []
    for uid in sorted(template_index):
        for role, row in (("a", reviewer_a[uid]), ("b", reviewer_b[uid])):
            label, evidence, _ = decision(row)
            kind = master_index[uid]["candidate_kind"]
            expected_positive = (
                "same_controller_component_anchor"
                if kind == "evidence_expert_component_closure_control"
                else "same_controller_direct_identifier"
            )
            if label == "positive" and evidence != expected_positive:
                invalid_kind_decisions.append((uid, role, kind, evidence))
        if decision(reviewer_a[uid]) != decision(reviewer_b[uid]):
            disagreement_uids.add(uid)
    if invalid_kind_decisions:
        raise ValueError(
            "Positive evidence type contradicts candidate provenance: "
            f"{invalid_kind_decisions[0]}"
        )

    adjudication_rows = []
    adjudicator_templates = {row["candidate_uid"]: row for row in load_csv(adjudicator_source_path)}
    if set(adjudicator_templates) != set(template_index):
        raise ValueError("Adjudicator template universe differs from blind reviewers")
    for uid in sorted(disagreement_uids):
        adjudication_rows.append(dict(adjudicator_templates[uid]))

    outputs = {
        "reviewer_a_formal": root / "reviewer_a_blind_packet.completed.csv",
        "reviewer_b_formal": root / "reviewer_b_blind_packet.completed.csv",
        "adjudication_template": root / "reviewer_adjudicator_disagreements.template.csv",
        "summary": root / "identity_control_dual_review_reconciliation_summary.json",
    }
    existing = [path for path in outputs.values() if path.exists()]
    if existing:
        raise FileExistsError(f"Refusing to overwrite identity reconciliation: {existing[0]}")
    outputs["reviewer_a_formal"].write_bytes(
        render_csv([reviewer_a[row["candidate_uid"]] for row in templates])
    )
    outputs["reviewer_b_formal"].write_bytes(
        render_csv([reviewer_b[row["candidate_uid"]] for row in templates])
    )
    outputs["adjudication_template"].write_bytes(render_csv(adjudication_rows))

    matching = [uid for uid in template_index if uid not in disagreement_uids]
    payload = {
        "step": "step16_reconcile_v8_identity_control_reviews",
        "run_id": summary["run_id"],
        "formal_candidate_count": len(template_index),
        "matching_decision_count": len(matching),
        "disagreement_count": len(disagreement_uids),
        "matching_decision_counts": dict(
            sorted(Counter("/".join(decision(reviewer_a[uid])) for uid in matching).items())
        ),
        "adjudication_candidate_uids": sorted(disagreement_uids),
        "reviewer_decisions_hidden_from_adjudication_packet": True,
        "candidate_kind_positive_evidence_contract_enforced": True,
        "inputs": {
            relative(summary_path): sha256(summary_path),
            relative(master_path): sha256(master_path),
            relative(template_path): sha256(template_path),
            relative(reviewer_a_path): sha256(reviewer_a_path),
            relative(reviewer_b_path): sha256(reviewer_b_path),
        },
        "outputs": {
            key: {"path": relative(path), "sha256": sha256(path)}
            for key, path in outputs.items()
            if key != "summary"
        },
    }
    payload["summary_sha256"] = canonical_hash(payload)
    outputs["summary"].write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
