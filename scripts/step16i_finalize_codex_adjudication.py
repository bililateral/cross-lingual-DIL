#!/usr/bin/env python3
"""Freeze an owner-authorized Codex adjudication without unblinding pair identities."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import shutil
import uuid
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent

BATCH_FIELDS = (
    "review_index",
    "blind_id",
    "adjudicated_identity_decision",
    "adjudicated_evidence_type",
    "adjudication_confidence",
    "adjudication_rationale",
    "reviewer_a_identity",
    "reviewer_b_identity",
)

FINAL_FIELDS = (
    "review_index",
    "blind_id",
    "reviewer_a_identity_decision",
    "reviewer_a_evidence_type_decision",
    "reviewer_a_review_confidence",
    "reviewer_b_identity_decision",
    "reviewer_b_evidence_type_decision",
    "reviewer_b_review_confidence",
    "adjudicated_identity_decision",
    "adjudicated_evidence_type",
    "adjudication_confidence",
    "adjudication_rationale",
    "adjudicator_id",
    "annotation_origin",
    "dataset_owner_authorized",
    "human_verified_per_row",
    "prospective_final_eligible",
    "step5_supervision_eligible",
)

CONFIDENCE_VALUES = {"high", "medium", "low"}
SAME_EVIDENCE = {
    "same_controller_direct_identifier",
    "same_controller_component_anchor",
}
NEGATIVE_EVIDENCE = {
    "template_clone_not_controller",
    "semantic_topic_not_controller",
    "public_contact_or_url_noise",
    "ordinary_negative",
}
UNCERTAIN_EVIDENCE = {
    "same_controller_style_structural_soft",
    "template_clone_not_controller",
    "semantic_topic_not_controller",
    "public_contact_or_url_noise",
    "uncertain_insufficient_evidence",
}


def workspace_path(value: str | Path) -> Path:
    path = Path(value)
    resolved = (path if path.is_absolute() else ROOT / path).resolve()
    try:
        resolved.relative_to(ROOT.resolve())
    except ValueError as exc:
        raise ValueError(f"Path must remain inside the workspace: {resolved}") from exc
    if resolved.name.casefold() == "blind_mapping.csv":
        raise ValueError("Codex adjudication must be frozen before blind_mapping.csv is read")
    return resolved


def relative(path: Path) -> str:
    return str(path.resolve().relative_to(ROOT.resolve())).replace("\\", "/")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = list(reader.fieldnames or [])
        rows = list(reader)
    if not fields or len(fields) != len(set(fields)):
        raise ValueError(f"Invalid CSV schema: {path}")
    if not rows or any(None in row for row in rows):
        raise ValueError(f"Empty or malformed CSV: {path}")
    return fields, rows


def render_csv(rows: list[dict[str, object]], fields: tuple[str, ...]) -> bytes:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(
        buffer,
        fieldnames=list(fields),
        extrasaction="raise",
        lineterminator="\n",
    )
    writer.writeheader()
    writer.writerows(rows)
    return ("\ufeff" + buffer.getvalue()).encode("utf-8")


def load_policy(path: Path) -> tuple[set[str], set[str]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    blinding = payload.get("blinding", {})
    identities = {str(value) for value in blinding.get("allowed_identity_decisions", [])}
    evidence = {str(value) for value in blinding.get("allowed_evidence_type_decisions", [])}
    if identities != {"same_controller", "different_controller", "uncertain"}:
        raise ValueError("Unexpected Step16I identity vocabulary")
    if not (SAME_EVIDENCE | NEGATIVE_EVIDENCE | UNCERTAIN_EVIDENCE) <= evidence:
        raise ValueError("Step16I evidence vocabulary is incomplete")
    return identities, evidence


def index_completed(path: Path, role: str) -> tuple[list[dict[str, str]], dict[str, dict[str, str]]]:
    fields, rows = load_csv(path)
    required = {
        "review_index",
        "blind_id",
        "independent_identity_decision",
        "evidence_type_decision",
        "review_confidence",
        "review_rationale",
    }
    missing = sorted(required - set(fields))
    if missing:
        raise ValueError(f"Reviewer {role} completed file is missing fields: {missing}")
    index: dict[str, dict[str, str]] = {}
    for row in rows:
        blind_id = row["blind_id"].strip()
        if not blind_id or blind_id in index:
            raise ValueError(f"Reviewer {role} has a missing or duplicate blind_id")
        index[blind_id] = row
    return rows, index


def validate_identity_evidence(identity: str, evidence: str) -> None:
    if identity == "same_controller" and evidence not in SAME_EVIDENCE:
        raise ValueError(f"same_controller cannot use evidence type {evidence}")
    if identity == "different_controller" and evidence not in NEGATIVE_EVIDENCE:
        raise ValueError(f"different_controller cannot use evidence type {evidence}")
    if identity == "uncertain" and evidence not in UNCERTAIN_EVIDENCE:
        raise ValueError(f"uncertain cannot use evidence type {evidence}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", required=True)
    parser.add_argument("--reviewer-a-completed", required=True)
    parser.add_argument("--reviewer-b-completed", required=True)
    parser.add_argument("--batch", action="append", required=True)
    parser.add_argument("--output-directory", required=True)
    args = parser.parse_args()

    policy_path = workspace_path(args.policy)
    reviewer_a_path = workspace_path(args.reviewer_a_completed)
    reviewer_b_path = workspace_path(args.reviewer_b_completed)
    batch_paths = [workspace_path(value) for value in args.batch]
    output_directory = workspace_path(args.output_directory)
    if output_directory.exists():
        raise ValueError(f"Refusing to overwrite adjudication output: {output_directory}")

    allowed_identity, allowed_evidence = load_policy(policy_path)
    reviewer_a_rows, reviewer_a = index_completed(reviewer_a_path, "A")
    _, reviewer_b = index_completed(reviewer_b_path, "B")
    if set(reviewer_a) != set(reviewer_b):
        raise ValueError("Reviewer A and B completed blind_id universes differ")

    batch_index: dict[str, dict[str, str]] = {}
    batch_records: list[dict[str, object]] = []
    for batch_path in batch_paths:
        fields, rows = load_csv(batch_path)
        if tuple(fields) != BATCH_FIELDS:
            raise ValueError(f"Unexpected adjudication batch schema: {batch_path}")
        for row in rows:
            blind_id = row["blind_id"].strip()
            if blind_id in batch_index:
                raise ValueError(f"Duplicate adjudication blind_id: {blind_id}")
            if blind_id not in reviewer_a:
                raise ValueError(f"Unknown adjudication blind_id: {blind_id}")
            a_row = reviewer_a[blind_id]
            b_row = reviewer_b[blind_id]
            if row["review_index"].strip() != a_row["review_index"].strip():
                raise ValueError(f"Adjudication review_index mismatch: {blind_id}")
            if row["reviewer_a_identity"].strip() != a_row["independent_identity_decision"].strip():
                raise ValueError(f"Adjudication changed Reviewer A identity: {blind_id}")
            if row["reviewer_b_identity"].strip() != b_row["independent_identity_decision"].strip():
                raise ValueError(f"Adjudication changed Reviewer B identity: {blind_id}")

            identity = row["adjudicated_identity_decision"].strip()
            evidence = row["adjudicated_evidence_type"].strip()
            confidence = row["adjudication_confidence"].strip().lower()
            rationale = row["adjudication_rationale"].strip()
            if identity not in allowed_identity:
                raise ValueError(f"Invalid adjudicated identity for {blind_id}: {identity}")
            if evidence not in allowed_evidence:
                raise ValueError(f"Invalid adjudicated evidence for {blind_id}: {evidence}")
            if confidence not in CONFIDENCE_VALUES or not rationale:
                raise ValueError(f"Invalid adjudication confidence/rationale for {blind_id}")
            validate_identity_evidence(identity, evidence)
            batch_index[blind_id] = row
            batch_records.append(
                {
                    "path": relative(batch_path),
                    "sha256": sha256(batch_path),
                    "row_count": len(rows),
                }
            )

    if set(batch_index) != set(reviewer_a):
        missing = sorted(set(reviewer_a) - set(batch_index))
        extra = sorted(set(batch_index) - set(reviewer_a))
        raise ValueError(f"Adjudication coverage mismatch: missing={missing[:3]} extra={extra[:3]}")

    final_rows: list[dict[str, object]] = []
    for a_row in sorted(reviewer_a_rows, key=lambda row: int(row["review_index"])):
        blind_id = a_row["blind_id"].strip()
        b_row = reviewer_b[blind_id]
        decision = batch_index[blind_id]
        final_rows.append(
            {
                "review_index": int(a_row["review_index"]),
                "blind_id": blind_id,
                "reviewer_a_identity_decision": a_row["independent_identity_decision"].strip(),
                "reviewer_a_evidence_type_decision": a_row["evidence_type_decision"].strip(),
                "reviewer_a_review_confidence": a_row["review_confidence"].strip().lower(),
                "reviewer_b_identity_decision": b_row["independent_identity_decision"].strip(),
                "reviewer_b_evidence_type_decision": b_row["evidence_type_decision"].strip(),
                "reviewer_b_review_confidence": b_row["review_confidence"].strip().lower(),
                "adjudicated_identity_decision": decision["adjudicated_identity_decision"].strip(),
                "adjudicated_evidence_type": decision["adjudicated_evidence_type"].strip(),
                "adjudication_confidence": decision["adjudication_confidence"].strip().lower(),
                "adjudication_rationale": decision["adjudication_rationale"].strip(),
                "adjudicator_id": "codex_ai_adjudicator",
                "annotation_origin": "two_ai_blind_reviews_plus_codex_adjudication_owner_authorized",
                "dataset_owner_authorized": "true",
                "human_verified_per_row": "false",
                "prospective_final_eligible": "false",
                "step5_supervision_eligible": "false",
            }
        )

    identity_counts = Counter(str(row["adjudicated_identity_decision"]) for row in final_rows)
    evidence_counts = Counter(str(row["adjudicated_evidence_type"]) for row in final_rows)
    confidence_counts = Counter(str(row["adjudication_confidence"]) for row in final_rows)
    csv_bytes = render_csv(final_rows, FINAL_FIELDS)
    final_csv_hash = hashlib.sha256(csv_bytes).hexdigest()
    unique_batch_records = {record["path"]: record for record in batch_records}
    summary = {
        "status": "completed_owner_authorized_codex_adjudication",
        "scope": "retrospective_internal_dev_sensitivity_only",
        "row_count": len(final_rows),
        "identity_counts": dict(sorted(identity_counts.items())),
        "evidence_type_counts": dict(sorted(evidence_counts.items())),
        "confidence_counts": dict(sorted(confidence_counts.items())),
        "annotation_origin": "two_ai_blind_reviews_plus_codex_adjudication_owner_authorized",
        "dataset_owner_authorized": True,
        "human_verified_per_row": False,
        "may_be_described_as_two_independent_human_reviews": False,
        "prospective_claim_allowed": False,
        "step5_labels_created_or_modified": False,
        "binary_primary_eligible_rows": identity_counts["same_controller"] + identity_counts["different_controller"],
        "uncertain_excluded_from_binary_evaluation": identity_counts["uncertain"],
        "inputs": {
            "policy": {"path": relative(policy_path), "sha256": sha256(policy_path)},
            "reviewer_a_completed": {"path": relative(reviewer_a_path), "sha256": sha256(reviewer_a_path)},
            "reviewer_b_completed": {"path": relative(reviewer_b_path), "sha256": sha256(reviewer_b_path)},
            "adjudication_batches": [unique_batch_records[key] for key in sorted(unique_batch_records)],
        },
        "outputs": {
            "adjudication_by_blind_id": {
                "path": relative(output_directory / "adjudication_by_blind_id.csv"),
                "sha256": final_csv_hash,
                "row_count": len(final_rows),
            }
        },
    }

    temp_directory = output_directory.with_name(f".{output_directory.name}.tmp-{uuid.uuid4().hex}")
    temp_directory.mkdir(parents=True)
    try:
        (temp_directory / "adjudication_by_blind_id.csv").write_bytes(csv_bytes)
        (temp_directory / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        output_directory.parent.mkdir(parents=True, exist_ok=True)
        os.replace(temp_directory, output_directory)
    except Exception:
        shutil.rmtree(temp_directory, ignore_errors=True)
        raise

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
