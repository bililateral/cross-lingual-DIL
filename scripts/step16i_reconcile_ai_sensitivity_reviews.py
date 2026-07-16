#!/usr/bin/env python3
"""Reconcile two immutable Step16I AI-sensitivity reviews without creating labels."""

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

DECISION_FIELDS = (
    "independent_identity_decision",
    "evidence_type_decision",
    "review_confidence",
    "review_rationale",
)
CONFIDENCE_VALUES = {"high", "medium", "low"}
AGREEMENT_FIELDS = (
    "blind_id",
    "reviewer_a_identity_decision",
    "reviewer_a_evidence_type_decision",
    "reviewer_a_review_confidence",
    "reviewer_b_identity_decision",
    "reviewer_b_evidence_type_decision",
    "reviewer_b_review_confidence",
    "exact_identity_agreement",
    "exact_evidence_agreement",
    "high_confidence_exact_agreement",
    "needs_human_adjudication",
)


def resolve_workspace_path(value: str | Path) -> Path:
    path = Path(value)
    resolved = (path if path.is_absolute() else ROOT / path).resolve()
    try:
        resolved.relative_to(ROOT.resolve())
    except ValueError as exc:
        raise ValueError(f"Step16I paths must remain within the workspace: {resolved}") from exc
    return resolved


def relative_path(path: Path) -> str:
    return str(path.resolve().relative_to(ROOT.resolve())).replace("\\", "/")


def reject_blind_mapping_input(path: Path) -> None:
    if path.name.casefold() == "blind_mapping.csv":
        raise ValueError("The Step16I AI-sensitivity reconciler must never read blind_mapping.csv")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_hash(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, ensure_ascii=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def load_json_object(path: Path, description: str) -> dict:
    reject_blind_mapping_input(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{description} must contain a JSON object: {path}")
    return payload


def load_csv(path: Path, description: str) -> tuple[list[str], list[dict[str, str]]]:
    reject_blind_mapping_input(path)
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = list(reader.fieldnames or [])
        if not fields or len(fields) != len(set(fields)):
            raise ValueError(f"{description} has an empty or duplicate CSV schema: {path}")
        rows = list(reader)
    if not rows:
        raise ValueError(f"{description} is empty: {path}")
    if any(None in row for row in rows):
        raise ValueError(f"{description} contains cells outside its declared schema: {path}")
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


def bool_text(value: bool) -> str:
    return "true" if value else "false"


def policy_decisions(policy: dict) -> tuple[set[str], set[str]]:
    blinding = policy.get("blinding")
    if not isinstance(blinding, dict):
        raise ValueError("Policy is missing the blinding object")
    identity = {
        str(value).strip()
        for value in blinding.get("allowed_identity_decisions", [])
        if str(value).strip()
    }
    evidence = {
        str(value).strip()
        for value in blinding.get("allowed_evidence_type_decisions", [])
        if str(value).strip()
    }
    if not identity or not evidence:
        raise ValueError("Policy must define nonempty identity and evidence decision vocabularies")
    return identity, evidence


def verify_manifest_contract(
    manifest: dict,
    manifest_path: Path,
    policy_path: Path,
    reviewer_a_queue: Path,
    reviewer_b_queue: Path,
) -> dict[str, dict]:
    if manifest.get("prospective_claim_allowed") is not False:
        raise ValueError("Preparation manifest must prohibit prospective claims")
    if manifest.get("automatic_identity_labels_assigned") is not False:
        raise ValueError("Preparation manifest must state that no identity labels were assigned")

    inputs = manifest.get("inputs", {})
    policy_record = inputs.get("policy", {})
    expected_policy_path = resolve_workspace_path(str(policy_record.get("path", "")))
    if policy_path != expected_policy_path:
        raise ValueError("Provided policy is not the policy frozen in the preparation manifest")
    if sha256(policy_path) != str(policy_record.get("sha256", "")):
        raise ValueError("Provided policy hash does not match the preparation manifest")

    outputs = manifest.get("outputs", {})
    records = {
        "reviewer_a_queue": outputs.get("reviewer_a_queue", {}),
        "reviewer_b_queue": outputs.get("reviewer_b_queue", {}),
    }
    for role, queue_path in (
        ("reviewer_a_queue", reviewer_a_queue),
        ("reviewer_b_queue", reviewer_b_queue),
    ):
        record = records[role]
        if not isinstance(record, dict) or not record:
            raise ValueError(f"Preparation manifest is missing {role}")
        expected_path = resolve_workspace_path(str(record.get("path", "")))
        if queue_path != expected_path:
            raise ValueError(f"Provided {role} path differs from the preparation manifest")
        observed_hash = sha256(queue_path)
        if observed_hash != str(record.get("sha256", "")):
            raise ValueError(f"Original {role} hash does not match the preparation manifest")
        try:
            row_count = int(record.get("row_count"))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Preparation manifest has an invalid {role} row count") from exc
        records[role] = {
            "path": relative_path(queue_path),
            "sha256": observed_hash,
            "row_count": row_count,
        }
    records["preparation_manifest"] = {
        "path": relative_path(manifest_path),
        "sha256": sha256(manifest_path),
    }
    records["policy"] = {
        "path": relative_path(policy_path),
        "sha256": sha256(policy_path),
    }
    return records


def validate_queue(
    fields: list[str], rows: list[dict[str, str]], expected_count: int, role: str
) -> dict[str, dict[str, str]]:
    required = {"blind_id", *DECISION_FIELDS}
    missing = sorted(required - set(fields))
    if missing:
        raise ValueError(f"Reviewer {role} queue is missing required fields: {missing}")
    if len(rows) != expected_count:
        raise ValueError(
            f"Reviewer {role} queue row count differs from manifest: "
            f"expected={expected_count} observed={len(rows)}"
        )
    index: dict[str, dict[str, str]] = {}
    for row in rows:
        blind_id = row["blind_id"].strip()
        if not blind_id or blind_id in index:
            raise ValueError(f"Reviewer {role} queue has a missing or duplicate blind_id")
        if any(row[field].strip() for field in DECISION_FIELDS):
            raise ValueError(f"Reviewer {role} original queue already contains decisions: {blind_id}")
        index[blind_id] = row
    return index


def validate_completed_review(
    queue_fields: list[str],
    queue_rows: list[dict[str, str]],
    completed_path: Path,
    role: str,
    allowed_identity: set[str],
    allowed_evidence: set[str],
) -> dict[str, dict[str, str]]:
    completed_fields, completed_rows = load_csv(
        completed_path, f"reviewer {role} completed review"
    )
    if completed_fields != queue_fields:
        raise ValueError(f"Reviewer {role} completed schema differs from the original queue")
    if len(completed_rows) != len(queue_rows):
        raise ValueError(f"Reviewer {role} completed row count differs from the original queue")

    immutable_fields = [field for field in queue_fields if field not in DECISION_FIELDS]
    completed_index: dict[str, dict[str, str]] = {}
    for row_number, (original, completed) in enumerate(
        zip(queue_rows, completed_rows, strict=True), start=1
    ):
        blind_id = completed["blind_id"].strip()
        if blind_id != original["blind_id"].strip():
            raise ValueError(
                f"Reviewer {role} blind_id order changed at row {row_number}: {blind_id}"
            )
        if not blind_id or blind_id in completed_index:
            raise ValueError(f"Reviewer {role} completed review has a duplicate blind_id")
        for field in immutable_fields:
            if completed[field] != original[field]:
                raise ValueError(
                    f"Reviewer {role} changed immutable field {field} for {blind_id}"
                )

        identity = completed["independent_identity_decision"].strip()
        evidence = completed["evidence_type_decision"].strip()
        confidence = completed["review_confidence"].strip().lower()
        rationale = completed["review_rationale"].strip()
        if identity not in allowed_identity:
            raise ValueError(f"Reviewer {role} has invalid identity decision for {blind_id}: {identity}")
        if evidence not in allowed_evidence:
            raise ValueError(f"Reviewer {role} has invalid evidence decision for {blind_id}: {evidence}")
        if confidence not in CONFIDENCE_VALUES:
            raise ValueError(f"Reviewer {role} has invalid confidence for {blind_id}: {confidence}")
        if not rationale:
            raise ValueError(f"Reviewer {role} rationale is blank for {blind_id}")
        completed_index[blind_id] = {
            "identity": identity,
            "evidence": evidence,
            "confidence": confidence,
        }
    return completed_index


def build_agreement_rows(
    reviewer_a: dict[str, dict[str, str]], reviewer_b: dict[str, dict[str, str]]
) -> tuple[list[dict[str, object]], dict[str, int]]:
    if set(reviewer_a) != set(reviewer_b):
        raise ValueError("Reviewer A/B blind_id universes differ")

    rows: list[dict[str, object]] = []
    counts: Counter[str] = Counter()
    for blind_id in sorted(reviewer_a):
        a = reviewer_a[blind_id]
        b = reviewer_b[blind_id]
        identity_agreement = a["identity"] == b["identity"]
        evidence_agreement = a["evidence"] == b["evidence"]
        exact_agreement = identity_agreement and evidence_agreement
        high_confidence_exact = (
            exact_agreement and a["confidence"] == "high" and b["confidence"] == "high"
        )
        needs_adjudication = not high_confidence_exact
        counts["reviewed"] += 1
        counts["exact_identity_agreement"] += int(identity_agreement)
        counts["exact_evidence_agreement"] += int(evidence_agreement)
        counts["exact_identity_and_evidence_agreement"] += int(exact_agreement)
        counts["high_confidence_exact_agreement"] += int(high_confidence_exact)
        counts["needs_human_adjudication"] += int(needs_adjudication)
        rows.append(
            {
                "blind_id": blind_id,
                "reviewer_a_identity_decision": a["identity"],
                "reviewer_a_evidence_type_decision": a["evidence"],
                "reviewer_a_review_confidence": a["confidence"],
                "reviewer_b_identity_decision": b["identity"],
                "reviewer_b_evidence_type_decision": b["evidence"],
                "reviewer_b_review_confidence": b["confidence"],
                "exact_identity_agreement": bool_text(identity_agreement),
                "exact_evidence_agreement": bool_text(evidence_agreement),
                "high_confidence_exact_agreement": bool_text(high_confidence_exact),
                "needs_human_adjudication": bool_text(needs_adjudication),
            }
        )
    return rows, dict(counts)


def reconcile(
    *,
    reviewer_a_queue: str | Path,
    reviewer_b_queue: str | Path,
    reviewer_a_completed: str | Path,
    reviewer_b_completed: str | Path,
    preparation_manifest: str | Path,
    policy: str | Path,
    output_directory: str | Path,
) -> dict:
    paths = {
        "reviewer_a_queue": resolve_workspace_path(reviewer_a_queue),
        "reviewer_b_queue": resolve_workspace_path(reviewer_b_queue),
        "reviewer_a_completed": resolve_workspace_path(reviewer_a_completed),
        "reviewer_b_completed": resolve_workspace_path(reviewer_b_completed),
        "preparation_manifest": resolve_workspace_path(preparation_manifest),
        "policy": resolve_workspace_path(policy),
    }
    if paths["reviewer_a_queue"] == paths["reviewer_b_queue"]:
        raise ValueError("Reviewer A/B original queue paths must differ")
    if paths["reviewer_a_completed"] == paths["reviewer_b_completed"]:
        raise ValueError("Reviewer A/B completed review paths must differ")
    for path in paths.values():
        reject_blind_mapping_input(path)
        if not path.is_file():
            raise FileNotFoundError(path)

    output_dir = resolve_workspace_path(output_directory)
    if output_dir == ROOT.resolve():
        raise ValueError("Output directory cannot be the workspace root")
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite immutable output directory: {output_dir}")

    manifest = load_json_object(paths["preparation_manifest"], "preparation manifest")
    policy_payload = load_json_object(paths["policy"], "policy")
    allowed_identity, allowed_evidence = policy_decisions(policy_payload)
    input_records = verify_manifest_contract(
        manifest,
        paths["preparation_manifest"],
        paths["policy"],
        paths["reviewer_a_queue"],
        paths["reviewer_b_queue"],
    )

    fields_a, queue_a = load_csv(paths["reviewer_a_queue"], "reviewer A original queue")
    fields_b, queue_b = load_csv(paths["reviewer_b_queue"], "reviewer B original queue")
    if fields_a != fields_b:
        raise ValueError("Reviewer A/B original queue schemas differ")
    index_a = validate_queue(
        fields_a, queue_a, input_records["reviewer_a_queue"]["row_count"], "A"
    )
    index_b = validate_queue(
        fields_b, queue_b, input_records["reviewer_b_queue"]["row_count"], "B"
    )
    if set(index_a) != set(index_b):
        raise ValueError("Reviewer A/B original queue blind_id universes differ")

    completed_a = validate_completed_review(
        fields_a,
        queue_a,
        paths["reviewer_a_completed"],
        "A",
        allowed_identity,
        allowed_evidence,
    )
    completed_b = validate_completed_review(
        fields_b,
        queue_b,
        paths["reviewer_b_completed"],
        "B",
        allowed_identity,
        allowed_evidence,
    )
    agreement_rows, counts = build_agreement_rows(completed_a, completed_b)
    reviewed_count = counts["reviewed"]

    input_records["reviewer_a_completed"] = {
        "path": relative_path(paths["reviewer_a_completed"]),
        "sha256": sha256(paths["reviewer_a_completed"]),
        "row_count": len(queue_a),
    }
    input_records["reviewer_b_completed"] = {
        "path": relative_path(paths["reviewer_b_completed"]),
        "sha256": sha256(paths["reviewer_b_completed"]),
        "row_count": len(queue_b),
    }

    agreement_payload = render_csv(agreement_rows, AGREEMENT_FIELDS)
    agreement_sha256 = hashlib.sha256(agreement_payload).hexdigest()
    final_agreement_path = output_dir / "agreement_by_blind_id.csv"
    final_summary_path = output_dir / "summary.json"
    summary = {
        "step": "step16i_reconcile_ai_sensitivity_reviews",
        "version": "2026-07-16-step16i-ai-sensitivity-reconciliation-v1",
        "status": "completed_ai_sensitivity_reconciliation",
        "scope": "ai_sensitivity_only_not_human_gold",
        "automatic_labels_created": False,
        "step5_modified": False,
        "prospective_claim_allowed": False,
        "join_key": "blind_id_only",
        "validation_contract": {
            "original_queue_hashes_verified_against_preparation_manifest": True,
            "completed_schema_row_count_and_blind_id_order_verified": True,
            "all_non_decision_cell_values_verified_unchanged": True,
            "blind_mapping_read": False,
            "label_inference_performed": False,
        },
        "reviewed_row_count": reviewed_count,
        "counts": counts,
        "agreement_rates": {
            "exact_identity_agreement": counts["exact_identity_agreement"] / reviewed_count,
            "exact_evidence_agreement": counts["exact_evidence_agreement"] / reviewed_count,
            "exact_identity_and_evidence_agreement": (
                counts["exact_identity_and_evidence_agreement"] / reviewed_count
            ),
            "high_confidence_exact_agreement": (
                counts["high_confidence_exact_agreement"] / reviewed_count
            ),
            "needs_human_adjudication": counts["needs_human_adjudication"] / reviewed_count,
        },
        "decision_counts": {
            "reviewer_a_identity": dict(
                sorted(Counter(row["identity"] for row in completed_a.values()).items())
            ),
            "reviewer_a_evidence": dict(
                sorted(Counter(row["evidence"] for row in completed_a.values()).items())
            ),
            "reviewer_b_identity": dict(
                sorted(Counter(row["identity"] for row in completed_b.values()).items())
            ),
            "reviewer_b_evidence": dict(
                sorted(Counter(row["evidence"] for row in completed_b.values()).items())
            ),
        },
        "input_artifacts": input_records,
        "output_artifacts": {
            "agreement_by_blind_id": {
                "path": relative_path(final_agreement_path),
                "sha256": agreement_sha256,
                "row_count": reviewed_count,
            }
        },
    }
    summary["summary_payload_sha256"] = canonical_hash(summary)
    summary_payload = (json.dumps(summary, indent=2, ensure_ascii=False) + "\n").encode("utf-8")

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging_dir = output_dir.with_name(
        f".{output_dir.name}.incomplete.{os.getpid()}.{uuid.uuid4().hex}"
    )
    if staging_dir.exists():
        raise FileExistsError(f"Unexpected Step16I staging directory exists: {staging_dir}")
    staging_dir.mkdir(parents=False, exist_ok=False)
    try:
        (staging_dir / final_agreement_path.name).write_bytes(agreement_payload)
        (staging_dir / final_summary_path.name).write_bytes(summary_payload)
        if output_dir.exists():
            raise FileExistsError(
                f"Refusing to overwrite immutable output directory: {output_dir}"
            )
        staging_dir.rename(output_dir)
    except Exception:
        if staging_dir.exists():
            shutil.rmtree(staging_dir)
        raise
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reviewer-a-queue", required=True)
    parser.add_argument("--reviewer-b-queue", required=True)
    parser.add_argument("--reviewer-a-completed", required=True)
    parser.add_argument("--reviewer-b-completed", required=True)
    parser.add_argument("--preparation-manifest", required=True)
    parser.add_argument("--policy", required=True)
    parser.add_argument("--output-directory", required=True)
    args = parser.parse_args()

    summary = reconcile(
        reviewer_a_queue=args.reviewer_a_queue,
        reviewer_b_queue=args.reviewer_b_queue,
        reviewer_a_completed=args.reviewer_a_completed,
        reviewer_b_completed=args.reviewer_b_completed,
        preparation_manifest=args.preparation_manifest,
        policy=args.policy,
        output_directory=args.output_directory,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
