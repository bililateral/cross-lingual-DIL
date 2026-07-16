#!/usr/bin/env python3
"""Unblind a frozen Codex adjudication into an isolated retrospective Dev2 table."""

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

OUTPUT_FIELDS = (
    "blind_id",
    "pair_uid",
    "candidate_component_id",
    "candidate_component_seller_count",
    "candidate_component_pair_count",
    "seller_uid_left",
    "seller_uid_right",
    "identity_decision",
    "binary_label",
    "evidence_type",
    "adjudication_confidence",
    "annotation_origin",
    "dataset_owner_authorized",
    "human_verified_per_row",
    "retrospective_development_only",
    "prospective_final_eligible",
    "step5_supervision_eligible",
)


def workspace_path(value: str | Path) -> Path:
    path = Path(value)
    resolved = (path if path.is_absolute() else ROOT / path).resolve()
    try:
        resolved.relative_to(ROOT.resolve())
    except ValueError as exc:
        raise ValueError(f"Path must remain inside the workspace: {resolved}") from exc
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
    if not fields or not rows or any(None in row for row in rows):
        raise ValueError(f"Invalid CSV: {path}")
    return fields, rows


def render_csv(rows: list[dict[str, object]]) -> bytes:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(
        buffer,
        fieldnames=list(OUTPUT_FIELDS),
        extrasaction="raise",
        lineterminator="\n",
    )
    writer.writeheader()
    writer.writerows(rows)
    return ("\ufeff" + buffer.getvalue()).encode("utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--adjudication-summary", required=True)
    parser.add_argument("--blind-mapping", required=True)
    parser.add_argument("--preparation-manifest", required=True)
    parser.add_argument("--output-directory", required=True)
    args = parser.parse_args()

    adjudication_summary_path = workspace_path(args.adjudication_summary)
    blind_mapping_path = workspace_path(args.blind_mapping)
    preparation_manifest_path = workspace_path(args.preparation_manifest)
    output_directory = workspace_path(args.output_directory)
    if output_directory.exists():
        raise ValueError(f"Refusing to overwrite Dev2 materialization: {output_directory}")

    adjudication_summary = json.loads(adjudication_summary_path.read_text(encoding="utf-8"))
    if adjudication_summary.get("status") != "completed_owner_authorized_codex_adjudication":
        raise ValueError("Adjudication summary is not a completed Codex adjudication")
    if adjudication_summary.get("human_verified_per_row") is not False:
        raise ValueError("Unexpected human-verification claim in Codex adjudication")
    if adjudication_summary.get("prospective_claim_allowed") is not False:
        raise ValueError("Codex-adjudicated Dev2 cannot be prospective")

    adjudication_record = adjudication_summary.get("outputs", {}).get("adjudication_by_blind_id", {})
    adjudication_path = workspace_path(str(adjudication_record.get("path", "")))
    if sha256(adjudication_path) != str(adjudication_record.get("sha256", "")):
        raise ValueError("Frozen adjudication CSV hash mismatch")

    preparation_manifest = json.loads(preparation_manifest_path.read_text(encoding="utf-8"))
    mapping_record = preparation_manifest.get("outputs", {}).get("blind_mapping", {})
    expected_mapping_path = workspace_path(str(mapping_record.get("path", "")))
    if blind_mapping_path != expected_mapping_path:
        raise ValueError("Blind mapping differs from the preparation manifest")
    if sha256(blind_mapping_path) != str(mapping_record.get("sha256", "")):
        raise ValueError("Blind mapping hash mismatch")

    adjudication_fields, adjudication_rows = load_csv(adjudication_path)
    mapping_fields, mapping_rows = load_csv(blind_mapping_path)
    required_adjudication = {
        "blind_id",
        "adjudicated_identity_decision",
        "adjudicated_evidence_type",
        "adjudication_confidence",
        "annotation_origin",
        "dataset_owner_authorized",
        "human_verified_per_row",
        "prospective_final_eligible",
        "step5_supervision_eligible",
    }
    required_mapping = {
        "blind_id",
        "pair_uid",
        "candidate_component_id",
        "candidate_component_seller_count",
        "candidate_component_pair_count",
        "seller_uid_left",
        "seller_uid_right",
        "retrospective_development_only",
        "prospective_final_eligible",
        "automatic_label_assigned",
    }
    if required_adjudication - set(adjudication_fields):
        raise ValueError("Frozen adjudication CSV is missing required fields")
    if required_mapping - set(mapping_fields):
        raise ValueError("Blind mapping is missing required fields")

    adjudication_index = {row["blind_id"].strip(): row for row in adjudication_rows}
    mapping_index = {row["blind_id"].strip(): row for row in mapping_rows}
    if len(adjudication_index) != len(adjudication_rows) or len(mapping_index) != len(mapping_rows):
        raise ValueError("Duplicate blind_id in adjudication or mapping")
    if set(adjudication_index) != set(mapping_index):
        raise ValueError("Adjudication and blind mapping universes differ")

    output_rows: list[dict[str, object]] = []
    for blind_id in sorted(adjudication_index):
        adjudication = adjudication_index[blind_id]
        mapping = mapping_index[blind_id]
        if mapping["retrospective_development_only"].strip() != "1":
            raise ValueError(f"Mapping row is not retrospective-only: {blind_id}")
        if mapping["prospective_final_eligible"].strip() != "0":
            raise ValueError(f"Mapping row unexpectedly claims prospective eligibility: {blind_id}")
        if mapping["automatic_label_assigned"].strip() != "0":
            raise ValueError(f"Mapping row already had an automatic label: {blind_id}")
        if adjudication["dataset_owner_authorized"].strip() != "true":
            raise ValueError(f"Adjudication is not owner-authorized: {blind_id}")
        if adjudication["human_verified_per_row"].strip() != "false":
            raise ValueError(f"Adjudication falsely claims per-row human verification: {blind_id}")
        if adjudication["prospective_final_eligible"].strip() != "false":
            raise ValueError(f"Adjudication unexpectedly claims prospective eligibility: {blind_id}")
        if adjudication["step5_supervision_eligible"].strip() != "false":
            raise ValueError(f"Adjudication unexpectedly claims Step5 eligibility: {blind_id}")

        identity = adjudication["adjudicated_identity_decision"].strip()
        binary_label = "1" if identity == "same_controller" else "0" if identity == "different_controller" else ""
        output_rows.append(
            {
                "blind_id": blind_id,
                "pair_uid": mapping["pair_uid"],
                "candidate_component_id": mapping["candidate_component_id"],
                "candidate_component_seller_count": mapping["candidate_component_seller_count"],
                "candidate_component_pair_count": mapping["candidate_component_pair_count"],
                "seller_uid_left": mapping["seller_uid_left"],
                "seller_uid_right": mapping["seller_uid_right"],
                "identity_decision": identity,
                "binary_label": binary_label,
                "evidence_type": adjudication["adjudicated_evidence_type"].strip(),
                "adjudication_confidence": adjudication["adjudication_confidence"].strip(),
                "annotation_origin": adjudication["annotation_origin"].strip(),
                "dataset_owner_authorized": "true",
                "human_verified_per_row": "false",
                "retrospective_development_only": "true",
                "prospective_final_eligible": "false",
                "step5_supervision_eligible": "false",
            }
        )

    identity_counts = Counter(str(row["identity_decision"]) for row in output_rows)
    evidence_counts = Counter(str(row["evidence_type"]) for row in output_rows)
    binary_rows = [row for row in output_rows if row["binary_label"] != ""]
    csv_bytes = render_csv(output_rows)
    output_csv_hash = hashlib.sha256(csv_bytes).hexdigest()
    summary = {
        "status": "materialized_retrospective_dev2_owner_authorized_codex_adjudication",
        "scope": "retrospective_internal_dev_sensitivity_only",
        "row_count": len(output_rows),
        "identity_counts": dict(sorted(identity_counts.items())),
        "evidence_type_counts": dict(sorted(evidence_counts.items())),
        "binary_evaluation_row_count": len(binary_rows),
        "binary_positive_count": identity_counts["same_controller"],
        "binary_negative_count": identity_counts["different_controller"],
        "binary_positive_prevalence": (
            identity_counts["same_controller"] / len(binary_rows) if binary_rows else None
        ),
        "uncertain_excluded_from_binary_evaluation": identity_counts["uncertain"],
        "annotation_origin": "two_ai_blind_reviews_plus_codex_adjudication_owner_authorized",
        "dataset_owner_authorized": True,
        "human_verified_per_row": False,
        "prospective_claim_allowed": False,
        "step5_labels_created_or_modified": False,
        "paper_primary_benchmark_eligible": False,
        "inputs": {
            "adjudication_summary": {
                "path": relative(adjudication_summary_path),
                "sha256": sha256(adjudication_summary_path),
            },
            "adjudication_by_blind_id": {
                "path": relative(adjudication_path),
                "sha256": sha256(adjudication_path),
            },
            "preparation_manifest": {
                "path": relative(preparation_manifest_path),
                "sha256": sha256(preparation_manifest_path),
            },
            "blind_mapping": {
                "path": relative(blind_mapping_path),
                "sha256": sha256(blind_mapping_path),
            },
        },
        "outputs": {
            "retrospective_dev2_labels": {
                "path": relative(output_directory / "retrospective_dev2_labels.csv"),
                "sha256": output_csv_hash,
                "row_count": len(output_rows),
            }
        },
    }

    temp_directory = output_directory.with_name(f".{output_directory.name}.tmp-{uuid.uuid4().hex}")
    temp_directory.mkdir(parents=True)
    try:
        (temp_directory / "retrospective_dev2_labels.csv").write_bytes(csv_bytes)
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
