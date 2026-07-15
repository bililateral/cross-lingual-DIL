#!/usr/bin/env python3
"""Reconcile two score-blind reviews of supplemental profile URL controls."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CANDIDATES = ROOT / "schema" / "step16_v8_profile_url_control_candidates.json"
DEFAULT_REVIEW_INPUT_ROOT = (
    ROOT / "reports" / "step15_v8" / "profile_url_control_review_20260715"
)
DEFAULT_OUTPUT_ROOT = (
    ROOT / "reports" / "step15_v8" / "profile_url_control_review_v3_20260715"
)

ALLOWED_DECISIONS = {
    ("negative", "public_contact_or_url_noise"),
    ("positive", "same_controller_direct_identifier"),
    ("uncertain", "uncertain_insufficient_evidence"),
}


def resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def rel(path: Path) -> str:
    return str(path.resolve().relative_to(ROOT.resolve())).replace("\\", "/")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_hash(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, ensure_ascii=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def render_csv(rows: list[dict], fields: list[str]) -> bytes:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(
        buffer, fieldnames=fields, extrasaction="raise", lineterminator="\n"
    )
    writer.writeheader()
    writer.writerows(rows)
    return ("\ufeff" + buffer.getvalue()).encode("utf-8")


def context_window(text: str, needle: str, radius: int = 220) -> str:
    folded = text.casefold()
    position = folded.find(needle.casefold())
    if position < 0:
        raise ValueError(f"Profile text does not contain reviewed URL literal: {needle}")
    start = max(0, position - radius)
    end = min(len(text), position + len(needle) + radius)
    return " ".join(text[start:end].split())


def load_decisions(path: Path, candidate_ids: set[str], expected_role: str) -> tuple[dict, dict[str, dict]]:
    payload = load_json(path)
    if payload.get("reviewer_role") != expected_role:
        raise ValueError(f"Unexpected reviewer role in {path}")
    if payload.get("model_scores_seen") is not False:
        raise ValueError(f"Reviewer was exposed to model scores: {path}")
    if payload.get("split_assignments_seen") is not False:
        raise ValueError(f"Reviewer was exposed to split assignments: {path}")
    review_lane_id = str(payload.get("review_lane_id", "")).strip()
    if not review_lane_id:
        raise ValueError(f"Missing review lane id: {path}")
    decisions = {}
    for row in payload.get("decisions", []):
        candidate_id = str(row.get("candidate_id", "")).strip()
        reviewer_id = str(row.get("reviewer_id", "")).strip()
        decision = (
            str(row.get("identity_label", "")).strip(),
            str(row.get("evidence_type", "")).strip(),
        )
        confidence = str(row.get("confidence", "")).strip()
        if candidate_id in decisions or decision not in ALLOWED_DECISIONS:
            raise ValueError(f"Invalid or duplicate reviewer decision: {candidate_id}")
        if not reviewer_id:
            raise ValueError(f"Missing per-candidate reviewer id: {candidate_id}")
        if confidence not in {"low", "medium", "high"}:
            raise ValueError(f"Invalid reviewer confidence: {candidate_id}")
        if not str(row.get("notes", "")).strip():
            raise ValueError(f"Reviewer notes are required: {candidate_id}")
        decisions[candidate_id] = dict(row)
    if set(decisions) != candidate_ids:
        raise ValueError(
            f"Reviewer candidate universe differs in {path}: "
            f"missing={sorted(candidate_ids - set(decisions))} "
            f"extra={sorted(set(decisions) - candidate_ids)}"
        )
    return payload, decisions


def write_immutable(path: Path, payload: bytes) -> None:
    if path.exists():
        if path.read_bytes() != payload:
            raise ValueError(f"Refusing to overwrite a different reviewed artifact: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", default=str(DEFAULT_CANDIDATES))
    parser.add_argument(
        "--reviewer-a", default=str(DEFAULT_REVIEW_INPUT_ROOT / "reviewer_lane_a.json")
    )
    parser.add_argument(
        "--reviewer-b", default=str(DEFAULT_REVIEW_INPUT_ROOT / "reviewer_lane_b.json")
    )
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()

    candidate_path = resolve(args.candidates)
    reviewer_a_path = resolve(args.reviewer_a)
    reviewer_b_path = resolve(args.reviewer_b)
    output_root = resolve(args.output_root)

    spec = load_json(candidate_path)
    if spec.get("candidate_selection_used_model_scores") is not False:
        raise ValueError("Supplemental URL candidates must be score blind")
    if spec.get("split_assignments_exposed_to_reviewers") is not False:
        raise ValueError("Supplemental URL reviewers must be split blind")
    profile_path = resolve(spec["source_profiles"])
    profiles = {row["seller_uid"]: row for row in load_jsonl(profile_path)}

    candidates = spec.get("candidates", [])
    candidate_ids = {str(row.get("candidate_id", "")).strip() for row in candidates}
    if not candidates or len(candidate_ids) != len(candidates) or "" in candidate_ids:
        raise ValueError("Supplemental URL candidate ids are empty or duplicated")
    reviewer_a, decisions_a = load_decisions(
        reviewer_a_path, candidate_ids, "blind_evidence_reviewer_a"
    )
    reviewer_b, decisions_b = load_decisions(
        reviewer_b_path, candidate_ids, "blind_evidence_reviewer_b"
    )
    if reviewer_a["review_lane_id"].casefold() == reviewer_b["review_lane_id"].casefold():
        raise ValueError("Supplemental URL reviews must use distinct review lanes")

    evidence_rows = []
    resolved_rows = []
    statuses = Counter()
    seen_pairs = set()
    for candidate in sorted(candidates, key=lambda row: row["candidate_id"]):
        candidate_id = candidate["candidate_id"]
        left_uid = candidate["seller_uid_left"]
        right_uid = candidate["seller_uid_right"]
        if left_uid == right_uid or left_uid not in profiles or right_uid not in profiles:
            raise ValueError(f"Invalid supplemental URL seller pair: {candidate_id}")
        pair_key = tuple(sorted((left_uid, right_uid)))
        if pair_key in seen_pairs:
            raise ValueError(f"Duplicate supplemental URL seller pair: {candidate_id}")
        seen_pairs.add(pair_key)
        literal = str(candidate["shared_url_literal"]).strip().casefold()
        if not literal:
            raise ValueError(f"Missing shared URL literal: {candidate_id}")
        left_context = context_window(profiles[left_uid]["profile_text"], literal)
        right_context = context_window(profiles[right_uid]["profile_text"], literal)
        a = decisions_a[candidate_id]
        b = decisions_b[candidate_id]
        if a["reviewer_id"].casefold() == b["reviewer_id"].casefold():
            raise ValueError(
                f"Supplemental URL candidate lacks two independent reviewers: {candidate_id}"
            )
        decision_a = (a["identity_label"], a["evidence_type"], a["confidence"])
        decision_b = (b["identity_label"], b["evidence_type"], b["confidence"])
        accepted = decision_a == decision_b == (
            "negative",
            "public_contact_or_url_noise",
            "high",
        )
        status = "resolved_high_confidence" if accepted else "excluded_no_high_confidence_agreement"
        statuses[status] += 1
        evidence_rows.append(
            {
                "candidate_id": candidate_id,
                "seller_uid_left": left_uid,
                "seller_uid_right": right_uid,
                "shared_url_literal": literal,
                "left_context": left_context,
                "right_context": right_context,
                "reviewer_a_decision": "|".join(decision_a),
                "reviewer_b_decision": "|".join(decision_b),
                "resolution_status": status,
            }
        )
        if accepted:
            resolved_rows.append(
                {
                    "review_candidate_uid": candidate_id,
                    "queue_kind": "risky_only_public_noise",
                    "seller_uid_left": left_uid,
                    "seller_uid_right": right_uid,
                    "shared_identifier_types": "external_url",
                    "shared_identifier_values": literal,
                    "status": status,
                    "identity_label": "negative",
                    "evidence_type": "public_contact_or_url_noise",
                    "review_confidence": "high",
                    "reviewer_ids": f"{a['reviewer_id']}+{b['reviewer_id']}",
                    "review_reason": f"{a['notes']} | {b['notes']}",
                    "selection_uid": candidate_id,
                }
            )

    evidence_fields = [
        "candidate_id",
        "seller_uid_left",
        "seller_uid_right",
        "shared_url_literal",
        "left_context",
        "right_context",
        "reviewer_a_decision",
        "reviewer_b_decision",
        "resolution_status",
    ]
    resolved_fields = [
        "review_candidate_uid",
        "queue_kind",
        "seller_uid_left",
        "seller_uid_right",
        "shared_identifier_types",
        "shared_identifier_values",
        "status",
        "identity_label",
        "evidence_type",
        "review_confidence",
        "reviewer_ids",
        "review_reason",
        "selection_uid",
    ]
    evidence_payload = render_csv(evidence_rows, evidence_fields)
    resolved_payload = render_csv(resolved_rows, resolved_fields)
    evidence_path = output_root / "profile_url_candidate_evidence.csv"
    resolved_path = output_root / "profile_url_resolved_controls.csv"
    summary_path = output_root / "profile_url_review_summary.json"
    summary = {
        "step": "step16_reconcile_v8_profile_url_reviews",
        "version": spec["version"],
        "producer": rel(Path(__file__).resolve()),
        "producer_sha256": sha256(Path(__file__).resolve()),
        "model_scores_read": False,
        "split_assignments_read": False,
        "accepted_control_scope": spec["accepted_control_scope"],
        "inputs": {
            rel(path): sha256(path)
            for path in (candidate_path, reviewer_a_path, reviewer_b_path, profile_path)
        },
        "candidate_count": len(candidates),
        "accepted_count": len(resolved_rows),
        "status_counts": dict(sorted(statuses.items())),
        "outputs": {
            "candidate_evidence": {
                "path": rel(evidence_path),
                "sha256": hashlib.sha256(evidence_payload).hexdigest(),
            },
            "resolved_controls": {
                "path": rel(resolved_path),
                "sha256": hashlib.sha256(resolved_payload).hexdigest(),
            },
        },
        "summary_hash_scope": "all_fields_except_summary_sha256",
    }
    summary["summary_sha256"] = canonical_hash(summary)
    summary_payload = (json.dumps(summary, indent=2, ensure_ascii=False) + "\n").encode("utf-8")
    diagnostics = {
        "status": "pass",
        "candidate_count": len(candidates),
        "accepted_count": len(resolved_rows),
        "status_counts": dict(sorted(statuses.items())),
        "summary": rel(summary_path),
    }
    if args.check_only:
        print(json.dumps(diagnostics, indent=2, ensure_ascii=False))
        return
    write_immutable(evidence_path, evidence_payload)
    write_immutable(resolved_path, resolved_payload)
    write_immutable(summary_path, summary_payload)
    print(json.dumps(diagnostics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
