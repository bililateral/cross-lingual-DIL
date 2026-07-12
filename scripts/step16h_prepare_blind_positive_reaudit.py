#!/usr/bin/env python3
"""Prepare two independently ordered, score-blind Step16H review queues."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
from collections import defaultdict, deque
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_POLICY = ROOT / "schema" / "step16h_blind_positive_reaudit_policy.json"


def resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def load_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def deterministic_rank(pair_uid: str, seed: str) -> str:
    return hashlib.sha256(f"{seed}|{pair_uid}".encode("utf-8")).hexdigest()


def split_contact_tokens(value: str) -> list[str]:
    return [token.strip().lower() for token in str(value or "").split("||") if token.strip()]


def build_signal_indexes(signal_rows: list[dict]) -> tuple[dict[tuple[str, str], list[dict]], dict[str, set[str]]]:
    by_seller_token: dict[tuple[str, str], list[dict]] = defaultdict(list)
    sellers_by_token: dict[str, set[str]] = defaultdict(set)
    for row in signal_rows:
        token = f"{row.get('contact_type', '')}:{str(row.get('normalized_value', '')).lower()}"
        seller_uid = str(row.get("seller_uid", ""))
        if not seller_uid or token.endswith(":"):
            continue
        by_seller_token[(seller_uid, token)].append(row)
        if row.get("direct_identity_eligible") == "1":
            sellers_by_token[token].add(seller_uid)
    return by_seller_token, sellers_by_token


def raw_occurrence_payload(signals: list[dict]) -> list[dict]:
    return [
        {
            "source_field": row.get("source_field", ""),
            "raw_value": row.get("raw_value", ""),
            "context": row.get("context", ""),
            "title_snippet": row.get("title_snippet", ""),
            "description_snippet": row.get("description_snippet", ""),
        }
        for row in signals[:6]
    ]


def contact_occurrence_evidence(
    candidate: dict,
    by_seller_token: dict[tuple[str, str], list[dict]],
) -> str:
    payload = []
    for token in split_contact_tokens(candidate.get("shared_contact_values", "")):
        payload.append(
            {
                "token": token,
                "left_occurrences": raw_occurrence_payload(
                    by_seller_token.get((candidate["seller_uid_left"], token), [])
                ),
                "right_occurrences": raw_occurrence_payload(
                    by_seller_token.get((candidate["seller_uid_right"], token), [])
                ),
            }
        )
    return json.dumps(payload, ensure_ascii=False, sort_keys=True)


def build_identity_adjacency(sellers_by_token: dict[str, set[str]], max_token_sellers: int = 5) -> dict[str, list[tuple[str, str]]]:
    adjacency: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for token, sellers in sellers_by_token.items():
        ordered = sorted(sellers)
        if len(ordered) < 2 or len(ordered) > max_token_sellers:
            continue
        for index, left in enumerate(ordered):
            for right in ordered[index + 1 :]:
                adjacency[left].append((right, token))
                adjacency[right].append((left, token))
    return adjacency


def component_candidate_path(
    left: str,
    right: str,
    adjacency: dict[str, list[tuple[str, str]]],
    by_seller_token: dict[tuple[str, str], list[dict]],
    max_hops: int = 3,
) -> list[dict]:
    queue = deque([(left, [])])
    visited = {left}
    while queue:
        node, path = queue.popleft()
        if len(path) >= max_hops:
            continue
        for neighbor, token in sorted(adjacency.get(node, [])):
            if neighbor in visited:
                continue
            next_path = [
                *path,
                {
                    "seller_uid_left": node,
                    "seller_uid_right": neighbor,
                    "shared_token": token,
                    "left_occurrences": raw_occurrence_payload(
                        by_seller_token.get((node, token), [])
                    ),
                    "right_occurrences": raw_occurrence_payload(
                        by_seller_token.get((neighbor, token), [])
                    ),
                },
            ]
            if neighbor == right:
                return next_path
            visited.add(neighbor)
            queue.append((neighbor, next_path))
    return []


def blinded_row(
    candidate: dict,
    label_row: dict,
    review_index: int,
    review_key_field: str = "pair_uid",
    review_key: str | None = None,
    raw_contact_occurrences_json: str = "",
    component_candidate_path_json: str = "",
) -> dict:
    row = {
        "review_index": review_index,
        review_key_field: review_key or candidate["pair_uid"],
        "split_name": label_row["split_name"],
        "source_market_raw_left": candidate["source_market_raw_left"],
        "source_market_raw_right": candidate["source_market_raw_right"],
        "source_seller_raw_left": candidate["source_seller_raw_left"],
        "source_seller_raw_right": candidate["source_seller_raw_right"],
        "alias_normalized_left": candidate.get("alias_normalized_left", ""),
        "alias_normalized_right": candidate.get("alias_normalized_right", ""),
        "alias_relation": candidate.get("alias_relation", ""),
        "item_count_left": candidate.get("item_count_left", ""),
        "item_count_right": candidate.get("item_count_right", ""),
        "shared_contact_count": candidate.get("shared_contact_count", ""),
        "shared_contact_types": candidate.get("shared_contact_types", ""),
        "shared_contact_values": candidate.get("shared_contact_values", ""),
        "shared_pgp_fingerprint_count": candidate.get("shared_pgp_fingerprint_count", ""),
        "shared_pgp_fingerprint_values": candidate.get("shared_pgp_fingerprint_values", ""),
        "pgp_alias_hit_count_left": candidate.get("pgp_alias_hit_count_left", ""),
        "pgp_alias_hit_count_right": candidate.get("pgp_alias_hit_count_right", ""),
        "shared_title_count": candidate.get("shared_title_count", ""),
        "shared_title_values": candidate.get("shared_title_values", ""),
        "shared_description_count": candidate.get("shared_description_count", ""),
        "shared_description_values": candidate.get("shared_description_values", ""),
        "shared_category_count": candidate.get("shared_category_count", ""),
        "shared_category_values": candidate.get("shared_category_values", ""),
        "left_preview": candidate.get("left_preview", ""),
        "right_preview": candidate.get("right_preview", ""),
        "raw_contact_occurrences_json": raw_contact_occurrences_json,
        "component_candidate_path_json": component_candidate_path_json,
        "independent_decision": "",
        "seller_facing_direct_evidence": "",
        "alternative_explanation_ruled_out": "",
        "review_confidence": "",
        "review_rationale": "",
    }
    if not raw_contact_occurrences_json:
        row.pop("raw_contact_occurrences_json")
    if not component_candidate_path_json:
        row.pop("component_candidate_path_json")
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", default=str(DEFAULT_POLICY))
    args = parser.parse_args()
    policy_path = resolve(args.policy)
    policy = json.loads(policy_path.read_text(encoding="utf-8"))
    input_cfg = policy["inputs"]
    positive_path = resolve(input_cfg["positive_reaudit"])
    labels_path = resolve(input_cfg["frozen_labels"])
    candidates_path = resolve(input_cfg["step4_candidates"])
    signal_path = resolve(input_cfg["step3_item_signals"]) if input_cfg.get("step3_item_signals") else None
    positive_rows = load_csv(positive_path)
    labels = load_csv(labels_path)
    candidates = load_csv(candidates_path)
    signal_rows = load_csv(signal_path) if signal_path else []
    by_seller_token, sellers_by_token = build_signal_indexes(signal_rows)
    identity_adjacency = build_identity_adjacency(
        sellers_by_token,
        int(policy.get("component_candidate_path", {}).get("max_token_sellers", 5)),
    )
    if len(positive_rows) != int(policy["expected_positive_candidate_count"]):
        raise ValueError(
            f"Step16H expected {policy['expected_positive_candidate_count']} positive candidates, "
            f"found {len(positive_rows)}"
        )
    label_index = {row["pair_uid"]: row for row in labels}
    candidate_index = {row["pair_uid"]: row for row in candidates}
    if len(label_index) != len(labels) or len(candidate_index) != len(candidates):
        raise ValueError("Step16H inputs contain duplicate pair_uid values")
    positive_uids = {row["pair_uid"] for row in positive_rows}
    for pair_uid in sorted(positive_uids):
        label_row = label_index.get(pair_uid)
        if label_row is None or label_row.get("review_label") != "positive":
            raise ValueError(f"Step16H positive candidate is not positive in the frozen labels: {pair_uid}")
        if pair_uid not in candidate_index:
            raise ValueError(f"Step16H positive candidate lacks raw Step4 evidence: {pair_uid}")
    control_cfg = policy["negative_control_selection"]
    control_seed = str(control_cfg["selection_seed"])
    controls: list[dict] = []
    for split_name, target in control_cfg["split_targets"].items():
        eligible = [
            row
            for row in labels
            if row.get("split_name") == split_name
            and row.get("review_label") == "negative"
            and row.get("usable_for_supervision") == "1"
            and row.get("usable_for_core_transfer") == "1"
            and row["pair_uid"] not in positive_uids
            and row["pair_uid"] in candidate_index
        ]
        eligible.sort(key=lambda row: deterministic_rank(row["pair_uid"], control_seed))
        if len(eligible) < int(target):
            raise ValueError(
                f"Step16H has too few eligible negative controls for {split_name}: "
                f"required={target} available={len(eligible)}"
            )
        controls.extend(eligible[: int(target)])
    if len(controls) != int(policy["negative_control_count"]):
        raise ValueError("Step16H negative control count does not match policy")
    selected_label_rows = [label_index[pair_uid] for pair_uid in sorted(positive_uids)] + controls
    selected_uids = [row["pair_uid"] for row in selected_label_rows]
    if len(selected_uids) != len(set(selected_uids)) or len(selected_uids) != int(policy["expected_row_count"]):
        raise ValueError("Step16H review universe is not the expected unique pair set")
    outputs = policy["outputs"]
    review_key_field = str(policy.get("review_key_field", "pair_uid"))
    blind_seed = str(policy.get("blind_id_seed", "step16h-review-key"))
    review_key_by_uid = {
        pair_uid: (
            f"blind_{deterministic_rank(pair_uid, blind_seed)[:20]}"
            if review_key_field == "blind_id"
            else pair_uid
        )
        for pair_uid in selected_uids
    }
    mapping_path = resolve(outputs["blind_mapping"]) if outputs.get("blind_mapping") else None
    if mapping_path is not None:
        mapping_rows = [
            {
                review_key_field: review_key_by_uid[row["pair_uid"]],
                "pair_uid": row["pair_uid"],
                "split_name": row["split_name"],
                "reference_subset": (
                    "positive_candidate" if row["pair_uid"] in positive_uids else "negative_control"
                ),
            }
            for row in sorted(selected_label_rows, key=lambda value: value["pair_uid"])
        ]
        write_csv(mapping_path, mapping_rows)
    queue_paths = {
        "reviewer_a": resolve(outputs["reviewer_a_queue"]),
        "reviewer_b": resolve(outputs["reviewer_b_queue"]),
    }
    for reviewer, seed in (("reviewer_a", 2026071101), ("reviewer_b", 2026071102)):
        if queue_paths[reviewer].exists() and any(
            row.get("independent_decision", "").strip() for row in load_csv(queue_paths[reviewer])
        ):
            raise ValueError(
                f"Refusing to overwrite a completed Step16H reviewer queue: {queue_paths[reviewer]}"
            )
        ordered = list(selected_label_rows)
        random.Random(seed).shuffle(ordered)
        write_csv(
            queue_paths[reviewer],
            [
                blinded_row(
                    candidate_index[row["pair_uid"]],
                    row,
                    idx + 1,
                    review_key_field=review_key_field,
                    review_key=review_key_by_uid[row["pair_uid"]],
                    raw_contact_occurrences_json=(
                        contact_occurrence_evidence(candidate_index[row["pair_uid"]], by_seller_token)
                        if signal_path
                        else ""
                    ),
                    component_candidate_path_json=(
                        json.dumps(
                            component_candidate_path(
                                candidate_index[row["pair_uid"]]["seller_uid_left"],
                                candidate_index[row["pair_uid"]]["seller_uid_right"],
                                identity_adjacency,
                                by_seller_token,
                                int(policy.get("component_candidate_path", {}).get("max_hops", 3)),
                            ),
                            ensure_ascii=False,
                            sort_keys=True,
                        )
                        if signal_path
                        else ""
                    ),
                )
                for idx, row in enumerate(ordered)
            ],
        )
    manifest = {
        "step": "step16h_prepare_blind_positive_reaudit",
        "policy": str(policy_path.relative_to(ROOT)),
        "policy_version": policy["version"],
        "inputs": [
            {"path": str(path.relative_to(ROOT)), "sha256": sha256(path)}
            for path in (positive_path, labels_path, candidates_path, *([signal_path] if signal_path else []))
        ],
        "row_count": len(selected_uids),
        "concealed_reference_counts": {
            "positive_candidates": len(positive_uids),
            "negative_controls": len(controls),
        },
        "concealed_pair_universe_sha256": hashlib.sha256(
            "\n".join(sorted(selected_uids)).encode("utf-8")
        ).hexdigest(),
        "old_decision_fields_removed": [
            "review_label",
            "evidence_type",
            "paper_evidence_tier",
            "recommended_use",
            "confidence",
            "needs_manual_recheck",
            "risk_flags",
            "rationale",
            "reviewer_id",
            "review_notes",
        ],
        "model_or_graph_score_fields_present": False,
        "prior_label_or_conclusion_fields_present": False,
        "raw_step4_previews_present": True,
        "raw_item_contact_occurrences_present": bool(signal_path),
        "opaque_review_keys": review_key_field == "blind_id",
        "blind_mapping": str(mapping_path.relative_to(ROOT)) if mapping_path else None,
        "blind_mapping_sha256": sha256(mapping_path) if mapping_path else None,
        "blindness_scope": (
            "procedural_score_and_prior-label_blinding; shared-workspace filesystem ACL isolation "
            "is not claimed"
        ),
        "queues": {
            reviewer: {"path": str(path.relative_to(ROOT)), "sha256": sha256(path)}
            for reviewer, path in queue_paths.items()
        },
    }
    manifest_path = resolve(outputs["manifest"])
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_manifest = manifest_path.with_name(f".{manifest_path.name}.tmp")
    temporary_manifest.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    temporary_manifest.replace(manifest_path)
    print(json.dumps({"manifest": str(manifest_path.relative_to(ROOT)), "row_count": len(selected_uids)}, indent=2))


if __name__ == "__main__":
    main()
