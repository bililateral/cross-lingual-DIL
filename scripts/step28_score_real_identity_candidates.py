#!/usr/bin/env python3
"""Freeze Step28/v4 and rank unlabeled real candidates plus graph expansions."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict

import numpy as np

import step28_common as base
import step28_history_common as history


POLICY_PATH = history.POLICY_PATH
BLIND_OCCURRENCE_FIELDS = (
    "source_dataset",
    "source_row_number",
    "source_market_raw",
    "source_field",
    "contact_type",
    "normalized_value",
    "raw_value",
    "title_snippet",
    "description_snippet",
)
BLIND_PACKET_FIELDS = [
    "blind_id", "seller_a_uid", "seller_b_uid", "seller_a_market",
    "seller_b_market", "seller_a_alias", "seller_b_alias",
    "shared_identity_evidence_json", "rotation_identity_evidence_json",
]
BLIND_ADJUDICATION_FIELDS = [
    "blind_id", "reviewer_id", "identity_decision",
    "evidence_sufficiency", "confidence", "review_notes",
]


def load_e5(policy: dict) -> tuple[dict[str, int], np.ndarray]:
    metadata = base.load_json(policy["inputs"]["frozen_e5_metadata"])
    matrix = np.load(base.resolve(policy["inputs"]["frozen_e5_matrix"]), allow_pickle=False)
    sellers = list(metadata["seller_uids"])
    if matrix.shape[0] != len(sellers) or len(set(sellers)) != len(sellers):
        raise ValueError("Step28/v4 frozen E5 cache is inconsistent")
    return {seller: index for index, seller in enumerate(sellers)}, matrix


def candidate_universe(
    policy: dict,
    graph: dict,
    eligible: set[str],
    known_reviewed_pair_uids: set[str],
) -> tuple[dict, dict]:
    rows = base.load_csv(policy["inputs"]["real_unlabeled_candidate_pool"])
    expected_status = policy["real_scoring"]["candidate_status_must_equal"]
    for row in rows:
        if str(row.get("review_label", "")).strip():
            raise ValueError("Step28/v4 refuses to open a nonblank real candidate label")
        if row.get("review_status") != expected_status:
            raise ValueError("Step28/v4 real candidate status boundary changed")
    universe: dict[tuple[str, str], dict] = {}
    origins: dict[tuple[str, str], set[str]] = defaultdict(set)
    excluded_origins: Counter = Counter()
    for row in rows:
        left, right = sorted((row["seller_uid_left"], row["seller_uid_right"]))
        if left not in eligible or right not in eligible:
            raise ValueError("Step28/v4 Step4 candidate lacks a frozen E5 endpoint")
        edge = (left, right)
        if history.canonical_pair_uid(left, right) in known_reviewed_pair_uids:
            excluded_origins["existing_step4"] += 1
            continue
        universe[edge] = row
        origins[edge].add("existing_step4")
    direct_pairs, rotation_pairs = history.expansion_pairs(graph, eligible)
    for edge in direct_pairs:
        if history.canonical_pair_uid(*edge) in known_reviewed_pair_uids:
            excluded_origins["identity_graph_direct"] += 1
            continue
        universe.setdefault(edge, {})
        origins[edge].add("identity_graph_direct")
    for edge in rotation_pairs:
        if history.canonical_pair_uid(*edge) in known_reviewed_pair_uids:
            excluded_origins["identity_graph_rotation"] += 1
            continue
        universe.setdefault(edge, {})
        origins[edge].add("identity_graph_rotation")
    metadata = {
        "step4_candidate_count": len(rows),
        "identity_graph_direct_pair_count": len(direct_pairs),
        "identity_graph_rotation_pair_count": len(rotation_pairs),
        "union_candidate_count": len(universe),
        "new_outside_step4_count": sum("existing_step4" not in origins[edge] for edge in universe),
        "known_reviewed_registry_count": len(known_reviewed_pair_uids),
        "known_reviewed_exclusion_counts_by_origin": dict(excluded_origins),
        "known_reviewed_pair_uid_remaining_in_universe_count": sum(
            history.canonical_pair_uid(*edge) in known_reviewed_pair_uids
            for edge in universe
        ),
    }
    return {edge: {"step4": universe[edge], "origins": origins[edge]} for edge in universe}, metadata


def load_known_reviewed_pair_uids(policy: dict) -> set[str]:
    path = policy["inputs"]["known_reviewed_pair_uid_exclusions"]
    rows = base.load_csv(path)
    if any(set(row) != {"pair_uid"} for row in rows):
        raise ValueError(
            "Step28 reviewed-pair exclusion registry must expose only pair_uid"
        )
    values = [row["pair_uid"].strip() for row in rows]
    if not values or any(not value for value in values) or len(values) != len(set(values)):
        raise ValueError("Step28 reviewed-pair exclusion registry is empty or duplicated")
    summary = base.load_json(policy["inputs"]["known_reviewed_pair_uid_exclusions_summary"])
    recorded_count = int(
        summary.get("unique_pair_uid_count", summary.get("pair_uid_count", -1))
    )
    if recorded_count != len(values):
        raise ValueError("Step28 reviewed-pair exclusion summary count changed")
    return set(values)


def recompute_observable_state_support(
    rows: list[dict], names: list[str]
) -> dict[str, dict]:
    support: dict[str, dict] = {}
    for row in rows:
        if row["synthetic_split"] not in {
            "synthetic_train",
            "synthetic_development",
        }:
            continue
        values = np.asarray([float(row[name]) for name in names], dtype=float)
        state_hash = history.observable_state_hash(values)
        current = support.setdefault(
            state_hash,
            {
                "positive_count": 0,
                "negative_count": 0,
                "splits": set(),
                "recipes": set(),
            },
        )
        current[f"{row['review_label']}_count"] += 1
        current["splits"].add(row["synthetic_split"])
        current["recipes"].add(row["recipe_id"])
    output = {}
    for state_hash, current in sorted(support.items()):
        positive = int(current["positive_count"])
        negative = int(current["negative_count"])
        status = (
            "ambiguous"
            if positive and negative
            else "positive_only"
            if positive
            else "negative_only"
        )
        output[state_hash] = {
            "positive_count": positive,
            "negative_count": negative,
            "status": status,
            "splits": sorted(current["splits"]),
            "recipes": sorted(current["recipes"]),
        }
    return output


def occurrence_evidence(row: dict) -> dict:
    return {
        key: row.get(key, "")
        for key in BLIND_OCCURRENCE_FIELDS
    }


def pair_evidence(
    left: str,
    right: str,
    by_seller: dict,
    graph: dict,
) -> tuple[list[dict], list[dict]]:
    left_tokens = by_seller.get(left, {})
    right_tokens = by_seller.get(right, {})
    direct_shared = []
    for token in sorted(set(left_tokens) & set(right_tokens)):
        direct_shared.append(
            {
                "contact_type": token[0],
                "normalized_value": token[1],
                "left_occurrences": [
                    occurrence_evidence(row) for row in left_tokens[token]
                ],
                "right_occurrences": [
                    occurrence_evidence(row) for row in right_tokens[token]
                ],
            }
        )
    rotations = []
    strong = graph["strong_adjacency"]
    for middle in sorted(strong.get(left, set()) & strong.get(right, set())):
        left_edge = tuple(sorted((left, middle)))
        right_edge = tuple(sorted((middle, right)))
        left_edge_tokens = graph["strong_edge_tokens"].get(left_edge, set())
        right_edge_tokens = graph["strong_edge_tokens"].get(right_edge, set())
        if not left_edge_tokens.isdisjoint(right_edge_tokens):
            continue
        rotations.append(
            {
                "middle_seller_uid": middle,
                "left_to_middle": [
                    {
                        "contact_type": token[0],
                        "normalized_value": token[1],
                        "left_occurrences": [
                            occurrence_evidence(row)
                            for row in by_seller[left][token]
                        ],
                        "middle_occurrences": [
                            occurrence_evidence(row)
                            for row in by_seller[middle][token]
                        ],
                    }
                    for token in sorted(left_edge_tokens)
                ],
                "middle_to_right": [
                    {
                        "contact_type": token[0],
                        "normalized_value": token[1],
                        "middle_occurrences": [
                            occurrence_evidence(row)
                            for row in by_seller[middle][token]
                        ],
                        "right_occurrences": [
                            occurrence_evidence(row)
                            for row in by_seller[right][token]
                        ],
                    }
                    for token in sorted(right_edge_tokens)
                ],
            }
        )
    return direct_shared, rotations


def score_candidates(
    policy: dict,
) -> tuple[list[dict], list[dict], list[dict], list[dict], dict]:
    artifact_path = policy["inputs"].get("frozen_synthetic_model_artifacts")
    if artifact_path is None:
        artifact_path = base.output_root(policy) / policy["outputs"]["model_artifacts"]
    artifacts = base.load_json(artifact_path)
    if artifacts.get("decision") != "GO":
        raise RuntimeError("Step28/v4 real scoring is forbidden unless synthetic audit is GO")
    source_independence_verified = False
    if policy["generation"].get("source_carrier_assignment") == "label_blind_exact_pairing":
        training_summary_path = policy["inputs"].get("frozen_synthetic_training_summary")
        if training_summary_path is None:
            training_summary_path = (
                base.output_root(policy) / policy["outputs"]["training_summary"]
            )
        frozen_training = base.load_json(training_summary_path)
        preflight = frozen_training.get("audit_diagnostics", {}).get(
            "source_carrier_independence_preflight", {}
        )
        source_independence_verified = bool(
            frozen_training.get("decision") == "GO"
            and preflight.get("required") is True
            and preflight.get("passed") is True
            and all(
                float(details.get("source_only_roc_auc", -1.0)) == 0.5
                for details in preflight.get("splits", {}).values()
            )
        )
        if not source_independence_verified:
            raise RuntimeError(
                "Step28 label-blind real scoring refuses a model without a clean "
                "source-carrier independence preflight"
            )
    model = artifacts["primary_model"]
    names = policy["model"]["feature_names"]
    if model.get("feature_names") != names:
        raise RuntimeError("Step28 frozen model feature order differs from policy")
    if artifacts.get("frozen_source_scorer") != policy["frozen_source_scorer"]:
        raise RuntimeError("Step28 frozen model source scorer differs from policy")
    model_input_path = policy["inputs"].get("frozen_synthetic_model_inputs")
    if model_input_path is None:
        model_input_path = base.output_root(policy) / policy["outputs"]["model_inputs"]
    all_synthetic_rows = base.load_csv(model_input_path)
    support_rows = [
        row for row in all_synthetic_rows
        if row["synthetic_split"] in {
            "synthetic_train", "synthetic_development"
        }
    ]
    support_matrix = np.asarray(
        [[float(row[name]) for name in names] for row in support_rows], dtype=float
    )
    if not support_rows:
        raise ValueError("Step28 guarded application lacks synthetic support rows")
    observable_support = recompute_observable_state_support(support_rows, names)
    if model.get("observable_state_support") != observable_support:
        raise RuntimeError(
            "Step28 observable-state support in artifact does not recompute exactly"
        )
    feature_minimum = np.min(support_matrix, axis=0)
    feature_maximum = np.max(support_matrix, axis=0)
    support_corrections = history.identity_correction(support_matrix, model)
    correction_minimum = float(np.min(support_corrections))
    correction_maximum = float(np.max(support_corrections))
    e5_index, e5_matrix = load_e5(policy)
    signal_rows = base.load_csv(policy["inputs"]["real_item_identity_signals"])
    by_seller, token_df = history.build_signal_index(signal_rows)
    graph = history.build_identity_graph(by_seller, token_df, policy)
    known_reviewed_pair_uids = load_known_reviewed_pair_uids(policy)
    universe, inventory = candidate_universe(
        policy, graph, set(e5_index), known_reviewed_pair_uids
    )
    profiles = {row["seller_uid"]: row for row in base.load_jsonl(policy["inputs"]["real_seller_profiles"])}

    scored: list[dict] = []
    for (left, right), candidate in universe.items():
        left_vector = np.asarray(e5_matrix[e5_index[left]], dtype=float)
        right_vector = np.asarray(e5_matrix[e5_index[right]], dtype=float)
        cosine = float(np.dot(left_vector, right_vector))
        source = history.source_probability_from_cosine(cosine, policy)
        features, details = history.history_feature_details(
            left, right, by_seller, token_df, graph, policy
        )
        matrix = history.feature_vector(features, policy)[None, :]
        state_hash = history.observable_state_hash(matrix[0])
        exact_support = observable_support.get(
            state_hash,
            {
                "positive_count": 0,
                "negative_count": 0,
                "status": "unseen",
                "splits": [],
                "recipes": [],
            },
        )
        positive_support_count = int(exact_support["positive_count"])
        negative_support_count = int(exact_support["negative_count"])
        support_status = str(exact_support["status"])
        support_splits = list(exact_support.get("splits", []))
        minimum_positive_support = int(
            policy["real_scoring"].get(
                "minimum_positive_observable_state_support_count", 1
            )
        )
        repeated_positive_support = int(
            support_status == "positive_only"
            and positive_support_count >= minimum_positive_support
        )
        both_split_support = int(
            set(support_splits)
            == {"synthetic_train", "synthetic_development"}
        )
        out_of_support = np.where(
            (matrix[0] < feature_minimum - 1e-12)
            | (matrix[0] > feature_maximum + 1e-12)
        )[0]
        clipped_matrix = np.clip(matrix, feature_minimum[None, :], feature_maximum[None, :])
        unbounded_correction = float(history.identity_correction(matrix, model)[0])
        feature_bounded_correction = float(history.identity_correction(clipped_matrix, model)[0])
        correction = float(np.clip(
            feature_bounded_correction, correction_minimum, correction_maximum
        ))
        production_review_eligible = history.positive_review_eligible(
            identity_correction=correction,
            out_of_support=bool(len(out_of_support)),
            support=exact_support,
            policy=policy,
        )
        model_score = float(base.sigmoid(base.logit(source) + correction))
        step4 = candidate["step4"]
        left_profile = profiles.get(left, {})
        right_profile = profiles.get(right, {})
        scored.append({
            "pair_uid": history.canonical_pair_uid(left, right),
            "candidate_origins": "|".join(sorted(candidate["origins"])),
            "new_outside_step4": int("existing_step4" not in candidate["origins"]),
            "seller_uid_left": left,
            "seller_uid_right": right,
            "source_market_raw_left": left_profile.get("source_market_raw", step4.get("source_market_raw_left", "")),
            "source_market_raw_right": right_profile.get("source_market_raw", step4.get("source_market_raw_right", "")),
            "alias_normalized_left": left_profile.get("alias_normalized", step4.get("alias_normalized_left", "")),
            "alias_normalized_right": right_profile.get("alias_normalized", step4.get("alias_normalized_right", "")),
            "source_cosine": f"{cosine:.12f}",
            "frozen_source_score": f"{source:.12f}",
            "unbounded_identity_logit_diagnostic": f"{unbounded_correction:.12f}",
            "feature_bounded_identity_logit_diagnostic": f"{feature_bounded_correction:.12f}",
            "identity_logit_correction": f"{correction:.12f}",
            "synthetic_scale_model_score": f"{model_score:.12f}",
            "model_score_is_real_probability": 0,
            "synthetic_support_clipped": int(len(out_of_support) > 0),
            "observable_state_hash": state_hash,
            "synthetic_train_development_support_status": support_status,
            "synthetic_train_development_positive_support_count": positive_support_count,
            "synthetic_train_development_negative_support_count": negative_support_count,
            "synthetic_support_splits": ";".join(support_splits),
            "synthetic_positive_support_repeated": repeated_positive_support,
            "synthetic_positive_support_in_both_splits": both_split_support,
            "production_review_eligible": int(production_review_eligible),
            "out_of_support_feature_count": len(out_of_support),
            "out_of_support_feature_names": ";".join(names[index] for index in out_of_support),
            "shared_token_count": details["shared_token_count"],
            "shared_token_hashes": ";".join(details["shared_token_hashes"]),
            "shared_identifier_types": ";".join(details["shared_identifier_types"]),
            "strong_common_middle_count": details["strong_common_middle_count"],
            "weak_common_middle_count": details["weak_common_middle_count"],
            "step4_candidate_rule_hits": step4.get("candidate_rule_hits", ""),
            **{name: f"{features[name]:.12f}" for name in policy["model"]["feature_names"]},
        })
    scored.sort(
        key=lambda row: (
            -float(row["synthetic_scale_model_score"]),
            -float(row["identity_logit_correction"]),
            row["pair_uid"],
        )
    )
    for rank, row in enumerate(scored, 1):
        row["rank"] = rank

    eligible_queue = [
        row for row in scored if int(row["production_review_eligible"]) == 1
    ]
    queue_size = int(policy["real_scoring"]["review_queue_size"])
    queue = []
    for queue_rank, row in enumerate(eligible_queue[:queue_size], 1):
        queue.append({
            "queue_rank": queue_rank,
            "blind_id": "B" + base.opaque_uid(
                f"{policy['generation']['synthetic_namespace']}-blind",
                row["pair_uid"],
            )[:15],
            **{key: value for key, value in row.items() if key != "rank"},
            "review_status": "pending_prospective_blind_review",
            "review_label": "",
            "review_notes": "",
        })
    blind_packet = []
    blind_adjudication = []
    blind_order = sorted(
        queue,
        key=lambda row: base.opaque_uid(
            f"{policy['generation']['synthetic_namespace']}-blind-order",
            row["pair_uid"],
        ),
    )
    for row in blind_order:
        left, right = row["seller_uid_left"], row["seller_uid_right"]
        direct_evidence, rotation_evidence = pair_evidence(
            left, right, by_seller, graph
        )
        blind_packet.append(
            {
                "blind_id": row["blind_id"],
                "seller_a_uid": left,
                "seller_b_uid": right,
                "seller_a_market": row["source_market_raw_left"],
                "seller_b_market": row["source_market_raw_right"],
                "seller_a_alias": row["alias_normalized_left"],
                "seller_b_alias": row["alias_normalized_right"],
                "shared_identity_evidence_json": json.dumps(
                    direct_evidence, ensure_ascii=False, separators=(",", ":")
                ),
                "rotation_identity_evidence_json": json.dumps(
                    rotation_evidence, ensure_ascii=False, separators=(",", ":")
                ),
            }
        )
        blind_adjudication.append(
            {
                "blind_id": row["blind_id"],
                "reviewer_id": "",
                "identity_decision": "",
                "evidence_sufficiency": "",
                "confidence": "",
                "review_notes": "",
            }
        )
    active = [row for row in scored if abs(float(row["identity_logit_correction"])) > 1e-15]
    positive = [row for row in scored if float(row["identity_logit_correction"]) > 0.0]
    negative = [row for row in scored if float(row["identity_logit_correction"]) < 0.0]
    out_of_support_rows = [row for row in scored if int(row["synthetic_support_clipped"]) == 1]
    cross_label_ambiguous_rows = [
        row for row in scored
        if row["synthetic_train_development_support_status"] == "ambiguous"
    ]
    summary = {
        **inventory,
        "scored_candidate_count": len(scored),
        "identity_active_candidate_count": len(active),
        "positive_identity_correction_count": len(positive),
        "negative_identity_correction_count": len(negative),
        "source_exact_fallback_count": len(scored) - len(active),
        "out_of_synthetic_support_count": len(out_of_support_rows),
        "synthetic_train_development_ambiguous_candidate_count": len(
            cross_label_ambiguous_rows
        ),
        "ambiguous_candidate_in_queue_count": sum(
            row["synthetic_train_development_support_status"] == "ambiguous"
            for row in queue
        ),
        "queue_without_repeated_positive_support_count": sum(
            int(int(row["synthetic_positive_support_repeated"]) == 0)
            for row in queue
        ),
        "queue_without_train_and_development_support_count": sum(
            int(int(row["synthetic_positive_support_in_both_splits"]) == 0)
            for row in queue
        ),
        "prospective_review_queue_count": len(queue),
        "blind_evidence_packet_count": len(blind_packet),
        "blind_packet_model_output_column_count": 0,
        "blind_packet_contains_pair_uid": False,
        "new_outside_step4_in_queue": sum(int(row["new_outside_step4"]) for row in queue),
        "queue_origin_counts": dict(Counter(row["candidate_origins"] for row in queue)),
        "maximum_unbounded_identity_logit_diagnostic": max((float(row["unbounded_identity_logit_diagnostic"]) for row in scored), default=0.0),
        "minimum_unbounded_identity_logit_diagnostic": min((float(row["unbounded_identity_logit_diagnostic"]) for row in scored), default=0.0),
        "maximum_identity_logit_correction": max((float(row["identity_logit_correction"]) for row in scored), default=0.0),
        "minimum_identity_logit_correction": min((float(row["identity_logit_correction"]) for row in scored), default=0.0),
        "synthetic_train_development_identity_correction_bounds": [
            correction_minimum, correction_maximum
        ],
        "feature_support_action": "clip_for_scoring_and_exclude_from_primary_review_queue",
        "real_candidate_labels_opened": 0,
        "old_valid_test_open_count": 0,
        "real_candidate_rows_used_for_model_fitting_selection_or_gating": 0,
        "source_carrier_independence_verified_before_real_scoring": (
            source_independence_verified
        ),
        "model_score_is_real_probability": False,
        "real_performance_claim_allowed": False,
        "review_queue_empty_is_valid_abstention": len(queue) == 0,
        "next_required_action": (
            "prospective blind review of the frozen evidence packet, followed by a newly collected holdout"
            if queue
            else "do not relax the guards; acquire new real item-level multitype or corroborated-rotation evidence before another prospective review"
        ),
    }
    if any(row["pair_uid"] in known_reviewed_pair_uids for row in queue):
        raise RuntimeError("Step28 known reviewed pair survived into review queue")
    return scored, queue, blind_packet, blind_adjudication, summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", required=True)
    args = parser.parse_args()
    policy = history.load_policy(args.policy)
    base.validate_frozen_inputs(policy)
    scores, queue, blind_packet, blind_adjudication, summary = score_candidates(policy)
    root = base.output_root(policy)
    outputs = policy["outputs"]
    score_fields = ["rank", *[key for key in scores[0] if key != "rank"]]
    base.write_csv_immutable(root / outputs["real_candidate_scores"], scores, score_fields)
    queue_fields = [
        "queue_rank",
        "blind_id",
        *[key for key in score_fields if key != "rank"],
        "review_status",
        "review_label",
        "review_notes",
    ]
    base.write_csv_immutable(root / outputs["prospective_review_queue"], queue, queue_fields)
    base.write_csv_immutable(
        root / outputs["blind_evidence_packet"], blind_packet, BLIND_PACKET_FIELDS
    )
    base.write_csv_immutable(
        root / outputs["blind_adjudication_template"],
        blind_adjudication,
        BLIND_ADJUDICATION_FIELDS,
    )
    base.write_json_immutable(root / outputs["real_scoring_summary"], summary)
    print(json.dumps({"status": "ok", **summary}, ensure_ascii=False))


if __name__ == "__main__":
    main()
