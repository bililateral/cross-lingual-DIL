#!/usr/bin/env python3
"""Build score-blind Step15-v8 occurrence-context review queues."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from collections import Counter, defaultdict
from pathlib import Path

import step15_v8_common as common


ROOT = Path(__file__).resolve().parent.parent


QUEUE_FIELDS = [
    "review_candidate_uid",
    "queue_kind",
    "pair_uid_if_in_step4",
    "seller_uid_left",
    "seller_uid_right",
    "candidate_component_id",
    "candidate_component_size",
    "split_eligibility",
    "existing_seller_split_membership",
    "evidence_state",
    "shared_identifier_types",
    "shared_identifier_values",
    "verified_direct_token_count",
    "risky_only_token_count",
    "support_only_token_count",
    "mixed_context_token_count",
    "high_frequency_token_count",
    "maximum_train_seller_token_frequency",
    "left_context_preview",
    "right_context_preview",
    "rule_generated_candidate_only",
    "identity_label_from_rule_forbidden",
    "blind_review_label",
    "blind_review_evidence_type",
    "blind_review_notes",
    "reviewer_id",
]


def occurrence_preview(rows: list[dict], limit: int = 800) -> str:
    parts = []
    for row in rows:
        context = str(row.get("context") or row.get("description_snippet") or row.get("title_snippet") or "")
        context = " ".join(context.split())
        flags = (
            f"direct={row.get('direct_identity_eligible','0')};"
            f"seller_facing={row.get('seller_facing_context','0')};"
            f"risky={row.get('product_data_risk_context','0')};"
            f"support={row.get('support_only','0')}"
        )
        parts.append(f"[{row.get('contact_type','')}:{flags}] {context[:300]}")
    return " || ".join(parts)[:limit]


def pair_uid_key(left: str, right: str) -> tuple[str, str]:
    return tuple(sorted((left, right)))


def deterministic_unseen_component_split(sellers: tuple[str, ...]) -> str:
    digest = hashlib.sha256(
        ("20260714|" + "|".join(sellers)).encode("utf-8")
    ).hexdigest()
    return "valid_candidate" if int(digest[:8], 16) % 5 == 0 else "train_candidate"


def candidate_component_index(
    candidate_pairs: set[tuple[str, str]],
) -> tuple[dict[str, tuple[str, ...]], dict[str, str]]:
    adjacency: dict[str, set[str]] = defaultdict(set)
    for left, right in candidate_pairs:
        adjacency[left].add(right)
        adjacency[right].add(left)
    seller_to_component: dict[str, tuple[str, ...]] = {}
    seller_to_component_id: dict[str, str] = {}
    for start in sorted(adjacency):
        if start in seller_to_component:
            continue
        pending = [start]
        members = set()
        while pending:
            seller = pending.pop()
            if seller in members:
                continue
            members.add(seller)
            pending.extend(sorted(adjacency[seller] - members, reverse=True))
        component = tuple(sorted(members))
        component_id = common.canonical_hash(
            ["step15_v8_candidate_component", *component]
        )[:24]
        for seller in component:
            seller_to_component[seller] = component
            seller_to_component_id[seller] = component_id
    return seller_to_component, seller_to_component_id


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", default=str(common.DEFAULT_POLICY))
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--validate-config-only", action="store_true")
    args = parser.parse_args()

    policy_path, policy, v7_policy = common.load_policy(args.policy)
    validation = common.validate_policy_contract(policy, v7_policy)
    if args.validate_config_only:
        print(json.dumps(validation, indent=2))
        return
    run_id = args.run_id or policy["default_run_id"]
    root = common.run_root(policy, run_id)
    cfg = policy["context_review_queues"]
    final_root = root / cfg["output_subdirectory"]
    staging_root = final_root.with_name(f".{final_root.name}.incomplete")
    if final_root.exists() or staging_root.exists():
        raise FileExistsError(
            f"Refusing to overwrite Step15-v8 context review queues: {final_root} / {staging_root}"
        )
    pool_name = cfg["source_pool"]
    pool = policy["pools"][pool_name]
    signals = common.load_csv(common.resolve(pool["item_identity_signals"]))
    by_seller = defaultdict(lambda: defaultdict(list))
    sellers_by_token = defaultdict(set)
    for row in signals:
        seller = str(row.get("seller_uid", "")).strip()
        contact_type = str(row.get("contact_type", "")).strip().lower()
        value = str(row.get("normalized_value", "")).strip().lower()
        if not seller or not contact_type or not value:
            continue
        token = (contact_type, value)
        by_seller[seller][token].append(row)
        sellers_by_token[token].add(seller)

    assignments = {
        row["pair_uid"]: row
        for row in common.load_csv(
            common.resolve(policy["frozen_dependencies"]["representative_validation_assignments"])
        )
    }
    labels = [
        row
        for row in common.load_csv(common.resolve(pool["frozen_labels"]))
        if row.get("review_label") in {"positive", "negative"}
        and row.get("usable_for_supervision") == "1"
        and row.get("usable_for_core_transfer") == "1"
    ]
    supervised_pair_keys = {pair_uid_key(row["seller_uid_left"], row["seller_uid_right"]) for row in labels}
    seller_splits = defaultdict(set)
    for row in labels:
        assignment = assignments.get(row["pair_uid"])
        if assignment is None:
            raise ValueError(f"Missing v7 assignment for current Chinese supervision: {row['pair_uid']}")
        split = assignment["v7_split_name"]
        seller_splits[row["seller_uid_left"]].add(split)
        seller_splits[row["seller_uid_right"]].add(split)
    train_sellers = {
        seller for seller, splits in seller_splits.items() if splits == {"train"}
    }
    token_df = Counter(
        {
            token: len(sellers & train_sellers)
            for token, sellers in sellers_by_token.items()
        }
    )
    step4 = common.load_csv(common.resolve(pool["step4_candidates"]))
    step4_uid = {
        pair_uid_key(row["seller_uid_left"], row["seller_uid_right"]): row["pair_uid"]
        for row in step4
    }

    max_sellers = int(cfg["maximum_sellers_per_token_for_pair_enumeration"])
    candidate_pairs = set()
    skipped_high_frequency_tokens = 0
    for token, sellers in sellers_by_token.items():
        ordered = sorted(
            sellers,
            key=lambda seller: (
                hashlib.sha256(
                    f"20260714|{token[0]}|{token[1]}|{seller}".encode("utf-8")
                ).hexdigest(),
                seller,
            ),
        )
        if len(ordered) > max_sellers:
            skipped_high_frequency_tokens += 1
            ordered = ordered[:max_sellers]
        candidate_pairs.update(
            pair_uid_key(left, right) for left, right in itertools.combinations(ordered, 2)
        )
    seller_components, seller_component_ids = candidate_component_index(candidate_pairs)

    queue_rows = defaultdict(list)
    for left, right in sorted(candidate_pairs):
        key = pair_uid_key(left, right)
        if key in supervised_pair_keys:
            continue
        row = {
            "seller_uid_left": left,
            "seller_uid_right": right,
            "domain": "zh",
        }
        evidence = common.occurrence_evidence(
            row,
            by_seller,
            token_df,
            int(policy["occurrence_evidence_expert"]["public_identifier_train_seller_frequency_threshold"]),
        )
        state_to_queue = {
            "risky_only_shared": "risky_only_public_noise",
            "support_only_shared": "risky_only_public_noise",
            "high_frequency_public": "risky_only_public_noise",
            "direct_with_mixed_context": "mixed_context_identifier",
            "verified_direct_both_sides": "verified_direct_both_sides",
        }
        queue_kind = state_to_queue.get(evidence["evidence_state"])
        if queue_kind is None:
            continue
        shared = sorted(set(by_seller[left]) & set(by_seller[right]))
        left_occ = [item for token in shared for item in by_seller[left][token]]
        right_occ = [item for token in shared for item in by_seller[right][token]]
        component = seller_components[left]
        if seller_components[right] != component:
            raise ValueError("Candidate pair endpoints were assigned to different components")
        membership = sorted(
            {
                split
                for seller in component
                for split in seller_splits.get(seller, set())
            }
        )
        if len(membership) > 1:
            split_eligibility = "blocked_cross_split_seller_overlap"
        elif membership:
            split = membership[0]
            split_eligibility = (
                f"{split}_only" if split != "internal_development_test" else "diagnostic_test_only"
            )
        else:
            split_eligibility = deterministic_unseen_component_split(component)
        shared_values = [f"{token[0]}:{token[1]}" for token in shared]
        queue_rows[queue_kind].append(
            {
                "review_candidate_uid": common.canonical_hash([queue_kind, left, right])[:24],
                "queue_kind": queue_kind,
                "pair_uid_if_in_step4": step4_uid.get(key, ""),
                "seller_uid_left": left,
                "seller_uid_right": right,
                "candidate_component_id": seller_component_ids[left],
                "candidate_component_size": len(component),
                "split_eligibility": split_eligibility,
                "existing_seller_split_membership": "|".join(membership),
                "evidence_state": evidence["evidence_state"],
                "shared_identifier_types": "|".join(evidence["shared_identifier_types"]),
                "shared_identifier_values": "|".join(shared_values),
                "verified_direct_token_count": evidence["verified_direct_token_count"],
                "risky_only_token_count": evidence["risky_only_token_count"],
                "support_only_token_count": evidence["support_only_token_count"],
                "mixed_context_token_count": evidence["mixed_context_token_count"],
                "high_frequency_token_count": evidence["high_frequency_token_count"],
                "maximum_train_seller_token_frequency": evidence[
                    "maximum_train_seller_token_frequency"
                ],
                "left_context_preview": occurrence_preview(left_occ),
                "right_context_preview": occurrence_preview(right_occ),
                "rule_generated_candidate_only": "1",
                "identity_label_from_rule_forbidden": "1",
                "blind_review_label": "",
                "blind_review_evidence_type": "",
                "blind_review_notes": "",
                "reviewer_id": "",
            }
        )

    staging_root.mkdir(parents=True, exist_ok=False)
    output_records = {}
    for queue_kind, limit in cfg["queue_limits"].items():
        rows = queue_rows.get(queue_kind, [])
        rows.sort(
            key=lambda row: (
                row["split_eligibility"].startswith("blocked"),
                -int(row["verified_direct_token_count"]),
                -int(row["mixed_context_token_count"]),
                -int(row["risky_only_token_count"]),
                int(row["maximum_train_seller_token_frequency"]),
                row["review_candidate_uid"],
            )
        )
        selected = rows[: int(limit)]
        path = staging_root / f"{queue_kind}_blind_review_queue.csv"
        path.write_bytes(common.render_csv(selected, QUEUE_FIELDS))
        output_records[queue_kind] = {
            "candidate_count_before_limit": len(rows),
            "output_count": len(selected),
            "split_eligibility_counts": dict(
                sorted(Counter(row["split_eligibility"] for row in selected).items())
            ),
            "path": str((final_root / path.name).relative_to(ROOT)).replace("\\", "/"),
            "sha256": common.sha256(path),
        }
    targets = cfg["review_targets"]
    summary = {
        "step": "step16_build_v8_context_review_queues",
        "version": policy["version"],
        "run_id": run_id,
        "source_occurrence_count": len(signals),
        "distinct_identifier_token_count": len(sellers_by_token),
        "candidate_pair_count_before_supervision_exclusion": len(candidate_pairs),
        "candidate_component_count": len(set(seller_component_ids.values())),
        "candidate_split_assignment_unit": "shared_identifier_seller_component",
        "skipped_or_truncated_high_frequency_token_count": skipped_high_frequency_tokens,
        "high_frequency_token_sampling": "deterministic_sha256_not_lexicographic_prefix",
        "outputs": output_records,
        "review_targets": targets,
        "candidate_rules_assign_identity_labels": False,
        "model_scores_read": False,
        "review_required_before_any_step5_update": True,
        "policy_sha256": common.sha256(policy_path),
        "inputs": {
            "item_identity_signals_sha256": common.sha256(
                common.resolve(pool["item_identity_signals"])
            ),
            "frozen_labels_sha256": common.sha256(common.resolve(pool["frozen_labels"])),
            "step4_candidates_sha256": common.sha256(
                common.resolve(pool["step4_candidates"])
            ),
        },
    }
    summary["summary_sha256"] = common.canonical_hash(summary)
    summary_path = staging_root / "step16_v8_context_review_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    staging_root.replace(final_root)
    print(
        json.dumps(
            {
                "status": "pass",
                "run_id": run_id,
                "outputs": {
                    key: value["output_count"] for key, value in output_records.items()
                },
                "summary": str((final_root / summary_path.name).relative_to(ROOT)).replace("\\", "/"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
