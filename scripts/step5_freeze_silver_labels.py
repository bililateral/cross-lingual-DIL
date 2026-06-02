from __future__ import annotations

import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
POLICY_PATH = ROOT / "schema" / "step5_freeze_policy.json"
QUEUE_PATHS = {
    "en_content_train_pool": ROOT / "reports" / "step5_en_balanced_review_queue.csv",
    "zh_target_strict": ROOT / "reports" / "step5_zh_target_strict_balanced_review_queue.csv",
    "zh_target_aux": ROOT / "reports" / "step5_zh_target_aux_balanced_review_queue.csv",
}
CANDIDATE_PATHS = {
    "en_content_train_pool": ROOT / "reports" / "step4_en_silver_candidate_pairs.csv",
    "zh_target_strict": ROOT / "reports" / "step4_zh_target_strict_silver_candidate_pairs.csv",
    "zh_target_aux": ROOT / "reports" / "step4_zh_target_aux_silver_candidate_pairs.csv",
}
OUTPUT_PATHS = {
    "en_content_train_pool": ROOT / "reports" / "step5_en_frozen_silver_labels.csv",
    "zh_target_strict": ROOT / "reports" / "step5_zh_target_strict_frozen_silver_labels.csv",
    "zh_target_aux": ROOT / "reports" / "step5_zh_target_aux_frozen_silver_labels.csv",
}
SUMMARY_PATH = ROOT / "reports" / "step5_frozen_silver_summary.json"
SPLIT_ORDER = ("train", "valid", "test")
TRUST_SUFFIX_RE = re.compile(r"\s*\(\d+%\)\s*$", re.I)
NON_ALIAS_CHARS_RE = re.compile(r"[^0-9a-z\u3400-\u9fff/]+", re.I)
DEFAULT_SPLIT_BALANCE_OBJECTIVE = {
    "row_weight": 0.4,
    "label_weight": 1.0,
    "label_stratum_weight": 4.0,
    "rare_label_stratum_threshold": 6,
    "rare_label_stratum_weight_multiplier": 0.4,
    "overfill_tolerance": 0.2,
    "overfill_penalty_weight": 3.0,
}


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def normalize_text(value: str) -> str:
    return str(value or "").strip()


def normalize_label(value: str) -> str:
    return normalize_text(value).lower()


def normalize_status(value: str) -> str:
    return normalize_text(value).lower()


def normalize_soft_alias(value: str) -> str:
    text = normalize_text(value).casefold()
    text = TRUST_SUFFIX_RE.sub("", text)
    text = NON_ALIAS_CHARS_RE.sub("", text)
    return text


def is_soft_same_alias(left_raw: str, right_raw: str) -> bool:
    left_norm = normalize_soft_alias(left_raw)
    right_norm = normalize_soft_alias(right_raw)
    if not left_norm or not right_norm:
        return False
    if left_norm != right_norm:
        return False
    return normalize_text(left_raw) != normalize_text(right_raw)


def as_int(value: str) -> int:
    if value in {"", None}:
        return 0
    return int(value)


def label_stratum_key(row: dict) -> str:
    return f"{row['review_label']}|{row['review_stratum']}"


def candidate_map(rows: list[dict]) -> dict[str, dict]:
    mapping = {}
    for row in rows:
        pair_uid = row["pair_uid"]
        if pair_uid in mapping:
            raise ValueError(f"Duplicate pair_uid in candidate table: {pair_uid}")
        mapping[pair_uid] = row
    return mapping


def reviewed_rows_from_queue(queue_rows: list[dict], policy: dict, pair_map: dict[str, dict], pool: str) -> tuple[list[dict], dict]:
    valid_labels = set(policy["label_vocab"])
    pending_statuses = set(policy["pending_status_values"])
    audit_scopes = set(policy["audit_only_candidate_scopes"])
    primary_scope = policy["primary_candidate_scope"]

    reviewed_rows: list[dict] = []
    issues = {
        "status_without_label": [],
        "invalid_label": [],
        "missing_reviewer_id": [],
        "pair_uid_missing_from_candidates": [],
        "status_inferred_from_label_count": 0,
    }

    for queue_row in queue_rows:
        pair_uid = queue_row["pair_uid"]
        candidate_row = pair_map.get(pair_uid)
        if not candidate_row:
            issues["pair_uid_missing_from_candidates"].append(pair_uid)
            continue

        review_label = normalize_label(queue_row.get("review_label", ""))
        review_status = normalize_status(queue_row.get("review_status", ""))
        reviewer_id = normalize_text(queue_row.get("reviewer_id", ""))
        review_notes = normalize_text(queue_row.get("review_notes", ""))

        if not review_label:
            if review_status not in pending_statuses:
                issues["status_without_label"].append(pair_uid)
            continue

        if review_label not in valid_labels:
            issues["invalid_label"].append(pair_uid)
            continue

        if review_status in pending_statuses:
            issues["status_inferred_from_label_count"] += 1
            review_status = "reviewed"

        if not reviewer_id:
            issues["missing_reviewer_id"].append(pair_uid)
            continue

        candidate_scope = candidate_row["candidate_scope"]
        soft_same_alias = is_soft_same_alias(
            queue_row.get("source_seller_raw_left", ""),
            queue_row.get("source_seller_raw_right", ""),
        )
        usable_for_supervision = int(candidate_scope == primary_scope and review_label in {"positive", "negative"})
        usable_for_core_transfer = usable_for_supervision

        split_name = ""
        if candidate_scope in audit_scopes:
            split_name = "audit_only"
            usable_for_supervision = 0
            usable_for_core_transfer = 0
        elif soft_same_alias:
            split_name = "audit_only_soft_alias"
            usable_for_supervision = 0
            usable_for_core_transfer = 0
        elif review_label == "uncertain":
            split_name = "uncertain_holdout"

        reviewed_row = {
            "balanced_review_rank": queue_row["balanced_review_rank"],
            "pair_uid": pair_uid,
            "data_bucket": candidate_row["data_bucket"],
            "candidate_language": candidate_row["candidate_language"],
            "candidate_scope": candidate_scope,
            "review_stratum": queue_row["review_stratum"],
            "review_priority": queue_row["review_priority"],
            "review_status": review_status,
            "review_label": review_label,
            "reviewer_id": reviewer_id,
            "review_notes": review_notes,
            "soft_same_alias_continuity_bool": int(soft_same_alias),
            "usable_for_supervision": usable_for_supervision,
            "usable_for_core_transfer": usable_for_core_transfer,
            "split_name": split_name,
            "split_component_id": "",
            "split_component_size": "",
            "seller_uid_left": candidate_row["seller_uid_left"],
            "seller_uid_right": candidate_row["seller_uid_right"],
            "source_market_raw_left": queue_row["source_market_raw_left"],
            "source_market_raw_right": queue_row["source_market_raw_right"],
            "source_seller_raw_left": queue_row["source_seller_raw_left"],
            "source_seller_raw_right": queue_row["source_seller_raw_right"],
            "alias_relation": queue_row["alias_relation"],
            "same_market_raw": queue_row["same_market_raw"],
            "candidate_rule_hits": queue_row["candidate_rule_hits"],
            "candidate_rank_score": queue_row["candidate_rank_score"],
            "lexical_similarity": queue_row["lexical_similarity"],
            "structural_support_score": queue_row["structural_support_score"],
            "shared_contact_count": candidate_row["shared_contact_count"],
            "shared_contact_values": queue_row["shared_contact_values"],
            "shared_title_count": candidate_row["shared_title_count"],
            "shared_title_values": queue_row["shared_title_values"],
            "shared_description_count": candidate_row["shared_description_count"],
            "shared_description_values": queue_row["shared_description_values"],
            "shared_category_count": candidate_row["shared_category_count"],
            "shared_category_values": queue_row["shared_category_values"],
            "shared_pgp_fingerprint_count": candidate_row["shared_pgp_fingerprint_count"],
            "shared_pgp_fingerprint_values": queue_row["shared_pgp_fingerprint_values"],
            "left_preview": queue_row["left_preview"],
            "right_preview": queue_row["right_preview"],
        }
        reviewed_rows.append(reviewed_row)

    blocking_issue_count = sum(
        len(issues[name])
        for name in ("status_without_label", "invalid_label", "missing_reviewer_id", "pair_uid_missing_from_candidates")
    )
    if blocking_issue_count:
        detail = {
            name: issues[name]
            for name in ("status_without_label", "invalid_label", "missing_reviewer_id", "pair_uid_missing_from_candidates")
            if issues[name]
        }
        raise ValueError(f"{pool} has invalid Step-5 review rows: {json.dumps(detail, ensure_ascii=False)}")

    return reviewed_rows, issues


def build_components(rows: list[dict]) -> dict[str, dict]:
    adjacency: dict[str, set[str]] = defaultdict(set)
    edge_rows: dict[str, list[dict]] = defaultdict(list)
    alias_to_sellers: dict[str, set[str]] = defaultdict(set)

    for row in rows:
        left = row["seller_uid_left"]
        right = row["seller_uid_right"]
        adjacency[left].add(right)
        adjacency[right].add(left)
        edge_rows[left].append(row)
        edge_rows[right].append(row)
        left_alias = normalize_soft_alias(row.get("source_seller_raw_left", ""))
        right_alias = normalize_soft_alias(row.get("source_seller_raw_right", ""))
        if left_alias:
            alias_to_sellers[left_alias].add(left)
        if right_alias:
            alias_to_sellers[right_alias].add(right)

    for sellers in alias_to_sellers.values():
        if len(sellers) <= 1:
            continue
        ordered_sellers = sorted(sellers)
        anchor = ordered_sellers[0]
        for seller_uid in ordered_sellers[1:]:
            adjacency[anchor].add(seller_uid)
            adjacency[seller_uid].add(anchor)

    visited: set[str] = set()
    components: dict[str, dict] = {}
    component_idx = 1

    for seller_uid in sorted(adjacency):
        if seller_uid in visited:
            continue
        stack = [seller_uid]
        sellers = []
        pair_index = {}
        while stack:
            current = stack.pop()
            if current in visited:
                continue
            visited.add(current)
            sellers.append(current)
            for row in edge_rows[current]:
                pair_index[row["pair_uid"]] = row
            for neighbor in adjacency[current]:
                if neighbor not in visited:
                    stack.append(neighbor)

        component_id = f"comp_{component_idx:05d}"
        component_rows = sorted(pair_index.values(), key=lambda item: int(item["balanced_review_rank"]))
        components[component_id] = {
            "component_id": component_id,
            "seller_uids": sorted(sellers),
            "rows": component_rows,
            "row_count": len(component_rows),
            "positive_count": sum(1 for row in component_rows if row["review_label"] == "positive"),
            "negative_count": sum(1 for row in component_rows if row["review_label"] == "negative"),
            "label_stratum_counts": Counter(label_stratum_key(row) for row in component_rows),
        }
        component_idx += 1

    return components


def split_balance_objective(policy: dict) -> dict:
    configured = policy.get("split_strategy", {}).get("balance_objective", {})
    objective = dict(DEFAULT_SPLIT_BALANCE_OBJECTIVE)
    objective.update(configured)
    return objective


def add_component_to_current(current: dict[str, dict], split_name: str, component: dict) -> None:
    current[split_name]["row_count"] += component["row_count"]
    current[split_name]["positive_count"] += component["positive_count"]
    current[split_name]["negative_count"] += component["negative_count"]
    current[split_name]["label_stratum_counts"].update(component["label_stratum_counts"])


def label_stratum_imbalance_cost(
    current: dict[str, dict],
    component: dict,
    split_name: str,
    totals: dict[str, int],
    label_stratum_totals: Counter,
    ratios: dict[str, float],
    objective: dict,
) -> float:
    row_weight = float(objective["row_weight"])
    label_weight = float(objective["label_weight"])
    label_stratum_weight = float(objective["label_stratum_weight"])
    rare_threshold = int(objective["rare_label_stratum_threshold"])
    rare_multiplier = float(objective["rare_label_stratum_weight_multiplier"])
    overfill_tolerance = float(objective["overfill_tolerance"])
    overfill_penalty_weight = float(objective["overfill_penalty_weight"])

    cost = 0.0
    for split in SPLIT_ORDER:
        projected_row_count = current[split]["row_count"] + (
            component["row_count"] if split == split_name else 0
        )
        projected_positive_count = current[split]["positive_count"] + (
            component["positive_count"] if split == split_name else 0
        )
        projected_negative_count = current[split]["negative_count"] + (
            component["negative_count"] if split == split_name else 0
        )

        desired_row_count = totals["row_count"] * ratios[split]
        desired_positive_count = totals["positive_count"] * ratios[split]
        desired_negative_count = totals["negative_count"] * ratios[split]
        cost += row_weight * abs(projected_row_count - desired_row_count) / max(desired_row_count, 1.0)
        cost += (
            label_weight
            * abs(projected_positive_count - desired_positive_count)
            / max(desired_positive_count, 1.0)
        )
        cost += (
            label_weight
            * abs(projected_negative_count - desired_negative_count)
            / max(desired_negative_count, 1.0)
        )

        for key, total_count in label_stratum_totals.items():
            projected_count = current[split]["label_stratum_counts"][key] + (
                component["label_stratum_counts"][key] if split == split_name else 0
            )
            desired_count = total_count * ratios[split]
            key_weight = label_stratum_weight
            if total_count < rare_threshold:
                key_weight *= rare_multiplier
            cost += key_weight * abs(projected_count - desired_count) / max(desired_count, 1.0)

    desired_target_rows = totals["row_count"] * ratios[split_name]
    projected_target_rows = current[split_name]["row_count"] + component["row_count"]
    if projected_target_rows > desired_target_rows * (1.0 + overfill_tolerance):
        cost += (
            (projected_target_rows / max(desired_target_rows, 1.0) - (1.0 + overfill_tolerance))
            * overfill_penalty_weight
        )
    return cost


def choose_split(
    component: dict,
    ratios: dict[str, float],
    current: dict[str, dict],
    totals: dict[str, int],
    label_stratum_totals: Counter,
    objective: dict,
) -> str:
    desired_total = {split: totals["row_count"] * ratios[split] for split in SPLIT_ORDER}
    desired_positive = {split: totals["positive_count"] * ratios[split] for split in SPLIT_ORDER}
    desired_negative = {split: totals["negative_count"] * ratios[split] for split in SPLIT_ORDER}

    best_split = SPLIT_ORDER[0]
    best_cost = None
    for split in SPLIT_ORDER:
        projected = {
            name: {
                "row_count": current[name]["row_count"] + (component["row_count"] if name == split else 0),
                "positive_count": current[name]["positive_count"] + (component["positive_count"] if name == split else 0),
                "negative_count": current[name]["negative_count"] + (component["negative_count"] if name == split else 0),
            }
            for name in SPLIT_ORDER
        }

        legacy_total_cost = sum(
            abs(projected[name]["row_count"] - desired_total[name]) / max(desired_total[name], 1.0)
            for name in SPLIT_ORDER
        )
        legacy_positive_cost = sum(
            abs(projected[name]["positive_count"] - desired_positive[name]) / max(desired_positive[name], 1.0)
            for name in SPLIT_ORDER
        )
        legacy_negative_cost = sum(
            abs(projected[name]["negative_count"] - desired_negative[name]) / max(desired_negative[name], 1.0)
            for name in SPLIT_ORDER
        )
        cost = label_stratum_imbalance_cost(
            current,
            component,
            split,
            totals,
            label_stratum_totals,
            ratios,
            objective,
        )

        empty_split_penalty = 0 if current[split]["row_count"] == 0 else 1
        tie_key = (
            cost,
            legacy_total_cost + 0.5 * legacy_positive_cost + 0.5 * legacy_negative_cost,
            empty_split_penalty,
            projected[split]["row_count"],
            projected[split]["positive_count"],
            split,
        )
        if best_cost is None or tie_key < best_cost:
            best_cost = tie_key
            best_split = split
    return best_split


def requirement_matches_component(component: dict, requirement: dict) -> int:
    return sum(
        1
        for row in component["rows"]
        if row["review_label"] == requirement["review_label"]
        and row["review_stratum"] == requirement["review_stratum"]
    )


def seed_coverage_assignments(
    ordered_components: list[dict],
    current: dict[str, dict],
    coverage_requirements: list[dict],
) -> dict[str, str]:
    assignments: dict[str, str] = {}
    severity_order = {"error": 0, "warning": 1}

    applicable_requirements = []
    for requirement in coverage_requirements:
        pool_available_count = sum(
            requirement_matches_component(component, requirement)
            for component in ordered_components
        )
        minimum_available = int(requirement.get("require_if_pool_available_at_least", 1))
        if pool_available_count < minimum_available:
            continue
        applicable_requirements.append(
            {
                "requirement": requirement,
                "pool_available_count": pool_available_count,
            }
        )

    applicable_requirements.sort(
        key=lambda item: (
            severity_order.get(str(item["requirement"].get("severity", "warning")), 9),
            item["pool_available_count"],
            item["requirement"]["review_label"],
            item["requirement"]["review_stratum"],
            ",".join(item["requirement"]["required_splits"]),
        )
    )

    for item in applicable_requirements:
        requirement = item["requirement"]
        required_splits = [str(split) for split in requirement["required_splits"]]
        for required_split in required_splits:
            already_covered = any(
                split_name == required_split
                and requirement_matches_component(component, requirement) > 0
                for component in ordered_components
                for component_id, split_name in assignments.items()
                if component["component_id"] == component_id
            )
            if already_covered:
                continue

            candidates = []
            for component in ordered_components:
                component_id = component["component_id"]
                if component_id in assignments:
                    continue
                match_count = requirement_matches_component(component, requirement)
                if match_count <= 0:
                    continue
                candidates.append(
                    (
                        component["row_count"],
                        -match_count,
                        -component["positive_count"],
                        component_id,
                        component,
                    )
                )

            if not candidates:
                continue

            _row_count, _neg_match_count, _neg_positive_count, component_id, component = min(candidates)
            assignments[component_id] = required_split
            add_component_to_current(current, required_split, component)

    return assignments


def component_sort_key(component: dict, label_stratum_totals: Counter) -> tuple:
    rare_share = max(
        (
            count / max(label_stratum_totals[key], 1)
            for key, count in component["label_stratum_counts"].items()
        ),
        default=0.0,
    )
    return (
        -len(component["label_stratum_counts"]),
        -rare_share,
        -component["row_count"],
        -component["positive_count"],
        component["component_id"],
    )


def assign_splits(
    rows: list[dict],
    ratios: dict[str, float],
    policy: dict,
    coverage_requirements: list[dict] | None = None,
) -> dict[str, dict]:
    supervision_rows = [row for row in rows if int(row["usable_for_supervision"]) == 1]
    if not supervision_rows:
        return {}

    components = build_components(supervision_rows)
    label_stratum_totals = Counter(label_stratum_key(row) for row in supervision_rows)
    ordered_components = sorted(
        components.values(),
        key=lambda item: component_sort_key(item, label_stratum_totals),
    )

    current = {
        split: {
            "row_count": 0,
            "positive_count": 0,
            "negative_count": 0,
            "label_stratum_counts": Counter(),
        }
        for split in SPLIT_ORDER
    }
    totals = {
        "row_count": len(supervision_rows),
        "positive_count": sum(1 for row in supervision_rows if row["review_label"] == "positive"),
        "negative_count": sum(1 for row in supervision_rows if row["review_label"] == "negative"),
    }
    objective = split_balance_objective(policy)

    seeded_assignments = seed_coverage_assignments(
        ordered_components,
        current,
        coverage_requirements or [],
    )

    assignments: dict[str, dict] = {}
    for component_id, split_name in seeded_assignments.items():
        component = components[component_id]
        assignments[component_id] = {
            "split_name": split_name,
            "component_size": component["row_count"],
            "pair_uids": [row["pair_uid"] for row in component["rows"]],
        }

    for component in ordered_components:
        if component["component_id"] in assignments:
            continue
        split_name = choose_split(
            component,
            ratios,
            current,
            totals,
            label_stratum_totals,
            objective,
        )
        assignments[component["component_id"]] = {
            "split_name": split_name,
            "component_size": component["row_count"],
            "pair_uids": [row["pair_uid"] for row in component["rows"]],
        }
        add_component_to_current(current, split_name, component)

    pair_to_assignment = {}
    for component_id, info in assignments.items():
        for pair_uid in info["pair_uids"]:
            pair_to_assignment[pair_uid] = {
                "split_name": info["split_name"],
                "split_component_id": component_id,
                "split_component_size": info["component_size"],
            }
    return pair_to_assignment


def finalize_rows(rows: list[dict], pair_assignments: dict[str, dict]) -> list[dict]:
    finalized = []
    for row in rows:
        result = dict(row)
        assignment = pair_assignments.get(row["pair_uid"])
        if assignment:
            result.update(assignment)
        elif not result["split_name"]:
            result["split_name"] = "reviewed_not_for_supervision"
        finalized.append(result)
    return finalized


def split_seller_sets(rows: list[dict]) -> dict[str, set[str]]:
    seller_sets = {split: set() for split in SPLIT_ORDER}
    for row in rows:
        if row["split_name"] not in seller_sets:
            continue
        seller_sets[row["split_name"]].add(row["seller_uid_left"])
        seller_sets[row["split_name"]].add(row["seller_uid_right"])
    return seller_sets


def split_seller_overlap_counts(rows: list[dict]) -> dict[str, int]:
    seller_sets = split_seller_sets(rows)
    overlap_counts = {}
    for left_idx, left_split in enumerate(SPLIT_ORDER):
        for right_split in SPLIT_ORDER[left_idx + 1 :]:
            key = f"{left_split}__{right_split}"
            overlap_counts[key] = len(seller_sets[left_split] & seller_sets[right_split])
    return overlap_counts


def split_alias_sets(rows: list[dict]) -> dict[str, set[str]]:
    alias_sets = {split: set() for split in SPLIT_ORDER}
    for row in rows:
        if row["split_name"] not in alias_sets:
            continue
        left_alias = normalize_soft_alias(row.get("source_seller_raw_left", ""))
        right_alias = normalize_soft_alias(row.get("source_seller_raw_right", ""))
        if left_alias:
            alias_sets[row["split_name"]].add(left_alias)
        if right_alias:
            alias_sets[row["split_name"]].add(right_alias)
    return alias_sets


def split_alias_overlap_counts(rows: list[dict]) -> dict[str, int]:
    alias_sets = split_alias_sets(rows)
    overlap_counts = {}
    for left_idx, left_split in enumerate(SPLIT_ORDER):
        for right_split in SPLIT_ORDER[left_idx + 1 :]:
            key = f"{left_split}__{right_split}"
            overlap_counts[key] = len(alias_sets[left_split] & alias_sets[right_split])
    return overlap_counts


def split_label_stratum_counts(rows: list[dict]) -> dict[str, dict[str, dict[str, int]]]:
    counts: dict[str, dict[str, dict[str, int]]] = {split: {} for split in SPLIT_ORDER}
    for split in SPLIT_ORDER:
        split_rows = [row for row in rows if row["split_name"] == split]
        labels = sorted({row["review_label"] for row in split_rows})
        for label in labels:
            counts[split][label] = dict(
                Counter(row["review_stratum"] for row in split_rows if row["review_label"] == label)
            )
    return counts


def evaluate_coverage_requirements(pool: str, rows: list[dict], policy: dict) -> tuple[list[dict], list[str], list[str]]:
    supervision_rows = [row for row in rows if int(row["usable_for_supervision"]) == 1]
    requirements = policy.get("coverage_requirements", {}).get(pool, [])
    results: list[dict] = []
    warnings: list[str] = []
    errors: list[str] = []

    for requirement in requirements:
        label = requirement["review_label"]
        stratum = requirement["review_stratum"]
        required_splits = [str(split) for split in requirement["required_splits"]]
        available_rows = [
            row
            for row in supervision_rows
            if row["review_label"] == label and row["review_stratum"] == stratum
        ]
        split_counts = {
            split: sum(1 for row in available_rows if row["split_name"] == split)
            for split in SPLIT_ORDER
        }
        minimum_available = int(requirement.get("require_if_pool_available_at_least", 1))
        applicable = len(available_rows) >= minimum_available
        missing_splits = [split for split in required_splits if split_counts.get(split, 0) == 0] if applicable else []
        passed = not applicable or not missing_splits
        severity = str(requirement.get("severity", "warning"))
        description = str(requirement.get("description", "")).strip()

        result = {
            "review_label": label,
            "review_stratum": stratum,
            "required_splits": required_splits,
            "require_if_pool_available_at_least": minimum_available,
            "pool_available_count": len(available_rows),
            "split_counts": split_counts,
            "applicable": applicable,
            "missing_splits": missing_splits,
            "severity": severity,
            "passed": passed,
        }
        if description:
            result["description"] = description
        results.append(result)

        if not passed:
            message = (
                f"{pool}: missing required split coverage for "
                f"{label}+{stratum}; required_splits={required_splits}; "
                f"pool_available_count={len(available_rows)}; split_counts={split_counts}"
            )
            if severity == "error":
                errors.append(message)
            else:
                warnings.append(message)

    return results, warnings, errors


def summarize_pool(rows: list[dict], policy: dict, inference_count: int) -> dict:
    reviewed_rows = rows
    supervision_rows = [row for row in rows if int(row["usable_for_supervision"]) == 1]
    label_counts = Counter(row["review_label"] for row in reviewed_rows)
    review_stratum_counts = Counter(row["review_stratum"] for row in reviewed_rows)
    supervision_strata = Counter(row["review_stratum"] for row in supervision_rows)
    split_counts = Counter(row["split_name"] for row in supervision_rows)
    seller_sets = split_seller_sets(supervision_rows)
    seller_overlap_counts = split_seller_overlap_counts(supervision_rows)
    alias_overlap_counts = split_alias_overlap_counts(supervision_rows)
    split_label_counts = {
        split: dict(Counter(row["review_label"] for row in supervision_rows if row["split_name"] == split))
        for split in SPLIT_ORDER
    }
    split_label_stratum = split_label_stratum_counts(supervision_rows)

    non_identifier_strata = set(policy["non_identifier_positive_strata"])
    positive_rows = [row for row in supervision_rows if row["review_label"] == "positive"]
    non_identifier_positive_count = sum(1 for row in positive_rows if row["review_stratum"] in non_identifier_strata)
    non_identifier_positive_share = (
        round(non_identifier_positive_count / len(positive_rows), 6) if positive_rows else None
    )

    return {
        "reviewed_row_count": len(reviewed_rows),
        "supervision_row_count": len(supervision_rows),
        "label_counts": dict(label_counts),
        "review_stratum_counts": dict(review_stratum_counts),
        "supervision_review_stratum_counts": dict(supervision_strata),
        "split_counts": dict(split_counts),
        "split_seller_counts": {split: len(seller_sets[split]) for split in SPLIT_ORDER},
        "split_seller_overlap_counts": seller_overlap_counts,
        "split_alias_overlap_counts": alias_overlap_counts,
        "split_label_counts": split_label_counts,
        "split_label_stratum_counts": split_label_stratum,
        "positive_supervision_count": len(positive_rows),
        "non_identifier_positive_count": non_identifier_positive_count,
        "non_identifier_positive_share": non_identifier_positive_share,
        "soft_same_alias_reviewed_count": sum(as_int(row["soft_same_alias_continuity_bool"]) for row in reviewed_rows),
        "status_inferred_from_label_count": inference_count,
        "top_frozen_rows": [
            {
                "balanced_review_rank": row["balanced_review_rank"],
                "pair_uid": row["pair_uid"],
                "review_stratum": row["review_stratum"],
                "review_label": row["review_label"],
                "split_name": row["split_name"],
            }
            for row in reviewed_rows[:12]
        ],
    }


def main() -> None:
    policy = load_json(POLICY_PATH)
    summary = {
        "schema_path": str(POLICY_PATH.relative_to(ROOT)),
        "input_dependencies": policy["input_dependencies"],
        "output_files": {
            "frozen_labels": {pool: str(path.relative_to(ROOT)) for pool, path in OUTPUT_PATHS.items()},
            "summary": str(SUMMARY_PATH.relative_to(ROOT)),
        },
        "pool_summaries": {},
        "acceptance_checks": {},
    }

    all_finalized_rows: list[dict] = []
    finalized_by_pool: dict[str, list[dict]] = {}
    coverage_warnings: list[str] = []
    coverage_errors: list[str] = []
    for pool in QUEUE_PATHS:
        queue_rows = load_csv(QUEUE_PATHS[pool])
        pair_rows = load_csv(CANDIDATE_PATHS[pool])
        pair_map = candidate_map(pair_rows)
        reviewed_rows, issues = reviewed_rows_from_queue(queue_rows, policy, pair_map, pool)
        pair_assignments = assign_splits(
            reviewed_rows,
            policy["split_strategy"]["pool_ratios"][pool],
            policy,
            policy.get("coverage_requirements", {}).get(pool, []),
        )
        finalized_rows = finalize_rows(reviewed_rows, pair_assignments)
        pool_summary = summarize_pool(
            finalized_rows,
            policy,
            issues["status_inferred_from_label_count"],
        )
        coverage_results, pool_warnings, pool_errors = evaluate_coverage_requirements(pool, finalized_rows, policy)
        pool_summary["coverage_requirement_results"] = coverage_results
        summary["pool_summaries"][pool] = pool_summary
        finalized_by_pool[pool] = finalized_rows
        coverage_warnings.extend(pool_warnings)
        coverage_errors.extend(pool_errors)
        all_finalized_rows.extend(finalized_rows)

    non_identifier_strata = set(policy["non_identifier_positive_strata"])
    positive_supervision_rows = [
        row for row in all_finalized_rows
        if int(row["usable_for_supervision"]) == 1 and row["review_label"] == "positive"
    ]
    non_identifier_positive_count = sum(
        1 for row in positive_supervision_rows if row["review_stratum"] in non_identifier_strata
    )
    non_identifier_positive_share = (
        round(non_identifier_positive_count / len(positive_supervision_rows), 6)
        if positive_supervision_rows else None
    )

    summary["acceptance_checks"] = {
        "no_same_alias_in_supervision": not any(
            row["candidate_scope"] == "same_alias_identity_continuity" and int(row["usable_for_supervision"]) == 1
            for row in all_finalized_rows
        ),
        "no_soft_same_alias_in_supervision": not any(
            as_int(row["soft_same_alias_continuity_bool"]) == 1 and int(row["usable_for_supervision"]) == 1
            for row in all_finalized_rows
        ),
        "all_frozen_rows_have_reviewer_id": all(bool(row["reviewer_id"]) for row in all_finalized_rows),
        "all_supervision_rows_have_split_name": all(
            bool(row["split_name"]) for row in all_finalized_rows if int(row["usable_for_supervision"]) == 1
        ),
        "no_seller_overlap_across_supervision_splits": all(
            overlap_count == 0
            for pool, pool_summary in summary["pool_summaries"].items()
            if pool != "zh_target_aux"
            for overlap_count in pool_summary.get("split_seller_overlap_counts", {}).values()
        ),
        "no_normalized_alias_overlap_across_supervision_splits": all(
            overlap_count == 0
            for pool, pool_summary in summary["pool_summaries"].items()
            if pool != "zh_target_aux"
            for overlap_count in pool_summary.get("split_alias_overlap_counts", {}).values()
        ),
        "global_positive_supervision_count": len(positive_supervision_rows),
        "global_non_identifier_positive_count": non_identifier_positive_count,
        "global_non_identifier_positive_share": non_identifier_positive_share,
        "non_identifier_positive_share_pass": (
            True
            if non_identifier_positive_share is None
            else non_identifier_positive_share >= 0.3
        ),
        "coverage_requirement_warning_count": len(coverage_warnings),
        "coverage_requirement_error_count": len(coverage_errors),
        "coverage_requirements_pass": len(coverage_errors) == 0,
        "coverage_requirement_warnings": coverage_warnings,
        "coverage_requirement_errors": coverage_errors,
    }

    with SUMMARY_PATH.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)

    if coverage_errors:
        raise SystemExit(
            "Step 5 freeze coverage requirements failed:\n- " + "\n- ".join(coverage_errors)
        )

    for pool, finalized_rows in finalized_by_pool.items():
        write_csv(OUTPUT_PATHS[pool], finalized_rows, policy["frozen_output_fields"])

    print(f"Wrote {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
