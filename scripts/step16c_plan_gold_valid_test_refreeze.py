from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LABELS = ROOT / "reports" / "step5_zh_target_strict_frozen_silver_labels.csv"
DEFAULT_EVIDENCE = ROOT / "reports" / "step15_evidence_type_labels.zh_target_strict.csv"
DEFAULT_PLAN = ROOT / "reports" / "step16c_gold_valid_test_refreeze_plan.csv"
DEFAULT_SUMMARY = ROOT / "reports" / "step16c_gold_valid_test_refreeze_plan_summary.json"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: Iterable[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def split_counts(rows: list[dict[str, str]]) -> dict[str, dict[str, int]]:
    counts: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        split_name = row.get("split_name", "")
        label = row.get("review_label", "")
        if split_name in {"train", "valid", "test"} and label in {"positive", "negative"}:
            counts[split_name][label] += 1
    return {split: dict(label_counts) for split, label_counts in sorted(counts.items())}


def seller_overlap_counts(rows: list[dict[str, str]]) -> dict[str, int]:
    sellers: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        split_name = row.get("split_name", "")
        if split_name not in {"train", "valid", "test"}:
            continue
        sellers[split_name].add(row.get("seller_uid_left", ""))
        sellers[split_name].add(row.get("seller_uid_right", ""))
    pairs = [("train", "valid"), ("train", "test"), ("valid", "test")]
    return {f"{a}__{b}": len(sellers[a] & sellers[b]) for a, b in pairs}


def row_is_silver(row: dict[str, str]) -> bool:
    return row.get("silver_train_only") == "1" or str(row.get("label_tier", "")).startswith("silver_")


class DisjointSet:
    def __init__(self) -> None:
        self.parent: dict[str, str] = {}

    def find(self, item: str) -> str:
        self.parent.setdefault(item, item)
        if self.parent[item] != item:
            self.parent[item] = self.find(self.parent[item])
        return self.parent[item]

    def union(self, left: str, right: str) -> None:
        root_left = self.find(left)
        root_right = self.find(right)
        if root_left != root_right:
            self.parent[root_right] = root_left


def build_train_seller_components(rows: list[dict[str, str]]) -> tuple[dict[str, list[dict[str, str]]], dict[str, str]]:
    """Build complete train-only connected components by seller endpoints.

    The old split_component_id is not enough for refreezing because two rows can
    share a seller while carrying different legacy component ids. Benchmark
    refreezing must move the whole seller-connected component, otherwise a
    seller can leak across train/valid/test.
    """

    dsu = DisjointSet()
    train_rows = [row for row in rows if row.get("split_name") == "train"]
    for row in train_rows:
        left = row.get("seller_uid_left", "")
        right = row.get("seller_uid_right", "")
        if left and right:
            dsu.union(left, right)

    root_to_rows: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in train_rows:
        left = row.get("seller_uid_left", "")
        right = row.get("seller_uid_right", "")
        root = dsu.find(left or right or row.get("pair_uid", ""))
        root_to_rows[root].append(row)

    root_to_component_id = {
        root: f"train_seller_component_{index:05d}" for index, root in enumerate(sorted(root_to_rows), start=1)
    }
    components = {root_to_component_id[root]: comp_rows for root, comp_rows in root_to_rows.items()}
    row_to_component = {
        row.get("pair_uid", ""): root_to_component_id[root]
        for root, comp_rows in root_to_rows.items()
        for row in comp_rows
    }
    return components, row_to_component


def component_summary(rows: list[dict[str, str]], evidence_by_pair: dict[str, str]) -> dict[str, object]:
    labels = Counter(row.get("review_label", "") for row in rows)
    strata = Counter(row.get("review_stratum", "") for row in rows)
    evidence = Counter(evidence_by_pair.get(row.get("pair_uid", ""), "missing") for row in rows)
    sellers = set()
    direct_contact_edges = 0
    for row in rows:
        sellers.add(row.get("seller_uid_left", ""))
        sellers.add(row.get("seller_uid_right", ""))
        shared_contacts = row.get("shared_contact_count") or "0"
        shared_pgp = row.get("shared_pgp_fingerprint_count") or "0"
        try:
            contact_count = int(float(shared_contacts))
        except ValueError:
            contact_count = 0
        try:
            pgp_count = int(float(shared_pgp))
        except ValueError:
            pgp_count = 0
        if contact_count > 0 or pgp_count > 0:
            direct_contact_edges += 1
    return {
        "row_count": len(rows),
        "positive_count": labels.get("positive", 0),
        "negative_count": labels.get("negative", 0),
        "seller_count": len(sellers),
        "direct_contact_edges": direct_contact_edges,
        "direct_identifier_positive_count": evidence.get("same_controller_direct_identifier", 0),
        "public_contact_or_url_noise_count": evidence.get("public_contact_or_url_noise", 0),
        "review_strata": dict(strata),
        "evidence_types": dict(evidence),
    }


def eligible_component(
    rows: list[dict[str, str]],
    *,
    label: str,
    evidence_by_pair: dict[str, str],
) -> bool:
    if any(row.get("split_name") != "train" for row in rows):
        return False
    if any(row_is_silver(row) for row in rows):
        return False
    labels = {row.get("review_label", "") for row in rows}
    if labels != {label}:
        return False
    if label == "positive":
        # Keep only original/gold rows. The script is intentionally not allowed
        # to promote Step16B weak positives into validation or test.
        return all(not row_is_silver(row) for row in rows) and any(
            evidence_by_pair.get(row.get("pair_uid", ""), "").startswith("same_controller")
            for row in rows
        )
    return True


def greedy_select(
    components: list[tuple[str, list[dict[str, str]], dict[str, object]]],
    needed: int,
    count_key: str,
) -> tuple[list[str], int]:
    selected: list[str] = []
    total = 0
    for comp_id, _rows, summary in components:
        if total >= needed:
            break
        selected.append(comp_id)
        total += int(summary[count_key])
    return selected, total


def greedy_select_with_quota(
    components: list[tuple[str, list[dict[str, str]], dict[str, object]]],
    needed: int,
    count_key: str,
    quota_key: str,
    quota_needed: int,
) -> tuple[list[str], int, int]:
    selected: list[str] = []
    selected_set: set[str] = set()
    total = 0
    quota_total = 0

    for comp_id, _rows, summary in components:
        if total >= needed or quota_total >= quota_needed:
            break
        quota_value = int(summary.get(quota_key, 0))
        if quota_value <= 0:
            continue
        selected.append(comp_id)
        selected_set.add(comp_id)
        total += int(summary[count_key])
        quota_total += quota_value

    for comp_id, _rows, summary in components:
        if total >= needed:
            break
        if comp_id in selected_set:
            continue
        selected.append(comp_id)
        selected_set.add(comp_id)
        total += int(summary[count_key])
        quota_total += int(summary.get(quota_key, 0))

    return selected, total, quota_total


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Plan a component-safe Chinese gold valid/test refreeze by moving only "
            "existing original train gold rows. This script never promotes silver rows."
        )
    )
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--evidence", type=Path, default=DEFAULT_EVIDENCE)
    parser.add_argument("--target-valid-positive", type=int, default=30)
    parser.add_argument("--target-valid-negative", type=int, default=90)
    parser.add_argument("--target-test-positive", type=int, default=50)
    parser.add_argument("--target-test-negative", type=int, default=150)
    parser.add_argument("--valid-direct-positive-move-quota", type=int, default=0)
    parser.add_argument("--test-direct-positive-move-quota", type=int, default=10)
    parser.add_argument("--valid-public-noise-negative-move-quota", type=int, default=4)
    parser.add_argument("--test-public-noise-negative-move-quota", type=int, default=4)
    parser.add_argument("--output-plan", type=Path, default=DEFAULT_PLAN)
    parser.add_argument("--output-summary", type=Path, default=DEFAULT_SUMMARY)
    args = parser.parse_args()

    rows = read_csv(args.labels)
    evidence_rows = read_csv(args.evidence) if args.evidence.exists() else []
    evidence_by_pair = {
        row.get("pair_uid", ""): row.get("evidence_type", "missing") for row in evidence_rows
    }

    current_counts = split_counts(rows)
    required = {
        "valid": {
            "positive": max(0, args.target_valid_positive - current_counts.get("valid", {}).get("positive", 0)),
            "negative": max(0, args.target_valid_negative - current_counts.get("valid", {}).get("negative", 0)),
        },
        "test": {
            "positive": max(0, args.target_test_positive - current_counts.get("test", {}).get("positive", 0)),
            "negative": max(0, args.target_test_negative - current_counts.get("test", {}).get("negative", 0)),
        },
    }

    by_component, row_to_component = build_train_seller_components(rows)

    positive_components: list[tuple[str, list[dict[str, str]], dict[str, object]]] = []
    negative_components: list[tuple[str, list[dict[str, str]], dict[str, object]]] = []
    for comp_id, comp_rows in by_component.items():
        if eligible_component(comp_rows, label="positive", evidence_by_pair=evidence_by_pair):
            summary = component_summary(comp_rows, evidence_by_pair)
            positive_components.append((comp_id, comp_rows, summary))
        elif eligible_component(comp_rows, label="negative", evidence_by_pair=evidence_by_pair):
            summary = component_summary(comp_rows, evidence_by_pair)
            negative_components.append((comp_id, comp_rows, summary))

    positive_components.sort(
        key=lambda item: (
            int(item[2]["direct_contact_edges"]) > 0,
            int(item[2]["positive_count"]),
            int(item[2]["seller_count"]),
            item[0],
        ),
        reverse=True,
    )
    negative_components.sort(
        key=lambda item: (
            int(item[2]["negative_count"]),
            int(item[2]["seller_count"]),
            item[0],
        ),
        reverse=True,
    )

    selected: dict[str, str] = {}
    plan_rows: list[dict[str, object]] = []

    direct_positive_quotas = {
        "valid": args.valid_direct_positive_move_quota,
        "test": args.test_direct_positive_move_quota,
    }
    public_noise_quotas = {
        "valid": args.valid_public_noise_negative_move_quota,
        "test": args.test_public_noise_negative_move_quota,
    }

    for target_split in ["test", "valid"]:
        needed = required[target_split]["positive"]
        available = [item for item in positive_components if item[0] not in selected]
        comps, total, quota_total = greedy_select_with_quota(
            available,
            needed,
            "positive_count",
            "direct_identifier_positive_count",
            direct_positive_quotas[target_split],
        )
        for comp_id in comps:
            selected[comp_id] = target_split
        required[target_split]["positive_selected"] = total
        required[target_split]["direct_identifier_positive_selected"] = quota_total

    for target_split in ["valid", "test"]:
        needed = required[target_split]["negative"]
        available = [item for item in negative_components if item[0] not in selected]
        comps, total, quota_total = greedy_select_with_quota(
            available,
            needed,
            "negative_count",
            "public_contact_or_url_noise_count",
            public_noise_quotas[target_split],
        )
        for comp_id in comps:
            selected[comp_id] = target_split
        required[target_split]["negative_selected"] = total
        required[target_split]["public_contact_or_url_noise_selected"] = quota_total

    component_lookup = {comp_id: (comp_rows, summary) for comp_id, comp_rows, summary in positive_components + negative_components}
    for comp_id, target_split in sorted(selected.items(), key=lambda item: (item[1], item[0])):
        comp_rows, summary = component_lookup[comp_id]
        example = comp_rows[0]
        plan_rows.append(
            {
                "component_id": comp_id,
                "from_split": "train",
                "to_split": target_split,
                "row_count": summary["row_count"],
                "positive_count": summary["positive_count"],
                "negative_count": summary["negative_count"],
                "seller_count": summary["seller_count"],
                "direct_contact_edges": summary["direct_contact_edges"],
                "direct_identifier_positive_count": summary["direct_identifier_positive_count"],
                "public_contact_or_url_noise_count": summary["public_contact_or_url_noise_count"],
                "evidence_types": json.dumps(summary["evidence_types"], ensure_ascii=False, sort_keys=True),
                "review_strata": json.dumps(summary["review_strata"], ensure_ascii=False, sort_keys=True),
                "example_pair_uid": example.get("pair_uid", ""),
                "example_shared_contact_values": example.get("shared_contact_values", ""),
            }
        )

    simulated_rows: list[dict[str, str]] = []
    for row in rows:
        updated = dict(row)
        target_split = selected.get(row_to_component.get(row.get("pair_uid", ""), ""))
        if target_split:
            updated["split_name"] = target_split
        simulated_rows.append(updated)

    fieldnames = [
        "component_id",
        "from_split",
        "to_split",
        "row_count",
        "positive_count",
        "negative_count",
        "seller_count",
        "direct_contact_edges",
        "direct_identifier_positive_count",
        "public_contact_or_url_noise_count",
        "evidence_types",
        "review_strata",
        "example_pair_uid",
        "example_shared_contact_values",
    ]
    write_csv(args.output_plan, plan_rows, fieldnames)

    summary = {
        "step": "step16c_gold_valid_test_refreeze_plan",
        "mode": "plan_only_no_label_file_modified",
        "input_labels": str(args.labels.relative_to(ROOT) if args.labels.is_absolute() else args.labels),
        "input_evidence": str(args.evidence.relative_to(ROOT) if args.evidence.is_absolute() else args.evidence),
        "targets": {
            "valid": {"positive": args.target_valid_positive, "negative": args.target_valid_negative},
            "test": {"positive": args.target_test_positive, "negative": args.target_test_negative},
        },
        "minimum_move_quotas": {
            "valid_direct_identifier_positive": args.valid_direct_positive_move_quota,
            "test_direct_identifier_positive": args.test_direct_positive_move_quota,
            "valid_public_contact_or_url_noise_negative": args.valid_public_noise_negative_move_quota,
            "test_public_contact_or_url_noise_negative": args.test_public_noise_negative_move_quota,
        },
        "current_split_counts": current_counts,
        "required_moves_from_train": required,
        "eligible_component_counts": {
            "positive": len(positive_components),
            "negative": len(negative_components),
        },
        "selected_component_count": len(selected),
        "selected_row_counts": dict(Counter(row["to_split"] for row in plan_rows)),
        "simulated_split_counts": split_counts(simulated_rows),
        "simulated_seller_overlap_counts": seller_overlap_counts(simulated_rows),
        "hard_rules": {
            "silver_rows_allowed_in_valid_or_test": False,
            "moves_only_from_current_train": True,
            "moves_whole_split_components": True,
            "script_does_not_apply_changes": True,
        },
        "outputs": {
            "plan_csv": str(args.output_plan.relative_to(ROOT) if args.output_plan.is_absolute() else args.output_plan),
            "summary_json": str(args.output_summary.relative_to(ROOT) if args.output_summary.is_absolute() else args.output_summary),
        },
    }
    write_json(args.output_summary, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
