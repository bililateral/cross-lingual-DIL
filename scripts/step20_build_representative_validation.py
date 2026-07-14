#!/usr/bin/env python3
"""Build a score-blind, component-disjoint validation overlay for Step15-v7."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_POLICY = ROOT / "schema" / "step15_v7_two_stage_policy.json"


def resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def load_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_hash(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, ensure_ascii=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def deterministic_rank(component_id: str, seed: int) -> str:
    return hashlib.sha256(f"{seed}|{component_id}".encode("utf-8")).hexdigest()


def render_csv(rows: list[dict], fields: list[str]) -> bytes:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return ("\ufeff" + buffer.getvalue()).encode("utf-8")


def write_fail_closed(path: Path, payload: bytes, allow_identical_replay: bool) -> str:
    expected = hashlib.sha256(payload).hexdigest()
    if path.exists():
        observed = sha256(path)
        if allow_identical_replay and observed == expected:
            return "identical_replay_noop"
        raise FileExistsError(f"Refusing to overwrite Step15-v7 split artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_bytes(payload)
    temporary.replace(path)
    return "created"


def eligible(row: dict) -> bool:
    return (
        row.get("review_label") in {"positive", "negative"}
        and row.get("usable_for_supervision") == "1"
        and row.get("usable_for_core_transfer") == "1"
    )


def evidence_counts(rows: list[dict]) -> Counter[str]:
    return Counter(row["evidence_type"] for row in rows)


def attach_seller_graph_components(rows: list[dict]) -> None:
    parent: dict[str, str] = {}

    def find(value: str) -> str:
        parent.setdefault(value, value)
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    def union(left: str, right: str) -> None:
        root_left = find(left)
        root_right = find(right)
        if root_left == root_right:
            return
        if root_left < root_right:
            parent[root_right] = root_left
        else:
            parent[root_left] = root_right

    for row in rows:
        union(str(row["seller_uid_left"]), str(row["seller_uid_right"]))
    sellers_by_root: dict[str, list[str]] = defaultdict(list)
    for seller in sorted(parent):
        sellers_by_root[find(seller)].append(seller)
    component_id = {
        seller: f"v7comp_{canonical_hash(sellers)[:16]}"
        for sellers in sellers_by_root.values()
        for seller in sellers
    }
    for row in rows:
        left_component = component_id[str(row["seller_uid_left"])]
        right_component = component_id[str(row["seller_uid_right"])]
        if left_component != right_component:
            raise AssertionError("Seller graph union failed to close a pair component")
        row["v7_component_id"] = left_component


def select_components(
    train_rows: list[dict],
    current_valid_rows: list[dict],
    minimums: dict[str, int],
    minimum_component_counts: dict[str, int],
    minimum_remaining_train_counts: dict[str, int],
    seed: int,
) -> tuple[set[str], dict]:
    current = evidence_counts(current_valid_rows)
    deficits = {key: max(0, int(target) - current.get(key, 0)) for key, target in minimums.items()}
    current_components: dict[str, set[str]] = defaultdict(set)
    for row in current_valid_rows:
        current_components[row["evidence_type"]].add(row["v7_component_id"])
    component_deficits = {
        key: max(0, int(target) - len(current_components.get(key, set())))
        for key, target in minimum_component_counts.items()
    }
    by_component: dict[str, list[dict]] = defaultdict(list)
    for row in train_rows:
        component = str(row.get("v7_component_id", "")).strip()
        if not component:
            raise ValueError(f"Missing split_component_id for {row.get('pair_uid')}")
        by_component[component].append(row)

    selected: set[str] = set()
    total_train_counts = evidence_counts(train_rows)
    moved_counts: Counter[str] = Counter()
    trace = []
    while any(value > 0 for value in deficits.values()) or any(
        value > 0 for value in component_deficits.values()
    ):
        candidates = []
        for component, rows in by_component.items():
            if component in selected:
                continue
            counts = evidence_counts(rows)
            if any(
                total_train_counts.get(key, 0) - moved_counts.get(key, 0) - counts.get(key, 0)
                < minimum
                for key, minimum in minimum_remaining_train_counts.items()
            ):
                continue
            row_gain = sum(min(deficits[key], counts.get(key, 0)) for key in deficits)
            component_gain = sum(
                1
                for key, deficit in component_deficits.items()
                if deficit > 0 and counts.get(key, 0) > 0
            )
            gain = row_gain + 3 * component_gain
            if gain <= 0:
                continue
            spillover = len(rows) - row_gain
            target_coverage = sum(1 for key in deficits if deficits[key] > 0 and counts.get(key, 0) > 0)
            candidates.append(
                (
                    spillover / max(gain, 1),
                    -target_coverage,
                    -gain,
                    len(rows),
                    deterministic_rank(component, seed),
                    component,
                    counts,
                )
            )
        if not candidates:
            raise ValueError(f"Cannot satisfy representative validation evidence deficits: {deficits}")
        _, _, _, _, _, component, counts = min(candidates)
        selected.add(component)
        moved_counts.update(counts)
        before = dict(deficits)
        component_before = dict(component_deficits)
        for key in deficits:
            deficits[key] = max(0, deficits[key] - counts.get(key, 0))
        for key in component_deficits:
            if counts.get(key, 0) > 0:
                component_deficits[key] = max(0, component_deficits[key] - 1)
        trace.append(
            {
                "component_id": component,
                "row_count": len(by_component[component]),
                "evidence_counts": dict(sorted(counts.items())),
                "deficits_before": before,
                "deficits_after": dict(deficits),
                "component_deficits_before": component_before,
                "component_deficits_after": dict(component_deficits),
            }
        )
    return selected, {
        "initial_valid_counts": dict(sorted(current.items())),
        "initial_valid_component_counts": {
            key: len(value) for key, value in sorted(current_components.items())
        },
        "selection_trace": trace,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", default=str(DEFAULT_POLICY))
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--allow-identical-replay", action="store_true")
    args = parser.parse_args()

    policy_path = resolve(args.policy)
    policy = json.loads(policy_path.read_text(encoding="utf-8"))
    cfg = policy["representative_validation"]
    if not cfg.get("never_read_model_scores") or not cfg.get("move_complete_train_components_only"):
        raise ValueError("Representative validation must be score-blind and component-complete")
    pool_cfg = policy["pools"][cfg["source_pool"]]
    labels_path = resolve(pool_cfg["frozen_labels"])
    evidence_path = resolve(pool_cfg["evidence_labels"])
    label_rows = [row for row in load_csv(labels_path) if eligible(row)]
    evidence_index = {row["pair_uid"]: row for row in load_csv(evidence_path)}
    missing = [row["pair_uid"] for row in label_rows if row["pair_uid"] not in evidence_index]
    if missing:
        raise ValueError(f"Evidence labels are missing supervised pairs; first={missing[0]}")
    rows = []
    for label in label_rows:
        evidence = evidence_index[label["pair_uid"]]
        rows.append({**label, "evidence_type": evidence["evidence_type"]})
    attach_seller_graph_components(rows)
    original_component_splits: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        original_component_splits[row["v7_component_id"]].add(row["split_name"])
    original_leaks = {
        key: sorted(value) for key, value in original_component_splits.items() if len(value) > 1
    }
    if original_leaks:
        first = next(iter(original_leaks.items()))
        raise ValueError(
            "Existing frozen supervision has seller-connected components across original splits; "
            f"count={len(original_leaks)} first={first}"
        )
    train_rows = [row for row in rows if row["split_name"] == "train"]
    valid_rows = [row for row in rows if row["split_name"] == "valid"]
    test_rows = [row for row in rows if row["split_name"] == "test"]
    selected_components, diagnostics = select_components(
        train_rows,
        valid_rows,
        {key: int(value) for key, value in cfg["minimum_total_evidence_counts"].items()},
        {
            key: int(value)
            for key, value in cfg.get("minimum_distinct_component_counts", {}).items()
        },
        {
            key: int(value)
            for key, value in cfg.get("minimum_remaining_train_counts", {}).items()
        },
        int(cfg["seed"]),
    )

    output_rows = []
    for row in sorted(rows, key=lambda item: item["pair_uid"]):
        original = row["split_name"]
        if original == "test":
            v7_split = "internal_development_test"
            reason = "frozen_current_test_role_renamed_only"
        elif original == "valid":
            v7_split = "valid"
            reason = "retained_original_valid"
        elif row["v7_component_id"] in selected_components:
            v7_split = "valid"
            reason = "score_blind_component_transfer_for_evidence_coverage"
        else:
            v7_split = "train"
            reason = "retained_train"
        output_rows.append(
            {
                "pair_uid": row["pair_uid"],
                "split_component_id": row["split_component_id"],
                "v7_component_id": row["v7_component_id"],
                "seller_uid_left": row["seller_uid_left"],
                "seller_uid_right": row["seller_uid_right"],
                "review_label": row["review_label"],
                "evidence_type": row["evidence_type"],
                "original_split_name": original,
                "v7_split_name": v7_split,
                "assignment_reason": reason,
            }
        )

    component_splits: dict[str, set[str]] = defaultdict(set)
    seller_splits: dict[str, set[str]] = defaultdict(set)
    for row in output_rows:
        component_splits[row["v7_component_id"]].add(row["v7_split_name"])
        seller_splits[row["seller_uid_left"]].add(row["v7_split_name"])
        seller_splits[row["seller_uid_right"]].add(row["v7_split_name"])
    component_leaks = {key: sorted(value) for key, value in component_splits.items() if len(value) > 1}
    seller_leaks = {key: sorted(value) for key, value in seller_splits.items() if len(value) > 1}
    if component_leaks or seller_leaks:
        raise ValueError(
            f"Representative validation overlay leaks components={len(component_leaks)} sellers={len(seller_leaks)}"
        )

    by_split = defaultdict(list)
    for row in output_rows:
        by_split[row["v7_split_name"]].append(row)
    final_valid_counts = evidence_counts(by_split["valid"])
    final_valid_components: dict[str, set[str]] = defaultdict(set)
    for row in by_split["valid"]:
        final_valid_components[row["evidence_type"]].add(row["v7_component_id"])
    unmet = {
        key: {"required": int(value), "observed": final_valid_counts.get(key, 0)}
        for key, value in cfg["minimum_total_evidence_counts"].items()
        if final_valid_counts.get(key, 0) < int(value)
    }
    if unmet:
        raise ValueError(f"Representative validation minimums were not met: {unmet}")
    component_unmet = {
        key: {"required": int(value), "observed": len(final_valid_components.get(key, set()))}
        for key, value in cfg.get("minimum_distinct_component_counts", {}).items()
        if len(final_valid_components.get(key, set())) < int(value)
    }
    if component_unmet:
        raise ValueError(f"Representative validation component minimums were not met: {component_unmet}")
    manifest = {
        "step": "step20_build_representative_validation",
        "version": policy["version"],
        "selection_is_model_score_blind": True,
        "current_test_used_for_selection": False,
        "current_test_role": "internal_development_test_only",
        "component_disjoint": True,
        "seller_disjoint": True,
        "selected_train_component_count": len(selected_components),
        "selected_train_components_sha256": canonical_hash(sorted(selected_components)),
        "row_counts": {key: len(value) for key, value in sorted(by_split.items())},
        "label_counts": {
            key: dict(sorted(Counter(row["review_label"] for row in value).items()))
            for key, value in sorted(by_split.items())
        },
        "evidence_counts": {
            key: dict(sorted(evidence_counts(value).items())) for key, value in sorted(by_split.items())
        },
        "minimum_total_evidence_counts": cfg["minimum_total_evidence_counts"],
        "minimum_distinct_component_counts": cfg.get("minimum_distinct_component_counts", {}),
        "minimum_remaining_train_counts": cfg.get("minimum_remaining_train_counts", {}),
        "valid_evidence_component_counts": {
            key: len(value) for key, value in sorted(final_valid_components.items())
        },
        "diagnostics": diagnostics,
        "inputs": {
            str(labels_path.relative_to(ROOT)).replace("\\", "/"): sha256(labels_path),
            str(evidence_path.relative_to(ROOT)).replace("\\", "/"): sha256(evidence_path),
        },
        "policy": str(policy_path.relative_to(ROOT)).replace("\\", "/"),
        "policy_sha256": sha256(policy_path),
        "pair_uid_sha256": canonical_hash(sorted(row["pair_uid"] for row in output_rows)),
    }
    manifest["manifest_sha256"] = canonical_hash(manifest)
    fields = list(output_rows[0])
    csv_payload = render_csv(output_rows, fields)
    manifest["assignment_csv_sha256"] = hashlib.sha256(csv_payload).hexdigest()
    manifest_payload = (json.dumps(manifest, indent=2, ensure_ascii=False) + "\n").encode("utf-8")
    if args.validate_only:
        print(json.dumps({"status": "pass", "manifest": manifest}, indent=2, ensure_ascii=False))
        return
    assignment_path = resolve(cfg["split_assignment_output"])
    manifest_path = resolve(cfg["manifest_output"])
    publication_root = assignment_path.parent
    if manifest_path.parent != publication_root:
        raise ValueError("Representative validation outputs must share one publication directory")
    if publication_root.exists():
        if (
            args.allow_identical_replay
            and assignment_path.is_file()
            and manifest_path.is_file()
            and sha256(assignment_path) == hashlib.sha256(csv_payload).hexdigest()
            and sha256(manifest_path) == hashlib.sha256(manifest_payload).hexdigest()
        ):
            actions = {"assignments": "identical_replay_noop", "manifest": "identical_replay_noop"}
            print(json.dumps({"status": "pass", "actions": actions, "manifest": manifest}, indent=2, ensure_ascii=False))
            return
        raise FileExistsError(f"Refusing to overwrite representative validation directory: {publication_root}")
    staging_root = publication_root.with_name(f".{publication_root.name}.incomplete")
    if staging_root.exists():
        raise FileExistsError(f"Incomplete representative validation directory exists: {staging_root}")
    staged_assignment = staging_root / assignment_path.name
    staged_manifest = staging_root / manifest_path.name
    actions = {
        "assignments": write_fail_closed(staged_assignment, csv_payload, False),
        "manifest": write_fail_closed(staged_manifest, manifest_payload, False),
    }
    staging_root.replace(publication_root)
    print(json.dumps({"status": "pass", "actions": actions, "manifest": manifest}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
