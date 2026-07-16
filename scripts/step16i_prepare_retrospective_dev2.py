#!/usr/bin/env python3
"""Prepare component-disjoint, score-blind retrospective zh_dev2 review queues.

This tool only prepares candidates for independent review. It never assigns an
identity label, never upgrades a candidate to gold, and never makes a
prospective-evaluation claim.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_POLICY = ROOT / "schema" / "step16i_retrospective_dev2_policy.json"


class UnionFind:
    def __init__(self) -> None:
        self.parent: dict[str, str] = {}
        self.rank: dict[str, int] = {}

    def add(self, value: str) -> None:
        if value not in self.parent:
            self.parent[value] = value
            self.rank[value] = 0

    def find(self, value: str) -> str:
        self.add(value)
        parent = self.parent[value]
        if parent != value:
            self.parent[value] = self.find(parent)
        return self.parent[value]

    def union(self, left: str, right: str) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return
        if self.rank[left_root] < self.rank[right_root]:
            left_root, right_root = right_root, left_root
        self.parent[right_root] = left_root
        if self.rank[left_root] == self.rank[right_root]:
            self.rank[left_root] += 1


def resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT.resolve())).replace("\\", "/")
    except ValueError:
        return str(path.resolve()).replace("\\", "/")


def ensure_within_workspace(path: Path, label: str) -> None:
    try:
        path.resolve().relative_to(ROOT.resolve())
    except ValueError as exc:
        raise ValueError(f"{label} must stay inside the project workspace: {path}") from exc


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def render_csv(rows: list[dict[str, Any]], fieldnames: list[str]) -> bytes:
    from io import StringIO

    buffer = StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fieldnames, extrasaction="raise")
    writer.writeheader()
    writer.writerows(rows)
    return ("\ufeff" + buffer.getvalue()).encode("utf-8")


def write_bytes_exclusive(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(payload)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_hash(*values: str) -> str:
    return hashlib.sha256("|".join(values).encode("utf-8")).hexdigest()


def normalize_scalar(value: Any) -> str:
    return str(value or "").strip()


def normalize_seller_alias(value: Any) -> str:
    return unicodedata.normalize("NFKC", normalize_scalar(value)).casefold()


def portable_seller_alias(value: Any) -> str:
    token = normalize_seller_alias(value)
    compact = token.strip("/")
    if not token or re.fullmatch(r"(?:shop/)?\d+", compact):
        return ""
    return token


def flatten_strings(value: Any) -> Iterable[str]:
    if isinstance(value, str):
        text = value.strip()
        if text:
            yield text
    elif isinstance(value, (list, tuple, set)):
        for item in value:
            yield from flatten_strings(item)


def collect_json_exclusions(
    value: Any,
    pair_keys: set[str],
    seller_keys: set[str],
    component_keys: set[str],
    pairs: set[str],
    sellers: set[str],
    components: set[str],
    seller_aliases: set[str],
) -> None:
    if isinstance(value, dict):
        entity_type = normalize_scalar(
            value.get("entity_type") or value.get("exclusion_type") or value.get("type")
        ).lower()
        entity_value = (
            value.get("entity_id")
            or value.get("entity_value")
            or value.get("exclusion_value")
            or value.get("value")
        )
        if entity_type and entity_value is not None:
            target = None
            if "pair" in entity_type:
                target = pairs
            elif "component" in entity_type:
                target = components
            elif "alias" in entity_type:
                target = seller_aliases
            elif "seller" in entity_type or "account" in entity_type:
                target = sellers
            if target is not None:
                target.update(flatten_strings(entity_value))
        for key, item in value.items():
            normalized_key = str(key).strip().lower()
            if normalized_key in pair_keys:
                pairs.update(flatten_strings(item))
            elif normalized_key in seller_keys:
                sellers.update(flatten_strings(item))
            elif normalized_key in component_keys:
                components.update(flatten_strings(item))
            collect_json_exclusions(
                item,
                pair_keys,
                seller_keys,
                component_keys,
                pairs,
                sellers,
                components,
                seller_aliases,
            )
    elif isinstance(value, list):
        for item in value:
            collect_json_exclusions(
                item,
                pair_keys,
                seller_keys,
                component_keys,
                pairs,
                sellers,
                components,
                seller_aliases,
            )


def load_permanent_exclusions(path: Path, cfg: dict[str, Any]) -> dict[str, set[str]]:
    pair_keys = {str(key).lower() for key in cfg["pair_uid_keys"]}
    seller_keys = {str(key).lower() for key in cfg["seller_uid_keys"]}
    component_keys = {str(key).lower() for key in cfg["component_id_keys"]}
    pairs: set[str] = set()
    sellers: set[str] = set()
    components: set[str] = set()
    seller_aliases: set[str] = set()
    if path.suffix.lower() == ".json":
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
        collect_json_exclusions(
            payload,
            pair_keys,
            seller_keys,
            component_keys,
            pairs,
            sellers,
            components,
            seller_aliases,
        )
    elif path.suffix.lower() == ".csv":
        for row in load_csv(path):
            lower_row = {str(key).lower(): value for key, value in row.items()}
            for key in pair_keys:
                pairs.update(flatten_strings(lower_row.get(key)))
            for key in seller_keys:
                sellers.update(flatten_strings(lower_row.get(key)))
            for key in component_keys:
                components.update(flatten_strings(lower_row.get(key)))
            entity_type = normalize_scalar(
                lower_row.get("entity_type")
                or lower_row.get("exclusion_type")
                or lower_row.get("type")
            ).lower()
            entity_value = (
                lower_row.get("entity_id")
                or lower_row.get("entity_value")
                or lower_row.get("exclusion_value")
                or lower_row.get("value")
            )
            if "pair" in entity_type:
                pairs.update(flatten_strings(entity_value))
            elif "component" in entity_type:
                components.update(flatten_strings(entity_value))
            elif "alias" in entity_type:
                seller_aliases.update(flatten_strings(entity_value))
            elif "seller" in entity_type or "account" in entity_type:
                sellers.update(flatten_strings(entity_value))
    else:
        raise ValueError("Permanent exclusion manifest must be JSON or CSV")
    if cfg.get("require_nonempty_seller_uids", True) and not sellers:
        raise ValueError(
            "Permanent exclusion manifest did not expose any seller UIDs; "
            "retrospective dev2 preparation fails closed"
        )
    return {
        "pair_uids": {normalize_scalar(value) for value in pairs if normalize_scalar(value)},
        "seller_uids": {
            normalize_scalar(value) for value in sellers if normalize_scalar(value)
        },
        "seller_aliases": {
            portable_seller_alias(value)
            for value in seller_aliases
            if portable_seller_alias(value)
        },
        "component_ids": {
            normalize_scalar(value) for value in components if normalize_scalar(value)
        },
    }


def split_rule_hits(value: str) -> set[str]:
    normalized = value.replace(",", "|").replace(";", "|")
    return {token.strip().lower() for token in normalized.split("|") if token.strip()}


def numeric(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def candidate_stratum(row: dict[str, str], cfg: dict[str, Any]) -> str:
    priority = normalize_scalar(row.get("review_priority")).lower()
    hits = split_rule_hits(row.get("candidate_rule_hits", ""))
    if priority in {str(value).lower() for value in cfg["high_priority_values"]}:
        return "high_priority"
    if hits & {str(value).lower() for value in cfg["template_rule_tokens"]}:
        return "template_clone"
    if (
        hits & {str(value).lower() for value in cfg["semantic_rule_tokens"]}
        or numeric(row.get("lexical_similarity")) >= float(cfg["semantic_min_lexical_similarity"])
    ):
        return "semantic_similarity"
    return "other"


def split_shared_tokens(row: dict[str, str]) -> list[tuple[str, str]]:
    tokens = []
    for raw_token in normalize_scalar(row.get("shared_contact_values")).split("||"):
        raw_token = raw_token.strip()
        if not raw_token:
            continue
        if ":" in raw_token:
            contact_type, value = raw_token.split(":", 1)
        else:
            contact_type, value = "unknown", raw_token
        tokens.append((contact_type.strip().lower(), value.strip().lower()))
    return tokens


def occurrence_index(rows: list[dict[str, str]]) -> dict[tuple[str, str, str], list[dict[str, str]]]:
    index: dict[tuple[str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        seller = normalize_scalar(row.get("seller_uid"))
        contact_type = normalize_scalar(row.get("contact_type")).lower()
        value = normalize_scalar(row.get("normalized_value")).lower()
        if seller and contact_type and value:
            index[(seller, contact_type, value)].append(row)
    return index


def raw_occurrence(row: dict[str, str]) -> dict[str, str]:
    return {
        "source_field": row.get("source_field", ""),
        "raw_value": row.get("raw_value", ""),
        "context": row.get("context", ""),
        "title_snippet": row.get("title_snippet", ""),
        "description_snippet": row.get("description_snippet", ""),
    }


def occurrence_payload(
    row: dict[str, str],
    index: dict[tuple[str, str, str], list[dict[str, str]]],
    maximum_per_side: int,
) -> tuple[str, str]:
    tokens = split_shared_tokens(row)
    if not tokens:
        return "", "not_applicable"
    payload = []
    complete = True
    left_seller = row["seller_uid_left"]
    right_seller = row["seller_uid_right"]
    for contact_type, value in tokens:
        left = index.get((left_seller, contact_type, value), [])
        right = index.get((right_seller, contact_type, value), [])
        if not left or not right:
            complete = False
        payload.append(
            {
                "token": f"{contact_type}:{value}",
                "left_occurrences": [raw_occurrence(item) for item in left[:maximum_per_side]],
                "right_occurrences": [raw_occurrence(item) for item in right[:maximum_per_side]],
            }
        )
    status = "complete" if complete else "incomplete_candidate_only"
    return json.dumps(payload, ensure_ascii=False, sort_keys=True), status


def component_id(sellers: list[str]) -> str:
    return f"retdev2_component_{stable_hash(*sellers)[:20]}"


def build_components(rows: list[dict[str, str]]) -> tuple[dict[str, list[dict[str, str]]], dict[str, list[str]]]:
    union_find = UnionFind()
    for row in rows:
        union_find.union(row["seller_uid_left"], row["seller_uid_right"])
    sellers_by_root: dict[str, set[str]] = defaultdict(set)
    for seller in union_find.parent:
        sellers_by_root[union_find.find(seller)].add(seller)
    id_by_root = {
        root: component_id(sorted(sellers)) for root, sellers in sellers_by_root.items()
    }
    rows_by_component: dict[str, list[dict[str, str]]] = defaultdict(list)
    sellers_by_component: dict[str, list[str]] = {}
    for root, sellers in sellers_by_root.items():
        sellers_by_component[id_by_root[root]] = sorted(sellers)
    for row in rows:
        cid = id_by_root[union_find.find(row["seller_uid_left"])]
        rows_by_component[cid].append(row)
    return dict(rows_by_component), sellers_by_component


def select_representatives(
    rows_by_component: dict[str, list[dict[str, str]]],
    cfg: dict[str, Any],
) -> list[dict[str, Any]]:
    seed = str(cfg["selection_seed"])
    stratum_order = [str(value) for value in cfg["stratum_order"]]
    stratum_rank = {name: index for index, name in enumerate(stratum_order)}
    representatives: list[dict[str, Any]] = []
    for cid, rows in rows_by_component.items():
        ranked = sorted(
            rows,
            key=lambda row: (
                stratum_rank.get(row["_selection_stratum"], len(stratum_rank)),
                stable_hash(seed, row["pair_uid"]),
                row["pair_uid"],
            ),
        )
        representatives.append(
            {
                "component_id": cid,
                "component_pair_count": len(rows),
                "row": ranked[0],
                "stratum": ranked[0]["_selection_stratum"],
            }
        )
    representatives.sort(
        key=lambda item: (
            stratum_rank.get(item["stratum"], len(stratum_rank)),
            stable_hash(seed, item["component_id"]),
            item["component_id"],
        )
    )
    maximum = int(cfg["maximum_selected_components"])
    selected: list[dict[str, Any]] = []
    selected_ids: set[str] = set()
    targets = {str(key): int(value) for key, value in cfg["stratum_component_targets"].items()}
    for stratum in stratum_order:
        available = [item for item in representatives if item["stratum"] == stratum]
        for item in available[: targets.get(stratum, 0)]:
            selected.append(item)
            selected_ids.add(item["component_id"])
    if len(selected) < maximum:
        for item in representatives:
            if item["component_id"] in selected_ids:
                continue
            selected.append(item)
            selected_ids.add(item["component_id"])
            if len(selected) >= maximum:
                break
    return selected[:maximum]


def reviewer_row(row: dict[str, str], blind_id: str, occurrence_json: str) -> dict[str, Any]:
    return {
        "review_index": 0,
        "blind_id": blind_id,
        "review_scope": "retrospective_development_candidate_only",
        "source_market_raw_left": row.get("source_market_raw_left", ""),
        "source_market_raw_right": row.get("source_market_raw_right", ""),
        "source_seller_raw_left": row.get("source_seller_raw_left", ""),
        "source_seller_raw_right": row.get("source_seller_raw_right", ""),
        "alias_normalized_left": row.get("alias_normalized_left", ""),
        "alias_normalized_right": row.get("alias_normalized_right", ""),
        "alias_relation": row.get("alias_relation", ""),
        "same_market_raw": row.get("same_market_raw", ""),
        "item_count_left": row.get("item_count_left", ""),
        "item_count_right": row.get("item_count_right", ""),
        "shared_contact_count": row.get("shared_contact_count", ""),
        "shared_contact_types": row.get("shared_contact_types", ""),
        "shared_contact_values": row.get("shared_contact_values", ""),
        "shared_pgp_fingerprint_count": row.get("shared_pgp_fingerprint_count", ""),
        "shared_pgp_fingerprint_values": row.get("shared_pgp_fingerprint_values", ""),
        "shared_title_count": row.get("shared_title_count", ""),
        "shared_title_values": row.get("shared_title_values", ""),
        "shared_description_count": row.get("shared_description_count", ""),
        "shared_description_values": row.get("shared_description_values", ""),
        "shared_category_count": row.get("shared_category_count", ""),
        "shared_category_values": row.get("shared_category_values", ""),
        "left_preview": row.get("left_preview", ""),
        "right_preview": row.get("right_preview", ""),
        "raw_contact_occurrences_json": occurrence_json,
        "independent_identity_decision": "",
        "evidence_type_decision": "",
        "review_confidence": "",
        "review_rationale": "",
    }


def assert_reviewer_blinding(fieldnames: list[str], forbidden_fragments: list[str]) -> None:
    violations = [
        field
        for field in fieldnames
        if any(fragment.lower() in field.lower() for fragment in forbidden_fragments)
    ]
    if violations:
        raise ValueError(f"Reviewer queue exposes forbidden fields: {violations}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", default=str(DEFAULT_POLICY))
    parser.add_argument("--step4-candidates", help="Override policy Step4 candidate CSV")
    parser.add_argument(
        "--permanent-exclusion-manifest",
        help="Override policy permanent exclusion JSON/CSV",
    )
    parser.add_argument(
        "--step3-occurrences",
        help="Override optional Step3 occurrence CSV; pass '-' to disable",
    )
    parser.add_argument("--output-dir", help="Override the isolated output directory")
    parser.add_argument("--max-components", type=int, help="Override maximum selected components")
    args = parser.parse_args()

    policy_path = resolve(args.policy)
    policy = json.loads(policy_path.read_text(encoding="utf-8"))
    if policy.get("prospective_claim_allowed") is not False:
        raise ValueError("Step16I retrospective policy must explicitly forbid prospective claims")
    input_cfg = policy["inputs"]
    step4_path = resolve(args.step4_candidates or input_cfg["step4_candidates"])
    exclusion_path = resolve(
        args.permanent_exclusion_manifest or input_cfg["permanent_exclusion_manifest"]
    )
    occurrence_value = (
        args.step3_occurrences
        if args.step3_occurrences is not None
        else input_cfg.get("step3_occurrences", "")
    )
    occurrence_path = None if occurrence_value in {"", "-", None} else resolve(occurrence_value)
    output_dir = resolve(args.output_dir or policy["outputs"]["output_directory"])
    ensure_within_workspace(output_dir, "Step16I output directory")
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite an existing Step16I output directory: {output_dir}")
    for label, path in (
        ("Step4 candidates", step4_path),
        ("permanent exclusion manifest", exclusion_path),
        *(([("Step3 occurrences", occurrence_path)]) if occurrence_path else []),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"{label} not found: {path}")

    candidates = load_csv(step4_path)
    required_fields = {"pair_uid", "seller_uid_left", "seller_uid_right"}
    missing_fields = required_fields - set(candidates[0] if candidates else {})
    if missing_fields:
        raise ValueError(f"Step4 candidates are missing required fields: {sorted(missing_fields)}")
    pair_uids = [row["pair_uid"] for row in candidates]
    if len(pair_uids) != len(set(pair_uids)):
        raise ValueError("Step4 candidates contain duplicate pair_uid values")

    exclusions = load_permanent_exclusions(exclusion_path, policy["permanent_exclusion"])
    normalized_excluded_aliases = exclusions["seller_aliases"]
    allowed_statuses = {
        str(value).strip().lower() for value in policy["selection"]["allowed_review_statuses"]
    }
    exclusion_counts: Counter[str] = Counter()
    eligible: list[dict[str, str]] = []
    excluded_components = exclusions["component_ids"]
    candidate_component_fields = policy["permanent_exclusion"].get(
        "candidate_component_fields", []
    )
    for source_row in candidates:
        row = dict(source_row)
        pair_uid = normalize_scalar(row.get("pair_uid"))
        left = normalize_scalar(row.get("seller_uid_left"))
        right = normalize_scalar(row.get("seller_uid_right"))
        reasons = []
        if not pair_uid or not left or not right or left == right:
            reasons.append("invalid_pair_identity")
        if pair_uid in exclusions["pair_uids"]:
            reasons.append("permanently_excluded_pair")
        if left in exclusions["seller_uids"] or right in exclusions["seller_uids"]:
            reasons.append("permanently_excluded_seller")
        candidate_aliases = {
            portable_seller_alias(row.get("alias_normalized_left") or row.get("source_seller_raw_left")),
            portable_seller_alias(row.get("alias_normalized_right") or row.get("source_seller_raw_right")),
        } - {""}
        if candidate_aliases & normalized_excluded_aliases:
            reasons.append("permanently_excluded_seller_alias")
        if any(
            normalize_scalar(row.get(field)) in excluded_components
            for field in candidate_component_fields
            if normalize_scalar(row.get(field))
        ):
            reasons.append("permanently_excluded_component")
        review_status = normalize_scalar(row.get("review_status")).lower()
        if review_status not in allowed_statuses:
            reasons.append("not_pending_review")
        if normalize_scalar(row.get("review_label")):
            reasons.append("existing_supervision_label")
        if reasons:
            exclusion_counts.update(set(reasons))
            continue
        row["_selection_stratum"] = candidate_stratum(row, policy["selection"])
        eligible.append(row)

    if not eligible:
        raise ValueError("No unsupervised, seller-exclusion-safe Step4 candidates remain")
    rows_by_component, sellers_by_component = build_components(eligible)
    selection_cfg = dict(policy["selection"])
    if args.max_components is not None:
        if args.max_components <= 0:
            raise ValueError("--max-components must be positive")
        selection_cfg["maximum_selected_components"] = args.max_components
    selected = select_representatives(rows_by_component, selection_cfg)
    minimum = int(selection_cfg["minimum_selected_components"])
    if len(selected) < minimum:
        raise ValueError(
            f"Too few component-disjoint retrospective candidates: required={minimum} "
            f"available={len(selected)}"
        )

    selected_component_ids = [item["component_id"] for item in selected]
    if len(selected_component_ids) != len(set(selected_component_ids)):
        raise AssertionError("Selected candidate components are not unique")
    seen_sellers: set[str] = set()
    for item in selected:
        component_sellers = set(sellers_by_component[item["component_id"]])
        if seen_sellers & component_sellers:
            raise AssertionError("Selected retrospective rows are not seller-component-disjoint")
        seen_sellers.update(component_sellers)
        row = item["row"]
        if row["pair_uid"] in exclusions["pair_uids"]:
            raise AssertionError("Selected pair intersects the permanent exclusion manifest")
        if {row["seller_uid_left"], row["seller_uid_right"]} & exclusions["seller_uids"]:
            raise AssertionError("Selected seller intersects the permanent exclusion manifest")
        selected_aliases = {
            portable_seller_alias(
                row.get("alias_normalized_left") or row.get("source_seller_raw_left")
            ),
            portable_seller_alias(
                row.get("alias_normalized_right") or row.get("source_seller_raw_right")
            ),
        } - {""}
        if selected_aliases & exclusions["seller_aliases"]:
            raise AssertionError("Selected seller alias intersects the permanent exclusion manifest")

    occurrence_rows = load_csv(occurrence_path) if occurrence_path else []
    occurrences = occurrence_index(occurrence_rows)
    occurrence_limit = int(policy["occurrence_context"]["maximum_occurrences_per_side"])
    blind_seed = str(policy["blinding"]["blind_id_seed"])
    mapping_rows = []
    queue_by_blind_id: dict[str, dict[str, Any]] = {}
    occurrence_status_counts: Counter[str] = Counter()
    stratum_counts: Counter[str] = Counter()
    for item in selected:
        row = item["row"]
        blind_id = f"retdev2_{stable_hash(blind_seed, row['pair_uid'])[:20]}"
        occurrence_json, occurrence_status = occurrence_payload(
            row, occurrences, occurrence_limit
        ) if occurrence_path else ("", "not_provided")
        occurrence_status_counts[occurrence_status] += 1
        stratum_counts[item["stratum"]] += 1
        mapping_rows.append(
            {
                "blind_id": blind_id,
                "pair_uid": row["pair_uid"],
                "candidate_component_id": item["component_id"],
                "candidate_component_seller_count": len(
                    sellers_by_component[item["component_id"]]
                ),
                "candidate_component_pair_count": item["component_pair_count"],
                "seller_uid_left": row["seller_uid_left"],
                "seller_uid_right": row["seller_uid_right"],
                "occurrence_context_status": occurrence_status,
                "retrospective_development_only": "1",
                "prospective_final_eligible": "0",
                "automatic_label_assigned": "0",
            }
        )
        queue_by_blind_id[blind_id] = reviewer_row(row, blind_id, occurrence_json)

    reviewer_fields = list(next(iter(queue_by_blind_id.values())))
    assert_reviewer_blinding(
        reviewer_fields, [str(value) for value in policy["blinding"]["forbidden_field_fragments"]]
    )
    reviewer_queues: dict[str, list[dict[str, Any]]] = {}
    for reviewer in ("reviewer_a", "reviewer_b"):
        ordered_ids = list(queue_by_blind_id)
        random.Random(str(policy["blinding"][f"{reviewer}_order_seed"])).shuffle(ordered_ids)
        queue_rows = []
        for review_index, blind_id in enumerate(ordered_ids, start=1):
            output_row = dict(queue_by_blind_id[blind_id])
            output_row["review_index"] = review_index
            queue_rows.append(output_row)
        reviewer_queues[reviewer] = queue_rows

    output_names = policy["outputs"]
    staging_dir = output_dir.with_name(f".{output_dir.name}.staging-{os.getpid()}")
    if staging_dir.exists():
        raise FileExistsError(f"Step16I staging directory already exists: {staging_dir}")
    staging_dir.mkdir(parents=True, exist_ok=False)
    mapping_path = staging_dir / output_names["blind_mapping_filename"]
    reviewer_a_path = staging_dir / output_names["reviewer_a_filename"]
    reviewer_b_path = staging_dir / output_names["reviewer_b_filename"]
    manifest_path = staging_dir / output_names["preparation_manifest_filename"]
    mapping_fields = list(mapping_rows[0])
    write_bytes_exclusive(mapping_path, render_csv(mapping_rows, mapping_fields))
    write_bytes_exclusive(
        reviewer_a_path, render_csv(reviewer_queues["reviewer_a"], reviewer_fields)
    )
    write_bytes_exclusive(
        reviewer_b_path, render_csv(reviewer_queues["reviewer_b"], reviewer_fields)
    )

    final_mapping_path = output_dir / mapping_path.name
    final_reviewer_a_path = output_dir / reviewer_a_path.name
    final_reviewer_b_path = output_dir / reviewer_b_path.name
    manifest = {
        "step": "step16i_prepare_retrospective_dev2",
        "version": policy["version"],
        "scientific_scope": "retrospective_internal_development_review_queue",
        "prospective_claim_allowed": False,
        "automatic_identity_labels_assigned": False,
        "component_selection_unit": "eligible_step4_candidate_seller_connected_component",
        "one_pair_per_selected_component": True,
        "seller_component_disjoint_verified": True,
        "inputs": {
            "policy": {"path": display_path(policy_path), "sha256": sha256(policy_path)},
            "step4_candidates": {"path": display_path(step4_path), "sha256": sha256(step4_path)},
            "permanent_exclusion_manifest": {
                "path": display_path(exclusion_path),
                "sha256": sha256(exclusion_path),
            },
            "step3_occurrences": (
                {"path": display_path(occurrence_path), "sha256": sha256(occurrence_path)}
                if occurrence_path
                else None
            ),
        },
        "permanent_exclusion_counts": {
            "pair_uids": len(exclusions["pair_uids"]),
            "seller_uids": len(exclusions["seller_uids"]),
            "seller_aliases": len(exclusions["seller_aliases"]),
            "component_ids": len(exclusions["component_ids"]),
        },
        "candidate_counts": {
            "step4_rows": len(candidates),
            "eligible_unsupervised_rows": len(eligible),
            "eligible_candidate_components": len(rows_by_component),
            "selected_pairs": len(selected),
            "selected_components": len(selected),
            "selected_sellers_in_full_components": len(seen_sellers),
            "excluded_reason_counts": dict(sorted(exclusion_counts.items())),
            "selected_stratum_counts": dict(sorted(stratum_counts.items())),
            "occurrence_context_status_counts": dict(sorted(occurrence_status_counts.items())),
        },
        "selection": {
            "maximum_selected_components": int(selection_cfg["maximum_selected_components"]),
            "stratum_order": selection_cfg["stratum_order"],
            "stratum_component_targets": selection_cfg["stratum_component_targets"],
            "deterministic_seed": selection_cfg["selection_seed"],
            "candidate_rank_used_for_selection": False,
            "model_or_step11_score_used_for_selection": False,
        },
        "blinding": {
            "opaque_blind_ids": True,
            "reviewer_orders_independent": True,
            "pair_uid_hidden_from_reviewer_queues": True,
            "candidate_rank_hidden": True,
            "candidate_rule_hits_hidden": True,
            "model_scores_hidden": True,
            "step11_graph_information_hidden": True,
            "prior_review_status_and_label_hidden": True,
            "raw_step4_evidence_present": True,
            "raw_step3_occurrence_context_present": bool(occurrence_path),
        },
        "pair_universe_sha256": hashlib.sha256(
            "\n".join(sorted(row["pair_uid"] for row in eligible)).encode("utf-8")
        ).hexdigest(),
        "selected_pair_universe_sha256": hashlib.sha256(
            "\n".join(sorted(row["pair_uid"] for row in (item["row"] for item in selected))).encode("utf-8")
        ).hexdigest(),
        "outputs": {
            "blind_mapping": {
                "path": display_path(final_mapping_path),
                "sha256": sha256(mapping_path),
                "row_count": len(mapping_rows),
            },
            "reviewer_a_queue": {
                "path": display_path(final_reviewer_a_path),
                "sha256": sha256(reviewer_a_path),
                "row_count": len(reviewer_queues["reviewer_a"]),
            },
            "reviewer_b_queue": {
                "path": display_path(final_reviewer_b_path),
                "sha256": sha256(reviewer_b_path),
                "row_count": len(reviewer_queues["reviewer_b"]),
            },
        },
    }
    write_bytes_exclusive(
        manifest_path,
        (json.dumps(manifest, indent=2, ensure_ascii=False) + "\n").encode("utf-8"),
    )
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging_dir.rename(output_dir)
    print(
        json.dumps(
            {
                "status": "prepared_retrospective_review_only",
                "output_directory": display_path(output_dir),
                "selected_pairs": len(selected),
                "selected_components": len(selected),
                "prospective_claim_allowed": False,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
