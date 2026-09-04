#!/usr/bin/env python3
"""Build controller-isolated synthetic English source supervision.

The identity topology is real full-PGP reuse evidence.  The model-visible text is
real Agora seller text repartitioned into synthetic accounts.  Consequently the
result is training augmentation, not recovered observations from missing markets.
"""

from __future__ import annotations

import csv
import difflib
import hashlib
import itertools
import json
import math
import os
import platform
import re
import shutil
import sys
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Callable, Iterable, Sequence

ROOT = Path(__file__).resolve().parents[1]
POLICY_PATH = ROOT / "schema" / "step7_v6_synthetic_english_source_policy.json"
BASE_BUILDER_PATH = ROOT / "scripts" / "step7_v5_build_english_source_dataset.py"
sys.path.insert(0, str(ROOT / "scripts"))
import step7_v5_build_english_source_dataset as base  # noqa: E402


class SyntheticSourceError(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(payload: object) -> str:
    return hashlib.sha256(
        json.dumps(
            payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()


def load_policy() -> dict:
    policy = json.loads(POLICY_PATH.read_text(encoding="utf-8"))
    for relative_path, expected in policy["inputs"].items():
        path = ROOT / relative_path
        if not path.is_file():
            raise SyntheticSourceError(f"Missing frozen input: {relative_path}")
        observed = sha256_file(path)
        if observed != expected:
            raise SyntheticSourceError(
                f"Frozen input drift: {relative_path}; expected={expected}; observed={observed}"
            )
    return policy


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, fieldnames: Sequence[str], rows: Iterable[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})


def write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def build_global_identity_components(
    registry: dict[tuple[str, str], dict],
    vendor_fingerprints: dict[str, set[str]],
    weak_rows: list[dict[str, str]],
    strong_rows: list[dict[str, str]],
) -> dict:
    """Close every frozen identity-evidence bridge before candidate filtering."""

    evidence_by_key: dict[tuple[str, str], dict[str, set[str]]] = defaultdict(
        lambda: {
            "aux_fingerprints": set(),
            "weak_fingerprints": set(),
            "strong_fingerprints": set(),
            "strong_key_aliases": set(),
        }
    )
    for key in registry:
        evidence_by_key[key]["aux_fingerprints"].update(
            vendor_fingerprints.get(key[1], set())
        )
    for row in weak_rows:
        key = (row["market_id"], row["vendor_id"])
        fingerprint = base.normalize_fingerprint(row.get("fingerprint", ""))
        if fingerprint:
            evidence_by_key[key]["weak_fingerprints"].add(fingerprint)
    for row in strong_rows:
        key = (row["market_id"], row["vendor_id"])
        fingerprint = base.normalize_fingerprint(row.get("fingerprint", ""))
        if fingerprint:
            evidence_by_key[key]["strong_fingerprints"].add(fingerprint)
        key_alias = base.normalize_alias(row.get("key_alias", ""))
        if key_alias:
            evidence_by_key[key]["strong_key_aliases"].add(key_alias)

    account_uid_by_key = {
        key: base.stable_hash("step7-v6-global-identity-account-v1", *key)
        for key in evidence_by_key
    }
    graph = {
        account_uid_by_key[key]: evidence for key, evidence in evidence_by_key.items()
    }
    component_by_uid = base.build_identity_conflict_components(graph)
    component_by_key = {
        key: component_by_uid[uid] for key, uid in account_uid_by_key.items()
    }
    component_keys: dict[str, set[tuple[str, str]]] = defaultdict(set)
    component_tokens: dict[str, set[tuple[str, str]]] = defaultdict(set)
    for key, evidence in evidence_by_key.items():
        component_uid = component_by_key[key]
        component_keys[component_uid].add(key)
        for field, evidence_type in (
            ("aux_fingerprints", "auxiliary_pgp"),
            ("weak_fingerprints", "weak_pgp"),
            ("strong_fingerprints", "strong_pgp"),
            ("strong_key_aliases", "strong_key_alias"),
        ):
            component_tokens[component_uid].update(
                (evidence_type, value) for value in evidence[field] if value
            )
    return {
        "evidence_by_key": evidence_by_key,
        "component_by_key": component_by_key,
        "component_keys": component_keys,
        "component_tokens": component_tokens,
        "account_count": len(evidence_by_key),
        "component_count": len(component_keys),
    }


def recover_v5_holdout_fingerprints(
    construction: dict,
    registry: dict[tuple[str, str], dict],
    global_identity: dict,
) -> tuple[set[str], set[str], set[tuple[str, str]], set[str], set[str]]:
    pair_path = (
        ROOT
        / "reports"
        / "step7_v5_english_source_dataset"
        / "v3_20260903"
        / "public_pairs.csv"
    )
    public_accounts: set[str] = set()
    for row in read_csv(pair_path):
        public_accounts.update((row["account_left_uid"], row["account_right_uid"]))

    matched_aliases: set[str] = set()
    matched_public_accounts: set[str] = set()
    matched_keys: set[tuple[str, str]] = set()
    for key, registry_row in registry.items():
        market_id, _vendor_id = key
        if market_id != "1":
            continue
        account_uid = base.stable_hash(
            "step7-v5-en-source-account-v1", "agora", registry_row["user_name"]
        )
        if account_uid in public_accounts:
            matched_aliases.add(registry_row["user_name"])
            matched_public_accounts.add(account_uid)
            matched_keys.add(key)
    if matched_public_accounts != public_accounts:
        raise SyntheticSourceError(
            "Could not recover every V5 V3 public account for disjointness"
        )
    component_by_key = global_identity["component_by_key"]
    if not matched_keys <= set(component_by_key):
        raise SyntheticSourceError(
            "Could not place every V5 V3 public account in the identity graph"
        )
    holdout_components = {component_by_key[key] for key in matched_keys}
    component_aliases = {
        registry[key]["user_name"]
        for component_uid in holdout_components
        for key in global_identity["component_keys"][component_uid]
        if key in registry and registry[key]["user_name"]
    }
    fingerprints: set[str] = set()
    evidence_tokens: set[tuple[str, str]] = set()
    for component_uid in holdout_components:
        tokens = global_identity["component_tokens"][component_uid]
        evidence_tokens.update(tokens)
        fingerprints.update(
            value for evidence_type, value in tokens if evidence_type != "strong_key_alias"
        )
    return (
        fingerprints,
        matched_aliases,
        evidence_tokens,
        component_aliases,
        holdout_components,
    )


def build_valid_topology_groups(
    policy: dict,
    strong_rows: list[dict[str, str]],
    registry: dict[tuple[str, str], dict],
    fingerprint_vendor_ids: dict[str, set[str]],
    vendor_fingerprints: dict[str, set[str]],
    global_identity: dict,
    holdout_fingerprints: set[str],
    holdout_evidence_tokens: set[tuple[str, str]],
    holdout_components: set[str],
) -> tuple[dict[str, list[dict]], dict]:
    allowed_markets = set(policy["construction"]["topology_market_ids"])
    groups: dict[str, dict[tuple[str, str], dict]] = defaultdict(dict)
    exclusions: Counter[str] = Counter()
    all_strong_fingerprints: set[str] = set()
    strong_by_account: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in strong_rows:
        if row["market_id"] in allowed_markets:
            strong_by_account[(row["market_id"], row["vendor_id"])].append(row)
    component_by_key = global_identity["component_by_key"]
    component_tokens = global_identity["component_tokens"]
    for row in strong_rows:
        fingerprint = base.normalize_fingerprint(row.get("fingerprint", ""))
        if fingerprint:
            all_strong_fingerprints.add(fingerprint)
        if row["market_id"] not in allowed_markets:
            exclusions["market_not_allowed"] += 1
            continue
        registry_row = registry.get((row["market_id"], row["vendor_id"]))
        if registry_row is None:
            exclusions["no_registry_row"] += 1
            continue
        if registry_row["user_name"] != base.normalize_alias(row["user_name"]):
            exclusions["registry_alias_mismatch"] += 1
            continue
        if registry_row["imposter"]:
            exclusions["imposter"] += 1
            continue
        if len(fingerprint) != 40:
            exclusions["invalid_full_fingerprint"] += 1
            continue
        auxiliary = vendor_fingerprints.get(row["vendor_id"], set())
        if len(auxiliary) != 1:
            exclusions[
                "no_auxiliary_fingerprint" if not auxiliary else "multiple_auxiliary_fingerprints"
            ] += 1
            continue
        if (
            fingerprint not in auxiliary
            or row["vendor_id"] not in fingerprint_vendor_ids.get(fingerprint, set())
        ):
            exclusions["cross_source_fingerprint_mismatch"] += 1
            continue
        account_key = (row["market_id"], row["vendor_id"])
        component_uid = component_by_key.get(account_key)
        if component_uid is None:
            exclusions["no_global_identity_component"] += 1
            continue
        groups[fingerprint][account_key] = {
            "market_id": row["market_id"],
            "vendor_id": row["vendor_id"],
            "conflict_component_uid": component_uid,
            "conflict_evidence_tokens": set(component_tokens[component_uid]),
        }

    valid_before_holdout = {
        fingerprint: list(accounts.values())
        for fingerprint, accounts in groups.items()
        if len(accounts) >= 2
    }
    valid = {}
    holdout_component_groups_excluded = 0
    for fingerprint, accounts in valid_before_holdout.items():
        components = {row["conflict_component_uid"] for row in accounts}
        if len(components) != 1:
            raise SyntheticSourceError(
                "One topology fingerprint spans multiple conflict components"
            )
        component_uid = next(iter(components))
        if (
            fingerprint in holdout_fingerprints
            or component_uid in holdout_components
            or component_tokens[component_uid] & holdout_evidence_tokens
        ):
            holdout_component_groups_excluded += 1
            continue
        valid[fingerprint] = accounts
    maximum_accounts = policy["construction"]["maximum_accounts_per_controller"]
    eligible_by_component: dict[str, list[tuple[str, list[dict]]]] = defaultdict(list)
    for fingerprint, accounts in valid.items():
        retained_accounts = sorted(
            accounts,
            key=lambda row: base.stable_hash(
                policy["construction"]["ordering_seed"],
                "topology-account",
                fingerprint,
                row["market_id"],
                row["vendor_id"],
            ),
        )[:maximum_accounts]
        component_uid = retained_accounts[0]["conflict_component_uid"]
        eligible_by_component[component_uid].append((fingerprint, retained_accounts))
    eligible: dict[str, list[dict]] = {}
    multiple_group_component_exclusions = 0
    for component_uid, values in eligible_by_component.items():
        fingerprint, accounts = min(
            values,
            key=lambda value: base.stable_hash(
                policy["construction"]["ordering_seed"],
                "topology-component-group",
                component_uid,
                value[0],
            ),
        )
        eligible[fingerprint] = accounts
        multiple_group_component_exclusions += len(values) - 1
    retained_fingerprints = sorted(
        eligible,
        key=lambda fingerprint: base.stable_hash(
            policy["construction"]["ordering_seed"], "topology-retain", fingerprint
        ),
    )[: policy["construction"]["controller_group_count"]]
    selected = {fingerprint: eligible[fingerprint] for fingerprint in retained_fingerprints}
    selected_components = {
        accounts[0]["conflict_component_uid"] for accounts in selected.values()
    }
    selected_holdout_evidence_overlaps = sum(
        bool(component_tokens[component_uid] & holdout_evidence_tokens)
        for component_uid in selected_components
    )
    return selected, {
        "raw_strong_rows": len(strong_rows),
        "global_identity_accounts": global_identity["account_count"],
        "global_identity_components": global_identity["component_count"],
        "valid_groups_before_holdout_exclusion": len(valid_before_holdout),
        "holdout_components_excluded": len(holdout_components),
        "holdout_fingerprints_excluded": len(holdout_fingerprints),
        "holdout_evidence_tokens": len(holdout_evidence_tokens),
        "holdout_component_groups_excluded": holdout_component_groups_excluded,
        "multiple_groups_same_component_excluded": multiple_group_component_exclusions,
        "eligible_groups_after_holdout_exclusion": len(eligible),
        "selected_groups": len(selected),
        "selected_conflict_components": len(selected_components),
        "selected_component_reuse": len(selected) - len(selected_components),
        "selected_holdout_evidence_overlaps": selected_holdout_evidence_overlaps,
        "selected_accounts": sum(len(value) for value in selected.values()),
        "selected_positive_pairs": sum(
            len(value) * (len(value) - 1) // 2 for value in selected.values()
        ),
        "row_exclusions": dict(sorted(exclusions.items())),
        "all_strong_fingerprints": all_strong_fingerprints,
    }


def balanced_partition(
    group_sizes: dict[str, int], capacities: dict[str, int], seed: str
) -> dict[str, str]:
    if sum(capacities.values()) != len(group_sizes):
        raise SyntheticSourceError("Split capacities do not equal topology group count")
    totals = {name: 0 for name in capacities}
    counts = {name: 0 for name in capacities}
    assignment: dict[str, str] = {}
    ordered = sorted(
        group_sizes,
        key=lambda uid: (
            -(group_sizes[uid] * (group_sizes[uid] - 1) // 2),
            base.stable_hash(seed, uid),
        ),
    )
    for uid in ordered:
        options = [name for name in capacities if counts[name] < capacities[name]]
        split = min(options, key=lambda name: (totals[name], counts[name], name))
        assignment[uid] = split
        counts[split] += 1
        totals[split] += group_sizes[uid] * (group_sizes[uid] - 1) // 2
    if counts != capacities:
        raise SyntheticSourceError(f"Split count mismatch: {counts!r}")
    return assignment


def donor_identity_components(
    registry: dict[tuple[str, str], dict],
    vendor_fingerprints: dict[str, set[str]],
    global_identity: dict,
) -> tuple[dict[str, dict], dict[str, str]]:
    registry_by_alias: dict[str, list[tuple[str, dict]]] = defaultdict(list)
    for (market_id, vendor_id), row in registry.items():
        if market_id == "1":
            registry_by_alias[row["user_name"]].append((vendor_id, row))

    metadata: dict[str, dict] = {}
    for alias, values in registry_by_alias.items():
        if len(values) != 1:
            continue
        vendor_id, registry_row = values[0]
        fingerprints = vendor_fingerprints.get(vendor_id, set())
        if registry_row["imposter"] or len(fingerprints) != 1:
            continue
        key = ("1", vendor_id)
        if key not in global_identity["component_by_key"]:
            raise SyntheticSourceError("Eligible donor is absent from the global identity graph")
        evidence = global_identity["evidence_by_key"][key]
        metadata[alias] = {
            "vendor_id": vendor_id,
            "fingerprint": next(iter(fingerprints)),
            "weak_fingerprints": set(evidence["weak_fingerprints"]),
            "strong_fingerprints": set(evidence["strong_fingerprints"]),
            "strong_key_aliases": set(evidence["strong_key_aliases"]),
        }
    component_by_alias = {
        alias: global_identity["component_by_key"][("1", row["vendor_id"])]
        for alias, row in metadata.items()
    }
    return metadata, component_by_alias


def remove_within_donor_exact_duplicates(items: list[dict]) -> list[dict]:
    title_counts = Counter(item["title_key"] for item in items if item["title_key"])
    description_counts = Counter(
        item["description_key"] for item in items if item["description_key"]
    )
    cleaned = []
    for item in items:
        row = dict(item)
        if row["title_key"] and title_counts[row["title_key"]] > 1:
            row["title_clean"] = ""
            row["title_key"] = ""
        if row["description_key"] and description_counts[row["description_key"]] > 1:
            row["description_clean"] = ""
            row["description_key"] = ""
        if row["title_clean"] or row["description_clean"]:
            cleaned.append(row)
    return cleaned


def normalized_copy_text(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.casefold())


def fields_share_local_copy(left: str, right: str, width: int = 40) -> bool:
    """Return whether two normalized fields share an exact local window."""

    left_key = normalized_copy_text(left)
    right_key = normalized_copy_text(right)
    if len(left_key) < width or len(right_key) < width:
        return False
    if len(left_key) > len(right_key):
        left_key, right_key = right_key, left_key
    windows = {
        left_key[index : index + width]
        for index in range(len(left_key) - width + 1)
    }
    return any(
        right_key[index : index + width] in windows
        for index in range(len(right_key) - width + 1)
    )


def source_alias_residuals(value: str, aliases: Iterable[str], minimum_length: int) -> set[str]:
    """Detect credible source-alias forms without short-word substring false hits."""

    normalized = base.normalize_alias(value)
    hits = set()
    for alias in aliases:
        normalized_alias = base.normalize_alias(alias)
        compact = base.compact_alias(alias)
        if len(compact) < minimum_length:
            continue
        core = r"[^a-z0-9]*".join(re.escape(character) for character in compact)
        if len(compact) < 5:
            core = r"(?<![a-z0-9])" + core + r"(?![a-z0-9])"
        if re.search(core, normalized, flags=re.IGNORECASE):
            hits.add(normalized_alias)
    return hits


def shingle_containment(left: str, right: str, width: int) -> float:
    if min(len(left), len(right)) < max(40, width):
        return 0.0
    left_shingles = {left[index : index + width] for index in range(len(left) - width + 1)}
    right_shingles = {
        right[index : index + width] for index in range(len(right) - width + 1)
    }
    denominator = min(len(left_shingles), len(right_shingles))
    return len(left_shingles & right_shingles) / denominator if denominator else 0.0


def fields_are_near_duplicates(
    left: str,
    right: str,
    *,
    long_width: int,
    long_threshold: float,
    short_threshold: float = 0.85,
) -> bool:
    """Detect long near-copies and short templates without treating all short text as zero."""

    if not left or not right:
        return False
    if left == right:
        return True
    if min(len(left), len(right)) < 40:
        return (
            difflib.SequenceMatcher(None, left, right, autojunk=False).ratio()
            >= short_threshold
        )
    return shingle_containment(left, right, long_width) >= long_threshold


def account_pair_has_near_duplicate(left: dict, right: dict) -> bool:
    fields = ("title_clean", "description_clean")
    for left_field, right_field in itertools.product(fields, repeat=2):
        left_values = [
            normalized_copy_text(item[left_field])
            for item in left["items"]
            if item[left_field]
        ]
        right_values = [
            normalized_copy_text(item[right_field])
            for item in right["items"]
            if item[right_field]
        ]
        if any(
            fields_are_near_duplicates(
                left_value,
                right_value,
                long_width=10,
                long_threshold=0.90,
            )
            for left_value in left_values
            for right_value in right_values
        ):
            return True
    return False


def select_copy_distinct_items(items: list[dict], needed: int, seed: str) -> list[dict]:
    """Select a deterministic candidate pool after exact field de-duplication.

    Near copies are deliberately retained here: repeated templates are legitimate
    within one real author's account.  They are clustered first and only removed
    if they would cross two synthetic accounts derived from the same donor.
    """

    ordered = sorted(
        items,
        key=lambda item: (
            -item_token_count(item),
            base.stable_hash(seed, "copy-distinct", item["source_row_number"]),
        ),
    )
    return [dict(item) for item in ordered[:needed]]


def prepare_donors(
    policy: dict,
    registry: dict[tuple[str, str], dict],
    vendor_fingerprints: dict[str, set[str]],
    fingerprint_aliases: dict[str, set[str]],
    global_identity: dict,
    holdout_components: set[str],
    required_count: int,
) -> tuple[list[dict], dict]:
    construction = policy["construction"]
    metadata, component_by_alias = donor_identity_components(
        registry, vendor_fingerprints, global_identity
    )
    component_fingerprints: dict[str, set[str]] = defaultdict(set)
    component_aliases: dict[str, set[str]] = defaultdict(set)
    for component, tokens in global_identity["component_tokens"].items():
        identity_fingerprints = {
            value for evidence_type, value in tokens if evidence_type != "strong_key_alias"
        }
        component_fingerprints[component].update(identity_fingerprints)
        component_aliases[component].update(
            value for evidence_type, value in tokens if evidence_type == "strong_key_alias"
        )
        for fingerprint in identity_fingerprints:
            component_aliases[component].update(
                fingerprint_aliases.get(fingerprint, set())
            )
    for component, keys in global_identity["component_keys"].items():
        component_aliases[component].update(
            registry[key]["user_name"]
            for key in keys
            if key in registry and registry[key]["user_name"]
        )
    all_aliases = base.read_agora_aliases()
    safe_aliases = {
        alias
        for alias in all_aliases
        if len(base.compact_alias(alias))
        >= construction["alias_minimum_redaction_length"]
    }
    aliases_to_redact = {
        alias for aliases in component_aliases.values() for alias in aliases
    }
    source_aliases_by_account: dict[str, set[str]] = {}
    for alias, row in metadata.items():
        source_aliases = set(component_aliases[component_by_alias[alias]])
        source_aliases_by_account[alias] = source_aliases
    alias_pattern = base.compile_alias_pattern(
        aliases_to_redact, construction["alias_minimum_redaction_length"]
    )
    account_alias_patterns = {
        alias: base.compile_account_alias_pattern(
            aliases, construction["alias_minimum_redaction_length"]
        )
        for alias, aliases in source_aliases_by_account.items()
    }
    all_items, text_audit = base.read_and_clean_agora(
        alias_pattern, account_alias_patterns
    )

    forbidden_components = set(holdout_components)
    candidates_by_component: dict[str, list[dict]] = defaultdict(list)
    exclusions: Counter[str] = Counter()
    for alias, row in metadata.items():
        if alias not in safe_aliases:
            exclusions["unsafe_alias"] += 1
            continue
        component = component_by_alias[alias]
        if component in forbidden_components:
            exclusions["v5_holdout_identity_component"] += 1
            continue
        if any(
            len(base.compact_alias(source_alias))
            < construction["alias_minimum_redaction_length"]
            for source_alias in source_aliases_by_account[alias]
        ):
            exclusions["source_identity_alias_too_short_for_safe_redaction"] += 1
            continue
        clean_items = remove_within_donor_exact_duplicates(
            [
                item
                for item in all_items.get(alias, [])
                if item["title_clean"] or item["description_clean"]
            ]
        )
        # Keep the complete cleaned donor history as a reserve.  Allocation later
        # chooses the required 16/24 rows and can therefore supplement a cluster
        # instead of rejecting the author merely because the first 24 rows repeat
        # one legitimate within-account template.
        maximum_needed_items = len(clean_items)
        diverse_items = select_copy_distinct_items(
            clean_items,
            maximum_needed_items,
            base.stable_hash(construction["ordering_seed"], "diverse", component, alias),
        )
        summary = base.account_summary(diverse_items)
        if summary["item_count"] < construction["minimum_clean_items_per_donor"]:
            exclusions["too_few_copy_distinct_items"] += 1
            continue
        if summary["token_count"] < construction["minimum_clean_tokens_per_donor"]:
            exclusions["too_few_clean_tokens"] += 1
            continue
        candidates_by_component[component].append(
            {
                "alias": alias,
                "component_uid": component,
                "fingerprint": row["fingerprint"],
                "source_aliases": source_aliases_by_account[alias],
                "identity_fingerprints": set(component_fingerprints[component]),
                "identity_evidence_tokens": set(
                    global_identity["component_tokens"][component]
                ),
                "account_capacity": len(diverse_items)
                // construction["items_per_synthetic_account"],
                "items": diverse_items,
                "summary": summary,
            }
        )

    component_representatives = []
    for component, candidates in candidates_by_component.items():
        maximum_capacity = max(row["account_capacity"] for row in candidates)
        representative = min(
            [row for row in candidates if row["account_capacity"] == maximum_capacity],
            key=lambda row: base.stable_hash(
                construction["ordering_seed"],
                "donor",
                component,
                row["alias"],
            ),
        )
        representative["donor_uid"] = base.stable_hash(
            construction["donor_uid_namespace"], component
        )
        component_representatives.append(representative)
    component_representatives.sort(
        key=lambda row: base.stable_hash(
            construction["ordering_seed"], "eligible-donor", row["donor_uid"]
        )
    )
    if len(component_representatives) < required_count:
        raise SyntheticSourceError(
            f"Only {len(component_representatives)} isolated donors for {required_count} groups"
        )
    return component_representatives, {
        "text_cleaning": text_audit,
        "candidate_identity_components": len(candidates_by_component),
        "eligible_component_representatives": len(component_representatives),
        "required_donors": required_count,
        "capacity_two_or_more": sum(
            row["account_capacity"] >= 2 for row in component_representatives
        ),
        "capacity_three_or_more": sum(
            row["account_capacity"] >= 3 for row in component_representatives
        ),
        "forbidden_components": len(forbidden_components),
        "exclusions": dict(sorted(exclusions.items())),
    }


def item_token_count(item: dict) -> int:
    return len(base.text_tokens(item["title_clean"])) + len(
        base.text_tokens(item["description_clean"])
    )


def item_content_tokens(item: dict) -> set[str]:
    return {
        token
        for token in base.text_tokens(
            f"{item['title_clean']} {item['description_clean']}"
        )
        if len(token) >= 3
    }


def transferable_style_projection(value: str) -> str:
    """Map source categories to a frozen alphabet without compatibility expansion."""

    result = []
    # NFC removes decomposed-mark side channels but, unlike NFKC, does not turn
    # semantic compatibility symbols (for example TM or TEL signs) into words.
    value = unicodedata.normalize("NFC", value)
    index = 0
    while index < len(value):
        character = value[index]
        category = unicodedata.category(character)
        if category.startswith("N"):
            end = index + 1
            while end < len(value) and unicodedata.category(value[end]).startswith("N"):
                end += 1
            result.append(f"N{min(end - index, 99)}")
            index = end
        elif category.startswith(("L", "M")):
            end = index + 1
            while end < len(value) and unicodedata.category(value[end]).startswith(
                ("L", "M")
            ):
                end += 1
            letter_count = sum(
                unicodedata.category(value[position]).startswith("L")
                for position in range(index, end)
            )
            if letter_count:
                result.append(f"W{min(letter_count, 99)}")
            index = end
        elif character.isspace():
            result.append("\n" if character in "\r\n" else " ")
            index += 1
        elif character in ASCII_STYLE_LITERALS:
            result.append(character)
            index += 1
        elif category.startswith("P"):
            result.append(UNICODE_PUNCTUATION_CLASSES.get(category, "."))
            index += 1
        elif category.startswith("S"):
            result.append(UNICODE_SYMBOL_CLASSES.get(category, "*"))
            index += 1
        else:
            index += 1
    return "".join(result)


ASCII_STYLE_LITERALS = frozenset(
    " \n.,;:!?\"'()[]{}-_/\\@#%&*+=<>|~^$"
)
UNICODE_PUNCTUATION_CLASSES = {
    "Pc": "_",
    "Pd": "-",
    "Ps": "(",
    "Pe": ")",
    "Pi": "\"",
    "Pf": "\"",
    "Po": ".",
}
UNICODE_SYMBOL_CLASSES = {"Sc": "$", "Sk": "^", "Sm": "+", "So": "*"}
STYLE_PLACEHOLDER_RE = re.compile(r"[WN][1-9][0-9]?")
ADJACENT_WORD_PLACEHOLDER_RE = re.compile(r"W[1-9][0-9]?(?=W[1-9])")
ADJACENT_NUMBER_PLACEHOLDER_RE = re.compile(r"N[1-9][0-9]?(?=N[1-9])")
STYLE_MISSINGNESS_ACCOUNT_FIELDS = (
    "empty_title_item_count",
    "empty_description_item_count",
    "both_empty_item_count",
)
STYLE_SEAM_ACCOUNT_FIELDS = (
    "title_adjacent_word_placeholder_count",
    "title_adjacent_number_placeholder_count",
    "description_adjacent_word_placeholder_count",
    "description_adjacent_number_placeholder_count",
)
STYLE_LENGTH_DISTRIBUTION_ACCOUNT_FIELDS = (
    "title_tokens_per_item_mean",
    "title_tokens_per_item_std",
    "title_tokens_per_item_minimum",
    "title_tokens_per_item_maximum",
    "description_tokens_per_item_mean",
    "description_tokens_per_item_std",
    "description_tokens_per_item_minimum",
    "description_tokens_per_item_maximum",
)
STYLE_DISTRIBUTION_ACCOUNT_FIELDS = (
    *STYLE_MISSINGNESS_ACCOUNT_FIELDS,
    *STYLE_SEAM_ACCOUNT_FIELDS,
    *STYLE_LENGTH_DISTRIBUTION_ACCOUNT_FIELDS,
)


def symmetric_pair_feature_names(account_fields: Sequence[str]) -> list[str]:
    return [
        feature
        for field in account_fields
        for feature in (
            f"minimum_{field}",
            f"maximum_{field}",
            f"absolute_{field}_difference",
        )
    ]


def style_projection_residual_count(value: str) -> int:
    residuals = 0
    index = 0
    while index < len(value):
        match = STYLE_PLACEHOLDER_RE.match(value, index)
        if match is not None:
            index = match.end()
            continue
        if value[index] not in ASCII_STYLE_LITERALS:
            residuals += 1
        index += 1
    return residuals


def style_account_summary(items: list[dict]) -> dict:
    title_token_counts = [
        len(STYLE_PLACEHOLDER_RE.findall(item["title_style"])) for item in items
    ]
    description_token_counts = [
        len(STYLE_PLACEHOLDER_RE.findall(item["description_style"]))
        for item in items
    ]
    title_tokens = sum(title_token_counts)
    description_tokens = sum(description_token_counts)
    item_count = len(items)

    def distribution(values: list[int], prefix: str) -> dict:
        mean = sum(values) / len(values) if values else 0.0
        variance = (
            sum((value - mean) ** 2 for value in values) / len(values)
            if values
            else 0.0
        )
        return {
            f"{prefix}_mean": mean,
            f"{prefix}_std": math.sqrt(variance),
            f"{prefix}_minimum": min(values, default=0),
            f"{prefix}_maximum": max(values, default=0),
        }

    tokens = title_tokens + description_tokens
    return {
        "item_count": item_count,
        "token_count": tokens,
        "title_token_count": title_tokens,
        "description_token_count": description_tokens,
        "empty_title_item_count": sum(value == 0 for value in title_token_counts),
        "empty_description_item_count": sum(
            value == 0 for value in description_token_counts
        ),
        "both_empty_item_count": sum(
            title == 0 and description == 0
            for title, description in zip(
                title_token_counts, description_token_counts
            )
        ),
        "title_adjacent_word_placeholder_count": sum(
            len(ADJACENT_WORD_PLACEHOLDER_RE.findall(item["title_style"]))
            for item in items
        ),
        "title_adjacent_number_placeholder_count": sum(
            len(ADJACENT_NUMBER_PLACEHOLDER_RE.findall(item["title_style"]))
            for item in items
        ),
        "description_adjacent_word_placeholder_count": sum(
            len(
                ADJACENT_WORD_PLACEHOLDER_RE.findall(
                    item["description_style"]
                )
            )
            for item in items
        ),
        "description_adjacent_number_placeholder_count": sum(
            len(
                ADJACENT_NUMBER_PLACEHOLDER_RE.findall(
                    item["description_style"]
                )
            )
            for item in items
        ),
        **distribution(title_token_counts, "title_tokens_per_item"),
        **distribution(description_token_counts, "description_tokens_per_item"),
        "category_set": set(),
        "token_set": {
            token
            for item in items
            for field in ("title_style", "description_style")
            for token in STYLE_PLACEHOLDER_RE.findall(item[field])
        },
    }


def style_pair_covariates(left: dict, right: dict) -> dict[str, float]:
    """Model-visible nonlexical structure available to audits and matching."""

    result = base.pair_covariates(left, right)
    for field in STYLE_DISTRIBUTION_ACCOUNT_FIELDS:
        left_value = float(left[field])
        right_value = float(right[field])
        result[f"minimum_{field}"] = min(left_value, right_value)
        result[f"maximum_{field}"] = max(left_value, right_value)
        result[f"absolute_{field}_difference"] = abs(left_value - right_value)
    return result


def unified_account_style_stream(style_items: Sequence[dict]) -> str:
    """Join field-neutral author-style fragments without publishing boundaries."""

    ordered = sorted(style_items, key=lambda row: row["item_uid"])
    return " ".join(
        value
        for row in ordered
        for value in (row["title_style"], row["description_style"])
        if value
    )


def style_placeholder_units(value: str) -> list[str]:
    """Split a projected stream into ordered one-placeholder style units."""

    matches = list(STYLE_PLACEHOLDER_RE.finditer(value))
    if not matches:
        return []
    units = []
    for index, match in enumerate(matches):
        start = 0 if index == 0 else match.start()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(value)
        units.append(value[start:end])
    if any(len(STYLE_PLACEHOLDER_RE.findall(unit)) != 1 for unit in units):
        raise SyntheticSourceError("Style unit does not contain exactly one placeholder")
    return units


def budget_style_stream(value: str, budget: int) -> tuple[str, dict]:
    """Select fixed, nonoverlapping start/middle/end placeholder windows."""

    if budget < 3:
        raise SyntheticSourceError("Style stream budget must support three windows")
    units = style_placeholder_units(value)
    source_count = len(units)
    if source_count < budget:
        raise SyntheticSourceError(
            f"Style stream has {source_count} placeholders below budget {budget}"
        )
    start_length = (budget + 2) // 3
    middle_length = budget // 3
    end_length = budget - start_length - middle_length
    middle_start = start_length + (
        source_count - start_length - end_length - middle_length
    ) // 2
    ranges = (
        (0, start_length),
        (middle_start, middle_start + middle_length),
        (source_count - end_length, source_count),
    )
    if not (
        0 <= ranges[0][0] < ranges[0][1] <= ranges[1][0]
        < ranges[1][1] <= ranges[2][0] < ranges[2][1] <= source_count
    ):
        raise SyntheticSourceError("Style stream budget windows overlap or drift")
    segments = [
        "".join(units[start:end]).strip(" \t\r\n")
        for start, end in ranges
    ]
    if any(not segment for segment in segments):
        raise SyntheticSourceError("Style stream budget produced an empty segment")
    result = " ".join(segments)
    selected_count = len(STYLE_PLACEHOLDER_RE.findall(result))
    if selected_count != budget:
        raise SyntheticSourceError("Style stream budget placeholder count drift")
    return result, {
        "source_placeholder_count": source_count,
        "selected_placeholder_count": selected_count,
        "source_ranges": [list(value_range) for value_range in ranges],
    }


def account_pair_has_style_near_duplicate(left: dict, right: dict) -> bool:
    fields = ("title_style", "description_style")
    for left_field, right_field in itertools.product(fields, repeat=2):
        left_values = [
            normalized_copy_text(item[left_field])
            for item in left["style_items"]
            if item[left_field]
        ]
        right_values = [
            normalized_copy_text(item[right_field])
            for item in right["style_items"]
            if item[right_field]
        ]
        if any(
            fields_are_near_duplicates(
                left_value,
                right_value,
                long_width=10,
                long_threshold=0.90,
            )
            for left_value in left_values
            for right_value in right_values
        ):
            return True
    return False


def account_pair_has_style_local_copy(left: dict, right: dict) -> bool:
    """Check the final published streams for an exact local projected copy."""

    return fields_share_local_copy(
        left["style_stream"],
        right["style_stream"],
        width=40,
    )


def find_local_copy_donor_uids(
    accounts: dict[str, dict], accounts_by_controller: dict[str, list[str]]
) -> list[str]:
    """Find donors whose pseudo-accounts still share a long local style copy."""

    return sorted(
        {
            accounts[left_uid]["donor_uid"]
            for account_uids in accounts_by_controller.values()
            for left_uid, right_uid in itertools.combinations(account_uids, 2)
            if account_pair_has_style_local_copy(
                accounts[left_uid], accounts[right_uid]
            )
        }
    )


def clear_cross_account_exact_style_fields(public_items: list[dict]) -> dict:
    owners: dict[str, set[str]] = defaultdict(set)
    for item in public_items:
        for field in ("title_style", "description_style"):
            if item[field]:
                owners[item[field]].add(item["account_uid"])
    shared = {
        key for key, account_uids in owners.items() if len(account_uids) > 1
    }
    cleared = Counter()
    for item in public_items:
        for field in ("title_style", "description_style"):
            if item[field] in shared:
                item[field] = ""
                cleared[field] += 1
    return {
        "shared_style_values_removed": len(shared),
        "title_rows_cleared": cleared["title_style"],
        "description_rows_cleared": cleared["description_style"],
    }


def clear_cross_bucket_near_fields(
    buckets: list[list[dict]],
    fields: Sequence[str],
    local_copy_width: int | None = None,
) -> dict:
    """Clear a field only when its near-copy crosses two synthetic accounts.

    Buckets come from one source donor and are formed without pair labels.  All
    matches are marked before mutation, making the result independent of scan
    order.  Repetition inside a single bucket is intentionally preserved.
    """

    marked: set[tuple[int, int, str]] = set()
    match_count = Counter()
    local_copy_match_count = Counter()
    for left_index, right_index in itertools.combinations(range(len(buckets)), 2):
        left_bucket = buckets[left_index]
        right_bucket = buckets[right_index]
        for left_field, right_field in itertools.product(fields, repeat=2):
            left_values = [
                normalized_copy_text(item[left_field]) for item in left_bucket
            ]
            right_values = [
                normalized_copy_text(item[right_field]) for item in right_bucket
            ]
            for left_item_index, left_value in enumerate(left_values):
                if not left_value:
                    continue
                for right_item_index, right_value in enumerate(right_values):
                    if not right_value:
                        continue
                    near_match = fields_are_near_duplicates(
                        left_value,
                        right_value,
                        long_width=10,
                        long_threshold=0.90,
                    )
                    local_copy_match = bool(local_copy_width) and fields_share_local_copy(
                        left_value, right_value, int(local_copy_width)
                    )
                    if not near_match and not local_copy_match:
                        continue
                    marked.add((left_index, left_item_index, left_field))
                    marked.add((right_index, right_item_index, right_field))
                    field_pair = f"{left_field}->{right_field}"
                    if near_match:
                        match_count[field_pair] += 1
                    if local_copy_match:
                        local_copy_match_count[field_pair] += 1

    cleared = Counter()
    for bucket_index, item_index, field in sorted(marked):
        item = buckets[bucket_index][item_index]
        if not item[field]:
            continue
        item[field] = ""
        key_field = field.replace("_clean", "_key")
        if key_field != field and key_field in item:
            item[key_field] = ""
        cleared[field] += 1
    return {
        "cross_bucket_near_matches": sum(match_count.values()),
        "matches_by_field": dict(sorted(match_count.items())),
        "cross_bucket_local_copy_matches": sum(local_copy_match_count.values()),
        "local_copy_matches_by_field": dict(sorted(local_copy_match_count.items())),
        "rows_cleared_by_field": dict(sorted(cleared.items())),
    }


def allocate_items(
    items: list[dict],
    account_count: int,
    per_account: int,
    seed: str,
    minimum_tokens: int = 0,
) -> list[list[dict]]:
    needed = account_count * per_account
    ordered = sorted(
        items,
        key=lambda item: (
            -item_token_count(item),
            base.stable_hash(seed, "allocation-order", item["source_row_number"]),
        ),
    )
    if len(ordered) < needed:
        raise SyntheticSourceError("Donor lost required items during allocation")

    content_tokens = {
        item["source_row_number"]: item_content_tokens(item) for item in ordered
    }
    categories = {
        item["source_row_number"]: base.exact_text_key(item.get("category_clean", ""))
        for item in ordered
    }
    style_values = {
        item["source_row_number"]: {
            field: normalized_copy_text(
                transferable_style_projection(item[field.replace("_style", "_clean")])
            )
            for field in ("title_style", "description_style")
        }
        for item in ordered
    }

    def topic_similarity(left: dict, right: dict) -> float:
        left_key = left["source_row_number"]
        right_key = right["source_row_number"]
        category_score = (
            1.0
            if categories[left_key] and categories[left_key] == categories[right_key]
            else 0.0
        )
        style_similarity = sum(
            bool(style_values[left_key][field])
            and fields_are_near_duplicates(
                style_values[left_key][field],
                style_values[right_key][field],
                long_width=10,
                long_threshold=0.90,
            )
            for field in ("title_style", "description_style")
        )
        return (
            4.0 * category_score
            + base.jaccard(content_tokens[left_key], content_tokens[right_key])
            + 4.0 * style_similarity
        )

    seeds = [ordered.pop(0)]
    while len(seeds) < account_count:
        seed_item = min(
            ordered,
            key=lambda item: (
                max(topic_similarity(item, prior) for prior in seeds),
                base.stable_hash(seed, "cluster-seed", item["source_row_number"]),
            ),
        )
        seeds.append(seed_item)
        ordered.remove(seed_item)
    buckets: list[list[dict]] = [[item] for item in seeds]
    while sum(map(len, buckets)) < needed:
        choices = []
        for item in ordered:
            for index, bucket in enumerate(buckets):
                if len(bucket) >= per_account:
                    continue
                mean_similarity = sum(
                    topic_similarity(item, prior) for prior in bucket
                ) / len(bucket)
                choices.append(
                    (
                        -mean_similarity,
                        base.stable_hash(
                            seed, "cluster-assignment", item["source_row_number"], index
                        ),
                        item,
                        index,
                    )
                )
        _, _, item, target = min(choices, key=lambda value: (value[0], value[1]))
        buckets[target].append(item)
        ordered.remove(item)
    if any(len(bucket) != per_account for bucket in buckets):
        raise SyntheticSourceError("Synthetic account item count imbalance")
    for _ in range(needed):
        token_totals = [sum(item_token_count(item) for item in bucket) for bucket in buckets]
        low_indices = [
            index for index, total in enumerate(token_totals) if total < minimum_tokens
        ]
        if not low_indices:
            break
        low_index = min(low_indices, key=lambda index: (token_totals[index], index))
        swaps = []
        for low_item_index, low_item in enumerate(buckets[low_index]):
            low_tokens = item_token_count(low_item)
            for high_index, high_bucket in enumerate(buckets):
                if high_index == low_index:
                    continue
                for high_item_index, high_item in enumerate(high_bucket):
                    high_tokens = item_token_count(high_item)
                    new_low_total = token_totals[low_index] - low_tokens + high_tokens
                    new_high_total = token_totals[high_index] - high_tokens + low_tokens
                    if (
                        new_low_total <= token_totals[low_index]
                        or new_high_total < minimum_tokens
                    ):
                        continue
                    swaps.append(
                        (
                            max(0, minimum_tokens - new_low_total),
                            -new_low_total,
                            base.stable_hash(
                                seed,
                                "minimum-token-swap",
                                low_item["source_row_number"],
                                high_item["source_row_number"],
                            ),
                            "bucket",
                            low_item_index,
                            high_index,
                            high_item_index,
                        )
                    )
            for reserve_index, reserve_item in enumerate(ordered):
                reserve_tokens = item_token_count(reserve_item)
                new_low_total = token_totals[low_index] - low_tokens + reserve_tokens
                if new_low_total <= token_totals[low_index]:
                    continue
                swaps.append(
                    (
                        max(0, minimum_tokens - new_low_total),
                        -new_low_total,
                        base.stable_hash(
                            seed,
                            "minimum-token-reserve",
                            low_item["source_row_number"],
                            reserve_item["source_row_number"],
                        ),
                        "reserve",
                        low_item_index,
                        reserve_index,
                        -1,
                    )
                )
        if not swaps:
            raise SyntheticSourceError(
                "Could not satisfy the per-account minimum token requirement"
            )
        _, _, _, source, low_item_index, source_index, source_item_index = min(swaps)
        if source == "bucket":
            buckets[low_index][low_item_index], buckets[source_index][source_item_index] = (
                buckets[source_index][source_item_index],
                buckets[low_index][low_item_index],
            )
        else:
            displaced = buckets[low_index][low_item_index]
            buckets[low_index][low_item_index] = ordered.pop(source_index)
            ordered.append(displaced)
    if any(
        sum(item_token_count(item) for item in bucket) < minimum_tokens
        for bucket in buckets
    ):
        raise SyntheticSourceError("Synthetic account remains below the token minimum")
    return buckets


def donor_is_disjoint_from_topology(
    donor: dict, topology_accounts: Sequence[dict], fingerprint: str
) -> bool:
    topology_component_uid = topology_accounts[0]["conflict_component_uid"]
    topology_evidence_tokens = topology_accounts[0]["conflict_evidence_tokens"]
    return (
        fingerprint not in donor["identity_fingerprints"]
        and donor["component_uid"] != topology_component_uid
        and not donor["identity_evidence_tokens"] & topology_evidence_tokens
    )


def synthesize_accounts(
    policy: dict,
    topology_groups: dict[str, list[dict]],
    split_by_group: dict[str, str],
    donors: list[dict],
    allocation_cache: dict[tuple[str, int], tuple | None] | None = None,
    allocation_rejections: Counter | None = None,
) -> tuple[dict[str, dict], dict[str, list[str]], list[dict], dict]:
    construction = policy["construction"]
    group_order = sorted(
        topology_groups,
        key=lambda fingerprint: (
            -len(topology_groups[fingerprint]),
            base.stable_hash(
                construction["ordering_seed"], "topology-group", fingerprint
            ),
        ),
    )
    donor_order = sorted(
        donors,
        key=lambda row: base.stable_hash(
            construction["ordering_seed"], "donor-map", row["donor_uid"]
        ),
    )
    mapped_allocations = []
    if allocation_cache is None:
        allocation_cache = {}
    if allocation_rejections is None:
        allocation_rejections = Counter()
    unused = {row["donor_uid"]: row for row in donor_order}
    for fingerprint in group_order:
        topology_accounts = topology_groups[fingerprint]
        required_capacity = len(topology_accounts)
        choices = sorted(
            (
                donor
                for donor in unused.values()
                if donor["account_capacity"] >= required_capacity
                and donor_is_disjoint_from_topology(
                    donor, topology_accounts, fingerprint
                )
            ),
            key=lambda row: base.stable_hash(
                construction["ordering_seed"],
                "donor-assignment",
                fingerprint,
                row["donor_uid"],
            ),
        )
        selected = None
        for donor in choices:
            cache_key = (donor["donor_uid"], required_capacity)
            if cache_key not in allocation_cache:
                try:
                    buckets = allocate_items(
                        donor["items"],
                        required_capacity,
                        construction["items_per_synthetic_account"],
                        base.stable_hash(
                            construction["ordering_seed"], donor["donor_uid"]
                        ),
                        construction["minimum_clean_tokens_per_synthetic_account"],
                    )
                except SyntheticSourceError:
                    allocation_rejections["clean_token_allocation"] += 1
                    allocation_cache[cache_key] = None
                    continue
                buckets = [[dict(item) for item in bucket] for bucket in buckets]
                raw_cleanup = clear_cross_bucket_near_fields(
                    buckets, ("title_clean", "description_clean")
                )
                if any(
                    sum(item_token_count(item) for item in bucket)
                    < construction["minimum_clean_tokens_per_synthetic_account"]
                    for bucket in buckets
                ):
                    allocation_rejections["clean_tokens_after_near_cleanup"] += 1
                    allocation_cache[cache_key] = None
                    continue
                style_probe_buckets = [
                    [
                        {
                            "title_style": transferable_style_projection(
                                item["title_clean"]
                            ),
                            "description_style": transferable_style_projection(
                                item["description_clean"]
                            ),
                        }
                        for item in bucket
                    ]
                    for bucket in buckets
                ]
                clear_cross_bucket_near_fields(
                    style_probe_buckets,
                    ("title_style", "description_style"),
                    local_copy_width=40,
                )
                style_probe_summaries = [
                    style_account_summary(bucket) for bucket in style_probe_buckets
                ]
                if any(
                    summary["token_count"]
                    < construction["minimum_style_tokens_per_synthetic_account"]
                    for summary in style_probe_summaries
                ):
                    allocation_rejections["style_tokens_after_near_cleanup"] += 1
                    allocation_cache[cache_key] = None
                    continue
                allocation_cache[cache_key] = (buckets, raw_cleanup)
            cached = allocation_cache[cache_key]
            if cached is None:
                continue
            buckets, raw_cleanup = cached
            selected = (donor, buckets, raw_cleanup)
            break
        if selected is None:
            raise SyntheticSourceError(
                f"No fully allocatable capacity-{required_capacity} donor remains "
                "for topology group"
            )
        donor, _, _ = selected
        mapped_allocations.append(selected)
        del unused[donor["donor_uid"]]
    accounts: dict[str, dict] = {}
    accounts_by_controller: dict[str, list[str]] = defaultdict(list)
    public_items: list[dict] = []
    source_rows_seen: set[tuple[str, int]] = set()
    source_row_reuse = 0
    low_token_accounts = 0
    donor_topology_identity_collisions = 0
    donor_topology_component_collisions = 0
    donor_topology_evidence_overlaps = 0
    raw_near_cleanup_matches = Counter()
    raw_near_cleanup_rows = Counter()
    style_near_cleanup_matches = Counter()
    style_local_copy_cleanup_matches = Counter()
    style_near_cleanup_rows = Counter()
    for fingerprint, (donor, buckets, raw_cleanup) in zip(
        group_order, mapped_allocations
    ):
        if fingerprint in donor["identity_fingerprints"]:
            donor_topology_identity_collisions += 1
        topology_accounts = topology_groups[fingerprint]
        topology_component_uid = topology_accounts[0]["conflict_component_uid"]
        topology_evidence_tokens = topology_accounts[0]["conflict_evidence_tokens"]
        donor_topology_component_collisions += (
            donor["component_uid"] == topology_component_uid
        )
        donor_topology_evidence_overlaps += bool(
            donor["identity_evidence_tokens"] & topology_evidence_tokens
        )
        controller_uid = base.stable_hash(
            construction["controller_uid_namespace"], fingerprint
        )
        raw_near_cleanup_matches.update(raw_cleanup["matches_by_field"])
        raw_near_cleanup_rows.update(raw_cleanup["rows_cleared_by_field"])
        controller_public_buckets: list[list[dict]] = []
        for role, (topology_account, bucket) in enumerate(zip(topology_accounts, buckets)):
            account_uid = base.stable_hash(
                construction["account_uid_namespace"], controller_uid, role
            )
            summary = base.account_summary(bucket)
            if summary["token_count"] < construction["minimum_clean_tokens_per_synthetic_account"]:
                low_token_accounts += 1
            accounts[account_uid] = {
                "account_uid": account_uid,
                "controller_uid": controller_uid,
                "donor_uid": donor["donor_uid"],
                "donor_component_uid": donor["component_uid"],
                "source_aliases": donor["source_aliases"],
                "source_fingerprint": donor["fingerprint"],
                "topology_market_id": topology_account["market_id"],
                "role": role,
                "split": split_by_group[fingerprint],
                "items": bucket,
                "summary": summary,
            }
            accounts_by_controller[controller_uid].append(account_uid)
            public_bucket = []
            for item in bucket:
                source_key = (donor["donor_uid"], int(item["source_row_number"]))
                if source_key in source_rows_seen:
                    source_row_reuse += 1
                source_rows_seen.add(source_key)
                item_uid = base.stable_hash(
                    construction["item_uid_namespace"], account_uid, item["source_row_number"]
                )
                public_bucket.append(
                    {
                        "account_uid": account_uid,
                        "item_uid": item_uid,
                        "split": split_by_group[fingerprint],
                        "title_clean": item["title_clean"],
                        "description_clean": item["description_clean"],
                        "title_style": transferable_style_projection(
                            item["title_clean"]
                        ),
                        "description_style": transferable_style_projection(
                            item["description_clean"]
                        ),
                        "_source_row_number": int(item["source_row_number"]),
                        "_donor_uid": donor["donor_uid"],
                    }
                )
            controller_public_buckets.append(public_bucket)
        style_cleanup = clear_cross_bucket_near_fields(
            controller_public_buckets,
            ("title_style", "description_style"),
            local_copy_width=40,
        )
        style_near_cleanup_matches.update(style_cleanup["matches_by_field"])
        style_local_copy_cleanup_matches.update(
            style_cleanup["local_copy_matches_by_field"]
        )
        style_near_cleanup_rows.update(style_cleanup["rows_cleared_by_field"])
        for public_bucket in controller_public_buckets:
            public_items.extend(public_bucket)
    for values in accounts_by_controller.values():
        values.sort()
    style_deduplication = clear_cross_account_exact_style_fields(public_items)
    style_items_by_account: dict[str, list[dict]] = defaultdict(list)
    for item in public_items:
        style_items_by_account[item["account_uid"]].append(item)
    style_budget = construction["style_stream_budget"]
    low_style_token_donor_uids = set()
    style_stream_budget_mismatches = 0
    style_stream_budget_audit = Counter()
    for uid, style_items in style_items_by_account.items():
        accounts[uid]["style_items"] = style_items
        raw_stream = unified_account_style_stream(style_items)
        raw_summary = style_account_summary(
            [{"title_style": raw_stream, "description_style": ""}]
        )
        accounts[uid]["raw_style_summary"] = raw_summary
        if (
            raw_summary["token_count"]
            < construction["minimum_style_tokens_per_synthetic_account"]
        ):
            low_style_token_donor_uids.add(accounts[uid]["donor_uid"])
            accounts[uid]["style_stream"] = raw_stream
            accounts[uid]["style_summary"] = raw_summary
            style_stream_budget_mismatches += 1
            continue
        budgeted_stream, budget_audit = budget_style_stream(
            raw_stream, style_budget["total_placeholders"]
        )
        accounts[uid]["style_stream"] = budgeted_stream
        accounts[uid]["style_summary"] = style_account_summary(
            [{"title_style": budgeted_stream, "description_style": ""}]
        )
        style_stream_budget_audit.update(
            {
                "accounts_budgeted": 1,
                "source_placeholders": budget_audit[
                    "source_placeholder_count"
                ],
                "selected_placeholders": budget_audit[
                    "selected_placeholder_count"
                ],
            }
        )
        if (
            accounts[uid]["style_summary"]["token_count"]
            != style_budget["total_placeholders"]
        ):
            style_stream_budget_mismatches += 1
    local_copy_donor_uids = find_local_copy_donor_uids(
        accounts, accounts_by_controller
    )
    low_style_token_accounts = sum(
        accounts[uid]["raw_style_summary"]["token_count"]
        < construction["minimum_style_tokens_per_synthetic_account"]
        for uid in accounts
    )
    low_clean_token_donor_uids = sorted(
        {
            accounts[uid]["donor_uid"]
            for uid in accounts
            if accounts[uid]["summary"]["token_count"]
            < construction["minimum_clean_tokens_per_synthetic_account"]
        }
    )
    return accounts, accounts_by_controller, public_items, {
        "source_item_reuse": source_row_reuse,
        "low_token_accounts": low_token_accounts,
        "source_rows_used": len(source_rows_seen),
        "eligible_donor_pool": len(donor_order),
        "selected_donors": len(mapped_allocations),
        "evaluated_donor_capacity_combinations": len(allocation_cache),
        "allocation_rejections": dict(sorted(allocation_rejections.items())),
        "donor_assignment": "capacity-first deterministic identity derangement with cached post-cleaning feasibility",
        "negative_matching": (
            "independent capacity-one minimum-cost two-factor matching inside "
            "each split, role-pair and controller-size cell with exact endpoint "
            "degrees; the frozen 26-feature distance only selects among feasible "
            "negative edges"
        ),
        "donor_topology_identity_collisions": donor_topology_identity_collisions,
        "donor_topology_component_collisions": donor_topology_component_collisions,
        "donor_topology_evidence_overlaps": donor_topology_evidence_overlaps,
        "style_deduplication": style_deduplication,
        "raw_cross_account_near_cleanup": {
            "matches_by_field": dict(sorted(raw_near_cleanup_matches.items())),
            "rows_cleared_by_field": dict(sorted(raw_near_cleanup_rows.items())),
        },
        "style_cross_account_near_cleanup": {
            "matches_by_field": dict(sorted(style_near_cleanup_matches.items())),
            "local_copy_matches_by_field": dict(
                sorted(style_local_copy_cleanup_matches.items())
            ),
            "rows_cleared_by_field": dict(sorted(style_near_cleanup_rows.items())),
        },
        "low_style_token_accounts": low_style_token_accounts,
        "low_style_token_donor_uids": sorted(low_style_token_donor_uids),
        "style_stream_budget_mismatches": style_stream_budget_mismatches,
        "style_stream_budget": {
            "selection": style_budget["selection"],
            "total_placeholders_per_account": style_budget[
                "total_placeholders"
            ],
            **dict(sorted(style_stream_budget_audit.items())),
        },
        "low_clean_token_donor_uids": low_clean_token_donor_uids,
        "local_copy_donor_uids": local_copy_donor_uids,
    }


def positive_pairs(
    policy: dict,
    accounts: dict[str, dict],
    accounts_by_controller: dict[str, list[str]],
) -> list[dict]:
    rows = []
    for controller_uid, account_uids in sorted(accounts_by_controller.items()):
        positive_count = len(account_uids) * (len(account_uids) - 1) // 2
        for left_uid, right_uid in itertools.combinations(account_uids, 2):
            left = accounts[left_uid]
            right = accounts[right_uid]
            rows.append(
                {
                    "left_uid": left_uid,
                    "right_uid": right_uid,
                    "anchor_controller_uid": controller_uid,
                    "split": left["split"],
                    "label": 1,
                    "sample_weight": 1.0 / positive_count,
                    "role_pair": tuple(sorted((left["role"], right["role"]))),
                    "raw_pair_key": base.stable_hash("step7-v6-raw-pair", left_uid, right_uid),
                    "covariates": style_pair_covariates(
                        left["style_summary"], right["style_summary"]
                    ),
                    "full_covariates": base.pair_covariates(
                        left["summary"], right["summary"]
                    ),
                }
            )
    return rows


def standardized_pair_cost(
    target: dict,
    candidate: dict,
    scales: dict,
    weights: dict,
) -> float:
    """Measure one candidate edge against its target in standardized units."""

    return sum(
        weights[name]
        * ((target[name] - candidate[name]) / scales[name]) ** 2
        for name in weights
    )


def select_negative_pairs(
    policy: dict,
    accounts: dict[str, dict],
    accounts_by_controller: dict[str, list[str]],
    positives: list[dict],
) -> tuple[list[dict], dict]:
    """Build deterministic capacity-one negatives with exact account degrees.

    Matching stays inside each split, role-pair and controller-size cell.  The
    fixed 26 model-visible covariates are used only to choose among otherwise
    valid degree-balanced edges; labels never affect account text.
    """

    try:
        import numpy as np
        from scipy.optimize import Bounds, LinearConstraint, milp
        from scipy.sparse import coo_matrix
    except ImportError as error:
        raise SyntheticSourceError(
            "NumPy and SciPy are required for negative matching"
        ) from error

    construction = policy["construction"]
    ratio = construction["negative_per_positive"]
    weights = construction["negative_matching_feature_weights"]
    feature_names = list(weights)
    account_by_controller_role = {
        controller_uid: {accounts[uid]["role"]: uid for uid in account_uids}
        for controller_uid, account_uids in accounts_by_controller.items()
    }
    positive_cells: dict[
        tuple[str, tuple[int, int], int], list[dict]
    ] = defaultdict(list)
    for row in positives:
        group_size = len(accounts_by_controller[row["anchor_controller_uid"]])
        positive_cells[(row["split"], row["role_pair"], group_size)].append(row)

    selected: list[dict] = []
    cell_diagnostics = []
    for cell_key, unsorted_targets in sorted(positive_cells.items()):
        split, role_pair, group_size = cell_key
        targets = sorted(unsorted_targets, key=lambda row: row["raw_pair_key"])
        target_count = len(targets)
        if target_count <= ratio:
            raise SyntheticSourceError(
                f"Negative derangement cell {cell_key} is too small"
            )
        left_role, right_role = role_pair
        left_uids = [
            account_by_controller_role[row["anchor_controller_uid"]][left_role]
            for row in targets
        ]
        right_uids = [
            account_by_controller_role[row["anchor_controller_uid"]][right_role]
            for row in targets
        ]
        candidate_covariates = {
            (left_index, right_index): style_pair_covariates(
                accounts[left_uid]["style_summary"],
                accounts[right_uid]["style_summary"],
            )
            for left_index, left_uid in enumerate(left_uids)
            for right_index, right_uid in enumerate(right_uids)
            if left_index != right_index
        }
        scales = {}
        for name in feature_names:
            values = np.asarray(
                [row["covariates"][name] for row in targets]
                + [row[name] for row in candidate_covariates.values()],
                dtype=np.float64,
            )
            scale = float(values.std())
            scales[name] = scale if scale > 1e-12 else 1.0

        forbidden_near_copy_edges: set[tuple[int, int]] = set()
        solve_rounds = 0
        while True:
            solve_rounds += 1
            available_edges = sorted(
                edge
                for edge in candidate_covariates
                if edge not in forbidden_near_copy_edges
            )
            if not available_edges:
                raise SyntheticSourceError(
                    f"No feasible negative edges for {cell_key}"
                )
            rows = []
            columns = []
            values = []
            raw_costs = []
            for edge_index, (left_index, right_index) in enumerate(
                available_edges
            ):
                rows.extend((left_index, target_count + right_index))
                columns.extend((edge_index, edge_index))
                values.extend((1.0, 1.0))
                raw_costs.append(
                    standardized_pair_cost(
                        targets[left_index]["covariates"],
                        candidate_covariates[(left_index, right_index)],
                        scales,
                        weights,
                    )
                )
            incidence = coo_matrix(
                (
                    np.asarray(values, dtype=np.float64),
                    (
                        np.asarray(rows, dtype=np.int32),
                        np.asarray(columns, dtype=np.int32),
                    ),
                ),
                shape=(2 * target_count, len(available_edges)),
            ).tocsc()
            maximum_cost = max(raw_costs, default=0.0)
            objective = np.asarray(raw_costs, dtype=np.float64) / (
                1.0 + maximum_cost
            )
            objective += np.asarray(
                [
                    int(
                        base.stable_hash(
                            construction["ordering_seed"],
                            "cell-negative-matching",
                            split,
                            role_pair,
                            group_size,
                            left_index,
                            right_index,
                        )[:12],
                        16,
                    )
                    / (16**12)
                    * 1e-9
                    for left_index, right_index in available_edges
                ],
                dtype=np.float64,
            )
            result = milp(
                c=objective,
                integrality=np.ones(len(available_edges), dtype=np.uint8),
                bounds=Bounds(
                    np.zeros(len(available_edges), dtype=np.float64),
                    np.ones(len(available_edges), dtype=np.float64),
                ),
                constraints=LinearConstraint(
                    incidence,
                    np.full(2 * target_count, float(ratio), dtype=np.float64),
                    np.full(2 * target_count, float(ratio), dtype=np.float64),
                ),
                options={"presolve": True, "mip_rel_gap": 0.0},
            )
            if not result.success or result.x is None:
                raise SyntheticSourceError(
                    f"Negative matching failed for {cell_key}: {result.message}"
                )
            rounded = np.rint(result.x)
            if float(np.max(np.abs(result.x - rounded))) > 1e-7:
                raise SyntheticSourceError(
                    f"Negative matching was nonintegral for {cell_key}"
                )
            chosen_indices = [
                index
                for index, value in enumerate(rounded.tolist())
                if int(value) == 1
            ]
            if len(chosen_indices) != target_count * ratio:
                raise SyntheticSourceError(
                    f"Negative matching edge count drift for {cell_key}"
                )
            chosen_edges = [available_edges[index] for index in chosen_indices]
            newly_forbidden = {
                edge
                for edge in chosen_edges
                if account_pair_has_near_duplicate(
                    accounts[left_uids[edge[0]]],
                    accounts[right_uids[edge[1]]],
                )
            }
            if not newly_forbidden:
                break
            forbidden_near_copy_edges.update(newly_forbidden)

        for left_index, right_index in chosen_edges:
            target = targets[left_index]
            left_uid = left_uids[left_index]
            right_uid = right_uids[right_index]
            endpoints = tuple(sorted((left_uid, right_uid)))
            selected.append(
                {
                    "left_uid": endpoints[0],
                    "right_uid": endpoints[1],
                    "raw_pair_key": base.stable_hash(
                        "step7-v6-raw-pair", *endpoints
                    ),
                    "covariates": candidate_covariates[
                        (left_index, right_index)
                    ],
                    "full_covariates": base.pair_covariates(
                        accounts[left_uid]["summary"],
                        accounts[right_uid]["summary"],
                    ),
                    "anchor_controller_uid": target["anchor_controller_uid"],
                    "split": split,
                    "label": 0,
                    "sample_weight": target["sample_weight"] / ratio,
                    "role_pair": role_pair,
                }
            )
        cell_diagnostics.append(
            {
                "split": split,
                "role_pair": list(role_pair),
                "group_size": group_size,
                "positive_targets": target_count,
                "candidate_edges": len(candidate_covariates),
                "selected_edges": len(chosen_edges),
                "forbidden_near_copy_edges": len(forbidden_near_copy_edges),
                "solve_rounds": solve_rounds,
                "selected_standardized_distance": float(
                    sum(
                        standardized_pair_cost(
                            targets[left_index]["covariates"],
                            candidate_covariates[(left_index, right_index)],
                            scales,
                            weights,
                        )
                        for left_index, right_index in chosen_edges
                    )
                ),
            }
        )

    return selected, {
        "protocol": (
            "independent capacity-one minimum-cost two-factor matching inside "
            "each split, role-pair and controller-size cell; every endpoint has "
            "exactly two negative incidents per positive incident; frozen 26 "
            "model-visible pair covariates and a deterministic tie break select "
            "among feasible edges"
        ),
        "distance_feature_names": feature_names,
        "cell_count": len(cell_diagnostics),
        "cells": cell_diagnostics,
    }

def orient_and_identify_pairs(policy: dict, rows: list[dict]) -> list[dict]:
    construction = policy["construction"]
    result = []
    for row in rows:
        left, right = sorted((row["left_uid"], row["right_uid"]))
        if int(base.stable_hash(construction["ordering_seed"], "orientation", left, right)[:2], 16) % 2:
            left, right = right, left
        pair_uid = base.stable_hash(construction["pair_uid_namespace"], *sorted((left, right)))
        result.append({**row, "pair_uid": pair_uid, "left_uid": left, "right_uid": right})
    return sorted(
        result,
        key=lambda row: base.stable_hash(
            construction["ordering_seed"], "published-pair", row["pair_uid"]
        ),
    )


def style_positive_control(rows: list[dict], accounts: dict[str, dict]) -> dict:
    try:
        import numpy as np
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics import average_precision_score, roc_auc_score
    except ImportError as error:
        raise SyntheticSourceError("scikit-learn and NumPy are required") from error

    account_uids = sorted(accounts)
    documents = []
    for uid in account_uids:
        documents.append(accounts[uid]["style_stream"])
    vectorizer = TfidfVectorizer(
        analyzer="char", ngram_range=(3, 5), min_df=2, max_features=30000, sublinear_tf=True
    )
    matrix = vectorizer.fit_transform(documents)
    index = {uid: i for i, uid in enumerate(account_uids)}
    scores = []
    labels = []
    for row in rows:
        left = matrix[index[row["left_uid"]]]
        right = matrix[index[row["right_uid"]]]
        scores.append(float(left.multiply(right).sum()))
        labels.append(row["label"])
    labels_array = np.asarray(labels, dtype=int)
    scores_array = np.asarray(scores, dtype=float)
    weights_array = np.asarray([row["sample_weight"] for row in rows], dtype=float)
    pair_auc = float(roc_auc_score(labels_array, scores_array))
    pair_ap = float(average_precision_score(labels_array, scores_array))
    pair_prevalence = float(labels_array.mean())
    weighted_auc = float(
        roc_auc_score(labels_array, scores_array, sample_weight=weights_array)
    )
    weighted_ap = float(
        average_precision_score(
            labels_array, scores_array, sample_weight=weights_array
        )
    )
    weighted_prevalence = float(
        np.average(labels_array.astype(float), weights=weights_array)
    )
    return {
        "protocol": "deterministic character 3-5 gram TF-IDF cosine on language-neutral lexical-free style projections; frozen gate uses controller-equal sample weights",
        "feature_count": len(vectorizer.vocabulary_),
        "roc_auc": weighted_auc,
        "average_precision": weighted_ap,
        "prevalence": weighted_prevalence,
        "average_precision_lift_over_prevalence": weighted_ap - weighted_prevalence,
        "unweighted_pair_metrics": {
            "roc_auc": pair_auc,
            "average_precision": pair_ap,
            "prevalence": pair_prevalence,
            "average_precision_lift_over_prevalence": pair_ap - pair_prevalence,
        },
    }


def pair_indicator_audit(
    rows: Sequence[dict],
    accounts: dict[str, dict],
    indicator: Callable[[dict, dict], bool],
    protocol: str,
) -> dict:
    """Measure both directions of a frozen one-bit shortcut indicator."""

    try:
        import numpy as np
        from sklearn.metrics import average_precision_score, roc_auc_score
    except ImportError as error:
        raise SyntheticSourceError("scikit-learn and NumPy are required") from error

    results = {}
    for split in ("train", "development", "synthetic_audit"):
        split_rows = [row for row in rows if row["split"] == split]
        labels = np.asarray([row["label"] for row in split_rows], dtype=int)
        scores = np.asarray(
            [
                indicator(accounts[row["left_uid"]], accounts[row["right_uid"]])
                for row in split_rows
            ],
            dtype=float,
        )
        weights = np.asarray([row["sample_weight"] for row in split_rows], dtype=float)

        def metrics(sample_weight: object | None) -> dict:
            prevalence = float(
                labels.mean()
                if sample_weight is None
                else np.average(labels.astype(float), weights=sample_weight)
            )
            auc = float(roc_auc_score(labels, scores, sample_weight=sample_weight))
            inverse_auc = float(
                roc_auc_score(labels, 1.0 - scores, sample_weight=sample_weight)
            )
            ap = float(
                average_precision_score(labels, scores, sample_weight=sample_weight)
            )
            inverse_ap = float(
                average_precision_score(
                    labels, 1.0 - scores, sample_weight=sample_weight
                )
            )
            return {
                "prevalence": prevalence,
                "roc_auc": auc,
                "inverse_roc_auc": inverse_auc,
                "bidirectional_roc_auc": max(auc, inverse_auc),
                "average_precision": ap,
                "inverse_average_precision": inverse_ap,
                "maximum_average_precision_lift_over_prevalence": max(
                    ap, inverse_ap
                )
                - prevalence,
            }

        results[split] = {
            "positive_with_indicator": int(scores[labels == 1].sum()),
            "positive_count": int((labels == 1).sum()),
            "negative_with_indicator": int(scores[labels == 0].sum()),
            "negative_count": int((labels == 0).sum()),
            "unweighted_pair": metrics(None),
            "controller_equal": metrics(weights),
        }
    heldout = [results["development"], results["synthetic_audit"]]
    return {
        "protocol": protocol,
        "splits": results,
        "maximum_heldout_bidirectional_roc_auc": max(
            values[weighting]["bidirectional_roc_auc"]
            for values in heldout
            for weighting in ("unweighted_pair", "controller_equal")
        ),
        "maximum_heldout_average_precision_lift_over_prevalence": max(
            values[weighting]["maximum_average_precision_lift_over_prevalence"]
            for values in heldout
            for weighting in ("unweighted_pair", "controller_equal")
        ),
        "positive_with_indicator": sum(
            values["positive_with_indicator"] for values in results.values()
        ),
        "negative_with_indicator": sum(
            values["negative_with_indicator"] for values in results.values()
        ),
    }


def near_style_indicator_audit(rows: Sequence[dict], accounts: dict[str, dict]) -> dict:
    return pair_indicator_audit(
        rows,
        accounts,
        account_pair_has_style_near_duplicate,
        "one-bit whole-field near-style indicator; report both score directions; "
        "frozen gates use the worst controller-equal and unweighted held-out metric",
    )


def local_style_copy_indicator_audit(
    rows: Sequence[dict], accounts: dict[str, dict]
) -> dict:
    return pair_indicator_audit(
        rows,
        accounts,
        account_pair_has_style_local_copy,
        "one-bit exact normalized 40-character local-copy indicator over all four "
        "final-stream field combinations; report both score directions; frozen gates "
        "use the worst controller-equal and unweighted held-out metric",
    )


def split_proxy_audit(
    rows: Sequence[dict], feature_names: Sequence[str], protocol_name: str
) -> dict:
    """Fit on train only and measure residual shortcuts on both held-out splits."""

    try:
        import numpy as np
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import average_precision_score, roc_auc_score
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
    except ImportError as error:
        raise SyntheticSourceError(
            "NumPy and scikit-learn are required for split shortcut audits"
        ) from error

    training_rows = [row for row in rows if row["split"] == "train"]
    if not training_rows:
        raise SyntheticSourceError("Shortcut audit has no training rows")
    train_matrix = np.asarray(
        [
            [row["covariates"][name] for name in feature_names]
            for row in training_rows
        ],
        dtype=np.float64,
    )
    train_labels = np.asarray([row["label"] for row in training_rows], dtype=int)
    train_weights = np.asarray(
        [row["sample_weight"] for row in training_rows], dtype=np.float64
    )
    if len(set(train_labels.tolist())) != 2:
        raise SyntheticSourceError("Shortcut audit training split lacks both labels")
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(C=1.0, solver="lbfgs", max_iter=5000, random_state=0),
    )
    model.fit(
        train_matrix,
        train_labels,
        logisticregression__sample_weight=train_weights,
    )

    heldout = {}
    for split in ("development", "synthetic_audit"):
        split_rows = [row for row in rows if row["split"] == split]
        matrix = np.asarray(
            [
                [row["covariates"][name] for name in feature_names]
                for row in split_rows
            ],
            dtype=np.float64,
        )
        labels = np.asarray([row["label"] for row in split_rows], dtype=int)
        weights = np.asarray(
            [row["sample_weight"] for row in split_rows], dtype=np.float64
        )
        if not len(split_rows) or len(set(labels.tolist())) != 2:
            raise SyntheticSourceError(
                f"Shortcut audit held-out split {split} lacks both labels"
            )
        probabilities = model.predict_proba(matrix)[:, 1]
        pair_auc = float(roc_auc_score(labels, probabilities))
        pair_average_precision = float(
            average_precision_score(labels, probabilities)
        )
        pair_prevalence = float(labels.mean())
        weighted_auc = float(
            roc_auc_score(labels, probabilities, sample_weight=weights)
        )
        weighted_average_precision = float(
            average_precision_score(labels, probabilities, sample_weight=weights)
        )
        weighted_prevalence = float(
            np.average(labels.astype(float), weights=weights)
        )
        heldout[split] = {
            "row_count": len(split_rows),
            "positive_count": int(labels.sum()),
            "controller_equal": {
                "prevalence": weighted_prevalence,
                "roc_auc": weighted_auc,
                "bidirectional_roc_auc": max(weighted_auc, 1.0 - weighted_auc),
                "average_precision": weighted_average_precision,
                "average_precision_lift_over_prevalence": weighted_average_precision
                - weighted_prevalence,
            },
            "unweighted_pair": {
                "prevalence": pair_prevalence,
                "roc_auc": pair_auc,
                "bidirectional_roc_auc": max(pair_auc, 1.0 - pair_auc),
                "average_precision": pair_average_precision,
                "average_precision_lift_over_prevalence": pair_average_precision
                - pair_prevalence,
            },
        }
    return {
        "feature_names": list(feature_names),
        "standardized_coefficients": {
            name: float(value)
            for name, value in zip(
                feature_names,
                model.named_steps["logisticregression"].coef_[0],
            )
        },
        "protocol": protocol_name,
        "training_row_count": len(training_rows),
        "heldout_splits": heldout,
        "maximum_heldout_bidirectional_roc_auc": max(
            metrics["bidirectional_roc_auc"]
            for value in heldout.values()
            for metrics in (value["controller_equal"], value["unweighted_pair"])
        ),
        "maximum_heldout_average_precision_lift_over_prevalence": max(
            metrics["average_precision_lift_over_prevalence"]
            for value in heldout.values()
            for metrics in (value["controller_equal"], value["unweighted_pair"])
        ),
    }


def audit_dataset(
    policy: dict,
    topology_groups: dict[str, list[dict]],
    split_by_group: dict[str, str],
    accounts: dict[str, dict],
    accounts_by_controller: dict[str, list[str]],
    public_items: list[dict],
    pair_rows: list[dict],
    synthesis_audit: dict,
    topology_audit: dict,
    donor_audit: dict,
) -> dict:
    construction = policy["construction"]
    gates = policy["quality_gates"]
    identity_residuals: Counter[str] = Counter()
    known_key_ids = {
        fingerprint[-width:]
        for fingerprint in {
            *topology_audit["all_strong_fingerprints"],
            *(row["source_fingerprint"] for row in accounts.values()),
        }
        for width in (8, 16, 40)
        if len(fingerprint) >= width
    }
    contact_patterns = {
        "pgp": base.PGP_ARMOR_RE,
        "url": base.URL_RE,
        "email": base.EMAIL_RE,
        "handle": base.HANDLE_RE,
        "phone": base.PHONE_RE,
        "hex_id": base.HEX_ID_RE,
        "crypto": base.CRYPTO_RE,
    }
    exact_owners: dict[str, set[str]] = defaultdict(set)
    source_item_owners: dict[tuple[str, int], set[str]] = defaultdict(set)
    for item in public_items:
        account = accounts[item["account_uid"]]
        for field in ("title_clean", "description_clean"):
            value = item[field]
            if value:
                exact_owners[base.exact_text_key(value)].add(item["account_uid"])
            for name, pattern in contact_patterns.items():
                if pattern.search(value):
                    identity_residuals[name] += 1
            if base.source_fingerprint_residuals(value, known_key_ids):
                identity_residuals["fingerprint"] += 1
            if source_alias_residuals(
                value,
                account["source_aliases"],
                construction["alias_minimum_redaction_length"],
            ):
                identity_residuals["source_alias"] += 1
        source_item_owners[(item["_donor_uid"], item["_source_row_number"])].add(
            item["account_uid"]
        )
    cross_account_exact = sum(len(owners) > 1 for owners in exact_owners.values())
    source_item_reuse = sum(len(owners) > 1 for owners in source_item_owners.values())

    pairs_with_near_duplicate_fields = 0
    pairs_with_near_duplicate_style_fields: Counter[int] = Counter()
    for pair in pair_rows:
        left = accounts[pair["left_uid"]]
        right = accounts[pair["right_uid"]]
        pairs_with_near_duplicate_fields += account_pair_has_near_duplicate(left, right)
        pairs_with_near_duplicate_style_fields[pair["label"]] += (
            account_pair_has_style_near_duplicate(left, right)
        )

    controller_splits: dict[str, set[str]] = defaultdict(set)
    donor_splits: dict[str, set[str]] = defaultdict(set)
    for account in accounts.values():
        controller_splits[account["controller_uid"]].add(account["split"])
        donor_splits[account["donor_component_uid"]].add(account["split"])
    split_overlaps = sum(len(v) > 1 for v in controller_splits.values())
    split_overlaps += sum(len(v) > 1 for v in donor_splits.values())

    pair_labels: dict[tuple[str, str], set[int]] = defaultdict(set)
    for row in pair_rows:
        pair_labels[tuple(sorted((row["left_uid"], row["right_uid"])))].add(row["label"])
    duplicate_or_conflicting = len(pair_rows) - len(pair_labels)
    duplicate_or_conflicting += sum(len(value) > 1 for value in pair_labels.values())

    positive_count = sum(row["label"] == 1 for row in pair_rows)
    negative_count = len(pair_rows) - positive_count
    seam_features = symmetric_pair_feature_names(STYLE_SEAM_ACCOUNT_FIELDS)
    seam_feature_set = set(seam_features)
    structural_features = [
        name
        for name in construction["negative_matching_feature_weights"]
        if name not in seam_feature_set
    ]
    topic_features = ["category_jaccard", "token_jaccard"]
    structural = split_proxy_audit(
        pair_rows,
        structural_features,
        "train-fitted structural proxy evaluated separately on development and synthetic audit",
    )
    seam_proxy = split_proxy_audit(
        pair_rows,
        seam_features,
        "train-fitted adjacent-placeholder seam proxy evaluated separately on "
        "development and synthetic audit",
    )
    full_covariate_rows = [
        {**row, "covariates": row["full_covariates"]} for row in pair_rows
    ]
    topic = split_proxy_audit(
        full_covariate_rows,
        topic_features,
        "train-fitted category/token-overlap proxy evaluated separately on development and synthetic audit",
    )
    style = style_positive_control(pair_rows, accounts)
    near_style_indicator = near_style_indicator_audit(pair_rows, accounts)
    local_style_copy_indicator = local_style_copy_indicator_audit(
        pair_rows, accounts
    )
    style_value_owners: dict[str, set[str]] = defaultdict(set)
    for item in public_items:
        for field in ("title_style", "description_style"):
            if item[field]:
                style_value_owners[item[field]].add(item["account_uid"])
    cross_account_exact_style_values = sum(
        len(owners) > 1 for owners in style_value_owners.values()
    )
    style_stream_value_owners: dict[str, set[str]] = defaultdict(set)
    for account_uid, account in accounts.items():
        value = account["style_stream"]
        if value:
            style_stream_value_owners[value].add(account_uid)
    cross_account_exact_style_stream_values = sum(
        len(owners) > 1 for owners in style_stream_value_owners.values()
    )
    style_lexical_residual_hits = 0
    for item in public_items:
        for field in ("title_style", "description_style"):
            style_lexical_residual_hits += style_projection_residual_count(item[field])
    role_counts = Counter((row["label"], row["role_pair"]) for row in pair_rows)
    role_balance = all(
        role_counts[(0, role_pair)]
        == role_counts[(1, role_pair)] * construction["negative_per_positive"]
        for role_pair in {row["role_pair"] for row in pair_rows}
    )
    positive_degrees: Counter[str] = Counter()
    negative_degrees: Counter[str] = Counter()
    positive_incident_weights: dict[str, float] = defaultdict(float)
    negative_incident_weights: dict[str, float] = defaultdict(float)
    for row in pair_rows:
        degrees = positive_degrees if row["label"] == 1 else negative_degrees
        incident_weights = (
            positive_incident_weights
            if row["label"] == 1
            else negative_incident_weights
        )
        for uid in (row["left_uid"], row["right_uid"]):
            degrees[uid] += 1
            incident_weights[uid] += row["sample_weight"]
    account_degree_mismatches = sum(
        negative_degrees[uid]
        != positive_degrees[uid] * construction["negative_per_positive"]
        or abs(negative_incident_weights[uid] - positive_incident_weights[uid])
        > 1e-12
        for uid in accounts
    )

    gate_results = {
        "minimum_controller_groups": len(accounts_by_controller)
        >= gates["minimum_controller_groups"],
        "positive_pair_range": gates["minimum_positive_pairs"]
        <= positive_count
        <= gates["maximum_positive_pairs"],
        "identity_residual_hits": sum(identity_residuals.values())
        <= gates["maximum_identity_residual_hits"],
        "cross_account_exact_text_values": cross_account_exact
        <= gates["maximum_cross_account_exact_text_values"],
        "published_pairs_with_near_duplicate_fields": pairs_with_near_duplicate_fields
        <= gates["maximum_published_pairs_with_near_duplicate_fields"],
        "split_overlaps": split_overlaps <= gates["maximum_split_overlaps"],
        "selected_topology_component_reuse": topology_audit[
            "selected_component_reuse"
        ]
        <= gates["maximum_selected_topology_component_reuse"],
        "selected_topology_holdout_evidence_overlaps": topology_audit[
            "selected_holdout_evidence_overlaps"
        ]
        <= gates["maximum_selected_topology_holdout_evidence_overlaps"],
        "source_item_reuse": source_item_reuse <= gates["maximum_source_item_reuse"],
        "duplicate_or_conflicting_pairs": duplicate_or_conflicting
        <= gates["maximum_duplicate_or_conflicting_pairs"],
        "synthetic_account_minimum_tokens": synthesis_audit["low_token_accounts"] == 0,
        "donor_topology_identity_collisions": synthesis_audit[
            "donor_topology_identity_collisions"
        ]
        == 0,
        "donor_topology_component_collisions": synthesis_audit[
            "donor_topology_component_collisions"
        ]
        <= gates["maximum_donor_topology_component_collisions"],
        "donor_topology_evidence_overlaps": synthesis_audit[
            "donor_topology_evidence_overlaps"
        ]
        <= gates["maximum_donor_topology_evidence_overlaps"],
        "negative_ratio": negative_count
        == positive_count * construction["negative_per_positive"],
        "account_role_label_balance": role_balance,
        "account_incident_degree_balance": account_degree_mismatches
        <= gates["maximum_account_incident_degree_mismatches"],
        "style_projection_lexical_residual_hits": style_lexical_residual_hits
        <= gates["maximum_style_projection_lexical_residual_hits"],
        "cross_account_exact_style_values": cross_account_exact_style_values
        <= gates["maximum_cross_account_exact_style_values"],
        "cross_account_exact_style_stream_values": (
            cross_account_exact_style_stream_values
            <= gates["maximum_cross_account_exact_style_stream_values"]
        ),
        "synthetic_account_minimum_style_tokens": synthesis_audit[
            "low_style_token_accounts"
        ]
        == 0,
        "style_stream_budget": synthesis_audit[
            "style_stream_budget_mismatches"
        ]
        <= gates["maximum_style_stream_budget_mismatches"],
        "positive_near_style_pairs": near_style_indicator["positive_with_indicator"]
        <= gates["maximum_positive_pairs_with_near_duplicate_style_fields"],
        "near_style_indicator_auc": near_style_indicator[
            "maximum_heldout_bidirectional_roc_auc"
        ]
        <= gates["maximum_near_style_indicator_bidirectional_roc_auc"],
        "near_style_indicator_ap": near_style_indicator[
            "maximum_heldout_average_precision_lift_over_prevalence"
        ]
        <= gates["maximum_near_style_indicator_ap_lift_over_prevalence"],
        "positive_local_style_copy_pairs": local_style_copy_indicator[
            "positive_with_indicator"
        ]
        <= gates["maximum_positive_pairs_with_local_style_copy"],
        "local_style_copy_indicator_auc": local_style_copy_indicator[
            "maximum_heldout_bidirectional_roc_auc"
        ]
        <= gates["maximum_local_style_copy_indicator_bidirectional_roc_auc"],
        "local_style_copy_indicator_ap": local_style_copy_indicator[
            "maximum_heldout_average_precision_lift_over_prevalence"
        ]
        <= gates["maximum_local_style_copy_indicator_ap_lift_over_prevalence"],
        "style_structural_proxy_auc": structural[
            "maximum_heldout_bidirectional_roc_auc"
        ]
        <= gates["maximum_style_structural_proxy_bidirectional_roc_auc"],
        "style_structural_proxy_ap": structural[
            "maximum_heldout_average_precision_lift_over_prevalence"
        ]
        <= gates["maximum_style_structural_proxy_ap_lift_over_prevalence"],
        "style_seam_proxy_auc": seam_proxy[
            "maximum_heldout_bidirectional_roc_auc"
        ]
        <= gates["maximum_style_seam_proxy_bidirectional_roc_auc"],
        "style_seam_proxy_ap": seam_proxy[
            "maximum_heldout_average_precision_lift_over_prevalence"
        ]
        <= gates["maximum_style_seam_proxy_ap_lift_over_prevalence"],
        "style_positive_control_auc": style["roc_auc"]
        >= gates["minimum_style_positive_control_roc_auc"],
        "style_positive_control_ap": style["average_precision_lift_over_prevalence"]
        >= gates["minimum_style_positive_control_ap_lift_over_prevalence"],
    }
    return {
        "status": "PASSED" if all(gate_results.values()) else "FAILED",
        "claim_boundary": policy["claim_boundary"],
        "counts": {
            "controller_groups": len(accounts_by_controller),
            "synthetic_accounts": len(accounts),
            "source_items_used": len(public_items),
            "public_accounts_per_view": len(accounts),
            "positive_pairs": positive_count,
            "negative_pairs": negative_count,
            "total_pairs": len(pair_rows),
        },
        "split_counts": {
            split: {
                "controller_groups": sum(value == split for value in split_by_group.values()),
                "accounts": sum(row["split"] == split for row in accounts.values()),
                "positive_pairs": sum(
                    row["split"] == split and row["label"] == 1 for row in pair_rows
                ),
                "negative_pairs": sum(
                    row["split"] == split and row["label"] == 0 for row in pair_rows
                ),
            }
            for split in sorted(set(split_by_group.values()))
        },
        "identity_residual_counts": dict(sorted(identity_residuals.items())),
        "cross_account_exact_text_values": cross_account_exact,
        "published_pairs_with_near_duplicate_fields": pairs_with_near_duplicate_fields,
        "pairs_with_near_duplicate_style_fields": {
            "positive": pairs_with_near_duplicate_style_fields[1],
            "negative": pairs_with_near_duplicate_style_fields[0],
        },
        "near_style_indicator": near_style_indicator,
        "local_style_copy_indicator": local_style_copy_indicator,
        "source_item_reuse": source_item_reuse,
        "split_overlaps": split_overlaps,
        "standardized_mean_differences": {
            view: {
                "all": base.standardized_mean_differences(rows),
                **{
                    split: base.standardized_mean_differences(
                        [row for row in rows if row["split"] == split]
                    )
                    for split in sorted(set(split_by_group.values()))
                },
            }
            for view, rows in (
                ("model_visible_style", pair_rows),
                ("hidden_full_text_diagnostic", full_covariate_rows),
            )
        },
        "duplicate_or_conflicting_pairs": duplicate_or_conflicting,
        "role_pair_counts": {
            f"label_{label}_roles_{left}_{right}": count
            for (label, (left, right)), count in sorted(role_counts.items())
        },
        "account_incident_degree_mismatches": account_degree_mismatches,
        "model_visibility": {
            "training_authorized_view": "transfer_style_projection",
            "full_clean_view_published": False,
            "style_projection_lexical_residual_hits": style_lexical_residual_hits,
            "cross_account_exact_style_values": cross_account_exact_style_values,
            "cross_account_exact_style_stream_values": (
                cross_account_exact_style_stream_values
            ),
            "style_structural_proxy_status": "MODEL_VISIBLE_HARD_GATE",
            "style_seam_proxy_status": "MODEL_VISIBLE_HARD_GATE",
            "item_slot_boundaries_published": False,
            "training_authorized_granularity": (
                "field_neutral_account_style_stream"
            ),
            "fixed_style_information_budget": construction[
                "style_stream_budget"
            ],
            "full_text_topic_proxy_status": "DIAGNOSTIC_ONLY_MODEL_INVISIBLE",
        },
        "structural_proxy": structural,
        "seam_proxy": seam_proxy,
        "topic_proxy": topic,
        "style_positive_control": style,
        "topology_audit": {
            key: value
            for key, value in topology_audit.items()
            if key != "all_strong_fingerprints"
        },
        "donor_audit": donor_audit,
        "synthesis_audit": synthesis_audit,
        "gate_results": gate_results,
    }


def publish() -> dict:
    policy = load_policy()
    construction = policy["construction"]
    output = ROOT / policy["output_directory"]
    building = output.with_name(output.name + ".building")
    if output.exists():
        raise SyntheticSourceError(f"Publication already exists: {output}")
    if building.exists():
        shutil.rmtree(building)
    building.mkdir(parents=True)
    try:
        strong_rows = read_csv(ROOT / "suspected_sockpuppet_strong.csv")
        weak_rows = read_csv(ROOT / "suspected_sockpuppet_weak.csv")
        registry, registry_parse = base.parse_vendor_registry()
        (
            fingerprint_vendor_ids,
            vendor_fingerprints,
            fingerprint_aliases,
            pgp_parse,
        ) = base.parse_auxiliary_pgp()
        global_identity = build_global_identity_components(
            registry, vendor_fingerprints, weak_rows, strong_rows
        )
        (
            holdout_fingerprints,
            holdout_aliases,
            holdout_evidence_tokens,
            holdout_component_aliases,
            holdout_components,
        ) = recover_v5_holdout_fingerprints(
            construction,
            registry,
            global_identity,
        )
        topology_groups, topology_audit = build_valid_topology_groups(
            policy,
            strong_rows,
            registry,
            fingerprint_vendor_ids,
            vendor_fingerprints,
            global_identity,
            holdout_fingerprints,
            holdout_evidence_tokens,
            holdout_components,
        )
        group_sizes = {key: len(value) for key, value in topology_groups.items()}
        split_by_group = balanced_partition(
            group_sizes, construction["split_controller_counts"], construction["split_seed"]
        )
        donors, donor_audit = prepare_donors(
            policy,
            registry,
            vendor_fingerprints,
            fingerprint_aliases,
            global_identity,
            holdout_components,
            len(topology_groups),
        )
        original_donor_count = len(donors)
        excluded_style_capacity_donors: set[str] = set()
        allocation_cache: dict[tuple[str, int], tuple | None] = {}
        allocation_rejections: Counter = Counter()
        while True:
            active_donors = [
                donor
                for donor in donors
                if donor["donor_uid"] not in excluded_style_capacity_donors
            ]
            if len(active_donors) < len(topology_groups):
                raise SyntheticSourceError(
                    "Donor reserve exhausted while enforcing post-allocation text minima"
                )
            accounts, accounts_by_controller, items, synthesis_audit = (
                synthesize_accounts(
                    policy,
                    topology_groups,
                    split_by_group,
                    active_donors,
                    allocation_cache,
                    allocation_rejections,
                )
            )
            low_donors = set(
                synthesis_audit["low_style_token_donor_uids"]
            ) | set(synthesis_audit["low_clean_token_donor_uids"])
            low_donors.update(synthesis_audit["local_copy_donor_uids"])
            if not low_donors:
                break
            excluded_style_capacity_donors.update(low_donors)
        synthesis_audit["eligible_donor_pool"] = original_donor_count
        synthesis_audit["style_capacity_donors_excluded"] = len(
            excluded_style_capacity_donors
        )
        positives = positive_pairs(policy, accounts, accounts_by_controller)
        negatives, negative_matching_diagnostic = select_negative_pairs(
            policy, accounts, accounts_by_controller, positives
        )
        synthesis_audit["negative_matching_diagnostic"] = (
            negative_matching_diagnostic
        )
        pair_rows = orient_and_identify_pairs(policy, positives + negatives)
        audit = audit_dataset(
            policy,
            topology_groups,
            split_by_group,
            accounts,
            accounts_by_controller,
            items,
            pair_rows,
            synthesis_audit,
            topology_audit,
            donor_audit,
        )
        if audit["status"] != "PASSED":
            print(json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True))
            raise SyntheticSourceError("Synthetic English source failed frozen quality gates")

        style_accounts = [
            {
                "account_uid": account_uid,
                "split": accounts[account_uid]["split"],
                "style_stream": accounts[account_uid]["style_stream"],
            }
            for account_uid in sorted(accounts)
        ]
        for name, values in (
            ("public_accounts_style_projection.jsonl", style_accounts),
        ):
            with (building / name).open("w", encoding="utf-8", newline="\n") as handle:
                for row in values:
                    handle.write(
                        json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
                        + "\n"
                    )
        write_csv(
            building / "public_pairs.csv",
            ["pair_uid", "account_left_uid", "account_right_uid", "split"],
            (
                {
                    "pair_uid": row["pair_uid"],
                    "account_left_uid": row["left_uid"],
                    "account_right_uid": row["right_uid"],
                    "split": row["split"],
                }
                for row in pair_rows
            ),
        )
        write_csv(
            building / "labels.csv",
            ["pair_uid", "label", "sample_weight"],
            (
                {
                    "pair_uid": row["pair_uid"],
                    "label": row["label"],
                    "sample_weight": format(row["sample_weight"], ".17g"),
                }
                for row in pair_rows
            ),
        )
        query_rows = []
        qrel_rows = []
        for controller_uid, account_uids in sorted(accounts_by_controller.items()):
            for query_uid in account_uids:
                query_rows.append(
                    {
                        "query_account_uid": query_uid,
                        "split": accounts[query_uid]["split"],
                    }
                )
                for candidate_uid in account_uids:
                    if candidate_uid != query_uid:
                        qrel_rows.append(
                            {
                                "query_account_uid": query_uid,
                                "relevant_account_uid": candidate_uid,
                            }
                        )
        write_csv(
            building / "retrieval_queries.csv",
            ["query_account_uid", "split"],
            sorted(query_rows, key=lambda row: row["query_account_uid"]),
        )
        write_csv(
            building / "retrieval_qrels.csv",
            ["query_account_uid", "relevant_account_uid"],
            sorted(
                qrel_rows,
                key=lambda row: (row["query_account_uid"], row["relevant_account_uid"]),
            ),
        )
        write_json(building / "quality_audit.json", audit)

        files = []
        for path in sorted(building.iterdir()):
            if path.name == "manifest.json" or not path.is_file():
                continue
            files.append(
                {
                    "path": path.name,
                    "size_bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
        manifest = {
            "version": policy["version"],
            "status": "PASSED_STYLE_ONLY_TRAINING_AUGMENTATION_QUALIFIED",
            "claim_boundary": policy["claim_boundary"],
            "policy_path": str(POLICY_PATH.relative_to(ROOT)).replace("\\", "/"),
            "policy_sha256": sha256_file(POLICY_PATH),
            "builder_path": str(Path(__file__).resolve().relative_to(ROOT)).replace("\\", "/"),
            "builder_sha256": sha256_file(Path(__file__).resolve()),
            "base_builder_sha256": sha256_file(BASE_BUILDER_PATH),
            "runtime_versions": {
                "python": platform.python_version(),
                "numpy": __import__("numpy").__version__,
                "scipy": __import__("scipy").__version__,
                "scikit_learn": __import__("sklearn").__version__,
            },
            "v5_real_holdout": {
                "public_accounts": len(holdout_aliases),
                "identity_fingerprints": len(holdout_fingerprints),
                "identity_component_unique_aliases": len(holdout_component_aliases),
                "identity_component_account_keys": sum(
                    len(global_identity["component_keys"][component_uid])
                    for component_uid in holdout_components
                ),
                "identity_components": len(holdout_components),
                "identity_evidence_tokens": len(holdout_evidence_tokens),
                "manifest_sha256": policy["inputs"][
                    "reports/step7_v5_english_source_dataset/v3_20260903/manifest.json"
                ],
            },
            "counts": audit["counts"],
            "split_counts": audit["split_counts"],
            "parse_coverage": {
                "vendor_registry": registry_parse,
                "auxiliary_pgp": pgp_parse,
            },
            "files": files,
        }
        manifest["manifest_self_sha256"] = canonical_sha256(manifest)
        write_json(building / "manifest.json", manifest)
        os.replace(building, output)
        return manifest
    except BaseException:
        if building.exists():
            shutil.rmtree(building)
        raise


def main() -> None:
    manifest = publish()
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
