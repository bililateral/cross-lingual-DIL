"""Independently verify new continual-dataset exports; no generator imports.

Reads only this run's source snapshots, manifest, public text and new synthetic
supervision. Does not run a model or use legacy labels/qrels/Audit A/B.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import string
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

SPLIT_COUNTS = {"train": 60, "development": 20, "heldout": 40}
ID = re.compile(r"[0-9a-f]{32}\Z")
ITEM_FIELDS = {"group_uid", "seller_uid", "item_uid", "title", "description"}
OWNER_FIELDS = ["group_uid", "seller_uid", "controller_uid"]
PAIR_FIELDS = ["group_uid", "seller_uid_left", "seller_uid_right", "label"]


def ensure(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def read_csv(path: Path, fields: list[str], maximum_line_bytes: int):
    with path.open("rb") as handle:
        ensure(all(len(line) <= maximum_line_bytes for line in handle), f"CSV row budget: {path.name}")
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        ensure(reader.fieldnames == fields, f"CSV schema: {path.name}")
        for row in reader:
            ensure(None not in row and all(value is not None for value in row.values()), "CSV shape")
            yield row


def pattern(template: str, fields: dict[str, list[str]]) -> re.Pattern:
    pieces = []
    for literal, field, format_spec, conversion in string.Formatter().parse(template):
        pieces.append(re.escape(literal))
        if field is not None:
            ensure(field in fields and not format_spec and not conversion, "Template field")
            pieces.append("(?:" + "|".join(re.escape(x) for x in fields[field]) + ")")
    return re.compile("".join(pieces))


class TextChecker:
    def __init__(self, catalog: dict[str, Any]):
        self.products: dict[str, tuple[int, int]] = {}
        self.titles: dict[str, tuple[str, int]] = {}
        self.descriptions: dict[str, list[tuple[re.Pattern, int, int]]] = {}
        self.topic_names = [topic["name"] for topic in catalog["topics"]]
        self.subcategory_names: list[str] = []
        for topic_index, topic in enumerate(catalog["topics"]):
            for sub in topic["subcategories"]:
                sub_index = len(self.subcategory_names)
                self.subcategory_names.append(sub["name"])
                for product in sub["products"]:
                    ensure(product not in self.products, "Duplicate product in catalog")
                    self.products[product] = (topic_index, sub_index)
                    for axis, template in enumerate(catalog["title_templates"]):
                        for qualifier in catalog["qualifiers"]:
                            title = template.format(product=product, qualifier=qualifier)
                            ensure(title not in self.titles, "Ambiguous title style")
                            self.titles[title] = (product, axis)
                    expressions = []
                    for layout, template in enumerate(catalog["description_templates"]):
                        for service_axis, services in enumerate(catalog["service_phrases"]):
                            fields = {
                                "intro": [x.format(product=product) for x in catalog["intros"]],
                                "detail": sub["details"], "packaging": catalog["packaging"],
                                "delivery": catalog["delivery"], "service": services,
                            }
                            expressions.append((pattern(template, fields), layout, service_axis))
                    self.descriptions[product] = expressions

    def check(self, row: dict[str, str]) -> tuple[int, int, tuple[int, int, int]]:
        ensure(set(row) == ITEM_FIELDS and all(isinstance(v, str) for v in row.values()),
               "Unexpected public fields")
        ensure(all(ID.fullmatch(row[field]) for field in ("group_uid", "seller_uid", "item_uid")),
               "Opaque ID format")
        for field, limit in (("title", 48), ("description", 720)):
            text = row[field]
            ensure(0 < len(text) <= limit and all(
                ord(c) <= 65535 and not 55296 <= ord(c) <= 57343
                and (ord(c) >= 32 or c in "\n\t") for c in text), "Invalid text/length")
        ensure(row["title"] in self.titles, "Title not from shared catalog")
        product, title_axis = self.titles[row["title"]]
        matches = [(layout, service) for regex, layout, service in self.descriptions[product]
                   if regex.fullmatch(row["description"])]
        ensure(len(matches) == 1, "Description is not an unambiguous shared-catalog rendering")
        topic, sub_index = self.products[product]
        return topic, sub_index, (title_axis, *matches[0])


def check_group(
    items: list[dict[str, str]], owners: list[dict[str, str]], pairs: list[dict[str, str]],
    checker: TextChecker,
) -> dict[str, Any]:
    """Reconstruct a complete group from exported ownership, independently of pairing code."""
    ensure(len(owners) == 28 and all(set(row) == set(OWNER_FIELDS) for row in owners),
           "Expected 28 ownership rows")
    owner = {row["seller_uid"]: row["controller_uid"] for row in owners}
    ensure(len(owner) == 28, "Duplicate account ownership")
    ensure(all(ID.fullmatch(v) for row in owners for v in row.values()), "Ownership ID")
    groups = {row["group_uid"] for row in owners}
    ensure(len(groups) == 1, "Ownership crosses groups")
    group_uid = next(iter(groups))
    sizes = Counter(Counter(owner.values()).values())
    ensure(sizes == {2: 8, 3: 4}, "Controller alias multiplicities")
    texts: dict[str, list[dict[str, str]]] = defaultdict(list)
    topic_counts: dict[str, Counter] = defaultdict(Counter)
    topic_totals: Counter = Counter()
    sub_totals: Counter = Counter()
    styles: dict[str, tuple[int, int, int]] = {}
    item_ids = set()
    duplicate_item_text = Counter()
    for row in items:
        ensure(row.get("seller_uid") in owner and row.get("group_uid") == group_uid,
               "Item account/group alignment")
        topic, sub_index, style = checker.check(row)
        ensure(row["item_uid"] not in item_ids, "Duplicate item ID")
        item_ids.add(row["item_uid"])
        texts[row["seller_uid"]].append(row)
        topic_counts[row["seller_uid"]][topic] += 1
        topic_totals[topic] += 1
        sub_totals[sub_index] += 1
        ensure(styles.setdefault(row["seller_uid"], style) == style,
               "Account style changed across records")
        duplicate_item_text[(row["title"], row["description"])] += 1
    ensure(set(texts) == set(owner) and all(2 <= len(v) <= 8 for v in texts.values()),
           "Every account must have 2-8 records")

    # Expected unordered edges from a sorted Cartesian product, not generator pair_rows.
    expected = {(left, right): int(owner[left] == owner[right])
                for left in sorted(owner) for right in sorted(owner) if left < right}
    actual = {}
    for row in pairs:
        ensure(set(row) == set(PAIR_FIELDS) and row["group_uid"] == group_uid,
               "Pair schema/group")
        key = (row["seller_uid_left"], row["seller_uid_right"])
        ensure(key in expected and key not in actual and str(row["label"]) in ("0", "1"),
               "Missing endpoint, self/reversed/duplicate edge or nonbinary label")
        actual[key] = int(row["label"])
    ensure(actual == expected, "Exported edges/labels do not equal ownership relation")
    ensure(len(actual) == 378 and sum(actual.values()) == 20, "Pair combinatorics")
    partners = Counter()
    for (left, right), label in actual.items():
        if label:
            partners[left] += 1
            partners[right] += 1
    ensure(Counter(partners.values()) == {1: 16, 2: 12}, "Query positives")

    overlaps = {str(label): {"pairs": 0, "topic_jaccard_sum": 0.0,
                            "same_style_axis_counts": [0, 0, 0]} for label in (0, 1)}
    for (left, right), label in actual.items():
        a, b = set(topic_counts[left]), set(topic_counts[right])
        entry = overlaps[str(label)]
        entry["pairs"] += 1
        entry["topic_jaccard_sum"] += len(a & b) / len(a | b)
        for axis in range(3):
            entry["same_style_axis_counts"][axis] += int(styles[left][axis] == styles[right][axis])
    return {
        "controllers": set(owner.values()), "accounts": set(owner), "item_ids": item_ids,
        "item_count_pmf": Counter(len(rows) for rows in texts.values()),
        "topic_counts": topic_totals, "subcategory_counts": sub_totals,
        "text_counts": duplicate_item_text,
        "account_text": {seller: tuple(sorted((r["title"], r["description"]) for r in rows))
                         for seller, rows in texts.items()},
        "overlap_sums": overlaps,
    }


def verify_dataset(root: Path) -> dict[str, Any]:
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    policy = json.loads((root / "sources/step28_continual_dataset_policy.json").read_text(encoding="utf-8"))
    catalog = json.loads((root / "sources/step28_continual_catalog.json").read_text(encoding="utf-8"))
    ensure(policy["groups_per_domain"] == SPLIT_COUNTS and policy["root_seed"] == 20260909,
           "Unapproved scale/seed")
    expected_payloads = {"groups.csv"} | {
        f"{split}/{name}" for split in SPLIT_COUNTS
        for name in ("items.jsonl", "supervision/owners.csv", "supervision/pairs.csv")
    } | {"sources/" + name for name in (
        "step28_continual_dataset.py", "step28_continual_dataset_verify.py",
        "step28_continual_dataset_policy.json", "step28_continual_catalog.json",
        "chinese_train_style_profile.json")}
    ensure(set(manifest["files"]) == expected_payloads, "Manifest payload set")
    current_files = {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()}
    ensure(current_files - expected_payloads <= {"manifest.json", "validation.json"}, "Unexpected files")
    for name, info in manifest["files"].items():
        path = root / name
        ensure(path.stat().st_size == info["bytes"] and sha256(path) == info["sha256"],
               f"Payload hash/size mismatch: {name}")
    checker = TextChecker(catalog)
    groups = {}
    group_numbers: dict[tuple[str, str], set[int]] = defaultdict(set)
    for row in read_csv(root / "groups.csv", [
        "domain", "split", "group_uid", "group_index", "accounts", "items"], 256):
        ensure(row["domain"] in ("A", "B", "C") and row["split"] in SPLIT_COUNTS, "Group schedule")
        uid = row["group_uid"]
        ensure(ID.fullmatch(uid) is not None and uid not in groups, "Duplicate/bad group ID")
        numbers = group_numbers[(row["domain"], row["split"])]
        number = int(row["group_index"])
        ensure(number not in numbers, "Duplicate group index")
        numbers.add(number)
        ensure(int(row["accounts"]) == 28, "Group account count")
        groups[uid] = row
    ensure(len(groups) == 360 and all(
        group_numbers[(domain, split)] == set(range(count))
        for domain in ("A", "B", "C") for split, count in SPLIT_COUNTS.items()), "Group coverage")

    seen = {key: set() for key in ("controllers", "accounts", "item_ids")}
    all_texts: Counter = Counter()
    all_account_texts: Counter = Counter()
    text_splits: dict[tuple[str, str], set[str]] = defaultdict(set)
    account_text_splits: dict[tuple, set[str]] = defaultdict(set)
    totals = Counter()
    item_count_pmf: Counter = Counter()
    domain_counts = {domain: Counter() for domain in ("A", "B", "C")}
    domain_subcounts = {domain: Counter() for domain in ("A", "B", "C")}
    cell_counts: dict[str, Any] = {}
    overlap_sums = {str(label): {"pairs": 0, "topic_jaccard_sum": 0.0,
                               "same_style_axis_counts": [0, 0, 0]} for label in (0, 1)}
    max_lengths = Counter()
    maximum_item_metadata_bytes = 0
    for split in SPLIT_COUNTS:
        by_group = {key: defaultdict(list) for key in ("items", "owners", "pairs")}
        with (root / split / "items.jsonl").open("rb") as handle:
            for line in handle:
                row = json.loads(line)
                ensure(set(row) == ITEM_FIELDS, "Public item schema")
                body_bytes = sum(len(row[k].encode("utf-8")) for k in ("title", "description"))
                overhead = len(line) - body_bytes
                ensure(overhead <= 256 and len(line) <= 768 * 3 + 256, "Item serialization budget")
                maximum_item_metadata_bytes = max(maximum_item_metadata_bytes, overhead)
                for key in ("title", "description"):
                    max_lengths[key] = max(max_lengths[key], len(row[key]))
                by_group["items"][row["group_uid"]].append(row)
        for key, filename, fields, row_budget in (
            ("owners", "owners.csv", OWNER_FIELDS, 192),
            ("pairs", "pairs.csv", PAIR_FIELDS, 128),
        ):
            for row in read_csv(root / split / "supervision" / filename, fields, row_budget):
                by_group[key][row["group_uid"]].append(row)
        expected_groups = {uid for uid, row in groups.items() if row["split"] == split}
        ensure(all(set(mapping) == expected_groups for mapping in by_group.values()),
               "Export split/group membership mismatch")
        for uid in sorted(expected_groups):
            items, owners, pairs = (by_group[key][uid] for key in ("items", "owners", "pairs"))
            ensure(int(groups[uid]["items"]) == len(items), "Group item count index")
            result = check_group(items, owners, pairs, checker)
            for key in seen:
                ensure(not seen[key] & result[key], f"{key} reused across independent groups/splits")
                seen[key].update(result[key])
            totals.update(groups=1, controllers=12, accounts=28, items=len(items),
                          pairs=378, positive_pairs=20, negative_pairs=358)
            item_count_pmf.update(result["item_count_pmf"])
            domain = groups[uid]["domain"]
            domain_counts[domain].update(result["topic_counts"])
            domain_subcounts[domain].update(result["subcategory_counts"])
            cell_key = f"{domain}/{split}"
            cell_counts.setdefault(cell_key, {"groups": 0, "items": 0, "topic_counts": [0, 0, 0]})
            cell_counts[cell_key]["groups"] += 1
            cell_counts[cell_key]["items"] += len(items)
            for topic, count in result["topic_counts"].items():
                cell_counts[cell_key]["topic_counts"][topic] += count
            all_texts.update(result["text_counts"])
            all_account_texts.update(result["account_text"].values())
            for value in result["text_counts"]:
                text_splits[value].add(split)
            for value in result["account_text"].values():
                account_text_splits[value].add(split)
            for label, entry in result["overlap_sums"].items():
                target = overlap_sums[label]
                target["pairs"] += entry["pairs"]
                target["topic_jaccard_sum"] += entry["topic_jaccard_sum"]
                for axis in range(3):
                    target["same_style_axis_counts"][axis] += entry["same_style_axis_counts"][axis]
    domain_report = {}
    for domain, counts in domain_counts.items():
        total = sum(counts.values())
        weights = [p / 4 for p in policy["domains"][domain] for _ in range(4)]
        # Independent marginal formula for selecting one of two sequential weighted draws.
        marginal = [p / 2 * (1 + math.fsum(
            weights[k] / (1 - weights[k]) for k in range(12) if k != i))
                    for i, p in enumerate(weights)]
        final = [0.7 * q + 0.3 * p for q, p in zip(marginal, weights)]
        domain_report[domain] = {
            "items": total,
            "topic_counts": {name: counts[i] for i, name in enumerate(checker.topic_names)},
            "topic_proportions": {name: counts[i] / total for i, name in enumerate(checker.topic_names)},
            "theoretical_topic_proportions": [sum(final[i:i+4]) for i in (0, 4, 8)],
            "subcategory_counts": {name: domain_subcounts[domain][i]
                                   for i, name in enumerate(checker.subcategory_names)},
        }
    for entry in overlap_sums.values():
        entry["mean_topic_set_jaccard"] = entry.pop("topic_jaccard_sum") / entry["pairs"]
        entry["style_axis_agreement"] = [x / entry["pairs"] for x in entry.pop("same_style_axis_counts")]
    total_bytes = sum(p.stat().st_size for p in root.rglob("*") if p.is_file())
    small_bytes = sum((root / name).stat().st_size for name in current_files
                      if name.startswith("sources/") or name in ("manifest.json", "validation.json"))
    ensure(total_bytes <= 256 * 1024**2 and small_bytes <= 4 * 1024**2, "Directory budget")
    return {
        "status": "PASS_GENERATION_CONTRACT_NOT_MODEL_QUALIFICATION",
        "counts": dict(totals), "item_count_histogram": dict(sorted(item_count_pmf.items())),
        "maximum_text_characters": dict(max_lengths),
        "maximum_item_metadata_bytes": maximum_item_metadata_bytes,
        "directory_bytes_at_verification": total_bytes, "small_record_bytes_at_verification": small_bytes,
        "domain_observations": domain_report, "domain_split_observations": cell_counts,
        "duplicate_text": {
            "item_text_unique": len(all_texts),
            "item_text_extra_occurrences": sum(n - 1 for n in all_texts.values()),
            "item_text_types_shared_across_splits": sum(len(s) > 1 for s in text_splits.values()),
            "account_text_extra_occurrences": sum(n - 1 for n in all_account_texts.values()),
            "account_text_types_shared_across_splits": sum(len(s) > 1 for s in account_text_splits.values()),
        },
        "observed_pair_overlap": overlap_sums,
        "checks": [
            "All payload hashes and sizes match the source manifest",
            "360 independent groups; controller/account/item IDs isolated",
            "New pair supervision equals complete ownership equivalence relation",
            "Every account has one fixed collection of 2-8 nonempty records",
            "Rendered product and all wording match the common catalog",
            "Account title/layout/service axes are constant across its records",
            "Only title and description are model inputs; IDs remain metadata",
            "27 candidates/query and 1 or 2 positives; no model was evaluated",
            "Actual category counts and overlap reported without reseeding or significance gates",
        ],
        "limits": [
            "Generator correctness and observed distribution shift do not establish learnability or forgetting.",
            "Duplicate common text may occur; counts are descriptive, not an automatic leakage verdict.",
            "No old labels, qrels, Audit A/B, Linux, GPU or model loading used.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(verify_dataset(args.root.resolve()), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
