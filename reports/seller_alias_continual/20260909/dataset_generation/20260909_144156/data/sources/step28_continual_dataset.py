"""Generate the approved Chinese seller-alias content-shift dataset (stdlib only).

Only title/description are model inputs. IDs are join metadata; ownership and
labels are new synthetic supervision. No legacy raw data, labels or models load.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
import platform
import random
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
POLICY = ROOT / "schema/step28_continual_dataset_policy.json"
CATALOG = ROOT / "schema/step28_continual_catalog.json"
SPLITS = ("train", "development", "heldout")
ITEM_FIELDS = ("group_uid", "seller_uid", "item_uid", "title", "description")
OWNER_FIELDS = ("group_uid", "seller_uid", "controller_uid")
PAIR_FIELDS = ("group_uid", "seller_uid_left", "seller_uid_right", "label")
GROUP_FIELDS = ("domain", "split", "group_uid", "group_index", "accounts", "items")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def digest(path: Path) -> str:
    with path.open("rb") as handle:
        value = hashlib.sha256()
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def json_line(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n"


def write_json(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def stream(seed: int, *parts: Any) -> random.Random:
    key = json_line([seed, *parts]).encode("utf-8")
    return random.Random(int.from_bytes(hashlib.sha256(key).digest(), "big"))


def opaque_id(seed: int, *parts: Any) -> str:
    return hashlib.sha256(json_line([seed, *parts]).encode("utf-8")).hexdigest()[:32]


def weighted_index(rng: random.Random, weights: list[float]) -> int:
    require(bool(weights) and all(math.isfinite(x) and x >= 0 for x in weights),
            "Invalid weights")
    total = math.fsum(weights)
    require(total > 0, "Zero total weight")
    value = rng.random() * total
    cumulative = 0.0
    last_positive = 0
    for index, weight in enumerate(weights):
        if weight:
            last_positive = index
        cumulative += weight
        if value < cumulative:
            return index
    return last_positive  # Protect only a floating-point final-boundary roundoff.


def preference(rng: random.Random, weights: list[float]) -> tuple[int, int]:
    first = weighted_index(rng, weights)
    remaining = weights.copy()
    remaining[first] = 0.0
    second = weighted_index(rng, remaining)
    return tuple(sorted((first, second)))


def inherit_preference(
    rng: random.Random, parent: tuple[int, int], weights: list[float], probability: float
) -> tuple[int, int]:
    # A redraw equal to the parent is valid; never retry it.
    return parent if rng.random() < probability else preference(rng, weights)


def inherit_style(rng: random.Random, parent: int, size: int, probability: float) -> int:
    return parent if rng.random() < probability else rng.randrange(size)


def subcategories(catalog: dict[str, Any]) -> list[dict[str, Any]]:
    return [sub for topic in catalog["topics"] for sub in topic["subcategories"]]


def valid_text(text: str, limit: int) -> bool:
    return bool(text) and len(text) <= limit and all(
        ord(char) <= 0xFFFF and not 0xD800 <= ord(char) <= 0xDFFF
        and (ord(char) >= 32 or char in "\n\t")
        for char in text
    )


def render(
    rng: random.Random, catalog: dict[str, Any], sub_index: int, style: tuple[int, ...]
) -> tuple[str, str]:
    sub = subcategories(catalog)[sub_index]
    product = rng.choice(sub["products"])
    title = catalog["title_templates"][style[0]].format(
        product=product, qualifier=rng.choice(catalog["qualifiers"])
    )
    description = catalog["description_templates"][style[1]].format(
        intro=rng.choice(catalog["intros"]).format(product=product),
        detail=rng.choice(sub["details"]),
        packaging=rng.choice(catalog["packaging"]),
        delivery=rng.choice(catalog["delivery"]),
        service=rng.choice(catalog["service_phrases"][style[2]]),
    )
    return title, description


def check_contract(policy: dict[str, Any], catalog: dict[str, Any]) -> dict[str, int]:
    """Check approved parameters and the maximum over common rendering resources."""
    require(policy["schema_version"] == catalog["schema_version"] == 1, "Schema mismatch")
    require(policy["root_seed"] == 20260909, "Unapproved root seed")
    require(policy["domains"] == {
        "A": [0.6, 0.2, 0.2], "B": [0.2, 0.6, 0.2], "C": [0.2, 0.2, 0.6]
    }, "Unapproved domain mixtures")
    require(policy["groups_per_domain"] == {
        "train": 60, "development": 20, "heldout": 40
    }, "Unapproved scale")
    require(policy["controllers_per_group"] == 12
            and policy["controller_alias_counts"] == {"2": 8, "3": 4}, "Group structure")
    require(policy["preference_inheritance"] == 0.75
            and policy["preferred_item_probability"] == 0.70, "Mechanism mismatch")
    require(policy["style_axis_sizes"] == [3, 3, 4]
            and policy["style_sampling"] == "independent_uniform_controller_and_redraw"
            and policy["redraw_may_equal_parent"] is True, "Style/redraw mismatch")
    require(policy["text_limits"] == {"title": 48, "description": 720}, "Text limits")
    require(policy["model_text_fields"] == ["title", "description"]
            and policy["public_item_fields"] == list(ITEM_FIELDS), "Input boundary")
    require(policy["training_authorized"] is False, "Generation scope only")
    require(policy["output_budget_bytes"] == 256 * 1024**2
            and policy["small_records_budget_bytes"] == 4 * 1024**2, "Budget mismatch")
    pmf = policy["item_count_pmf"]
    require(set(pmf) == set(map(str, range(2, 9)))
            and all(x > 0 for x in pmf.values())
            and abs(math.fsum(pmf.values()) - 1) < 1e-10, "Item count PMF")
    require(len(catalog["topics"]) == 3
            and all(len(t["subcategories"]) == 4 for t in catalog["topics"]), "Categories")
    require([len(catalog[k]) for k in
             ("title_templates", "description_templates", "service_phrases")] == [3, 3, 4],
            "Shared style axes")
    subs = subcategories(catalog)
    products = [p for sub in subs for p in sub["products"]]
    require(len(products) == len(set(products)) == 48, "48 shared, distinct product names")
    require(not any(a in b for a in products for b in products if a != b),
            "Product names must be unambiguously recoverable from rendered text")

    def strings(value: Any):
        if isinstance(value, str):
            yield value
        elif isinstance(value, list):
            for child in value:
                yield from strings(child)
        elif isinstance(value, dict):
            for child in value.values():
                yield from strings(child)

    require(all(valid_text(s, 1000) for s in strings(catalog)), "Invalid catalog characters")
    max_title = max(len(t.format(product=p, qualifier=q))
                    for t, p, q in itertools.product(
                        catalog["title_templates"], products, catalog["qualifiers"]))
    # Each field occurs once; maximizing each common component gives the exact bound.
    template_fields = ("intro", "detail", "packaging", "delivery", "service")
    for template in catalog["description_templates"]:
        require(all(template.count("{" + field + "}") == 1 for field in template_fields),
                "Description template field use")
    max_description = 0
    for sub, template in itertools.product(subs, catalog["description_templates"]):
        components = {
            "intro": max((t.format(product=p) for t, p in itertools.product(
                catalog["intros"], sub["products"])), key=len),
            "detail": max(sub["details"], key=len),
            "packaging": max(catalog["packaging"], key=len),
            "delivery": max(catalog["delivery"], key=len),
            "service": max(itertools.chain.from_iterable(catalog["service_phrases"]), key=len),
        }
        max_description = max(max_description, len(template.format(**components)))
    require(max_title <= 48 and max_description <= 720, "Catalog exceeds text budget")
    return {"maximum_title_characters": max_title,
            "maximum_description_characters": max_description}


def make_group(
    policy: dict[str, Any], catalog: dict[str, Any], domain: str, split: str, index: int
) -> tuple[dict[str, Any], list[dict[str, str]], list[dict[str, str]]]:
    seed = policy["root_seed"]
    context = (domain, split, index)
    group_uid = opaque_id(seed, *context, "group")
    weights = [weight / 4 for weight in policy["domains"][domain] for _ in range(4)]
    sizes = [int(size) for size, count in sorted(policy["controller_alias_counts"].items())
             for _ in range(count)]
    stream(seed, *context, "alias_counts").shuffle(sizes)
    account_ids = [opaque_id(seed, *context, "account", n) for n in range(sum(sizes))]
    stream(seed, *context, "id_assignment").shuffle(account_ids)
    item_numbers = sorted(map(int, policy["item_count_pmf"]))
    count_weights = [policy["item_count_pmf"][str(n)] for n in item_numbers]
    items: list[dict[str, str]] = []
    owners: list[dict[str, str]] = []
    account_slot = 0
    for controller, size in enumerate(sizes):
        controller_uid = opaque_id(seed, *context, "controller", controller)
        parent = preference(stream(seed, *context, "controller_preference", controller), weights)
        parent_style = tuple(stream(seed, *context, "controller_style", controller, axis)
                             .randrange(n) for axis, n in enumerate(policy["style_axis_sizes"]))
        for alias in range(size):
            seller_uid = account_ids[account_slot]
            account_slot += 1
            account_context = (*context, controller, alias)
            own_preference = inherit_preference(
                stream(seed, *account_context, "account_preference"), parent,
                weights, policy["preference_inheritance"]
            )
            style = tuple(inherit_style(
                stream(seed, *account_context, "account_style", axis), parent_style[axis],
                n, policy["preference_inheritance"]
            ) for axis, n in enumerate(policy["style_axis_sizes"]))
            number = item_numbers[weighted_index(
                stream(seed, *account_context, "item_count"), count_weights)]
            owners.append(dict(group_uid=group_uid, seller_uid=seller_uid,
                               controller_uid=controller_uid))
            for item in range(number):
                category_rng = stream(seed, *account_context, "item_category", item)
                if category_rng.random() < policy["preferred_item_probability"]:
                    sub_index = category_rng.choice(own_preference)
                else:
                    sub_index = weighted_index(category_rng, weights)
                title, description = render(
                    stream(seed, *account_context, "item_wording", item), catalog, sub_index, style
                )
                require(valid_text(title, 48) and valid_text(description, 720), "Rendered text")
                items.append(dict(
                    group_uid=group_uid, seller_uid=seller_uid,
                    item_uid=opaque_id(seed, *account_context, "item_id", item),
                    title=title, description=description
                ))
    items.sort(key=lambda row: (row["seller_uid"], row["item_uid"]))
    owners.sort(key=lambda row: row["seller_uid"])
    group = dict(domain=domain, split=split, group_uid=group_uid, group_index=index,
                 accounts=len(owners), items=len(items))
    return group, items, owners


def pair_rows(owners: list[dict[str, str]]):
    for left, right in itertools.combinations(sorted(owners, key=lambda r: r["seller_uid"]), 2):
        yield dict(group_uid=left["group_uid"], seller_uid_left=left["seller_uid"],
                   seller_uid_right=right["seller_uid"],
                   label=int(left["controller_uid"] == right["controller_uid"]))


def build_dataset(
    output: Path, policy_path: Path, catalog_path: Path, reference_path: Path
) -> dict[str, Any]:
    policy, catalog = load_json(policy_path), load_json(catalog_path)
    catalog_bounds = check_contract(policy, catalog)
    ref = policy["item_count_reference"]
    require(digest(reference_path) == ref["sha256"], "Public PMF reference hash")
    reference = load_json(reference_path)
    require(reference["seller_count"] == ref["seller_count"]
            and reference["item_count_pmf"] == policy["item_count_pmf"]
            and reference["item_count_clip"] == [2, 8], "Public PMF reference contents")
    require(not output.exists(), "Output already exists; never overwrite an independent run")
    output.mkdir(parents=True)
    sources = output / "sources"
    sources.mkdir()
    source_inputs = [Path(__file__), Path(__file__).with_name("step28_continual_dataset_verify.py"),
                     policy_path, catalog_path, reference_path]
    for path in source_inputs:
        shutil.copyfile(path, sources / path.name)
    with (output / "groups.csv").open("x", encoding="utf-8", newline="") as group_handle:
        group_writer = csv.DictWriter(group_handle, fieldnames=GROUP_FIELDS, lineterminator="\n")
        group_writer.writeheader()
        for split in SPLITS:
            split_root = output / split
            supervision = split_root / "supervision"
            supervision.mkdir(parents=True)
            with (split_root / "items.jsonl").open("x", encoding="utf-8", newline="\n") as item_handle, \
                    (supervision / "owners.csv").open("x", encoding="utf-8", newline="") as owner_handle, \
                    (supervision / "pairs.csv").open("x", encoding="utf-8", newline="") as pair_handle:
                owner_writer = csv.DictWriter(owner_handle, OWNER_FIELDS, lineterminator="\n")
                pair_writer = csv.DictWriter(pair_handle, PAIR_FIELDS, lineterminator="\n")
                owner_writer.writeheader()
                pair_writer.writeheader()
                for domain in sorted(policy["domains"]):
                    for index in range(policy["groups_per_domain"][split]):
                        group, items, owners = make_group(policy, catalog, domain, split, index)
                        group_writer.writerow(group)
                        for row in items:
                            item_handle.write(json_line(row))
                        owner_writer.writerows(owners)
                        pair_writer.writerows(pair_rows(owners))
                    print(json_line({"written": split, "domain": domain}).strip(), flush=True)
    payloads = {str(p.relative_to(output).as_posix()):
                {"bytes": p.stat().st_size, "sha256": digest(p)}
                for p in sorted(output.rglob("*")) if p.is_file()}
    write_json(output / "manifest.json", {
        "study": policy["study"], "root_seed": policy["root_seed"],
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version, "platform": platform.platform(),
        "argv": sys.argv, "catalog_bounds": catalog_bounds, "files": payloads,
        "model_text_fields": policy["model_text_fields"],
        "supervision_scope": policy["generation_label_access"],
        "limitations": policy["limitations"],
        "status": "BUILT_AWAITING_INDEPENDENT_VALIDATION",
    })
    # Separate verifier derives relations and observed topics from exported files.
    from step28_continual_dataset_verify import verify_dataset
    validation = verify_dataset(output)
    write_json(output / "validation.json", validation)
    require(sum(p.stat().st_size for p in output.rglob("*") if p.is_file())
            <= policy["output_budget_bytes"], "Final directory exceeds budget")
    return validation


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("build", help="Generate once in a new directory, then validate")
    build.add_argument("--out", type=Path, required=True)
    build.add_argument("--policy", type=Path, default=POLICY)
    build.add_argument("--catalog", type=Path, default=CATALOG)
    build.add_argument("--reference", type=Path, default=None)
    args = parser.parse_args()
    policy = load_json(args.policy)
    reference = args.reference or ROOT / policy["item_count_reference"]["path"]
    result = build_dataset(args.out.resolve(), args.policy.resolve(), args.catalog.resolve(),
                           reference.resolve())
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
