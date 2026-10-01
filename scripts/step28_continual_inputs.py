#!/usr/bin/env python3
"""Audit public train/development inputs and freeze random whole-world arrivals.

This is an input audit, not a training or label-reading entry point. Published
arrays are not regenerated. Existing Audit A/B payloads are never opened.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import platform
import sys
from collections import Counter
from itertools import combinations, zip_longest
from pathlib import Path
from typing import Any, Iterator

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "reports/step28_synthetic_chinese_dataset/v9_4_1_formal_500x4_attempt1_20260829"
PROJECTION = ROOT / "reports/step28_model_experiment/v9_4_1_public_projection_v1_20260831"
OUTPUT = ROOT / "reports/seller_alias_continual/20260907/inputs"
SPLITS = ("train", "development")
SEEDS = (11, 23, 37)
ROW_FIELDS = ("split", "world_ordinal", "world_uid", "canonical_pair_uid",
              "seller_uid_left", "seller_uid_right")
ENDPOINT_FIELDS = ("canonical_pair_uid", "world_uid", "seller_uid_left", "seller_uid_right")
ROOT_HASH = "9190ecdead719ace3bdf74f635c1c23e0a0d19482cb4a9d4d2f6fc1f31dbbbe3"
PROJECTION_HASH = "eee19a09978e40a9016e1d9722579fae04feff90ffa7b5a19b08481e7a2bc4f2"


def file_record(path: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return {"path": path.relative_to(ROOT).as_posix(),
            "size_bytes": path.stat().st_size, "sha256": digest.hexdigest()}


def verify(path: Path, spec: dict[str, Any], records: list[dict]) -> Path:
    actual = file_record(path)
    if actual["sha256"] != spec["sha256"] or (
        "size_bytes" in spec and actual["size_bytes"] != spec["size_bytes"]
    ):
        raise ValueError(f"Input bytes differ: {actual['path']}")
    records.append(actual)
    return path


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def csv_rows(path: Path, fields: tuple[str, ...]) -> Iterator[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != fields:
            raise ValueError(f"CSV schema differs: {path.name}")
        yield from reader


def validate_world_rows(rows: list[dict[str, str]], world: dict[str, Any],
                        sellers: list[str], split: str, ordinal: int) -> None:
    expected = list(combinations(sorted(sellers), 2))
    if len(sellers) != 28 or len(set(sellers)) != 28 or len(rows) != 378:
        raise ValueError("Incomplete world")
    if world != {"world_uid": world["world_uid"], "split": split,
                 "world_ordinal": ordinal, "seller_count": 28,
                 "item_count": 99, "pair_count": 378}:
        raise ValueError("World metadata differs")
    for row, (left, right) in zip(rows, expected, strict=True):
        wanted = dict(zip(ROW_FIELDS, (split, str(ordinal), world["world_uid"],
                                      f"{left}||{right}", left, right), strict=True))
        if row != wanted:
            raise ValueError("World/pair alignment differs")


def random_arrivals(world_count: int = 500, stage_count: int = 5) -> dict[str, Any]:
    if world_count <= 0 or stage_count <= 0 or world_count % stage_count:
        raise ValueError("Random stages require equal complete-world blocks")
    orders = []
    for seed in SEEDS:
        permutation = np.random.default_rng(seed).permutation(world_count)
        stages = [group.tolist() for group in np.split(permutation, stage_count)]
        orders.append({"seed": seed, "train_world_ordinals_by_stage": stages,
                       "evaluation": "all fixed development worlds at every stage"})
    return {"kind": "random_whole_world_reference", "distinct_domains_claimed": False,
            "world_count": world_count, "stage_count": stage_count,
            "rng": "numpy.default_rng PCG64", "orders": orders}


def summarize_features(matrix: np.ndarray, names: list[str]) -> tuple[list[dict], np.ndarray]:
    if matrix.shape != (189000, len(names)) or np.isinf(matrix).any():
        raise ValueError("Feature shape or infinity")
    blocks = matrix.reshape(500, 378, len(names))
    count = np.isfinite(blocks).sum(axis=1)
    means = np.divide(np.nansum(blocks, axis=1), count,
                      out=np.zeros((500, len(names))), where=count > 0)
    result = []
    for column, name in enumerate(names):
        values = matrix[:, column]
        finite = values[np.isfinite(values)]
        result.append({"name": name, "missing": int(np.isnan(values).sum()),
                       "unique_finite_values": int(len(np.unique(finite))),
                       "min": float(finite.min()) if len(finite) else None,
                       "max": float(finite.max()) if len(finite) else None,
                       "world_mean_min_median_max": np.quantile(
                           means[:, column], [0, .5, 1]).tolist(),
                       "world_means_distinct": int(len(np.unique(means[:, column])))})
    return result, means


def profile_summary(path: Path, seller_world: dict[str, int]) -> tuple[dict, list[str]]:
    category_counts = [Counter() for _ in range(500)]
    numeric = [[] for _ in range(500)]
    seen: set[str] = set()
    for row in jsonl(path):
        seller = row["seller_uid"]
        if seller not in seller_world or seller in seen:
            raise ValueError("Profile seller isolation/alignment")
        seen.add(seller)
        world = seller_world[seller]
        # Counts account-level category presence, NOT item frequency.
        categories = {part.strip() for part in row["category_concat_top"].split(" || ")
                      if part.strip()}
        category_counts[world].update(categories)
        numeric[world].append([row["item_count"], row["title_length_stats"]["median"],
                               row["description_length_stats"]["median"],
                               row["style_stats"]["punct_ratio_mean"],
                               row["style_stats"]["digit_ratio_mean"]])
    if seen != set(seller_world):
        raise ValueError("Missing public profile")
    categories = sorted(set().union(*(set(row) for row in category_counts)))
    presence = np.asarray([[row[name] for name in categories] for row in category_counts])
    numeric_means = np.asarray([np.mean(row, axis=0) for row in numeric])
    winners, ties = [], 0
    for row in category_counts:
        maximum = max(row.values())
        tied = sorted(name for name, value in row.items() if value == maximum)
        ties += int(len(tied) > 1)
        winners.append(tied[0])
    return {"category_measure": "number of accounts mentioning category in profile",
            "categories": categories,
            "category_world_presence": {name: int(np.count_nonzero(presence[:, i]))
                                        for i, name in enumerate(categories)},
            "category_accounts_per_world_min_median_max": {
                name: np.quantile(presence[:, i], [0, .5, 1]).tolist()
                for i, name in enumerate(categories)},
            "dominant_category_tie_worlds": ties,
            "dominant_category_support_utf8_tie_break": dict(sorted(Counter(winners).items())),
            "world_mean_profile_min_median_max": {
                name: np.quantile(numeric_means[:, i], [0, .5, 1]).tolist()
                for i, name in enumerate(("item_count", "title_length_median",
                                         "description_length_median", "punct_ratio_mean",
                                         "digit_ratio_mean"))}}, winners


def between_group_fraction(means: np.ndarray, groups: list[str]) -> list[float]:
    """Descriptive ANOVA fraction on world means, no labels or inference test."""
    grand = means.mean(axis=0)
    total = np.square(means - grand).sum(axis=0)
    between = np.zeros(means.shape[1])
    group_array = np.asarray(groups)
    for group in sorted(set(groups)):
        block = means[group_array == group]
        between += len(block) * np.square(block.mean(axis=0) - grand)
    return np.divide(between, total, out=np.zeros_like(total), where=total > 1e-24).tolist()


def audit() -> tuple[dict, dict]:
    records: list[dict] = []
    source = read_json(verify(DATA / "root_manifest.json", {"sha256": ROOT_HASH}, records))
    publication = read_json(verify(PROJECTION / "public_projection_manifest.json",
                                   {"sha256": PROJECTION_HASH}, records))
    base = read_json(verify(PROJECTION / publication["base_manifest_file"]["path"],
                            publication["base_manifest_file"], records))
    identity = read_json(verify(PROJECTION / publication["identity_manifest_file"]["path"],
                                publication["identity_manifest_file"], records))
    policy_path = ROOT / "schema/step28_model_experiment_policy.json"
    records.append(file_record(policy_path))
    feature_contract = read_json(policy_path)["feature_contract"]
    names = [*feature_contract["legacy18"], *feature_contract["labse6"],
             *feature_contract["identity33"]]
    # Pin the column order through its historical scientific digest.
    name_hash = hashlib.sha256(json.dumps(names, ensure_ascii=False, sort_keys=True,
                                         separators=(",", ":")).encode()).hexdigest()
    if name_hash != "3afa0e88404c97907d0c1527efe19aa0580837fc3a849242b6433cab3d392c73":
        raise ValueError("Joint57 column order differs")
    registered = {row["path"]: row for row in source["public_files"]}
    summaries, all_sellers, all_worlds = {}, set(), set()
    for split in SPLITS:
        print(f"Auditing public {split}", flush=True)
        base_record = next(row for row in base["splits"] if row["split"] == split)
        base_split = read_json(verify(PROJECTION / "base_v1" / base_record["manifest_file"]["path"],
                                      base_record["manifest_file"], records))
        id_split = next(row for row in identity["splits"] if row["split"] == split)
        bpath = verify(PROJECTION / "base_v1" / base_split["base24_file"]["path"],
                       base_split["base24_file"], records)
        ipath = verify(PROJECTION / "identity_v1" / id_split["identity33_file"]["path"],
                       id_split["identity33_file"], records)
        row_path = verify(PROJECTION / "base_v1" / base_split["row_keys_file"]["path"],
                          base_split["row_keys_file"], records)
        verify(PROJECTION / "identity_v1" / id_split["row_keys_file"]["path"],
               id_split["row_keys_file"], records)
        if base_split["row_keys_file"]["sha256"] != id_split["row_keys_file"]["sha256"]:
            raise ValueError("Base/identity row bytes differ")
        paths = {}
        for filename in ("worlds.jsonl", "sellers.jsonl", "model_seller_profiles.jsonl",
                         "complete_model_pair_endpoints.csv", "identity33_all_pairs.csv"):
            relative = f"{split}/observed/{filename}"
            paths[filename] = verify(DATA / relative, registered[relative], records)
        worlds = list(jsonl(paths["worlds.jsonl"]))
        if len(worlds) != 500 or len({w["world_uid"] for w in worlds}) != 500:
            raise ValueError("World count/uniqueness")
        world_index = {row["world_uid"]: i for i, row in enumerate(worlds)}
        sellers = [[] for _ in worlds]
        seller_world: dict[str, int] = {}
        for row in jsonl(paths["sellers.jsonl"]):
            seller, world = row["seller_uid"], world_index[row["world_uid"]]
            if seller in seller_world:
                raise ValueError("Duplicate seller")
            seller_world[seller] = world
            sellers[world].append(seller)
        if all_sellers.intersection(seller_world) or all_worlds.intersection(world_index):
            raise ValueError("Train/development identity overlap")
        all_sellers.update(seller_world)
        all_worlds.update(world_index)
        bmatrix = np.load(bpath, allow_pickle=False)
        imatrix = np.load(ipath, allow_pickle=False)
        if bmatrix.dtype.str != "<f8" or imatrix.dtype.str != "<f8" or (
            bmatrix.shape != (189000, 24) or imatrix.shape != (189000, 33)
            or not np.isfinite(imatrix).all()
        ):
            raise ValueError("Published matrix schema/finite identity")
        row_iter = csv_rows(row_path, ROW_FIELDS)
        source_iter = csv_rows(paths["complete_model_pair_endpoints.csv"], ENDPOINT_FIELDS)
        identity_iter = csv_rows(paths["identity33_all_pairs.csv"],
                                  ("canonical_pair_uid", "world_uid", *names[24:]))
        block, row_count = [], 0
        for index, (row, endpoint, ident) in enumerate(zip_longest(row_iter, source_iter, identity_iter)):
            if row is None or endpoint is None or ident is None or index >= 189000:
                raise ValueError("Unequal source/array row count")
            if {name: row[name] for name in ENDPOINT_FIELDS} != endpoint or (
                ident["canonical_pair_uid"] != row["canonical_pair_uid"]
                or ident["world_uid"] != row["world_uid"]
            ):
                raise ValueError("Source/array row alignment")
            if not np.array_equal(imatrix[index], [float(ident[name]) for name in names[24:]]):
                raise ValueError("Identity CSV/array differs")
            block.append(row)
            if len(block) == 378:
                ordinal = index // 378
                validate_world_rows(block, worlds[ordinal], sellers[ordinal], split, ordinal)
                block = []
            row_count += 1
        if row_count != 189000 or block:
            raise ValueError("Incomplete pair stream")
        matrix = np.column_stack((bmatrix, imatrix))
        features, means = summarize_features(matrix, names)
        profiles, groups = profile_summary(paths["model_seller_profiles.jsonl"], seller_world)
        summaries[split] = {"worlds": len(worlds), "sellers": len(seller_world),
                            "pairs": row_count, "all_worlds_complete_k28": True,
                            "identity_csv_array_exact": True, "features": features,
                            "public_profiles": profiles,
                            "dominant_category_world_mean_between_group_fraction":
                                dict(zip(names, between_group_fraction(means, groups), strict=True))}
    arrivals = random_arrivals()
    arrivals["source_publication_sha256"] = PROJECTION_HASH
    result = {"status": "PUBLIC_INPUT_AUDIT_COMPLETE_NO_TRAINING", "inputs": records,
              "splits": summaries, "train_development_public_ids_disjoint": True,
              "column_order_sha256": name_hash, "arrival_kind": arrivals["kind"],
              "label_payloads_read": 0, "audit_payloads_read": 0,
              "models_fitted": 0, "linux_accessed": False,
              "environment": {"python": sys.version, "platform": platform.platform(),
                              "numpy": np.__version__,
                              "torch_installed": importlib.util.find_spec("torch") is not None},
              "implementation": file_record(Path(__file__)),
              "limits": ["Metadata and public feature assessment, not prediction effectiveness",
                         "No reproduction of encoder or production parser; archived payloads verified",
                         "Category ANOVA is descriptive and uses no pair labels",
                         "Random batches do not establish domains or real chronology"]}
    return result, arrivals


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("audit",))
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Do not overwrite a published input audit")
    result, arrivals = audit()
    args.output.mkdir(parents=True, exist_ok=False)
    for name, value in (("inputs.json", result), ("arrival.json", arrivals)):
        with (args.output / name).open("x", encoding="utf-8", newline="\n") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.write("\n")
    print(json.dumps({"status": result["status"], "checked_inputs": len(result["inputs"]),
                      "output": str(args.output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
