"""Approved diagnostic inputs, paired schedules and saved-metric comparisons.

No test text, test scores, owners or legacy truth is opened. Training supervision
is parsed only by the explicit training entry; valid truth has a separate gate.
"""
from __future__ import annotations

import csv
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

import step28_continual_population_data as data
import step28_continual_population_evaluate as metrics

POLICY = data.ROOT / "schema/step28_continual_diagnostic_policy.json"
SPLITS = ("train", "development")
ARMS = ("same", "cross")
POINTS = ("shared", *ARMS)
COMPLETE = "DIAGNOSTIC_COMPLETE_NINE_POINTS_VALID_TRUTH_UNREAD"


def policy() -> dict:
    config = data.read_json(POLICY)
    if (config["rotations"] != [["A", "B"], ["B", "C"], ["C", "A"]]
            or config["epochs_per_stage"] != 3 or config["physical_updates"] != 1620
            or config["evaluation"]["splits"] != list(SPLITS)
            or config["evaluation"]["threshold_logit"] != 0
            or config["evaluation"]["minimum_matching_directions"] != 2):
        raise ValueError("Diagnostic approval differs")
    return config


def sources() -> list[dict]:
    paths = list((data.ROOT / "scripts").glob("step28_continual_diagnostic*.py"))
    paths += [data.ROOT / "scripts" / f"step28_continual_population{suffix}.py"
              for suffix in ("", "_data", "_run", "_evaluate")]
    paths += [POLICY, data.ROOT / "tests/test_step28_continual_diagnostic_contracts.py"]
    return [data.record(p, data.ROOT) for p in sorted(paths)]


class Archive:
    """Offline train/valid public supply; never construct the old three-split archive."""

    def __init__(self, config: dict, *, train_labels: bool = False):
        self.root = data.ROOT / config["data_root"]
        for name, key in (("manifest.json", "data_manifest_sha256"),
                          ("validation.json", "data_validation_sha256")):
            if data.sha256(self.root / name) != config[key]:
                raise ValueError("Pinned dataset metadata differs")
        self.manifest = data.read_json(self.root / "manifest.json")
        if data.read_json(self.root / "validation.json")["status"] != "PASS_GENERATION_CONTRACT_NOT_MODEL_QUALIFICATION":
            raise ValueError("Missing original dataset qualification")
        self.checked: dict = {}
        with self.path("groups.csv").open(encoding="utf-8", newline="") as handle:
            metadata = list(csv.DictReader(handle))
        if len(metadata) != 360 or len({r["group_uid"] for r in metadata}) != 360:
            raise ValueError("Original public group index differs")
        self.metadata = [r for r in metadata if r["split"] in SPLITS]
        self.by_split: dict = {}
        seller_ids, item_ids = set(), set()
        for split in SPLITS:
            meta = {r["group_uid"]: r for r in self.metadata if r["split"] == split}
            for domain in "ABC":
                selected = [r for r in meta.values() if r["domain"] == domain]
                if sorted(int(r["group_index"]) for r in selected) != list(range(config["groups_per_domain"][split])):
                    raise ValueError("Domain positional index differs")
            rows: dict = defaultdict(lambda: defaultdict(list))
            with self.path(f"{split}/items.jsonl").open(encoding="utf-8") as handle:
                import json
                for line in handle:
                    row = json.loads(line)
                    if (set(row) != {"group_uid", "seller_uid", "item_uid", "title", "description"}
                            or row["group_uid"] not in meta or row["item_uid"] in item_ids):
                        raise ValueError("Public text fields or boundaries differ")
                    item_ids.add(row["item_uid"])
                    rows[row["group_uid"]][row["seller_uid"]].append(
                        (row["item_uid"], row["title"], row["description"]))
            if set(rows) != set(meta):
                raise ValueError("Missing public groups")
            groups = {}
            for uid, accounts in rows.items():
                sellers = tuple(sorted(accounts))
                if seller_ids.intersection(sellers):
                    raise ValueError("Seller crosses group/split boundary")
                seller_ids.update(sellers)
                group = data.Group(uid, sellers, tuple(tuple(sorted(accounts[s])) for s in sellers))
                group.validate()
                if int(meta[uid]["accounts"]) != 28 or int(meta[uid]["items"]) != sum(map(len, group.items)):
                    raise ValueError("Recorded group size differs")
                groups[uid] = group
            self.by_split[split] = groups
        self.train_label_parses = 0
        if train_labels:
            path = self.path("train/supervision/pairs.csv")
            labels = read_labels(path, self.groups("train"))
            self.train_label_parses = 1
            for group, values in zip(self.groups("train"), labels, strict=True):
                self.by_split["train"][group.uid] = data.Group(
                    group.uid, group.sellers, group.items, tuple(map(int, values)))

    def path(self, relative: str) -> Path:
        if relative not in {"groups.csv", "train/items.jsonl", "development/items.jsonl",
                            "train/supervision/pairs.csv"}:
            raise ValueError("Input outside diagnostic supply")
        record = self.manifest["files"][relative]
        result = data.verify(self.root / relative, record)
        self.checked[relative] = record
        return result

    def groups(self, split: str, domain: str = "ABC") -> list[data.Group]:
        if split not in SPLITS or not domain or not set(domain) <= set("ABC"):
            raise ValueError("Unapproved diagnostic split/domain")
        rows = sorted(self.metadata, key=lambda r: (r["domain"], int(r["group_index"])))
        return [self.by_split[split][r["group_uid"]] for r in rows
                if r["split"] == split and r["domain"] in domain]


def read_labels(path: Path, groups: list[data.Group]) -> np.ndarray:
    collected: dict = defaultdict(list)
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            collected[row["group_uid"]].append(row)
    if set(collected) != {g.uid for g in groups}:
        raise ValueError("Supervision group alignment differs")
    labels = np.array([data.align_labels(g, collected[g.uid]) for g in groups], dtype=np.uint8)
    if not np.all(labels.sum(axis=1) == 20):
        raise ValueError("Complete pair supervision differs")
    # Also checks the approved one/two-positive per-query structure.
    metrics.retrieval(labels, np.zeros(labels.shape), 28)
    return labels


def positions(count: int, epochs: int, seed: int) -> list[int]:
    rng = random.Random(seed)
    result = []
    for _ in range(epochs):
        order = list(range(count))
        rng.shuffle(order)
        result.extend(order)
    return result


def segment_seed(config: dict, anchor: str, phase: str) -> int:
    if phase not in ("shared", "branch"):
        raise ValueError("Unknown schedule phase")
    return data.seed_for(config["initialization_seed"], "diagnostic", anchor, phase)


def domain_rows(archive: Archive, split: str) -> dict[str, np.ndarray]:
    domains = {r["group_uid"]: r["domain"] for r in archive.metadata}
    return {d: np.array([i for i, g in enumerate(archive.groups(split)) if domains[g.uid] == d])
            for d in "ABC"}


def summarize(matrix: np.ndarray, rows: dict[str, np.ndarray]) -> dict:
    by_domain = {d: dict(zip(metrics.COLUMNS, matrix[ix].mean(axis=0).tolist())) for d, ix in rows.items()}
    return {"by_domain": by_domain, "domain_equal": dict(zip(metrics.COLUMNS,
            np.stack([matrix[ix].mean(axis=0) for ix in rows.values()]).mean(axis=0).tolist()))}


def comparisons(values: dict, rows: dict[str, np.ndarray], config: dict) -> dict:
    """Keep one selected evaluation domain per rotation, not zero-filled full-domain means."""
    ev = config["evaluation"]
    rng = np.random.default_rng(ev["bootstrap_seed"])
    draws = {d: rng.choice(ix, size=(ev["bootstrap_replicates"], len(ix)), replace=True)
             for d, ix in rows.items()}
    definitions = {
        "same_old_gain": ("same", "shared", 0),
        "cross_new_gain": ("cross", "shared", 1),
        "cross_old_change": ("cross", "shared", 0),
        "cross_minus_same_old": ("cross", "same", 0),
    }
    result = {}
    tail = (1 - ev["confidence_level"]) / 2
    for metric_index, metric in enumerate(metrics.COLUMNS):
        result[metric] = {}
        for name, (after, before, which_domain) in definitions.items():
            directions, samples = [], []
            for anchor, target in config["rotations"]:
                domain = (anchor, target)[which_domain]
                delta = values[anchor][after][:, metric_index] - values[anchor][before][:, metric_index]
                directions.append(float(delta[rows[domain]].mean()))
                samples.append(delta[draws[domain]].mean(axis=1))
            boot = np.stack(samples).mean(axis=0)
            result[metric][name] = {"mean": float(np.mean(directions)), "per_direction": directions,
                "conditional_95pct_interval": np.quantile(boot, [tail, 1 - tail]).tolist()}
    return result


def decision(comparison: dict, minimum: int = 2) -> dict:
    ap, query = comparison["average_precision"], comparison["map"]
    gain, loss = ap["cross_new_gain"], ap["cross_old_change"]
    matched = [i for i, (g, l) in enumerate(zip(gain["per_direction"], loss["per_direction"], strict=True))
               if g > 0 and l < 0]
    map_matched = [i for i in matched if query["cross_new_gain"]["per_direction"][i] > 0
                   and query["cross_old_change"]["per_direction"][i] < 0]
    conflict = (gain["conditional_95pct_interval"][0] > 0
                and loss["conditional_95pct_interval"][1] < 0 and len(matched) >= minimum)
    room = ap["same_old_gain"]
    return {"remaining_learning_room_supported": room["conditional_95pct_interval"][0] > 0
            and sum(x > 0 for x in room["per_direction"]) >= minimum,
            "ap_conflict_supported": conflict, "ap_matching_direction_indices": matched,
            "map_matching_direction_indices": map_matched,
            "action": "CONFLICT_SUPPORTED_METHOD_PLAN_STILL_REQUIRES_CONFIRMATION"
            if conflict and len(map_matched) >= minimum else "STOP_DIRECT_ANTI_FORGETTING_EXPANSION",
            "boundary": "Development diagnostic at fixed budget, not proof of convergence, absence of all forgetting, or real-market efficacy"}
