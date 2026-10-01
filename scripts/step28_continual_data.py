"""Pinned public inputs and stage-bounded access for the continual comparison."""
from __future__ import annotations

import csv
import hashlib
import json
from itertools import zip_longest
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
POLICY = ROOT / "schema/step28_continual_policy.json"
LABEL_ROOT = "private_custody/step28_v13_v1_13_v9_4_1_formal_500x4_attempt1_20260829"
ROW_FIELDS = ("split", "world_ordinal", "world_uid", "canonical_pair_uid",
              "seller_uid_left", "seller_uid_right")
LABEL_FIELDS = ("canonical_pair_uid", "world_uid", "label")
CODE_FILES = ("scripts/step28_continual_core.py", "scripts/step28_continual_data.py",
              "scripts/step28_continual_run.py", "scripts/step28_continual_evaluate.py")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def record(path: Path, base: Path = ROOT) -> dict[str, Any]:
    # CLI output/check paths are normally relative; normalize both operands
    # before recording a relative artifact path. No experiment bytes change.
    path, base = path.resolve(), base.resolve()
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1048576), b""):
            digest.update(block)
    return {"path": path.relative_to(base).as_posix(), "size_bytes": path.stat().st_size,
            "sha256": digest.hexdigest()}


def verify(spec: dict[str, Any], base: Path = ROOT) -> Path:
    path = base / spec["path"]
    if record(path, base) != spec:
        raise ValueError(f"File differs: {spec['path']}")
    return path


def write_json(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")


def code_records() -> list[dict[str, Any]]:
    return [record(ROOT / path) for path in (*CODE_FILES, "schema/step28_continual_policy.json")]


def load_policy() -> dict[str, Any]:
    policy = read_json(POLICY)
    if (policy["world_count"], policy["sellers_per_world"], policy["positive_pairs_per_world"],
            policy["stage_count"], policy["epochs"], policy["batch_size"], policy["memory_bytes"],
            policy["initialization_seed"], policy["threshold_logit"]) != (
            500, 28, 20, 5, 5, 256, 524288, 20260907, 0.0):
        raise ValueError("Frozen comparison settings differ")
    if policy["order_seeds"] != [11, 23, 37] or policy["feature_count"] != 57:
        raise ValueError("Frozen seeds or features differ")
    for split in ("train", "development"):
        if policy[split + "_labels"]["path"] != f"{LABEL_ROOT}/{split}/pair_labels.csv":
            raise ValueError("Unapproved supervision path; Audit truth must remain sealed")
    if policy["no_strong_identity_columns"] != [0, 19] or (
            policy["bootstrap_seed"], policy["bootstrap_replicates"]) != (20260907, 2000):
        raise ValueError("Frozen evaluation configuration differs")
    return policy


def verify_public(policy: dict[str, Any]) -> dict[str, Any]:
    for spec in policy["public_files"].values():
        verify(spec)
    for key in ("arrival", "input_audit", "core_runtime", "core_code"):
        verify(policy[key])
    arrival = read_json(ROOT / policy["arrival"]["path"])
    validate_arrival(arrival, policy["world_count"], policy["stage_count"], policy["order_seeds"])
    if read_json(ROOT / policy["core_runtime"]["path"])["script_sha256"] != policy["core_code"]["sha256"]:
        raise ValueError("Core runtime evidence covers another implementation")
    return arrival


def validate_arrival(arrival: dict, world_count: int, stage_count: int, seeds: list[int]) -> None:
    if (arrival["kind"] != "random_whole_world_reference" or arrival["distinct_domains_claimed"]
            or arrival["world_count"] != world_count or arrival["stage_count"] != stage_count
            or [x["seed"] for x in arrival["orders"]] != seeds):
        raise ValueError("Arrival scope differs")
    for order in arrival["orders"]:
        stages = order["train_world_ordinals_by_stage"]
        flat = [i for group in stages for i in group]
        if (len(stages) != stage_count or any(len(g) != world_count // stage_count for g in stages)
                or sorted(flat) != list(range(world_count))
                or flat != np.random.default_rng(order["seed"]).permutation(world_count).tolist()):
            raise ValueError("Arrival has repeated, missing, or reordered worlds")


def aligned_labels(label_path: Path, row_path: Path, split: str, world_count: int,
                   sellers_per_world: int, positives: int) -> np.ndarray:
    """Offline benchmark/evaluation parsing, NEVER a method's historical cache."""
    pairs = sellers_per_world * (sellers_per_world - 1) // 2
    labels = np.empty(world_count * pairs, dtype=np.uint8)
    with label_path.open(encoding="utf-8-sig", newline="") as lh, row_path.open(
            encoding="utf-8-sig", newline="") as rh:
        truth, rows = csv.DictReader(lh), csv.DictReader(rh)
        if tuple(truth.fieldnames or ()) != LABEL_FIELDS or tuple(rows.fieldnames or ()) != ROW_FIELDS:
            raise ValueError("Label/public CSV schema differs")
        count = 0
        for count, (label, row) in enumerate(zip_longest(truth, rows), 1):
            if label is None or row is None or count > len(labels):
                raise ValueError("Label/public row count differs")
            if (row["split"] != split or int(row["world_ordinal"]) != (count - 1) // pairs
                    or row["canonical_pair_uid"] != row["seller_uid_left"] + "||" + row["seller_uid_right"]
                    or label["canonical_pair_uid"] != row["canonical_pair_uid"]
                    or label["world_uid"] != row["world_uid"] or label["label"] not in ("0", "1")):
                raise ValueError("Label/public row alignment differs")
            labels[count - 1] = int(label["label"])
    if count != len(labels) or not np.all(labels.reshape(world_count, pairs).sum(axis=1) == positives):
        raise ValueError("Label count/class balance differs")
    return labels


def selected_rows(worlds: list[int], world_count: int, pairs: int) -> np.ndarray:
    if not worlds or len(worlds) != len(set(worlds)) or any(type(w) is not int or not 0 <= w < world_count for w in worlds):
        raise ValueError("Invalid world selection")
    return (np.asarray(worlds, dtype=np.int64)[:, None] * pairs + np.arange(pairs)).ravel()


def array_rows(path: Path, indices: np.ndarray) -> np.ndarray:
    mapped = np.load(path, mmap_mode="r", allow_pickle=False)
    try:
        return np.array(mapped[indices], copy=True)
    finally:
        mapped._mmap.close()


class StageSource:
    """An immutable benchmark archive with explicit, logged stage access.

    Methods receive current rows only. Cumulative retraining alone gets the
    arrived prefix. No full feature/label array is retained in this object.
    This is an ordinary audited data boundary, not filesystem access control.
    """

    def __init__(self, base: Path, identity: Path, labels: Path, stages: list[list[int]],
                 pairs: int, world_count: int):
        self.base, self.identity, self.labels = base, identity, labels
        self.stages, self.pairs, self.world_count = stages, pairs, world_count
        self.accesses: list[dict[str, Any]] = []

    def load(self, stage: int, cumulative: bool = False) -> tuple[np.ndarray, np.ndarray]:
        if not 1 <= stage <= len(self.stages):
            raise ValueError("Stage outside arrival schedule")
        worlds = ([w for group in self.stages[:stage] for w in group]
                  if cumulative else list(self.stages[stage - 1]))
        indices = selected_rows(worlds, self.world_count, self.pairs)
        x = np.concatenate((array_rows(self.base, indices), array_rows(self.identity, indices)), axis=1)
        y = array_rows(self.labels, indices)
        if x.shape != (len(indices), 57) or y.shape != (len(indices),):
            raise ValueError("Selected input shapes differ")
        self.accesses.append({"stage": stage, "scope": "arrived_prefix" if cumulative else "current",
                              "world_ordinals": worlds, "rows": len(indices)})
        return x, y
