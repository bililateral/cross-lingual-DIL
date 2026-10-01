"""World-aligned public text/numeric access; no audit truth or hidden replay lookup."""
from __future__ import annotations

import csv
import itertools
import json
from pathlib import Path
from typing import Any, Iterator

import numpy as np

import step28_continual_data as shared
import step28_continual_text as core

ROOT = Path(__file__).resolve().parents[1]
POLICY = ROOT / "schema/step28_continual_text_run_policy.json"
CODE_FILES = (
    "scripts/step28_continual_text_data.py", "scripts/step28_continual_text_run.py",
    "scripts/step28_continual_text_evaluate.py", "tests/test_step28_continual_text_run_contracts.py",
    "tests/test_step28_continual_text_contracts.py",
    "scripts/step28_continual_text.py", "scripts/step28_continual_text_check.py",
    "scripts/step28_continual_core.py", "scripts/step28_continual_data.py",
    "scripts/step28_continual_evaluate.py", "scripts/step28_labse_finetune_common.py",
    "scripts/step7_authorship_common.py", "schema/step28_continual_text_policy.json",
    "schema/step28_continual_policy.json", "schema/step28_continual_text_run_policy.json",
)


def code_records() -> list[dict]:
    return [shared.record(ROOT / name) for name in CODE_FILES]


def load_settings() -> dict:
    policy = shared.read_json(POLICY)
    benchmark = shared.read_json(shared.verify(policy["benchmark_policy"]))
    component = shared.read_json(shared.verify(policy["component_policy"]))
    if component != core.load_policy():
        raise ValueError("Different component contract")
    if (component["world_count"], component["stage_count"], component["worlds_per_stage"],
            component["initialization_seed"], component["planned_physical_updates"]) != (
            500, 5, 100, 20260907, 13800):
        raise ValueError("Different world or training schedule")
    if (policy["retained_inference_points"] != ["shared", "sequential_stage5", "er_stage5", "cumulative_stage5"]
            or policy["arm_execution_order"] != ["sequential", "er", "cumulative"]
            or policy["bootstrap_seed"] != 20260907 or policy["bootstrap_replicates"] != 2000
            or policy["no_strong_identity_columns"] != [0, 19] or policy["threshold_logit"] != 0
            or (policy["planned_physical_updates"], policy["unique_score_points"], policy["retained_inference_models"])
            != (13800, 42, 12)):
        raise ValueError("Different publication or evaluation contract")
    for split in ("train", "development"):
        if benchmark[split + "_labels"]["path"] != f"{shared.LABEL_ROOT}/{split}/pair_labels.csv":
            raise ValueError("Only authorized train/development binary supervision is allowed")
    return {"run": policy, "benchmark": benchmark, "component": component}


def verify_public(settings: dict) -> dict:
    benchmark, component = settings["benchmark"], settings["component"]
    for entry in benchmark["public_files"].values():
        shared.verify(entry)
    for name in ("arrival", "input_audit"):
        shared.verify(benchmark[name])
    if (component["arrival_path"] != benchmark["arrival"]["path"]
            or component["arrival_sha256"] != benchmark["arrival"]["sha256"]):
        raise ValueError("Text and benchmark arrival differ")
    for entry in component["public_texts"].values():
        shared.verify(entry)
    arrival = shared.read_json(ROOT / benchmark["arrival"]["path"])
    shared.validate_arrival(arrival, 500, 5, [11, 23, 37])
    checked = shared.read_json(shared.verify(settings["run"]["component_runtime"]))
    if checked["status"] != "PASSED_NEW_TEXT_COMPONENT_CHECK_NO_FORMAL_EXPERIMENT":
        raise ValueError("Required real text component evidence is absent")
    for entry in checked["files"]:
        shared.verify(entry)
    return arrival


class WorldArchive:
    """Immutable benchmark index, not a method's historical memory.

    Retains only public world IDs, endpoint IDs and text byte offsets. A read
    materializes one complete world's raw 51 columns and all its redacted text.
    The optional offline label NPY is accessed only for requested train worlds.
    """

    def __init__(self, split: str, base: Path, identity: Path, rows: Path, texts: Path,
                 *, world_count: int = 500, sellers: int = 28, items: int = 99,
                 labels: Path | None = None):
        if split not in ("train", "development") or (split == "development" and labels is not None):
            raise ValueError("Development archive must be label-free")
        self.split, self.base, self.identity, self.texts, self.labels = split, base, identity, texts, labels
        self.world_count, self.sellers, self.items = world_count, sellers, items
        self.pairs = sellers * (sellers - 1) // 2
        self.worlds: list[tuple[str, tuple[str, ...]]] = []
        with rows.open(encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            if tuple(reader.fieldnames or ()) != shared.ROW_FIELDS:
                raise ValueError("Public pair row schema differs")
            seen = set()
            for ordinal in range(world_count):
                block = list(itertools.islice(reader, self.pairs))
                if len(block) != self.pairs:
                    raise ValueError("Missing public pair rows")
                uid = block[0]["world_uid"]
                endpoints = tuple(sorted({r[k] for r in block for k in ("seller_uid_left", "seller_uid_right")}))
                expected = list(itertools.combinations(endpoints, 2))
                if len(endpoints) != sellers or seen.intersection(endpoints):
                    raise ValueError("Wrong or cross-world public endpoints")
                for row, edge in zip(block, expected, strict=True):
                    if (row["split"] != split or row["world_uid"] != uid
                            or int(row["world_ordinal"]) != ordinal
                            or (row["seller_uid_left"], row["seller_uid_right"]) != edge
                            or row["canonical_pair_uid"] != "||".join(edge)):
                        raise ValueError("Pair order is not canonical world-aligned K_n")
                self.worlds.append((uid, endpoints))
                seen.update(endpoints)
            if next(reader, None) is not None or len({w[0] for w in self.worlds}) != world_count:
                raise ValueError("Extra rows or duplicate world identifiers")
        self.text_offsets = []
        with texts.open("rb") as handle:
            for uid, _ in self.worlds:
                start = handle.tell()
                for _ in range(items):
                    raw = handle.readline()
                    if not raw:
                        raise ValueError("Missing public text items")
                    row = json.loads(raw)
                    if tuple(row) != core.text_common.REDACTED_ITEM_FIELDS or row["world_uid"] != uid:
                        raise ValueError("Text schema or world order differs from numeric rows")
                self.text_offsets.append((start, handle.tell()))
            if handle.read(1):
                raise ValueError("Extra text items")
        for path, columns in ((base, 24), (identity, 33)):
            mapped = np.load(path, mmap_mode="r", allow_pickle=False)
            try:
                if mapped.shape != (world_count * self.pairs, columns) or mapped.dtype.kind != "f":
                    raise ValueError("Wrong public numeric shape/dtype")
            finally:
                mapped._mmap.close()

    def read(self, ordinal: int) -> core.PairBatch:
        if type(ordinal) is not int or not 0 <= ordinal < self.world_count:
            raise ValueError("World outside archive")
        uid, endpoints = self.worlds[ordinal]
        start, end = self.text_offsets[ordinal]
        with self.texts.open("rb") as handle:
            handle.seek(start)
            rows = [json.loads(raw) for raw in handle.read(end - start).splitlines()]
        index = core.text_common.build_redacted_text_index(
            rows, expected_worlds=1, expected_sellers_per_world=self.sellers, expected_items_per_world=self.items)
        if set(index) != {uid} or tuple(sorted(index[uid])) != endpoints:
            raise ValueError("Text endpoints do not align with public numeric pairs")
        selected = shared.selected_rows([ordinal], self.world_count, self.pairs)
        numeric = np.column_stack((shared.array_rows(self.base, selected)[:, :18],
                                   shared.array_rows(self.identity, selected))).astype(np.float64, copy=False)
        labels = shared.array_rows(self.labels, selected) if self.labels is not None else None
        batch = core.PairBatch(index[uid], tuple(itertools.combinations(endpoints, 2)), numeric, labels)
        batch.validate(training=self.labels is not None, complete_world=self.sellers == 28)
        return batch

    def all_worlds(self) -> Iterator[core.PairBatch]:
        for ordinal in range(self.world_count):
            yield self.read(ordinal)


def archive(settings: dict, split: str, labels: Path | None = None) -> WorldArchive:
    files = settings["benchmark"]["public_files"]
    return WorldArchive(split, *(ROOT / files[split + "_" + name]["path"] for name in ("base", "identity", "rows")),
                        ROOT / settings["component"]["public_texts"][split]["path"], labels=labels)


def disjoint_archives(train: WorldArchive, development: WorldArchive) -> None:
    for index in (0, 1):
        left = {x for row in train.worlds for x in ([row[0]] if index == 0 else row[1])}
        right = {x for row in development.worlds for x in ([row[0]] if index == 0 else row[1])}
        if left.intersection(right):
            raise ValueError("Train/development public identifiers overlap")


class StageSource:
    def __init__(self, archive: WorldArchive, stages: list[list[int]]):
        flat = [w for group in stages for w in group]
        if not stages or any(not group for group in stages) or sorted(flat) != list(range(archive.world_count)):
            raise ValueError("Incomplete, repeated or invalid arrival worlds")
        self.archive, self.stages = archive, stages
        self.accesses: list[dict[str, Any]] = []

    def load(self, arm: str, stage: int) -> list[core.PairBatch]:
        if arm not in ("shared", "sequential", "er", "cumulative") or not 1 <= stage <= len(self.stages):
            raise ValueError("Invalid requested arm/stage")
        if (stage == 1) != (arm == "shared"):
            raise ValueError("First stage is physically shared once")
        worlds = ([w for group in self.stages[:stage] for w in group]
                  if arm == "cumulative" else self.stages[stage - 1])
        self.accesses.append({"arm": arm, "stage": stage,
                              "scope": "arrived_prefix" if arm == "cumulative" else "current",
                              "world_ordinals": list(worlds), "rows": len(worlds) * self.archive.pairs})
        return [self.archive.read(w) for w in worlds]
