"""Pinned public groups, authorized train supervision, and self-contained replay.

The offline archive can index the full training export. Training functions receive
only current groups, a retained memory object, or the cumulative arrived prefix.
No function in this module opens owners, valid labels, or test labels.
"""
from __future__ import annotations

import csv
import hashlib
import itertools
import json
import random
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
POLICY_PATH = ROOT / "schema/step28_continual_population_policy.json"
SPLITS = ("train", "development", "heldout")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def json_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True,
                       separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")


def write_json(path: Path, value: Any) -> None:
    path.write_bytes(json_bytes(value))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def record(path: Path, base: Path) -> dict:
    return {"path": path.relative_to(base).as_posix(), "bytes": path.stat().st_size,
            "sha256": sha256(path)}


def verify(path: Path, expected: dict) -> Path:
    if path.stat().st_size != expected["bytes"] or sha256(path) != expected["sha256"]:
        raise ValueError(f"File differs: {path.name}")
    return path


def policy() -> dict:
    result = read_json(POLICY_PATH)
    if (result["orders"] != ["ABC", "BCA", "CAB"]
            or result["evaluation"]["heldout_labels_allowed"] is not False
            or result["model"]["pair_head"] != [1536, 128, 1]):
        raise ValueError("Population experiment contract differs")
    return result


@dataclass(frozen=True)
class Group:
    uid: str
    sellers: tuple[str, ...]
    # (item UID, title, description), account order matches sellers exactly.
    items: tuple[tuple[tuple[str, str, str], ...], ...]
    labels: tuple[int, ...] | None = None

    def validate(self, accounts: int = 28) -> None:
        if (len(self.sellers) != accounts or self.sellers != tuple(sorted(set(self.sellers)))
                or len(self.items) != accounts):
            raise ValueError("Account identities or alignment differ")
        item_ids = []
        for rows in self.items:
            if not 2 <= len(rows) <= 8 or rows != tuple(sorted(rows)):
                raise ValueError("Record multiplicity/order differs")
            for uid, title, description in rows:
                if not uid or not title or not description:
                    raise ValueError("Empty public item")
                item_ids.append(uid)
        if len(item_ids) != len(set(item_ids)):
            raise ValueError("Repeated item identity")
        if self.labels is not None:
            if len(self.labels) != accounts * (accounts - 1) // 2:
                raise ValueError("Incomplete pair labels")
            if any(type(v) is not int or v not in (0, 1) for v in self.labels):
                raise ValueError("Nonbinary supervision")

    def payload(self) -> dict:
        return asdict(self)

    @classmethod
    def restore(cls, value: dict) -> Group:
        result = cls(value["uid"], tuple(value["sellers"]),
                     tuple(tuple(tuple(row) for row in rows) for rows in value["items"]),
                     None if value["labels"] is None else tuple(value["labels"]))
        result.validate()
        return result


def align_labels(group: Group, rows: list[dict]) -> tuple[int, ...]:
    expected = list(itertools.combinations(group.sellers, 2))
    found = {}
    for row in rows:
        key = (row["seller_uid_left"], row["seller_uid_right"])
        if (row["group_uid"] != group.uid or key in found or key[0] >= key[1]
                or row["label"] not in ("0", "1")):
            raise ValueError("Pair identity, uniqueness or label differs")
        found[key] = int(row["label"])
    if set(found) != set(expected):
        raise ValueError("Pair set is not the complete account graph")
    return tuple(found[key] for key in expected)


class Archive:
    """Offline supply; this is not the method's bounded historical memory."""

    def __init__(self, config: dict, *, train_labels: bool):
        self.root = ROOT / config["data_root"]
        for name, key in (("manifest.json", "data_manifest_sha256"),
                          ("validation.json", "data_validation_sha256")):
            if sha256(self.root / name) != config[key]:
                raise ValueError(f"Pinned {name} differs")
        self.manifest = read_json(self.root / "manifest.json")
        if read_json(self.root / "validation.json")["status"] != "PASS_GENERATION_CONTRACT_NOT_MODEL_QUALIFICATION":
            raise ValueError("Dataset export qualification missing")
        self.checked: dict[str, dict] = {}
        with self.path("groups.csv").open(encoding="utf-8", newline="") as stream:
            self.metadata = list(csv.DictReader(stream))
        if len(self.metadata) != 360 or len({r["group_uid"] for r in self.metadata}) != 360:
            raise ValueError("Group partition differs")
        self.by_split: dict[str, dict[str, Group]] = {}
        all_sellers, all_items = set(), set()
        for split in SPLITS:
            meta = {r["group_uid"]: r for r in self.metadata if r["split"] == split}
            for domain in "ABC":
                if sum(r["domain"] == domain for r in meta.values()) != config["groups_per_domain"][split]:
                    raise ValueError("Domain count differs")
            collected: dict = defaultdict(lambda: defaultdict(list))
            with self.path(f"{split}/items.jsonl").open(encoding="utf-8") as stream:
                for line in stream:
                    row = json.loads(line)
                    if set(row) != {"group_uid", "seller_uid", "item_uid", "title", "description"}:
                        raise ValueError("Unexpected model-visible input field")
                    if row["group_uid"] not in meta or row["item_uid"] in all_items:
                        raise ValueError("Group boundary or repeated item")
                    all_items.add(row["item_uid"])
                    collected[row["group_uid"]][row["seller_uid"]].append(
                        (row["item_uid"], row["title"], row["description"]))
            if set(collected) != set(meta):
                raise ValueError("Missing public group")
            groups = {}
            for uid, sellers in collected.items():
                ordered = tuple(sorted(sellers))
                if all_sellers.intersection(ordered):
                    raise ValueError("Account reused across groups/splits")
                all_sellers.update(ordered)
                group = Group(uid, ordered, tuple(tuple(sorted(sellers[s])) for s in ordered))
                group.validate()
                if int(meta[uid]["items"]) != sum(map(len, group.items)) or int(meta[uid]["accounts"]) != 28:
                    raise ValueError("Public group size differs")
                groups[uid] = group
            self.by_split[split] = groups
        self.train_label_parses = 0
        if train_labels:
            rows = defaultdict(list)
            with self.path("train/supervision/pairs.csv").open(encoding="utf-8", newline="") as stream:
                self.train_label_parses += 1
                for row in csv.DictReader(stream):
                    rows[row["group_uid"]].append(row)
            if set(rows) != set(self.by_split["train"]):
                raise ValueError("Train label groups differ")
            for uid, group in self.by_split["train"].items():
                labels = align_labels(group, rows[uid])
                if sum(labels) != 20:
                    raise ValueError("Train positive count differs")
                self.by_split["train"][uid] = Group(uid, group.sellers, group.items, labels)

    def path(self, relative: str) -> Path:
        allowed = {"groups.csv", "train/supervision/pairs.csv"} | {f"{s}/items.jsonl" for s in SPLITS}
        if relative not in allowed:
            raise ValueError("This archive cannot open this supervision/input")
        expected = self.manifest["files"][relative]
        path = verify(self.root / relative, expected)
        self.checked[relative] = expected
        return path

    def groups(self, split: str, domains: str = "ABC") -> list[Group]:
        ids = [r["group_uid"] for r in sorted(self.metadata, key=lambda r: (r["domain"], int(r["group_index"])))
               if r["split"] == split and r["domain"] in domains]
        return [self.by_split[split][uid] for uid in ids]


def tuple_state(value: Any) -> Any:
    return tuple(tuple_state(v) for v in value) if isinstance(value, list) else value


class Memory:
    """Algorithm R; all retained contents and selection RNG count toward bytes."""

    def __init__(self, seed: int, capacity: int = 6, maximum_bytes: int = 1048576):
        self.rng = random.Random(seed)
        self.capacity, self.maximum_bytes = capacity, maximum_bytes
        self.seen = 0
        self.groups: list[Group] = []

    def add_stage(self, current: list[Group]) -> None:
        # The runner supplies disjoint frozen stages exactly once. No all-history ID set.
        if len({g.uid for g in current}) != len(current) or {g.uid for g in current} & {g.uid for g in self.groups}:
            raise ValueError("Repeated current group")
        for group in current:
            if group.labels is None:
                raise ValueError("Replay lacks authorized binary labels")
            self.seen += 1
            index = len(self.groups) if len(self.groups) < self.capacity else self.rng.randrange(self.seen)
            if index < self.capacity:
                if index == len(self.groups):
                    self.groups.append(group)
                else:
                    self.groups[index] = group
        self.to_bytes()

    def to_bytes(self) -> bytes:
        result = json_bytes({"capacity": self.capacity, "maximum_bytes": self.maximum_bytes,
                             "seen": self.seen, "rng": self.rng.getstate(),
                             "groups": [g.payload() for g in self.groups]})
        if len(result) > self.maximum_bytes:
            raise ValueError("Historical state exceeds the approved byte budget")
        return result

    @classmethod
    def from_bytes(cls, payload: bytes) -> Memory:
        obj = json.loads(payload)
        result = cls(0, obj["capacity"], obj["maximum_bytes"])
        result.seen = obj["seen"]
        result.rng.setstate(tuple_state(obj["rng"]))
        result.groups = [Group.restore(row) for row in obj["groups"]]
        if len(result.groups) != min(result.capacity, result.seen) or result.to_bytes() != payload:
            raise ValueError("Memory serialization differs")
        return result


def seed_for(seed: int, *parts: Any) -> int:
    return int.from_bytes(hashlib.sha256(json_bytes([seed, *parts])).digest()[:8], "big") % (2**63 - 1)


def schedule(groups: list[Group], epochs: int, seed: int) -> list[Group]:
    rng = random.Random(seed)
    result = []
    for _ in range(epochs):
        rows = sorted(groups, key=lambda g: g.uid)
        rng.shuffle(rows)
        result.extend(rows)
    return result
