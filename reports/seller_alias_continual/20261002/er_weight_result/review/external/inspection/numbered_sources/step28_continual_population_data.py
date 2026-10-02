    1  """Pinned public groups, authorized train supervision, and self-contained replay.
    2  
    3  The offline archive can index the full training export. Training functions receive
    4  only current groups, a retained memory object, or the cumulative arrived prefix.
    5  No function in this module opens owners, valid labels, or test labels.
    6  """
    7  from __future__ import annotations
    8  
    9  import csv
   10  import hashlib
   11  import itertools
   12  import json
   13  import random
   14  from collections import defaultdict
   15  from dataclasses import asdict, dataclass
   16  from pathlib import Path
   17  from typing import Any
   18  
   19  ROOT = Path(__file__).resolve().parents[1]
   20  POLICY_PATH = ROOT / "schema/step28_continual_population_policy.json"
   21  SPLITS = ("train", "development", "heldout")
   22  
   23  
   24  def read_json(path: Path) -> Any:
   25      return json.loads(path.read_text(encoding="utf-8"))
   26  
   27  
   28  def json_bytes(value: Any) -> bytes:
   29      return (json.dumps(value, ensure_ascii=False, sort_keys=True,
   30                         separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")
   31  
   32  
   33  def write_json(path: Path, value: Any) -> None:
   34      path.write_bytes(json_bytes(value))
   35  
   36  
   37  def sha256(path: Path) -> str:
   38      digest = hashlib.sha256()
   39      with path.open("rb") as stream:
   40          for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
   41              digest.update(block)
   42      return digest.hexdigest()
   43  
   44  
   45  def record(path: Path, base: Path) -> dict:
   46      return {"path": path.relative_to(base).as_posix(), "bytes": path.stat().st_size,
   47              "sha256": sha256(path)}
   48  
   49  
   50  def verify(path: Path, expected: dict) -> Path:
   51      if path.stat().st_size != expected["bytes"] or sha256(path) != expected["sha256"]:
   52          raise ValueError(f"File differs: {path.name}")
   53      return path
   54  
   55  
   56  def policy() -> dict:
   57      result = read_json(POLICY_PATH)
   58      if (result["orders"] != ["ABC", "BCA", "CAB"]
   59              or result["evaluation"]["heldout_labels_allowed"] is not False
   60              or result["model"]["pair_head"] != [1536, 128, 1]):
   61          raise ValueError("Population experiment contract differs")
   62      return result
   63  
   64  
   65  @dataclass(frozen=True)
   66  class Group:
   67      uid: str
   68      sellers: tuple[str, ...]
   69      # (item UID, title, description), account order matches sellers exactly.
   70      items: tuple[tuple[tuple[str, str, str], ...], ...]
   71      labels: tuple[int, ...] | None = None
   72  
   73      def validate(self, accounts: int = 28) -> None:
   74          if (len(self.sellers) != accounts or self.sellers != tuple(sorted(set(self.sellers)))
   75                  or len(self.items) != accounts):
   76              raise ValueError("Account identities or alignment differ")
   77          item_ids = []
   78          for rows in self.items:
   79              if not 2 <= len(rows) <= 8 or rows != tuple(sorted(rows)):
   80                  raise ValueError("Record multiplicity/order differs")
   81              for uid, title, description in rows:
   82                  if not uid or not title or not description:
   83                      raise ValueError("Empty public item")
   84                  item_ids.append(uid)
   85          if len(item_ids) != len(set(item_ids)):
   86              raise ValueError("Repeated item identity")
   87          if self.labels is not None:
   88              if len(self.labels) != accounts * (accounts - 1) // 2:
   89                  raise ValueError("Incomplete pair labels")
   90              if any(type(v) is not int or v not in (0, 1) for v in self.labels):
   91                  raise ValueError("Nonbinary supervision")
   92  
   93      def payload(self) -> dict:
   94          return asdict(self)
   95  
   96      @classmethod
   97      def restore(cls, value: dict) -> Group:
   98          result = cls(value["uid"], tuple(value["sellers"]),
   99                       tuple(tuple(tuple(row) for row in rows) for rows in value["items"]),
  100                       None if value["labels"] is None else tuple(value["labels"]))
  101          result.validate()
  102          return result
  103  
  104  
  105  def align_labels(group: Group, rows: list[dict]) -> tuple[int, ...]:
  106      expected = list(itertools.combinations(group.sellers, 2))
  107      found = {}
  108      for row in rows:
  109          key = (row["seller_uid_left"], row["seller_uid_right"])
  110          if (row["group_uid"] != group.uid or key in found or key[0] >= key[1]
  111                  or row["label"] not in ("0", "1")):
  112              raise ValueError("Pair identity, uniqueness or label differs")
  113          found[key] = int(row["label"])
  114      if set(found) != set(expected):
  115          raise ValueError("Pair set is not the complete account graph")
  116      return tuple(found[key] for key in expected)
  117  
  118  
  119  class Archive:
  120      """Offline supply; this is not the method's bounded historical memory."""
  121  
  122      def __init__(self, config: dict, *, train_labels: bool):
  123          self.root = ROOT / config["data_root"]
  124          for name, key in (("manifest.json", "data_manifest_sha256"),
  125                            ("validation.json", "data_validation_sha256")):
  126              if sha256(self.root / name) != config[key]:
  127                  raise ValueError(f"Pinned {name} differs")
  128          self.manifest = read_json(self.root / "manifest.json")
  129          if read_json(self.root / "validation.json")["status"] != "PASS_GENERATION_CONTRACT_NOT_MODEL_QUALIFICATION":
  130              raise ValueError("Dataset export qualification missing")
  131          self.checked: dict[str, dict] = {}
  132          with self.path("groups.csv").open(encoding="utf-8", newline="") as stream:
  133              self.metadata = list(csv.DictReader(stream))
  134          if len(self.metadata) != 360 or len({r["group_uid"] for r in self.metadata}) != 360:
  135              raise ValueError("Group partition differs")
  136          self.by_split: dict[str, dict[str, Group]] = {}
  137          all_sellers, all_items = set(), set()
  138          for split in SPLITS:
  139              meta = {r["group_uid"]: r for r in self.metadata if r["split"] == split}
  140              for domain in "ABC":
  141                  if sum(r["domain"] == domain for r in meta.values()) != config["groups_per_domain"][split]:
  142                      raise ValueError("Domain count differs")
  143              collected: dict = defaultdict(lambda: defaultdict(list))
  144              with self.path(f"{split}/items.jsonl").open(encoding="utf-8") as stream:
  145                  for line in stream:
  146                      row = json.loads(line)
  147                      if set(row) != {"group_uid", "seller_uid", "item_uid", "title", "description"}:
  148                          raise ValueError("Unexpected model-visible input field")
  149                      if row["group_uid"] not in meta or row["item_uid"] in all_items:
  150                          raise ValueError("Group boundary or repeated item")
  151                      all_items.add(row["item_uid"])
  152                      collected[row["group_uid"]][row["seller_uid"]].append(
  153                          (row["item_uid"], row["title"], row["description"]))
  154              if set(collected) != set(meta):
  155                  raise ValueError("Missing public group")
  156              groups = {}
  157              for uid, sellers in collected.items():
  158                  ordered = tuple(sorted(sellers))
  159                  if all_sellers.intersection(ordered):
  160                      raise ValueError("Account reused across groups/splits")
  161                  all_sellers.update(ordered)
  162                  group = Group(uid, ordered, tuple(tuple(sorted(sellers[s])) for s in ordered))
  163                  group.validate()
  164                  if int(meta[uid]["items"]) != sum(map(len, group.items)) or int(meta[uid]["accounts"]) != 28:
  165                      raise ValueError("Public group size differs")
  166                  groups[uid] = group
  167              self.by_split[split] = groups
  168          self.train_label_parses = 0
  169          if train_labels:
  170              rows = defaultdict(list)
  171              with self.path("train/supervision/pairs.csv").open(encoding="utf-8", newline="") as stream:
  172                  self.train_label_parses += 1
  173                  for row in csv.DictReader(stream):
  174                      rows[row["group_uid"]].append(row)
  175              if set(rows) != set(self.by_split["train"]):
  176                  raise ValueError("Train label groups differ")
  177              for uid, group in self.by_split["train"].items():
  178                  labels = align_labels(group, rows[uid])
  179                  if sum(labels) != 20:
  180                      raise ValueError("Train positive count differs")
  181                  self.by_split["train"][uid] = Group(uid, group.sellers, group.items, labels)
  182  
  183      def path(self, relative: str) -> Path:
  184          allowed = {"groups.csv", "train/supervision/pairs.csv"} | {f"{s}/items.jsonl" for s in SPLITS}
  185          if relative not in allowed:
  186              raise ValueError("This archive cannot open this supervision/input")
  187          expected = self.manifest["files"][relative]
  188          path = verify(self.root / relative, expected)
  189          self.checked[relative] = expected
  190          return path
  191  
  192      def groups(self, split: str, domains: str = "ABC") -> list[Group]:
  193          ids = [r["group_uid"] for r in sorted(self.metadata, key=lambda r: (r["domain"], int(r["group_index"])))
  194                 if r["split"] == split and r["domain"] in domains]
  195          return [self.by_split[split][uid] for uid in ids]
  196  
  197  
  198  def tuple_state(value: Any) -> Any:
  199      return tuple(tuple_state(v) for v in value) if isinstance(value, list) else value
  200  
  201  
  202  class Memory:
  203      """Algorithm R; all retained contents and selection RNG count toward bytes."""
  204  
  205      def __init__(self, seed: int, capacity: int = 6, maximum_bytes: int = 1048576):
  206          self.rng = random.Random(seed)
  207          self.capacity, self.maximum_bytes = capacity, maximum_bytes
  208          self.seen = 0
  209          self.groups: list[Group] = []
  210  
  211      def add_stage(self, current: list[Group]) -> None:
  212          # The runner supplies disjoint frozen stages exactly once. No all-history ID set.
  213          if len({g.uid for g in current}) != len(current) or {g.uid for g in current} & {g.uid for g in self.groups}:
  214              raise ValueError("Repeated current group")
  215          for group in current:
  216              if group.labels is None:
  217                  raise ValueError("Replay lacks authorized binary labels")
  218              self.seen += 1
  219              index = len(self.groups) if len(self.groups) < self.capacity else self.rng.randrange(self.seen)
  220              if index < self.capacity:
  221                  if index == len(self.groups):
  222                      self.groups.append(group)
  223                  else:
  224                      self.groups[index] = group
  225          self.to_bytes()
  226  
  227      def to_bytes(self) -> bytes:
  228          result = json_bytes({"capacity": self.capacity, "maximum_bytes": self.maximum_bytes,
  229                               "seen": self.seen, "rng": self.rng.getstate(),
  230                               "groups": [g.payload() for g in self.groups]})
  231          if len(result) > self.maximum_bytes:
  232              raise ValueError("Historical state exceeds the approved byte budget")
  233          return result
  234  
  235      @classmethod
  236      def from_bytes(cls, payload: bytes) -> Memory:
  237          obj = json.loads(payload)
  238          result = cls(0, obj["capacity"], obj["maximum_bytes"])
  239          result.seen = obj["seen"]
  240          result.rng.setstate(tuple_state(obj["rng"]))
  241          result.groups = [Group.restore(row) for row in obj["groups"]]
  242          if len(result.groups) != min(result.capacity, result.seen) or result.to_bytes() != payload:
  243              raise ValueError("Memory serialization differs")
  244          return result
  245  
  246  
  247  def seed_for(seed: int, *parts: Any) -> int:
  248      return int.from_bytes(hashlib.sha256(json_bytes([seed, *parts])).digest()[:8], "big") % (2**63 - 1)
  249  
  250  
  251  def schedule(groups: list[Group], epochs: int, seed: int) -> list[Group]:
  252      rng = random.Random(seed)
  253      result = []
  254      for _ in range(epochs):
  255          rows = sorted(groups, key=lambda g: g.uid)
  256          rng.shuffle(rows)
  257          result.extend(rows)
  258      return result
