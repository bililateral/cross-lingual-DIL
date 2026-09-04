"""Shared, label-bounded primitives for the V6 style-transfer experiment."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
POLICY_PATH = (
    REPO_ROOT
    / "schema"
    / "step28_v13_v1_13_v9_4_1_v6_style_transfer_v2_policy.json"
)
PLACEHOLDER_RE = re.compile(r"[WN][0-9]{1,2}")


class StyleTransferContractError(RuntimeError):
    """Raised when a frozen scientific or ordinary replay boundary drifts."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_hash(payload: Mapping[str, object]) -> str:
    value = dict(payload)
    value.pop("canonical_self_hash", None)
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def verify_file_record(record: Mapping[str, object], *, label: str) -> Path:
    path = REPO_ROOT / str(record["path"])
    if not path.is_file():
        raise StyleTransferContractError(f"{label} is missing: {path}")
    if path.stat().st_size != int(record["size_bytes"]):
        raise StyleTransferContractError(f"{label} size drift")
    if sha256_file(path) != str(record["sha256"]):
        raise StyleTransferContractError(f"{label} SHA-256 drift")
    return path


def _load_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise StyleTransferContractError(f"Expected an object in {path}")
    return value


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise StyleTransferContractError(
                    f"Invalid JSONL at {path}:{line_number}"
                ) from exc
            if not isinstance(row, dict):
                raise StyleTransferContractError(
                    f"Expected object at {path}:{line_number}"
                )
            rows.append(row)
    return rows


def load_policy() -> dict:
    policy = _load_json(POLICY_PATH)
    if canonical_hash(policy) != policy.get("canonical_self_hash"):
        raise StyleTransferContractError("V6 style-transfer policy self-hash drift")
    if policy.get("status") != (
        "IMPLEMENTATION_AUTHORIZED_TRAIN_DEVELOPMENT_ONLY_AUDIT_TRUTH_SEALED"
    ):
        raise StyleTransferContractError("V6 style-transfer policy status drift")
    if policy["arms"] != [
        "generic",
        "v6_correct",
        "v6_permuted",
        "v6_order_format_mass_ablated",
    ]:
        raise StyleTransferContractError("V6 style-transfer arm registry drift")
    if policy["evaluation"].get("chinese_primary_aggregation") != "world_equal":
        raise StyleTransferContractError("Chinese primary AP aggregation drift")
    evaluation = policy["evaluation"]
    if evaluation.get("ranking_score") != "raw_account_cosine_float64":
        raise StyleTransferContractError("Target ranking score contract drift")
    source_replay = policy.get("source_replay", {})
    if source_replay.get("reference_split") != "v6_synthetic_audit":
        raise StyleTransferContractError("Source replay reference split drift")
    if source_replay.get("comparison") != "exact_float64_raw_cosine_array":
        raise StyleTransferContractError("Source replay comparison drift")
    if not source_replay.get("verify_before_checkpoint_save") or not source_replay.get(
        "verify_after_checkpoint_reload"
    ):
        raise StyleTransferContractError("Source replay checkpoint boundary drift")
    budgets = policy["target_optimization"]["budgets"]
    if evaluation.get("learning_curve_budgets") != list(budgets):
        raise StyleTransferContractError("Chinese learning-curve order drift")
    if evaluation.get("primary_budget") != "worlds_025":
        raise StyleTransferContractError("Chinese low-resource primary budget drift")
    if evaluation.get("saturation_budget") != "worlds_500":
        raise StyleTransferContractError("Chinese saturation budget drift")
    truth = policy["truth_boundary"]
    if truth["audit_a_labels_qrels_controllers_allowed"]:
        raise StyleTransferContractError("Audit A truth must remain sealed")
    if truth["audit_b_labels_qrels_controllers_allowed"]:
        raise StyleTransferContractError("Audit B truth must remain sealed")
    for label, record in policy["frozen_inputs"].items():
        path = verify_file_record(record, label=label)
        expected_self_hash = record.get("canonical_self_hash")
        if expected_self_hash is not None:
            observed = _load_json(path)
            observed_self_hash = observed.get(
                "canonical_self_hash", observed.get("manifest_self_sha256")
            )
            if observed_self_hash != expected_self_hash:
                raise StyleTransferContractError(f"{label} canonical hash drift")
    return policy


def _verify_manifest_payload(root: Path, manifest: Mapping[str, object]) -> None:
    for record in manifest["files"]:
        path = root / str(record["path"])
        if not path.is_file():
            raise StyleTransferContractError(f"Manifest payload is missing: {path}")
        if path.stat().st_size != int(record["size_bytes"]):
            raise StyleTransferContractError(f"Manifest payload size drift: {path}")
        if sha256_file(path) != record["sha256"]:
            raise StyleTransferContractError(f"Manifest payload hash drift: {path}")


def transferable_style_projection(value: str) -> str:
    # Import the one frozen implementation rather than maintaining a second
    # near-copy whose Unicode behavior could silently diverge.
    import step7_v6_build_synthetic_english_source as v6_builder

    return v6_builder.transferable_style_projection(value)


def count_placeholders(style_stream: str) -> int:
    return sum(1 for _ in PLACEHOLDER_RE.finditer(style_stream))


def split_style_windows(
    style_stream: str, maximum_placeholders: int = 100
) -> tuple[str, ...]:
    if not isinstance(style_stream, str) or not style_stream:
        raise StyleTransferContractError("Cannot split an empty style stream")
    if maximum_placeholders <= 0:
        raise StyleTransferContractError("Invalid placeholder window budget")
    matches = list(PLACEHOLDER_RE.finditer(style_stream))
    if not matches:
        raise StyleTransferContractError("Style stream contains no placeholders")
    boundaries = [0]
    for index in range(maximum_placeholders, len(matches), maximum_placeholders):
        boundaries.append(matches[index].start())
    boundaries.append(len(style_stream))
    windows = tuple(
        style_stream[boundaries[index] : boundaries[index + 1]]
        for index in range(len(boundaries) - 1)
    )
    if "".join(windows) != style_stream or any(not value for value in windows):
        raise StyleTransferContractError("Style window reconstruction failed")
    counts = [count_placeholders(value) for value in windows]
    if any(value < 1 or value > maximum_placeholders for value in counts):
        raise StyleTransferContractError("Style window placeholder budget failed")
    return windows


def ablate_order_format_mass(style_stream: str) -> str:
    placeholders = [
        (match.group(0)[0], int(match.group(0)[1:]))
        for match in PLACEHOLDER_RE.finditer(style_stream)
    ]
    if not placeholders:
        raise StyleTransferContractError("Cannot ablate a placeholder-free stream")
    placeholders.sort(key=lambda value: (value[0], value[1]))
    return " ".join(f"{kind}{length}" for kind, length in placeholders)


def deterministic_stream_derangement(
    account_uids: Sequence[str], seed: int
) -> dict[str, str]:
    unique = set(account_uids)
    if len(unique) != len(account_uids) or len(unique) < 2:
        raise StyleTransferContractError(
            "Derangement requires at least two unique account identifiers"
        )
    domain = f"step28-v6-style-permutation-v2:{seed}:".encode("ascii")
    order = sorted(
        unique,
        key=lambda uid: (
            hashlib.sha256(domain + uid.encode("utf-8")).digest(),
            uid.encode("utf-8"),
        ),
    )
    mapping = {
        uid: order[(index + 1) % len(order)] for index, uid in enumerate(order)
    }
    if set(mapping) != unique or set(mapping.values()) != unique:
        raise StyleTransferContractError("Derangement is not bijective")
    if any(uid == source_uid for uid, source_uid in mapping.items()):
        raise StyleTransferContractError("Derangement contains a fixed point")
    return mapping


class _DisjointSet:
    def __init__(self, values: Iterable[str]):
        self.parent = {value: value for value in values}

    def find(self, value: str) -> str:
        parent = self.parent[value]
        while parent != self.parent[parent]:
            self.parent[parent] = self.parent[self.parent[parent]]
            parent = self.parent[parent]
        while value != parent:
            next_value = self.parent[value]
            self.parent[value] = parent
            value = next_value
        return parent

    def union(self, left: str, right: str) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return
        first, second = sorted((left_root, right_root), key=lambda x: x.encode("utf-8"))
        self.parent[second] = first


def positive_components(
    account_uids: Iterable[str], pairs: Sequence[Mapping[str, object]]
) -> tuple[dict[str, str], dict[str, tuple[str, ...]]]:
    accounts = tuple(sorted(set(account_uids), key=lambda x: x.encode("utf-8")))
    dsu = _DisjointSet(accounts)
    for row in pairs:
        if int(row["label"]) == 1:
            dsu.union(str(row["account_left_uid"]), str(row["account_right_uid"]))
    members: dict[str, list[str]] = defaultdict(list)
    for uid in accounts:
        members[dsu.find(uid)].append(uid)
    normalized_members = {
        min(values, key=lambda x: x.encode("utf-8")): tuple(
            sorted(values, key=lambda x: x.encode("utf-8"))
        )
        for values in members.values()
    }
    account_to_component = {
        uid: component
        for component, values in normalized_members.items()
        for uid in values
    }
    return account_to_component, dict(
        sorted(normalized_members.items(), key=lambda value: value[0].encode("utf-8"))
    )


def component_loss_contributions(
    pairs: Sequence[Mapping[str, object]],
    account_to_component: Mapping[str, str],
) -> dict[str, tuple[tuple[int, float], ...]]:
    contributions: dict[str, dict[int, float]] = defaultdict(
        lambda: defaultdict(float)
    )
    for index, row in enumerate(pairs):
        left_component = account_to_component[str(row["account_left_uid"])]
        right_component = account_to_component[str(row["account_right_uid"])]
        weight = float(row["sample_weight"])
        if int(row["label"]) == 1:
            if left_component != right_component:
                raise StyleTransferContractError(
                    "Positive pair crosses recovered components"
                )
            contributions[left_component][index] += weight
        else:
            if left_component == right_component:
                raise StyleTransferContractError(
                    "Negative pair stays inside a recovered component"
                )
            contributions[left_component][index] += weight / 2.0
            contributions[right_component][index] += weight / 2.0
    return {
        component: tuple(sorted(values.items()))
        for component, values in sorted(contributions.items())
    }


def audit_component_class_mass(
    pairs: Sequence[Mapping[str, object]],
    contributions: Mapping[str, Sequence[tuple[int, float]]],
) -> dict[str, dict[str, float]]:
    output = {}
    for component, values in contributions.items():
        positive = sum(
            weight for index, weight in values if int(pairs[index]["label"]) == 1
        )
        negative = sum(
            weight for index, weight in values if int(pairs[index]["label"]) == 0
        )
        if not math.isclose(positive, 1.0, abs_tol=1e-12):
            raise StyleTransferContractError(
                f"Positive component mass drift for {component}: {positive}"
            )
        if not math.isclose(negative, 1.0, abs_tol=1e-12):
            raise StyleTransferContractError(
                f"Negative component mass drift for {component}: {negative}"
            )
        output[component] = {"positive": positive, "negative": negative}
    return output


def load_v6(policy: Mapping[str, object]) -> dict:
    root = REPO_ROOT / policy["roots"]["v6"]
    manifest = _load_json(root / "manifest.json")
    if manifest.get("manifest_self_sha256") != (
        policy["frozen_inputs"]["v6_manifest"]["canonical_self_hash"]
    ):
        raise StyleTransferContractError("V6 manifest canonical hash drift")
    _verify_manifest_payload(root, manifest)
    accounts = {
        row["account_uid"]: row
        for row in _read_jsonl(root / "public_accounts_style_projection.jsonl")
    }
    public_pairs = _read_csv(root / "public_pairs.csv")
    labels = {row["pair_uid"]: row for row in _read_csv(root / "labels.csv")}
    if set(labels) != {row["pair_uid"] for row in public_pairs}:
        raise StyleTransferContractError("V6 pair/label key mismatch")
    pairs = []
    for row in public_pairs:
        label_row = labels[row["pair_uid"]]
        merged = {
            **row,
            "label": int(label_row["label"]),
            "sample_weight": float(label_row["sample_weight"]),
        }
        left = merged["account_left_uid"]
        right = merged["account_right_uid"]
        if left not in accounts or right not in accounts:
            raise StyleTransferContractError("V6 pair references a missing account")
        if accounts[left]["split"] != row["split"] or accounts[right]["split"] != row["split"]:
            raise StyleTransferContractError("V6 pair/account split mismatch")
        pairs.append(merged)
    return {"root": root, "manifest": manifest, "accounts": accounts, "pairs": pairs}


def _join_projected_items(
    rows: Iterable[Mapping[str, object]],
    *,
    account_key: str,
    title_key: str,
    description_key: str,
) -> dict[str, str]:
    grouped: dict[str, list[Mapping[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[str(row[account_key])].append(row)
    streams = {}
    for account_uid, items in grouped.items():
        parts = []
        for row in sorted(items, key=lambda value: str(value["item_uid"]).encode("utf-8")):
            for field in (title_key, description_key):
                value = str(row.get(field, ""))
                if value:
                    projected = transferable_style_projection(value)
                    if projected:
                        parts.append(projected)
        stream = " ".join(parts)
        if not stream or count_placeholders(stream) == 0:
            raise StyleTransferContractError(
                f"Account has no projected style content: {account_uid}"
            )
        streams[account_uid] = stream
    return streams


def load_v5_style_streams(policy: Mapping[str, object]) -> dict[str, str]:
    root = REPO_ROOT / policy["roots"]["v5"]
    manifest = _load_json(root / "manifest.json")
    _verify_manifest_payload(root, manifest)
    rows = _read_jsonl(root / "public_items_full_clean.jsonl")
    streams = _join_projected_items(
        rows,
        account_key="account_uid",
        title_key="title_clean",
        description_key="description_clean",
    )
    if len(streams) != int(manifest["counts"]["published_accounts"]):
        raise StyleTransferContractError("V5 projected account count drift")
    return streams


def load_v5_pairs(policy: Mapping[str, object]) -> list[dict]:
    root = REPO_ROOT / policy["roots"]["v5"]
    public_rows = _read_csv(root / "public_pairs.csv")
    labels = {row["pair_uid"]: row for row in _read_csv(root / "labels.csv")}
    if set(labels) != {row["pair_uid"] for row in public_rows}:
        raise StyleTransferContractError("V5 pair/label key mismatch")
    rows = [
        {**row, "label": int(labels[row["pair_uid"]]["label"])}
        for row in public_rows
    ]
    if len(rows) != 84 or sum(row["label"] for row in rows) != 28:
        raise StyleTransferContractError("V5 frozen pair count drift")
    return rows


def _chinese_public_record(
    policy: Mapping[str, object], split: str, basename: str
) -> Mapping[str, object]:
    if split not in {"train", "development"}:
        raise StyleTransferContractError("Only Chinese train/development are allowed")
    manifest = _load_json(
        REPO_ROOT / policy["roots"]["chinese_public"] / "root_manifest.json"
    )
    relative = f"{split}/observed/{basename}"
    records = {row["path"]: row for row in manifest["public_files"]}
    if relative not in records:
        raise StyleTransferContractError(f"Missing Chinese public record: {relative}")
    return records[relative]


def verify_chinese_public_file(
    policy: Mapping[str, object], split: str, basename: str
) -> Path:
    record = _chinese_public_record(policy, split, basename)
    root = REPO_ROOT / policy["roots"]["chinese_public"]
    path = root / record["path"]
    if path.stat().st_size != int(record["size_bytes"]):
        raise StyleTransferContractError(f"Chinese {split} {basename} size drift")
    if sha256_file(path) != record["sha256"]:
        raise StyleTransferContractError(f"Chinese {split} {basename} hash drift")
    return path


def load_chinese_style_streams(
    policy: Mapping[str, object], split: str
) -> dict[str, dict[str, str]]:
    path = verify_chinese_public_file(policy, split, "redacted_items.jsonl")
    rows = _read_jsonl(path)
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[str(row["world_uid"])].append(row)
    worlds = {}
    for world_uid, world_rows in grouped.items():
        worlds[world_uid] = _join_projected_items(
            world_rows,
            account_key="seller_uid",
            title_key="title",
            description_key="description",
        )
        if len(worlds[world_uid]) != 28:
            raise StyleTransferContractError(
                f"Chinese seller count drift in {world_uid}"
            )
    if len(worlds) != 500:
        raise StyleTransferContractError(f"Chinese {split} world count drift")
    return dict(sorted(worlds.items(), key=lambda value: value[0].encode("utf-8")))


def _verified_private_input(
    policy: Mapping[str, object], role: str
) -> Path:
    execution_record = policy["frozen_inputs"]["train_development_policy"]
    execution_policy = _load_json(REPO_ROOT / execution_record["path"])
    if role not in execution_policy["authorized_private_inputs"]:
        raise StyleTransferContractError(f"Private input is not authorized: {role}")
    record = execution_policy["authorized_private_inputs"][role]
    path = REPO_ROOT / policy["roots"]["chinese_private"] / record["path"]
    if not path.is_file():
        raise StyleTransferContractError(f"Authorized private input is missing: {role}")
    if path.stat().st_size != int(record["size_bytes"]):
        raise StyleTransferContractError(f"Private {role} size drift")
    if sha256_file(path) != record["sha256"]:
        raise StyleTransferContractError(f"Private {role} hash drift")
    return path


def load_chinese_pairs(
    policy: Mapping[str, object], split: str, *, include_labels: bool
) -> list[dict]:
    endpoints_path = verify_chinese_public_file(
        policy, split, "complete_model_pair_endpoints.csv"
    )
    rows = _read_csv(endpoints_path)
    if len(rows) != 189000:
        raise StyleTransferContractError(f"Chinese {split} pair count drift")
    if not include_labels:
        return rows
    if split not in {"train", "development"}:
        raise StyleTransferContractError("Only train/development labels are allowed")
    label_path = _verified_private_input(policy, f"{split}_labels")
    labels = {
        row["canonical_pair_uid"]: int(row["label"])
        for row in _read_csv(label_path)
    }
    if set(labels) != {row["canonical_pair_uid"] for row in rows}:
        raise StyleTransferContractError(f"Chinese {split} pair/label mismatch")
    merged = [
        {**row, "label": labels[row["canonical_pair_uid"]]} for row in rows
    ]
    if sum(row["label"] for row in merged) != 10000:
        raise StyleTransferContractError(f"Chinese {split} positive count drift")
    return merged


def load_chinese_development_qrels(
    policy: Mapping[str, object],
) -> list[dict]:
    path = _verified_private_input(policy, "development_qrels")
    rows = _read_jsonl(path)
    if len(rows) != 14000:
        raise StyleTransferContractError("Chinese development qrels count drift")
    return rows


def validate_static_inputs() -> dict:
    policy = load_policy()
    v6 = load_v6(policy)
    train_accounts = {
        uid: row for uid, row in v6["accounts"].items() if row["split"] == "train"
    }
    train_pairs = [row for row in v6["pairs"] if row["split"] == "train"]
    account_to_component, components = positive_components(train_accounts, train_pairs)
    contributions = component_loss_contributions(train_pairs, account_to_component)
    audit_component_class_mass(train_pairs, contributions)
    if len(components) != policy["source_optimization"]["expected_train_components"]:
        raise StyleTransferContractError("V6 training component count drift")
    expected_updates = math.ceil(
        len(components)
        / int(policy["source_optimization"]["components_per_gradient_step"])
    )
    if expected_updates != policy["source_optimization"]["expected_updates_per_epoch"]:
        raise StyleTransferContractError("V6 source update count drift")
    for account in v6["accounts"].values():
        windows = split_style_windows(account["style_stream"])
        if len(windows) != 1 or count_placeholders(windows[0]) != 100:
            raise StyleTransferContractError("V6 one-window contract drift")
    v5_streams = load_v5_style_streams(policy)
    chinese_counts = {}
    chinese_world_ids = {}
    chinese_seller_ids = {}
    for split in ("train", "development"):
        worlds = load_chinese_style_streams(policy, split)
        chinese_world_ids[split] = set(worlds)
        chinese_seller_ids[split] = {
            seller_uid for sellers in worlds.values() for seller_uid in sellers
        }
        chinese_counts[split] = {
            "worlds": len(worlds),
            "sellers": sum(len(value) for value in worlds.values()),
        }
        verify_chinese_public_file(policy, split, "complete_model_pair_endpoints.csv")
    if chinese_world_ids["train"] & chinese_world_ids["development"]:
        raise StyleTransferContractError("Chinese train/development world leakage")
    if chinese_seller_ids["train"] & chinese_seller_ids["development"]:
        raise StyleTransferContractError("Chinese train/development seller leakage")
    return {
        "status": "PASSED_V6_STYLE_TRANSFER_STATIC_INPUTS",
        "policy_canonical_self_hash": policy["canonical_self_hash"],
        "v6_accounts": len(v6["accounts"]),
        "v6_pairs": len(v6["pairs"]),
        "v6_train_components": len(components),
        "v6_updates_per_epoch": expected_updates,
        "v5_accounts": len(v5_streams),
        "chinese": chinese_counts,
        "audit_a_truth_reads": 0,
        "audit_b_truth_reads": 0,
    }


if __name__ == "__main__":
    print(json.dumps(validate_static_inputs(), ensure_ascii=False, sort_keys=True, indent=2))
