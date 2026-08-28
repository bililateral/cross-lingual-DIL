#!/usr/bin/env python3
"""Build the V9.4 1,004-world Chinese method-qualification root.

The public renderer never receives pair labels.  Controller membership is used
only to allocate intended author styles and identity assets, and truth is
projected after all public/model rows for a world have been frozen in memory.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import hmac
import json
import os
import re
import shutil
import subprocess
import sys
import unicodedata
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any, TextIO

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import step28_history_common as history
import step28_v13_identity_values as identity_values
import step28_v13_production_chain as production
import step28_v13_profiles as profiles_module
import step28_v13_text_renderer as text_renderer
import step28_v13_v1_13_balanced_world_schedule_v9_4 as schedule_v94
import step28_v13_v1_13_joint_noise_signatures_v9_4 as noise_v94
import step28_v13_v1_13_model_visible_prebuild_source_v9_4 as prebuild_v94
import step28_v13_v1_13_model_visible_public_replay_v9_4 as replay_v94
import step28_v13_v1_13_scientific_common_v9 as scientific_common


VERSION = "2026-08-28-step28-v13-v1-13-method-root-builder-v9-4-v1"
POLICY_PATH = ROOT / "schema" / "step28_v13_v1_13_v9_4_method_root_policy.json"
BASE_POLICY_PATH = ROOT / "schema" / "step28_v13_synthetic_chinese_dataset_policy.json"
TEMPLATE_PATH = ROOT / "schema" / "step28_v13_v1_13_candidate_text_templates_v9.json"
PREBUILD_RESULT_PATH = (
    ROOT
    / "reports"
    / "step28_synthetic_chinese_dataset"
    / "v9_4_prebuild_gate_attempt3_20260828"
    / "prebuild_gate_result.json"
)
TIME_KEY_PATH = (
    ROOT
    / "private_custody"
    / "step28_v13_v1_13_v9_4_prebuild_gate_attempt3_20260828"
    / "time_key.consumed.bin"
)
ARTIFICIAL_CODE_RE = re.compile(r"Q[A-P]{10}")
FORBIDDEN_VISIBLE_RE = re.compile(
    r"(?:v9[_-]?4|audit[_ -]?[ab]|development|controller|seller|world|"
    r"mechanism|candidate|控制者|候选编号|世界编号)",
    re.IGNORECASE,
)
SPLITS = ("train", "development", "audit_a", "audit_b")
FORMAL_COUNTS = {"train": 500, "development": 500, "audit_a": 2, "audit_b": 2}
SMOKE_COUNTS = {split: 1 for split in SPLITS}
MARKETS = ("青岚集市", "远山商街", "星港广场")
IDENTITY_TYPES = (
    "telegram",
    "email",
    "bat",
    "qq",
    "wechat",
    "phone",
    "crypto_wallet",
    "external_url",
)
DIRECT_TYPES = IDENTITY_TYPES[:-1]
STYLE_FIELDS = (
    "separator",
    "ending",
    "line_mode",
    "english_tag",
    "traditional_variant",
    "repeat_punctuation",
)
TITLE_PATTERNS = (
    "现货{product}{attribute} {modifier}",
    "{product}日常组合 {attribute} {modifier}",
    "可选规格的{product} {attribute}",
    "{product}基础套装 {attribute}",
    "本周整理：{product} {attribute} {modifier}",
    "{attribute}{product} {modifier}",
    "多用途{product} {attribute}",
    "{product}组合上架 {attribute} {modifier}",
    "今日整理{product} 规格{attribute}",
    "常规供应{product} {attribute} {modifier}",
    "{product}补充上架 版本{attribute}",
    "可直接选择的{product} {attribute} {modifier}",
)
DESCRIPTION_PATTERNS = (
    "页面内容对应{product}{separator}当前为{attribute}{separator}{delivery}{ending}{service}{ending}",
    "本条包含{product}{separator}规格标记为{attribute}{separator}{service}{ending}{delivery}{ending}",
    "此处提供{product}{separator}版本是{attribute}{ending}{delivery}{separator}{service}{ending}",
    "{product}采用常规包装{separator}{delivery}{separator}{service}{ending}页面参数可选{attribute}{ending}",
    "请先确认{product}的{attribute}{separator}{delivery}{ending}{service}{separator}以订单备注为准{ending}",
    "当前页面为{product}{separator}{delivery}{separator}版本说明为{attribute}{ending}{service}{ending}",
    "本页说明{product}的选项{separator}{attribute}{separator}{service}{ending}{delivery}{ending}",
    "{attribute}的{product}已完成整理{separator}{delivery}{ending}{service}{separator}请保留订单说明{ending}",
    "提供{product}的常规选项{separator}{attribute}{ending}{service}{separator}{delivery}{ending}",
    "下单前核对{product}与{attribute}{separator}{delivery}{ending}{service}{ending}",
    "本次整理的是{product}{separator}可选{attribute}{ending}{delivery}{separator}{service}{ending}",
    "页面所列{product}按{attribute}准备{separator}{service}{ending}{delivery}{ending}",
)
TITLE_DETAIL_POOLS = (
    ("日常适用", "桌面适用", "出行适用", "临时备用", "长期备用", "基础选配", "组合选配", "按需选配"),
    ("常规整理", "分批整理", "小量整理", "集中整理", "本期补充", "近期补充", "现行版本", "通用版本"),
    ("独立包装", "简易包装", "加固包装", "轻便包装", "附带说明", "清单齐全", "规格可选", "参数可选"),
    ("可先核对", "按单准备", "依次处理", "备注优先", "选项清楚", "说明完整", "用途明确", "数量可调"),
)
DESCRIPTION_DETAIL_POOLS = (
    ("包装会按所选规格分别整理", "不同选项会在清单中分开标明", "准备前会再次核对页面备注", "内容会依照确认后的清单安排", "多个版本会按顺序分别处理", "随附说明会与对应选项放在一起", "数量和外观会在交付前复核", "临时调整需要在处理前写明"),
    ("页面说明用于区分当前选项", "下单备注应写清保留项目", "收到后可按清单逐项核对", "规格相近时请同时查看用途", "组合内容以最终确认项为准", "缺少参数时会先等待补充", "基础检查完成后再安排交付", "售后反馈请说明具体差异"),
    ("常规需求按现有顺序处理", "特殊需求需要单独列出", "同类版本不会混在一个选项中", "配件与说明会分别核对", "外观和数量均以备注为准", "页面更新不会改变已确认清单", "每项内容都会保留对应说明", "选择完成后再进入准备环节"),
    ("如有疑问可先确认适用范围", "请在开始处理前补全信息", "不同批次应分别选择", "同一订单可保留多项备注", "说明不清时不要直接混选", "交付前可再次检查清单", "收到内容后先核对数量", "组合调整应列出替换项目"),
)
MECHANISM_SLOTS = {
    "G_A": {
        3: (
            "single_hop_rotation",
            "corroborated_two_hop_rotation",
            "corroborated_two_hop_rotation",
            "cross_market_stable_reuse",
        ),
        2: (
            "single_identity_stable_reuse",
            "single_identity_stable_reuse",
            "multi_type_identity_reuse",
            "multi_type_identity_reuse",
            "cross_market_stable_reuse",
            "sparse_history",
            "same_controller_no_direct_share",
            "zero_visible_identity_history",
        ),
    },
    "G_B": {
        3: (
            "single_hop_rotation",
            "corroborated_two_hop_rotation",
            "corroborated_two_hop_rotation",
            "corroborated_two_hop_rotation",
        ),
        2: (
            "single_identity_stable_reuse",
            "multi_type_identity_reuse",
            "cross_market_stable_reuse",
            "sparse_history",
            "sparse_history",
            "same_controller_no_direct_share",
            "zero_visible_identity_history",
            "zero_visible_identity_history",
        ),
    },
}


class MethodRootBuildError(ValueError):
    """Fail-closed error for the V9.4 method-root builder."""


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def hmac_digest(key: bytes, *parts: object) -> bytes:
    payload = "\x1f".join(str(value) for value in parts).encode("utf-8")
    return hmac.new(key, payload, hashlib.sha256).digest()


def ranked(values: Iterable[str], key: bytes, *parts: object) -> list[str]:
    return sorted(
        values,
        key=lambda value: (hmac_digest(key, *parts, value), value.encode("utf-8")),
    )


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise MethodRootBuildError(f"JSON object required: {path}")
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")


def git_head() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True,
        text=True, encoding="utf-8", capture_output=True,
    )
    return result.stdout.strip()


@dataclass(frozen=True)
class Authorities:
    text: bytes
    identity: bytes
    style: bytes
    audit_schedule: bytes
    uid: bytes
    time: bytes


@dataclass(frozen=True)
class PublicWorld:
    split: str
    ordinal: int
    world_uid: str
    seller_uids: tuple[str, ...]
    noise_slots: tuple[int, ...]
    controller_groups: tuple[tuple[str, ...], ...]


@dataclass
class Asset:
    asset_uid: str
    identity_type: str
    value: str
    role: str
    mechanism: str
    seller_occurrences: dict[str, int]


class JsonlWriter:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.stream: TextIO = path.open("w", encoding="utf-8", newline="\n")
        self.rows = 0

    def write(self, row: Mapping[str, Any]) -> None:
        self.stream.write(json.dumps(dict(row), ensure_ascii=False, separators=(",", ":")))
        self.stream.write("\n")
        self.rows += 1

    def close(self) -> None:
        self.stream.close()


class CsvWriter:
    def __init__(self, path: Path, fields: Sequence[str]):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.fields = tuple(fields)
        self.stream: TextIO = path.open("w", encoding="utf-8", newline="")
        self.writer = csv.DictWriter(self.stream, fieldnames=self.fields, lineterminator="\n")
        self.writer.writeheader()
        self.rows = 0

    def write(self, row: Mapping[str, Any]) -> None:
        if tuple(row) != self.fields:
            raise MethodRootBuildError(f"CSV schema drift: {self.path.name}")
        self.writer.writerow(dict(row))
        self.rows += 1

    def close(self) -> None:
        self.stream.close()


@dataclass
class SplitWriters:
    worlds: JsonlWriter
    sellers: JsonlWriter
    items: JsonlWriter
    replay_items: JsonlWriter
    redacted_items: JsonlWriter
    profiles: JsonlWriter
    endpoints: CsvWriter
    identity33: CsvWriter
    membership: JsonlWriter
    labels: CsvWriter
    qrels: JsonlWriter
    identity_plan: JsonlWriter
    parser_rows: JsonlWriter
    generation_audit: JsonlWriter

    def close(self) -> None:
        for value in self.__dict__.values():
            value.close()


def load_formal_authorities(policy: Mapping[str, Any]) -> Authorities:
    auth_path = ROOT / str(policy["formal_authorization_path"])
    if not auth_path.is_file():
        raise MethodRootBuildError("Formal method-root authorization is absent")
    auth = read_json(auth_path)
    required = {
        "version", "status", "implementation_commit", "policy_sha256",
        "prebuild_result_sha256", "output_root", "private_root", "key_files",
        "time_key", "canonical_self_hash",
    }
    if set(auth) != required or auth["status"] != "AUTHORIZED_ONCE_NOT_CONSUMED":
        raise MethodRootBuildError("Formal method-root authorization schema/status drift")
    payload = dict(auth)
    claimed = payload.pop("canonical_self_hash")
    if claimed != canonical_sha256(payload):
        raise MethodRootBuildError("Formal authorization self-hash drift")
    if auth["implementation_commit"] != git_head():
        raise MethodRootBuildError("Formal authorization implementation commit drift")
    if auth["policy_sha256"] != sha256_file(POLICY_PATH):
        raise MethodRootBuildError("Formal authorization policy hash drift")
    if auth["prebuild_result_sha256"] != sha256_file(PREBUILD_RESULT_PATH):
        raise MethodRootBuildError("Formal authorization prebuild-result hash drift")
    if (
        auth["output_root"] != policy["formal_output_root"]
        or auth["private_root"] != policy["formal_private_root"]
    ):
        raise MethodRootBuildError("Formal authorization output binding drift")
    status = subprocess.run(
        ["git", "status", "--porcelain=v1", "--untracked-files=all"],
        cwd=ROOT, check=True, capture_output=True, text=True, encoding="utf-8",
    ).stdout.splitlines()
    expected_untracked = f"?? {auth_path.relative_to(ROOT).as_posix()}"
    if status != [expected_untracked]:
        raise MethodRootBuildError(
            "Formal build requires the frozen commit plus only its authorization file"
        )
    keys: dict[str, bytes] = {}
    expected_names = {"text", "identity", "style", "audit_schedule", "uid"}
    if set(auth["key_files"]) != expected_names:
        raise MethodRootBuildError("Formal random-authority keyset drift")
    commitments: set[str] = set()
    for name in sorted(expected_names):
        spec = auth["key_files"][name]
        path = ROOT / str(spec["path"])
        data = path.read_bytes() if path.is_file() else b""
        commitment = hashlib.sha256(data).hexdigest()
        if len(data) != 32 or commitment != spec["commitment_sha256"]:
            raise MethodRootBuildError(f"Formal authority drift: {name}")
        if commitment in commitments:
            raise MethodRootBuildError("Formal authorities are not independent")
        commitments.add(commitment)
        keys[name] = data
    time_spec = auth["time_key"]
    time_data = TIME_KEY_PATH.read_bytes() if TIME_KEY_PATH.is_file() else b""
    if (
        str(time_spec["path"]) != TIME_KEY_PATH.relative_to(ROOT).as_posix()
        or len(time_data) != 32
        or hashlib.sha256(time_data).hexdigest() != time_spec["commitment_sha256"]
        or time_spec["commitment_sha256"]
        != "b99fe117617313ec2cda0228d8d40d56ccea8f63891425fe5b2332dc5b338c82"
        or time_spec["commitment_sha256"] in commitments
    ):
        raise MethodRootBuildError("Retained V9.4 time authority drift")
    return Authorities(time=time_data, **keys)


def consume_formal_authorization(policy: Mapping[str, Any]) -> dict[str, Any]:
    auth_path = ROOT / str(policy["formal_authorization_path"])
    auth = read_json(auth_path)
    text_key_path = ROOT / str(auth["key_files"]["text"]["path"])
    marker_path = text_key_path.parent / "method_root_build.consumed.json"
    payload = {
        "version": "2026-08-28-step28-v13-v1-13-v9-4-method-root-consumption-v1",
        "status": "METHOD_ROOT_BUILD_AUTHORITY_CONSUMED",
        "authorization_sha256": sha256_file(auth_path),
        "authorization_canonical_self_hash": auth["canonical_self_hash"],
        "implementation_commit": auth["implementation_commit"],
        "output_root": auth["output_root"],
        "private_root": auth["private_root"],
        "rerun_authorized": False,
    }
    payload["canonical_self_hash"] = canonical_sha256(payload)
    marker_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with marker_path.open("x", encoding="utf-8", newline="\n") as stream:
            json.dump(payload, stream, ensure_ascii=False, indent=2, sort_keys=True)
            stream.write("\n")
    except FileExistsError as exc:
        raise MethodRootBuildError("Formal method-root authority was already consumed") from exc
    return {
        "path": marker_path.relative_to(ROOT).as_posix(),
        "sha256": sha256_file(marker_path),
        "canonical_self_hash": payload["canonical_self_hash"],
    }


def smoke_authorities() -> Authorities:
    def key(name: str) -> bytes:
        return hashlib.sha256(f"{VERSION}::smoke::{name}".encode("ascii")).digest()
    return Authorities(
        text=key("text"), identity=key("identity"), style=key("style"),
        audit_schedule=key("audit_schedule"), uid=key("uid"), time=key("time"),
    )


def _audit_world(split: str, ordinal: int, key: bytes) -> PublicWorld:
    world_uid = f"v9_4_{split}_world_{ordinal:03d}"
    sellers = tuple(f"{world_uid}_seller_{index:02d}" for index in range(28))
    seller_order = ranked(sellers, key, "audit-groups", split, ordinal)
    sizes = (3, 3, 3, 3, 2, 2, 2, 2, 2, 2, 2, 2)
    groups: list[tuple[str, ...]] = []
    cursor = 0
    for size in sizes:
        groups.append(tuple(sorted(seller_order[cursor:cursor + size], key=lambda x: x.encode("utf-8"))))
        cursor += size
    noise_rank = ranked([str(i) for i in range(28)], key, "audit-noise", split, ordinal)
    noise_slots = tuple(int(value) for value in noise_rank)
    return PublicWorld(split, ordinal, world_uid, sellers, noise_slots, tuple(groups))


def _smoke_world(split: str) -> PublicWorld:
    return _audit_world(split, 0, hashlib.sha256(f"smoke::{split}".encode()).digest())


def build_world_schedules(formal: bool, auth: Authorities) -> dict[str, tuple[PublicWorld, ...]]:
    if not formal:
        return {split: (_smoke_world(split),) for split in SPLITS}
    output: dict[str, tuple[PublicWorld, ...]] = {}
    for split in ("train", "development"):
        schedule = schedule_v94.build_split_schedule(split)
        expected = read_json(POLICY_PATH)["model_visible_noise"][
            f"{split}_schedule_commitment_sha256"
        ]
        if schedule.commitment["split_schedule_commitment_sha256"] != expected:
            raise MethodRootBuildError(f"{split} V9.4 schedule commitment drift")
        output[split] = tuple(
            PublicWorld(
                split=split,
                ordinal=int(public["world_ordinal"]),
                world_uid=str(public["world_uid"]),
                seller_uids=tuple(public["seller_uids"]),
                noise_slots=tuple(public["noise_slot_by_seller_slot"]),
                controller_groups=tuple(tuple(group) for group in groups),
            )
            for public, groups in zip(
                schedule.public_worlds,
                schedule.controller_groups_by_world,
                strict=True,
            )
        )
    for split in ("audit_a", "audit_b"):
        output[split] = tuple(
            _audit_world(split, ordinal, auth.audit_schedule)
            for ordinal in range(FORMAL_COUNTS[split])
        )
    return output


def style_for_seller(
    *, world: PublicWorld, seller_uid: str, group_index: int,
    group_style: Mapping[str, Any], template: Mapping[str, Any], key: bytes,
) -> dict[str, Any]:
    style = {field: group_style[field] for field in STYLE_FIELDS}
    selected = ranked(STYLE_FIELDS, key, "seller-style", world.world_uid, seller_uid)[:2]
    domains = template["renderer_contract"]["style_factor_domains"]
    for field in selected:
        domain = list(domains[field])
        style[field] = domain[(domain.index(style[field]) + 1) % len(domain)]
    style["base_style_id"] = str(group_style["style_id"])
    style["perturbed_fields"] = selected
    style["controller_group_index"] = group_index
    return style


def assign_styles(
    world: PublicWorld, template: Mapping[str, Any], key: bytes
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    prototypes = list(template["style_prototypes"])
    selected_ids = ranked(
        [str(row["style_id"]) for row in prototypes], key, "world-styles", world.world_uid
    )[:4]
    by_id = {str(row["style_id"]): row for row in prototypes}
    group_order = ranked([str(i) for i in range(12)], key, "style-groups", world.world_uid)
    group_style: dict[int, Mapping[str, Any]] = {}
    for rank, group_text in enumerate(group_order):
        group_style[int(group_text)] = by_id[selected_ids[rank // 3]]
    styles: dict[str, dict[str, Any]] = {}
    audit: list[dict[str, Any]] = []
    for group_index, group in enumerate(world.controller_groups):
        for seller_uid in group:
            style = style_for_seller(
                world=world, seller_uid=seller_uid, group_index=group_index,
                group_style=group_style[group_index], template=template, key=key,
            )
            styles[seller_uid] = style
            audit.append({"seller_uid": seller_uid, **style})
    if len(styles) != 28:
        raise MethodRootBuildError("Seller author-style universe drift")
    controller_style_counts = Counter(
        str(group_style[index]["style_id"]) for index in range(12)
    )
    if sorted(controller_style_counts.values()) != [3, 3, 3, 3]:
        raise MethodRootBuildError("Base author styles are not shared by three controllers")
    return styles, audit


def perturb_style_text(text: str, style: Mapping[str, Any], template: Mapping[str, Any]) -> str:
    if bool(style["traditional_variant"]):
        table = str.maketrans(template["renderer_contract"]["traditional_substitutions"])
        text = text.translate(table)
    return unicodedata.normalize("NFC", text)


def render_base_item(
    *, world: PublicWorld, seller_uid: str, noise_slot: int, ordinal: int,
    title_nonempty: bool, description_nonempty: bool, style: Mapping[str, Any],
    template: Mapping[str, Any], key: bytes,
) -> tuple[str, str, str, dict[str, str]]:
    lexicon = template["generic_lexicon"]
    digest = hmac_digest(key, "content", world.split, world.ordinal, noise_slot, ordinal)
    category = lexicon["categories"][digest[0] % len(lexicon["categories"])]
    products = lexicon["category_products"][category]
    product = products[digest[1] % len(products)]
    attribute = lexicon["attributes"][digest[2] % len(lexicon["attributes"])]
    modifier = lexicon["title_modifiers"][digest[3] % len(lexicon["title_modifiers"])]
    delivery = lexicon["delivery"][digest[4] % len(lexicon["delivery"])]
    service = lexicon["service"][digest[5] % len(lexicon["service"])]
    title = ""
    if title_nonempty:
        title = TITLE_PATTERNS[digest[6] % len(TITLE_PATTERNS)].format(
            product=product, attribute=attribute, modifier=modifier
        )
        tag = str(style["english_tag"])
        if tag and digest[7] % 16 < 3:
            title += " " + tag
        detail_digest = hmac_digest(
            key, "title-details", world.split, world.ordinal, noise_slot, ordinal
        )
        title += " " + " ".join(
            pool[detail_digest[index] % len(pool)]
            for index, pool in enumerate(TITLE_DETAIL_POOLS)
        )
        title = perturb_style_text(title, style, template)
    description = ""
    if description_nonempty:
        ending = str(style["ending"])
        if bool(style["repeat_punctuation"]):
            ending *= 2
        separator = str(style["separator"])
        if style["line_mode"] == "double":
            separator += "\n"
        elif style["line_mode"] == "bullet":
            separator = "\n• "
        description = DESCRIPTION_PATTERNS[digest[8] % len(DESCRIPTION_PATTERNS)].format(
            product=product, attribute=attribute, delivery=delivery,
            service=service, separator=separator, ending=ending,
        )
        if style["line_mode"] == "bullet":
            description = "• " + description
        detail_digest = hmac_digest(
            key, "description-details", world.split, world.ordinal, noise_slot, ordinal
        )
        description += separator.join(
            pool[detail_digest[index] % len(pool)] + ending
            for index, pool in enumerate(DESCRIPTION_DETAIL_POOLS)
        )
        description = perturb_style_text(description, style, template)
    return title, description, category, {
        "product": product, "attribute": attribute, "modifier": modifier,
        "delivery": delivery, "service": service,
    }


def assign_markets(world: PublicWorld) -> dict[str, str]:
    return {
        seller_uid: MARKETS[world.noise_slots[index] % len(MARKETS)]
        for index, seller_uid in enumerate(world.seller_uids)
    }


def assign_mechanisms(
    world: PublicWorld, markets: Mapping[str, str], key: bytes
) -> dict[int, str]:
    graph = "G_B" if world.split == "audit_b" else "G_A"
    assigned: dict[int, str] = {}
    for size in (3, 2):
        indexes = [i for i, group in enumerate(world.controller_groups) if len(group) == size]
        slots = list(MECHANISM_SLOTS[graph][size])
        if slots.count("cross_market_stable_reuse"):
            eligible = [
                index for index in indexes
                if len({markets[seller] for seller in world.controller_groups[index]}) > 1
            ]
            if not eligible:
                raise MethodRootBuildError("No cross-market controller is available")
            chosen = int(ranked([str(i) for i in eligible], key, "cross-market", world.world_uid, size)[0])
            assigned[chosen] = "cross_market_stable_reuse"
            indexes.remove(chosen)
            slots.remove("cross_market_stable_reuse")
        ordered_indexes = [int(value) for value in ranked([str(i) for i in indexes], key, "mechanism-groups", world.world_uid, size)]
        ordered_slots = ranked(
            [f"{slot}#{ordinal}" for ordinal, slot in enumerate(slots)],
            key, "mechanism-slots", world.world_uid, size,
        )
        for index, slot in zip(ordered_indexes, ordered_slots, strict=True):
            assigned[index] = slot.rsplit("#", 1)[0]
    expected = Counter(MECHANISM_SLOTS[graph][3] + MECHANISM_SLOTS[graph][2])
    if len(assigned) != 12 or Counter(assigned.values()) != expected:
        raise MethodRootBuildError("Mechanism assignment cardinality drift")
    return assigned


def identity_value_for_asset(key: bytes, identity_type: str, asset_uid: str) -> str:
    modulus = identity_values.domain_size(identity_type, "parser_safe_hex_v2")
    integer = int.from_bytes(hmac_digest(key, "identity-value", identity_type, asset_uid), "big") % modulus
    return identity_values.encode_identity_value(
        identity_type, integer, handle_encoding="parser_safe_hex_v2"
    )


def plan_assets(
    world: PublicWorld, mechanisms: Mapping[int, str], markets: Mapping[str, str], key: bytes,
) -> tuple[list[Asset], list[dict[str, Any]], tuple[tuple[str, str], ...]]:
    assets: list[Asset] = []
    used_values: set[tuple[str, str]] = set()
    zero_sellers = {
        seller for index, group in enumerate(world.controller_groups)
        if mechanisms[index] == "zero_visible_identity_history" for seller in group
    }

    def add_asset(
        *, identity_type: str, sellers: Sequence[str], role: str,
        mechanism: str, repeats: int | Mapping[str, int] = 1,
    ) -> Asset:
        serial = len(assets)
        asset_uid = "asset_" + hashlib.sha256(
            hmac_digest(key, "asset", world.world_uid, serial, identity_type, mechanism)
        ).hexdigest()
        value = identity_value_for_asset(key, identity_type, asset_uid)
        if (identity_type, value) in used_values:
            raise MethodRootBuildError("Identity value collision")
        used_values.add((identity_type, value))
        occurrence_map = {
            seller: int(repeats[seller] if isinstance(repeats, Mapping) else repeats)
            for seller in sellers
        }
        asset = Asset(asset_uid, identity_type, value, role, mechanism, occurrence_map)
        assets.append(asset)
        return asset

    def choose_types(label: str, count: int, allowed: Sequence[str] = DIRECT_TYPES) -> list[str]:
        return ranked(allowed, key, "asset-types", world.world_uid, label, len(assets))[:count]

    for seller_uid in world.seller_uids:
        types = choose_types(f"background::{seller_uid}", 2)
        for identity_type in types:
            add_asset(
                identity_type=identity_type, sellers=[seller_uid],
                role="direct_or_private", mechanism="seller_private_background",
                repeats=1,
            )

    mechanism_audit: list[dict[str, Any]] = []
    for group_index, group in enumerate(world.controller_groups):
        mechanism = mechanisms[group_index]
        ordered = ranked(group, key, "mechanism-members", world.world_uid, group_index)
        created: list[str] = []
        if mechanism == "single_identity_stable_reuse":
            repeat = 2 if hmac_digest(key, "repeat", world.world_uid, group_index)[0] % 2 else 1
            created.append(add_asset(
                identity_type=choose_types(mechanism, 1)[0], sellers=ordered,
                role="direct_or_private", mechanism=mechanism, repeats=repeat,
            ).asset_uid)
        elif mechanism == "multi_type_identity_reuse":
            for identity_type in choose_types(mechanism, 2):
                repeat = 2 if hmac_digest(key, "repeat", world.world_uid, group_index, identity_type)[0] % 2 else 1
                created.append(add_asset(
                    identity_type=identity_type, sellers=ordered,
                    role="direct_or_private", mechanism=mechanism, repeats=repeat,
                ).asset_uid)
        elif mechanism == "cross_market_stable_reuse":
            pairs = [pair for pair in combinations(ordered, 2) if markets[pair[0]] != markets[pair[1]]]
            pair = sorted(pairs, key=lambda value: (hmac_digest(key, "cross-pair", world.world_uid, *value), value))[0]
            repeat = 2 if hmac_digest(key, "repeat", world.world_uid, group_index)[0] % 2 else 1
            created.append(add_asset(
                identity_type=choose_types(mechanism, 1)[0], sellers=pair,
                role="direct_or_private", mechanism=mechanism, repeats=repeat,
            ).asset_uid)
        elif mechanism == "single_hop_rotation":
            types = choose_types(mechanism, 2)
            for identity_type, endpoints in zip(types, (ordered[:2], ordered[1:]), strict=True):
                created.append(add_asset(
                    identity_type=identity_type, sellers=endpoints,
                    role="direct_or_private", mechanism=mechanism, repeats=1,
                ).asset_uid)
        elif mechanism == "corroborated_two_hop_rotation":
            types = choose_types(mechanism, 4)
            for identity_type, endpoints in zip(types, (ordered[:2], ordered[:2], ordered[1:], ordered[1:]), strict=True):
                created.append(add_asset(
                    identity_type=identity_type, sellers=endpoints,
                    role="direct_or_private", mechanism=mechanism, repeats=1,
                ).asset_uid)
        elif mechanism == "sparse_history":
            created.append(add_asset(
                identity_type=choose_types(mechanism, 1)[0], sellers=ordered,
                role="direct_or_private", mechanism=mechanism, repeats=1,
            ).asset_uid)
        elif mechanism not in {"same_controller_no_direct_share", "zero_visible_identity_history"}:
            raise MethodRootBuildError(f"Unknown mechanism: {mechanism}")
        mechanism_audit.append({
            "controller_group_index": group_index,
            "members": list(group),
            "mechanism": mechanism,
            "identity_asset_uids": created,
        })

    seller_controller = {
        seller: index for index, group in enumerate(world.controller_groups) for seller in group
    }
    eligible = [seller for seller in world.seller_uids if seller not in zero_sellers]
    negative_pairs = [
        pair for pair in combinations(eligible, 2)
        if seller_controller[pair[0]] != seller_controller[pair[1]]
    ]
    ordered_pairs = sorted(
        negative_pairs,
        key=lambda pair: (hmac_digest(key, "hard-negative-pairs", world.world_uid, *pair), pair),
    )
    selected: list[tuple[str, str]] = []
    used_sellers: set[str] = set()
    for pair in ordered_pairs:
        if not used_sellers.intersection(pair):
            selected.append(pair)
            used_sellers.update(pair)
        if len(selected) == 6:
            break
    if len(selected) != 6:
        raise MethodRootBuildError("Hard-negative endpoint selection failed")
    hub_sellers = [pair[0] for pair in selected[:4]]
    add_asset(
        identity_type=choose_types("high-frequency", 1)[0], sellers=hub_sellers,
        role="high_frequency_direct", mechanism="negative_high_frequency_hub",
    )
    for pair in selected[:2]:
        add_asset(
            identity_type=choose_types("risky", 1, IDENTITY_TYPES)[0], sellers=pair,
            role="risky_product", mechanism="negative_risky_shared_token",
        )
    for pair in selected[2:4]:
        add_asset(
            identity_type="external_url", sellers=pair, role="public_support",
            mechanism="negative_public_support_token",
        )
    return assets, mechanism_audit, tuple(selected)


def allocate_assets_to_items(
    *, world: PublicWorld, assets: Sequence[Asset], item_rows: Sequence[Mapping[str, Any]],
    key: bytes,
) -> tuple[dict[str, list[Asset]], list[dict[str, Any]]]:
    description_items: dict[str, list[str]] = defaultdict(list)
    for row in item_rows:
        if row["description"]:
            description_items[str(row["seller_uid"])].append(str(row["item_uid"]))
    for values in description_items.values():
        values.sort(key=lambda value: value.encode("utf-8"))
    item_assets: dict[str, list[Asset]] = defaultdict(list)
    occurrences: list[dict[str, Any]] = []
    per_seller_cursor: Counter[str] = Counter()
    for asset in assets:
        for seller_uid in sorted(asset.seller_occurrences, key=lambda value: value.encode("utf-8")):
            count = asset.seller_occurrences[seller_uid]
            candidates = ranked(
                description_items[seller_uid], key, "asset-items", world.world_uid,
                asset.asset_uid, seller_uid,
            )
            if len(candidates) < count:
                raise MethodRootBuildError("Identity repeat lacks distinct description items")
            for occurrence_ordinal, item_uid in enumerate(candidates[:count]):
                item_assets[item_uid].append(asset)
                occurrences.append({
                    "asset_uid": asset.asset_uid,
                    "seller_uid": seller_uid,
                    "item_uid": item_uid,
                    "identity_type": asset.identity_type,
                    "normalized_value": asset.value,
                    "role": asset.role,
                    "mechanism": asset.mechanism,
                    "occurrence_ordinal": occurrence_ordinal,
                })
                per_seller_cursor[seller_uid] += 1
    for values in item_assets.values():
        values.sort(key=lambda asset: asset.asset_uid.encode("utf-8"))
    return dict(item_assets), occurrences


def build_identity33(
    *, base_policy: Mapping[str, Any], endpoints: Sequence[Mapping[str, str]],
    history_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, str]]:
    names = list(base_policy["history_features"]["feature_names"])
    excluded = set(base_policy["history_features"]["excluded_history_feature_names"])
    if len(names) != 33:
        raise MethodRootBuildError("Identity33 feature contract drift")
    by_seller, token_df = history.build_signal_index([dict(row) for row in history_rows])
    graph = history.build_identity_graph(
        by_seller,
        token_df,
        {"generation": {
            "direct_token_seller_frequency_maximum": int(base_policy["history_features"]["direct_token_seller_frequency_maximum"]),
            "weak_graph_token_seller_frequency_maximum": int(base_policy["history_features"]["weak_graph_token_seller_frequency_maximum"]),
        }},
    )
    output: list[dict[str, str]] = []
    for endpoint in endpoints:
        features, _ = history.history_feature_details(
            endpoint["seller_uid_left"], endpoint["seller_uid_right"],
            by_seller, token_df, graph,
            {"generation": {
                "direct_token_seller_frequency_maximum": int(base_policy["history_features"]["direct_token_seller_frequency_maximum"]),
                "weak_graph_token_seller_frequency_maximum": int(base_policy["history_features"]["weak_graph_token_seller_frequency_maximum"]),
            }},
        )
        if set(features) != set(names) | excluded:
            raise MethodRootBuildError("Frozen history helper feature keyset drift")
        output.append({
            "canonical_pair_uid": endpoint["canonical_pair_uid"],
            "world_uid": endpoint["world_uid"],
            **{name: f"{float(features[name]):.12f}" for name in names},
        })
    return output


def _pair_rows(world: PublicWorld) -> list[dict[str, str]]:
    return [
        {
            "canonical_pair_uid": f"{left}||{right}",
            "world_uid": world.world_uid,
            "seller_uid_left": left,
            "seller_uid_right": right,
        }
        for left, right in combinations(world.seller_uids, 2)
    ]


def production_parse_with_uid_aliases(
    *, world: PublicWorld, sellers: Sequence[Mapping[str, Any]],
    items: Sequence[Mapping[str, Any]], base_policy: Mapping[str, Any], key: bytes,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Run the frozen parser on an isomorphic UID namespace.

    V9.4's registered public coordinates intentionally use readable schedule
    identifiers, while the older frozen parser validates w_/sel_/itm_ hashes.
    Only metadata identifiers are aliased; text, markets, item identifiers and
    item order are byte-identical.  The history projection is mapped back after
    the production parser has completed.
    """

    alias_world = "w_" + hashlib.sha256(
        hmac_digest(key, "parser-world-alias", world.world_uid)
    ).hexdigest()
    seller_alias = {
        seller_uid: "sel_" + hashlib.sha256(
            hmac_digest(key, "parser-seller-alias", world.world_uid, seller_uid)
        ).hexdigest()
        for seller_uid in world.seller_uids
    }
    reverse = {value: name for name, value in seller_alias.items()}
    if len(reverse) != 28:
        raise MethodRootBuildError("Parser seller alias collision")
    alias_sellers = [
        {
            "world_uid": alias_world,
            "seller_uid": seller_alias[str(row["seller_uid"])],
            "market": row["market"],
        }
        for row in sellers
    ]
    alias_items = [
        {
            **dict(row),
            "world_uid": alias_world,
            "seller_uid": seller_alias[str(row["seller_uid"])],
        }
        for row in items
    ]
    parsed_alias = production.parse_observed_world(
        base_policy, mode="formal", split=world.split,
        sellers=alias_sellers, items=alias_items,
    )
    history_alias = production.project_history_safe_occurrences(
        base_policy, mode="formal", split=world.split,
        sellers=alias_sellers, items=alias_items, parsed_rows=parsed_alias,
    )
    mapped_history = [
        {
            **dict(row),
            "world_uid": world.world_uid,
            "seller_uid": reverse[str(row["seller_uid"])],
        }
        for row in history_alias
    ]
    alias_audit = {
        "version": "2026-08-28-step28-v13-v1-13-v9-4-parser-uid-alias-v1",
        "world_uid": world.world_uid,
        "alias_world_uid_sha256": hashlib.sha256(alias_world.encode()).hexdigest(),
        "seller_alias_mapping_sha256": canonical_sha256(seller_alias),
        "seller_count": len(seller_alias),
        "item_count": len(items),
        "text_bytes_changed": False,
        "market_bytes_changed": False,
        "item_uid_bytes_changed": False,
        "production_parser_row_count": len(parsed_alias),
    }
    return parsed_alias, mapped_history, alias_audit


def build_one_world(
    *, world: PublicWorld, auth: Authorities, base_policy: Mapping[str, Any],
    template: Mapping[str, Any], signatures: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    styles, style_audit = assign_styles(world, template, auth.style)
    markets = assign_markets(world)
    mechanisms = assign_mechanisms(world, markets, auth.identity)
    assets, mechanism_audit, negative_pairs = plan_assets(
        world, mechanisms, markets, auth.identity
    )
    signature_by_slot = {int(row["noise_slot"]): row for row in signatures}
    base_items: list[dict[str, Any]] = []
    replay_items: list[dict[str, Any]] = []
    for seller_index, seller_uid in enumerate(world.seller_uids):
        noise_slot = world.noise_slots[seller_index]
        signature = signature_by_slot[noise_slot]
        for ordinal in range(int(signature["item_count"])):
            item_uid = "itm_" + hashlib.sha256(
                hmac_digest(auth.uid, "item", world.world_uid, seller_uid, ordinal)
            ).hexdigest()
            title, description, category, _components = render_base_item(
                world=world, seller_uid=seller_uid, noise_slot=noise_slot,
                ordinal=ordinal,
                title_nonempty=signature["title_present_mask"][ordinal] == "1",
                description_nonempty=signature["description_present_mask"][ordinal] == "1",
                style=styles[seller_uid], template=template, key=auth.text,
            )
            time_bucket = prebuild_v94._time_bucket(
                time_key_hex=auth.time.hex(), split=world.split,
                world_ordinal=world.ordinal, noise_slot=noise_slot,
                logical_item_ordinal=ordinal,
            )
            base_items.append({
                "world_uid": world.world_uid, "seller_uid": seller_uid,
                "item_uid": item_uid, "time_bucket": time_bucket,
                "category": category, "title": title, "description": description,
            })
            replay_items.append({
                "world_uid": world.world_uid, "seller_uid": seller_uid,
                "item_uid": item_uid, "logical_item_ordinal": ordinal,
                "title": title, "description": description,
                "time_bucket": time_bucket,
            })
    by_seller_items: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in base_items:
        by_seller_items[str(row["seller_uid"])].append(row)
    registered_negative_controls: list[dict[str, Any]] = []
    for control_index, pair in enumerate(negative_pairs):
        left_items = [row for row in by_seller_items[pair[0]] if row["title"]]
        right_items = [row for row in by_seller_items[pair[1]] if row["title"]]
        if not left_items or not right_items:
            raise MethodRootBuildError("Registered text hard negative lacks a title")
        source = sorted(left_items, key=lambda row: row["item_uid"].encode("utf-8"))[0]
        target = sorted(right_items, key=lambda row: row["item_uid"].encode("utf-8"))[0]
        if control_index < 2:
            target["title"] = source["title"]
            control_type = "exact_title_clone_negative"
        else:
            target["title"] = source["title"] + " 配置另选"
            control_type = "high_semantic_similarity_negative"
        for replay_row in replay_items:
            if replay_row["item_uid"] == target["item_uid"]:
                replay_row["title"] = target["title"]
                break
        registered_negative_controls.append({
            "canonical_pair_uid": f"{pair[0]}||{pair[1]}",
            "control_type": control_type,
            "source_item_uid": source["item_uid"],
            "target_item_uid": target["item_uid"],
        })
    item_assets, planned_occurrences = allocate_assets_to_items(
        world=world, assets=assets, item_rows=base_items, key=auth.identity
    )
    raw_items: list[dict[str, Any]] = []
    redacted_items: list[dict[str, Any]] = []
    replay_by_uid = {row["item_uid"]: row for row in replay_items}
    for base in base_items:
        item_uid = str(base["item_uid"])
        clauses = [
            text_renderer.identity_clause(
                template_family=base_policy["identity_design"]["role_to_template_family"][asset.role],
                identity_type=asset.identity_type,
                normalized_value=asset.value,
                template=template,
            )
            for asset in item_assets.get(item_uid, [])
        ]
        description = text_renderer.render_description(
            base_description=str(base["description"]), noise_clause="",
            identity_clauses=clauses, selector_uid=item_uid, template=template,
        )
        raw = {**base, "description": description}
        raw_items.append(raw)
        redacted_items.append({
            "world_uid": base["world_uid"], "seller_uid": base["seller_uid"],
            "item_uid": item_uid, "title": base["title"],
            "description": base["description"],
        })
        replay_by_uid[item_uid]["description"] = description
    sellers = [
        {"world_uid": world.world_uid, "seller_uid": seller, "market": markets[seller]}
        for seller in world.seller_uids
    ]
    parsed, history_rows, parser_alias_audit = production_parse_with_uid_aliases(
        world=world, sellers=sellers, items=raw_items,
        base_policy=base_policy, key=auth.uid,
    )
    planned_keys = Counter(
        (row["seller_uid"], row["item_uid"], row["identity_type"], row["normalized_value"])
        for row in planned_occurrences
    )
    alias_to_public = {
        "sel_" + hashlib.sha256(
            hmac_digest(auth.uid, "parser-seller-alias", world.world_uid, seller_uid)
        ).hexdigest(): seller_uid
        for seller_uid in world.seller_uids
    }
    parsed_keys = Counter(
        (alias_to_public[row["seller_uid"]], row["item_uid"], row["contact_type"], row["normalized_value"])
        for row in parsed
    )
    if planned_keys != parsed_keys or any(row["source_field"] != "description" for row in parsed):
        raise MethodRootBuildError("Production parser does not exactly replay the private identity plan")
    safe_items = [
        {**row, "title": redacted_items[index]["title"],
         "description": redacted_items[index]["description"]}
        for index, row in enumerate(raw_items)
    ]
    profiles, profile_audit = profiles_module.build_world_profiles(
        base_policy, mode="formal", split=world.split,
        sellers=sellers, items=safe_items,
    )
    model_profiles = list(scientific_common.project_model_seller_profiles(profiles))
    endpoints = _pair_rows(world)
    identity33 = build_identity33(
        base_policy=base_policy, endpoints=endpoints, history_rows=history_rows
    )
    if world.split in {"train", "development"}:
        public_schedule = {
            "split": world.split, "world_ordinal": world.ordinal,
            "world_uid": world.world_uid, "seller_uids": list(world.seller_uids),
            "noise_slot_by_seller_slot": list(world.noise_slots),
        }
        registered_endpoints, proxy_items = prebuild_v94.build_truth_free_world_source(
            world=public_schedule, noise_signatures=signatures,
            time_key_hex=auth.time.hex(),
        )
        registered_rows = prebuild_v94.build_truth_free_world_projection(
            world=public_schedule, noise_signatures=signatures,
            time_key_hex=auth.time.hex(),
        )
        replay_endpoints = [
            {
                "world_uid": row["world_uid"],
                "canonical_pair_uid": row["canonical_pair_uid"],
                "seller_uid_left": row["seller_uid_left"],
                "seller_uid_right": row["seller_uid_right"],
            }
            for row in endpoints
        ]
        replay_v94.require_exact_replay(
            registered_rows=registered_rows,
            public_endpoint_rows=replay_endpoints,
            public_item_rows=[replay_by_uid[row["item_uid"]] for row in replay_items],
        )
        if len(proxy_items) != len(raw_items):
            raise MethodRootBuildError("V9.4 actual/proxy item count drift")
    forbidden_hits = [
        (row["item_uid"], field)
        for row in raw_items for field in ("title", "description")
        if ARTIFICIAL_CODE_RE.search(str(row[field])) or FORBIDDEN_VISIBLE_RE.search(str(row[field]))
    ]
    if forbidden_hits:
        raise MethodRootBuildError(f"Forbidden visible marker: {forbidden_hits[0]}")
    seller_controller = {
        seller: group_index
        for group_index, group in enumerate(world.controller_groups)
        for seller in group
    }
    labels: list[dict[str, Any]] = []
    positive = 0
    for endpoint in endpoints:
        label = int(
            seller_controller[endpoint["seller_uid_left"]]
            == seller_controller[endpoint["seller_uid_right"]]
        )
        positive += label
        labels.append({
            "canonical_pair_uid": endpoint["canonical_pair_uid"],
            "world_uid": world.world_uid, "label": label,
        })
    if len(labels) != 378 or positive != 20:
        raise MethodRootBuildError("World pair truth cardinality drift")
    membership = [
        {"world_uid": world.world_uid, "controller_group_index": index, "seller_uid": seller}
        for index, group in enumerate(world.controller_groups) for seller in group
    ]
    qrels = []
    for seller in world.seller_uids:
        relevant = sorted(
            [other for other in world.controller_groups[seller_controller[seller]] if other != seller],
            key=lambda value: value.encode("utf-8"),
        )
        qrels.append({
            "world_uid": world.world_uid, "query_seller_uid": seller,
            "relevant_seller_uids": relevant,
        })
    active_rows = sum(
        any(float(row[name]) != 0.0 for name in base_policy["history_features"]["feature_names"])
        for row in identity33
    )
    if active_rows == 0:
        raise MethodRootBuildError("Identity33 positive control is degenerate")
    return {
        "world": {
            "world_uid": world.world_uid, "split": world.split,
            "world_ordinal": world.ordinal, "seller_count": 28,
            "item_count": len(raw_items), "pair_count": 378,
        },
        "sellers": sellers, "items": raw_items,
        "replay_items": list(replay_by_uid.values()),
        "redacted_items": redacted_items, "model_profiles": model_profiles,
        "endpoints": endpoints, "identity33": identity33,
        "membership": membership, "labels": labels, "qrels": qrels,
        "identity_plan": [{
            "world_uid": world.world_uid, "asset_uid": asset.asset_uid,
            "identity_type": asset.identity_type,
            "value_sha256": hashlib.sha256(asset.value.encode("utf-8")).hexdigest(),
            "role": asset.role, "mechanism": asset.mechanism,
            "seller_occurrences": asset.seller_occurrences,
        } for asset in assets],
        "parser_rows": parsed,
        "generation_audit": {
            "world_uid": world.world_uid,
            "style_assignments": style_audit,
            "mechanism_assignments": mechanism_audit,
            "registered_negative_controls": registered_negative_controls,
            "profile_audit": profile_audit,
            "parser_uid_alias_audit": parser_alias_audit,
            "identity33_active_pair_count": active_rows,
            "public_projection_exact_replay": world.split in {"train", "development"},
            "truth_projected_after_public_rows": True,
        },
    }


def open_writers(
    public_root: Path, private_root: Path, split: str, identity_fields: Sequence[str]
) -> SplitWriters:
    observed = public_root / split / "observed"
    private = private_root / split
    return SplitWriters(
        worlds=JsonlWriter(observed / "worlds.jsonl"),
        sellers=JsonlWriter(observed / "sellers.jsonl"),
        items=JsonlWriter(observed / "items.jsonl"),
        replay_items=JsonlWriter(observed / "model_visible_replay_items.jsonl"),
        redacted_items=JsonlWriter(observed / "redacted_items.jsonl"),
        profiles=JsonlWriter(observed / "model_seller_profiles.jsonl"),
        endpoints=CsvWriter(observed / "complete_model_pair_endpoints.csv", (
            "canonical_pair_uid", "world_uid", "seller_uid_left", "seller_uid_right",
        )),
        identity33=CsvWriter(observed / "identity33_all_pairs.csv", (
            "canonical_pair_uid", "world_uid", *identity_fields,
        )),
        membership=JsonlWriter(private / "controller_membership.jsonl"),
        labels=CsvWriter(private / "pair_labels.csv", (
            "canonical_pair_uid", "world_uid", "label",
        )),
        qrels=JsonlWriter(private / "qrels.jsonl"),
        identity_plan=JsonlWriter(private / "identity_plan.jsonl"),
        parser_rows=JsonlWriter(private / "parsed_identity_occurrences.jsonl"),
        generation_audit=JsonlWriter(private / "generation_audit.jsonl"),
    )


def write_world(writers: SplitWriters, value: Mapping[str, Any]) -> None:
    writers.worlds.write(value["world"])
    for name, writer_name in (
        ("sellers", "sellers"), ("items", "items"),
        ("replay_items", "replay_items"), ("redacted_items", "redacted_items"),
        ("model_profiles", "profiles"), ("endpoints", "endpoints"),
        ("identity33", "identity33"), ("membership", "membership"),
        ("labels", "labels"), ("qrels", "qrels"),
        ("identity_plan", "identity_plan"), ("parser_rows", "parser_rows"),
    ):
        writer = getattr(writers, writer_name)
        for row in value[name]:
            writer.write(row)
    writers.generation_audit.write(value["generation_audit"])


def file_manifest(root: Path) -> list[dict[str, Any]]:
    return [
        {
            "path": path.relative_to(root).as_posix(),
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in sorted(
            (value for value in root.rglob("*") if value.is_file()),
            key=lambda value: value.relative_to(root).as_posix().encode("utf-8"),
        )
    ]


def build_dataset(*, formal: bool, output_root: Path | None = None) -> dict[str, Any]:
    policy = read_json(POLICY_PATH)
    base_policy = read_json(BASE_POLICY_PATH)
    template = read_json(TEMPLATE_PATH)
    prebuild_result = read_json(PREBUILD_RESULT_PATH)
    if (
        prebuild_result.get("status") != "PASSED_PREBUILD_SHORTCUT_GATE"
        or prebuild_result.get("decision", {}).get("method_root_build_eligible") is not True
    ):
        raise MethodRootBuildError("V9.4 prebuild pass boundary is absent")
    auth = load_formal_authorities(policy) if formal else smoke_authorities()
    if formal:
        root = ROOT / str(policy["formal_output_root"])
        private = ROOT / str(policy["formal_private_root"])
    else:
        root = output_root or (
            ROOT / "reports" / "step28_synthetic_chinese_dataset" / "_v9_4_method_root_smoke"
        )
        private = root.parent / f".{root.name}.private"
    temp = root.parent / f".{root.name}.building"
    private_temp = private.parent / f".{private.name}.building"
    if any(path.exists() for path in (root, private, temp, private_temp)):
        raise MethodRootBuildError("Output or temporary method-root path already exists")
    consumption = consume_formal_authorization(policy) if formal else None
    temp.mkdir(parents=True)
    private_temp.mkdir(parents=True)
    writers: dict[str, SplitWriters] = {}
    try:
        signatures_capability = noise_v94.build_noise_signatures()
        signatures = [dict(row) for row in signatures_capability.rows]
        schedules = build_world_schedules(formal, auth)
        identity_fields = list(base_policy["history_features"]["feature_names"])
        for split in SPLITS:
            writers[split] = open_writers(temp, private_temp, split, identity_fields)
        completed = 0
        for split in SPLITS:
            for world in schedules[split]:
                value = build_one_world(
                    world=world, auth=auth, base_policy=base_policy,
                    template=template, signatures=signatures,
                )
                write_world(writers[split], value)
                completed += 1
                if completed % 25 == 0 or completed == sum(
                    len(values) for values in schedules.values()
                ):
                    print(json.dumps({
                        "event": "progress", "worlds_completed": completed,
                        "worlds_total": sum(len(values) for values in schedules.values()),
                        "split": split,
                    }, ensure_ascii=False), flush=True)
        for writer in writers.values():
            writer.close()
        writers.clear()
        public_files = file_manifest(temp)
        private_files = file_manifest(private_temp)
        manifest = {
            "version": VERSION,
            "status": "BUILT_NOT_TRAINING_QUALIFIED",
            "formal": formal,
            "world_counts": {split: len(schedules[split]) for split in SPLITS},
            "seller_count": sum(len(schedules[split]) for split in SPLITS) * 28,
            "pair_count": sum(len(schedules[split]) for split in SPLITS) * 378,
            "positive_pair_count": sum(len(schedules[split]) for split in SPLITS) * 20,
            "negative_pair_count": sum(len(schedules[split]) for split in SPLITS) * 358,
            "policy_sha256": sha256_file(POLICY_PATH),
            "prebuild_result_sha256": sha256_file(PREBUILD_RESULT_PATH),
            "formal_authorization_sha256": (
                sha256_file(ROOT / str(policy["formal_authorization_path"]))
                if formal else None
            ),
            "formal_authorization_consumption": consumption,
            "noise_signature_commitment_sha256": signatures_capability.commitment[
                "signature_set_commitment_sha256"
            ],
            "public_files": public_files,
            "private_file_commitments": private_files,
            "audit_truth_read_counts": {"audit_a": 0, "audit_b": 0},
            "training_authorized": False,
        }
        manifest["canonical_self_hash"] = canonical_sha256(manifest)
        write_json(temp / "root_manifest.json", manifest)
        root.parent.mkdir(parents=True, exist_ok=True)
        private.parent.mkdir(parents=True, exist_ok=True)
        temp.rename(root)
        private_temp.rename(private)
        return manifest
    except BaseException:
        for writer in writers.values():
            try:
                writer.close()
            except Exception:
                pass
        if temp.exists():
            shutil.rmtree(temp)
        if private_temp.exists():
            shutil.rmtree(private_temp)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--smoke", action="store_true")
    group.add_argument("--formal", action="store_true")
    parser.add_argument("--output-root", type=Path)
    args = parser.parse_args()
    if args.formal and args.output_root is not None:
        raise SystemExit("--output-root is smoke-only")
    result = build_dataset(formal=args.formal, output_root=args.output_root)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
