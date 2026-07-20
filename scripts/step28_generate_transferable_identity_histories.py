#!/usr/bin/env python3
"""Generate multi-item latent-controller histories for Step28.

Legacy v4/v5 policies stratified frozen source-score carriers by their original
labels.  The v6 ``label_blind_exact_pairing`` contract deliberately has a
separate code path: it refuses any carrier-label input and assigns the exact
same carrier multiset to synthetic positives and negatives in every split.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter
from pathlib import Path

import numpy as np

import step3_build_seller_profiles as step3
import step28_common as base
import step28_history_common as history


POLICY_PATH = history.POLICY_PATH
SIGNAL_FIELDS = list(step3.ITEM_SIGNAL_FIELDS)


def stringify(row: dict) -> dict:
    return {key: str(row.get(key, "")) for key in SIGNAL_FIELDS}


def carrier_pools(policy: dict) -> dict[str, dict[str, list[dict]]]:
    rows = base.load_csv(policy["inputs"]["real_train_source_carriers"])
    assignment = policy["generation"].get(
        "source_carrier_assignment", "legacy_label_correlated"
    )
    if assignment == "label_blind_exact_pairing":
        if policy["inputs"].get("source_carrier_labels"):
            raise ValueError(
                "Step28 label-blind generation forbids a source-carrier label file"
            )
        forbidden_columns = {
            "review_label", "label", "same_controller", "same_latent_controller"
        }
        present = sorted(forbidden_columns & set(rows[0] if rows else {}))
        if present:
            raise ValueError(
                "Step28 label-blind source carriers contain forbidden label columns: "
                + ",".join(present)
            )
        expected_count = int(policy["generation"]["source_carrier_expected_count"])
        if len(rows) != expected_count or {row.get("split_name") for row in rows} != {"train"}:
            raise ValueError(
                "Step28 label-blind source carriers do not match the frozen train-only contract"
            )
        output = {
            split: {"all": []}
            for split in policy["generation"]["splits"]
        }
        partition_salt = str(policy["generation"]["source_carrier_partition_salt"])
        partition_unit = str(
            policy["generation"].get("source_carrier_partition_unit", "pair_uid")
        )
        component_key_by_seller: dict[str, str] = {}
        if partition_unit == "seller_connected_component":
            parent: dict[str, str] = {}

            def find(seller: str) -> str:
                parent.setdefault(seller, seller)
                if parent[seller] != seller:
                    parent[seller] = find(parent[seller])
                return parent[seller]

            def union(left: str, right: str) -> None:
                left_root, right_root = find(left), find(right)
                if left_root != right_root:
                    parent[max(left_root, right_root)] = min(left_root, right_root)

            for row in rows:
                union(row["seller_uid_left"], row["seller_uid_right"])
            members: dict[str, list[str]] = {}
            for seller in sorted(parent):
                members.setdefault(find(seller), []).append(seller)
            for component_members in members.values():
                component_key = hashlib.sha256(
                    "|".join(component_members).encode("utf-8")
                ).hexdigest()
                for seller in component_members:
                    component_key_by_seller[seller] = component_key
        elif partition_unit != "pair_uid":
            raise ValueError(
                f"unknown Step28 source-carrier partition unit: {partition_unit}"
            )
        for row in rows:
            partition_key = (
                component_key_by_seller[row["seller_uid_left"]]
                if partition_unit == "seller_connected_component"
                else row["pair_uid"]
            )
            bucket = int(
                hashlib.sha256(f"{partition_salt}|{partition_key}".encode()).hexdigest()[:8],
                16,
            ) % 5
            split = (
                "synthetic_audit"
                if bucket == 0
                else "synthetic_development"
                if bucket == 1
                else "synthetic_train"
            )
            output[split]["all"].append(row)
        minimum = int(policy["generation"]["source_carrier_partition_minimum"])
        for split, pools in output.items():
            if len(pools["all"]) < minimum:
                raise ValueError(
                    f"Step28 label-blind source carrier partition too small: "
                    f"{split}:{len(pools['all'])}"
                )
        return output
    if assignment != "legacy_label_correlated":
        raise ValueError(f"unknown Step28 source-carrier assignment: {assignment}")

    label_path = policy["inputs"].get("source_carrier_labels")
    if label_path:
        labels = {
            row["pair_uid"]: row["review_label"]
            for row in base.load_csv(label_path)
            if (
                row.get("split_name") == "train"
                and row.get("usable_for_supervision") == "1"
                and row.get("usable_for_core_transfer") == "1"
            )
        }
        if set(labels) != {row["pair_uid"] for row in rows}:
            raise ValueError("Step28 source-carrier feature/label pair UIDs differ")
        rows = [{**row, "review_label": labels[row["pair_uid"]]} for row in rows]
    expected_counts = Counter(policy["generation"].get(
        "source_carrier_expected_label_counts", {"negative": 344, "positive": 229}
    ))
    if len(rows) != sum(expected_counts.values()) or {row.get("split_name") for row in rows} != {"train"}:
        raise ValueError("Step28 source carriers do not match the frozen train-only contract")
    if Counter(row["review_label"] for row in rows) != expected_counts:
        raise ValueError("Step28 source-carrier label counts changed")
    output: dict[str, dict[str, list[dict]]] = {
        split: {"positive": [], "negative": []}
        for split in policy["generation"]["splits"]
    }
    partition_salt = str(policy["generation"].get("source_carrier_partition_salt", "v4"))
    for row in rows:
        bucket = int(
            hashlib.sha256(f"{partition_salt}|{row['pair_uid']}".encode()).hexdigest()[:8], 16
        ) % 5
        split = (
            "synthetic_audit"
            if bucket == 0
            else "synthetic_development"
            if bucket == 1
            else "synthetic_train"
        )
        output[split][row["review_label"]].append(row)
    for split, labels in output.items():
        for label, selected in labels.items():
            minimum = int(policy["generation"].get("source_carrier_partition_minimum_per_label", 20))
            if len(selected) < minimum:
                raise ValueError(f"Step28/v4 source carrier partition too small: {split}:{label}:{len(selected)}")
    return output


def token_value(
    policy: dict,
    world_uid: str,
    split: str,
    world_index: int,
    token_type: str,
    variant: int,
) -> str:
    namespace = str(policy["generation"].get("synthetic_namespace", "step28-v4"))
    digest = (
        base.opaque_uid("step28-v4-token", split, world_index, token_type, variant)
        if namespace == "step28-v4"
        else base.opaque_uid(f"{namespace}-token", world_uid, token_type, variant)
    )
    prefix = str(policy["generation"].get("synthetic_token_prefix", "s28v4"))
    numeric_namespace = str(
        policy["generation"].get("synthetic_numeric_namespace", "")
    )
    if numeric_namespace and (
        not numeric_namespace.isdigit() or len(numeric_namespace) > 3
    ):
        raise ValueError("synthetic numeric namespace must contain at most three digits")
    if token_type == "telegram":
        return prefix + digest[:10]
    if token_type == "email":
        return prefix + digest[:10] + "@example.invalid"
    if token_type == "bat":
        return prefix + "b" + digest[:10]
    if token_type == "qq":
        if numeric_namespace:
            remaining = 12 - len(numeric_namespace)
            return numeric_namespace + str(
                int(digest[:16], 16) % (10**remaining)
            ).zfill(remaining)
        return str(100000000 + int(digest[:12], 16) % 900000000)
    if token_type == "wechat":
        return prefix + "w" + digest[:10]
    if token_type == "phone":
        if numeric_namespace:
            remaining = 13 - len(numeric_namespace)
            return "86" + numeric_namespace + str(
                int(digest[:16], 16) % (10**remaining)
            ).zfill(remaining)
        return "8613" + str(100000000 + int(digest[:12], 16) % 900000000)
    if token_type == "crypto_wallet":
        return "0x" + (digest * 2)[:40]
    if token_type == "external_url":
        return f"{prefix}{digest[:10]}.example.com/contact"
    raise ValueError(f"unsupported synthetic token type: {token_type}")


def rendered_token(token_type: str, value: str) -> str:
    prefixes = {
        "telegram": "Telegram @",
        "bat": "BAT ID: ",
        "qq": "QQ: ",
        "wechat": "微信: ",
        "phone": "电话: ",
        "crypto_wallet": "收款钱包地址: ",
        "external_url": "https://",
        "email": "",
    }
    if token_type not in prefixes:
        raise ValueError(f"unsupported synthetic rendered token type: {token_type}")
    return prefixes[token_type] + value


def render_occurrence(bank: str, role: str, token_type: str, value: str) -> tuple[str, str, str]:
    rendered = rendered_token(token_type, value)
    templates = {
        "train": {
            "direct": ("数字商品交付", f"本店卖家联系方式为 {rendered}，付款后联系卖家完成交付。"),
            "risk": ("数据库样例", f"商品数据样例中的联系人字段为 {rendered}，仅属于交付数据。"),
            "support": ("公开教程", f"课程讲义收录 {rendered}，仅供课堂演示。"),
            "neutral": ("标准商品说明", "拍下后系统自动发送数字商品，请按页面步骤操作。"),
        },
        "development": {
            "direct": ("订单售后信息", f"购买后请通过 {rendered} 联系本店商家处理订单。"),
            "risk": ("记录字段展示", f"所售数据记录里包含 {rendered}，这是商品内容中的样例字段。"),
            "support": ("帮助文档", f"帮助页面列出 {rendered}，仅作课堂演示。"),
            "neutral": ("订单页面", "请阅读商品详情，自动发货后按说明完成操作。"),
        },
        "audit": {
            "direct": ("交易完成说明", f"完成订单后联系商家 {rendered} 获取后续服务。"),
            "risk": ("资料内容预览", f"商品数据预览含联系卖家字段 {rendered}，该账号来自资料内容。"),
            "support": ("使用指引", f"公共指引以 {rendered} 充当演示值，仅供阅读示范。"),
            "neutral": ("交付须知", "数字内容由系统发放，使用前请先查看随附说明。"),
        },
        "train_v7": {
            "direct": ("交易联络", f"下单后请联系本店卖家 {rendered} 核对交付。"),
            "risk": ("数据字段样本", f"所售数据库样例字段包含 {rendered}，该值属于商品数据。"),
            "support": ("公共说明页", f"公开帮助文档使用 {rendered} 作为课堂演示。"),
            "neutral": ("自动交付商品", "付款后系统自动发送文件，请依照商品说明使用。"),
        },
        "development_v7": {
            "direct": ("订单联系方法", f"购买完成后通过 {rendered} 联系商家处理本次订单。"),
            "risk": ("资料预览字段", f"资料数据预览中的账号字段是 {rendered}，并非本店联系方式。"),
            "support": ("公开操作手册", f"公共操作手册列出示范值 {rendered}，仅供阅读。"),
            "neutral": ("数字内容订单", "订单完成后自动发货，具体步骤见随附文档。"),
        },
        "audit_v7": {
            "direct": ("售后联络信息", f"需要售后时请联系当前卖家 {rendered}。"),
            "risk": ("商品记录示例", f"出售记录中的联系人字段显示 {rendered}，它来自数据内容。"),
            "support": ("公共教程示例", f"公共教程以 {rendered} 作为示例值，不代表卖家身份。"),
            "neutral": ("电子商品须知", "电子商品由页面自动交付，请先阅读压缩包内说明。"),
        },
        "train_v8": {
            "direct": ("商家交付渠道", f"付款后请通过 {rendered} 与当前商家核对交付内容。"),
            "risk": ("数据列预览", f"本商品的数据字段示例含有 {rendered}，该值来自所售资料。"),
            "support": ("公开演示页面", f"公开页面展示 {rendered}，仅作演示。"),
            "neutral": ("文件自动发送", "完成付款后页面自动发送文件，请按压缩包说明操作。"),
        },
        "development_v8": {
            "direct": ("订单商家通道", f"订单完成后使用 {rendered} 联系本店商家确认售后。"),
            "risk": ("记录内容样本", f"待售数据库的记录样本中出现 {rendered}，属于资料字段。"),
            "support": ("演示说明页面", f"说明页面采用 {rendered} 作为演示值。"),
            "neutral": ("电子文件交付", "系统会在订单完成后自动交付电子文件，请查看附件说明。"),
        },
        "audit_v8": {
            "direct": ("当前卖家售后", f"如需处理本次购买，请使用 {rendered} 联络当前卖家。"),
            "risk": ("商品数据片段", f"商品所含数据片段的账号字段为 {rendered}，不属于店铺身份。"),
            "support": ("阅读示范页面", f"示范页面展示 {rendered}，用于阅读示例。"),
            "neutral": ("数字附件说明", "数字附件将在付款后自动发放，解压后可查看完整说明。"),
        },
        "train_v9": {
            "direct": ("店铺订单联络", f"购买后请用 {rendered} 联系该店铺完成交付。"),
            "risk": ("数据库列样本", f"出售的数据库字段片段中含 {rendered}，该值属于商品记录。"),
            "support": ("公开演示网页", f"公开网页以 {rendered} 进行演示。"),
            "neutral": ("在线自动发货", "订单支付完成后自动发放电子文件，使用方法见附件。"),
        },
        "development_v9": {
            "direct": ("本店售后入口", f"本次下单后可用 {rendered} 联络本店处理售后。"),
            "risk": ("待售记录预览", f"待售数据记录的示例字段显示 {rendered}，这是商品内容。"),
            "support": ("虚构示范页面", f"阅读页面列出 {rendered} 作为虚构示范。"),
            "neutral": ("订单附件提示", "电子附件会随订单自动发放，请依照附件内步骤使用。"),
        },
        "audit_v9": {
            "direct": ("卖家订单沟通", f"处理该订单时请通过 {rendered} 联系当前卖家。"),
            "risk": ("所售数据预览", f"所售数据的账号样例包含 {rendered}，该值源自商品内容。"),
            "support": ("阅读参考页面", f"说明页面展示 {rendered} 供读者参考。"),
            "neutral": ("自动交付提醒", "付款完成后系统自动提供下载内容，请阅读文件内提示。"),
        },
        "train_v10": {
            "direct": ("商户订单联系", f"完成购买后请使用 {rendered} 联系商户办理交付。"),
            "risk": ("商品库字段", f"商品数据库的样例字段出现 {rendered}，它来自所售记录。"),
            "support": ("课堂演示网页", f"公开网页展示 {rendered} 供课堂演示。"),
            "neutral": ("电子内容发送", "支付成功后电子内容自动发送，请查阅随附操作说明。"),
        },
        "development_v10": {
            "direct": ("交易售后联络", f"交易完成后请通过 {rendered} 联络本店获得售后。"),
            "risk": ("数据内容节选", f"出售数据的内容节选中含有 {rendered}，这是记录字段。"),
            "support": ("阅读演示网页", f"演示网页采用 {rendered} 供阅读。"),
            "neutral": ("附件自动提供", "订单完成后附件自动提供，具体使用方式写在文件中。"),
        },
        "audit_v10": {
            "direct": ("订单卖家渠道", f"本次订单需要沟通时请用 {rendered} 联系卖家。"),
            "risk": ("出售记录片段", f"出售记录片段里的账号样例是 {rendered}，并非卖家身份。"),
            "support": ("公共示范页面", f"公共页面显示 {rendered} 用于示范。"),
            "neutral": ("下载内容提示", "完成付款即可自动获取下载内容，请按照文件提示操作。"),
        },
        "train_v11": {
            "direct": ("成交商家入口", f"成交后请用 {rendered} 联系本店完成后续交付。"),
            "risk": ("销售库样本", f"销售数据库的样例字段中出现 {rendered}，它属于商品数据。"),
            "support": ("公开教学页面", f"公开页面用 {rendered} 进行教学演示。"),
            "neutral": ("自动发送说明", "付款成功后系统自动发送电子内容，请查看包内使用说明。"),
        },
        "development_v11": {
            "direct": ("店铺沟通方式", f"订单支付后请使用 {rendered} 联系店铺处理交付。"),
            "risk": ("商品记录样本", f"商品数据库记录样本包含 {rendered}，这是内容字段。"),
            "support": ("示范阅读页面", f"阅读页面使用 {rendered} 进行公开演示。"),
            "neutral": ("文件发放须知", "支付完成后文件会自动发放，操作细节见随附说明。"),
        },
        "audit_v11": {
            "direct": ("当前订单联络", f"当前订单完成后请通过 {rendered} 联络卖家。"),
            "risk": ("出售库片段", f"出售数据库片段的账号字段是 {rendered}，它来自商品内容。"),
            "support": ("公共演示页面", f"公共页面用 {rendered} 展示演示写法。"),
            "neutral": ("电子交付提示", "订单付款后电子文件自动交付，请阅读其中的操作提示。"),
        },
        "train_v12": {
            "direct": ("本店履约联络", f"付款后请通过 {rendered} 联系本店完成履约。"),
            "risk": ("待售资料样例", f"待售资料的样例字段含 {rendered}，该值属于商品内容。"),
            "support": ("公开教程示例", f"公开教程使用 {rendered} 作为示例。"),
            "neutral": ("数字订单说明", "数字订单将在支付后自动交付，使用步骤见文件说明。"),
        },
        "development_v12": {
            "direct": ("商家交付入口", f"订单支付完成后可用 {rendered} 联系商家办理交付。"),
            "risk": ("销售数据预览", f"销售数据预览字段出现 {rendered}，它不是店铺联络方式。"),
            "support": ("阅读教程页面", f"教程页面列出 {rendered} 供公开阅读。"),
            "neutral": ("电子附件说明", "付款完成后系统自动提供电子附件，操作方法见附件。"),
        },
        "audit_v12": {
            "direct": ("订单履约渠道", f"如需处理当前订单，请用 {rendered} 联络本店。"),
            "risk": ("商品数据截取", f"商品数据截取样例包含 {rendered}，该值来自所售内容。"),
            "support": ("公共教程页面", f"公共教程页面展示 {rendered} 作为写法示例。"),
            "neutral": ("自动交付须知", "完成支付后电子资料会自动交付，请查看随附使用须知。"),
        },
    }
    title, description = templates[bank][role]
    fingerprint_source = f"{role}|{token_type}|{title}|" + description.replace(rendered, "<IDENTIFIER>")
    return title, description, hashlib.sha256(fingerprint_source.encode()).hexdigest()


def add_item(
    items: list[dict],
    planned: list[dict],
    *,
    world_uid: str,
    world_index: int,
    seller_uid: str,
    side: str,
    bank: str,
    role: str,
    uid_namespace: str = "step28-v4",
    token_type: str | None = None,
    value: str | None = None,
) -> str:
    if (token_type is None) != (value is None):
        raise ValueError("Step28/v4 occurrence token type/value must be both present or absent")
    effective_type = token_type or "email"
    title, description, fingerprint = render_occurrence(bank, role, effective_type, value or "neutral@example.invalid")
    if role == "neutral":
        title, description, fingerprint = render_occurrence(bank, role, effective_type, "neutral@example.invalid")
    ordinal = len(items)
    item_uid = base.opaque_uid(f"{uid_namespace}-item", world_uid, ordinal)
    source_namespace = uid_namespace.replace("-", "_")
    item = {
        "world_uid": world_uid,
        "item_uid": item_uid,
        "seller_uid": seller_uid,
        "source_dataset": f"{source_namespace}_virtual_history",
        "source_row_number": str(world_index * 100 + ordinal + 1),
        "data_bucket": f"{source_namespace}_synthetic_only",
        "source_market_raw": f"virtual_market_{side}",
        "source_seller_raw": seller_uid,
        "source_seller_id_raw": seller_uid,
        "alias_normalized": seller_uid,
        "title_raw": title,
        "description_raw": description,
        "structured_snapshot": "",
    }
    items.append(item)
    if role != "neutral":
        planned.append({
            "source_dataset": item["source_dataset"],
            "source_row_number": item["source_row_number"],
            "seller_uid": seller_uid,
            "contact_type": token_type,
            "normalized_value": value,
            "expected_context_role": role,
        })
    return fingerprint


def repeat_occurrence(
    items: list[dict], planned: list[dict], fingerprints: set[str], *,
    count: int, world_uid: str, world_index: int, seller_uid: str, side: str,
    bank: str, role: str, token_type: str, value: str,
    uid_namespace: str = "step28-v4",
) -> None:
    for _ in range(count):
        fingerprints.add(add_item(
            items, planned, world_uid=world_uid, world_index=world_index,
            seller_uid=seller_uid, side=side, bank=bank, role=role,
            token_type=token_type, value=value, uid_namespace=uid_namespace,
        ))


def build_world(
    split: str,
    recipe: str,
    local_index: int,
    bank: str,
    carrier: dict,
    policy: dict,
    rng: random.Random,
) -> tuple[dict, list[dict], list[dict], dict]:
    uid_namespace = str(policy["generation"].get("synthetic_namespace", "step28-v4"))
    world_index = int(
        base.opaque_uid(f"{uid_namespace}-world-index", split, recipe, local_index)[:10],
        16,
    ) % 80_000_000
    world_uid = base.opaque_uid(f"{uid_namespace}-world", split, recipe, local_index)
    left = base.opaque_uid(f"{uid_namespace}-seller", world_uid, "left")
    right = base.opaque_uid(f"{uid_namespace}-seller", world_uid, "right")
    middle = base.opaque_uid(f"{uid_namespace}-seller", world_uid, "middle")
    target_label = "positive" if recipe.startswith("positive_") else "negative"
    if target_label == "positive":
        controller = base.opaque_uid(f"{uid_namespace}-controller", world_uid, 0)
        controllers = {left: controller, right: controller, middle: controller}
    else:
        controllers = {
            seller: base.opaque_uid(f"{uid_namespace}-controller", world_uid, seller)
            for seller in (left, right, middle)
        }
    items: list[dict] = []
    planned: list[dict] = []
    fingerprints: set[str] = set()
    values: list[str] = []

    def new_token(token_type: str, variant: int) -> str:
        value = token_value(
            policy, world_uid, split, world_index, token_type, variant
        )
        values.append(value)
        return value

    def occurrence(seller: str, side: str, role: str, token_type: str, value: str, count: int) -> None:
        repeat_occurrence(
            items, planned, fingerprints, count=count, world_uid=world_uid,
            world_index=world_index, seller_uid=seller, side=side, bank=bank,
            role=role, token_type=token_type, value=value,
            uid_namespace=uid_namespace,
        )

    handled = False
    if policy["generation"].get("history_variation_version") == "v7_diverse":
        handled = True
        canonical = recipe.removesuffix("_holdout")
        aliases = {
            "positive_email_reuse": "positive_stable_reuse",
            "positive_multitype_reuse": "positive_multi_token_reuse",
            "positive_cross_channel_rotation": "positive_rotation_chain",
            "positive_noisy_rotation": "positive_noisy_reuse",
            "negative_sparse_collision": "negative_private_collision",
            "negative_adversarial_repeat": "negative_repeated_collision",
            "negative_email_hub": "negative_public_hub",
            "negative_product_telegram": "negative_product_leakage",
            "negative_support_email": "negative_support_leakage",
        }
        canonical = aliases.get(canonical, canonical)
        seed_material = (
            f"{policy['generation']['seed']}|{uid_namespace}|{split}|{recipe}|{local_index}"
        )
        world_rng = random.Random(
            int(hashlib.sha256(seed_material.encode("utf-8")).hexdigest()[:16], 16)
        )
        identifier_types = tuple(
            policy["generation"].get(
                "synthetic_identifier_types", history.SUPPORTED_IDENTITY_TYPES
            )
        )
        unsupported = sorted(set(identifier_types) - set(history.SUPPORTED_IDENTITY_TYPES))
        if unsupported or len(set(identifier_types)) != len(identifier_types):
            raise ValueError(
                "invalid Step28/v7 synthetic identifier types: " + ",".join(unsupported)
            )
        role_types = {
            "direct": tuple(
                policy["generation"].get(
                    "synthetic_direct_identifier_types", identifier_types
                )
            ),
            "risk": tuple(
                policy["generation"].get(
                    "synthetic_risk_identifier_types", identifier_types
                )
            ),
            "support": tuple(
                policy["generation"].get(
                    "synthetic_support_identifier_types", ("email", "external_url")
                )
            ),
        }
        if any(not values for values in role_types.values()) or any(
            set(values) - set(identifier_types) for values in role_types.values()
        ):
            raise ValueError("invalid Step28/v7 role-specific identifier type set")
        variant_counter = 0

        def fresh(token_type: str) -> str:
            nonlocal variant_counter
            value = new_token(token_type, variant_counter)
            variant_counter += 1
            return value

        def choose_types(count: int, role: str = "direct") -> list[str]:
            available = role_types[role]
            if count > len(available):
                raise ValueError("Step28/v7 requested more independent types than available")
            selected = list(available)
            world_rng.shuffle(selected)
            return selected[:count]

        def add_shared(role: str, token_count: int, minimum: int, maximum: int) -> None:
            for token_type in choose_types(token_count, role):
                value = fresh(token_type)
                left_count = world_rng.randint(minimum, maximum)
                right_count = world_rng.randint(minimum, maximum)
                occurrence(left, "a", role, token_type, value, left_count)
                occurrence(right, "b", role, token_type, value, right_count)

        def path_middle(path_index: int) -> str:
            seller = (
                middle
                if path_index == 0
                else base.opaque_uid(
                    f"{uid_namespace}-seller", world_uid, f"middle-{path_index}"
                )
            )
            if seller not in controllers:
                controllers[seller] = (
                    controllers[left]
                    if target_label == "positive"
                    else base.opaque_uid(
                        f"{uid_namespace}-controller", world_uid, seller
                    )
                )
            return seller

        def add_rotation_paths(
            path_count: int,
            *,
            repeated_probability: float,
            maximum_repeat: int,
            hub_distractors: bool = False,
        ) -> None:
            for path_index in range(path_count):
                current_middle = path_middle(path_index)
                first_type, second_type = choose_types(2)
                first, second = fresh(first_type), fresh(second_type)
                first_count = (
                    world_rng.randint(2, maximum_repeat)
                    if world_rng.random() < repeated_probability
                    else 1
                )
                second_count = (
                    world_rng.randint(2, maximum_repeat)
                    if world_rng.random() < repeated_probability
                    else 1
                )
                for seller, side, token_type, value, count in (
                    (left, "a", first_type, first, first_count),
                    (current_middle, f"m{path_index}", first_type, first, first_count),
                    (current_middle, f"m{path_index}", second_type, second, second_count),
                    (right, "b", second_type, second, second_count),
                ):
                    occurrence(seller, side, "direct", token_type, value, count)
                if hub_distractors:
                    for edge_index, (token_type, value) in enumerate(
                        ((first_type, first), (second_type, second))
                    ):
                        for distractor_index in range(world_rng.randint(3, 6)):
                            seller = base.opaque_uid(
                                f"{uid_namespace}-seller",
                                world_uid,
                                f"graph-hub-{path_index}-{edge_index}-{distractor_index}",
                            )
                            controllers[seller] = base.opaque_uid(
                                f"{uid_namespace}-controller", world_uid, seller
                            )
                            occurrence(
                                seller,
                                f"g{path_index}{edge_index}{distractor_index}",
                                "direct",
                                token_type,
                                value,
                                1,
                            )

        if canonical == "positive_stable_reuse":
            add_shared("direct", world_rng.randint(1, 4), 1, 5)
        elif canonical == "positive_multi_token_reuse":
            add_shared("direct", world_rng.randint(3, 6), 1, 4)
        elif canonical == "positive_sparse_multitoken":
            add_shared("direct", world_rng.randint(2, 6), 1, 1)
        elif canonical == "positive_rotation_chain":
            add_rotation_paths(
                world_rng.randint(1, 4),
                repeated_probability=0.80,
                maximum_repeat=4,
            )
        elif canonical == "positive_sparse_rotation":
            add_rotation_paths(
                world_rng.randint(1, 4),
                repeated_probability=0.0,
                maximum_repeat=2,
            )
        elif canonical == "positive_noisy_reuse":
            add_shared("direct", world_rng.randint(2, 5), 1, 4)
            add_shared("risk", world_rng.randint(1, 3), 1, 3)
        elif canonical == "positive_support_noise":
            add_shared("direct", world_rng.randint(2, 5), 1, 4)
            add_shared("support", world_rng.randint(1, 2), 1, 3)
        elif canonical == "positive_source_only":
            for seller, side in ((left, "a"), (right, "b")):
                fingerprints.add(
                    add_item(
                        items,
                        planned,
                        world_uid=world_uid,
                        world_index=world_index,
                        seller_uid=seller,
                        side=side,
                        bank=bank,
                        role="neutral",
                        uid_namespace=uid_namespace,
                    )
                )
        elif canonical == "negative_private_collision":
            add_shared("direct", world_rng.randint(1, 4), 1, 1)
        elif canonical == "negative_repeated_collision":
            add_shared("direct", world_rng.randint(1, 5), 2, 5)
        elif canonical == "negative_public_hub":
            for token_type in choose_types(world_rng.randint(1, 3)):
                value = fresh(token_type)
                occurrence(left, "a", "direct", token_type, value, 1)
                occurrence(right, "b", "direct", token_type, value, 1)
                for distractor_index in range(world_rng.randint(3, 7)):
                    seller = base.opaque_uid(
                        f"{uid_namespace}-seller",
                        world_uid,
                        f"hub-{variant_counter}-{distractor_index}",
                    )
                    controllers[seller] = base.opaque_uid(
                        f"{uid_namespace}-controller", world_uid, seller
                    )
                    occurrence(
                        seller,
                        f"h{distractor_index}",
                        "direct",
                        token_type,
                        value,
                        1,
                    )
        elif canonical == "negative_product_leakage":
            add_shared("risk", world_rng.randint(1, 4), 1, 4)
            if world_rng.random() < 0.70:
                add_shared("direct", world_rng.randint(1, 5), 1, 4)
        elif canonical == "negative_support_leakage":
            add_shared("support", world_rng.randint(1, 2), 1, 4)
            if world_rng.random() < 0.70:
                add_shared("direct", world_rng.randint(1, 5), 1, 4)
        elif canonical == "negative_graph_hub":
            add_rotation_paths(
                world_rng.randint(1, 3),
                repeated_probability=0.35,
                maximum_repeat=3,
                hub_distractors=True,
            )
        elif canonical == "negative_clean_graph_collision":
            add_rotation_paths(
                world_rng.randint(1, 4),
                repeated_probability=0.25,
                maximum_repeat=4,
            )
        elif canonical == "negative_source_only":
            for seller, side in ((left, "a"), (right, "b")):
                fingerprints.add(
                    add_item(
                        items,
                        planned,
                        world_uid=world_uid,
                        world_index=world_index,
                        seller_uid=seller,
                        side=side,
                        bank=bank,
                        role="neutral",
                        uid_namespace=uid_namespace,
                    )
                )
        else:
            raise ValueError(f"unknown Step28/v7 recipe: {recipe}")

    if handled:
        pass
    elif recipe in {"positive_stable_reuse", "positive_email_reuse_holdout"}:
        token_type = "email" if "email" in recipe else "telegram"
        value = new_token(token_type, 0)
        occurrence(left, "a", "direct", token_type, value, rng.randint(2, 4))
        occurrence(right, "b", "direct", token_type, value, rng.randint(2, 4))
    elif recipe in {"positive_multi_token_reuse", "positive_sparse_multitoken_holdout"}:
        count = 1 if "sparse" in recipe else 2
        for variant, token_type in enumerate(("telegram", "email")):
            value = new_token(token_type, variant)
            occurrence(left, "a", "direct", token_type, value, count)
            occurrence(right, "b", "direct", token_type, value, count)
    elif recipe in {
        "positive_rotation_chain",
        "positive_cross_channel_rotation_holdout",
        "positive_sparse_rotation",
        "positive_sparse_rotation_holdout",
    }:
        types = ("telegram", "email") if "cross_channel" in recipe else ("telegram", "telegram")
        first, second = new_token(types[0], 0), new_token(types[1], 1)
        occurrence_count = 1 if "sparse_rotation" in recipe else 2
        for seller, side, token_type, value in (
            (left, "a", types[0], first), (middle, "m", types[0], first),
            (middle, "m", types[1], second), (right, "b", types[1], second),
        ):
            occurrence(seller, side, "direct", token_type, value, occurrence_count)
    elif recipe == "positive_noisy_reuse":
        clean, noisy = new_token("telegram", 0), new_token("email", 1)
        occurrence(left, "a", "direct", "telegram", clean, 2)
        occurrence(right, "b", "direct", "telegram", clean, 2)
        occurrence(left, "a", "direct", "email", noisy, 1)
        occurrence(right, "b", "risk", "email", noisy, 1)
    elif recipe == "positive_noisy_rotation_holdout":
        first, second, noise = new_token("telegram", 0), new_token("email", 1), new_token("email", 2)
        for seller, side, token_type, value in (
            (left, "a", "telegram", first), (middle, "m", "telegram", first),
            (middle, "m", "email", second), (right, "b", "email", second),
        ):
            occurrence(seller, side, "direct", token_type, value, 2)
        occurrence(left, "a", "risk", "email", noise, 1)
        occurrence(right, "b", "risk", "email", noise, 1)
    elif recipe in {"positive_support_noise", "positive_support_noise_holdout"}:
        clean, support = new_token("telegram", 0), new_token("email", 1)
        occurrence(left, "a", "direct", "telegram", clean, 2)
        occurrence(right, "b", "direct", "telegram", clean, 2)
        occurrence(left, "a", "support", "email", support, 1)
        occurrence(right, "b", "support", "email", support, 1)
    elif recipe in {"positive_source_only", "positive_source_only_holdout"}:
        fingerprints.add(add_item(items, planned, world_uid=world_uid, world_index=world_index, seller_uid=left, side="a", bank=bank, role="neutral", uid_namespace=uid_namespace))
        fingerprints.add(add_item(items, planned, world_uid=world_uid, world_index=world_index, seller_uid=right, side="b", bank=bank, role="neutral", uid_namespace=uid_namespace))
    elif recipe in {"negative_private_collision", "negative_sparse_collision_holdout"}:
        value = new_token("telegram", 0)
        occurrence(left, "a", "direct", "telegram", value, 1)
        occurrence(right, "b", "direct", "telegram", value, 1)
    elif recipe in {"negative_repeated_collision", "negative_adversarial_repeat_holdout"}:
        value = new_token("telegram", 0)
        occurrence(left, "a", "direct", "telegram", value, 2)
        occurrence(right, "b", "direct", "telegram", value, 2)
    elif recipe in {"negative_public_hub", "negative_email_hub_holdout"}:
        token_type = "email" if "email" in recipe else "telegram"
        value = new_token(token_type, 0)
        for seller, side in ((left, "a"), (right, "b")):
            occurrence(seller, side, "direct", token_type, value, 1)
        for distractor_index in range(3):
            seller = base.opaque_uid(f"{uid_namespace}-seller", world_uid, f"hub-{distractor_index}")
            controllers[seller] = base.opaque_uid(f"{uid_namespace}-controller", world_uid, f"hub-{distractor_index}")
            occurrence(seller, f"h{distractor_index}", "direct", token_type, value, 1)
    elif recipe in {"negative_product_leakage", "negative_product_telegram_holdout"}:
        token_type = "telegram" if "telegram" in recipe else "email"
        value = new_token(token_type, 0)
        occurrence(left, "a", "risk", token_type, value, 2)
        occurrence(right, "b", "risk", token_type, value, 2)
    elif recipe in {"negative_graph_hub", "negative_graph_hub_holdout"}:
        first, second = new_token("telegram", 0), new_token("email", 1)
        for seller, side, token_type, value in (
            (left, "a", "telegram", first), (middle, "m", "telegram", first),
            (middle, "m", "email", second), (right, "b", "email", second),
        ):
            occurrence(seller, side, "direct", token_type, value, 1)
        for variant, (token_type, value) in enumerate((("telegram", first), ("email", second))):
            for distractor_index in range(3):
                seller = base.opaque_uid(f"{uid_namespace}-seller", world_uid, f"g{variant}-{distractor_index}")
                controllers[seller] = base.opaque_uid(f"{uid_namespace}-controller", world_uid, f"g{variant}-{distractor_index}")
                occurrence(seller, f"g{variant}{distractor_index}", "direct", token_type, value, 1)
    elif recipe in {"negative_clean_graph_collision", "negative_clean_graph_collision_holdout"}:
        first, second = new_token("telegram", 0), new_token("email", 1)
        for seller, side, token_type, value in (
            (left, "a", "telegram", first), (middle, "m", "telegram", first),
            (middle, "m", "email", second), (right, "b", "email", second),
        ):
            occurrence(seller, side, "direct", token_type, value, 1)
    elif recipe in {"negative_support_leakage", "negative_support_email_holdout"}:
        value = new_token("email", 0)
        occurrence(left, "a", "support", "email", value, 2)
        occurrence(right, "b", "support", "email", value, 2)
    else:
        raise ValueError(f"unknown Step28/v4 recipe: {recipe}")

    signals: list[dict] = []
    for item in items:
        meta = {key: item[key] for key in (
            "data_bucket", "source_dataset", "source_row_number", "seller_uid",
            "source_market_raw", "source_seller_raw", "source_seller_id_raw", "alias_normalized",
        )}
        extracted = step3.extract_item_identity_signals(
            meta,
            title_raw=item["title_raw"],
            description_raw=item["description_raw"],
            structured_snapshot=item["structured_snapshot"],
        )
        for row in extracted:
            record = stringify(row)
            record["world_uid"] = world_uid
            signals.append(record)
    def context_role(row: dict) -> str:
        if history.is_risky(row):
            return "risk"
        if history.is_support(row):
            return "support"
        if history.is_direct(row):
            return "direct"
        return "other"

    observed = Counter(
        (
            row["source_dataset"],
            row["source_row_number"],
            row["seller_uid"],
            row["contact_type"],
            row["normalized_value"],
            context_role(row),
        )
        for row in signals
    )
    wanted = Counter(
        (
            row["source_dataset"],
            row["source_row_number"],
            row["seller_uid"],
            row["contact_type"],
            row["normalized_value"],
            row["expected_context_role"],
        )
        for row in planned
    )
    parser_recovery = (
        1.0
        if not wanted and not observed
        else sum((observed & wanted).values()) / max(sum(wanted.values()), 1)
    )
    if observed != wanted:
        raise ValueError(
            f"Step28/v4 production parser did not exactly recover planned occurrences: {recipe}:"
            f" wanted={wanted - observed}, extra={observed - wanted}"
        )
    by_seller, token_df = history.build_signal_index(signals)
    graph = history.build_identity_graph(by_seller, token_df, policy)
    features, details = history.history_feature_details(left, right, by_seller, token_df, graph, policy)
    source_probability = history.source_probability_from_cosine(
        float(carrier["identifier_redacted_e5_cosine"]), policy
    )
    pair_uid = base.opaque_uid(f"{uid_namespace}-pair", world_uid, left, right)
    truth = {
        "pair_uid": pair_uid,
        "world_uid": world_uid,
        "synthetic_namespace": uid_namespace,
        "synthetic_split": split,
        "recipe_id": recipe,
        "review_label": target_label,
        "target_seller_uids": [left, right],
        "synthetic_seller_uids": sorted(controllers),
        "controller_uids": [controllers[left], controllers[right]],
        "same_latent_controller": controllers[left] == controllers[right],
        "identifier_values": sorted(set(values)),
        "template_text_sha256": sorted(fingerprints),
        "source_carrier_pair_uid": carrier["pair_uid"],
        "source_carrier_seller_uids": sorted(
            value
            for value in (
                carrier.get("seller_uid_left", ""),
                carrier.get("seller_uid_right", ""),
            )
            if value
        ),
        "source_carrier_cosine": float(carrier["identifier_redacted_e5_cosine"]),
        "parser_recovery": parser_recovery,
        "parser_observable_details": details,
        "benchmark_eligible": False,
        "real_identity_claim_allowed": False,
    }
    if policy["generation"].get(
        "source_carrier_assignment", "legacy_label_correlated"
    ) == "legacy_label_correlated":
        truth["source_carrier_review_label"] = carrier["review_label"]
    model_row = {
        "pair_uid": pair_uid,
        "synthetic_split": split,
        "recipe_id": recipe,
        "review_label": target_label,
        "source_carrier_pair_uid": carrier["pair_uid"],
        "source_probability": f"{source_probability:.12f}",
        **{name: f"{features[name]:.12f}" for name in policy["model"]["feature_names"]},
    }
    return truth, items, signals, model_row


def generate(policy: dict) -> tuple[list[dict], list[dict], list[dict], list[dict], dict]:
    pools = carrier_pools(policy)
    rng = random.Random(int(policy["generation"]["seed"]))
    assignment = policy["generation"].get(
        "source_carrier_assignment", "legacy_label_correlated"
    )
    truth_rows: list[dict] = []
    items: list[dict] = []
    signals: list[dict] = []
    model_rows: list[dict] = []
    for split, cfg in policy["generation"]["splits"].items():
        if assignment == "label_blind_exact_pairing":
            positive_recipes = [
                recipe for recipe in cfg["recipes"] if recipe.startswith("positive_")
            ]
            negative_recipes = [
                recipe for recipe in cfg["recipes"] if recipe.startswith("negative_")
            ]
            if len(positive_recipes) != len(negative_recipes):
                raise ValueError(
                    f"Step28 exact pairing requires equal positive/negative recipe counts: {split}"
                )
            carriers = list(pools[split]["all"])
            shuffle_seed = int(
                hashlib.sha256(
                    f"{policy['generation']['seed']}|{split}|label-blind-carriers".encode()
                ).hexdigest()[:16],
                16,
            )
            random.Random(shuffle_seed).shuffle(carriers)
            rows_per_recipe = int(cfg["rows_per_recipe"])
            for recipe_index, (positive_recipe, negative_recipe) in enumerate(
                zip(positive_recipes, negative_recipes)
            ):
                for local_index in range(rows_per_recipe):
                    carrier = carriers[
                        (recipe_index * rows_per_recipe + local_index) % len(carriers)
                    ]
                    for recipe in (positive_recipe, negative_recipe):
                        truth, world_items, world_signals, model_row = build_world(
                            split, recipe, local_index, cfg["template_bank"], carrier,
                            policy, rng,
                        )
                        truth_rows.append(truth)
                        items.extend(world_items)
                        signals.extend(world_signals)
                        model_rows.append(model_row)
        else:
            match_probability = float(
                policy["generation"]["source_carrier_label_match_probability"]
            )
            for recipe in cfg["recipes"]:
                label = "positive" if recipe.startswith("positive_") else "negative"
                for local_index in range(int(cfg["rows_per_recipe"])):
                    carrier_label = label if rng.random() < match_probability else (
                        "negative" if label == "positive" else "positive"
                    )
                    carrier = rng.choice(pools[split][carrier_label])
                    truth, world_items, world_signals, model_row = build_world(
                        split, recipe, local_index, cfg["template_bank"], carrier,
                        policy, rng,
                    )
                    truth_rows.append(truth)
                    items.extend(world_items)
                    signals.extend(world_signals)
                    model_rows.append(model_row)

    overlap: dict[str, dict[str, int]] = {}
    split_rows = {
        split: [row for row in truth_rows if row["synthetic_split"] == split]
        for split in policy["generation"]["splits"]
    }
    field_extractors = {
        "world_uid": lambda row: {row["world_uid"]},
        "synthetic_seller_uid": lambda row: set(row["synthetic_seller_uids"]),
        "identifier_value": lambda row: set(row["identifier_values"]),
        "source_carrier_pair_uid": lambda row: {row["source_carrier_pair_uid"]},
        "source_carrier_seller_uid": lambda row: set(row["source_carrier_seller_uids"]),
        "template_text_sha256": lambda row: set(row["template_text_sha256"]),
    }
    splits = list(split_rows)
    for field, extractor in field_extractors.items():
        sets = {
            split: set().union(*(extractor(row) for row in rows)) if rows else set()
            for split, rows in split_rows.items()
        }
        overlap[field] = {
            f"{splits[left]}__{splits[right]}": len(sets[splits[left]] & sets[splits[right]])
            for left in range(len(splits))
            for right in range(left + 1, len(splits))
        }
    source_balance_by_split = {}
    for split in split_rows:
        current = [row for row in model_rows if row["synthetic_split"] == split]
        positive = [row for row in current if row["review_label"] == "positive"]
        negative = [row for row in current if row["review_label"] == "negative"]
        positive_scores = sorted(float(row["source_probability"]) for row in positive)
        negative_scores = sorted(float(row["source_probability"]) for row in negative)
        same_length = len(positive_scores) == len(negative_scores)
        maximum_difference = (
            max(
                (abs(left - right) for left, right in zip(positive_scores, negative_scores)),
                default=0.0,
            )
            if same_length
            else None
        )
        positive_carriers = Counter(row["source_carrier_pair_uid"] for row in positive)
        negative_carriers = Counter(row["source_carrier_pair_uid"] for row in negative)
        labels = [int(row["review_label"] == "positive") for row in current]
        scores = [float(row["source_probability"]) for row in current]
        source_balance_by_split[split] = {
            "positive_row_count": len(positive),
            "negative_row_count": len(negative),
            "source_carrier_uid_multiset_exactly_equal": positive_carriers == negative_carriers,
            "source_probability_multiset_exactly_equal": (
                same_length and maximum_difference == 0.0
            ),
            "source_probability_sorted_max_abs_difference": maximum_difference,
            "source_only_roc_auc": base.roc_auc(labels, scores),
        }

    feature_names = policy["model"]["feature_names"]
    feature_states_by_split: dict[str, set[tuple[str, ...]]] = {}
    feature_state_diagnostics = {}
    for split in split_rows:
        current = [row for row in model_rows if row["synthetic_split"] == split]
        states = {
            tuple(row[name] for name in feature_names)
            for row in current
        }
        feature_states_by_split[split] = states
        matrix = np.asarray(
            [[float(row[name]) for name in feature_names] for row in current],
            dtype=float,
        )
        labels_by_state: dict[tuple[str, ...], set[str]] = {}
        for row in current:
            state = tuple(row[name] for name in feature_names)
            labels_by_state.setdefault(state, set()).add(row["review_label"])
        feature_state_diagnostics[split] = {
            "row_count": len(current),
            "unique_feature_state_count": len(states),
            "feature_matrix_rank": int(np.linalg.matrix_rank(matrix)),
            "cross_label_ambiguous_state_count": sum(
                len(labels) > 1 for labels in labels_by_state.values()
            ),
            "unique_feature_states_by_recipe": {
                recipe: len({
                    tuple(row[name] for name in feature_names)
                    for row in current
                    if row["recipe_id"] == recipe
                })
                for recipe in sorted({row["recipe_id"] for row in current})
            },
        }
    feature_state_overlap = {
        f"{splits[left]}__{splits[right]}": len(
            feature_states_by_split[splits[left]]
            & feature_states_by_split[splits[right]]
        )
        for left in range(len(splits))
        for right in range(left + 1, len(splits))
    }

    cross_version_overlap: dict[str, dict[str, int]] = {}
    for key in policy["generation"].get("cross_version_reference_world_truth_keys", []):
        prior_rows = base.load_jsonl(policy["inputs"][key])
        cross_version_overlap[key] = {
            "world_uid": len(
                {row["world_uid"] for row in truth_rows}
                & {row["world_uid"] for row in prior_rows}
            ),
            "synthetic_seller_uid": len(
                set().union(*(set(row["synthetic_seller_uids"]) for row in truth_rows))
                & set().union(*(set(row["synthetic_seller_uids"]) for row in prior_rows))
            ),
            "identifier_value": len(
                set().union(*(set(row["identifier_values"]) for row in truth_rows))
                & set().union(*(set(row["identifier_values"]) for row in prior_rows))
            ),
            "pair_uid": len(
                {row["pair_uid"] for row in truth_rows}
                & {row["pair_uid"] for row in prior_rows}
            ),
        }
    maximum_cross_version_overlap = max(
        (
            value
            for values in cross_version_overlap.values()
            for value in values.values()
        ),
        default=0,
    )
    allowed_cross_version_overlap = int(
        policy["generation"].get("cross_version_synthetic_overlap_maximum", 0)
    )
    if maximum_cross_version_overlap > allowed_cross_version_overlap:
        raise RuntimeError(
            "Step28 synthetic namespace overlaps a prior version: "
            f"{maximum_cross_version_overlap}>{allowed_cross_version_overlap}"
        )

    summary = {
        "row_count": len(truth_rows),
        "item_count": len(items),
        "parsed_occurrence_count": len(signals),
        "rows_by_split": dict(Counter(row["synthetic_split"] for row in truth_rows)),
        "labels_by_split": {
            split: dict(Counter(row["review_label"] for row in rows))
            for split, rows in split_rows.items()
        },
        "recipes": dict(Counter(row["recipe_id"] for row in truth_rows)),
        "source_carrier_partition_counts": {
            split: {label: len(rows) for label, rows in labels.items()}
            for split, labels in pools.items()
        },
        "source_carrier_domain": policy["generation"].get("source_carrier_domain", "zh_train_legacy"),
        "source_carrier_assignment": assignment,
        "source_carrier_label_file_open_count": int(
            bool(policy["inputs"].get("source_carrier_labels"))
        ),
        "source_carrier_label_column_count": int(
            any(
                "review_label" in row
                for split_pools in pools.values()
                for rows in split_pools.values()
                for row in rows
            )
        ),
        "source_label_balance_by_split": source_balance_by_split,
        "parser_recovery": min(row["parser_recovery"] for row in truth_rows),
        "split_overlap_counts": overlap,
        "feature_state_diagnostics": feature_state_diagnostics,
        "feature_state_overlap_counts": feature_state_overlap,
        "cross_version_synthetic_overlap_counts": cross_version_overlap,
        "maximum_cross_version_synthetic_overlap": maximum_cross_version_overlap,
        "duplicate_world_uid_count": len(truth_rows) - len({row["world_uid"] for row in truth_rows}),
        "duplicate_pair_uid_count": len(truth_rows) - len({row["pair_uid"] for row in truth_rows}),
        "duplicate_item_uid_count": len(items) - len({row["item_uid"] for row in items}),
        "old_valid_test_open_count": 0,
        "real_candidate_label_open_count": 0,
    }
    return truth_rows, items, signals, model_rows, summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", required=True)
    args = parser.parse_args()
    policy_path = base.resolve(args.policy)
    policy = history.load_policy(policy_path)
    base.validate_frozen_inputs(policy)
    base.frozen_source_artifact(policy)
    if Path(step3.__file__).resolve() != base.resolve(policy["inputs"]["production_item_parser"]).resolve():
        raise RuntimeError("Step28/v4 imported an unexpected production parser")
    truth, items, signals, model_rows, summary = generate(policy)
    root = base.output_root(policy)
    outputs = policy["outputs"]
    base.write_jsonl_immutable(root / outputs["world_truth"], truth)
    base.write_jsonl_immutable(root / outputs["synthetic_items"], items)
    base.write_csv_immutable(
        root / outputs["parsed_occurrences"], signals, ["world_uid", *SIGNAL_FIELDS]
    )
    base.write_csv_immutable(
        root / outputs["model_inputs"],
        model_rows,
        [*policy["feature_boundary"]["model_input_metadata_columns"], *policy["model"]["feature_names"]],
    )
    base.write_json_immutable(root / outputs["generation_summary"], summary)
    print(json.dumps({"status": "ok", **summary}, ensure_ascii=False))


if __name__ == "__main__":
    main()
