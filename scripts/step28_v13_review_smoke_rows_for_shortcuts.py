#!/usr/bin/env python3
"""Inspect every train/development smoke row for learnable shortcuts.

This is an explicitly development-only diagnostic.  It never opens the
Audit A/B oracle directories and cannot grant a formal dataset status.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
import re
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = (
    ROOT
    / "reports"
    / "step28_synthetic_chinese_dataset"
    / "v13_dev_smoke_v2_20260728"
    / "dataset_smoke_v3"
)
SPLITS = ("train", "development")
UNIVERSES = {
    "complete": "complete_model_pair_endpoints.csv",
    "c40": "candidate_pairs.csv",
}


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def reachable_style_to_base(
    template: Mapping[str, Any],
) -> dict[str, set[str]]:
    renderer = template["renderer_contract"]
    factor_order = list(renderer["style_factor_order"])
    domains = renderer["style_factor_domains"]
    output: dict[str, set[str]] = defaultdict(set)
    for prototype in template["style_prototypes"]:
        base_style = str(prototype["style_id"])
        for selected in itertools.combinations(factor_order, 2):
            style = {
                name: prototype[name] for name in factor_order
            }
            for factor in selected:
                domain = list(domains[factor])
                position = domain.index(style[factor])
                style[factor] = domain[(position + 1) % len(domain)]
            effective_uid = "estyle_" + hashlib.sha256(
                canonical_json_bytes(style)
            ).hexdigest()
            output[effective_uid].add(base_style)
    return output


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def normalized_segments(text: str) -> set[str]:
    return {
        re.sub(r"\s+", "", part)
        for part in re.split(r"[。！？!?；;\n•]+", text)
        if len(re.sub(r"\s+", "", part)) >= 8
    }


def character_ngrams(text: str, width: int = 3) -> Counter[str]:
    value = re.sub(r"\s+", "", text.lower())
    return Counter(
        value[index : index + width]
        for index in range(max(0, len(value) - width + 1))
    )


def counter_cosine(left: Counter[str], right: Counter[str]) -> float:
    numerator = sum(
        count * right.get(key, 0) for key, count in left.items()
    )
    left_norm = math.sqrt(sum(value * value for value in left.values()))
    right_norm = math.sqrt(
        sum(value * value for value in right.values())
    )
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return numerator / (left_norm * right_norm)


def set_jaccard(left: set[Any], right: set[Any]) -> float:
    union = left | right
    return len(left & right) / len(union) if union else 0.0


def safe_float(value: Any) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"Nonfinite value: {value!r}")
    return number


def auc_summary(labels: np.ndarray, values: np.ndarray) -> dict[str, Any]:
    raw = float(roc_auc_score(labels, values))
    return {
        "auc": raw,
        "auc_symmetric": max(raw, 1.0 - raw),
        "positive_direction": "higher" if raw >= 0.5 else "lower",
        "positive_mean": float(values[labels == 1].mean()),
        "negative_mean": float(values[labels == 0].mean()),
    }


def binary_rule_summary(
    labels: np.ndarray,
    values: np.ndarray,
) -> dict[str, Any]:
    selected = values.astype(bool)
    positives = labels == 1
    negatives = labels == 0
    return {
        **auc_summary(labels, values.astype(float)),
        "true_count": int(selected.sum()),
        "positive_recall": float(
            (selected & positives).sum() / positives.sum()
        ),
        "negative_false_positive_rate": float(
            (selected & negatives).sum() / negatives.sum()
        ),
        "positive_rate_when_true": (
            float(labels[selected].mean()) if selected.any() else None
        ),
        "positive_rate_when_false": (
            float(labels[~selected].mean())
            if (~selected).any()
            else None
        ),
    }


def grouped_oof(
    rows: Sequence[Mapping[str, Any]],
    feature_names: Sequence[str],
) -> dict[str, Any]:
    labels = np.asarray([int(row["label"]) for row in rows])
    groups = np.asarray([str(row["world_uid"]) for row in rows])
    matrix = np.asarray(
        [
            [safe_float(row[name]) for name in feature_names]
            for row in rows
        ],
        dtype=np.float64,
    )
    if len(set(groups)) < 5:
        raise ValueError("Grouped OOF requires at least five worlds")
    splitter = GroupKFold(n_splits=5)
    models = {
        "logistic": make_pipeline(
            StandardScaler(),
            LogisticRegression(
                C=1.0,
                max_iter=10000,
                random_state=20260729,
            ),
        ),
        "gradient_tree": HistGradientBoostingClassifier(
            max_depth=2,
            max_iter=200,
            learning_rate=0.03,
            l2_regularization=1.0,
            early_stopping=False,
            random_state=20260729,
        ),
    }
    output: dict[str, Any] = {}
    for model_name, model in models.items():
        scores = np.full(len(rows), np.nan, dtype=np.float64)
        for train_index, test_index in splitter.split(
            matrix,
            labels,
            groups,
        ):
            if (
                len(set(labels[train_index])) != 2
                or len(set(labels[test_index])) != 2
            ):
                raise ValueError("Grouped fold is single-class")
            model.fit(matrix[train_index], labels[train_index])
            if hasattr(model, "decision_function"):
                fold_scores = model.decision_function(
                    matrix[test_index]
                )
            else:
                fold_scores = model.predict_proba(
                    matrix[test_index]
                )[:, 1]
            scores[test_index] = np.asarray(fold_scores, dtype=float)
        if not np.isfinite(scores).all():
            raise ValueError("OOF scores are incomplete or nonfinite")
        auc = float(roc_auc_score(labels, scores))
        output[model_name] = {
            "auc": auc,
            "auc_symmetric": max(auc, 1.0 - auc),
            "average_precision": float(
                average_precision_score(labels, scores)
            ),
        }
    return output


def load_split(
    dataset: Path,
    split: str,
    style_to_base: Mapping[str, set[str]],
    *,
    oracle_directory_name: str = "oracle",
    audit_directory_name: str = "structural_audit",
) -> dict[str, Any]:
    if split not in SPLITS:
        raise ValueError("Only train/development supervision is allowed")
    root = dataset / split
    observed = root / "observed"
    oracle = root / oracle_directory_name
    sellers = read_csv(observed / "sellers.csv")
    items = read_jsonl(observed / "redacted_items.jsonl")
    profiles = read_jsonl(observed / "seller_profiles.jsonl")
    history = read_csv(observed / "history_item_index.csv")
    asts = read_jsonl(root / audit_directory_name / "render_asts.jsonl")
    membership_rows = read_csv(oracle / "controller_membership.csv")
    membership = {
        (row["world_uid"], row["seller_uid"]): row["controller_uid"]
        for row in membership_rows
    }
    if len(membership) != len(membership_rows):
        raise ValueError("Duplicate controller membership")
    controller_styles = {
        (row["world_uid"], row["controller_uid"]): row["style_id"]
        for row in read_csv(oracle / "controller_style_groups.csv")
    }

    item_by_seller: dict[tuple[str, str], list[dict[str, Any]]] = (
        defaultdict(list)
    )
    for item in items:
        item_by_seller[(item["world_uid"], item["seller_uid"])].append(
            item
        )
    time_by_seller: dict[tuple[str, str], Counter[int]] = defaultdict(
        Counter
    )
    for row in history:
        time_by_seller[(row["world_uid"], row["seller_uid"])][
            int(row["time_bucket"])
        ] += 1
    ast_by_seller: dict[tuple[str, str], list[dict[str, Any]]] = (
        defaultdict(list)
    )
    for row in asts:
        ast_by_seller[(row["world_uid"], row["seller_uid"])].append(
            row
        )
    profile_by_seller = {
        (row["world_uid"], row["seller_uid"]): row for row in profiles
    }
    seller_rank: dict[tuple[str, str], int] = {}
    sellers_by_world: dict[str, list[str]] = defaultdict(list)
    for row in sellers:
        sellers_by_world[row["world_uid"]].append(row["seller_uid"])
    for world_uid, seller_uids in sellers_by_world.items():
        for rank, seller_uid in enumerate(seller_uids):
            seller_rank[(world_uid, seller_uid)] = rank

    seller_features: dict[tuple[str, str], dict[str, Any]] = {}
    for source in sellers:
        world_uid = source["world_uid"]
        seller_uid = source["seller_uid"]
        key = (world_uid, seller_uid)
        seller_items = item_by_seller[key]
        seller_asts = ast_by_seller[key]
        profile = profile_by_seller[key]
        if not seller_items or not seller_asts:
            raise ValueError(f"Seller lacks items or AST rows: {key}")
        titles = [str(row["title"]) for row in seller_items]
        descriptions = [
            str(row["description"]) for row in seller_items
        ]
        text = "\n".join(
            f"{title}\n{description}"
            for title, description in zip(
                titles,
                descriptions,
                strict=True,
            )
        )
        categories = {
            str(row["value"]) for row in profile["top_categories"]
        }
        style = profile["style_stats"]
        effective_styles = {
            str(row["effective_style_uid"]) for row in seller_asts
        }
        inferred_base_styles: set[str] = set()
        for effective_style in effective_styles:
            inferred_base_styles.update(
                style_to_base.get(effective_style, set())
            )
        controller_uid = membership[key]
        item_count = len(seller_items)
        buckets = time_by_seller[key]
        seller_features[key] = {
            "market": source["market"],
            "item_count": float(item_count),
            "title_missing_rate": sum(
                not value for value in titles
            )
            / item_count,
            "description_missing_rate": sum(
                not value for value in descriptions
            )
            / item_count,
            "title_length_mean": sum(map(len, titles)) / item_count,
            "description_length_mean": (
                sum(map(len, descriptions)) / item_count
            ),
            "text_length": float(len(text)),
            "digit_ratio": safe_float(style["digit_ratio_mean"]),
            "punct_ratio": safe_float(style["punct_ratio_mean"]),
            "uppercase_ratio": safe_float(
                style["uppercase_ratio_mean"]
            ),
            "newline_count": safe_float(
                style["newline_count_mean"]
            ),
            "max_category_share": safe_float(
                style["max_category_share"]
            ),
            "categories": categories,
            "segments": normalized_segments(text),
            "char3": character_ngrams(text),
            "effective_styles": effective_styles,
            "inferred_base_styles": inferred_base_styles,
            "oracle_base_style": controller_styles[
                (world_uid, controller_uid)
            ],
            "title_skeletons": {
                int(row["title_skeleton_index"]) for row in seller_asts
            },
            "description_skeletons": {
                int(row["description_skeleton_index"])
                for row in seller_asts
            },
            "services": {
                str(row["service"]) for row in seller_asts
            },
            "deliveries": {
                str(row["delivery"]) for row in seller_asts
            },
            "products": {
                str(row["product"]) for row in seller_asts
            },
            "attributes": {
                str(row["attribute"]) for row in seller_asts
            },
            "time_probs": tuple(
                buckets[bucket] / item_count for bucket in range(4)
            ),
        }
    return {
        "root": root,
        "membership": membership,
        "seller_features": seller_features,
        "seller_rank": seller_rank,
        "sellers_by_world": sellers_by_world,
        "identity33": {
            row["canonical_pair_uid"]: row
            for row in read_csv(
                observed / "identity33_all_pairs.csv"
            )
        },
        "universes": {
            name: read_csv(observed / filename)
            for name, filename in UNIVERSES.items()
        },
        "row_counts": {
            "sellers": len(sellers),
            "redacted_items": len(items),
            "profiles": len(profiles),
            "history_items": len(history),
            "render_asts": len(asts),
            "membership": len(membership_rows),
            **{
                name: len(read_csv(observed / filename))
                for name, filename in UNIVERSES.items()
            },
        },
    }


def pair_record(
    *,
    split: str,
    source: Mapping[str, str],
    row_index_in_world: int,
    data: Mapping[str, Any],
) -> dict[str, Any]:
    world_uid = source["world_uid"]
    left_uid = source["seller_uid_left"]
    right_uid = source["seller_uid_right"]
    left = data["seller_features"][(world_uid, left_uid)]
    right = data["seller_features"][(world_uid, right_uid)]
    label = int(
        data["membership"][(world_uid, left_uid)]
        == data["membership"][(world_uid, right_uid)]
    )
    left_rank = data["seller_rank"][(world_uid, left_uid)]
    right_rank = data["seller_rank"][(world_uid, right_uid)]
    identity = data["identity33"][source["canonical_pair_uid"]]
    identity_values = [
        safe_float(value)
        for key, value in identity.items()
        if key not in {"canonical_pair_uid", "world_uid"}
    ]
    left_prefix = int(left_uid.removeprefix("sel_")[:8], 16)
    right_prefix = int(right_uid.removeprefix("sel_")[:8], 16)
    output: dict[str, Any] = {
        "split": split,
        "canonical_pair_uid": source["canonical_pair_uid"],
        "world_uid": world_uid,
        "seller_uid_left": left_uid,
        "seller_uid_right": right_uid,
        "label": label,
        "pair_position": float(row_index_in_world),
        "pair_position_mod2": float(row_index_in_world % 2 == 0),
        "pair_position_mod4": float(row_index_in_world % 4),
        "rank_absdiff": float(abs(left_rank - right_rank)),
        "rank_sum": float(left_rank + right_rank),
        "rank_same_mod2": float(left_rank % 2 == right_rank % 2),
        "rank_same_mod3": float(left_rank % 3 == right_rank % 3),
        "uid_prefix_absdiff": (
            abs(left_prefix - right_prefix) / (2**32 - 1)
        ),
        "uid_prefix_xor_popcount": float(
            (left_prefix ^ right_prefix).bit_count()
        ),
        "same_market": float(left["market"] == right["market"]),
        "item_count_absdiff": abs(
            left["item_count"] - right["item_count"]
        ),
        "item_count_sum": left["item_count"] + right["item_count"],
        "title_missing_absdiff": abs(
            left["title_missing_rate"]
            - right["title_missing_rate"]
        ),
        "description_missing_absdiff": abs(
            left["description_missing_rate"]
            - right["description_missing_rate"]
        ),
        "time_probability_l1": sum(
            abs(a - b)
            for a, b in zip(
                left["time_probs"],
                right["time_probs"],
                strict=True,
            )
        ),
        "title_length_absdiff": abs(
            left["title_length_mean"] - right["title_length_mean"]
        ),
        "description_length_absdiff": abs(
            left["description_length_mean"]
            - right["description_length_mean"]
        ),
        "text_length_absdiff": abs(
            left["text_length"] - right["text_length"]
        ),
        "digit_ratio_absdiff": abs(
            left["digit_ratio"] - right["digit_ratio"]
        ),
        "punct_ratio_absdiff": abs(
            left["punct_ratio"] - right["punct_ratio"]
        ),
        "uppercase_ratio_absdiff": abs(
            left["uppercase_ratio"] - right["uppercase_ratio"]
        ),
        "newline_count_absdiff": abs(
            left["newline_count"] - right["newline_count"]
        ),
        "max_category_share_absdiff": abs(
            left["max_category_share"]
            - right["max_category_share"]
        ),
        "category_jaccard": set_jaccard(
            left["categories"],
            right["categories"],
        ),
        "category_any_overlap": float(
            bool(left["categories"] & right["categories"])
        ),
        "text_char3_cosine": counter_cosine(
            left["char3"],
            right["char3"],
        ),
        "exact_segment_jaccard": set_jaccard(
            left["segments"],
            right["segments"],
        ),
        "same_effective_style": float(
            left["effective_styles"] == right["effective_styles"]
        ),
        "both_base_style_uniquely_inferred": float(
            len(left["inferred_base_styles"]) == 1
            and len(right["inferred_base_styles"]) == 1
        ),
        "same_inferred_base_style": float(
            len(left["inferred_base_styles"]) == 1
            and len(right["inferred_base_styles"]) == 1
            and left["inferred_base_styles"]
            == right["inferred_base_styles"]
        ),
        "same_oracle_base_style": float(
            left["oracle_base_style"] == right["oracle_base_style"]
        ),
        "title_skeleton_jaccard": set_jaccard(
            left["title_skeletons"],
            right["title_skeletons"],
        ),
        "description_skeleton_jaccard": set_jaccard(
            left["description_skeletons"],
            right["description_skeletons"],
        ),
        "service_jaccard": set_jaccard(
            left["services"],
            right["services"],
        ),
        "delivery_jaccard": set_jaccard(
            left["deliveries"],
            right["deliveries"],
        ),
        "product_jaccard": set_jaccard(
            left["products"],
            right["products"],
        ),
        "attribute_jaccard": set_jaccard(
            left["attributes"],
            right["attributes"],
        ),
        "identity33_nonzero_count": float(
            sum(value != 0.0 for value in identity_values)
        ),
        "identity33_l1": float(sum(abs(value) for value in identity_values)),
    }
    return output


def build_pair_rows(
    split_data: Mapping[str, Mapping[str, Any]],
    universe: str,
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for split, data in split_data.items():
        indices: Counter[str] = Counter()
        for source in data["universes"][universe]:
            world_uid = source["world_uid"]
            output.append(
                pair_record(
                    split=split,
                    source=source,
                    row_index_in_world=indices[world_uid],
                    data=data,
                )
            )
            indices[world_uid] += 1
    return output


def per_world_class_counts(
    rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    counts: dict[tuple[str, str], Counter[int]] = defaultdict(Counter)
    for row in rows:
        counts[(row["split"], row["world_uid"])][int(row["label"])] += 1
    return [
        {
            "split": split,
            "world_uid": world_uid,
            "negative": value[0],
            "positive": value[1],
        }
        for (split, world_uid), value in sorted(counts.items())
    ]


def audit_universe(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    labels = np.asarray([int(row["label"]) for row in rows])
    nuisance_features = (
        "item_count_absdiff",
        "item_count_sum",
        "title_missing_absdiff",
        "description_missing_absdiff",
        "time_probability_l1",
    )
    uid_order_features = (
        "pair_position",
        "pair_position_mod2",
        "pair_position_mod4",
        "rank_absdiff",
        "rank_sum",
        "rank_same_mod2",
        "rank_same_mod3",
        "uid_prefix_absdiff",
        "uid_prefix_xor_popcount",
    )
    visible_features = (
        *nuisance_features,
        "same_market",
        "title_length_absdiff",
        "description_length_absdiff",
        "text_length_absdiff",
        "digit_ratio_absdiff",
        "punct_ratio_absdiff",
        "uppercase_ratio_absdiff",
        "newline_count_absdiff",
        "max_category_share_absdiff",
        "category_jaccard",
        "category_any_overlap",
        "text_char3_cosine",
        "exact_segment_jaccard",
        "title_skeleton_jaccard",
        "description_skeleton_jaccard",
        "service_jaccard",
        "delivery_jaccard",
        "product_jaccard",
        "attribute_jaccard",
    )
    oracle_style_features = (
        "same_effective_style",
        "same_inferred_base_style",
        "same_oracle_base_style",
    )
    identity_features = (
        "identity33_nonzero_count",
        "identity33_l1",
    )
    single_feature: dict[str, Any] = {}
    for feature in (
        *uid_order_features,
        *visible_features,
        *oracle_style_features,
        *identity_features,
    ):
        values = np.asarray(
            [safe_float(row[feature]) for row in rows],
            dtype=np.float64,
        )
        single_feature[feature] = auc_summary(labels, values)
    ranked = sorted(
        single_feature.items(),
        key=lambda item: item[1]["auc_symmetric"],
        reverse=True,
    )
    binary_rules = {
        name: binary_rule_summary(
            labels,
            np.asarray(
                [safe_float(row[name]) for row in rows],
                dtype=float,
            ),
        )
        for name in (
            "same_effective_style",
            "same_inferred_base_style",
            "same_oracle_base_style",
            "same_market",
            "category_any_overlap",
        )
    }
    feature_groups = {
        "uid_and_file_order": uid_order_features,
        "nuisance_only": nuisance_features,
        "visible_m0_proxy": visible_features,
        "visible_plus_oracle_style": (
            *visible_features,
            *oracle_style_features,
        ),
        "identity33_intended_signal": identity_features,
    }
    examples = {
        "same_style_negative": [
            {
                key: row[key]
                for key in (
                    "split",
                    "world_uid",
                    "canonical_pair_uid",
                    "text_char3_cosine",
                    "item_count_absdiff",
                )
            }
            for row in rows
            if row["label"] == 0
            and row["same_effective_style"] == 1.0
        ][:8],
        "same_style_positive": [
            {
                key: row[key]
                for key in (
                    "split",
                    "world_uid",
                    "canonical_pair_uid",
                    "text_char3_cosine",
                    "item_count_absdiff",
                )
            }
            for row in rows
            if row["label"] == 1
            and row["same_effective_style"] == 1.0
        ][:8],
        "largest_positive_item_count_difference": [
            {
                key: row[key]
                for key in (
                    "split",
                    "world_uid",
                    "canonical_pair_uid",
                    "item_count_absdiff",
                    "text_char3_cosine",
                )
            }
            for row in sorted(
                (row for row in rows if row["label"] == 1),
                key=lambda row: row["item_count_absdiff"],
                reverse=True,
            )[:8]
        ],
    }
    return {
        "row_count": len(rows),
        "positive_count": int(labels.sum()),
        "negative_count": int((labels == 0).sum()),
        "per_world_class_counts": per_world_class_counts(rows),
        "single_feature_ranked": [
            {"feature": name, **summary}
            for name, summary in ranked
        ],
        "binary_rules": binary_rules,
        "grouped_world_oof": {
            group: grouped_oof(rows, features)
            for group, features in feature_groups.items()
        },
        "examples": examples,
    }


def build_report(dataset: Path) -> dict[str, Any]:
    if not (dataset / "release_manifest.json").is_file():
        raise FileNotFoundError("dataset_smoke_v3 release is absent")
    template = json.loads(
        (
            ROOT / "schema" / "step28_v13_synthetic_text_templates.json"
        ).read_text(encoding="utf-8")
    )
    style_to_base = reachable_style_to_base(template)
    split_data = {
        split: load_split(dataset, split, style_to_base)
        for split in SPLITS
    }
    seller_style_rows = [
        features
        for data in split_data.values()
        for features in data["seller_features"].values()
    ]
    report = {
        "version": (
            "2026-07-29-step28-v13-development-row-shortcut-review-v2"
        ),
        "status": (
            "DEVELOPMENT_DIAGNOSTIC_ONLY_NOT_FORMAL_EVIDENCE"
        ),
        "explicit_boundary": {
            "supervised_splits_opened": list(SPLITS),
            "audit_a_b_oracle_opened": False,
            "all_rows_processed": True,
            "formal_status_granted": False,
            "purpose": (
                "User-directed row-by-row shortcut review of the "
                "existing development smoke dataset"
            ),
        },
        "dataset_root": dataset.as_posix(),
        "style_recoverability": {
            "reachable_effective_style_count": len(style_to_base),
            "uniquely_mapped_effective_style_count": sum(
                len(base_styles) == 1
                for base_styles in style_to_base.values()
            ),
            "seller_count": len(seller_style_rows),
            "seller_unique_base_recovery_count": sum(
                len(row["inferred_base_styles"]) == 1
                for row in seller_style_rows
            ),
            "seller_unique_base_recovery_rate": (
                sum(
                    len(row["inferred_base_styles"]) == 1
                    for row in seller_style_rows
                )
                / len(seller_style_rows)
            ),
            "seller_unique_recovery_correct_count": sum(
                len(row["inferred_base_styles"]) == 1
                and next(iter(row["inferred_base_styles"]))
                == row["oracle_base_style"]
                for row in seller_style_rows
            ),
        },
        "split_row_counts": {
            split: data["row_counts"]
            for split, data in split_data.items()
        },
        "universes": {
            universe: audit_universe(
                build_pair_rows(split_data, universe)
            )
            for universe in UNIVERSES
        },
    }
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET,
    )
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = build_report(args.dataset)
    payload = json.dumps(
        report,
        ensure_ascii=False,
        sort_keys=True,
        indent=2,
    ) + "\n"
    if args.output is None:
        print(payload, end="")
        return
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise FileExistsError(
            f"Refusing to overwrite development review: {args.output}"
        )
    args.output.write_text(payload, encoding="utf-8")
    print(f"Wrote development row shortcut review: {args.output}")


if __name__ == "__main__":
    main()
