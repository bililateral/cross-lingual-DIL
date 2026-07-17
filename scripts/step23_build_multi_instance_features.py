#!/usr/bin/env python3
"""Build symmetric item-level multi-instance features for real Step23 train pairs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

import step7_build_semantic_pair_features as semantic


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_POLICY = ROOT / "schema" / "step23_item_multi_instance_policy.json"


def resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def bool_value(value: object) -> bool:
    return str(value or "").strip().casefold() in {"1", "true", "yes"}


def jaccard(left: set[str], right: set[str]) -> float:
    union = left | right
    return len(left & right) / len(union) if union else 0.0


def quantile(values: np.ndarray, value: float) -> float:
    return float(np.quantile(np.asarray(values, dtype=float), value))


def pair_features(
    left_items: list[dict],
    right_items: list[dict],
    embedding_index: dict[str, int],
    embeddings: np.ndarray,
    cfg: dict,
) -> dict[str, float]:
    left_matrix = np.asarray([embeddings[embedding_index[item["item_uid"]]] for item in left_items], dtype=np.float64)
    right_matrix = np.asarray([embeddings[embedding_index[item["item_uid"]]] for item in right_items], dtype=np.float64)
    similarities = np.asarray(left_matrix @ right_matrix.T, dtype=np.float64)
    left_mean = np.mean(left_matrix, axis=0)
    right_mean = np.mean(right_matrix, axis=0)
    left_mean_norm = float(np.linalg.norm(left_mean))
    right_mean_norm = float(np.linalg.norm(right_mean))
    if left_mean_norm <= 1e-12 or right_mean_norm <= 1e-12:
        raise ValueError("Step23 mean-pooled seller embedding has zero norm")
    mean_pool_cosine = float(
        np.dot(left_mean / left_mean_norm, right_mean / right_mean_norm)
    )
    flat = similarities.ravel()
    left_nearest = np.max(similarities, axis=1)
    right_nearest = np.max(similarities, axis=0)
    bidirectional_nearest = np.concatenate([left_nearest, right_nearest])
    left_argmax = np.argmax(similarities, axis=1)
    right_argmax = np.argmax(similarities, axis=0)
    mutual_count = sum(right_argmax[right_index] == left_index for left_index, right_index in enumerate(left_argmax))

    output = {
        "mi_mean_pool_cosine": mean_pool_cosine,
        "mi_item_count_min": float(min(len(left_items), len(right_items))),
        "mi_item_count_max": float(max(len(left_items), len(right_items))),
        "mi_item_count_log_gap": abs(math.log1p(len(left_items)) - math.log1p(len(right_items))),
        "mi_cross_pair_count_log": math.log1p(len(flat)),
        "mi_cosine_mean": float(np.mean(flat)),
        "mi_cosine_std": float(np.std(flat)),
        "mi_cosine_min": float(np.min(flat)),
        "mi_cosine_max": float(np.max(flat)),
        "mi_nn_mean": float(np.mean(bidirectional_nearest)),
        "mi_nn_std": float(np.std(bidirectional_nearest)),
        "mi_nn_min": float(np.min(bidirectional_nearest)),
        "mi_nn_max": float(np.max(bidirectional_nearest)),
        "mi_nn_side_mean_abs_gap": abs(float(np.mean(left_nearest)) - float(np.mean(right_nearest))),
        "mi_mutual_top1_share": mutual_count / max(min(len(left_items), len(right_items)), 1),
    }
    for q in cfg["cosine_distribution_quantiles"]:
        output[f"mi_cosine_q{int(round(float(q) * 100)):02d}"] = quantile(flat, float(q))
    for q in cfg["nearest_neighbor_quantiles"]:
        output[f"mi_nn_q{int(round(float(q) * 100)):02d}"] = quantile(bidirectional_nearest, float(q))
    descending = np.sort(flat)[::-1]
    for top_k in cfg["top_k_means"]:
        k = min(int(top_k), len(descending))
        output[f"mi_cosine_top{int(top_k)}_mean"] = float(np.mean(descending[:k]))
    output["mi_cosine_max_minus_median"] = output["mi_cosine_max"] - quantile(flat, 0.5)
    output["mi_cosine_q95_minus_median"] = quantile(flat, 0.95) - quantile(flat, 0.5)

    left_titles = {item["title_hash"] for item in left_items if item.get("title_hash")}
    right_titles = {item["title_hash"] for item in right_items if item.get("title_hash")}
    left_descriptions = {item["description_hash"] for item in left_items if item.get("description_hash")}
    right_descriptions = {item["description_hash"] for item in right_items if item.get("description_hash")}
    left_categories = {item["category_key"] for item in left_items if item.get("category_key")}
    right_categories = {item["category_key"] for item in right_items if item.get("category_key")}
    output.update({
        "mi_exact_title_intersection": float(len(left_titles & right_titles)),
        "mi_exact_title_jaccard": jaccard(left_titles, right_titles),
        "mi_exact_description_intersection": float(len(left_descriptions & right_descriptions)),
        "mi_exact_description_jaccard": jaccard(left_descriptions, right_descriptions),
        "mi_category_jaccard": jaccard(left_categories, right_categories),
    })

    matched_pairs = {(left_index, int(right_index)) for left_index, right_index in enumerate(left_argmax)}
    matched_pairs.update((int(left_index), right_index) for right_index, left_index in enumerate(right_argmax))
    for style_field in cfg["style_fields"]:
        gaps = [
            abs(float(left_items[left_index][style_field]) - float(right_items[right_index][style_field]))
            for left_index, right_index in sorted(matched_pairs)
        ]
        output[f"mi_best_match_{style_field}_gap_mean"] = float(np.mean(gaps))
        output[f"mi_best_match_{style_field}_gap_max"] = float(np.max(gaps))
    if any(not np.isfinite(value) for value in output.values()):
        raise ValueError("Step23 emitted a non-finite multi-instance feature")
    return output


def write_csv_immutable(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"Step23 refuses to write an empty feature file: {path}")
    fieldnames = list(rows[0])
    rendered_lines = []
    import io
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    rendered = ("\ufeff" + buffer.getvalue()).encode("utf-8")
    if path.exists():
        if path.read_bytes() != rendered:
            raise ValueError(f"Refusing to overwrite different Step23 features: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(rendered)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", default=str(DEFAULT_POLICY))
    args = parser.parse_args()
    policy_path = resolve(args.policy)
    policy = json.loads(policy_path.read_text(encoding="utf-8"))
    output_root = resolve(policy["outputs_root"])
    output_cfg = policy["outputs"]
    item_path = output_root / output_cfg["selected_items"]
    matrix_path = output_root / output_cfg["item_embedding_matrix"]
    metadata_path = output_root / output_cfg["item_embedding_metadata"]
    for path in (item_path, matrix_path, metadata_path):
        if not path.is_file():
            raise FileNotFoundError(f"Step23 feature input missing: {path}")

    items = semantic.load_jsonl(item_path)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    embeddings = np.load(matrix_path, mmap_mode="r")
    if list(embeddings.shape) != list(metadata["shape"]):
        raise ValueError("Step23 item embedding matrix/metadata shape mismatch")
    if metadata.get("valid_test_items_encoded") is not False:
        raise ValueError("Step23 feature builder refuses a valid/test item cache")
    embedding_norms = np.linalg.norm(np.asarray(embeddings, dtype=np.float32), axis=1)
    if np.max(np.abs(embedding_norms - 1.0)) > 1e-3:
        raise ValueError("Step23 item embeddings are not unit-normalized; dot product is not cosine")
    embedding_index = {uid: index for index, uid in enumerate(metadata["item_uids"])}
    if len(embedding_index) != len(metadata["item_uids"]):
        raise ValueError("Step23 embedding metadata contains duplicate item UIDs")
    items_by_pool_seller: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for item in items:
        if item["item_uid"] not in embedding_index:
            raise ValueError(f"Step23 item missing embedding: {item['item_uid']}")
        if not bool_value(item.get("exact_overlap_eligible", True)) and (
            item.get("title_hash") or item.get("description_hash")
        ):
            raise ValueError(
                f"Step23 cross-field-redacted item retained an exact-overlap hash: {item['item_uid']}"
            )
        items_by_pool_seller[(item["pool"], item["seller_uid"])].append(item)
    for values in items_by_pool_seller.values():
        values.sort(key=lambda row: (int(row["seller_item_rank"]), row["item_uid"]))
        if len(values) > int(policy["item_selection"]["maximum_items_per_seller"]):
            raise ValueError("Step23 selected-item cap was exceeded")

    feature_cfg = policy["multi_instance_features"]
    if not feature_cfg["forbid_identifier_features"] or not feature_cfg["forbid_global_idf_or_test_fitted_statistics"]:
        raise ValueError("Step23 multi-instance feature isolation was relaxed")
    output_paths = {}
    pool_summary = {}
    for pool_name, pool_cfg in policy["pools"].items():
        labels = load_csv(resolve(pool_cfg["frozen_labels"]))
        pairs = [
            row for row in labels
            if row.get("split_name") == "train"
            and row.get("review_label") in {"positive", "negative"}
            and bool_value(row.get("usable_for_supervision"))
        ]
        feature_rows = []
        for row in pairs:
            left_key = (pool_name, row["seller_uid_left"])
            right_key = (pool_name, row["seller_uid_right"])
            if left_key not in items_by_pool_seller or right_key not in items_by_pool_seller:
                raise ValueError(f"Step23 pair seller missing selected items: {row['pair_uid']}")
            features = pair_features(
                items_by_pool_seller[left_key], items_by_pool_seller[right_key], embedding_index, embeddings, feature_cfg
            )
            feature_rows.append({
                "pair_uid": row["pair_uid"],
                "pool": pool_name,
                "domain": pool_cfg["domain"],
                "split_name": "train",
                "seller_uid_left": row["seller_uid_left"],
                "seller_uid_right": row["seller_uid_right"],
                "review_label": row["review_label"],
                **{key: f"{value:.12f}" for key, value in features.items()},
            })
        output_name = output_cfg["pair_features_en"] if pool_cfg["domain"] == "en" else output_cfg["pair_features_zh"]
        output_path = output_root / output_name
        write_csv_immutable(output_path, feature_rows)
        output_paths[pool_name] = output_path
        pool_summary[pool_name] = {
            "pair_count": len(feature_rows),
            "positive_count": sum(row["review_label"] == "positive" for row in feature_rows),
            "negative_count": sum(row["review_label"] == "negative" for row in feature_rows),
            "feature_count": len(feature_rows[0]) - 7,
            "output_sha256": sha256_file(output_path),
        }

    summary_path = output_root / output_cfg["pair_feature_summary"]
    summary = {
        "step": "step23_item_multi_instance_feature_build",
        "policy_version": policy["version"],
        "status": "real_train_pairs_only_symmetric_identifier_free",
        "pool_summary": pool_summary,
        "valid_or_test_pairs_featurized": 0,
        "feature_names": list(load_csv(next(iter(output_paths.values())))[0].keys())[7:],
        "input_hashes": {
            "policy": sha256_file(policy_path),
            "producer": sha256_file(Path(__file__)),
            "items": sha256_file(item_path),
            "embedding_matrix": sha256_file(matrix_path),
            "embedding_metadata": sha256_file(metadata_path),
            **{
                f"{pool_name}:frozen_labels": sha256_file(resolve(pool_cfg["frozen_labels"]))
                for pool_name, pool_cfg in policy["pools"].items()
            },
        },
    }
    declared_feature_sets = policy["evaluation"]["model_feature_sets"]
    produced_feature_names = set(summary["feature_names"])
    for model_name, names in declared_feature_sets.items():
        if not names or len(names) != len(set(names)):
            raise ValueError(f"Step23 model feature set is empty or duplicated: {model_name}")
        missing = sorted(set(names) - produced_feature_names)
        if missing:
            raise ValueError(
                f"Step23 model feature set references missing features for {model_name}: {missing}"
            )
    rendered = json.dumps(summary, ensure_ascii=False, indent=2) + "\n"
    if summary_path.exists() and summary_path.read_text(encoding="utf-8") != rendered:
        raise ValueError("Refusing to overwrite a different Step23 feature summary")
    summary_path.write_text(rendered, encoding="utf-8")
    print(json.dumps({"status": summary["status"], "pool_summary": pool_summary}, indent=2))


if __name__ == "__main__":
    main()
