"""Train and evaluate the frozen V6 English-style initialization controls.

This stage selects source epochs on V6 development, then evaluates the fixed
source initializations on V6 synthetic audit, V5 real-English holdout, and
Chinese development zero-shot.  It never reads Audit A/B truth.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

import step28_v13_v1_13_v9_4_1_labse_finetune_common_v1 as token_common
import step28_v13_v1_13_v9_4_1_v6_style_transfer_common_v2 as common


ROOT = common.REPO_ROOT


class StyleTransferRuntimeError(RuntimeError):
    pass


def _json_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False
        )
        + "\n"
    ).encode("utf-8")


def _write_json(path: Path, value: object) -> None:
    path.write_bytes(_json_bytes(value))


def set_determinism(torch: Any, seed: int) -> None:
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def require_gpu_runtime(policy: Mapping[str, Any]) -> tuple[Any, Any]:
    try:
        for package in ("scipy", "sklearn", "transformers"):
            __import__(package)
        import torch
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise StyleTransferRuntimeError(
            "PyTorch and sentence-transformers are required on Linux"
        ) from exc
    if not torch.cuda.is_available():
        raise StyleTransferRuntimeError("A CUDA GPU is required")
    import step7_v4_common as step7_common

    step7_policy = step7_common.load_policy()
    observed = step7_common.validate_model_payload(
        "labse", step7_policy["embedding_models"]["labse"]
    )
    for field in ("file_count", "total_size_bytes", "content_sha256"):
        if observed[field] != policy["labse_model"][field]:
            raise StyleTransferRuntimeError(f"LaBSE payload drift: {field}")
    return torch, SentenceTransformer


def runtime_package_versions() -> dict[str, str]:
    return {
        name: importlib.metadata.version(name)
        for name in (
            "sentence-transformers",
            "transformers",
            "scikit-learn",
            "scipy",
        )
    }


def implementation_file_records(relative_paths: Sequence[str]) -> list[dict[str, Any]]:
    records = []
    for relative in relative_paths:
        path = ROOT / relative
        if not path.is_file():
            raise StyleTransferRuntimeError(
                f"Implementation file is missing: {relative}"
            )
        records.append(
            {
                "path": relative,
                "size_bytes": path.stat().st_size,
                "sha256": common.sha256_file(path),
            }
        )
    return records


def load_encoder(
    policy: Mapping[str, Any], SentenceTransformer: Any, torch: Any, seed: int
) -> tuple[Any, Any]:
    set_determinism(torch, seed)
    encoder = SentenceTransformer(
        str(ROOT / policy["labse_model"]["path"]),
        device="cuda:0",
        local_files_only=True,
    )
    if int(encoder.max_seq_length) != int(
        policy["labse_model"]["native_max_sequence_length"]
    ):
        raise StyleTransferRuntimeError("LaBSE maximum sequence length drift")
    if getattr(encoder, "default_prompt_name", None) is not None:
        encoder.default_prompt_name = None
    set_determinism(torch, seed)
    return encoder, encoder.tokenizer


def prepare_stream_chunks(
    tokenizer: Any,
    streams: Mapping[str, str],
    policy: Mapping[str, Any],
    *,
    ablated: bool = False,
) -> dict[str, tuple[tuple[str, ...], ...]]:
    result = {}
    maximum = int(policy["style_input"]["maximum_placeholders_per_window"])
    token_budget = int(policy["style_input"]["token_budget_including_special_tokens"])
    for number, uid in enumerate(
        sorted(streams, key=lambda value: value.encode("utf-8")), start=1
    ):
        stream = streams[uid]
        if ablated:
            stream = common.ablate_order_format_mass(stream)
        windows = common.split_style_windows(stream, maximum)
        subchunks = []
        for window in windows:
            chunks = tuple(token_common.chunk_text_exact(tokenizer, window, token_budget))
            if "".join(chunks) != window:
                raise StyleTransferRuntimeError("Token subchunk reconstruction failed")
            subchunks.append(chunks)
        if "".join("".join(value) for value in subchunks) != stream:
            raise StyleTransferRuntimeError("Hierarchical style reconstruction failed")
        result[uid] = tuple(subchunks)
        if number % 5000 == 0:
            print(f"风格流分块：{number}/{len(streams)}", flush=True)
    return result


def _unit_mean(torch: Any, matrix: Any) -> Any:
    return torch.nn.functional.normalize(
        matrix.float().mean(dim=0), p=2, dim=0, eps=1e-12
    )


def encode_accounts(
    torch: Any,
    encoder: Any,
    tokenizer: Any,
    prepared: Mapping[str, Sequence[Sequence[str]]],
    account_uids: Sequence[str],
    *,
    batch_size: int = 24,
    use_autocast: bool = True,
) -> Any:
    ordered = tuple(sorted(account_uids, key=lambda value: value.encode("utf-8")))
    flat_chunks: list[str] = []
    window_slices: list[slice] = []
    account_window_slices: list[slice] = []
    for uid in ordered:
        if uid not in prepared:
            raise StyleTransferRuntimeError(f"Missing prepared account: {uid}")
        window_start = len(window_slices)
        for chunks in prepared[uid]:
            chunk_start = len(flat_chunks)
            flat_chunks.extend(chunks)
            window_slices.append(slice(chunk_start, len(flat_chunks)))
        account_window_slices.append(slice(window_start, len(window_slices)))
    chunk_vectors = []
    device = torch.device("cuda:0")
    for start in range(0, len(flat_chunks), batch_size):
        texts = flat_chunks[start : start + batch_size]
        features = tokenizer(
            texts,
            padding=True,
            truncation=False,
            add_special_tokens=True,
            return_tensors="pt",
        )
        if int(features["attention_mask"].sum(dim=1).max()) > 256:
            raise StyleTransferRuntimeError("A style subchunk exceeds 256 tokens")
        features = {key: value.to(device) for key, value in features.items()}
        with torch.autocast(
            device_type="cuda",
            dtype=torch.bfloat16,
            enabled=use_autocast,
        ):
            encoded = encoder(features)["sentence_embedding"]
        chunk_vectors.append(
            torch.nn.functional.normalize(
                encoded.float(), p=2, dim=1, eps=1e-12
            )
        )
    chunks = torch.cat(chunk_vectors, dim=0)
    windows = torch.stack([_unit_mean(torch, chunks[value]) for value in window_slices])
    return torch.stack(
        [_unit_mean(torch, windows[value]) for value in account_window_slices]
    )


def pair_cosines(
    embeddings: Any,
    account_uids: Sequence[str],
    pairs: Sequence[Mapping[str, Any]],
    *,
    left_key: str,
    right_key: str,
) -> np.ndarray:
    index = {uid: position for position, uid in enumerate(account_uids)}
    left_indices = [index[str(row[left_key])] for row in pairs]
    right_indices = [index[str(row[right_key])] for row in pairs]
    result = (
        (
            embeddings[left_indices].float()
            * embeddings[right_indices].float()
        )
        .sum(dim=1)
        .detach()
        .double()
        .cpu()
        .numpy()
    )
    result = np.ascontiguousarray(result, dtype="<f8")
    if not np.isfinite(result).all():
        raise StyleTransferRuntimeError("Non-finite cosine scores")
    return result


def ranking_metrics(
    labels: Sequence[int],
    scores: Sequence[float],
    weights: Sequence[float] | None = None,
) -> dict[str, float]:
    from sklearn.metrics import (
        average_precision_score,
        auc,
        precision_recall_curve,
        roc_auc_score,
        roc_curve,
    )

    y = np.asarray(labels, dtype=np.int8)
    s = np.asarray(scores, dtype="<f8")
    w = None if weights is None else np.asarray(weights, dtype="<f8")
    if len(y) != len(s) or set(np.unique(y).tolist()) != {0, 1}:
        raise StyleTransferRuntimeError("Ranking metric input is invalid")
    precision, recall, _ = precision_recall_curve(y, s, sample_weight=w)
    fpr, tpr, _ = roc_curve(
        y, s, sample_weight=w, drop_intermediate=False
    )
    prevalence = float(np.average(y.astype(float), weights=w))
    return {
        "average_precision": float(average_precision_score(y, s, sample_weight=w)),
        "trapezoidal_pr_auc": float(auc(recall, precision)),
        "roc_auc": float(roc_auc_score(y, s, sample_weight=w)),
        "recall_at_fpr_1pct": float(np.max(np.r_[0.0, tpr[fpr <= 0.01]])),
        "prevalence": prevalence,
        "average_precision_lift": float(
            average_precision_score(y, s, sample_weight=w) - prevalence
        ),
    }


def fit_monotonic_calibrator(
    scores: Sequence[float], labels: Sequence[int], weights: Sequence[float]
) -> dict[str, float]:
    from scipy.optimize import minimize
    from scipy.special import expit

    x = np.asarray(scores, dtype="<f8")
    y = np.asarray(labels, dtype="<f8")
    w = np.asarray(weights, dtype="<f8")
    total = float(w.sum())

    def objective(parameters: np.ndarray) -> tuple[float, np.ndarray]:
        log_scale, bias = parameters
        scale = math.exp(float(log_scale))
        logits = scale * x + bias
        loss = float(np.sum(w * (np.logaddexp(0.0, logits) - y * logits)) / total)
        error = w * (expit(logits) - y) / total
        gradient = np.asarray(
            [np.sum(error * scale * x), np.sum(error)], dtype="<f8"
        )
        return loss, gradient

    result = minimize(
        objective,
        np.asarray([0.0, 0.0], dtype="<f8"),
        jac=True,
        method="L-BFGS-B",
        bounds=((-8.0, 8.0), (-20.0, 20.0)),
        options={"ftol": 1e-15, "gtol": 1e-10, "maxiter": 1000},
    )
    if not result.success or not np.isfinite(result.x).all():
        raise StyleTransferRuntimeError(f"Calibration failed: {result.message}")
    return {
        "log_scale": float(result.x[0]),
        "scale": float(math.exp(float(result.x[0]))),
        "bias": float(result.x[1]),
        "loss": float(result.fun),
        "iterations": int(result.nit),
    }


def calibrated_probabilities(
    scores: Sequence[float], calibrator: Mapping[str, float]
) -> np.ndarray:
    from scipy.special import expit

    result = expit(
        float(calibrator["scale"]) * np.asarray(scores, dtype="<f8")
        + float(calibrator["bias"])
    )
    return np.asarray(result, dtype="<f8")


def select_f1_threshold(
    probabilities: Sequence[float],
    labels: Sequence[int],
    weights: Sequence[float],
) -> dict[str, float]:
    p = np.asarray(probabilities, dtype="<f8")
    y = np.asarray(labels, dtype=np.int8)
    w = np.asarray(weights, dtype="<f8")
    order = np.argsort(-p, kind="stable")
    sorted_p = p[order]
    sorted_y = y[order]
    sorted_w = w[order]
    ends = np.flatnonzero(np.r_[sorted_p[1:] != sorted_p[:-1], True])
    tp = np.cumsum(sorted_w * sorted_y)[ends]
    fp = np.cumsum(sorted_w * (1 - sorted_y))[ends]
    positive = float(np.sum(sorted_w * sorted_y))
    fn = positive - tp
    denominator = 2.0 * tp + fp + fn
    f1 = np.divide(2.0 * tp, denominator, out=np.zeros_like(tp), where=denominator > 0)
    best = float(np.max(f1))
    candidates = ends[np.isclose(f1, best, rtol=0.0, atol=1e-15)]
    chosen = int(candidates[np.argmax(sorted_p[candidates])])
    return {"threshold": float(sorted_p[chosen]), "development_f1": best}


def probability_metrics(
    probabilities: Sequence[float],
    labels: Sequence[int],
    weights: Sequence[float] | None,
    threshold: float,
) -> dict[str, float]:
    p = np.asarray(probabilities, dtype="<f8")
    y = np.asarray(labels, dtype=np.int8)
    w = np.ones(len(y), dtype="<f8") if weights is None else np.asarray(weights, dtype="<f8")
    predicted = p >= threshold
    tp = float(w[(y == 1) & predicted].sum())
    fp = float(w[(y == 0) & predicted].sum())
    tn = float(w[(y == 0) & ~predicted].sum())
    fn = float(w[(y == 1) & ~predicted].sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    mcc_denominator = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    clipped = np.clip(p, 1e-15, 1.0 - 1e-15)
    return {
        "threshold": float(threshold),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "specificity": specificity,
        "balanced_accuracy": (recall + specificity) / 2.0,
        "mcc": (tp * tn - fp * fn) / mcc_denominator if mcc_denominator else 0.0,
        "brier": float(np.average(np.square(p - y), weights=w)),
        "log_loss": float(
            np.average(-(y * np.log(clipped) + (1 - y) * np.log1p(-clipped)), weights=w)
        ),
    }


def _one_query_retrieval(
    candidates: Sequence[tuple[str, float]], relevant: set[str]
) -> dict[str, float]:
    ordered = sorted(
        candidates, key=lambda value: (-value[1], value[0].encode("utf-8"))
    )
    relevance = np.asarray(
        [candidate in relevant for candidate, _score in ordered], dtype=np.int8
    )
    relevant_total = len(relevant)
    if relevant_total <= 0:
        raise StyleTransferRuntimeError("Retrieval query has no relevant candidate")
    ranks = np.flatnonzero(relevance) + 1
    average_precision = float(
        np.mean(
            [
                float(np.sum(relevance[:rank])) / rank
                for rank in ranks.tolist()
            ]
        )
    )
    output = {
        "mrr": 1.0 / float(ranks[0]),
        "map": average_precision,
    }
    for k in (1, 3, 5, 10):
        hits = int(np.sum(relevance[:k]))
        output[f"recall_at_{k}"] = hits / relevant_total
        discounts = 1.0 / np.log2(np.arange(2, min(k, len(relevance)) + 2))
        dcg = float(np.sum(relevance[:k] * discounts))
        ideal_count = min(k, relevant_total)
        ideal = float(np.sum(1.0 / np.log2(np.arange(2, ideal_count + 2))))
        output[f"ndcg_at_{k}"] = dcg / ideal if ideal else 0.0
    return output


def aggregate_retrieval(
    rows: Mapping[str, Mapping[str, float]],
    qrels: Mapping[str, set[str]],
    *,
    query_weights: Mapping[str, float] | None = None,
) -> dict[str, float]:
    per_query = {}
    for query, relevant in qrels.items():
        if query not in rows:
            raise StyleTransferRuntimeError(f"Retrieval query is missing: {query}")
        candidates = tuple(rows[query].items())
        if not relevant.issubset(rows[query]):
            raise StyleTransferRuntimeError("A relevant item is absent from candidates")
        per_query[query] = _one_query_retrieval(candidates, relevant)
    metric_names = tuple(next(iter(per_query.values())))
    if query_weights is None:
        weights = {query: 1.0 for query in per_query}
    else:
        weights = {query: float(query_weights[query]) for query in per_query}
    denominator = sum(weights.values())
    return {
        name: float(
            sum(weights[query] * values[name] for query, values in per_query.items())
            / denominator
        )
        for name in metric_names
    }


def v6_retrieval(
    embeddings: Any,
    account_uids: Sequence[str],
    pairs: Sequence[Mapping[str, Any]],
    component: Mapping[str, str],
    component_members: Mapping[str, Sequence[str]],
) -> dict[str, dict[str, float]]:
    matrix = embeddings.detach().float().cpu().numpy()
    similarities = matrix @ matrix.T
    candidates = {}
    for index, query in enumerate(account_uids):
        candidates[query] = {
            candidate: float(similarities[index, other])
            for other, candidate in enumerate(account_uids)
            if other != index
        }
    qrels: dict[str, set[str]] = defaultdict(set)
    for row in pairs:
        if int(row["label"]) != 1:
            continue
        left = str(row["account_left_uid"])
        right = str(row["account_right_uid"])
        qrels[left].add(right)
        qrels[right].add(left)
    weights = {
        uid: 1.0 / len(component_members[component[uid]]) for uid in qrels
    }
    return {
        "query_macro": aggregate_retrieval(candidates, qrels),
        "controller_equal": aggregate_retrieval(
            candidates, qrels, query_weights=weights
        ),
    }


def limited_pair_retrieval(
    pairs: Sequence[Mapping[str, Any]], scores: Sequence[float]
) -> dict[str, float]:
    candidates: dict[str, dict[str, float]] = defaultdict(dict)
    qrels: dict[str, set[str]] = defaultdict(set)
    for row, score in zip(pairs, scores):
        left = str(row["account_left_uid"])
        right = str(row["account_right_uid"])
        candidates[left][right] = float(score)
        candidates[right][left] = float(score)
        if int(row["label"]) == 1:
            qrels[left].add(right)
            qrels[right].add(left)
    eligible_candidates = {query: candidates[query] for query in qrels}
    return aggregate_retrieval(eligible_candidates, qrels)


def _source_head(torch: Any, policy: Mapping[str, Any]) -> Any:
    config = policy["source_optimization"]

    class PositiveCosineHead(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.log_scale = torch.nn.Parameter(
                torch.tensor(float(config["log_scale_initial"]), dtype=torch.float32)
            )
            self.bias = torch.nn.Parameter(
                torch.tensor(float(config["bias_initial"]), dtype=torch.float32)
            )

        def forward(self, cosines: Any) -> Any:
            lower, upper = config["log_scale_bounds"]
            scale = torch.exp(torch.clamp(self.log_scale, float(lower), float(upper)))
            return scale * cosines + self.bias

    return PositiveCosineHead().to("cuda:0")


def _training_view(
    streams: Mapping[str, str],
    arm: str,
    seed: int,
) -> tuple[dict[str, str], dict[str, str] | None]:
    if arm == "v6_permuted":
        mapping = common.deterministic_stream_derangement(tuple(streams), seed)
        return {uid: streams[source_uid] for uid, source_uid in mapping.items()}, mapping
    return dict(streams), None


def train_source_encoder(
    torch: Any,
    SentenceTransformer: Any,
    policy: Mapping[str, Any],
    v6: Mapping[str, Any],
    arm: str,
    seed: int,
    epochs: int,
    *,
    development_prepared: Mapping[str, Sequence[Sequence[str]]],
) -> tuple[Any, Any, list[dict[str, float]], dict[str, Any]]:
    if arm not in {
        "v6_correct",
        "v6_permuted",
        "v6_order_format_mass_ablated",
    }:
        raise StyleTransferRuntimeError(f"Invalid trainable source arm: {arm}")
    config = policy["source_optimization"]
    encoder, tokenizer = load_encoder(policy, SentenceTransformer, torch, seed)
    encoder.eval()
    head = _source_head(torch, policy)
    train_accounts = {
        uid: row
        for uid, row in v6["accounts"].items()
        if row["split"] == "train"
    }
    streams = {uid: row["style_stream"] for uid, row in train_accounts.items()}
    view_streams, permutation = _training_view(streams, arm, seed)
    prepared = prepare_stream_chunks(
        tokenizer,
        view_streams,
        policy,
        ablated=arm == "v6_order_format_mass_ablated",
    )
    pairs = [row for row in v6["pairs"] if row["split"] == "train"]
    account_component, components = common.positive_components(train_accounts, pairs)
    contributions = common.component_loss_contributions(pairs, account_component)
    common.audit_component_class_mass(pairs, contributions)
    component_ids = tuple(components)
    optimizer = torch.optim.AdamW(
        [
            {
                "params": encoder.parameters(),
                "lr": float(config["encoder_learning_rate"]),
                "weight_decay": float(config["encoder_weight_decay"]),
            },
            {
                "params": head.parameters(),
                "lr": float(config["head_learning_rate"]),
                "weight_decay": float(config["head_weight_decay"]),
            },
        ],
        betas=tuple(float(value) for value in config["adamw_betas"]),
        eps=float(config["adamw_eps"]),
    )
    development_pairs = [
        row for row in v6["pairs"] if row["split"] == "development"
    ]
    development_uids = tuple(
        sorted(
            (
                uid
                for uid, row in v6["accounts"].items()
                if row["split"] == "development"
            ),
            key=lambda value: value.encode("utf-8"),
        )
    )
    development_labels = np.asarray(
        [row["label"] for row in development_pairs], dtype=np.int8
    )
    development_weights = np.asarray(
        [row["sample_weight"] for row in development_pairs], dtype="<f8"
    )
    trace = []
    with torch.no_grad():
        initial_embeddings = encode_accounts(
            torch, encoder, tokenizer, development_prepared, development_uids
        )
    initial_scores = pair_cosines(
        initial_embeddings,
        development_uids,
        development_pairs,
        left_key="account_left_uid",
        right_key="account_right_uid",
    )
    trace.append(
        {
            "epoch": 0,
            "development_weighted_average_precision": ranking_metrics(
                development_labels, initial_scores, development_weights
            )["average_precision"],
            "optimizer_updates": 0,
        }
    )
    del initial_embeddings
    update_count = 0
    first_gradient_norm = None
    for epoch in range(1, epochs + 1):
        generator = np.random.Generator(np.random.PCG64(seed + epoch - 1))
        order = list(component_ids)
        generator.shuffle(order)
        losses = []
        row_forward_counts = np.zeros(len(pairs), dtype=np.int16)
        row_raw_weight_mass = np.zeros(len(pairs), dtype="<f8")
        width = int(config["components_per_gradient_step"])
        for start in range(0, len(order), width):
            batch_components = order[start : start + width]
            pair_weights: dict[int, float] = defaultdict(float)
            for component_uid in batch_components:
                for pair_index, weight in contributions[component_uid]:
                    pair_weights[pair_index] += float(weight)
            pair_indices = tuple(sorted(pair_weights))
            row_forward_counts[list(pair_indices)] += 1
            row_raw_weight_mass[list(pair_indices)] += np.asarray(
                [pair_weights[index] for index in pair_indices], dtype="<f8"
            )
            selected_pairs = [pairs[index] for index in pair_indices]
            selected_uids = tuple(
                sorted(
                    {
                        str(row["account_left_uid"])
                        for row in selected_pairs
                    }
                    | {
                        str(row["account_right_uid"])
                        for row in selected_pairs
                    },
                    key=lambda value: value.encode("utf-8"),
                )
            )
            optimizer.zero_grad(set_to_none=True)
            with torch.no_grad():
                frozen_embeddings = encode_accounts(
                    torch, encoder, tokenizer, prepared, selected_uids
                )
            leaf = frozen_embeddings.detach().requires_grad_(True)
            uid_index = {uid: index for index, uid in enumerate(selected_uids)}
            cosines = torch.stack(
                [
                    (
                        leaf[uid_index[str(row["account_left_uid"])]]
                        * leaf[uid_index[str(row["account_right_uid"])]]
                    ).sum()
                    for row in selected_pairs
                ]
            )
            labels = torch.tensor(
                [float(row["label"]) for row in selected_pairs],
                device="cuda:0",
                dtype=torch.float32,
            )
            weights = torch.tensor(
                [pair_weights[index] for index in pair_indices],
                device="cuda:0",
                dtype=torch.float32,
            )
            logits = head(cosines)
            individual = torch.nn.functional.binary_cross_entropy_with_logits(
                logits, labels, reduction="none"
            )
            expected_mass = 2.0 * len(batch_components)
            if not math.isclose(
                float(weights.sum().detach().cpu()), expected_mass, abs_tol=1e-5
            ):
                raise StyleTransferRuntimeError("Source batch objective mass drift")
            loss = torch.sum(individual * weights) / expected_mass
            loss.backward()
            if leaf.grad is None or not torch.isfinite(leaf.grad).all():
                raise StyleTransferRuntimeError("Invalid source embedding gradient")
            text_gradient = leaf.grad.detach()
            graph_embeddings = encode_accounts(
                torch, encoder, tokenizer, prepared, selected_uids
            )
            torch.sum(graph_embeddings.float() * text_gradient.float()).backward()
            gradient_norm = float(
                torch.nn.utils.clip_grad_norm_(
                    list(encoder.parameters()) + list(head.parameters()),
                    float(config["gradient_clip_norm"]),
                )
                .detach()
                .cpu()
            )
            if not math.isfinite(gradient_norm) or gradient_norm <= 0.0:
                raise StyleTransferRuntimeError("Invalid source parameter gradient")
            if first_gradient_norm is None:
                first_gradient_norm = gradient_norm
            optimizer.step()
            update_count += 1
            losses.append(float(loss.detach().cpu()))
            del frozen_embeddings, leaf, text_gradient, graph_embeddings
        expected_updates = epoch * int(config["expected_updates_per_epoch"])
        if update_count != expected_updates:
            raise StyleTransferRuntimeError("Source optimizer update count drift")
        base_row_weights = np.asarray(
            [float(row["sample_weight"]) for row in pairs], dtype="<f8"
        )
        mass_error = float(np.max(np.abs(row_raw_weight_mass - base_row_weights)))
        if mass_error > 1e-12:
            raise StyleTransferRuntimeError("Source row raw weight mass drift")
        labels_array = np.asarray(
            [int(row["label"]) for row in pairs], dtype=np.int8
        )
        if np.any(row_forward_counts[labels_array == 1] != 1):
            raise StyleTransferRuntimeError("A source positive row was not forwarded once")
        if np.any(
            (row_forward_counts[labels_array == 0] < 1)
            | (row_forward_counts[labels_array == 0] > 2)
        ):
            raise StyleTransferRuntimeError("A source negative row has invalid visits")
        with torch.no_grad():
            development_embeddings = encode_accounts(
                torch, encoder, tokenizer, development_prepared, development_uids
            )
        development_scores = pair_cosines(
            development_embeddings,
            development_uids,
            development_pairs,
            left_key="account_left_uid",
            right_key="account_right_uid",
        )
        development_ap = ranking_metrics(
            development_labels, development_scores, development_weights
        )["average_precision"]
        trace.append(
            {
                "epoch": epoch,
                "mean_component_batch_loss": float(np.mean(losses)),
                "development_weighted_average_precision": development_ap,
                "optimizer_updates": update_count,
                "positive_rows_forwarded_once": int(
                    np.sum(row_forward_counts[labels_array == 1] == 1)
                ),
                "negative_rows_forwarded_once": int(
                    np.sum(row_forward_counts[labels_array == 0] == 1)
                ),
                "negative_rows_forwarded_twice": int(
                    np.sum(row_forward_counts[labels_array == 0] == 2)
                ),
                "row_raw_weight_mass_max_abs_error": mass_error,
            }
        )
        print(
            f"{arm} seed={seed} 来源轮次 {epoch}/{epochs}："
            f"开发AP={development_ap:.6f}，累计更新={update_count}",
            flush=True,
        )
        del development_embeddings
    permutation_audit = None
    if permutation is not None:
        residual = {0: 0, 1: 0}
        totals = {0: 0, 1: 0}
        for row in pairs:
            totals[int(row["label"])] += 1
            same_component = (
                account_component[permutation[row["account_left_uid"]]]
                == account_component[permutation[row["account_right_uid"]]]
            )
            if same_component:
                residual[int(row["label"])] += 1
        permutation_audit = {
            "mapping_sha256": hashlib.sha256(
                json.dumps(
                    permutation,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest(),
            "fixed_points": sum(
                uid == source_uid for uid, source_uid in permutation.items()
            ),
            "positive_same_controller_residual": residual[1],
            "negative_same_controller_residual": residual[0],
            "positive_pair_count": totals[1],
            "negative_pair_count": totals[0],
            "positive_same_controller_residual_rate": residual[1] / totals[1],
            "negative_same_controller_residual_rate": residual[0] / totals[0],
        }
    audit = {
        "arm": arm,
        "seed": seed,
        "epochs": epochs,
        "optimizer_updates": update_count,
        "first_pre_clip_gradient_norm": first_gradient_norm,
        "permutation": permutation_audit,
        "audit_a_truth_reads": 0,
        "audit_b_truth_reads": 0,
    }
    return encoder, tokenizer, trace, audit


def _evaluate_v6_split(
    torch: Any,
    encoder: Any,
    tokenizer: Any,
    prepared: Mapping[str, Sequence[Sequence[str]]],
    v6: Mapping[str, Any],
    split: str,
) -> tuple[np.ndarray, dict[str, Any], Any, tuple[str, ...]]:
    accounts = tuple(
        sorted(
            (
                uid
                for uid, row in v6["accounts"].items()
                if row["split"] == split
            ),
            key=lambda value: value.encode("utf-8"),
        )
    )
    pairs = [row for row in v6["pairs"] if row["split"] == split]
    with torch.no_grad():
        embeddings = encode_accounts(torch, encoder, tokenizer, prepared, accounts)
    scores = pair_cosines(
        embeddings,
        accounts,
        pairs,
        left_key="account_left_uid",
        right_key="account_right_uid",
    )
    labels = np.asarray([row["label"] for row in pairs], dtype=np.int8)
    weights = np.asarray([row["sample_weight"] for row in pairs], dtype="<f8")
    account_component, components = common.positive_components(accounts, pairs)
    metrics = {
        "unweighted_pair": ranking_metrics(labels, scores),
        "controller_equal": ranking_metrics(labels, scores, weights),
        "retrieval": v6_retrieval(
            embeddings, accounts, pairs, account_component, components
        ),
    }
    return scores, metrics, embeddings, accounts


def _evaluate_v5(
    torch: Any,
    encoder: Any,
    tokenizer: Any,
    prepared: Mapping[str, Sequence[Sequence[str]]],
    pairs: Sequence[Mapping[str, Any]],
    calibrator: Mapping[str, float],
    threshold: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    accounts = tuple(sorted(prepared, key=lambda value: value.encode("utf-8")))
    with torch.no_grad():
        embeddings = encode_accounts(torch, encoder, tokenizer, prepared, accounts)
    scores = pair_cosines(
        embeddings,
        accounts,
        pairs,
        left_key="account_left_uid",
        right_key="account_right_uid",
    )
    labels = np.asarray([row["label"] for row in pairs], dtype=np.int8)
    probabilities = calibrated_probabilities(scores, calibrator)
    metrics = {
        "ranking": ranking_metrics(labels, scores),
        "probability_and_threshold": probability_metrics(
            probabilities, labels, None, threshold
        ),
        "limited_candidate_retrieval": limited_pair_retrieval(pairs, scores),
    }
    del embeddings
    return scores, metrics


def _evaluate_chinese_development(
    torch: Any,
    encoder: Any,
    tokenizer: Any,
    prepared: Mapping[str, Sequence[Sequence[str]]],
    worlds: Mapping[str, Mapping[str, str]],
    pairs: Sequence[Mapping[str, Any]],
    qrel_rows: Sequence[Mapping[str, Any]],
) -> tuple[np.ndarray, dict[str, Any]]:
    pair_groups: dict[str, list[tuple[int, Mapping[str, Any]]]] = defaultdict(list)
    for index, row in enumerate(pairs):
        pair_groups[str(row["world_uid"])].append((index, row))
    qrels = {
        str(row["query_seller_uid"]): set(row["relevant_seller_uids"])
        for row in qrel_rows
    }
    scores = np.empty(len(pairs), dtype="<f8")
    per_world_ranking = []
    per_world_retrieval = []
    for world_number, (world_uid, sellers) in enumerate(worlds.items(), start=1):
        account_uids = tuple(
            sorted(sellers, key=lambda value: value.encode("utf-8"))
        )
        indexed_pairs = pair_groups[world_uid]
        world_pairs = [row for _index, row in indexed_pairs]
        with torch.no_grad():
            embeddings = encode_accounts(
                torch, encoder, tokenizer, prepared, account_uids
            )
        world_scores = pair_cosines(
            embeddings,
            account_uids,
            world_pairs,
            left_key="seller_uid_left",
            right_key="seller_uid_right",
        )
        for (index, _row), score in zip(indexed_pairs, world_scores):
            scores[index] = score
        world_labels = [int(row["label"]) for row in world_pairs]
        per_world_ranking.append(ranking_metrics(world_labels, world_scores))
        candidates: dict[str, dict[str, float]] = defaultdict(dict)
        for row, score in zip(world_pairs, world_scores):
            left = str(row["seller_uid_left"])
            right = str(row["seller_uid_right"])
            candidates[left][right] = float(score)
            candidates[right][left] = float(score)
        world_qrels = {uid: qrels[uid] for uid in account_uids}
        per_world_retrieval.append(aggregate_retrieval(candidates, world_qrels))
        del embeddings
        if world_number % 50 == 0:
            print(f"中文零样本推理：{world_number}/{len(worlds)} 世界", flush=True)
    labels = np.asarray([row["label"] for row in pairs], dtype=np.int8)
    ranking_names = tuple(per_world_ranking[0])
    retrieval_names = tuple(per_world_retrieval[0])
    metrics = {
        "pooled_ranking": ranking_metrics(labels, scores),
        "world_equal_ranking": {
            name: float(np.mean([row[name] for row in per_world_ranking]))
            for name in ranking_names
        },
        "retrieval_world_equal": {
            name: float(np.mean([row[name] for row in per_world_retrieval]))
            for name in retrieval_names
        },
    }
    return scores, metrics


def evaluate_bundle(
    torch: Any,
    encoder: Any,
    tokenizer: Any,
    policy: Mapping[str, Any],
    v6: Mapping[str, Any],
    v6_prepared: Mapping[str, Sequence[Sequence[str]]],
    v5_prepared: Mapping[str, Sequence[Sequence[str]]],
    v5_pairs: Sequence[Mapping[str, Any]],
    chinese_prepared: Mapping[str, Sequence[Sequence[str]]],
    chinese_worlds: Mapping[str, Mapping[str, str]],
    chinese_pairs: Sequence[Mapping[str, Any]],
    chinese_qrels: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    split_results = {}
    split_scores = {}
    split_embeddings = {}
    split_accounts = {}
    for split in ("train", "development", "synthetic_audit"):
        scores, metrics, embeddings, accounts = _evaluate_v6_split(
            torch, encoder, tokenizer, v6_prepared, v6, split
        )
        split_scores[split] = scores
        split_results[split] = metrics
        split_embeddings[split] = embeddings
        split_accounts[split] = accounts
    train_pairs = [row for row in v6["pairs"] if row["split"] == "train"]
    train_labels = [row["label"] for row in train_pairs]
    train_weights = [row["sample_weight"] for row in train_pairs]
    calibrator = fit_monotonic_calibrator(
        split_scores["train"], train_labels, train_weights
    )
    development_pairs = [
        row for row in v6["pairs"] if row["split"] == "development"
    ]
    development_probabilities = calibrated_probabilities(
        split_scores["development"], calibrator
    )
    threshold_record = select_f1_threshold(
        development_probabilities,
        [row["label"] for row in development_pairs],
        [row["sample_weight"] for row in development_pairs],
    )
    threshold = threshold_record["threshold"]
    for split in ("train", "development", "synthetic_audit"):
        pairs = [row for row in v6["pairs"] if row["split"] == split]
        probabilities = calibrated_probabilities(split_scores[split], calibrator)
        split_results[split]["probability_and_threshold"] = probability_metrics(
            probabilities,
            [row["label"] for row in pairs],
            [row["sample_weight"] for row in pairs],
            threshold,
        )
    v5_scores, v5_metrics = _evaluate_v5(
        torch,
        encoder,
        tokenizer,
        v5_prepared,
        v5_pairs,
        calibrator,
        threshold,
    )
    chinese_scores, chinese_metrics = _evaluate_chinese_development(
        torch,
        encoder,
        tokenizer,
        chinese_prepared,
        chinese_worlds,
        chinese_pairs,
        chinese_qrels,
    )
    for value in split_embeddings.values():
        del value
    return (
        {
            "v6": split_results,
            "calibration": calibrator,
            "threshold_selection": threshold_record,
            "v5_real_holdout": v5_metrics,
            "chinese_development_zero_shot": chinese_metrics,
            "audit_a_truth_reads": 0,
            "audit_b_truth_reads": 0,
        },
        {
            "v6_train": split_scores["train"],
            "v6_development": split_scores["development"],
            "v6_synthetic_audit": split_scores["synthetic_audit"],
            "v5_real_holdout": v5_scores,
            "chinese_development_zero_shot": chinese_scores,
        },
    )


def _earliest_best_epoch(
    trace: Sequence[Mapping[str, float]], tolerance: float
) -> int:
    values = np.asarray(
        [row["development_weighted_average_precision"] for row in trace],
        dtype="<f8",
    )
    if not np.isfinite(values).all():
        raise StyleTransferRuntimeError("Non-finite source selection metric")
    maximum = float(np.max(values))
    return int(
        next(
            row["epoch"]
            for row in trace
            if maximum - float(row["development_weighted_average_precision"])
            <= tolerance
        )
    )


def select_source_epochs(
    torch: Any,
    SentenceTransformer: Any,
    policy: Mapping[str, Any],
    v6: Mapping[str, Any],
    development_prepared: Mapping[str, Sequence[Sequence[str]]],
) -> dict[str, Any]:
    config = policy["source_optimization"]
    seed = int(config["selection_seed"])
    initial = int(config["initial_max_epochs"])
    encoder, _tokenizer, trace, audit = train_source_encoder(
        torch,
        SentenceTransformer,
        policy,
        v6,
        "v6_correct",
        seed,
        initial,
        development_prepared=development_prepared,
    )
    selected = _earliest_best_epoch(trace, float(config["metric_tie_tolerance"]))
    del encoder
    gc.collect()
    torch.cuda.empty_cache()
    extended = False
    if selected == initial:
        extended = True
        maximum = int(config["extended_max_epochs"])
        encoder, _tokenizer, trace, audit = train_source_encoder(
            torch,
            SentenceTransformer,
            policy,
            v6,
            "v6_correct",
            seed,
            maximum,
            development_prepared=development_prepared,
        )
        selected = _earliest_best_epoch(
            trace, float(config["metric_tie_tolerance"])
        )
        del encoder
        gc.collect()
        torch.cuda.empty_cache()
    return {
        "selection_seed": seed,
        "selected_epochs": selected,
        "extended_to_30": extended,
        "trace": trace,
        "training_audit": audit,
        "v5_labels_read_during_selection": 0,
        "chinese_labels_read_during_selection": 0,
        "audit_a_truth_reads": 0,
        "audit_b_truth_reads": 0,
    }


def bootstrap_global_ap(
    labels: Sequence[int],
    scores: Sequence[float],
    base_weights: Sequence[float],
    group_left: Sequence[int],
    group_right: Sequence[int],
    draws: np.ndarray,
    *,
    batch_size: int = 128,
) -> np.ndarray:
    y = np.asarray(labels, dtype=np.int8)
    s = np.asarray(scores, dtype="<f8")
    base = np.asarray(base_weights, dtype="<f8")
    left = np.asarray(group_left, dtype=np.int32)
    right = np.asarray(group_right, dtype=np.int32)
    group_count = draws.shape[1]
    order = np.argsort(-s, kind="stable")
    sorted_scores = s[order]
    sorted_labels = y[order].astype("<f8")
    group_ends = np.flatnonzero(
        np.r_[sorted_scores[1:] != sorted_scores[:-1], True]
    )
    output = np.empty(len(draws), dtype="<f8")
    for start in range(0, len(draws), batch_size):
        stop = min(start + batch_size, len(draws))
        multiplicity = np.zeros((stop - start, group_count), dtype="<f8")
        for row_number, draw in enumerate(draws[start:stop]):
            multiplicity[row_number] = np.bincount(
                draw, minlength=group_count
            )
        row_weight = (
            base[None, :]
            * (multiplicity[:, left] + multiplicity[:, right])
            / 2.0
        )
        row_weight = row_weight[:, order]
        positive_total = row_weight @ sorted_labels
        if np.any(positive_total <= 0):
            raise StyleTransferRuntimeError("Bootstrap draw lost all positives")
        tp = np.cumsum(row_weight * sorted_labels, axis=1)[:, group_ends]
        fp = np.cumsum(row_weight * (1.0 - sorted_labels), axis=1)[:, group_ends]
        recall = tp / positive_total[:, None]
        precision = np.divide(
            tp,
            tp + fp,
            out=np.ones_like(tp),
            where=(tp + fp) != 0.0,
        )
        previous_recall = np.concatenate(
            (np.zeros((stop - start, 1)), recall[:, :-1]), axis=1
        )
        output[start:stop] = np.sum(
            (recall - previous_recall) * precision, axis=1
        )
    if not np.isfinite(output).all() or np.any(
        (output < -1e-12) | (output > 1.0 + 1e-12)
    ):
        raise StyleTransferRuntimeError("Bootstrap AP is outside [0, 1]")
    return output


def _component_bootstrap_inputs(
    account_uids: Sequence[str],
    pairs: Sequence[Mapping[str, Any]],
) -> tuple[np.ndarray, np.ndarray, int]:
    account_component, components = common.positive_components(account_uids, pairs)
    component_ids = tuple(components)
    ordinal = {component: index for index, component in enumerate(component_ids)}
    left = np.asarray(
        [ordinal[account_component[str(row["account_left_uid"])]] for row in pairs],
        dtype=np.int32,
    )
    right = np.asarray(
        [ordinal[account_component[str(row["account_right_uid"])]] for row in pairs],
        dtype=np.int32,
    )
    return left, right, len(component_ids)


def _paired_component_intervals(
    policy: Mapping[str, Any],
    pairs: Sequence[Mapping[str, Any]],
    account_uids: Sequence[str],
    generic_scores: np.ndarray,
    correct_scores: Sequence[np.ndarray],
    permuted_scores: Sequence[np.ndarray],
    weights: Sequence[float],
) -> dict[str, Any]:
    left, right, group_count = _component_bootstrap_inputs(account_uids, pairs)
    repetitions = int(policy["evaluation"]["bootstrap_repetitions"])
    rng = np.random.Generator(
        np.random.PCG64(int(policy["evaluation"]["bootstrap_seed"]))
    )
    draws = rng.integers(
        0, group_count, size=(repetitions, group_count), dtype=np.int32
    )
    labels = [int(row["label"]) for row in pairs]
    generic = bootstrap_global_ap(
        labels, generic_scores, weights, left, right, draws
    )
    correct = np.mean(
        np.vstack(
            [
                bootstrap_global_ap(
                    labels, scores, weights, left, right, draws
                )
                for scores in correct_scores
            ]
        ),
        axis=0,
    )
    permuted = np.mean(
        np.vstack(
            [
                bootstrap_global_ap(
                    labels, scores, weights, left, right, draws
                )
                for scores in permuted_scores
            ]
        ),
        axis=0,
    )

    def interval(values: np.ndarray) -> dict[str, float]:
        return {
            "q025": float(np.quantile(values, 0.025, method="linear")),
            "q975": float(np.quantile(values, 0.975, method="linear")),
        }

    return {
        "unit": "positive_components_plus_singletons",
        "group_count": group_count,
        "repetitions": repetitions,
        "correct_minus_generic": interval(correct - generic),
        "correct_minus_permuted": interval(correct - permuted),
    }


def _per_world_ap(
    pairs: Sequence[Mapping[str, Any]], scores: Sequence[float]
) -> tuple[tuple[str, ...], np.ndarray]:
    groups: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(pairs):
        groups[str(row["world_uid"])].append(index)
    world_uids = tuple(sorted(groups, key=lambda value: value.encode("utf-8")))
    values = []
    scores_array = np.asarray(scores, dtype="<f8")
    for world_uid in world_uids:
        indices = groups[world_uid]
        values.append(
            ranking_metrics(
                [int(pairs[index]["label"]) for index in indices],
                scores_array[indices],
            )["average_precision"]
        )
    return world_uids, np.asarray(values, dtype="<f8")


def _paired_world_intervals(
    policy: Mapping[str, Any],
    pairs: Sequence[Mapping[str, Any]],
    generic_scores: np.ndarray,
    correct_scores: Sequence[np.ndarray],
    permuted_scores: Sequence[np.ndarray],
) -> dict[str, Any]:
    world_uids, generic = _per_world_ap(pairs, generic_scores)
    correct = np.mean(
        np.vstack([_per_world_ap(pairs, scores)[1] for scores in correct_scores]),
        axis=0,
    )
    permuted = np.mean(
        np.vstack([_per_world_ap(pairs, scores)[1] for scores in permuted_scores]),
        axis=0,
    )
    repetitions = int(policy["evaluation"]["bootstrap_repetitions"])
    rng = np.random.Generator(
        np.random.PCG64(int(policy["evaluation"]["bootstrap_seed"]))
    )
    draws = rng.integers(
        0,
        len(world_uids),
        size=(repetitions, len(world_uids)),
        dtype=np.int32,
    )

    def interval(per_world_delta: np.ndarray) -> dict[str, float]:
        means = per_world_delta[draws].mean(axis=1)
        return {
            "q025": float(np.quantile(means, 0.025, method="linear")),
            "q975": float(np.quantile(means, 0.975, method="linear")),
        }

    return {
        "unit": "world",
        "world_count": len(world_uids),
        "repetitions": repetitions,
        "estimand": "mean_per_world_average_precision_delta",
        "correct_minus_generic": interval(correct - generic),
        "correct_minus_permuted": interval(correct - permuted),
    }


def _mean_metric(
    results: Mapping[str, Any],
    model_ids: Sequence[str],
    path: Sequence[str],
) -> float:
    values = []
    for model_id in model_ids:
        value: Any = results[model_id]
        for key in path:
            value = value[key]
        values.append(float(value))
    return float(np.mean(values))


def _file_manifest(root: Path) -> list[dict[str, Any]]:
    records = []
    for path in sorted(
        (value for value in root.rglob("*") if value.is_file()),
        key=lambda value: value.relative_to(root).as_posix(),
    ):
        records.append(
            {
                "path": path.relative_to(root).as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": common.sha256_file(path),
            }
        )
    return records


def run() -> dict[str, Any]:
    policy = common.load_policy()
    static = common.validate_static_inputs()
    torch, SentenceTransformer = require_gpu_runtime(policy)
    implementation_records = implementation_file_records(
        (
            "scripts/step28_v13_v1_13_v9_4_1_v6_style_transfer_common_v2.py",
            "scripts/step28_v13_v1_13_v9_4_1_v6_style_transfer_source_linux_v2.py",
            "scripts/run_step28_v13_v1_13_v9_4_1_v6_style_transfer_source_v2_linux_20260904.sh",
        )
    )
    runtime_versions = runtime_package_versions()
    output_root = ROOT / policy["output_roots"]["source_and_zero_shot"]
    building = output_root.with_name(output_root.name + ".building")
    if output_root.exists() or building.exists():
        raise StyleTransferRuntimeError("Source output path already exists")
    building.mkdir(parents=True)
    started = time.time()
    v6 = common.load_v6(policy)
    v6_streams = {
        uid: row["style_stream"] for uid, row in v6["accounts"].items()
    }

    # Tokenization is fixed by the one pinned LaBSE payload.  The temporary
    # generic encoder is released before epoch selection.
    temporary_encoder, tokenizer = load_encoder(
        policy, SentenceTransformer, torch, int(policy["source_optimization"]["selection_seed"])
    )
    v6_prepared = prepare_stream_chunks(tokenizer, v6_streams, policy)
    del temporary_encoder
    gc.collect()
    torch.cuda.empty_cache()
    selection = select_source_epochs(
        torch, SentenceTransformer, policy, v6, v6_prepared
    )
    _write_json(building / "source_epoch_selection.json", selection)
    selected_epochs = int(selection["selected_epochs"])

    # These labels are opened only after source selection and after the entire
    # protocol is fixed in the current policy.
    v5_streams = common.load_v5_style_streams(policy)
    v5_pairs = common.load_v5_pairs(policy)
    chinese_worlds = common.load_chinese_style_streams(policy, "development")
    chinese_pairs = common.load_chinese_pairs(
        policy, "development", include_labels=True
    )
    chinese_qrels = common.load_chinese_development_qrels(policy)
    flat_chinese_streams = {
        seller_uid: stream
        for sellers in chinese_worlds.values()
        for seller_uid, stream in sellers.items()
    }

    token_encoder, tokenizer = load_encoder(
        policy, SentenceTransformer, torch, int(policy["source_optimization"]["selection_seed"])
    )
    v5_prepared = prepare_stream_chunks(tokenizer, v5_streams, policy)
    chinese_prepared = prepare_stream_chunks(
        tokenizer, flat_chinese_streams, policy
    )
    results: dict[str, Any] = {}
    score_store: dict[str, dict[str, np.ndarray]] = {}
    generic_metrics, generic_scores = evaluate_bundle(
        torch,
        token_encoder,
        tokenizer,
        policy,
        v6,
        v6_prepared,
        v5_prepared,
        v5_pairs,
        chinese_prepared,
        chinese_worlds,
        chinese_pairs,
        chinese_qrels,
    )
    results["generic"] = generic_metrics
    score_store["generic"] = generic_scores
    np.savez_compressed(building / "scores_generic.npz", **generic_scores)
    del token_encoder
    gc.collect()
    torch.cuda.empty_cache()

    model_ids: dict[str, list[str]] = defaultdict(list)
    training_audits = {}
    for arm in (
        "v6_correct",
        "v6_permuted",
        "v6_order_format_mass_ablated",
    ):
        for seed in policy["source_optimization"]["confirmation_seeds"]:
            model_id = f"{arm}_seed_{seed}"
            print(f"开始来源确认模型：{model_id}", flush=True)
            encoder, tokenizer, trace, training_audit = train_source_encoder(
                torch,
                SentenceTransformer,
                policy,
                v6,
                arm,
                int(seed),
                selected_epochs,
                development_prepared=v6_prepared,
            )
            metrics, scores = evaluate_bundle(
                torch,
                encoder,
                tokenizer,
                policy,
                v6,
                v6_prepared,
                v5_prepared,
                v5_pairs,
                chinese_prepared,
                chinese_worlds,
                chinese_pairs,
                chinese_qrels,
            )
            results[model_id] = metrics
            score_store[model_id] = scores
            model_ids[arm].append(model_id)
            training_audits[model_id] = {
                **training_audit,
                "trace": trace,
            }
            np.savez_compressed(building / f"scores_{model_id}.npz", **scores)
            del encoder
            gc.collect()
            torch.cuda.empty_cache()

    audit_pairs = [
        row for row in v6["pairs"] if row["split"] == "synthetic_audit"
    ]
    audit_accounts = tuple(
        uid
        for uid, row in v6["accounts"].items()
        if row["split"] == "synthetic_audit"
    )
    audit_weights = [row["sample_weight"] for row in audit_pairs]
    v6_intervals = _paired_component_intervals(
        policy,
        audit_pairs,
        audit_accounts,
        score_store["generic"]["v6_synthetic_audit"],
        [
            score_store[model_id]["v6_synthetic_audit"]
            for model_id in model_ids["v6_correct"]
        ],
        [
            score_store[model_id]["v6_synthetic_audit"]
            for model_id in model_ids["v6_permuted"]
        ],
        audit_weights,
    )
    v5_accounts = tuple(v5_streams)
    v5_intervals = _paired_component_intervals(
        policy,
        v5_pairs,
        v5_accounts,
        score_store["generic"]["v5_real_holdout"],
        [
            score_store[model_id]["v5_real_holdout"]
            for model_id in model_ids["v6_correct"]
        ],
        [
            score_store[model_id]["v5_real_holdout"]
            for model_id in model_ids["v6_permuted"]
        ],
        np.ones(len(v5_pairs), dtype="<f8"),
    )
    chinese_intervals = _paired_world_intervals(
        policy,
        chinese_pairs,
        score_store["generic"]["chinese_development_zero_shot"],
        [
            score_store[model_id]["chinese_development_zero_shot"]
            for model_id in model_ids["v6_correct"]
        ],
        [
            score_store[model_id]["chinese_development_zero_shot"]
            for model_id in model_ids["v6_permuted"]
        ],
    )
    correct_ids = model_ids["v6_correct"]
    permuted_ids = model_ids["v6_permuted"]
    correct_audit_ap = _mean_metric(
        results, correct_ids, ("v6", "synthetic_audit", "controller_equal", "average_precision")
    )
    correct_audit_auc = _mean_metric(
        results, correct_ids, ("v6", "synthetic_audit", "controller_equal", "roc_auc")
    )
    correct_audit_lift = _mean_metric(
        results, correct_ids, ("v6", "synthetic_audit", "controller_equal", "average_precision_lift")
    )
    generic_audit_ap = float(
        results["generic"]["v6"]["synthetic_audit"]["controller_equal"]["average_precision"]
    )
    permuted_audit_ap = _mean_metric(
        results, permuted_ids, ("v6", "synthetic_audit", "controller_equal", "average_precision")
    )
    source_gate = {
        "correct_mean_weighted_roc_auc": correct_audit_auc,
        "correct_mean_weighted_average_precision": correct_audit_ap,
        "correct_mean_weighted_ap_lift": correct_audit_lift,
        "correct_minus_generic_mean_ap": correct_audit_ap - generic_audit_ap,
        "correct_minus_permuted_mean_ap": correct_audit_ap - permuted_audit_ap,
        "bootstrap": v6_intervals,
    }
    source_gate["confirmed"] = bool(
        correct_audit_auc
        >= float(policy["source_gate"]["correct_mean_weighted_roc_auc_minimum"])
        and correct_audit_lift
        >= float(policy["source_gate"]["correct_mean_weighted_ap_lift_minimum"])
        and source_gate["correct_minus_generic_mean_ap"] > 0.0
        and source_gate["correct_minus_permuted_mean_ap"] > 0.0
        and v6_intervals["correct_minus_generic"]["q025"] > 0.0
        and v6_intervals["correct_minus_permuted"]["q025"] > 0.0
    )
    generic_v5_ap = float(
        results["generic"]["v5_real_holdout"]["ranking"]["average_precision"]
    )
    correct_v5_ap = _mean_metric(
        results,
        correct_ids,
        ("v5_real_holdout", "ranking", "average_precision"),
    )
    permuted_v5_ap = _mean_metric(
        results,
        permuted_ids,
        ("v5_real_holdout", "ranking", "average_precision"),
    )
    generic_chinese_ap = float(
        results["generic"]["chinese_development_zero_shot"]["world_equal_ranking"][
            "average_precision"
        ]
    )
    correct_chinese_ap = _mean_metric(
        results,
        correct_ids,
        (
            "chinese_development_zero_shot",
            "world_equal_ranking",
            "average_precision",
        ),
    )
    permuted_chinese_ap = _mean_metric(
        results,
        permuted_ids,
        (
            "chinese_development_zero_shot",
            "world_equal_ranking",
            "average_precision",
        ),
    )
    comparisons = {
        "source_gate": source_gate,
        "v5_real_holdout": {
            "correct_mean_average_precision": correct_v5_ap,
            "correct_minus_generic_mean_ap": correct_v5_ap - generic_v5_ap,
            "correct_minus_permuted_mean_ap": correct_v5_ap - permuted_v5_ap,
            "bootstrap": v5_intervals,
            "interpretation": "support_only_never_stops_chinese_stage",
        },
        "chinese_development_zero_shot": {
            "primary_aggregation": "world_equal",
            "correct_mean_world_equal_average_precision": correct_chinese_ap,
            "correct_minus_generic_mean_world_equal_ap": (
                correct_chinese_ap - generic_chinese_ap
            ),
            "correct_minus_permuted_mean_world_equal_ap": (
                correct_chinese_ap - permuted_chinese_ap
            ),
            "pooled_average_precision_descriptive": {
                "generic": float(
                    results["generic"]["chinese_development_zero_shot"]
                    ["pooled_ranking"]["average_precision"]
                ),
                "correct_mean": _mean_metric(
                    results,
                    correct_ids,
                    (
                        "chinese_development_zero_shot",
                        "pooled_ranking",
                        "average_precision",
                    ),
                ),
                "permuted_mean": _mean_metric(
                    results,
                    permuted_ids,
                    (
                        "chinese_development_zero_shot",
                        "pooled_ranking",
                        "average_precision",
                    ),
                ),
            },
            "bootstrap": chinese_intervals,
            "interpretation": "development_candidate_not_blind_confirmation",
        },
    }
    _write_json(building / "model_metrics.json", results)
    _write_json(building / "training_audits.json", training_audits)
    _write_json(building / "comparisons.json", comparisons)
    run_summary = {
        "status": "COMPLETED_V6_STYLE_SOURCE_AND_CHINESE_DEVELOPMENT_ZERO_SHOT",
        "policy_canonical_self_hash": policy["canonical_self_hash"],
        "selected_source_epochs": selected_epochs,
        "models": ["generic"] + [value for arm in model_ids.values() for value in arm],
        "source_gate_confirmed": source_gate["confirmed"],
        "elapsed_seconds": time.time() - started,
        "audit_a_truth_reads": 0,
        "audit_b_truth_reads": 0,
        "static_input_validation": static,
    }
    _write_json(building / "run_summary.json", run_summary)
    manifest = {
        "version": policy["version"],
        "status": run_summary["status"],
        "policy_canonical_self_hash": policy["canonical_self_hash"],
        "files": _file_manifest(building),
        "runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
            "gpu": torch.cuda.get_device_name(0),
            "packages": runtime_versions,
        },
        "implementation_files": implementation_records,
        "audit_a_truth_reads": 0,
        "audit_b_truth_reads": 0,
    }
    _write_json(building / "manifest.json", manifest)
    os.replace(building, output_root)
    return run_summary


def smoke() -> dict[str, Any]:
    policy = common.load_policy()
    torch, SentenceTransformer = require_gpu_runtime(policy)
    encoder, tokenizer = load_encoder(
        policy, SentenceTransformer, torch, int(policy["source_optimization"]["selection_seed"])
    )
    streams = {
        "a": " ".join(["W5, N2!"] * 105),
        "b": " ".join(["W4. N3?"] * 103),
        "c": " ".join(["W7; N1."] * 104),
        "d": " ".join(["W3: N4!"] * 102),
    }
    prepared = prepare_stream_chunks(tokenizer, streams, policy)
    account_uids = ("a", "b", "c", "d")
    head = _source_head(torch, policy)
    encoder.zero_grad(set_to_none=True)
    head.zero_grad(set_to_none=True)
    with torch.no_grad():
        frozen = encode_accounts(
            torch, encoder, tokenizer, prepared, account_uids, batch_size=4
        )
    leaf = frozen.detach().requires_grad_(True)
    cosines = torch.stack(
        ((leaf[0] * leaf[1]).sum(), (leaf[2] * leaf[3]).sum())
    )
    labels = torch.tensor([1.0, 0.0], device="cuda:0")
    loss = torch.nn.functional.binary_cross_entropy_with_logits(
        head(cosines), labels
    )
    loss.backward()
    if leaf.grad is None or not torch.isfinite(leaf.grad).all():
        raise StyleTransferRuntimeError("GPU smoke produced no leaf gradient")
    head_gradient_before_vjp = (
        float(head.log_scale.grad.detach().cpu()),
        float(head.bias.grad.detach().cpu()),
    )
    graph = encode_accounts(
        torch, encoder, tokenizer, prepared, account_uids, batch_size=4
    )
    torch.sum(graph.float() * leaf.grad.detach().float()).backward()
    head_gradient_after_vjp = (
        float(head.log_scale.grad.detach().cpu()),
        float(head.bias.grad.detach().cpu()),
    )
    if head_gradient_before_vjp != head_gradient_after_vjp:
        raise StyleTransferRuntimeError(
            "Two-pass VJP incorrectly changed the source head gradient"
        )
    gradient = sum(
        float(parameter.grad.detach().abs().sum().cpu())
        for parameter in encoder.parameters()
        if parameter.grad is not None
    )
    if not math.isfinite(gradient) or gradient <= 0.0:
        raise StyleTransferRuntimeError("GPU smoke produced no encoder gradient")
    return {
        "status": "PASSED_V6_STYLE_TRANSFER_GPU_SMOKE",
        "embedding_shape": list(graph.shape),
        "encoder_gradient_l1": gradient,
        "head_gradient_unchanged_by_vjp": True,
        "audit_a_truth_reads": 0,
        "audit_b_truth_reads": 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("validate", "smoke", "run"))
    arguments = parser.parse_args()
    if arguments.command == "validate":
        result = common.validate_static_inputs()
    elif arguments.command == "smoke":
        result = smoke()
    else:
        result = run()
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
