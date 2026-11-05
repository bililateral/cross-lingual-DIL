"""Unmodified pure metric definitions copied from the project; no data imports."""

from __future__ import annotations
import math
from typing import Any
import numpy as np

KS = (1, 3, 5, 10)
CURVE_KEYS = ("average_precision", "trapezoidal_pr_auc", "roc_auc", "recall_at_fpr_1pct")
CLASS_KEYS = CURVE_KEYS + ("brier", "log_loss", "precision", "recall", "f1",
                         "specificity", "balanced_accuracy", "mcc")
RETRIEVAL_KEYS = ("map", "mrr") + tuple(f"recall_at_{k}" for k in KS) + tuple(f"ndcg_at_{k}" for k in KS)
COLUMNS = CLASS_KEYS + RETRIEVAL_KEYS

def curve_metrics(y: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    y, scores = np.asarray(y), np.asarray(scores, dtype=np.float64)
    if (y.ndim != 1 or y.shape != scores.shape or not len(y) or not np.isin(y, [0, 1]).all()
            or not np.isfinite(scores).all() or not 0 < y.sum() < len(y)):
        raise ValueError("Curve inputs need aligned finite scores and both classes")
    order = np.argsort(-scores, kind="stable")
    ordered = scores[order]
    end = np.r_[ordered[1:] != ordered[:-1], True]
    tp = np.cumsum(y[order], dtype=np.float64)[end]
    fp = np.cumsum(1 - y[order], dtype=np.float64)[end]
    recall, fpr = tp / y.sum(), fp / (len(y) - y.sum())
    precision = tp / (tp + fp)
    dr = np.diff(np.r_[0., recall])
    return {"average_precision": float(dr @ precision),
            "trapezoidal_pr_auc": float(dr @ ((precision + np.r_[1., precision[:-1]]) / 2)),
            "roc_auc": float(np.diff(np.r_[0., fpr]) @ ((recall + np.r_[0., recall[:-1]]) / 2)),
            "recall_at_fpr_1pct": float(np.max(np.r_[0., recall[fpr <= .01]]))}

def confusion_metrics(tp: float, fp: float, fn: float, tn: float) -> dict[str, float]:
    precision = tp / (tp + fp) if tp + fp else 0.
    recall = tp / (tp + fn) if tp + fn else 0.
    specificity = tn / (tn + fp) if tn + fp else 0.
    denominator = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return {"precision": precision, "recall": recall, "specificity": specificity,
            "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.,
            "balanced_accuracy": (recall + specificity) / 2,
            "mcc": (tp * tn - fp * fn) / denominator if denominator else 0.}

def classification(y: np.ndarray, scores: np.ndarray, threshold: float = 0.) -> dict[str, Any]:
    result: dict[str, Any] = curve_metrics(y, scores)
    scores = np.asarray(scores, dtype=np.float64)
    probability = np.exp(-np.logaddexp(0., -scores))
    clipped = np.clip(probability, 1e-15, 1 - 1e-15)
    result["brier"] = float(np.mean((probability - y) ** 2))
    result["log_loss"] = float(-np.mean(y * np.log(clipped) + (1 - y) * np.log1p(-clipped)))
    predicted, positive = scores >= threshold, y == 1
    tp, fp = int(np.sum(predicted & positive)), int(np.sum(predicted & ~positive))
    fn, tn = int(np.sum(~predicted & positive)), int(np.sum(~predicted & ~positive))
    result.update(confusion_metrics(tp, fp, fn, tn))
    result["confusion"] = {"tp": tp, "fp": fp, "fn": fn, "tn": tn}
    return result

def retrieval(y: np.ndarray, scores: np.ndarray, sellers: int) -> np.ndarray:
    """Complete sorted K_n edges; score ties use ascending public seller order."""
    y, scores = np.asarray(y), np.asarray(scores, dtype=np.float64)
    pairs = sellers * (sellers - 1) // 2
    if y.shape != scores.shape or y.ndim != 2 or y.shape[1] != pairs or not np.isfinite(scores).all():
        raise ValueError("Retrieval rows differ from complete worlds")
    if not np.isin(y, [0, 1]).all():
        raise ValueError("Nonbinary relevance")
    left, right = np.triu_indices(sellers, 1)
    adjacency = np.full((len(y), sellers, sellers), -np.inf)
    relevant = np.zeros((len(y), sellers, sellers), dtype=np.uint8)
    adjacency[:, left, right] = adjacency[:, right, left] = scores
    relevant[:, left, right] = relevant[:, right, left] = y
    order = np.argsort(-adjacency, axis=2, kind="stable")[:, :, :sellers - 1]
    ranked = np.take_along_axis(relevant, order, axis=2)
    total = relevant.sum(axis=2)
    if not np.all((total >= 1) & (total <= 2)):
        raise ValueError("Each qualified query must have one or two relevant candidates")
    ranks = np.arange(1, sellers)
    ap = (np.cumsum(ranked, axis=2) * ranked / ranks).sum(axis=2) / total
    rr = 1. / (np.argmax(ranked, axis=2) + 1)
    values = [ap, rr]
    values.extend(ranked[:, :, :k].sum(axis=2) / total for k in KS)
    discounts = 1. / np.log2(ranks + 1)
    for k in KS:
        ideal = np.asarray([discounts[:min(k, int(n))].sum() for n in total.ravel()]).reshape(total.shape)
        values.append((ranked[:, :, :k] * discounts[:k]).sum(axis=2) / ideal)
    return np.stack([v.mean(axis=1) for v in values], axis=1)

def group_metrics(labels: np.ndarray, scores: np.ndarray) -> tuple[np.ndarray, list[dict]]:
    results = [classification(y, s) for y, s in zip(labels, scores, strict=True)]
    numeric = np.array([[row[k] for k in CLASS_KEYS] for row in results])
    return np.column_stack((numeric, retrieval(labels, scores, 28))), [r["confusion"] for r in results]

def query_metrics(labels, scores):
    """Preserve per-query detail; aggregate agrees with the retained metric core."""
    left, right = np.triu_indices(28, 1)
    adjacency = np.full((len(labels), 28, 28), -np.inf)
    relevant = np.zeros((len(labels), 28, 28), np.uint8)
    adjacency[:, left, right] = adjacency[:, right, left] = scores
    relevant[:, left, right] = relevant[:, right, left] = labels
    order = np.argsort(-adjacency, axis=2, kind="stable")[:, :, :27]
    ranked = np.take_along_axis(relevant, order, axis=2)
    total = relevant.sum(2)
    if not np.all((total >= 1) & (total <= 2)):
        raise ValueError("Query relevance differs")
    ranks = np.arange(1, 28)
    values = [(np.cumsum(ranked, axis=2)*ranked/ranks).sum(2)/total, 1./(np.argmax(ranked, axis=2)+1)]
    values.extend(ranked[:, :, :k].sum(2)/total for k in (1, 3, 5, 10))
    discount = 1./np.log2(ranks+1)
    for k in (1, 3, 5, 10):
        ideal = np.asarray([discount[:min(k, int(n))].sum() for n in total.ravel()]).reshape(total.shape)
        values.append((ranked[:, :, :k]*discount[:k]).sum(2)/ideal)
    return np.stack(values, axis=2)

def undefined_metrics(counts):
    """Only precision/MCC can be undefined for the contracted two-class groups."""
    mask = np.zeros((len(counts), 22), dtype=bool)
    for i, r in enumerate(counts):
        tp, fp, fn, tn = (r[k] for k in ("tp", "fp", "fn", "tn"))
        mask[i, 6] = tp+fp == 0
        mask[i, 11] = (tp+fp)*(tp+fn)*(tn+fp)*(tn+fn) == 0
    return mask
