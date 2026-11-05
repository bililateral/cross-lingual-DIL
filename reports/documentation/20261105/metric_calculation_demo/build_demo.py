"""Construct artificial symmetric scores; evaluate them with copied pure metric functions.

No model, real labels, dataset text, calibration fitting, or training is used.
The requested MAP is a constraint on generated scores, not an edited metric.
"""
from __future__ import annotations

import time

START = time.monotonic()
import hashlib
import json
import os
from pathlib import Path
import resource
import signal
import sys

import numpy as np

import metric_functions as metrics

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "output"
TARGETS = (0.532146725, 0.524377507, 0.528396158)
POOL_SIZE = 32768
GROUPS = 1080


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def nullable(value):
    if isinstance(value, np.ndarray):
        return nullable(value.tolist())
    if isinstance(value, list):
        return [nullable(x) for x in value]
    if isinstance(value, dict):
        return {k: nullable(v) for k, v in value.items()}
    if isinstance(value, (float, np.floating)):
        return None if not np.isfinite(value) else float(value)
    return value


def write_json(path, obj):
    path.write_text(json.dumps(nullable(obj), ensure_ascii=False, indent=2,
                               allow_nan=False) + "\n", encoding="utf-8")


def save_array(name, value):
    path = OUTPUT / name
    np.save(path, value, allow_pickle=False)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha(path),
            "shape": list(value.shape), "dtype": str(value.dtype)}


def deadline(signum, frame):
    raise TimeoutError("600-second CPU-demo wall-time limit reached")


def choose(rng, maps, target):
    """Choose 1080 distinct templates, reserving two for a discrete two-sum fit."""
    low = rng.permutation(np.flatnonzero(maps < target)).tolist()
    high = rng.permutation(np.flatnonzero(maps >= target)).tolist()
    selected, total = [], 0.0
    for i in range(GROUPS - 2):
        side = high if total < (i + 1) * target else low
        if not side:
            raise ValueError("Artificial template pool does not bracket the target")
        idx = side.pop()
        selected.append(idx)
        total += float(maps[idx])
    available = np.ones(len(maps), dtype=bool)
    available[selected] = False
    ids = np.flatnonzero(available)
    ids = ids[np.argsort(maps[ids], kind="stable")]
    desired = target * GROUPS - total
    left, right = 0, len(ids) - 1
    best = (float("inf"), None, None)
    while left < right:
        i, j = int(ids[left]), int(ids[right])
        combined = float(maps[i] + maps[j])
        error = abs(combined - desired)
        if error < best[0]:
            best = (error, i, j)
        if error / GROUPS < 1e-11:
            break
        if combined < desired:
            left += 1
        else:
            right -= 1
    if best[1] is None:
        raise ValueError("No distinct terminal template pair")
    selected.extend(best[1:])
    rng.shuffle(selected)
    selected = np.asarray(selected, dtype=np.int64)
    if len(set(selected.tolist())) != GROUPS:
        raise ValueError("Repeated templates in selected batch")
    return selected


def calculate(batch, target, labels):
    rng = np.random.default_rng(2026110500 + batch)
    scores = np.empty((POOL_SIZE, 378), dtype=np.float64)
    maps = np.empty(POOL_SIZE, dtype=np.float64)
    # Entire undirected score vectors are constructed before metrics are calculated.
    # Varying positive separation gives the selection a broad range of legal rankings.
    for start in range(0, POOL_SIZE, 256):
        end = min(start + 256, POOL_SIZE)
        size = end - start
        separation = np.clip(rng.normal(1.85, 0.60, (size, 1)), 0.25, 3.8)
        block = rng.normal(0.0, 1.0, (size, 378)) + separation * labels - 3.0
        scores[start:end] = block
        truth = np.broadcast_to(labels, block.shape)
        maps[start:end] = metrics.query_metrics(truth, block)[:, :, 0].mean(1)
    selected = choose(rng, maps, target)
    chosen = scores[selected].copy()
    del scores
    truth = np.broadcast_to(labels, chosen.shape)
    values, counts = metrics.group_metrics(truth, chosen)
    queries = metrics.query_metrics(truth, chosen)
    if (values.shape != (GROUPS, 22)
            or queries.shape != (GROUPS, 28, 10)
            or not np.isfinite(chosen).all()
            or not np.allclose(values[:, 12:], queries.mean(1), atol=1e-12, rtol=1e-12)
            or any(c["tp"] + c["fn"] != 20 or c["tn"] + c["fp"] != 358 for c in counts)):
        raise ValueError("Scores, query metrics, group metrics or confusion counts disagree")
    values[metrics.undefined_metrics(counts)] = np.nan
    absolute = values.mean(0)
    actual = float(absolute[12])
    if abs(actual - float(maps[selected].mean())) > 1e-12:
        raise ValueError("MAP selection and full evaluation disagree")
    per_run = {}
    for order_index, order in enumerate(("ABC", "BCA", "CAB")):
        for seed_index, seed in enumerate((4, 5, 6)):
            begin = (order_index * 3 + seed_index) * 120
            cells = values[begin:begin + 120].reshape(3, 40, 22).mean(1)
            per_run[f"{order}_s{seed}"] = {"final_cells": cells,
                                              "final": cells.mean(0)}
    result = {
        "target_map": target,
        "actual_map": actual,
        "absolute_error": abs(actual - target),
        "matches_at_9_decimals": f"{actual:.9f}" == f"{target:.9f}",
        "seed": 2026110500 + batch,
        "candidate_pool_size": POOL_SIZE,
        "selected_template_indices": selected.tolist(),
        "absolute": absolute,
        "per_run": per_run,
        "undefined_group_counts": np.isnan(values).sum(0).tolist(),
        "scores": save_array(f"H_{batch}_scores.npy", chosen),
        "group_metrics": save_array(f"H_{batch}_metrics.npy", values),
        "query_metrics": save_array(f"H_{batch}_queries.npy", queries),
        "confusion_totals": {key: sum(x[key] for x in counts) for key in ("tp", "fp", "fn", "tn")},
        "query_to_group_max_abs_error": float(np.max(np.abs(values[:, 12:] - queries.mean(1)))),
        "elapsed_seconds_cumulative": time.monotonic() - START,
    }
    write_json(OUTPUT / f"H_{batch}_result.json", result)
    print(json.dumps({"batch": batch, "target": target, "actual": actual,
                      "matches_at_9_decimals": result["matches_at_9_decimals"],
                      "elapsed": time.monotonic() - START}), flush=True)
    return result


def main():
    signal.signal(signal.SIGALRM, deadline)
    signal.setitimer(signal.ITIMER_REAL, max(0.01, 600 - (time.monotonic() - START)))
    OUTPUT.mkdir()
    cliques = [list(range(i, i + 2)) for i in range(0, 16, 2)]
    cliques += [list(range(i, i + 3)) for i in range(16, 28, 3)]
    owner = np.empty(28, dtype=np.int64)
    for i, members in enumerate(cliques):
        owner[members] = i
    left, right = np.triu_indices(28, 1)
    labels = (owner[left] == owner[right]).astype(np.uint8)
    if int(labels.sum()) != 20:
        raise ValueError("Artificial relation graph differs")
    label_record = save_array("artificial_labels.npy", labels)
    result = {"kind": "SYNTHETIC_NUMERICAL_DEMO_NOT_MODEL_EVALUATION",
              "status": "RUNNING", "targets": TARGETS,
              "columns": list(metrics.COLUMNS), "groups_per_batch": GROUPS,
              "grouping": "3 orders x 3 seeds x 3 final positions x 40 groups",
              "artificial_cliques": cliques, "labels": label_record,
              "scoring": "independent normal undirected scores; positive mean shift; offset -3; identity probability logit",
              "probability_and_threshold": "sigmoid of generated scores; threshold logit 0; no real stage-cal mapping",
              "real_label_reads": 0, "model_loads": 0, "training_updates": 0,
              "gpu_calls": 0, "cpu_affinity": sorted(os.sched_getaffinity(0)),
              "python": sys.version, "numpy": np.__version__,
              "source_sha256": {p.name: sha(p) for p in (Path(__file__), ROOT / "metric_functions.py", ROOT / "source_functions.json")},
              "batches": {}}
    try:
        for i, target in enumerate(TARGETS, 1):
            result["batches"][str(i)] = calculate(i, target, labels)
        result["status"] = "COMPLETE" if all(x["matches_at_9_decimals"] for x in result["batches"].values()) else "COMPLETE_WITH_TARGET_ERROR"
    except BaseException as exc:
        result["status"] = "FAILED"
        result["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        result["elapsed_seconds"] = time.monotonic() - START
        result["cpu_seconds"] = time.process_time()
        result["max_rss_kib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        write_json(OUTPUT / "result.json", result)


if __name__ == "__main__":
    main()
