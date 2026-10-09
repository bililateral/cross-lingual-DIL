"""Saved-matrix continual statistics; no labels, text, weights or fitting entry."""
from __future__ import annotations

import csv
import io
from pathlib import Path
from typing import Any

import numpy as np

import step28_bge_continual as method

data, metrics = method.data, method.metrics
ENDPOINTS = ("O", "N", "Z", "F_first", "F", "G", "final_all")
RANK_COLUMNS = tuple(metrics.COLUMNS.index(name) for name in
                     (*metrics.CURVE_KEYS, *metrics.RETRIEVAL_KEYS))


def point_name(order: str, arm: str, stage: int) -> str:
    if order not in method.ORDERS or arm not in method.METHODS or stage not in (1, 2, 3):
        raise ValueError("Unknown logical path point")
    return order + "_shared" if stage == 1 or arm == "frozen" else f"{order}_{arm}_stage{stage}"


def expected_points() -> list[str]:
    return [name for order in method.ORDERS for name in
            [order + "_shared", *(f"{order}_{arm}_stage{stage}"
                                  for arm in method.UPDATED for stage in (2, 3))]]


def domain_rows(domains: list[str]) -> list[np.ndarray]:
    rows = [np.flatnonzero(np.asarray(domains) == domain) for domain in "ABC"]
    if len(domains) != 60 or any(len(row) != 20 for row in rows):
        raise ValueError("Need the same 20 valid groups per actual domain")
    return rows


def bootstrap_draws() -> np.ndarray:
    return np.random.Generator(np.random.PCG64(20260930)).integers(0, 20, size=(5000, 3, 20))


def read_matrix(root: Path, record: dict) -> np.ndarray:
    path = data.verify(root / record["path"], record)
    result = np.load(path, allow_pickle=False)
    if result.shape != (60, 22) or result.dtype != np.float64 or not np.isfinite(result).all():
        raise ValueError("Complete finite 60x22 float64 matrix required")
    return result


def read_counts(root: Path, record: dict) -> list[dict]:
    values = data.read_json(data.verify(root / record["path"], record))
    if len(values) != 60:
        raise ValueError("Incomplete group confusion counts")
    for row in values:
        if (any(type(row[k]) is not int or row[k] < 0 for k in ("tp", "fp", "fn", "tn"))
                or row["tp"] + row["fn"] != 20 or row["fp"] + row["tn"] != 358):
            raise ValueError("Confusion counts violate the complete 378-pair group")
    return values


def write_once(path: Path, payload: bytes) -> None:
    if path.exists():
        if path.read_bytes() != payload:
            raise FileExistsError("Refusing to replace a published result with different bytes: " + str(path))
    else:
        path.write_bytes(payload)


def stage_table(arrays: dict, initial: np.ndarray, rows: list[np.ndarray]) -> bytes:
    """Explicit scientific indices, including unchanged logical frozen endpoints."""
    stream = io.StringIO(newline="")
    writer = csv.writer(stream, lineterminator="\n")
    writer.writerow(("metric", "output_role", "actual_domain", "arrival_index", "stage",
                     "order", "seed", "method", "source_point", "group_macro"))
    for order in method.ORDERS:
        for arrival, domain in enumerate(order, 1):
            r = rows["ABC".index(domain)]
            for k, metric in enumerate(metrics.COLUMNS):
                writer.writerow((metric, "raw", domain, arrival, 0, order, "s0", "initial",
                                 "initial", repr(float(initial[r, k].mean()))))
            for arm in method.METHODS:
                for stage in (1, 2, 3):
                    point = point_name(order, arm, stage)
                    for role in method.ROLES:
                        for k, metric in enumerate(metrics.COLUMNS):
                            writer.writerow((metric, role, domain, arrival, stage, order, "s0", arm,
                                             point, repr(float(arrays[point][role][r, k].mean()))))
    return stream.getvalue().encode("utf-8")


def endpoint_fields(arrays: dict, domains: list[str], arm: str, role: str,
                    endpoint: str) -> np.ndarray:
    """(order, actual_domain, group_within_actual_domain, metric) weighted fields.

    Domain sums, not another mean over domains: term coefficients already sum to
    one for absolute endpoints. Shared groups remain aligned across every path.
    """
    if role not in (*method.ROLES, "primary") or endpoint not in ENDPOINTS:
        raise ValueError("Unknown metric role or endpoint")
    terms = {
        "O": ((3, 1, .5), (3, 2, .5)),
        "N": ((2, 2, .5), (3, 3, .5)), "Z": ((3, 3, 1.),),
        "F_first": ((1, 1, 1.), (3, 1, -1.)),
        "F": ((1, 1, .5), (3, 1, -.5), (2, 2, .5), (3, 2, -.5)),
        "G": ((2, 2, .5), (1, 2, -.5), (3, 3, .5), (2, 3, -.5)),
        "final_all": ((3, 1, 1 / 3), (3, 2, 1 / 3), (3, 3, 1 / 3)),
    }[endpoint]
    rows = domain_rows(domains)
    fields = np.zeros((3, 3, 20, 22), dtype=np.float64)
    for i, order in enumerate(method.ORDERS):
        for stage, arrival, weight in terms:
            key = point_name(order, arm, stage)
            if role == "primary":
                values = arrays[key]["stage-cal"].copy()
                values[:, RANK_COLUMNS] = arrays[key]["raw"][:, RANK_COLUMNS]
            else:
                values = arrays[key][role]
            d = "ABC".index(order[arrival - 1])
            fields[i, d] += weight * values[rows[d]]
    if endpoint in ("F_first", "F", "G"):
        fields[..., [metrics.COLUMNS.index(n) for n in ("brier", "log_loss")]] *= -1
    return fields


def summarize_field(fields: np.ndarray, draws: np.ndarray) -> dict:
    if (fields.shape != (3, 3, 20, 22) or not np.isfinite(fields).all()
            or draws.shape != (5000, 3, 20) or draws.dtype.kind not in "iu"
            or np.any((draws < 0) | (draws >= 20))):
        raise ValueError("Bad fields or actual-domain draws")
    mean_field = fields.mean(0)
    per_order = fields.mean(2).sum(1)
    point = mean_field.mean(1).sum(0)
    boot = sum(mean_field[d][draws[:, d]].mean(1) for d in range(3))
    intervals = np.quantile(boot, [.025, .975], axis=0, method="linear")
    return {name: {"mean": float(point[k]),
                   "per_order": dict(zip(method.ORDERS, per_order[:, k].tolist())),
                   "conditional_95pct_interval": intervals[:, k].tolist()}
            for k, name in enumerate(metrics.COLUMNS)}


def comparison_checks(delta: dict, versus_raw: dict) -> dict:
    old, new, latest = delta["O"], delta["N"], delta["Z"]
    checks = {"old_map_improves": old["map"]["mean"] > 0,
              "old_map_interval_above_zero": old["map"]["conditional_95pct_interval"][0] > 0,
              "old_recall5_improves": old["recall_at_5"]["mean"] > 0,
              "new_map_non_decrease": new["map"]["mean"] >= 0,
              "new_recall5_non_decrease": new["recall_at_5"]["mean"] >= 0}
    for endpoint in ("O", "N"):
        for name, sign in ranking_guards().items():
            checks[f"{endpoint}_{name}_non_degradation"] = sign * delta[endpoint][name]["mean"] >= 0
        for name in ("brier", "log_loss"):
            checks[f"{endpoint}_{name}_against_raw_reference"] = versus_raw[endpoint][name]["mean"] <= 0
    for name, sign in {"map": 1, "recall_at_5": 1, **ranking_guards()}.items():
        checks[f"Z_{name}_non_degradation"] = sign * latest[name]["mean"] >= 0
    return {"pilot_observed_checks_pass": all(checks.values()), "checks": checks,
            "failed": [key for key, ok in checks.items() if not ok],
            "positive_old_map_orders": sum(v > 0 for v in old["map"]["per_order"].values()),
            "future_three_seed_qualification": "NOT_EVALUATED_SINGLE_SEED_PILOT",
            "scope": "Observed mean protection, not per-domain guarantees or population noninferiority"}


def ranking_guards() -> dict[str, int]:
    return {"average_precision": 1, "roc_auc": 1, "brier": -1, "log_loss": -1}


def forgetting_checks(first: dict, gain: dict) -> dict:
    matched = [o for o in method.ORDERS if first["map"]["per_order"][o] > 0
               and gain["map"]["per_order"][o] > 0]
    checks = {"seq_first_MAP_loss_positive": first["map"]["mean"] > 0,
              "seq_first_MAP_loss_interval_above_zero": first["map"]["conditional_95pct_interval"][0] > 0,
              "same_order_path_forgetting_and_new_learning": len(matched) >= 2}
    return {"established": all(checks.values()), "checks": checks, "matched_orders": matched,
            "claim": "Same complete path, not necessarily same transition or gradient conflict",
            "failure_scope": "Does not rule out intermediate or second-domain forgetting"}


def finalize(root: Path) -> dict:
    """Recoverable after full collection: uses only saved matrices and counts."""
    collected = data.read_json(root / "collected.json")
    if collected["status"] != "ALL_PILOT_MATRICES_SAVED_BEFORE_COMPARISONS":
        raise ValueError("Complete collection required")
    if collected["metric_columns"] != list(metrics.COLUMNS):
        raise ValueError("Metric schema changed")
    domains = collected["domains"]
    if len(collected["group_ids"]) != 60 or len(set(collected["group_ids"])) != 60:
        raise ValueError("Repeated or incomplete group identities")
    rows = domain_rows(domains)
    if set(collected["points"]) != set(expected_points()):
        raise ValueError("Incomplete stage endpoint set")
    arrays = {point: {role: read_matrix(root, record["matrix"])
                      for role, record in info.items()}
              for point, info in collected["points"].items()}
    for point, roles in arrays.items():
        if set(roles) != set(method.ROLES):
            raise ValueError("Incomplete output roles")
        for role in ("stage-cal", "first-cal"):
            if not np.array_equal(roles[role][:, RANK_COLUMNS], roles["raw"][:, RANK_COLUMNS]):
                raise ValueError("Calibrated ranking/curve metrics changed")
    initial = read_matrix(root, collected["initial"]["matrix"])
    read_counts(root, collected["initial"]["counts"])
    counts_by_point = {point: {role: read_counts(root, info["counts"]) for role, info in entries.items()}
                       for point, entries in collected["points"].items()}
    draws = bootstrap_draws()
    buffer = io.BytesIO()
    np.save(buffer, draws, allow_pickle=False)
    write_once(root / "bootstrap_draws.npy", buffer.getvalue())
    write_once(root / "stage_metrics.csv", stage_table(arrays, initial, rows))
    fields = {arm: {role: {name: endpoint_fields(arrays, domains, arm, role, name)
                          for name in ENDPOINTS} for role in (*method.ROLES, "primary")}
              for arm in method.METHODS}
    endpoints = {arm: {role: {name: summarize_field(value, draws) for name, value in entries.items()}
                       for role, entries in variants.items()} for arm, variants in fields.items()}
    comparisons = {}
    for candidate, reference in (("er", "seq"), ("logit", "er")):
        delta = {name: summarize_field(fields[candidate]["primary"][name]
                                       - fields[reference]["primary"][name], draws) for name in ENDPOINTS}
        against_raw = {name: summarize_field(fields[candidate]["primary"][name]
                                             - fields[reference]["raw"][name], draws) for name in ("O", "N")}
        comparisons[candidate + "_minus_" + reference] = {
            "primary": delta, "against_raw_reference": against_raw,
            "interpretation": comparison_checks(delta, against_raw)}
    absolute = {}
    for point, roles in arrays.items():
        absolute[point] = {}
        for role, values in roles.items():
            counts = counts_by_point[point][role]
            absolute[point][role] = {
                "macro_all": dict(zip(metrics.COLUMNS, values.mean(0).tolist())),
                "macro_by_domain": {d: dict(zip(metrics.COLUMNS, values[r].mean(0).tolist()))
                                    for d, r in zip("ABC", rows, strict=True)},
                "pooled_fixed_half_classification": method.base.fixed_classification(counts, domains)}
    first_learning = {}
    for order in method.ORDERS:
        r = rows["ABC".index(order[0])]
        first = arrays[order + "_shared"]["raw"][r]
        first_learning[order] = {
            "actual_domain": order[0], "initial_raw": dict(zip(metrics.COLUMNS, initial[r].mean(0).tolist())),
            "first_raw": dict(zip(metrics.COLUMNS, first.mean(0).tolist())),
            "raw_first_minus_initial": dict(zip(metrics.COLUMNS, (first - initial[r]).mean(0).tolist()))}
    result = {"status": "COMPLETE_BGE_CONTINUAL_DIAGNOSTIC", "source_files": collected["source_files"],
              "analysis_sources": [data.record(data.ROOT / name, data.ROOT) for name in
                                   ("scripts/step28_bge_continual_evaluate.py", "scripts/step28_bge_continual.py",
                                    "scripts/step28_continual_population_evaluate.py", "scripts/step28_chinese_base.py")],
              "collected": data.record(root / "collected.json", root),
              "stage_metrics": data.record(root / "stage_metrics.csv", root),
              "draws": data.record(root / "bootstrap_draws.npy", root), "endpoints": endpoints,
              "comparisons": comparisons, "absolute_stage_results": absolute,
              "initial_to_first_learning": first_learning,
              "seq_forgetting": forgetting_checks(endpoints["seq"]["raw"]["F_first"],
                                                   endpoints["seq"]["raw"]["G"]),
              "scope": "Fixed s0 conditional development diagnostic; no novel-method qualification or test",
              "next": "REVIEW_COMPLETE_RESULTS_BEFORE_ANY_NEW_STAGE"}
    write_once(root / "evaluation.json", data.json_bytes(result))
    return result
