"""Saved-matrix ER development comparisons; no formal label/model/text access."""
from __future__ import annotations

import csv
import io
from pathlib import Path

import numpy as np

import step28_er_weight as method
import step28_bge_continual_evaluate as previous

data, metrics = method.data, method.metrics


def select_configuration(endpoints: dict, comparisons: dict) -> dict:
    eligible = [arm for arm in method.ARMS
                if comparisons[arm + "_minus_er"]["interpretation"]["pilot_observed_checks_pass"]]
    selected = max(eligible, key=lambda arm: (endpoints[arm]["primary"]["O"]["map"]["mean"],
                                            endpoints[arm]["primary"]["O"]["recall_at_5"]["mean"],
                                            method.ARMS[arm])) if eligible else "er"
    return {"selected": selected, "history_weight": method.ARMS.get(selected, 1.),
            "eligible": eligible, "fallback_used": not eligible,
            "criterion": "All 23 versus original ER, then O MAP, O Recall@5, larger lambda",
            "scope": "Developed-valid configuration selection, not independent confirmation",
            "additional_configurations_authorized": False}


def evaluate_matrices(new: dict, old: dict, domains: list[str]) -> dict:
    if set(new) != set(method.expected_points()):
        raise ValueError("Incomplete new matrix set")
    draws = previous.bootstrap_draws()
    views = {"seq": (old, "seq"), "er": (old, "er")}
    for arm in method.ARMS:
        # The old endpoint function uses logical ER names. Explicit local views do
        # not rename a published artifact or mutate parent globals or source files.
        view = {order + "_shared": old[order + "_shared"] for order in method.ORDERS}
        view.update({f"{order}_er_stage{stage}": new[method.point_name(order, arm, stage)]
                     for order in method.ORDERS for stage in (2, 3)})
        views[arm] = (view, "er")
    fields = {arm: {role: {name: previous.endpoint_fields(arrays, domains, logical, role, name)
                          for name in previous.ENDPOINTS}
                    for role in (*method.ROLES, "primary")}
              for arm, (arrays, logical) in views.items()}
    endpoints = {arm: {role: {name: previous.summarize_field(value, draws)
                             for name, value in entries.items()}
                       for role, entries in variants.items()}
                 for arm, variants in fields.items()}
    comparisons = {}
    for arm in method.ARMS:
        for reference in ("er", "seq"):
            delta = {name: previous.summarize_field(fields[arm]["primary"][name]
                                                   - fields[reference]["primary"][name], draws)
                     for name in previous.ENDPOINTS}
            raw = {name: previous.summarize_field(fields[arm]["primary"][name]
                                                 - fields[reference]["raw"][name], draws)
                   for name in ("O", "N")}
            verdict = previous.comparison_checks(delta, raw)
            if len(verdict["checks"]) != 23:
                raise ValueError("Original 23 checks changed")
            comparisons[arm + "_minus_" + reference] = {
                "primary": delta, "against_raw_reference": raw, "interpretation": verdict}
    return {"endpoints": endpoints, "comparisons": comparisons,
            "selection": select_configuration(endpoints, comparisons)}


def read_points(root: Path, collection: dict, names: list[str]) -> tuple[dict, dict]:
    arrays, counts = {}, {}
    for name in names:
        records = collection["points"][name]
        if set(records) != set(method.ROLES):
            raise ValueError("Missing complete raw/stage-cal/first-cal metrics")
        arrays[name] = {role: previous.read_matrix(root, rec["matrix"]) for role, rec in records.items()}
        counts[name] = {role: previous.read_counts(root, rec["counts"]) for role, rec in records.items()}
        for role in ("stage-cal", "first-cal"):
            if not np.array_equal(arrays[name][role][:, previous.RANK_COLUMNS],
                                  arrays[name]["raw"][:, previous.RANK_COLUMNS]):
                raise ValueError("Calibrated rank/curve metrics differ")
    return arrays, counts


def stage_table(new: dict, old: dict, domains: list[str]) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.writer(stream, lineterminator="\n")
    writer.writerow(("order", "method", "history_weight", "stage", "source_point",
                     "role", "actual_domain", "metric", "group_macro"))
    rows = previous.domain_rows(domains)
    for order in method.ORDERS:
        for arm in ("seq", "er", *method.ARMS):
            for stage in (1, 2, 3):
                point = (order + "_shared" if stage == 1 else
                         method.point_name(order, arm, stage) if arm in method.ARMS else f"{order}_{arm}_stage{stage}")
                source = new if arm in method.ARMS and stage > 1 else old
                for role in method.ROLES:
                    for domain, indices in zip("ABC", rows, strict=True):
                        for k, metric in enumerate(metrics.COLUMNS):
                            writer.writerow((order, arm, method.ARMS.get(arm, 1. if arm == "er" else 0.),
                                             stage, point, role, domain, metric,
                                             repr(float(source[point][role][indices, k].mean()))))
    return stream.getvalue().encode("utf-8")


def finalize(root: Path, baseline_root: Path) -> dict:
    p = method.contract()
    reference = method.baseline(p, baseline_root.parent)
    collected = data.read_json(root / "collected.json")
    old = reference["collected"]
    if (collected["status"] != "ALL_36_ER_WEIGHT_MATRICES_SAVED_BEFORE_COMPARISONS"
            or collected["policy_sha256"] != method.POLICY_SHA256
            or collected["source_files"] != method.sources()
            or collected["metric_columns"] != list(metrics.COLUMNS)
            or set(collected["points"]) != set(method.expected_points())
            or collected["group_ids"] != old["group_ids"] or collected["domains"] != old["domains"]):
        raise ValueError("Complete aligned collection is required")
    new_arrays, new_counts = read_points(root, collected, method.expected_points())
    names = [name for order in method.ORDERS for name in
             [order + "_shared", *(f"{order}_{arm}_stage{stage}" for arm in ("seq", "er") for stage in (2, 3))]]
    old_arrays, old_counts = read_points(baseline_root, old, names)
    # Preserve the exact small matrices/counts used, separately from new observations.
    destination = root / "reference"
    destination.mkdir(exist_ok=True)
    for name in names:
        for info in old["points"][name].values():
            for rec in info.values():
                source = data.verify(baseline_root / rec["path"], rec)
                target = destination / rec["path"]
                target.parent.mkdir(parents=True, exist_ok=True)
                previous.write_once(target, source.read_bytes())
    previous.write_once(destination / "collected.json", data.json_bytes({
        "original_collected": p["baseline"]["records"]["evaluation/collected.json"],
        "group_ids": old["group_ids"], "domains": old["domains"],
        "metric_columns": old["metric_columns"], "points": {name: old["points"][name] for name in names},
        "status": "REUSED_45_ORIGINAL_PILOT_METRIC_COUNT_SETS"}))
    result = evaluate_matrices(new_arrays, old_arrays, collected["domains"])
    absolute = {}
    rows = previous.domain_rows(collected["domains"])
    for arrays, counts in ((old_arrays, old_counts), (new_arrays, new_counts)):
        for point, roles in arrays.items():
            absolute[point] = {role: {
                "macro_all": dict(zip(metrics.COLUMNS, values.mean(0).tolist())),
                "macro_by_domain": {d: dict(zip(metrics.COLUMNS, values[r].mean(0).tolist()))
                                    for d, r in zip("ABC", rows, strict=True)},
                "pooled_fixed_half_classification": method.base.fixed_classification(counts[point][role], collected["domains"])}
                for role, values in roles.items()}
    buffer = io.BytesIO()
    np.save(buffer, previous.bootstrap_draws(), allow_pickle=False)
    previous.write_once(root / "bootstrap_draws.npy", buffer.getvalue())
    previous.write_once(root / "stage_metrics.csv", stage_table(new_arrays, old_arrays, collected["domains"]))
    result.update(status="COMPLETE_ER_WEIGHT_DEVELOPMENT_COMPARISON",
                  source_files=collected["source_files"], policy_sha256=method.POLICY_SHA256,
                  collected=data.record(root / "collected.json", root),
                  reused_collection=data.record(destination / "collected.json", root),
                  new_metric_count_sets=36, reused_metric_count_sets=45,
                  stage_metrics=data.record(root / "stage_metrics.csv", root),
                  draws=data.record(root / "bootstrap_draws.npy", root), absolute_stage_results=absolute,
                  scope="Two fixed candidates on developed valid, conditional paired intervals; no independent test or new-method qualification",
                  next="Review full results; no further lambda or automatic training")
    previous.write_once(root / "evaluation.json", data.json_bytes(result))
    return result
