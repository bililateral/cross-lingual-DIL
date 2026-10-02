"""Saved-matrix ER development comparisons; no formal label/model/text access."""
from __future__ import annotations

import csv
import io
from pathlib import Path

import numpy as np

import step28_er_weight as method
import step28_bge_continual_evaluate as previous

data, metrics = method.data, method.metrics


def select_configuration(endpoints: dict, comparisons: dict, p: dict | None = None) -> dict:
    p = method.contract() if p is None else p
    reference = p["selection"].get("reference", "er")
    eligible = [arm for arm in p["arms"]
                if comparisons[arm + "_minus_" + reference]["interpretation"]["pilot_observed_checks_pass"]]
    selected = max(eligible, key=lambda arm: (endpoints[arm]["primary"]["O"]["map"]["mean"],
                                            endpoints[arm]["primary"]["O"]["recall_at_5"]["mean"],
                                            p["arms"][arm])) if eligible else reference
    return {"selected": selected, "history_weight": method.all_weights(p)[selected],
            "eligible": eligible, "fallback_used": not eligible,
            "criterion": "All 23 versus " + reference + ", then O MAP, O Recall@5, larger lambda",
            "scope": "Developed-valid configuration selection, not independent confirmation",
            "additional_configurations_authorized": False}


def evaluate_matrices(new: dict, old: dict, domains: list[str], p: dict | None = None) -> dict:
    p = method.contract() if p is None else p
    if set(new) != set(method.expected_points(p)):
        raise ValueError("Incomplete new matrix set")
    draws = previous.bootstrap_draws()
    views = {arm: (old, arm) for arm in p.get("baseline_methods", ("seq", "er"))}
    for arm in (*p.get("reference_arms", {}), *p["arms"]):
        # The old endpoint function uses logical ER names. Explicit local views do
        # not rename a published artifact or mutate parent globals or source files.
        view = {order + "_shared": old[order + "_shared"] for order in method.ORDERS}
        source = new if arm in p["arms"] else old
        view.update({f"{order}_er_stage{stage}": source[f"{order}_{arm}_stage{stage}"]
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
    pairs = p["evaluation"].get("comparison_pairs", [
        (arm, reference) for arm in p["arms"] for reference in p["evaluation"]["comparators"]])
    for arm, reference in pairs:
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
    result = {"endpoints": endpoints, "comparisons": comparisons}
    if method.with_logits(p):
        result["method_checks"] = {
            "increment_against_matched_er_passes": comparisons["logit_quarter_minus_quarter"]["interpretation"]["pilot_observed_checks_pass"],
            "all_guards_against_seq_pass": comparisons["logit_quarter_minus_seq"]["interpretation"]["pilot_observed_checks_pass"],
            "scope": "Separate fixed developed-valid comparisons; no automatic model replacement or new-method qualification"}
    else:
        result["selection"] = select_configuration(endpoints, comparisons, p)
    return result


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


def stage_table(new: dict, old: dict, domains: list[str], p: dict | None = None) -> bytes:
    p = method.contract() if p is None else p
    stream = io.StringIO(newline="")
    writer = csv.writer(stream, lineterminator="\n")
    writer.writerow(("order", "method", "history_weight", "stage", "source_point",
                     "role", "actual_domain", "metric", "group_macro"))
    rows = previous.domain_rows(domains)
    for order in method.ORDERS:
        for arm, weight in method.all_weights(p).items():
            for stage in (1, 2, 3):
                point = (order + "_shared" if stage == 1 else
                         f"{order}_{arm}_stage{stage}")
                source = new if arm in p["arms"] and stage > 1 else old
                for role in method.ROLES:
                    for domain, indices in zip("ABC", rows, strict=True):
                        for k, metric in enumerate(metrics.COLUMNS):
                            writer.writerow((order, arm, weight,
                                             stage, point, role, domain, metric,
                                             repr(float(source[point][role][indices, k].mean()))))
    return stream.getvalue().encode("utf-8")


def finalize(root: Path, baseline_root: Path, p: dict | None = None,
             weight_root: Path | None = None) -> dict:
    p = method.contract() if p is None else p
    reference = method.baseline(p, baseline_root.parent,
                                None if weight_root is None else weight_root.parent)
    collected = data.read_json(root / "collected.json")
    old = reference["collected"]
    if (collected["status"] != f"ALL_{p['metric_count_sets']}_{method.evidence_tag(p)}_MATRICES_SAVED_BEFORE_COMPARISONS"
            or collected["policy_sha256"] != method.policy_sha256(p)
            or collected["source_files"] != method.sources(p)
            or collected["metric_columns"] != list(metrics.COLUMNS)
            or set(collected["points"]) != set(method.expected_points(p))
            or collected["group_ids"] != old["group_ids"] or collected["domains"] != old["domains"]):
        raise ValueError("Complete aligned collection is required")
    new_arrays, new_counts = read_points(root, collected, method.expected_points(p))
    names = [name for order in method.ORDERS for name in
             [order + "_shared", *(f"{order}_{arm}_stage{stage}" for arm in p.get("baseline_methods", ("seq", "er")) for stage in (2, 3))]]
    old_arrays, old_counts = read_points(baseline_root, old, names)
    reuse = [(baseline_root, old, names)]
    if "weight_reference" in reference:
        wr = reference["weight_reference"]
        weight_names = [f"{order}_{arm}_stage{stage}" for order in method.ORDERS
                        for arm in p["reference_arms"] for stage in (2, 3)]
        weight_arrays, weight_counts = read_points(wr["job"] / "evaluation", wr["collected"], weight_names)
        old_arrays.update(weight_arrays)
        old_counts.update(weight_counts)
        reuse.append((wr["job"] / "evaluation", wr["collected"], weight_names))
    # Preserve the exact small matrices/counts used, separately from new observations.
    destination = root / "reference"
    destination.mkdir(exist_ok=True)
    reused_points = {}
    for source_root, collection, point_names in reuse:
        for name in point_names:
            reused_points[name] = collection["points"][name]
            for info in collection["points"][name].values():
                for rec in info.values():
                    source = data.verify(source_root / rec["path"], rec)
                    target = destination / rec["path"]
                    target.parent.mkdir(parents=True, exist_ok=True)
                    previous.write_once(target, source.read_bytes())
    reused_count = len(reused_points) * len(method.ROLES)
    previous.write_once(destination / "collected.json", data.json_bytes({
        "original_collected": p["baseline"]["records"]["evaluation/collected.json"],
        "weight_collected": p.get("weight_reference", {}).get("records", {}).get("evaluation/collected.json"),
        "group_ids": old["group_ids"], "domains": old["domains"],
        "metric_columns": old["metric_columns"], "points": reused_points,
        "status": f"REUSED_{reused_count}_FROZEN_METRIC_COUNT_SETS"}))
    result = evaluate_matrices(new_arrays, old_arrays, collected["domains"], p)
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
    previous.write_once(root / "stage_metrics.csv", stage_table(new_arrays, old_arrays, collected["domains"], p))
    result.update(status=f"COMPLETE_{method.evidence_tag(p)}_DEVELOPMENT_COMPARISON",
                  source_files=collected["source_files"], policy_sha256=method.policy_sha256(p),
                  collected=data.record(root / "collected.json", root),
                  reused_collection=data.record(destination / "collected.json", root),
                  new_metric_count_sets=p["metric_count_sets"], reused_metric_count_sets=reused_count,
                  stage_metrics=data.record(root / "stage_metrics.csv", root),
                  draws=data.record(root / "bootstrap_draws.npy", root), absolute_stage_results=absolute,
                  scope="Fixed confirmed candidates on developed valid, conditional paired intervals; no independent test or new-method qualification",
                  next="Review full results; no further lambda or automatic training")
    previous.write_once(root / "evaluation.json", data.json_bytes(result))
    return result
