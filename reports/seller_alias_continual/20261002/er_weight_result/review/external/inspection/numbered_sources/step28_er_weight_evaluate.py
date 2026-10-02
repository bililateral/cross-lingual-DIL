    1  """Saved-matrix ER development comparisons; no formal label/model/text access."""
    2  from __future__ import annotations
    3  
    4  import csv
    5  import io
    6  from pathlib import Path
    7  
    8  import numpy as np
    9  
   10  import step28_er_weight as method
   11  import step28_bge_continual_evaluate as previous
   12  
   13  data, metrics = method.data, method.metrics
   14  
   15  
   16  def select_configuration(endpoints: dict, comparisons: dict) -> dict:
   17      eligible = [arm for arm in method.ARMS
   18                  if comparisons[arm + "_minus_er"]["interpretation"]["pilot_observed_checks_pass"]]
   19      selected = max(eligible, key=lambda arm: (endpoints[arm]["primary"]["O"]["map"]["mean"],
   20                                              endpoints[arm]["primary"]["O"]["recall_at_5"]["mean"],
   21                                              method.ARMS[arm])) if eligible else "er"
   22      return {"selected": selected, "history_weight": method.ARMS.get(selected, 1.),
   23              "eligible": eligible, "fallback_used": not eligible,
   24              "criterion": "All 23 versus original ER, then O MAP, O Recall@5, larger lambda",
   25              "scope": "Developed-valid configuration selection, not independent confirmation",
   26              "additional_configurations_authorized": False}
   27  
   28  
   29  def evaluate_matrices(new: dict, old: dict, domains: list[str]) -> dict:
   30      if set(new) != set(method.expected_points()):
   31          raise ValueError("Incomplete new matrix set")
   32      draws = previous.bootstrap_draws()
   33      views = {"seq": (old, "seq"), "er": (old, "er")}
   34      for arm in method.ARMS:
   35          # The old endpoint function uses logical ER names. Explicit local views do
   36          # not rename a published artifact or mutate parent globals or source files.
   37          view = {order + "_shared": old[order + "_shared"] for order in method.ORDERS}
   38          view.update({f"{order}_er_stage{stage}": new[method.point_name(order, arm, stage)]
   39                       for order in method.ORDERS for stage in (2, 3)})
   40          views[arm] = (view, "er")
   41      fields = {arm: {role: {name: previous.endpoint_fields(arrays, domains, logical, role, name)
   42                            for name in previous.ENDPOINTS}
   43                      for role in (*method.ROLES, "primary")}
   44                for arm, (arrays, logical) in views.items()}
   45      endpoints = {arm: {role: {name: previous.summarize_field(value, draws)
   46                               for name, value in entries.items()}
   47                         for role, entries in variants.items()}
   48                   for arm, variants in fields.items()}
   49      comparisons = {}
   50      for arm in method.ARMS:
   51          for reference in ("er", "seq"):
   52              delta = {name: previous.summarize_field(fields[arm]["primary"][name]
   53                                                     - fields[reference]["primary"][name], draws)
   54                       for name in previous.ENDPOINTS}
   55              raw = {name: previous.summarize_field(fields[arm]["primary"][name]
   56                                                   - fields[reference]["raw"][name], draws)
   57                     for name in ("O", "N")}
   58              verdict = previous.comparison_checks(delta, raw)
   59              if len(verdict["checks"]) != 23:
   60                  raise ValueError("Original 23 checks changed")
   61              comparisons[arm + "_minus_" + reference] = {
   62                  "primary": delta, "against_raw_reference": raw, "interpretation": verdict}
   63      return {"endpoints": endpoints, "comparisons": comparisons,
   64              "selection": select_configuration(endpoints, comparisons)}
   65  
   66  
   67  def read_points(root: Path, collection: dict, names: list[str]) -> tuple[dict, dict]:
   68      arrays, counts = {}, {}
   69      for name in names:
   70          records = collection["points"][name]
   71          if set(records) != set(method.ROLES):
   72              raise ValueError("Missing complete raw/stage-cal/first-cal metrics")
   73          arrays[name] = {role: previous.read_matrix(root, rec["matrix"]) for role, rec in records.items()}
   74          counts[name] = {role: previous.read_counts(root, rec["counts"]) for role, rec in records.items()}
   75          for role in ("stage-cal", "first-cal"):
   76              if not np.array_equal(arrays[name][role][:, previous.RANK_COLUMNS],
   77                                    arrays[name]["raw"][:, previous.RANK_COLUMNS]):
   78                  raise ValueError("Calibrated rank/curve metrics differ")
   79      return arrays, counts
   80  
   81  
   82  def stage_table(new: dict, old: dict, domains: list[str]) -> bytes:
   83      stream = io.StringIO(newline="")
   84      writer = csv.writer(stream, lineterminator="\n")
   85      writer.writerow(("order", "method", "history_weight", "stage", "source_point",
   86                       "role", "actual_domain", "metric", "group_macro"))
   87      rows = previous.domain_rows(domains)
   88      for order in method.ORDERS:
   89          for arm in ("seq", "er", *method.ARMS):
   90              for stage in (1, 2, 3):
   91                  point = (order + "_shared" if stage == 1 else
   92                           method.point_name(order, arm, stage) if arm in method.ARMS else f"{order}_{arm}_stage{stage}")
   93                  source = new if arm in method.ARMS and stage > 1 else old
   94                  for role in method.ROLES:
   95                      for domain, indices in zip("ABC", rows, strict=True):
   96                          for k, metric in enumerate(metrics.COLUMNS):
   97                              writer.writerow((order, arm, method.ARMS.get(arm, 1. if arm == "er" else 0.),
   98                                               stage, point, role, domain, metric,
   99                                               repr(float(source[point][role][indices, k].mean()))))
  100      return stream.getvalue().encode("utf-8")
  101  
  102  
  103  def finalize(root: Path, baseline_root: Path) -> dict:
  104      p = method.contract()
  105      reference = method.baseline(p, baseline_root.parent)
  106      collected = data.read_json(root / "collected.json")
  107      old = reference["collected"]
  108      if (collected["status"] != "ALL_36_ER_WEIGHT_MATRICES_SAVED_BEFORE_COMPARISONS"
  109              or collected["policy_sha256"] != method.POLICY_SHA256
  110              or collected["source_files"] != method.sources()
  111              or collected["metric_columns"] != list(metrics.COLUMNS)
  112              or set(collected["points"]) != set(method.expected_points())
  113              or collected["group_ids"] != old["group_ids"] or collected["domains"] != old["domains"]):
  114          raise ValueError("Complete aligned collection is required")
  115      new_arrays, new_counts = read_points(root, collected, method.expected_points())
  116      names = [name for order in method.ORDERS for name in
  117               [order + "_shared", *(f"{order}_{arm}_stage{stage}" for arm in ("seq", "er") for stage in (2, 3))]]
  118      old_arrays, old_counts = read_points(baseline_root, old, names)
  119      # Preserve the exact small matrices/counts used, separately from new observations.
  120      destination = root / "reference"
  121      destination.mkdir(exist_ok=True)
  122      for name in names:
  123          for info in old["points"][name].values():
  124              for rec in info.values():
  125                  source = data.verify(baseline_root / rec["path"], rec)
  126                  target = destination / rec["path"]
  127                  target.parent.mkdir(parents=True, exist_ok=True)
  128                  previous.write_once(target, source.read_bytes())
  129      previous.write_once(destination / "collected.json", data.json_bytes({
  130          "original_collected": p["baseline"]["records"]["evaluation/collected.json"],
  131          "group_ids": old["group_ids"], "domains": old["domains"],
  132          "metric_columns": old["metric_columns"], "points": {name: old["points"][name] for name in names},
  133          "status": "REUSED_45_ORIGINAL_PILOT_METRIC_COUNT_SETS"}))
  134      result = evaluate_matrices(new_arrays, old_arrays, collected["domains"])
  135      absolute = {}
  136      rows = previous.domain_rows(collected["domains"])
  137      for arrays, counts in ((old_arrays, old_counts), (new_arrays, new_counts)):
  138          for point, roles in arrays.items():
  139              absolute[point] = {role: {
  140                  "macro_all": dict(zip(metrics.COLUMNS, values.mean(0).tolist())),
  141                  "macro_by_domain": {d: dict(zip(metrics.COLUMNS, values[r].mean(0).tolist()))
  142                                      for d, r in zip("ABC", rows, strict=True)},
  143                  "pooled_fixed_half_classification": method.base.fixed_classification(counts[point][role], collected["domains"])}
  144                  for role, values in roles.items()}
  145      buffer = io.BytesIO()
  146      np.save(buffer, previous.bootstrap_draws(), allow_pickle=False)
  147      previous.write_once(root / "bootstrap_draws.npy", buffer.getvalue())
  148      previous.write_once(root / "stage_metrics.csv", stage_table(new_arrays, old_arrays, collected["domains"]))
  149      result.update(status="COMPLETE_ER_WEIGHT_DEVELOPMENT_COMPARISON",
  150                    source_files=collected["source_files"], policy_sha256=method.POLICY_SHA256,
  151                    collected=data.record(root / "collected.json", root),
  152                    reused_collection=data.record(destination / "collected.json", root),
  153                    new_metric_count_sets=36, reused_metric_count_sets=45,
  154                    stage_metrics=data.record(root / "stage_metrics.csv", root),
  155                    draws=data.record(root / "bootstrap_draws.npy", root), absolute_stage_results=absolute,
  156                    scope="Two fixed candidates on developed valid, conditional paired intervals; no independent test or new-method qualification",
  157                    next="Review full results; no further lambda or automatic training")
  158      previous.write_once(root / "evaluation.json", data.json_bytes(result))
  159      return result
