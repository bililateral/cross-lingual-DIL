    1  """Saved-matrix continual statistics; no labels, text, weights or fitting entry."""
    2  from __future__ import annotations
    3  
    4  import csv
    5  import io
    6  from pathlib import Path
    7  from typing import Any
    8  
    9  import numpy as np
   10  
   11  import step28_bge_continual as method
   12  
   13  data, metrics = method.data, method.metrics
   14  ENDPOINTS = ("O", "N", "Z", "F_first", "F", "G", "final_all")
   15  RANK_COLUMNS = tuple(metrics.COLUMNS.index(name) for name in
   16                       (*metrics.CURVE_KEYS, *metrics.RETRIEVAL_KEYS))
   17  
   18  
   19  def point_name(order: str, arm: str, stage: int) -> str:
   20      if order not in method.ORDERS or arm not in method.METHODS or stage not in (1, 2, 3):
   21          raise ValueError("Unknown logical path point")
   22      return order + "_shared" if stage == 1 or arm == "frozen" else f"{order}_{arm}_stage{stage}"
   23  
   24  
   25  def expected_points() -> list[str]:
   26      return [name for order in method.ORDERS for name in
   27              [order + "_shared", *(f"{order}_{arm}_stage{stage}"
   28                                    for arm in method.UPDATED for stage in (2, 3))]]
   29  
   30  
   31  def domain_rows(domains: list[str]) -> list[np.ndarray]:
   32      rows = [np.flatnonzero(np.asarray(domains) == domain) for domain in "ABC"]
   33      if len(domains) != 60 or any(len(row) != 20 for row in rows):
   34          raise ValueError("Need the same 20 valid groups per actual domain")
   35      return rows
   36  
   37  
   38  def bootstrap_draws() -> np.ndarray:
   39      return np.random.Generator(np.random.PCG64(20260930)).integers(0, 20, size=(5000, 3, 20))
   40  
   41  
   42  def read_matrix(root: Path, record: dict) -> np.ndarray:
   43      path = data.verify(root / record["path"], record)
   44      result = np.load(path, allow_pickle=False)
   45      if result.shape != (60, 22) or result.dtype != np.float64 or not np.isfinite(result).all():
   46          raise ValueError("Complete finite 60x22 float64 matrix required")
   47      return result
   48  
   49  
   50  def read_counts(root: Path, record: dict) -> list[dict]:
   51      values = data.read_json(data.verify(root / record["path"], record))
   52      if len(values) != 60:
   53          raise ValueError("Incomplete group confusion counts")
   54      for row in values:
   55          if (any(type(row[k]) is not int or row[k] < 0 for k in ("tp", "fp", "fn", "tn"))
   56                  or row["tp"] + row["fn"] != 20 or row["fp"] + row["tn"] != 358):
   57              raise ValueError("Confusion counts violate the complete 378-pair group")
   58      return values
   59  
   60  
   61  def write_once(path: Path, payload: bytes) -> None:
   62      if path.exists():
   63          if path.read_bytes() != payload:
   64              raise FileExistsError("Refusing to replace a published result with different bytes: " + str(path))
   65      else:
   66          path.write_bytes(payload)
   67  
   68  
   69  def stage_table(arrays: dict, initial: np.ndarray, rows: list[np.ndarray]) -> bytes:
   70      """Explicit scientific indices, including unchanged logical frozen endpoints."""
   71      stream = io.StringIO(newline="")
   72      writer = csv.writer(stream, lineterminator="\n")
   73      writer.writerow(("metric", "output_role", "actual_domain", "arrival_index", "stage",
   74                       "order", "seed", "method", "source_point", "group_macro"))
   75      for order in method.ORDERS:
   76          for arrival, domain in enumerate(order, 1):
   77              r = rows["ABC".index(domain)]
   78              for k, metric in enumerate(metrics.COLUMNS):
   79                  writer.writerow((metric, "raw", domain, arrival, 0, order, "s0", "initial",
   80                                   "initial", repr(float(initial[r, k].mean()))))
   81              for arm in method.METHODS:
   82                  for stage in (1, 2, 3):
   83                      point = point_name(order, arm, stage)
   84                      for role in method.ROLES:
   85                          for k, metric in enumerate(metrics.COLUMNS):
   86                              writer.writerow((metric, role, domain, arrival, stage, order, "s0", arm,
   87                                               point, repr(float(arrays[point][role][r, k].mean()))))
   88      return stream.getvalue().encode("utf-8")
   89  
   90  
   91  def endpoint_fields(arrays: dict, domains: list[str], arm: str, role: str,
   92                      endpoint: str) -> np.ndarray:
   93      """(order, actual_domain, group_within_actual_domain, metric) weighted fields.
   94  
   95      Domain sums, not another mean over domains: term coefficients already sum to
   96      one for absolute endpoints. Shared groups remain aligned across every path.
   97      """
   98      if role not in (*method.ROLES, "primary") or endpoint not in ENDPOINTS:
   99          raise ValueError("Unknown metric role or endpoint")
  100      terms = {
  101          "O": ((3, 1, .5), (3, 2, .5)),
  102          "N": ((2, 2, .5), (3, 3, .5)), "Z": ((3, 3, 1.),),
  103          "F_first": ((1, 1, 1.), (3, 1, -1.)),
  104          "F": ((1, 1, .5), (3, 1, -.5), (2, 2, .5), (3, 2, -.5)),
  105          "G": ((2, 2, .5), (1, 2, -.5), (3, 3, .5), (2, 3, -.5)),
  106          "final_all": ((3, 1, 1 / 3), (3, 2, 1 / 3), (3, 3, 1 / 3)),
  107      }[endpoint]
  108      rows = domain_rows(domains)
  109      fields = np.zeros((3, 3, 20, 22), dtype=np.float64)
  110      for i, order in enumerate(method.ORDERS):
  111          for stage, arrival, weight in terms:
  112              key = point_name(order, arm, stage)
  113              if role == "primary":
  114                  values = arrays[key]["stage-cal"].copy()
  115                  values[:, RANK_COLUMNS] = arrays[key]["raw"][:, RANK_COLUMNS]
  116              else:
  117                  values = arrays[key][role]
  118              d = "ABC".index(order[arrival - 1])
  119              fields[i, d] += weight * values[rows[d]]
  120      if endpoint in ("F_first", "F", "G"):
  121          fields[..., [metrics.COLUMNS.index(n) for n in ("brier", "log_loss")]] *= -1
  122      return fields
  123  
  124  
  125  def summarize_field(fields: np.ndarray, draws: np.ndarray) -> dict:
  126      if (fields.shape != (3, 3, 20, 22) or not np.isfinite(fields).all()
  127              or draws.shape != (5000, 3, 20) or draws.dtype.kind not in "iu"
  128              or np.any((draws < 0) | (draws >= 20))):
  129          raise ValueError("Bad fields or actual-domain draws")
  130      mean_field = fields.mean(0)
  131      per_order = fields.mean(2).sum(1)
  132      point = mean_field.mean(1).sum(0)
  133      boot = sum(mean_field[d][draws[:, d]].mean(1) for d in range(3))
  134      intervals = np.quantile(boot, [.025, .975], axis=0, method="linear")
  135      return {name: {"mean": float(point[k]),
  136                     "per_order": dict(zip(method.ORDERS, per_order[:, k].tolist())),
  137                     "conditional_95pct_interval": intervals[:, k].tolist()}
  138              for k, name in enumerate(metrics.COLUMNS)}
  139  
  140  
  141  def comparison_checks(delta: dict, versus_raw: dict) -> dict:
  142      old, new, latest = delta["O"], delta["N"], delta["Z"]
  143      checks = {"old_map_improves": old["map"]["mean"] > 0,
  144                "old_map_interval_above_zero": old["map"]["conditional_95pct_interval"][0] > 0,
  145                "old_recall5_improves": old["recall_at_5"]["mean"] > 0,
  146                "new_map_non_decrease": new["map"]["mean"] >= 0,
  147                "new_recall5_non_decrease": new["recall_at_5"]["mean"] >= 0}
  148      for endpoint in ("O", "N"):
  149          for name, sign in ranking_guards().items():
  150              checks[f"{endpoint}_{name}_non_degradation"] = sign * delta[endpoint][name]["mean"] >= 0
  151          for name in ("brier", "log_loss"):
  152              checks[f"{endpoint}_{name}_against_raw_reference"] = versus_raw[endpoint][name]["mean"] <= 0
  153      for name, sign in {"map": 1, "recall_at_5": 1, **ranking_guards()}.items():
  154          checks[f"Z_{name}_non_degradation"] = sign * latest[name]["mean"] >= 0
  155      return {"pilot_observed_checks_pass": all(checks.values()), "checks": checks,
  156              "failed": [key for key, ok in checks.items() if not ok],
  157              "positive_old_map_orders": sum(v > 0 for v in old["map"]["per_order"].values()),
  158              "future_three_seed_qualification": "NOT_EVALUATED_SINGLE_SEED_PILOT",
  159              "scope": "Observed mean protection, not per-domain guarantees or population noninferiority"}
  160  
  161  
  162  def ranking_guards() -> dict[str, int]:
  163      return {"average_precision": 1, "roc_auc": 1, "brier": -1, "log_loss": -1}
  164  
  165  
  166  def forgetting_checks(first: dict, gain: dict) -> dict:
  167      matched = [o for o in method.ORDERS if first["map"]["per_order"][o] > 0
  168                 and gain["map"]["per_order"][o] > 0]
  169      checks = {"seq_first_MAP_loss_positive": first["map"]["mean"] > 0,
  170                "seq_first_MAP_loss_interval_above_zero": first["map"]["conditional_95pct_interval"][0] > 0,
  171                "same_order_path_forgetting_and_new_learning": len(matched) >= 2}
  172      return {"established": all(checks.values()), "checks": checks, "matched_orders": matched,
  173              "claim": "Same complete path, not necessarily same transition or gradient conflict",
  174              "failure_scope": "Does not rule out intermediate or second-domain forgetting"}
  175  
  176  
  177  def finalize(root: Path) -> dict:
  178      """Recoverable after full collection: uses only saved matrices and counts."""
  179      collected = data.read_json(root / "collected.json")
  180      if collected["status"] != "ALL_PILOT_MATRICES_SAVED_BEFORE_COMPARISONS":
  181          raise ValueError("Complete collection required")
  182      if collected["metric_columns"] != list(metrics.COLUMNS):
  183          raise ValueError("Metric schema changed")
  184      domains = collected["domains"]
  185      if len(collected["group_ids"]) != 60 or len(set(collected["group_ids"])) != 60:
  186          raise ValueError("Repeated or incomplete group identities")
  187      rows = domain_rows(domains)
  188      if set(collected["points"]) != set(expected_points()):
  189          raise ValueError("Incomplete stage endpoint set")
  190      arrays = {point: {role: read_matrix(root, record["matrix"])
  191                        for role, record in info.items()}
  192                for point, info in collected["points"].items()}
  193      for point, roles in arrays.items():
  194          if set(roles) != set(method.ROLES):
  195              raise ValueError("Incomplete output roles")
  196          for role in ("stage-cal", "first-cal"):
  197              if not np.array_equal(roles[role][:, RANK_COLUMNS], roles["raw"][:, RANK_COLUMNS]):
  198                  raise ValueError("Calibrated ranking/curve metrics changed")
  199      initial = read_matrix(root, collected["initial"]["matrix"])
  200      read_counts(root, collected["initial"]["counts"])
  201      counts_by_point = {point: {role: read_counts(root, info["counts"]) for role, info in entries.items()}
  202                         for point, entries in collected["points"].items()}
  203      draws = bootstrap_draws()
  204      buffer = io.BytesIO()
  205      np.save(buffer, draws, allow_pickle=False)
  206      write_once(root / "bootstrap_draws.npy", buffer.getvalue())
  207      write_once(root / "stage_metrics.csv", stage_table(arrays, initial, rows))
  208      fields = {arm: {role: {name: endpoint_fields(arrays, domains, arm, role, name)
  209                            for name in ENDPOINTS} for role in (*method.ROLES, "primary")}
  210                for arm in method.METHODS}
  211      endpoints = {arm: {role: {name: summarize_field(value, draws) for name, value in entries.items()}
  212                         for role, entries in variants.items()} for arm, variants in fields.items()}
  213      comparisons = {}
  214      for candidate, reference in (("er", "seq"), ("logit", "er")):
  215          delta = {name: summarize_field(fields[candidate]["primary"][name]
  216                                         - fields[reference]["primary"][name], draws) for name in ENDPOINTS}
  217          against_raw = {name: summarize_field(fields[candidate]["primary"][name]
  218                                               - fields[reference]["raw"][name], draws) for name in ("O", "N")}
  219          comparisons[candidate + "_minus_" + reference] = {
  220              "primary": delta, "against_raw_reference": against_raw,
  221              "interpretation": comparison_checks(delta, against_raw)}
  222      absolute = {}
  223      for point, roles in arrays.items():
  224          absolute[point] = {}
  225          for role, values in roles.items():
  226              counts = counts_by_point[point][role]
  227              absolute[point][role] = {
  228                  "macro_all": dict(zip(metrics.COLUMNS, values.mean(0).tolist())),
  229                  "macro_by_domain": {d: dict(zip(metrics.COLUMNS, values[r].mean(0).tolist()))
  230                                      for d, r in zip("ABC", rows, strict=True)},
  231                  "pooled_fixed_half_classification": method.base.fixed_classification(counts, domains)}
  232      first_learning = {}
  233      for order in method.ORDERS:
  234          r = rows["ABC".index(order[0])]
  235          first = arrays[order + "_shared"]["raw"][r]
  236          first_learning[order] = {
  237              "actual_domain": order[0], "initial_raw": dict(zip(metrics.COLUMNS, initial[r].mean(0).tolist())),
  238              "first_raw": dict(zip(metrics.COLUMNS, first.mean(0).tolist())),
  239              "raw_first_minus_initial": dict(zip(metrics.COLUMNS, (first - initial[r]).mean(0).tolist()))}
  240      result = {"status": "COMPLETE_BGE_CONTINUAL_DIAGNOSTIC", "source_files": collected["source_files"],
  241                "analysis_sources": [data.record(data.ROOT / name, data.ROOT) for name in
  242                                     ("scripts/step28_bge_continual_evaluate.py", "scripts/step28_bge_continual.py",
  243                                      "scripts/step28_continual_population_evaluate.py", "scripts/step28_chinese_base.py")],
  244                "collected": data.record(root / "collected.json", root),
  245                "stage_metrics": data.record(root / "stage_metrics.csv", root),
  246                "draws": data.record(root / "bootstrap_draws.npy", root), "endpoints": endpoints,
  247                "comparisons": comparisons, "absolute_stage_results": absolute,
  248                "initial_to_first_learning": first_learning,
  249                "seq_forgetting": forgetting_checks(endpoints["seq"]["raw"]["F_first"],
  250                                                     endpoints["seq"]["raw"]["G"]),
  251                "scope": "Fixed s0 conditional development diagnostic; no novel-method qualification or test",
  252                "next": "REVIEW_COMPLETE_RESULTS_BEFORE_ANY_NEW_STAGE"}
  253      write_once(root / "evaluation.json", data.json_bytes(result))
  254      return result
