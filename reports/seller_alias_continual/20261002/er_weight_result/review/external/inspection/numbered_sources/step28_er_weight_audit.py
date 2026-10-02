    1  """Independently audit saved ER-weight results on Linux; no labels or models."""
    2  from __future__ import annotations
    3  
    4  import argparse
    5  from collections import Counter
    6  import csv
    7  from datetime import datetime, timezone
    8  import json
    9  import math
   10  import os
   11  from pathlib import Path
   12  import time
   13  
   14  import numpy as np
   15  
   16  from step28_bge_continual_audit import Audit, METRICS, ORDERS, RANK, TERMS, bound, confusion, read, record
   17  
   18  ARMS = {"half": .5, "quarter": .25}
   19  ROLES = ("raw", "stage-cal", "first-cal", "primary")
   20  
   21  
   22  def point(order: str, arm: str, stage: int) -> str:
   23      return order + "_shared" if stage == 1 else f"{order}_{arm}_stage{stage}"
   24  
   25  
   26  def summarize(value: np.ndarray, weights: np.ndarray) -> dict:
   27      per_order = value.sum(axis=1) / 20
   28      means = per_order.mean(axis=0)
   29      ordered = np.sort(weights @ value.mean(axis=0), axis=0)
   30      intervals = []
   31      for fraction in (.025, .975):
   32          position = fraction * 4999
   33          lo, hi = math.floor(position), math.ceil(position)
   34          intervals.append(ordered[lo] + (position - lo) * (ordered[hi] - ordered[lo]))
   35      return {m: {"mean": float(means[k]),
   36                  "per_order": {o: float(per_order[i, k]) for i, o in enumerate(ORDERS)},
   37                  "conditional_95pct_interval": [float(a[k]) for a in intervals]}
   38              for k, m in enumerate(METRICS)}
   39  
   40  
   41  def main() -> None:
   42      parser = argparse.ArgumentParser(description=__doc__)
   43      for name in ("project", "job", "baseline", "inventory", "output"):
   44          parser.add_argument("--" + name, type=Path, required=True)
   45      parser.add_argument("--cpu", type=int, default=47)
   46      args = parser.parse_args()
   47      os.sched_setaffinity(0, {args.cpu})
   48      os.nice(15)
   49      started = time.monotonic()
   50      args.output.mkdir(parents=True, exist_ok=False)
   51      project, job, baseline = args.project.resolve(), args.job.resolve(), args.baseline.resolve()
   52      run, ev, old_run = job / "run", job / "evaluation", baseline / "run"
   53      audit = Audit()
   54      for item in read(args.inventory)["files"]:
   55          bound(project, item)
   56          audit.checks["returned_files"] += 1
   57      manifest, collected, saved = read(run / "manifest.json"), read(ev / "collected.json"), read(ev / "evaluation.json")
   58      old_manifest, old_collected = read(old_run / "manifest.json"), read(baseline / "evaluation/collected.json")
   59      policy = read(project / "schema/step28_er_weight_policy.json")
   60      for item in policy["baseline"]["records"].values():
   61          bound(baseline, item)
   62      completion, gate = read(job / "completion.json"), read(job / "before_valid.json")
   63      assert (job / "exit_status.txt").read_text().strip() == "0" and not (job / "failure.json").exists()
   64      assert completion["physical_updates"] == manifest["physical_updates"] == 3456
   65      assert manifest["gradient_group_presentations"] == 6912
   66      assert read(job / "access.json") == completion["label_parses"] == dict(train=1, valid=1, heldout=0, owners=0)
   67      assert gate["status"] == "PASS_ER_WEIGHT_COMPLETE_BLIND_GATE"
   68      assert gate["label_parses"] == dict(train=1, valid=0, heldout=0, owners=0)
   69      bound(job, gate["manifest"])
   70      bound(job, completion["evaluation"])
   71      for item in read(job / "execution.json")["source_files"]:
   72          bound(project, item)
   73          audit.checks["unchanged_frozen_sources"] += 1
   74      assert collected["group_ids"] == old_collected["group_ids"]
   75      assert collected["domains"] == old_collected["domains"]
   76      assert tuple(collected["metric_columns"]) == METRICS
   77      assert len(set(collected["group_ids"])) == 60
   78      domains = np.asarray(collected["domains"])
   79      rows = {d: np.flatnonzero(domains == d) for d in "ABC"}
   80      assert all(len(r) == 20 for r in rows.values())
   81      partition = read(bound(run, manifest["partition"]))
   82      assert partition == read(old_run / "partition.json")
   83      uid_domain = {x["group_uid"]: x["domain"] for x in partition["fit"]}
   84      expected = {point(o, a, s) for o in ORDERS for a in ARMS for s in (2, 3)}
   85      assert set(manifest["points"]) == set(manifest["training"]) == set(collected["points"]) == expected
   86      diagnostics = {"training": {}, "memory": {}, "calibration": {}}
   87      scores = {}
   88      for name in sorted(expected):
   89          p = read(bound(run, manifest["points"][name]))
   90          t = read(bound(run, manifest["training"][name]))
   91          order, arm, stage = p["order"], name.split("_")[1], p["stage"]
   92          reference = read(bound(old_run, old_manifest["training"][point(order, "er", stage)]))
   93          assert p["history_weight"] == t["history_weight"] == ARMS[arm]
   94          assert p["completed_updates"] == t["adam_step"] == 288 * stage
   95          assert p["full_model_adam_and_rng_restore_verified"] is True and t["updates"] == 288
   96          for key in ("current_ids", "history_ids", "current_dropout_stream", "adam_step", "updates"):
   97              assert t[key] == reference[key], (name, key)
   98          current_ids = {uid for uid, domain in uid_domain.items() if domain == order[stage - 1]}
   99          assert Counter(t["current_ids"]) == Counter({uid: 6 for uid in current_ids}) and len(current_ids) == 48
  100          memory = read(run / "memory" / (name + "_budget.json"))
  101          assert 0 < memory["serialized_bytes"] <= 1048576 and len(memory["members"]) == 6
  102          assert memory["members"] == reference["memory_after_training"]["members"]
  103          assert set(t["history_ids"]) <= set(memory["members"]) and not set(memory["members"]) & current_ids
  104          x = np.load(bound(run, t["update_file"]), allow_pickle=False)
  105          assert x.shape == (288, 14) and np.isfinite(x).all()
  106          c = {column: x[:, k] for k, column in enumerate(t["update_columns"])}
  107          for role in ("current", "history"):
  108              assert np.allclose(c[role + "_total"], c[role + "_bce"] + c[role + "_rank"] + .5 * c[role + "_hard"], atol=3e-6, rtol=3e-6)
  109          audit.array(c["history_weight"], np.full(288, ARMS[arm]), name + "/lambda", 0.)
  110          audit.array(c["weighted_history_total"], ARMS[arm] * c["history_total"], name + "/weighted", 0.)
  111          audit.array(c["total"], c["current_total"] + ARMS[arm] * c["history_total"], name + "/total", 0.)
  112          audit.array(c["encoder_lr"], np.array([1e-5 * (s / 29 if s <= 29 else (288 - s) / 259) for s in range(1, 289)]), name + "/lr")
  113          audit.array(c["head_lr"], np.full(288, .001), name + "/head_lr", 0.)
  114          for step in (1, 29, 30, 288):
  115              for module in ("encoder", "head"):
  116                  obs = t["observations"][str(step)][module]
  117                  assert obs["finite_nonzero_combined_gradient"] and obs["parameters_changed"] == (module == "head" or step < 288)
  118          mapping = read(bound(run, p["mapping"]))
  119          assert mapping["status"] == "PASS_CALIBRATION_FIT" and mapping["a"] > 0
  120          assert mapping["calibration_group_ids"] == [g["group_uid"] for g in partition["calibration"] if g["domain"] == order[stage - 1]]
  121          assert (mapping["group_count"], mapping["pair_count"], mapping["positive_count"]) == (12, 4536, 240)
  122          assert mapping["model_state_sha256"] == p["model_state_sha256"] and mapping["score_source"] == p["scores"]["calibration"]
  123          values = {role: np.load(bound(run, item), allow_pickle=False) for role, item in p["scores"].items()}
  124          assert all(v.shape == ((12, 378) if r == "calibration" else (60, 378)) and np.isfinite(v).all() for r, v in values.items())
  125          for role, parameters in (("stage-cal", mapping), ("first-cal", p["first_map_parameters"])):
  126              raw = values["development"].astype(np.float64)
  127              audit.array(values[role], raw * parameters["a"] + parameters["b"], name + role, 0.)
  128              ordering = np.argsort(raw, axis=1, kind="stable")
  129              assert np.array_equal(ordering, np.argsort(values[role], axis=1, kind="stable"))
  130              assert np.array_equal(np.diff(np.take_along_axis(raw, ordering, 1)) == 0, np.diff(np.take_along_axis(values[role], ordering, 1)) == 0)
  131          scores[name] = values
  132          diagnostics["training"][name] = {"seconds": t["training_seconds"], "gradient_norm_gt_one": int((c["gradient_norm"] > 1).sum()), "gradient_norm_mean": float(c["gradient_norm"].mean())}
  133          diagnostics["memory"][name] = {"bytes": memory["serialized_bytes"], "domains": dict(Counter(uid_domain[uid] for uid in memory["members"]))}
  134          diagnostics["calibration"][name] = {k: mapping[k] for k in ("a", "b", "initial_nll", "final_nll")}
  135      for order in ORDERS:
  136          shared = read(bound(old_run, old_manifest["points"][order + "_shared"]))
  137          for arm in ARMS:
  138              start = manifest["restored_starts"][order + "_" + arm]
  139              assert start["full_checkpoint"] == shared["full_checkpoint"] and start["adam_step"] == 288
  140              assert start["model_state_sha256"] == shared["model_state_sha256"] and start["first_scores_replayed_exactly"]
  141              assert start["first_map"] == shared["first_map_parameters"]
  142              assert start["memory_source"] == old_manifest["memories"][order + "_er_after1"]["file"]
  143              retained = manifest["memories"][point(order, arm, 2)]
  144              assert retained["members"] == old_manifest["memories"][point(order, "er", 2)]["members"]
  145      matrices, counts = {}, {}
  146      for collection, folder in ((read(ev / "reference/collected.json"), ev / "reference"), (collected, ev)):
  147          for name, variants in collection["points"].items():
  148              matrices[name], counts[name] = {}, {}
  149              for role, info in variants.items():
  150                  matrix = np.load(bound(folder, info["matrix"]), allow_pickle=False)
  151                  assert matrix.shape == (60, 22) and matrix.dtype == np.float64 and np.isfinite(matrix).all()
  152                  count = read(bound(folder, info["counts"]))
  153                  assert len(count) == 60
  154                  for i, row in enumerate(count):
  155                      assert all(type(row[k]) is int and row[k] >= 0 for k in ("tp", "fp", "fn", "tn"))
  156                      assert row["tp"] + row["fn"] == 20 and row["fp"] + row["tn"] == 358
  157                      for metric, value in confusion(row).items():
  158                          audit.equal(matrix[i, METRICS.index(metric)], value, name + role + metric)
  159                      if name in scores:
  160                          assert int((scores[name]["development" if role == "raw" else role][i] >= 0).sum()) == row["tp"] + row["fp"]
  161                  if name not in expected:
  162                      assert info == old_collected["points"][name][role]
  163                  matrices[name][role], counts[name][role] = matrix, count
  164                  audit.checks["matrix_count_sets"] += 1
  165              for role in ROLES[1:3]:
  166                  audit.array(matrices[name][role][:, RANK], matrices[name]["raw"][:, RANK], name + "/rank", 0.)
  167              matrices[name]["primary"] = matrices[name]["stage-cal"].copy()
  168      assert audit.checks["matrix_count_sets"] == 81
  169      draws = np.random.Generator(np.random.PCG64(20260930)).integers(0, 20, (5000, 3, 20))
  170      assert np.array_equal(draws, np.load(bound(ev, saved["draws"]), allow_pickle=False))
  171      weights = np.zeros((5000, 60))
  172      for i, domain in enumerate("ABC"):
  173          weights[:, rows[domain]] = np.stack([np.bincount(d, minlength=20) for d in draws[:, i]]) / 20
  174  
  175      def field(arm: str, role: str, endpoint: str) -> np.ndarray:
  176          value = np.zeros((3, 60, 22))
  177          for i, order in enumerate(ORDERS):
  178              for stage, arrival, coefficient in TERMS[endpoint]:
  179                  selected = rows[order[arrival - 1]]
  180                  value[i, selected] += coefficient * matrices[point(order, arm, stage)][role][selected]
  181          if endpoint in ("F_first", "F", "G"):
  182              value[:, :, [4, 5]] *= -1
  183          return value
  184  
  185      handmade = np.zeros((3, 60, 22))
  186      for domain, value in zip("ABC", (2, 4, 6)):
  187          handmade[:, rows[domain]] = value
  188      audit.equal(summarize(handmade, weights)["map"], {"mean": 12., "per_order": dict.fromkeys(ORDERS, 12.), "conditional_95pct_interval": [12., 12.]}, "handmade_bootstrap")
  189      assert confusion(dict(tp=2, fp=1, fn=2, tn=5))["recall"] == .5
  190      endpoints = {arm: {role: {ep: summarize(field(arm, role, ep), weights) for ep in TERMS} for role in ROLES} for arm in ("seq", "er", *ARMS)}
  191      audit.equal(saved["endpoints"], endpoints, "endpoints")
  192      comparisons = {}
  193      signs = dict(average_precision=1, roc_auc=1, brier=-1, log_loss=-1)
  194      for candidate in ARMS:
  195          for reference in ("er", "seq"):
  196              name = candidate + "_minus_" + reference
  197              delta = {ep: summarize(field(candidate, "primary", ep) - field(reference, "primary", ep), weights) for ep in TERMS}
  198              raw = {ep: summarize(field(candidate, "primary", ep) - field(reference, "raw", ep), weights) for ep in ("O", "N")}
  199              audit.equal(saved["comparisons"][name]["primary"], delta, name)
  200              audit.equal(saved["comparisons"][name]["against_raw_reference"], raw, name + "/raw")
  201              checks = {"old_map_improves": delta["O"]["map"]["mean"] > 0, "old_map_interval_above_zero": delta["O"]["map"]["conditional_95pct_interval"][0] > 0,
  202                        "old_recall5_improves": delta["O"]["recall_at_5"]["mean"] > 0, "new_map_non_decrease": delta["N"]["map"]["mean"] >= 0, "new_recall5_non_decrease": delta["N"]["recall_at_5"]["mean"] >= 0}
  203              for ep in ("O", "N"):
  204                  checks.update({f"{ep}_{m}_non_degradation": sign * delta[ep][m]["mean"] >= 0 for m, sign in signs.items()})
  205                  checks.update({f"{ep}_{m}_against_raw_reference": raw[ep][m]["mean"] <= 0 for m in ("brier", "log_loss")})
  206              checks.update({f"Z_{m}_non_degradation": sign * delta["Z"][m]["mean"] >= 0 for m, sign in dict(map=1, recall_at_5=1, **signs).items()})
  207              verdict = saved["comparisons"][name]["interpretation"]
  208              audit.equal(verdict["checks"], checks, name + "/checks")
  209              assert len(checks) == 23 and verdict["pilot_observed_checks_pass"] == all(checks.values())
  210              assert set(verdict["failed"]) == {k for k, value in checks.items() if not value}
  211              assert verdict["positive_old_map_orders"] == sum(v > 0 for v in delta["O"]["map"]["per_order"].values())
  212              comparisons[name] = {"passed": sum(checks.values()), "total": len(checks), "checks": checks}
  213      eligible = [a for a in ARMS if comparisons[a + "_minus_er"]["passed"] == 23]
  214      selected = max(eligible, key=lambda a: (endpoints[a]["primary"]["O"]["map"]["mean"], endpoints[a]["primary"]["O"]["recall_at_5"]["mean"], ARMS[a])) if eligible else "er"
  215      assert saved["selection"] == completion["selection"]
  216      assert saved["selection"]["selected"] == selected and saved["selection"]["eligible"] == eligible
  217      assert saved["selection"]["history_weight"] == ARMS.get(selected, 1.) and saved["selection"]["fallback_used"] == (not eligible)
  218      for name, roles in saved["absolute_stage_results"].items():
  219          for role, obj in roles.items():
  220              matrix = matrices[name][role]
  221              audit.equal(obj["macro_all"], dict(zip(METRICS, matrix.mean(0).tolist())), name + role)
  222              for domain, indices in rows.items():
  223                  audit.equal(obj["macro_by_domain"][domain], dict(zip(METRICS, matrix[indices].mean(0).tolist())), name + role + domain)
  224              pooled = obj["pooled_fixed_half_classification"]
  225              assert pooled["threshold"] == 0
  226              for domain, indices in {**rows, "pooled": np.arange(60)}.items():
  227                  total = {k: sum(counts[name][role][i][k] for i in indices) for k in ("tp", "fp", "fn", "tn")}
  228                  rates = confusion(total)
  229                  expected_count = {**total, **{k: rates[k] for k in ("precision", "recall", "f1")}, "fpr": total["fp"] / (total["fp"] + total["tn"])}
  230                  audit.equal(pooled["pooled"] if domain == "pooled" else pooled["by_domain"][domain], expected_count, name + role + domain + "/counts")
  231      with bound(ev, saved["stage_metrics"]).open(encoding="utf-8", newline="") as stream:
  232          keys = set()
  233          for row in csv.DictReader(stream):
  234              name = point(row["order"], row["method"], int(row["stage"]))
  235              assert name == row["source_point"]
  236              key = tuple(row[k] for k in ("order", "method", "stage", "role", "actual_domain", "metric"))
  237              assert key not in keys
  238              keys.add(key)
  239              audit.equal(float(row["group_macro"]), matrices[name][row["role"]][rows[row["actual_domain"]], METRICS.index(row["metric"])].mean(), "stage_csv")
  240          assert len(keys) == 3 * 4 * 3 * 3 * 3 * 22
  241          audit.checks["stage_csv_rows"] = len(keys)
  242      result = {"status": "PASS_ER_WEIGHT_SAVED_RESULT_AUDIT", "completed_at_utc": datetime.now(timezone.utc).isoformat(), "elapsed_seconds": time.monotonic() - started,
  243                "cpu_affinity": sorted(os.sched_getaffinity(0)), "script": record(Path(__file__)), "helper": record(Path(__file__).with_name("step28_bge_continual_audit.py")),
  244                "numeric_comparisons": audit.numbers, "maximum_absolute_error": audit.max_error, "checks": dict(audit.checks), "comparisons": comparisons, "selection": saved["selection"],
  245                "new_label_parses": dict(train=0, valid=0, heldout=0, owners=0), "formal_texts_or_models_loaded": False,
  246                "limitations": ["Saved aggregate metrics audited; label-dependent AP/MAP not recomputed from truth.", "Native execution not repeated; weights hashed separately without model loading.", "Single seed, developed valid and selection-conditional intervals; no independent test or method novelty claim."]}
  247      for name, obj in (("audit.json", result), ("diagnostics.json", diagnostics)):
  248          (args.output / name).write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
  249      print(json.dumps(result, ensure_ascii=False, indent=2))
  250  
  251  
  252  if __name__ == "__main__":
  253      main()
