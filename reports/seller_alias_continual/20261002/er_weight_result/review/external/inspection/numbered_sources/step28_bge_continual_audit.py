    1  """Independent saved-result audit. No labels, texts, models, fitting or torch.
    2  
    3  Run on Linux py310. Frozen experimental code is intentionally not imported.
    4  This checks saved metric algebra and execution evidence; it does not reconstruct
    5  label-dependent AP/MAP from blind scores or claim a second native execution.
    6  """
    7  from __future__ import annotations
    8  
    9  import argparse
   10  from collections import Counter
   11  import csv
   12  from datetime import datetime, timezone
   13  import hashlib
   14  import json
   15  import math
   16  import os
   17  from pathlib import Path
   18  import time
   19  
   20  import numpy as np
   21  
   22  ORDERS = ("ABC", "BCA", "CAB")
   23  ARMS = ("frozen", "seq", "er", "logit")
   24  ROLES = ("raw", "stage-cal", "first-cal", "primary")
   25  METRICS = (
   26      "average_precision", "trapezoidal_pr_auc", "roc_auc", "recall_at_fpr_1pct",
   27      "brier", "log_loss", "precision", "recall", "f1", "specificity",
   28      "balanced_accuracy", "mcc", "map", "mrr", "recall_at_1", "recall_at_3",
   29      "recall_at_5", "recall_at_10", "ndcg_at_1", "ndcg_at_3", "ndcg_at_5", "ndcg_at_10",
   30  )
   31  TERMS = {
   32      "O": [(3, 1, .5), (3, 2, .5)],
   33      "N": [(2, 2, .5), (3, 3, .5)],
   34      "Z": [(3, 3, 1.)],
   35      "F_first": [(1, 1, 1.), (3, 1, -1.)],
   36      "F": [(1, 1, .5), (3, 1, -.5), (2, 2, .5), (3, 2, -.5)],
   37      "G": [(2, 2, .5), (1, 2, -.5), (3, 3, .5), (2, 3, -.5)],
   38      "final_all": [(3, 1, 1 / 3), (3, 2, 1 / 3), (3, 3, 1 / 3)],
   39  }
   40  RANK = list(range(4)) + list(range(12, 22))
   41  
   42  
   43  def read(path: Path):
   44      return json.loads(path.read_text(encoding="utf-8"))
   45  
   46  
   47  def record(path: Path) -> dict:
   48      payload = path.read_bytes()
   49      return {"bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}
   50  
   51  
   52  def bound(root: Path, info: dict) -> Path:
   53      path = (root / info["path"]).resolve()
   54      path.relative_to(root.resolve())
   55      if record(path) != {k: info[k] for k in ("bytes", "sha256")}:
   56          raise ValueError("File binding differs: " + str(path))
   57      return path
   58  
   59  
   60  def point(order: str, arm: str, stage: int) -> str:
   61      return order + "_shared" if stage == 1 or arm == "frozen" else f"{order}_{arm}_stage{stage}"
   62  
   63  
   64  def confusion(row: dict) -> dict:
   65      tp, fp, fn, tn = (int(row[k]) for k in ("tp", "fp", "fn", "tn"))
   66      den = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
   67      recall = tp / (tp + fn) if tp + fn else 0.
   68      spec = tn / (tn + fp) if tn + fp else 0.
   69      return {"precision": tp / (tp + fp) if tp + fp else 0., "recall": recall,
   70              "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.,
   71              "specificity": spec, "balanced_accuracy": (recall + spec) / 2,
   72              "mcc": (tp * tn - fp * fn) / den if den else 0.}
   73  
   74  
   75  class Audit:
   76      def __init__(self):
   77          self.numbers = 0
   78          self.max_error = 0.
   79          self.checks = Counter()
   80  
   81      def equal(self, actual, expected, name: str, tolerance: float = 3e-12):
   82          if isinstance(expected, dict):
   83              assert set(actual) == set(expected), (name, "keys")
   84              for key in expected:
   85                  self.equal(actual[key], expected[key], name + "/" + key, tolerance)
   86          elif isinstance(expected, (list, tuple)):
   87              assert len(actual) == len(expected), (name, "length")
   88              for i, value in enumerate(expected):
   89                  self.equal(actual[i], value, name + "/" + str(i), tolerance)
   90          elif isinstance(expected, (int, float)) and not isinstance(expected, bool):
   91              error = abs(float(actual) - float(expected))
   92              assert math.isfinite(error) and error <= tolerance, (name, actual, expected, error)
   93              self.numbers += 1
   94              self.max_error = max(self.max_error, error)
   95          else:
   96              assert actual == expected, (name, actual, expected)
   97  
   98      def array(self, actual, expected, name: str, tolerance: float = 3e-12):
   99          assert actual.shape == expected.shape, name
  100          errors = np.abs(actual.astype(np.float64) - expected.astype(np.float64))
  101          maximum = float(errors.max(initial=0))
  102          assert np.isfinite(errors).all() and maximum <= tolerance, (name, maximum)
  103          self.numbers += actual.size
  104          self.max_error = max(self.max_error, maximum)
  105  
  106  
  107  def main() -> None:
  108      parser = argparse.ArgumentParser(description=__doc__)
  109      parser.add_argument("--project", type=Path, required=True)
  110      parser.add_argument("--job", type=Path, required=True)
  111      parser.add_argument("--inventory", type=Path, required=True)
  112      parser.add_argument("--output", type=Path, required=True)
  113      parser.add_argument("--cpu", type=int, default=47)
  114      args = parser.parse_args()
  115      os.sched_setaffinity(0, {args.cpu})
  116      os.nice(15)
  117      started = time.monotonic()
  118      args.output.mkdir(parents=True, exist_ok=False)
  119      audit = Audit()
  120      project, job = args.project.resolve(), args.job.resolve()
  121      run, ev = job / "run", job / "evaluation"
  122      inventory = read(args.inventory)
  123      for item in inventory["files"]:
  124          bound(project, item)
  125      audit.checks["returned_files"] = len(inventory["files"])
  126      manifest, collected, saved = read(run / "manifest.json"), read(ev / "collected.json"), read(ev / "evaluation.json")
  127      completion, gate = read(job / "completion.json"), read(job / "before_valid.json")
  128      assert completion["physical_updates"] == manifest["physical_updates"] == 6048
  129      assert manifest["gradient_group_presentations"] == 9504
  130      assert (job / "exit_status.txt").read_text().strip() == "0"
  131      assert not (job / "failure.json").exists()
  132      assert read(job / "access.json") == completion["label_parses"] == dict(train=1, valid=1, heldout=0, owners=0)
  133      assert gate["status"] == "PASS_COMPLETE_BLIND_GATE" and gate["points"] == 21
  134      assert gate["supervision"] == dict(train=1, valid=0, heldout=0, owners=0)
  135      bound(job, gate["manifest"])
  136      bound(job, completion["evaluation"])
  137      for item in read(job / "execution.json")["source_files"]:
  138          bound(project, item)
  139          audit.checks["unchanged_frozen_sources"] += 1
  140      assert tuple(collected["metric_columns"]) == METRICS
  141      domains = np.asarray(collected["domains"])
  142      groups = collected["group_ids"]
  143      assert len(groups) == len(set(groups)) == 60
  144      rows = {d: np.flatnonzero(domains == d) for d in "ABC"}
  145      assert all(len(r) == 20 for r in rows.values())
  146      partition = read(bound(run, manifest["partition"]))
  147      assert [x["group_uid"] for x in partition["development"]] == groups
  148      assert [x["domain"] for x in partition["development"]] == domains.tolist()
  149      uid_domain = {x["group_uid"]: x["domain"] for x in partition["fit"]}
  150      expected_points = {point(o, a, s) for o in ORDERS for a in ARMS for s in (1, 2, 3)}
  151      assert len(expected_points) == 21 and set(collected["points"]) == set(manifest["points"]) == expected_points
  152      matrices, counts, scores, points, mappings, training = {}, {}, {}, {}, {}, {}
  153      loss_error = 0.
  154      diagnostics = {"training": {}, "memory": {}, "calibration": {}}
  155      for name in sorted(expected_points):
  156          p = read(bound(run, manifest["points"][name]))
  157          points[name] = p
  158          assert p["name"] == name and p["completed_updates"] == 288 * p["stage"]
  159          assert p["full_model_adam_and_rng_restore_verified"] is True
  160          m = read(bound(run, p["mapping"]))
  161          mappings[name] = m
  162          assert m["a"] > 0 and m["status"] == "PASS_CALIBRATION_FIT" and m["optimizer_success"]
  163          assert (m["group_count"], m["pair_count"], m["positive_count"]) == (12, 4536, 240)
  164          assert m["actual_domain"] == p["actual_domain"] == p["order"][p["stage"] - 1]
  165          assert m["calibration_group_ids"] == [x["group_uid"] for x in partition["calibration"] if x["domain"] == p["actual_domain"]]
  166          assert m["model_state_sha256"] == p["model_state_sha256"]
  167          assert m["score_source"] == p["scores"]["calibration"]
  168          scores[name] = {r: np.load(bound(run, f), allow_pickle=False) for r, f in p["scores"].items()}
  169          for role, value in scores[name].items():
  170              assert value.shape == ((12, 378) if role == "calibration" else (60, 378)) and np.isfinite(value).all()
  171              assert value.dtype == (np.float32 if role in ("calibration", "development") else np.float64)
  172          raw = scores[name]["development"].astype(np.float64)
  173          for role, mapping in (("stage-cal", m), ("first-cal", p["first_map_parameters"])):
  174              transformed = raw * mapping["a"] + mapping["b"]
  175              audit.array(scores[name][role], transformed, name + role, 0.)
  176              ordering = np.argsort(raw, axis=1, kind="stable")
  177              assert np.array_equal(ordering, np.argsort(transformed, axis=1, kind="stable"))
  178              assert np.array_equal(np.diff(np.take_along_axis(raw, ordering, axis=1)) == 0,
  179                                    np.diff(np.take_along_axis(transformed, ordering, axis=1)) == 0)
  180              audit.checks["affine_group_orders_and_ties"] += 60
  181          diagnostics["calibration"][name] = {k: m[k] for k in ("a", "b", "initial_nll", "final_nll", "raw_brier", "calibrated_brier")}
  182          t = read(bound(run, manifest["training"][name]))
  183          training[name] = t
  184          assert t["updates"] == 288 and t["adam_step"] == p["completed_updates"]
  185          assert t["encoder_positive_lr_updates"] == 287
  186          expected_ids = {uid for uid, domain in uid_domain.items() if domain == p["actual_domain"]}
  187          assert len(expected_ids) == 48 and Counter(t["current_ids"]) == Counter({uid: 6 for uid in expected_ids})
  188          for start in range(0, 288, 48):
  189              assert set(t["current_ids"][start:start + 48]) == expected_ids
  190          x = np.load(bound(run, t["update_file"]), allow_pickle=False)
  191          assert x.shape == (288, 14) and np.isfinite(x).all()
  192          c = {column: x[:, k] for k, column in enumerate(t["update_columns"])}
  193          for prefix in ("current", "history"):
  194              delta = c[prefix + "_total"] - c[prefix + "_bce"] - c[prefix + "_rank"] - .5 * c[prefix + "_hard"]
  195              loss_error = max(loss_error, float(np.abs(delta).max()))
  196              assert np.allclose(c[prefix + "_total"], c[prefix + "_bce"] + c[prefix + "_rank"] + .5 * c[prefix + "_hard"], atol=2e-6, rtol=1e-6)
  197          audit.array(c["total"], c["current_total"] + c["history_total"] + .5 * c["logit_mse"], name + "/loss")
  198          lr = np.array([1e-5 * (s / 29 if s <= 29 else (288 - s) / 259) for s in range(1, 289)])
  199          audit.array(c["encoder_lr"], lr, name + "/lr")
  200          audit.array(c["head_lr"], np.full(288, .001), name + "/head_lr")
  201          for step in (1, 29, 30, 288):
  202              for module in ("encoder", "head"):
  203                  obs = t["observations"][str(step)][module]
  204                  assert obs["finite_nonzero_combined_gradient"] is True
  205                  assert obs["parameters_changed"] == (module == "head" or step < 288)
  206          if name.endswith("shared") or "_seq_" in name:
  207              assert not t["history_ids"]
  208              assert np.count_nonzero(x[:, 4:9]) == 0
  209          else:
  210              assert len(t["history_ids"]) == 288
  211              memory = read(run / "memory" / (name + "_budget.json"))
  212              assert 0 < memory["serialized_bytes"] <= 1048576 and len(memory["members"]) == 6
  213              assert memory["seen"] == 48 * (t["stage"] - 1)
  214              assert set(t["history_ids"]) <= set(memory["members"])
  215              assert not set(memory["members"]) & expected_ids
  216              diagnostics["memory"][name] = {
  217                  "bytes_including_auxiliary": memory["serialized_bytes"],
  218                  "domain_composition": dict(Counter(uid_domain[uid] for uid in memory["members"])),
  219                  "history_presentations_by_domain": dict(Counter(uid_domain[uid] for uid in t["history_ids"])),
  220                  "draw_count_range": [min(Counter(t["history_ids"]).values()), max(Counter(t["history_ids"]).values())],
  221                  "references": memory["references"], "origins": memory["reference_origins"],
  222              }
  223          diagnostics["training"][name] = {
  224              "seconds": t["training_seconds"],
  225              "epoch_means": [{col: float(values[e * 48:(e + 1) * 48].mean()) for col, values in c.items()} for e in range(6)],
  226          }
  227          audit.checks["physical_updates"] += 288
  228          audit.checks["history_gradient_presentations"] += len(t["history_ids"])
  229      for order in ORDERS:
  230          first = mappings[order + "_shared"]
  231          for name, p in points.items():
  232              if p["order"] == order:
  233                  audit.equal(p["first_map_parameters"], {k: first[k] for k in ("a", "b")}, name + "/first_map")
  234          for stage in (2, 3):
  235              paired = [training[point(order, arm, stage)] for arm in ("seq", "er", "logit")]
  236              assert all(t["current_ids"] == paired[0]["current_ids"] and t["current_dropout_stream"] == paired[0]["current_dropout_stream"] for t in paired)
  237              assert paired[1]["history_ids"] == paired[2]["history_ids"]
  238              a, b = (diagnostics["memory"][point(order, arm, stage)] for arm in ("er", "logit"))
  239              assert a["domain_composition"] == b["domain_composition"]
  240          early = read(run / "memory" / (order + "_logit_stage2_budget.json"))
  241          late = read(run / "memory" / (order + "_logit_stage3_budget.json"))
  242          for uid in set(early["references"]) & set(late["references"]):
  243              assert early["references"][uid] == late["references"][uid] and late["reference_origins"][uid] == 1
  244      assert audit.checks["physical_updates"] == 6048 and audit.checks["history_gradient_presentations"] == 3456
  245      entries = {"initial": {"raw": collected["initial"]}, **collected["points"]}
  246      scores["initial"] = {"development": np.load(bound(run, manifest["initial"]["scores"]), allow_pickle=False)}
  247      for name, variants in entries.items():
  248          matrices[name], counts[name] = {}, {}
  249          for role, info in variants.items():
  250              value = np.load(bound(ev, info["matrix"]), allow_pickle=False)
  251              assert value.shape == (60, 22) and value.dtype == np.float64 and np.isfinite(value).all()
  252              rows_counts = read(bound(ev, info["counts"]))
  253              assert len(rows_counts) == 60
  254              score = scores[name]["development" if role == "raw" else role]
  255              for i, row in enumerate(rows_counts):
  256                  assert all(type(row[k]) is int and row[k] >= 0 for k in ("tp", "fp", "fn", "tn"))
  257                  assert row["tp"] + row["fn"] == 20 and row["fp"] + row["tn"] == 358
  258                  assert int((score[i] >= 0).sum()) == row["tp"] + row["fp"]
  259                  for metric, expected in confusion(row).items():
  260                      audit.equal(value[i, METRICS.index(metric)], expected, name + role + metric)
  261              matrices[name][role], counts[name][role] = value, rows_counts
  262              audit.checks["matrix_count_sets"] += 1
  263          if name != "initial":
  264              for role in ("stage-cal", "first-cal"):
  265                  audit.array(matrices[name][role][:, RANK], matrices[name]["raw"][:, RANK], name + "/rank_invariance", 0.)
  266              matrices[name]["primary"] = matrices[name]["stage-cal"].copy()
  267              matrices[name]["primary"][:, RANK] = matrices[name]["raw"][:, RANK]
  268      draws = np.random.Generator(np.random.PCG64(20260930)).integers(0, 20, (5000, 3, 20))
  269      assert np.array_equal(draws, np.load(bound(ev, saved["draws"]), allow_pickle=False))
  270      weights = np.zeros((5000, 60))
  271      for d, domain in enumerate("ABC"):
  272          weights[:, rows[domain]] = np.stack([np.bincount(row, minlength=20) for row in draws[:, d]]) / 20
  273  
  274      def field(arm: str, role: str, endpoint: str) -> np.ndarray:
  275          result = np.zeros((3, 60, 22))
  276          for oi, order in enumerate(ORDERS):
  277              for stage, arrival, coefficient in TERMS[endpoint]:
  278                  selected = rows[order[arrival - 1]]
  279                  result[oi, selected] += coefficient * matrices[point(order, arm, stage)][role][selected]
  280          if endpoint in ("F_first", "F", "G"):
  281              result[:, :, [4, 5]] *= -1
  282          return result
  283  
  284      def summary(value: np.ndarray) -> dict:
  285          per_order = value.sum(axis=1) / 20
  286          average = per_order.mean(axis=0)
  287          replicates = weights @ value.mean(axis=0)
  288          ordered = np.sort(replicates, axis=0)
  289          ci = []
  290          for fraction in (.025, .975):
  291              position = fraction * 4999
  292              lo, hi = math.floor(position), math.ceil(position)
  293              ci.append(ordered[lo] + (position - lo) * (ordered[hi] - ordered[lo]))
  294          return {m: {"mean": float(average[k]), "per_order": {o: float(per_order[i, k]) for i, o in enumerate(ORDERS)},
  295                      "conditional_95pct_interval": [float(ci[0][k]), float(ci[1][k])]} for k, m in enumerate(METRICS)}
  296  
  297      # Independent small algebra checks, without the experiment's helpers.
  298      assert confusion(dict(tp=2, fp=1, fn=2, tn=5))["recall"] == .5
  299      handmade = np.zeros((3, 60, 22))
  300      handmade[:, rows["A"]] = 2
  301      handmade[:, rows["B"]] = 4
  302      handmade[:, rows["C"]] = 6
  303      audit.equal(summary(handmade)["map"], {"mean": 12., "per_order": dict.fromkeys(ORDERS, 12.), "conditional_95pct_interval": [12., 12.]}, "handmade_bootstrap")
  304      recomputed = {arm: {role: {ep: summary(field(arm, role, ep)) for ep in TERMS} for role in ROLES} for arm in ARMS}
  305      audit.equal(saved["endpoints"], recomputed, "endpoints")
  306      comparisons = {}
  307      signs = dict(average_precision=1, roc_auc=1, brier=-1, log_loss=-1)
  308      for candidate, reference in (("er", "seq"), ("logit", "er")):
  309          name = candidate + "_minus_" + reference
  310          delta = {ep: summary(field(candidate, "primary", ep) - field(reference, "primary", ep)) for ep in TERMS}
  311          raw_delta = {ep: summary(field(candidate, "primary", ep) - field(reference, "raw", ep)) for ep in ("O", "N")}
  312          audit.equal(saved["comparisons"][name]["primary"], delta, name)
  313          audit.equal(saved["comparisons"][name]["against_raw_reference"], raw_delta, name + "/raw")
  314          checks = {
  315              "old_map_improves": delta["O"]["map"]["mean"] > 0,
  316              "old_map_interval_above_zero": delta["O"]["map"]["conditional_95pct_interval"][0] > 0,
  317              "old_recall5_improves": delta["O"]["recall_at_5"]["mean"] > 0,
  318              "new_map_non_decrease": delta["N"]["map"]["mean"] >= 0,
  319              "new_recall5_non_decrease": delta["N"]["recall_at_5"]["mean"] >= 0,
  320          }
  321          for ep in ("O", "N"):
  322              for m, sign in signs.items():
  323                  checks[f"{ep}_{m}_non_degradation"] = sign * delta[ep][m]["mean"] >= 0
  324              for m in ("brier", "log_loss"):
  325                  checks[f"{ep}_{m}_against_raw_reference"] = raw_delta[ep][m]["mean"] <= 0
  326          for m, sign in dict(map=1, recall_at_5=1, **signs).items():
  327              checks[f"Z_{m}_non_degradation"] = sign * delta["Z"][m]["mean"] >= 0
  328          verdict = saved["comparisons"][name]["interpretation"]
  329          audit.equal(verdict["checks"], checks, name + "/checks")
  330          assert verdict["pilot_observed_checks_pass"] == all(checks.values())
  331          assert set(verdict["failed"]) == {k for k, value in checks.items() if not value}
  332          assert verdict["positive_old_map_orders"] == sum(x > 0 for x in delta["O"]["map"]["per_order"].values())
  333          comparisons[name] = {"passed": sum(checks.values()), "total": len(checks), "checks": checks}
  334      first = recomputed["seq"]["raw"]["F_first"]["map"]
  335      gain = recomputed["seq"]["raw"]["G"]["map"]
  336      matched = [o for o in ORDERS if first["per_order"][o] > 0 and gain["per_order"][o] > 0]
  337      fg = dict(seq_first_MAP_loss_positive=first["mean"] > 0,
  338                seq_first_MAP_loss_interval_above_zero=first["conditional_95pct_interval"][0] > 0,
  339                same_order_path_forgetting_and_new_learning=len(matched) >= 2)
  340      audit.equal(saved["seq_forgetting"]["checks"], fg, "forgetting_checks")
  341      assert saved["seq_forgetting"]["matched_orders"] == matched
  342      assert saved["seq_forgetting"]["established"] == all(fg.values())
  343      for name in expected_points:
  344          for role in ROLES[:3]:
  345              value = matrices[name][role]
  346              obj = saved["absolute_stage_results"][name][role]
  347              audit.equal(obj["macro_all"], dict(zip(METRICS, value.mean(0).tolist())), name + role + "/macro")
  348              for d in "ABC":
  349                  audit.equal(obj["macro_by_domain"][d], dict(zip(METRICS, value[rows[d]].mean(0).tolist())), name + role + d)
  350              pooled = obj["pooled_fixed_half_classification"]
  351              assert pooled["threshold"] == 0
  352              for d, selected in {**rows, "pooled": np.arange(60)}.items():
  353                  total = {k: sum(counts[name][role][i][k] for i in selected) for k in ("tp", "fp", "fn", "tn")}
  354                  rates = confusion(total)
  355                  expected = {**total, **{k: rates[k] for k in ("precision", "recall", "f1")}, "fpr": total["fp"] / (total["fp"] + total["tn"])}
  356                  audit.equal(pooled["pooled"] if d == "pooled" else pooled["by_domain"][d], expected, name + role + "/pooled" + d)
  357      for order in ORDERS:
  358          selected = rows[order[0]]
  359          initial, trained = matrices["initial"]["raw"][selected], matrices[order + "_shared"]["raw"][selected]
  360          for key, value in (("initial_raw", initial), ("first_raw", trained), ("raw_first_minus_initial", trained - initial)):
  361              audit.equal(saved["initial_to_first_learning"][order][key], dict(zip(METRICS, value.mean(0).tolist())), order + key)
  362      with bound(ev, saved["stage_metrics"]).open(encoding="utf-8", newline="") as stream:
  363          seen = set()
  364          for row in csv.DictReader(stream):
  365              name = row["source_point"]
  366              assert row["seed"] == "s0" and row["actual_domain"] == row["order"][int(row["arrival_index"]) - 1]
  367              if int(row["stage"]) == 0:
  368                  assert name == row["method"] == "initial" and row["output_role"] == "raw"
  369              else:
  370                  assert name == point(row["order"], row["method"], int(row["stage"]))
  371              key = tuple(row[k] for k in ("order", "actual_domain", "stage", "method", "output_role", "metric"))
  372              assert key not in seen
  373              seen.add(key)
  374              value = matrices[name][row["output_role"]][rows[row["actual_domain"]], METRICS.index(row["metric"])].mean()
  375              audit.equal(float(row["group_macro"]), float(value), "stage_table")
  376          assert len(seen) == 198 + 3 * 3 * 4 * 3 * 3 * 22
  377          audit.checks["stage_csv_rows"] = len(seen)
  378      with (args.output / "endpoints.csv").open("x", encoding="utf-8", newline="") as stream:
  379          writer = csv.writer(stream, lineterminator="\n")
  380          writer.writerow(("method", "role", "endpoint", "metric", "mean", "ci_lower", "ci_upper", *ORDERS))
  381          for arm in ARMS:
  382              for role in ROLES:
  383                  for ep, summary_value in recomputed[arm][role].items():
  384                      for metric, obj in summary_value.items():
  385                          writer.writerow((arm, role, ep, metric, obj["mean"], *obj["conditional_95pct_interval"], *(obj["per_order"][o] for o in ORDERS)))
  386      diagnostics["descriptive_logit_minus_seq"] = {
  387          ep: {m: recomputed["logit"]["primary"][ep][m]["mean"] - recomputed["seq"]["primary"][ep][m]["mean"] for m in METRICS}
  388          for ep in TERMS}
  389      diagnostics["descriptive_scope"] = "Absolute differences only; LOGIT-minus-SEQ is not an added prospective test or acceptance rule."
  390      result = {
  391          "status": "PASS_SAVED_RESULT_AUDIT", "completed_at_utc": datetime.now(timezone.utc).isoformat(),
  392          "elapsed_seconds": time.monotonic() - started, "cpu_affinity": sorted(os.sched_getaffinity(0)),
  393          "job": str(job.relative_to(project)), "script": record(Path(__file__)),
  394          "checks": dict(audit.checks), "numeric_comparisons": audit.numbers, "maximum_absolute_error": audit.max_error,
  395          "float32_objective_identity_max_error": loss_error, "comparisons": comparisons,
  396          "seq_first_forgetting": first, "seq_new_learning": gain,
  397          "new_label_parses": dict(train=0, valid=0, heldout=0, owners=0),
  398          "formal_texts_or_weights_loaded": False,
  399          "limitations": ["No label-dependent AP/MAP recomputation; saved matrices are the authorized collection.",
  400                          "No refit, native model rerun or full history payload read; execution evidence and summary bindings checked.",
  401                          "Conditional 5000-draw intervals, reused 60 development groups, one training seed.",
  402                          "No per-domain noninferiority, new method qualification or independent heldout claim."],
  403      }
  404      for filename, payload in (("audit.json", result), ("diagnostics.json", diagnostics)):
  405          with (args.output / filename).open("x", encoding="utf-8", newline="\n") as stream:
  406              json.dump(payload, stream, ensure_ascii=False, indent=2, allow_nan=False)
  407              stream.write("\n")
  408      print(json.dumps(result, ensure_ascii=False, indent=2))
  409  
  410  
  411  if __name__ == "__main__":
  412      main()
