    1  """Windows valid-only evaluation after complete blind output qualification.
    2  
    3  Pure curve/classification/retrieval functions preserve the audited text evaluator
    4  definitions. No old archive, consumed label parser or identity stratification is
    5  imported. The Linux-only model receipt operation hashes outputs without labels.
    6  """
    7  from __future__ import annotations
    8  
    9  import argparse
   10  import csv
   11  import math
   12  from collections import defaultdict
   13  from pathlib import Path
   14  from typing import Any
   15  
   16  import numpy as np
   17  
   18  import step28_continual_population_data as data
   19  import step28_continual_population_run as runner
   20  
   21  KS = (1, 3, 5, 10)
   22  CURVE_KEYS = ("average_precision", "trapezoidal_pr_auc", "roc_auc", "recall_at_fpr_1pct")
   23  CLASS_KEYS = CURVE_KEYS + ("brier", "log_loss", "precision", "recall", "f1",
   24                           "specificity", "balanced_accuracy", "mcc")
   25  RETRIEVAL_KEYS = ("map", "mrr") + tuple(f"recall_at_{k}" for k in KS) + tuple(f"ndcg_at_{k}" for k in KS)
   26  COLUMNS = CLASS_KEYS + RETRIEVAL_KEYS
   27  
   28  def curve_metrics(y: np.ndarray, scores: np.ndarray) -> dict[str, float]:
   29      y, scores = np.asarray(y), np.asarray(scores, dtype=np.float64)
   30      if (y.ndim != 1 or y.shape != scores.shape or not len(y) or not np.isin(y, [0, 1]).all()
   31              or not np.isfinite(scores).all() or not 0 < y.sum() < len(y)):
   32          raise ValueError("Curve inputs need aligned finite scores and both classes")
   33      order = np.argsort(-scores, kind="stable")
   34      ordered = scores[order]
   35      end = np.r_[ordered[1:] != ordered[:-1], True]
   36      tp = np.cumsum(y[order], dtype=np.float64)[end]
   37      fp = np.cumsum(1 - y[order], dtype=np.float64)[end]
   38      recall, fpr = tp / y.sum(), fp / (len(y) - y.sum())
   39      precision = tp / (tp + fp)
   40      dr = np.diff(np.r_[0., recall])
   41      return {"average_precision": float(dr @ precision),
   42              "trapezoidal_pr_auc": float(dr @ ((precision + np.r_[1., precision[:-1]]) / 2)),
   43              "roc_auc": float(np.diff(np.r_[0., fpr]) @ ((recall + np.r_[0., recall[:-1]]) / 2)),
   44              "recall_at_fpr_1pct": float(np.max(np.r_[0., recall[fpr <= .01]]))}
   45  
   46  
   47  def confusion_metrics(tp: float, fp: float, fn: float, tn: float) -> dict[str, float]:
   48      precision = tp / (tp + fp) if tp + fp else 0.
   49      recall = tp / (tp + fn) if tp + fn else 0.
   50      specificity = tn / (tn + fp) if tn + fp else 0.
   51      denominator = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
   52      return {"precision": precision, "recall": recall, "specificity": specificity,
   53              "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.,
   54              "balanced_accuracy": (recall + specificity) / 2,
   55              "mcc": (tp * tn - fp * fn) / denominator if denominator else 0.}
   56  
   57  
   58  def classification(y: np.ndarray, scores: np.ndarray, threshold: float = 0.) -> dict[str, Any]:
   59      result: dict[str, Any] = curve_metrics(y, scores)
   60      scores = np.asarray(scores, dtype=np.float64)
   61      probability = np.exp(-np.logaddexp(0., -scores))
   62      clipped = np.clip(probability, 1e-15, 1 - 1e-15)
   63      result["brier"] = float(np.mean((probability - y) ** 2))
   64      result["log_loss"] = float(-np.mean(y * np.log(clipped) + (1 - y) * np.log1p(-clipped)))
   65      predicted, positive = scores >= threshold, y == 1
   66      tp, fp = int(np.sum(predicted & positive)), int(np.sum(predicted & ~positive))
   67      fn, tn = int(np.sum(~predicted & positive)), int(np.sum(~predicted & ~positive))
   68      result.update(confusion_metrics(tp, fp, fn, tn))
   69      result["confusion"] = {"tp": tp, "fp": fp, "fn": fn, "tn": tn}
   70      return result
   71  
   72  
   73  def retrieval(y: np.ndarray, scores: np.ndarray, sellers: int) -> np.ndarray:
   74      """Complete sorted K_n edges; score ties use ascending public seller order."""
   75      y, scores = np.asarray(y), np.asarray(scores, dtype=np.float64)
   76      pairs = sellers * (sellers - 1) // 2
   77      if y.shape != scores.shape or y.ndim != 2 or y.shape[1] != pairs or not np.isfinite(scores).all():
   78          raise ValueError("Retrieval rows differ from complete worlds")
   79      if not np.isin(y, [0, 1]).all():
   80          raise ValueError("Nonbinary relevance")
   81      left, right = np.triu_indices(sellers, 1)
   82      adjacency = np.full((len(y), sellers, sellers), -np.inf)
   83      relevant = np.zeros((len(y), sellers, sellers), dtype=np.uint8)
   84      adjacency[:, left, right] = adjacency[:, right, left] = scores
   85      relevant[:, left, right] = relevant[:, right, left] = y
   86      order = np.argsort(-adjacency, axis=2, kind="stable")[:, :, :sellers - 1]
   87      ranked = np.take_along_axis(relevant, order, axis=2)
   88      total = relevant.sum(axis=2)
   89      if not np.all((total >= 1) & (total <= 2)):
   90          raise ValueError("Each qualified query must have one or two relevant candidates")
   91      ranks = np.arange(1, sellers)
   92      ap = (np.cumsum(ranked, axis=2) * ranked / ranks).sum(axis=2) / total
   93      rr = 1. / (np.argmax(ranked, axis=2) + 1)
   94      values = [ap, rr]
   95      values.extend(ranked[:, :, :k].sum(axis=2) / total for k in KS)
   96      discounts = 1. / np.log2(ranks + 1)
   97      for k in KS:
   98          ideal = np.asarray([discounts[:min(k, int(n))].sum() for n in total.ravel()]).reshape(total.shape)
   99          values.append((ranked[:, :, :k] * discounts[:k]).sum(axis=2) / ideal)
  100      return np.stack([v.mean(axis=1) for v in values], axis=1)
  101  
  102  
  103  def verify_models(run: Path, output: Path) -> dict:
  104      """Read-only Linux payload verification receipt for the Windows copy stage."""
  105      manifest = data.read_json(run / "manifest.json")
  106      if manifest["status"] != runner.COMPLETE or output.exists():
  107          raise ValueError("Requires a complete run and a new receipt file")
  108      found = []
  109      for order in manifest["orders"]:
  110          for record in order["models"].values():
  111              path = data.verify(run / record["path"], record)
  112              found.append(data.record(path, run))
  113      if len(found) != 12:
  114          raise ValueError("Missing final inference payload")
  115      result = {"status": "TWELVE_LINUX_MODEL_PAYLOADS_HASH_VERIFIED_NO_LABELS",
  116                "manifest_sha256": data.sha256(run / "manifest.json"),
  117                "files": sorted(found, key=lambda r: r["path"])}
  118      data.write_json(output, result)
  119      return result
  120  
  121  
  122  def require_memory_points(points: dict) -> None:
  123      actual = {name for name, record in points.items() if "memory" in record}
  124      if actual != {"shared", "er_stage2", "er_stage3"}:
  125          raise ValueError("Missing/extra required historical memory evidence")
  126  
  127  
  128  def validate_run(run: Path, config: dict, archive: data.Archive) -> tuple[dict, dict]:
  129      if (run / "failure.json").exists():
  130          raise ValueError("Failed run cannot be evaluated")
  131      manifest = data.read_json(run / "manifest.json")
  132      if (manifest["status"] != runner.COMPLETE or manifest["config"] != config
  133              or manifest["source_files"] != runner.sources()
  134              or manifest["label_reads"] != {"new_train_csv_offline_packaging": 1,
  135                                             "development": 0, "heldout": 0, "owners": 0, "old_assets": 0}
  136              or manifest["physical_updates"] != 5400
  137              or [o["order"] for o in manifest["orders"]] != config["orders"]
  138              or manifest["runtime_check"]["status"] != "PASS_REAL_LABSE_NEW_INPUT_HEAD_MEMORY_ADAM_RELOAD"
  139              or manifest["runtime_check"]["torch_contracts"] != {"passed": 3, "skipped": 0, "failed": 0}
  140              or manifest["runtime_check"]["formal_initialization_restored"] is not True):
  141          raise ValueError("Incomplete, stale, or unauthorized execution")
  142      if (manifest["budget"]["elapsed_seconds"] > config["runtime"]["maximum_gpu_stage_seconds"]
  143              or manifest["budget"]["peak_observed_bytes"] > config["runtime"]["maximum_output_bytes"]):
  144          raise ValueError("Unapproved resource expansion")
  145      for split in ("development", "heldout"):
  146          if manifest["evaluation_group_ids"][split] != [g.uid for g in archive.groups(split)]:
  147              raise ValueError("Blind score row identities differ")
  148      expected_inputs = {**archive.checked,
  149                         "train/supervision/pairs.csv": archive.manifest["files"]["train/supervision/pairs.csv"]}
  150      if manifest["verified_input_files"] != expected_inputs:
  151          raise ValueError("Training inputs/provenance differ")
  152      for key in ("file_count", "total_size_bytes", "content_sha256"):
  153          if manifest["pretrained_model"][key] != config["model"][key]:
  154              raise ValueError("Pretrained model source differs")
  155      receipt = data.read_json(run / "model_verification.json")
  156      expected_models = []
  157      scores: dict = {}
  158      for order in manifest["orders"]:
  159          order_id = order["order"]
  160          expected_points = {"initial", "shared"} | {f"{arm}_stage{s}" for arm in runner.ARMS for s in (2, 3)}
  161          if set(order["points"]) != expected_points or set(order["models"]) != {"frozen", *runner.ARMS}:
  162              raise ValueError("Missing/extra point or model")
  163          require_memory_points(order["points"])
  164          for arm, record in order["models"].items():
  165              if record["actual_loaded_model_equals_replayed_state"] is not True:
  166                  raise ValueError("Inference payload not actually loaded and compared")
  167              if record["path"] != f"models/{order_id}_{arm}.pt":
  168                  raise ValueError("Model point mapping differs")
  169              expected_models.append({k: record[k] for k in ("path", "bytes", "sha256")})
  170          if order["trajectory"]["frozen"] != ["initial", "shared", "shared", "shared"]:
  171              raise ValueError("Frozen trajectory differs")
  172          for arm in runner.ARMS:
  173              if order["trajectory"][arm] != ["initial", "shared", f"{arm}_stage2", f"{arm}_stage3"]:
  174                  raise ValueError("Model trajectory differs")
  175          for arm, stage, log in [("shared", 1, order["training"]["shared"])] + [
  176                  (arm, s, order["training"][arm][s - 2]) for arm in runner.ARMS for s in (2, 3)]:
  177              domains = order_id[:stage] if arm == "cumulative" else order_id[stage - 1]
  178              seed = data.seed_for(config["initialization_seed"], order_id, stage,
  179                                   "cumulative" if arm == "cumulative" else "current")
  180              rows = data.schedule(archive.groups("train", domains), 3, seed)
  181              if (log["current_group_ids"] != [g.uid for g in rows]
  182                      or log["current_dropout_stream"] != seed
  183                      or log["updates"] != len(rows) or log["current_pair_presentations"] != len(rows) * 378
  184                      or log["memory_frozen_during_stage"] is not True):
  185                  raise ValueError("Training supply, schedule or update counts differ")
  186              history = log["replay_group_ids"]
  187              if arm == "er":
  188                  previous = "shared" if stage == 2 else "er_stage2"
  189                  available = order["points"][previous]["memory"]["retained_groups"]
  190                  if len(history) != len(rows) or not set(history).issubset(available):
  191                      raise ValueError("Replay accessed an unretained/future group")
  192              elif history:
  193                  raise ValueError("Unexpected historical replay")
  194              if log["replay_pair_presentations"] != len(history) * 378:
  195                  raise ValueError("Replay computation accounting differs")
  196          scores[order_id] = {}
  197          for name, record in order["points"].items():
  198              if record["full_model_and_adam_reloaded"] is not True or set(record["scores"]) != {"development", "heldout"}:
  199                  raise ValueError("Incomplete actual checkpoint replay")
  200              if "memory" in record:
  201                  mem = record["memory"]
  202                  expected_seen = 60 if name == "shared" else 60 * int(name[-1])
  203                  if (mem["disk_roundtrip_exact"] is not True or mem["seen"] != expected_seen
  204                          or len(set(mem["retained_groups"])) != 6 or not 0 < mem["bytes"] <= 1048576):
  205                      raise ValueError("Historical memory evidence differs")
  206                  # Hash only; do not parse archived train labels during valid evaluation.
  207                  data.verify(run / mem["path"], mem)
  208              for split, expected in record["scores"].items():
  209                  if expected["path"] != f"scores/{order_id}_{name}_{split}.npy":
  210                      raise ValueError("Prediction point mapping differs")
  211                  path = data.verify(run / expected["path"], expected)
  212                  values = np.load(path, allow_pickle=False)
  213                  if (values.dtype != np.dtype("float32") or values.shape != (len(archive.groups(split)), 378)
  214                          or not np.isfinite(values).all()):
  215                      raise ValueError("Blind score dimensions or values differ")
  216                  if split == "development":
  217                      scores[order_id][name] = values
  218      if (receipt["status"] != "TWELVE_LINUX_MODEL_PAYLOADS_HASH_VERIFIED_NO_LABELS"
  219              or receipt["manifest_sha256"] != data.sha256(run / "manifest.json")
  220              or receipt["files"] != sorted(expected_models, key=lambda r: r["path"])):
  221          raise ValueError("Linux model verification is missing or belongs to another run")
  222      return manifest, scores
  223  
  224  
  225  def group_metrics(labels: np.ndarray, scores: np.ndarray) -> tuple[np.ndarray, list[dict]]:
  226      results = [classification(y, s) for y, s in zip(labels, scores, strict=True)]
  227      numeric = np.array([[row[k] for k in CLASS_KEYS] for row in results])
  228      return np.column_stack((numeric, retrieval(labels, scores, 28))), [r["confusion"] for r in results]
  229  
  230  
  231  def paired_interval(delta: np.ndarray, domain_rows: list[np.ndarray], config: dict) -> dict:
  232      """Average fixed orders first; bootstrap independent groups within each domain."""
  233      if delta.ndim != 2 or not np.isfinite(delta).all():
  234          raise ValueError("Paired differences must be finite order-by-group rows")
  235      flat = np.concatenate(domain_rows)
  236      if len(flat) != delta.shape[1] or set(flat.tolist()) != set(range(delta.shape[1])):
  237          raise ValueError("Domain bootstrap partition differs")
  238      rng = np.random.default_rng(config["bootstrap_seed"])
  239      averaged = delta.mean(axis=0)
  240      replicates = []
  241      for rows in domain_rows:
  242          indices = rng.choice(rows, size=(config["bootstrap_replicates"], len(rows)), replace=True)
  243          replicates.append(averaged[indices].mean(axis=1))
  244      boot = np.stack(replicates).mean(axis=0)
  245      tail = (1 - config["confidence_level"]) / 2
  246      return {"mean": float(np.mean([averaged[rows].mean() for rows in domain_rows])),
  247              "per_order": [float(np.mean([values[rows].mean() for rows in domain_rows])) for values in delta],
  248              "conditional_95pct_interval": np.quantile(boot, [tail, 1 - tail]).tolist(),
  249              "conditional_on": "This generated dataset, initialization and three fixed orders; independent groups only"}
  250  
  251  
  252  def old_domain_change(current: np.ndarray, when_learned: np.ndarray,
  253                        history: np.ndarray, rows: np.ndarray) -> dict:
  254      # R[t,j] averages independent groups FIRST, then takes a maximum over stages.
  255      # Averaging per-group maxima would overstate domain-level forgetting.
  256      current_mean = float(current[rows].mean())
  257      return {"current_minus_when_learned": current_mean - float(when_learned[rows].mean()),
  258              "best_previous_minus_current": float(history[:, rows].mean(axis=1).max()) - current_mean}
  259  
  260  
  261  def evaluate(run: Path, output: Path) -> dict:
  262      config = data.policy()
  263      if config["evaluation"]["split"] != "development" or config["evaluation"]["heldout_labels_allowed"]:
  264          raise ValueError("Only the confirmed valid-first stage is authorized")
  265      if output.exists():
  266          raise FileExistsError("Do not repeat label evaluation into an existing result directory")
  267      archive = data.Archive(config, train_labels=False)
  268      manifest, scores = validate_run(run, config, archive)
  269      groups = archive.groups("development")
  270      # All output/reload/supply gates above precede the single valid binary parse.
  271      output.mkdir(parents=True)
  272      data.write_json(output / "access.json", {"phase": "COMPLETE_OUTPUT_CHECKS_PASSED_BEFORE_VALID_PARSE",
  273                      "manifest_sha256": data.sha256(run / "manifest.json"), "heldout_labels": 0,
  274                      "train_labels": 0, "owners": 0, "old_assets": 0})
  275      path = data.verify(archive.root / "development/supervision/pairs.csv",
  276                         archive.manifest["files"]["development/supervision/pairs.csv"])
  277      grouped = defaultdict(list)
  278      with path.open(encoding="utf-8", newline="") as stream:
  279          for row in csv.DictReader(stream):
  280              grouped[row["group_uid"]].append(row)
  281      if set(grouped) != {g.uid for g in groups}:
  282          raise ValueError("Valid binary groups differ")
  283      labels = np.array([data.align_labels(g, grouped[g.uid]) for g in groups], dtype=np.uint8)
  284      if not np.all(labels.sum(axis=1) == 20):
  285          raise ValueError("Valid positive counts differ")
  286      domain_by_id = {r["group_uid"]: r["domain"] for r in archive.metadata}
  287      domain_rows = [np.array([i for i, g in enumerate(groups) if domain_by_id[g.uid] == d]) for d in "ABC"]
  288      result: dict = {"status": "VALID_EVALUATION_COMPLETE_TEST_TRUTH_UNREAD", "columns": COLUMNS,
  289                      "run_manifest_sha256": data.sha256(run / "manifest.json"),
  290                      "label_reads": {"development_binary_csv": 1, "heldout": 0, "train": 0, "owners": 0, "old_assets": 0},
  291                      "config": config, "groups": [g.uid for g in groups], "orders": {},
  292                      "threshold": {"logit": 0, "probability": 0.5},
  293                      "ranking": "raw logit; query score ties use ascending opaque account ID",
  294                      "recall_at_fpr_note": "Labeled-sample diagnostic, not a deployable threshold"}
  295      metrics = {}
  296      for order in manifest["orders"]:
  297          order_id = order["order"]
  298          metrics[order_id] = {}
  299          order_result: dict = {"points": {}, "transitions": {}}
  300          for name, values in scores[order_id].items():
  301              matrix, confusion = group_metrics(labels, values)
  302              metrics[order_id][name] = matrix
  303              target = output / f"{order_id}_{name}_metrics.npy"
  304              np.save(target, matrix, allow_pickle=False)
  305              domain = {d: dict(zip(COLUMNS, matrix[rows].mean(axis=0).tolist())) for d, rows in zip("ABC", domain_rows)}
  306              order_result["points"][name] = {
  307                  "domain_equal": dict(zip(COLUMNS, np.stack([matrix[r].mean(axis=0) for r in domain_rows]).mean(axis=0).tolist())),
  308                  "by_domain": domain, "per_group_confusion": confusion, "per_group_metrics": data.record(target, output),
  309                  "pooled_pair_metrics": classification(labels.ravel(), values.ravel())}
  310          for arm, trajectory in order["trajectory"].items():
  311              transitions = []
  312              for stage in (1, 2, 3):
  313                  now, before = (metrics[order_id][trajectory[t]][:, 0] for t in (stage, stage - 1))
  314                  new_rows = domain_rows["ABC".index(order_id[stage - 1])]
  315                  old = {}
  316                  for learned in range(1, stage):
  317                      domain = order_id[learned - 1]
  318                      rows = domain_rows["ABC".index(domain)]
  319                      at_learning = metrics[order_id][trajectory[learned]][:, 0]
  320                      previous = np.stack([metrics[order_id][trajectory[t]][:, 0] for t in range(learned, stage)])
  321                      old[domain] = old_domain_change(now, at_learning, previous, rows)
  322                  transitions.append({"stage": stage, "new_domain": order_id[stage - 1],
  323                                      "new_domain_after_minus_before": float((now[new_rows] - before[new_rows]).mean()),
  324                                      "old_domains": old})
  325              order_result["transitions"][arm] = transitions
  326          result["orders"][order_id] = order_result
  327      delta = np.stack([metrics[o]["er_stage3"][:, 0] - metrics[o]["sequential_stage3"][:, 0] for o in config["orders"]])
  328      result["primary_er_minus_sequential_ap"] = paired_interval(delta, domain_rows, config["evaluation"])
  329      result["final_fixed_order_equal"] = {
  330          arm: dict(zip(COLUMNS, np.stack([
  331              metrics[o]["shared" if arm == "frozen" else f"{arm}_stage3"].mean(axis=0)
  332              for o in config["orders"]]).mean(axis=0).tolist()))
  333          for arm in ("frozen", *runner.ARMS)}
  334      data.write_json(output / "evaluation.json", result)
  335      return {"status": result["status"], "primary": result["primary_er_minus_sequential_ap"], "output": str(output)}
  336  
  337  
  338  def main() -> None:
  339      parser = argparse.ArgumentParser(description=__doc__)
  340      parser.add_argument("mode", choices=("evaluate-valid", "verify-models"))
  341      parser.add_argument("--run", type=Path, required=True)
  342      parser.add_argument("--out", type=Path, required=True)
  343      args = parser.parse_args()
  344      result = verify_models(args.run, args.out) if args.mode == "verify-models" else evaluate(args.run, args.out)
  345      print(data.json_bytes(result).decode())
  346  
  347  
  348  if __name__ == "__main__":
  349      main()
