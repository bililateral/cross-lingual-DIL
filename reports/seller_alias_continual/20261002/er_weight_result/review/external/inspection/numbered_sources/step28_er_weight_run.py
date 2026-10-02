    1  """ER weight continuations with exact historical starts and one gated valid collection."""
    2  from __future__ import annotations
    3  
    4  import argparse
    5  import gc
    6  import os
    7  from pathlib import Path
    8  import platform
    9  import shutil
   10  import time
   11  from typing import Any
   12  
   13  import numpy as np
   14  
   15  import step28_er_weight as method
   16  import step28_er_weight_evaluate as evaluation
   17  
   18  parent, prior = method.parent, method.prior
   19  base, data, core = method.base, method.data, method.core
   20  save_array, rng_state, restore_rng = prior.save_array, prior.rng_state, prior.restore_rng
   21  COMPLETE = "COMPLETE_3456_ER_WEIGHT_UPDATES_VALID_BLIND"
   22  
   23  
   24  def old_point(reference: dict, name: str) -> dict:
   25      root = reference["job"] / "run"
   26      return data.read_json(data.verify(root / reference["manifest"]["points"][name]["path"],
   27                                        reference["manifest"]["points"][name]))
   28  
   29  
   30  def old_training(reference: dict, order: str, stage: int) -> dict:
   31      rec = reference["manifest"]["training"][f"{order}_er_stage{stage}"]
   32      return data.read_json(data.verify(reference["job"] / "run" / rec["path"], rec))
   33  
   34  
   35  def train_stage(model: Any, optimizer: Any, current: list, memory: Any, c: dict,
   36                  order: str, stage: int, arm: str, root: Path, reference_log: dict,
   37                  budget: Any) -> dict:
   38      import torch
   39  
   40      started = time.monotonic()
   41      old_policy = parent.contract()
   42      sequence, stream = parent.schedule(current, old_policy, order, stage)
   43      if [g.uid for g in sequence] != reference_log["current_ids"]:
   44          raise ValueError("Current schedule differs from original ER")
   45      memory.begin_stage(stage)
   46      updates, history_ids, observations = [], [], {}
   47      if next(model.parameters()).is_cuda:
   48          torch.cuda.synchronize()
   49          torch.cuda.reset_peak_memory_stats()
   50      for index, group in enumerate(sequence):
   51          history, target = memory.draw()
   52          if target is not None or history.uid != reference_log["history_ids"][index]:
   53              raise ValueError("ER history is unpaired or contains logits")
   54          history_ids.append(history.uid)
   55          record = method.update(
   56              model, optimizer, group, history, c, method.ARMS[arm], stage, index + 1,
   57              data.seed_for(stream, index, "dropout"),
   58              data.seed_for(old_policy["memory_seed"], order, stage, index, "history_dropout"),
   59              observe=index + 1 in (1, 29, 30, 288), check=budget.check)
   60          updates.append([record[key] for key in method.STEP_COLUMNS])
   61          if record["modules"]:
   62              observations[str(index + 1)] = record["modules"]
   63          if (index + 1) % 24 == 0:
   64              print(data.json_bytes({"event": "updates", "order": order, "arm": arm,
   65                                     "stage": stage, "stage_updates": index + 1,
   66                                     "logical_updates": parent.adam_step(optimizer),
   67                                     **budget.state()}).decode(), flush=True)
   68      summary = memory.summary()
   69      if (summary["draw_count"] != 288 or summary["members"] != reference_log["memory_after_training"]["members"]
   70              or history_ids != reference_log["history_ids"]):
   71          raise ValueError("Historical supply differs after training")
   72      name = method.point_name(order, arm, stage)
   73      result = {"name": name, "order": order, "arm": arm, "stage": stage,
   74                "history_weight": method.ARMS[arm], "updates": 288,
   75                "adam_step": parent.adam_step(optimizer), "actual_domain": order[stage - 1],
   76                "current_ids": [g.uid for g in sequence], "history_ids": history_ids,
   77                "current_dropout_stream": stream, "memory_after_training": summary,
   78                "observations": observations, "update_columns": list(method.STEP_COLUMNS),
   79                "update_file": save_array(root / "updates" / (name + ".npy"),
   80                                          np.asarray(updates, dtype=np.float64), root)}
   81      if next(model.parameters()).is_cuda:
   82          torch.cuda.synchronize()
   83          result["cuda_allocator"] = {"peak_allocated_bytes": torch.cuda.max_memory_allocated(),
   84                                       "peak_reserved_bytes": torch.cuda.max_memory_reserved()}
   85      result["training_seconds"] = time.monotonic() - started
   86      data.write_json(root / "updates" / (name + ".json"), result)
   87      return result
   88  
   89  
   90  def checkpoint(root: Path, name: str, model: Any, optimizer: Any, c: dict,
   91                 order: str, stage: int, current_cal: list, valid: list,
   92                 first_map: dict, budget: Any, weight: float) -> dict:
   93      """Real full state restore and complete score replay at every endpoint."""
   94      started = time.monotonic()
   95      arm = next((arm for arm, value in method.ARMS.items() if value == weight), None)
   96      if (arm is None or name != method.point_name(order, arm, stage) or first_map is None
   97              or len(current_cal) != 12 or len(valid) != 60 or parent.adam_step(optimizer) != stage * 288):
   98          raise ValueError("Checkpoint is not a complete stage with current-only calibration")
   99      state = rng_state()
  100      metadata = {"name": name, "order": order, "stage": stage, "seed": "s0",
  101                  "completed_updates": stage * 288, "policy_sha256": method.POLICY_SHA256,
  102                  "history_weight": weight, "parent_policy_sha256": parent.POLICY_SHA256,
  103                  "rng": state, "config": c}
  104      scores = {"calibration": parent.ranking.score(model, current_cal, c, budget.check),
  105                "development": parent.ranking.score(model, valid, c, budget.check)}
  106      full_path = root / "work" / (name + ".pt")
  107      budget.check(base.persistence.checkpoint_reserve(model, optimizer))
  108      full = core.save_state(full_path, model, optimizer, metadata)
  109      if core.restore_state(full_path, model, optimizer, full["state_sha256"]) != metadata:
  110          raise ValueError("Full model/Adam metadata differs")
  111      restore_rng(state)
  112      for role, groups in (("calibration", current_cal), ("development", valid)):
  113          if not np.array_equal(scores[role], parent.ranking.score(model, groups, c, budget.check)):
  114              raise ValueError("Actual full restore changed complete scores")
  115      model_digest = core.state_digest(model.state_dict())
  116      path = root / "models" / (name + ".pt")
  117      budget.check(base.persistence.checkpoint_reserve(model, None))
  118      inference = core.save_state(path, model, None, metadata)
  119      core.restore_state(path, model, None, inference["state_sha256"])
  120      if core.state_digest(model.state_dict()) != model_digest:
  121          raise ValueError("Inference state restore differs")
  122      score_records = {role: save_array(root / "scores" / (name + "_" + role + ".npy"), values, root)
  123                       for role, values in scores.items()}
  124      truth = np.asarray([group.labels for group in current_cal], dtype=np.uint8)
  125      mapping = parent.calibration.fit(scores["calibration"], truth, role="calibration", check=budget.check)
  126      mapping.update(name=name, actual_domain=order[stage - 1],
  127                     calibration_group_ids=[g.uid for g in current_cal], model_state_sha256=model_digest,
  128                     score_source=score_records["calibration"])
  129      map_path = root / "maps" / (name + ".json")
  130      data.write_json(map_path, mapping)  # Preserve actual solver failure before refusing continuation.
  131      if mapping["status"] != "PASS_CALIBRATION_FIT":
  132          raise RuntimeError("Stage calibration failed; no alternative fit or retry")
  133      restored_map = data.read_json(map_path)
  134      if restored_map != mapping:
  135          raise ValueError("Serialized mapping differs")
  136      first = first_map
  137      for role, use_map in (("stage-cal", mapping), ("first-cal", first)):
  138          values = parent.calibration.transform(scores["development"], use_map)
  139          parent.calibration.preserve_order(scores["development"], values)
  140          score_records[role] = save_array(root / "scores" / (name + "_" + role + ".npy"), values, root)
  141      # RNG is part of the shared branch state; checkpoint/evaluation cannot advance it.
  142      restore_rng(state)
  143      auxiliary = {"checkpoint_metadata": metadata,
  144                   "first_map": dict(zip(("a", "b"), parent.calibration.parameters(first))),
  145                   "stage_map": dict(zip(("a", "b"), parent.calibration.parameters(mapping))),
  146                   "history_weight": weight, "er_weight_policy_sha256": method.POLICY_SHA256}
  147      auxiliary_bytes = len(data.json_bytes(auxiliary))
  148      if auxiliary_bytes > 1048576:
  149          raise ValueError("Common phase/RNG/map metadata alone exceeds history allowance")
  150      result = {"name": name, "order": order, "stage": stage, "actual_domain": order[stage - 1],
  151                "completed_updates": stage * 288, "full_model_adam_and_rng_restore_verified": True,
  152                "history_weight": weight, "policy_sha256": method.POLICY_SHA256,
  153                "model_state_sha256": model_digest,
  154                "model": {**inference, "path": path.relative_to(root).as_posix()},
  155                "full_checkpoint": {**full, "path": full_path.relative_to(root).as_posix()},
  156                "full_checkpoint_retained": False,
  157                "scores": score_records, "mapping": data.record(map_path, root),
  158                "learner_auxiliary": auxiliary, "learner_auxiliary_serialized_bytes": auxiliary_bytes,
  159                "first_map_parameters": dict(zip(("a", "b"), parent.calibration.parameters(first)))}
  160      # Verified intermediate file only; all inference endpoints remain active research evidence.
  161      base.persistence.remove_work_file(full_path, root)
  162      result["intermediate_deleted_bytes"] = full["bytes"]
  163      result["checkpoint_calibration_score_seconds"] = time.monotonic() - started
  164      data.write_json(root / "points" / (name + ".json"), result)
  165      return result
  166  
  167  
  168  def train(job: Path, p: dict, budget: Any) -> tuple[dict, dict, list, dict]:
  169      import torch
  170  
  171      reference = method.baseline(p)
  172      old_root = reference["job"] / "run"
  173      root = job / "run"
  174      root.mkdir()
  175      for name in ("models", "work", "scores", "maps", "points", "memory", "updates"):
  176          (root / name).mkdir()
  177      c = method.config(p)
  178      groups, metadata, checked = base.public.public_inputs(c)
  179      _, partition = base.partition(groups, metadata, c)
  180      if partition != reference["partition"] or checked != reference["manifest"]["public_inputs"]:
  181          raise ValueError("Inputs/partition differ from the paired original experiment")
  182      data.write_json(root / "partition.json", partition)
  183      archive = core.model_files(base.model_config(c, "split_rank"))
  184      if any(archive[k] != c["models"]["split_rank"][k]
  185             for k in ("file_count", "total_size_bytes", "content_sha256")):
  186          raise ValueError("Pretrained archive differs")
  187      # Verify all required original payload identities before consuming new supervision.
  188      for order in method.ORDERS:
  189          shared = old_point(reference, order + "_shared")
  190          data.verify(old_root / shared["full_checkpoint"]["path"], shared["full_checkpoint"])
  191          rec = reference["manifest"]["memories"][order + "_er_after1"]["file"]
  192          data.verify(old_root / rec["path"], rec)
  193      groups["train"] = prior.parse_once(job, groups["train"], c, "train")
  194      selected, after = base.partition(groups, metadata, c)
  195      if after != partition:
  196          raise ValueError("Partition changed after training-label alignment")
  197      supply = method.ContinuationSupply(selected, partition)
  198      manifest = {"status": "RUNNING", "source_files": method.sources(), "policy_sha256": method.POLICY_SHA256,
  199                  "baseline_records": p["baseline"]["records"], "public_inputs": checked,
  200                  "pretrained_archive": archive, "partition": data.record(root / "partition.json", root),
  201                  "points": {}, "training": {}, "memories": {}, "restored_starts": {},
  202                  "physical_updates": 0, "gradient_group_presentations": 0}
  203      data.write_json(root / "startup.json", manifest)
  204      for order in method.ORDERS:
  205          shared = old_point(reference, order + "_shared")
  206          rec = reference["manifest"]["memories"][order + "_er_after1"]["file"]
  207          memory_payload = data.verify(old_root / rec["path"], rec).read_bytes()
  208          first_scores = prior.array(old_root, shared["scores"]["development"], (60, 378), np.float32)
  209          for arm, weight in method.ARMS.items():
  210              path = order + "_" + arm
  211              model, optimizer = prior.restore_branch(old_root, shared, c)
  212              state = rng_state()
  213              if not np.array_equal(first_scores, parent.ranking.score(model, groups["development"], c, budget.check)):
  214                  raise ValueError("Restored first-state full blind scores differ from original runtime")
  215              restore_rng(state)
  216              memory = parent.Memory.from_bytes(memory_payload)
  217              if (memory.with_logits or memory.order != order or memory.reservoir.seen != 48
  218                      or memory.draw_stage != 0 or memory.draw_count != 0
  219                      or memory.first_map != shared["first_map_parameters"]):
  220                  raise ValueError("Original ER cache/first-map restore differs")
  221              memory.auxiliary = {**shared["learner_auxiliary"], "history_weight": weight,
  222                                  "er_weight_policy_sha256": method.POLICY_SHA256}
  223              memory.to_bytes()
  224              supply.resume(path, order, shared)
  225              manifest["restored_starts"][path] = {
  226                  "full_checkpoint": shared["full_checkpoint"], "memory_source": rec,
  227                  "first_scores_replayed_exactly": True, "model_state_sha256": shared["model_state_sha256"],
  228                  "adam_step": parent.adam_step(optimizer), "first_map": memory.first_map,
  229                  "memory_summary": memory.summary()}
  230              for stage in (2, 3):
  231                  current, current_cal = supply.current(path, order, stage)
  232                  name = method.point_name(order, arm, stage)
  233                  train_stage(model, optimizer, current, memory, c, order, stage, arm,
  234                              root, old_training(reference, order, stage), budget)
  235                  point = checkpoint(root, name, model, optimizer, c, order, stage,
  236                                     current_cal, groups["development"], memory.first_map, budget, weight)
  237                  memory.auxiliary = point["learner_auxiliary"]
  238                  data.write_json(root / "memory" / (name + "_budget.json"), memory.summary())
  239                  if stage == 2:
  240                      memory.retain(current, 2, None)
  241                      expected = reference["manifest"]["memories"][f"{order}_er_stage2"]
  242                      if memory.summary()["members"] != expected["members"]:
  243                          raise ValueError("Stage-end reservoir members differ from original ER")
  244                      manifest["memories"][name] = prior.save_memory(root, name, memory)
  245                  manifest["points"][name] = data.record(root / "points" / (name + ".json"), root)
  246                  manifest["training"][name] = data.record(root / "updates" / (name + ".json"), root)
  247                  manifest["physical_updates"] += 288
  248                  manifest["gradient_group_presentations"] += 576
  249                  data.write_json(root / "progress.json", manifest)
  250                  del current, current_cal
  251              del model, optimizer, memory
  252              gc.collect()
  253              torch.cuda.empty_cache()
  254      if method.sources() != manifest["source_files"]:
  255          raise ValueError("Scientific sources changed")
  256      manifest.update(status=COMPLETE, budget=budget.state())
  257      data.write_json(root / "manifest.json", manifest)
  258      return manifest, partition, groups["development"], reference
  259  
  260  
  261  def blind_gate(root: Path, manifest: dict, reference: dict) -> dict:
  262      """Reject incomplete, unpaired, or incorrectly weighted updates before valid labels."""
  263      if (manifest["status"] != COMPLETE or manifest["source_files"] != method.sources()
  264              or manifest["policy_sha256"] != method.POLICY_SHA256
  265              or manifest["physical_updates"] != 3456 or manifest["gradient_group_presentations"] != 6912
  266              or set(manifest["points"]) != set(method.expected_points())
  267              or set(manifest["training"]) != set(method.expected_points())
  268              or set(manifest["memories"]) != {method.point_name(o, a, 2) for o in method.ORDERS for a in method.ARMS}
  269              or set(manifest["restored_starts"]) != {o + "_" + a for o in method.ORDERS for a in method.ARMS}):
  270          raise ValueError("Incomplete ER weight experiment")
  271      partition = data.read_json(data.verify(root / manifest["partition"]["path"], manifest["partition"]))
  272      if partition != reference["partition"]:
  273          raise ValueError("Blind partition changed")
  274      result = {}
  275      for order in method.ORDERS:
  276          for arm, weight in method.ARMS.items():
  277              start = manifest["restored_starts"][order + "_" + arm]
  278              shared = old_point(reference, order + "_shared")
  279              initial_memory = reference["manifest"]["memories"][order + "_er_after1"]
  280              if (start["full_checkpoint"] != shared["full_checkpoint"] or start["adam_step"] != 288
  281                      or not start["first_scores_replayed_exactly"]
  282                      or start["model_state_sha256"] != shared["model_state_sha256"]
  283                      or start["first_map"] != shared["first_map_parameters"]
  284                      or start["memory_source"] != initial_memory["file"]
  285                      or start["memory_summary"]["members"] != initial_memory["members"]
  286                      or start["memory_summary"]["with_logits"]
  287                      or start["memory_summary"]["seen"] != 48):
  288                  raise ValueError("Shared start evidence differs")
  289              retained = manifest["memories"][method.point_name(order, arm, 2)]
  290              data.verify(root / retained["file"]["path"], retained["file"])
  291              if (retained["members"] != reference["manifest"]["memories"][f"{order}_er_stage2"]["members"]
  292                      or retained["seen"] != 96 or retained["with_logits"]
  293                      or retained["serialized_bytes"] != retained["file"]["bytes"]
  294                      or retained["serialized_bytes"] > 1048576):
  295                  raise ValueError("Saved next-stage cache is incomplete or unpaired")
  296              for stage in (2, 3):
  297                  name = method.point_name(order, arm, stage)
  298                  point_rec, log_rec = manifest["points"][name], manifest["training"][name]
  299                  point = data.read_json(data.verify(root / point_rec["path"], point_rec))
  300                  log = data.read_json(data.verify(root / log_rec["path"], log_rec))
  301                  if (point["name"] != name or point["order"] != order or point["stage"] != stage
  302                          or point["actual_domain"] != order[stage - 1] or point["completed_updates"] != stage * 288
  303                          or point["history_weight"] != weight or not point["full_model_adam_and_rng_restore_verified"]
  304                          or point["policy_sha256"] != method.POLICY_SHA256):
  305                      raise ValueError("Native endpoint identity or actual restoration differs")
  306                  data.verify(root / point["model"]["path"], point["model"])
  307                  original_log = old_training(reference, order, stage)
  308                  for field in ("current_ids", "history_ids", "current_dropout_stream", "adam_step", "updates"):
  309                      if log[field] != original_log[field]:
  310                          raise ValueError("Training schedule is unpaired: " + field)
  311                  if (log["update_columns"] != list(method.STEP_COLUMNS)
  312                          or log["history_weight"] != weight
  313                          or log["memory_after_training"]["members"] != original_log["memory_after_training"]["members"]
  314                          or log["memory_after_training"]["serialized_bytes"] > 1048576):
  315                      raise ValueError("Replay configuration or memory differs")
  316                  values = prior.array(root, log["update_file"], (288, len(method.STEP_COLUMNS)), np.float64)
  317                  columns = {key: values[:, i] for i, key in enumerate(method.STEP_COLUMNS)}
  318                  for role in ("current", "history"):
  319                      if not np.allclose(columns[role + "_total"], columns[role + "_bce"]
  320                                         + columns[role + "_rank"] + .5 * columns[role + "_hard"],
  321                                         rtol=3e-6, atol=3e-6):
  322                          raise ValueError("Base objective decomposition differs")
  323                  if (not np.array_equal(columns["history_weight"], np.full(288, weight))
  324                          or not np.array_equal(columns["weighted_history_total"], weight * columns["history_total"])
  325                          or not np.array_equal(columns["total"], columns["current_total"] + columns["weighted_history_total"])
  326                          or not np.array_equal(columns["encoder_lr"], [parent.stage_lr(i) for i in range(1, 289)])
  327                          or not np.all(columns["head_lr"] == .001)):
  328                      raise ValueError("Historical weight, total, or learning-rate log is wrong")
  329                  if set(log["observations"]) != {"1", "29", "30", "288"}:
  330                      raise ValueError("Missing real gradient/update observations")
  331                  for step, modules in log["observations"].items():
  332                      for module in ("encoder", "head"):
  333                          if (not modules[module]["finite_nonzero_combined_gradient"]
  334                                  or modules[module]["parameters_changed"] != (module == "head" or step != "288")):
  335                              raise ValueError("Missing real encoder/head update")
  336                  scores = point["scores"]
  337                  raw = prior.array(root, scores["development"], (60, 378), np.float32)
  338                  prior.array(root, scores["calibration"], (12, 378), np.float32)
  339                  mapping = data.read_json(data.verify(root / point["mapping"]["path"], point["mapping"]))
  340                  expected_cal = [row["group_uid"] for row in partition["calibration"] if row["domain"] == order[stage - 1]]
  341                  if (mapping["calibration_group_ids"] != expected_cal
  342                          or mapping["actual_domain"] != order[stage - 1]
  343                          or mapping["group_count"] != 12 or mapping["pair_count"] != 4536
  344                          or mapping["positive_count"] != 240
  345                          or mapping["score_source"] != scores["calibration"]
  346                          or mapping["model_state_sha256"] != point["model_state_sha256"]
  347                          or mapping["status"] != "PASS_CALIBRATION_FIT"
  348                          or point["first_map_parameters"] != start["first_map"]):
  349                      raise ValueError("Current-only calibration or first map differs")
  350                  roles = {"raw": raw}
  351                  for role, use_map in (("stage-cal", mapping), ("first-cal", start["first_map"])):
  352                      transformed = prior.array(root, scores[role], (60, 378), np.float64)
  353                      if not np.array_equal(transformed, parent.calibration.transform(raw, use_map)):
  354                          raise ValueError("Saved calibration transform differs")
  355                      parent.calibration.preserve_order(raw, transformed)
  356                      roles[role] = transformed
  357                  result[name] = roles
  358      return result
  359  
  360  
  361  def collect(root: Path, scores: dict, labelled: list, partition: dict, source_files: list) -> dict:
  362      root.mkdir()
  363      if set(scores) != set(method.expected_points()):
  364          raise ValueError("Collect all new points together")
  365      if [g.uid for g in labelled] != [row["group_uid"] for row in partition["development"]]:
  366          raise ValueError("Valid truth and blind group identities differ")
  367      truth = np.asarray([g.labels for g in labelled], dtype=np.uint8)
  368      collected = {"status": "COLLECTING", "points": {}, "source_files": source_files,
  369                   "policy_sha256": method.POLICY_SHA256, "metric_columns": list(method.metrics.COLUMNS),
  370                   "group_ids": [g.uid for g in labelled],
  371                   "domains": [row["domain"] for row in partition["development"]]}
  372      for name in method.expected_points():
  373          if set(scores[name]) != set(method.ROLES):
  374              raise ValueError("Missing output role")
  375          entries = {}
  376          for role in method.ROLES:
  377              matrix, counts = method.metrics.group_metrics(truth, scores[name][role])
  378              rec = save_array(root / f"{name}_{role}.npy", matrix, root)
  379              counts_path = root / f"{name}_{role}_counts.json"
  380              data.write_json(counts_path, counts)
  381              entries[role] = {"matrix": rec, "counts": data.record(counts_path, root)}
  382          collected["points"][name] = entries
  383      collected["status"] = "ALL_36_ER_WEIGHT_MATRICES_SAVED_BEFORE_COMPARISONS"
  384      data.write_json(root / "collected.json", collected)
  385      return collected
  386  
  387  
  388  def execute(job: Path, audit_path: Path, authorization_path: Path) -> dict:
  389      import torch
  390  
  391      p, source_files = method.contract(), method.sources()
  392      auth = data.read_json(authorization_path)
  393      audit = data.read_json(audit_path)
  394      if (platform.system() != "Linux" or auth.get("status") != "AUTHORIZED_ER_WEIGHT_AFTER_REVIEW"
  395              or auth.get("policy_sha256") != method.POLICY_SHA256 or auth.get("source_files") != source_files
  396              or auth.get("job") != job.relative_to(data.ROOT).as_posix()
  397              or auth.get("runtime") != p["runtime"] or auth.get("supervision") != p["supervision"]
  398              or auth.get("review_and_primary_passed") is not True):
  399          raise ValueError("Matching reviewed formal authorization is required")
  400      if (audit.get("status") != "PASS_ER_WEIGHT_HANDMADE_CPU" or audit.get("source_files") != source_files
  401              or audit.get("contracts", {}).get("failed") != 0 or audit.get("contracts", {}).get("skipped") != 0
  402              or audit.get("formal_inputs") is not False or audit.get("formal_labels") is not False
  403              or set(audit.get("native", {})) != set(method.ARMS)
  404              or audit.get("native_history_gradient_scaling_verified") is not True):
  405          raise ValueError("Necessary current handmade/native CPU verification is missing")
  406      for record in audit["native"].values():
  407          data.verify(audit_path.parent / record["path"], record)
  408      if (not job.is_relative_to((data.ROOT / "reports").resolve())
  409              or any((job / n).exists() for n in ("run", "evaluation", "access.json"))):
  410          raise ValueError("Use a new reports job; formal entry cannot resume")
  411      if not torch.cuda.is_available() or torch.cuda.device_count() != 1 or not torch.cuda.is_bf16_supported():
  412          raise RuntimeError("One eligible visible GPU required")
  413      environment = {"python": platform.python_version(), "torch": torch.__version__,
  414                     "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name()}
  415      if environment != p["baseline"]["environment"]:
  416          raise RuntimeError("Paired runtime differs; do not silently retrain controls")
  417      available = next(int(row.split()[1]) * 1024 for row in Path("/proc/meminfo").read_text().splitlines()
  418                       if row.startswith("MemAvailable:"))
  419      if (torch.cuda.mem_get_info()[0] < p["runtime"]["minimum_free_gpu_bytes"]
  420              or available < p["runtime"]["minimum_free_host_bytes"]
  421              or shutil.disk_usage(data.ROOT).free < p["runtime"]["maximum_output_bytes"]):
  422          raise RuntimeError("Wait for resources without affecting other users")
  423      job.mkdir(parents=True, exist_ok=True)
  424      torch.set_num_threads(1)
  425      os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})
  426      torch.use_deterministic_algorithms(True)
  427      torch.backends.cuda.matmul.allow_tf32 = False
  428      torch.backends.cudnn.allow_tf32 = False
  429      torch.backends.cudnn.benchmark = False
  430      budget = base.persistence.Budget(job, {"runtime": p["runtime"]})
  431      data.write_json(job / "access.json", {"train": 0, "valid": 0, "heldout": 0, "owners": 0})
  432      data.write_json(job / "execution.json", {**environment, "source_files": source_files,
  433                                              "authorization": data.record(authorization_path, data.ROOT),
  434                                              "audit": data.record(audit_path, data.ROOT),
  435                                              "cpu_affinity": sorted(os.sched_getaffinity(0))})
  436      try:
  437          manifest, partition, valid, reference = train(job, p, budget)
  438          scores = blind_gate(job / "run", manifest, reference)
  439          data.write_json(job / "before_valid.json", {"status": "PASS_ER_WEIGHT_COMPLETE_BLIND_GATE",
  440                                                      "manifest": data.record(job / "run/manifest.json", job),
  441                                                      "label_parses": data.read_json(job / "access.json")})
  442          labelled = prior.parse_once(job, valid, method.config(p), "development")
  443          collect(job / "evaluation", scores, labelled, partition, source_files)
  444          del labelled, valid, scores
  445          result = evaluation.finalize(job / "evaluation", reference["job"] / "evaluation")
  446          budget.check(1)
  447          if method.sources() != source_files:
  448              raise ValueError("Sources changed before completion")
  449          completion = {"status": result["status"], "physical_updates": 3456,
  450                        "label_parses": data.read_json(job / "access.json"), "budget": budget.state(),
  451                        "selection": result["selection"],
  452                        "evaluation": data.record(job / "evaluation/evaluation.json", job)}
  453          data.write_json(job / "completion.json", completion)
  454          return completion
  455      except Exception as error:
  456          data.write_json(job / "failure.json", {"status": "FAILED_NO_AUTOMATIC_RETRY", "error": str(error),
  457                                                 "label_parses": data.read_json(job / "access.json"),
  458                                                 "recovery": "After all 36 matrices exist, finalize uses saved metrics only"})
  459          raise
  460  
  461  
  462  def main() -> None:
  463      parser = argparse.ArgumentParser(description=__doc__)
  464      parser.add_argument("action", choices=("execute", "finalize"))
  465      parser.add_argument("--out", type=Path, required=True)
  466      parser.add_argument("--audit", type=Path)
  467      parser.add_argument("--authorization", type=Path)
  468      args = parser.parse_args()
  469      if platform.system() != "Linux":
  470          parser.error("Research scripts run only on Linux py310")
  471      if args.action == "execute":
  472          if args.audit is None or args.authorization is None:
  473              parser.error("execute needs --audit and --authorization")
  474          result = execute(args.out.resolve(), args.audit.resolve(), args.authorization.resolve())
  475      else:
  476          p = method.contract()
  477          result = evaluation.finalize(args.out.resolve(), data.ROOT / p["baseline"]["linux_job"] / "evaluation")
  478      print(data.json_bytes({"status": result["status"], "out": str(args.out)}).decode())
  479  
  480  
  481  if __name__ == "__main__":
  482      main()
