    1  """Three sequential orders, shared first states, restricted replay and gated valid.
    2  
    3  Only an explicit matching formal authorization and native evidence enable execute.
    4  Preparing this file does not authorize Linux, supervision or training.
    5  """
    6  from __future__ import annotations
    7  
    8  import argparse
    9  import gc
   10  import hashlib
   11  import os
   12  from pathlib import Path
   13  import platform
   14  import shutil
   15  import time
   16  from typing import Any, Callable
   17  
   18  import numpy as np
   19  
   20  import step28_bge_continual as method
   21  import step28_bge_continual_evaluate as evaluation
   22  
   23  base, data, core = method.base, method.data, method.core
   24  COMPLETE = "COMPLETE_6048_UPDATES_21_ENDPOINTS_VALID_BLIND"
   25  
   26  
   27  def sources() -> list[dict]:
   28      paths = [
   29          "scripts/step28_bge_continual.py", "scripts/step28_bge_continual_run.py",
   30          "scripts/step28_bge_continual_evaluate.py", "scripts/step28_bge_continual_check.py",
   31          "tests/test_step28_bge_continual_contracts.py", "schema/step28_bge_continual_policy.json",
   32          "docs/SELLER_ALIAS_BGE_CONTINUAL.zh.md", "scripts/run_step28_bge_continual_linux_20260930.sh",
   33          "scripts/step28_alias_ranking.py", "scripts/step28_alias_calibration.py",
   34          "scripts/step28_chinese_base.py", "schema/step28_alias_ranking_policy.json",
   35          "schema/step28_chinese_base_policy.json", "scripts/step28_continual_population.py",
   36          "scripts/step28_continual_population_data.py", "scripts/step28_continual_population_evaluate.py",
   37          "scripts/step28_continual_population_run.py", "scripts/step28_continual_expression_run.py",
   38      ]
   39      return [data.record(data.ROOT / path, data.ROOT) for path in sorted(paths)]
   40  
   41  
   42  def array(root: Path, record: dict, shape: tuple, dtype: Any) -> np.ndarray:
   43      result = np.load(data.verify(root / record["path"], record), allow_pickle=False)
   44      if result.shape != shape or result.dtype != dtype or not np.isfinite(result).all():
   45          raise ValueError("Saved score/matrix schema differs")
   46      return result
   47  
   48  
   49  def save_array(path: Path, values: np.ndarray, root: Path) -> dict:
   50      if path.exists():
   51          raise FileExistsError(path)
   52      np.save(path, values, allow_pickle=False)
   53      if not np.array_equal(np.load(path, allow_pickle=False), values):
   54          raise ValueError("Array roundtrip changed values")
   55      return data.record(path, root)
   56  
   57  
   58  def rng_state() -> dict:
   59      import torch
   60      return {"cpu": torch.get_rng_state().tolist(),
   61              "cuda": [state.tolist() for state in torch.cuda.get_rng_state_all()]
   62              if torch.cuda.is_available() else []}
   63  
   64  
   65  def restore_rng(state: dict) -> None:
   66      import torch
   67      torch.set_rng_state(torch.tensor(state["cpu"], dtype=torch.uint8))
   68      if state["cuda"]:
   69          if len(state["cuda"]) != torch.cuda.device_count():
   70              raise ValueError("CUDA RNG count differs")
   71          torch.cuda.set_rng_state_all([torch.tensor(row, dtype=torch.uint8) for row in state["cuda"]])
   72      if rng_state() != state:
   73          raise ValueError("RNG restore changed state")
   74  
   75  
   76  def parse_once(job: Path, groups: list, c: dict, split: str,
   77                 loader: Callable = base.public.attach_labels) -> list:
   78      if split not in ("train", "development"):
   79          raise ValueError("Forbidden supervision split")
   80      path = job / "access.json"
   81      access = data.read_json(path)
   82      key = "train" if split == "train" else "valid"
   83      if access[key] != 0:
   84          raise ValueError("This stage's supervision parse has already been attempted")
   85      access[key] = 1
   86      data.write_json(path, access)  # Attempt recorded before opening any label CSV.
   87      return loader(groups, c, split)
   88  
   89  
   90  def save_memory(root: Path, name: str, memory: method.Memory) -> dict:
   91      path = root / "memory" / (name + ".json")
   92      if path.exists():
   93          raise FileExistsError(path)
   94      payload = memory.to_bytes()
   95      path.write_bytes(payload)
   96      restored = method.Memory.from_bytes(path.read_bytes())
   97      if restored.to_bytes() != payload:
   98          raise ValueError("Memory restoration changed payload")
   99      return {"file": data.record(path, root), **memory.summary(),
  100              "custody": "Linux-only training payload; contains formal texts and labels, exclude from review/small-result sync"}
  101  
  102  
  103  def train_stage(model: Any, optimizer: Any, current: list, memory: method.Memory | None,
  104                  c: dict, p: dict, order: str, stage: int, arm: str,
  105                  root: Path, budget: Any) -> dict:
  106      import torch
  107  
  108      started = time.monotonic()
  109      sequence, stream = method.schedule(current, p, order, stage)
  110      if memory is not None:
  111          memory.begin_stage(stage)
  112      history_ids, updates, observations = [], [], {}
  113      torch.cuda.synchronize() if next(model.parameters()).is_cuda else None
  114      if next(model.parameters()).is_cuda:
  115          torch.cuda.reset_peak_memory_stats()
  116      for index, group in enumerate(sequence):
  117          history, reference = memory.draw() if memory is not None else (None, None)
  118          if history is not None:
  119              history_ids.append(history.uid)
  120          record = method.update(
  121              model, optimizer, group, history, reference, c, arm, stage, index + 1,
  122              data.seed_for(stream, index, "dropout"),
  123              data.seed_for(p["memory_seed"], order, stage, index, "history_dropout"),
  124              observe=index + 1 in (1, 29, 30, 288), check=budget.check)
  125          updates.append([record[key] for key in method.STEP_COLUMNS])
  126          if record["modules"]:
  127              observations[str(index + 1)] = record["modules"]
  128          if (index + 1) % 24 == 0:
  129              print(data.json_bytes({"event": "updates", "order": order, "method": arm,
  130                                     "stage": stage, "stage_updates": index + 1,
  131                                     "logical_updates": method.adam_step(optimizer),
  132                                     **budget.state()}).decode(), flush=True)
  133      if memory is not None and memory.draw_count != 288:
  134          raise ValueError("Incomplete historical presentation count")
  135      name = order + "_shared" if stage == 1 else f"{order}_{arm}_stage{stage}"
  136      result = {"order": order, "method": "shared" if stage == 1 else arm, "stage": stage,
  137                "actual_domain": order[stage - 1], "seed": "s0", "updates": 288,
  138                "encoder_positive_lr_updates": 287, "adam_step": method.adam_step(optimizer),
  139                "current_ids": [g.uid for g in sequence], "history_ids": history_ids,
  140                "current_schedule_sha256": hashlib.sha256(data.json_bytes([g.uid for g in sequence])).hexdigest(),
  141                "current_dropout_stream": stream, "update_columns": list(method.STEP_COLUMNS),
  142                "scalar_timing_scope": "Host construction/enqueue time only; backward is joint with historical supervision, not separately isolated GPU compute",
  143                "observations": observations, "memory_after_training": memory.summary() if memory else None,
  144                "update_file": save_array(root / "updates" / (name + ".npy"),
  145                                          np.asarray(updates, dtype=np.float64), root)}
  146      if next(model.parameters()).is_cuda:
  147          torch.cuda.synchronize()
  148          result["cuda_allocator"] = {"peak_allocated_bytes": torch.cuda.max_memory_allocated(),
  149                                       "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
  150                                       "scope": "This stage training interval; not whole-card or other-process use"}
  151      result["training_seconds"] = time.monotonic() - started
  152      data.write_json(root / "updates" / (name + ".json"), result)
  153      return result
  154  
  155  
  156  def checkpoint(root: Path, name: str, model: Any, optimizer: Any, c: dict,
  157                 order: str, stage: int, current_cal: list, valid: list,
  158                 first_map: dict | None, budget: Any) -> dict:
  159      """Real full state restore and complete score replay at every endpoint."""
  160      started = time.monotonic()
  161      if len(current_cal) != 12 or len(valid) != 60 or method.adam_step(optimizer) != stage * 288:
  162          raise ValueError("Checkpoint is not a complete stage with current-only calibration")
  163      state = rng_state()
  164      metadata = {"name": name, "order": order, "stage": stage, "seed": "s0",
  165                  "completed_updates": stage * 288, "policy_sha256": method.POLICY_SHA256,
  166                  "rng": state, "config": c}
  167      scores = {"calibration": method.ranking.score(model, current_cal, c, budget.check),
  168                "development": method.ranking.score(model, valid, c, budget.check)}
  169      full_path = root / ("branches" if stage == 1 else "work") / (name + ".pt")
  170      budget.check(base.persistence.checkpoint_reserve(model, optimizer))
  171      full = core.save_state(full_path, model, optimizer, metadata)
  172      if core.restore_state(full_path, model, optimizer, full["state_sha256"]) != metadata:
  173          raise ValueError("Full model/Adam metadata differs")
  174      restore_rng(state)
  175      for role, groups in (("calibration", current_cal), ("development", valid)):
  176          if not np.array_equal(scores[role], method.ranking.score(model, groups, c, budget.check)):
  177              raise ValueError("Actual full restore changed complete scores")
  178      model_digest = core.state_digest(model.state_dict())
  179      path = root / "models" / (name + ".pt")
  180      budget.check(base.persistence.checkpoint_reserve(model, None))
  181      inference = core.save_state(path, model, None, metadata)
  182      core.restore_state(path, model, None, inference["state_sha256"])
  183      if core.state_digest(model.state_dict()) != model_digest:
  184          raise ValueError("Inference state restore differs")
  185      score_records = {role: save_array(root / "scores" / (name + "_" + role + ".npy"), values, root)
  186                       for role, values in scores.items()}
  187      truth = np.asarray([group.labels for group in current_cal], dtype=np.uint8)
  188      mapping = method.calibration.fit(scores["calibration"], truth, role="calibration", check=budget.check)
  189      mapping.update(name=name, actual_domain=order[stage - 1],
  190                     calibration_group_ids=[g.uid for g in current_cal], model_state_sha256=model_digest,
  191                     score_source=score_records["calibration"])
  192      map_path = root / "maps" / (name + ".json")
  193      data.write_json(map_path, mapping)  # Preserve actual solver failure before refusing continuation.
  194      if mapping["status"] != "PASS_CALIBRATION_FIT":
  195          raise RuntimeError("Stage calibration failed; no alternative fit or retry")
  196      restored_map = data.read_json(map_path)
  197      if restored_map != mapping:
  198          raise ValueError("Serialized mapping differs")
  199      first = first_map if first_map is not None else mapping
  200      for role, use_map in (("stage-cal", mapping), ("first-cal", first)):
  201          values = method.calibration.transform(scores["development"], use_map)
  202          method.calibration.preserve_order(scores["development"], values)
  203          score_records[role] = save_array(root / "scores" / (name + "_" + role + ".npy"), values, root)
  204      # RNG is part of the shared branch state; checkpoint/evaluation cannot advance it.
  205      restore_rng(state)
  206      auxiliary = {"checkpoint_metadata": metadata,
  207                   "first_map": dict(zip(("a", "b"), method.calibration.parameters(first))),
  208                   "stage_map": dict(zip(("a", "b"), method.calibration.parameters(mapping)))}
  209      auxiliary_bytes = len(data.json_bytes(auxiliary))
  210      if auxiliary_bytes > 1048576:
  211          raise ValueError("Common phase/RNG/map metadata alone exceeds history allowance")
  212      result = {"name": name, "order": order, "stage": stage, "actual_domain": order[stage - 1],
  213                "completed_updates": stage * 288, "full_model_adam_and_rng_restore_verified": True,
  214                "model_state_sha256": model_digest,
  215                "model": {**inference, "path": path.relative_to(root).as_posix()},
  216                "full_checkpoint": {**full, "path": full_path.relative_to(root).as_posix()},
  217                "full_checkpoint_retained": stage == 1,
  218                "scores": score_records, "mapping": data.record(map_path, root),
  219                "learner_auxiliary": auxiliary, "learner_auxiliary_serialized_bytes": auxiliary_bytes,
  220                "first_map_parameters": dict(zip(("a", "b"), method.calibration.parameters(first)))}
  221      if stage != 1:
  222          # Verified intermediate file only; all inference endpoints remain active research evidence.
  223          base.persistence.remove_work_file(full_path, root)
  224          result["intermediate_deleted_bytes"] = full["bytes"]
  225      result["checkpoint_calibration_score_seconds"] = time.monotonic() - started
  226      data.write_json(root / "points" / (name + ".json"), result)
  227      return result
  228  
  229  
  230  def restore_branch(root: Path, point: dict, c: dict) -> tuple[Any, Any]:
  231      model = base.load_model(c, "split_rank")
  232      optimizer = core.make_optimizer(model, c)
  233      record = point["full_checkpoint"]
  234      path = data.verify(root / record["path"], record)
  235      metadata = core.restore_state(path, model, optimizer, record["state_sha256"])
  236      if metadata["completed_updates"] != 288 or method.adam_step(optimizer) != 288:
  237          raise ValueError("Branch did not restore the shared full first stage")
  238      restore_rng(metadata["rng"])
  239      if core.state_digest(model.state_dict()) != point["model_state_sha256"]:
  240          raise ValueError("Shared branch model mismatch")
  241      return model, optimizer
  242  
  243  
  244  def blind_gate(root: Path, manifest: dict, p: dict) -> dict:
  245      """Inspect actual complete weights and score files before any valid parsing."""
  246      if (manifest["status"] != COMPLETE or manifest["source_files"] != sources()
  247              or manifest["physical_updates"] != 6048 or manifest["gradient_group_presentations"] != 9504
  248              or set(manifest["points"]) != set(evaluation.expected_points())):
  249          raise ValueError("Incomplete or changed pilot before valid")
  250      initial = array(root, manifest["initial"]["scores"], (60, 378), np.float32)
  251      partition = data.read_json(data.verify(root / manifest["partition"]["path"], manifest["partition"]))
  252      fitting = {r["group_uid"]: r["domain"] for r in partition["fit"]}
  253      points = {}
  254      schedules, memories, first_maps = {}, {}, {}
  255      for name in evaluation.expected_points():
  256          ref = manifest["points"][name]
  257          point = data.read_json(data.verify(root / ref["path"], ref))
  258          stage = point["stage"]
  259          expected_stage = 1 if name.endswith("_shared") else int(name[-1])
  260          if (point["name"] != name or stage != expected_stage or point["order"] != name[:3]
  261                  or point["actual_domain"] != name[stage - 1]
  262                  or point["completed_updates"] != stage * 288
  263                  or point["full_model_adam_and_rng_restore_verified"] is not True):
  264              raise ValueError("Stage identity/reload record differs")
  265          data.verify(root / point["model"]["path"], point["model"])
  266          if stage == 1:
  267              data.verify(root / point["full_checkpoint"]["path"], point["full_checkpoint"])
  268          raw = array(root, point["scores"]["development"], (60, 378), np.float32)
  269          array(root, point["scores"]["calibration"], (12, 378), np.float32)
  270          mapping = data.read_json(data.verify(root / point["mapping"]["path"], point["mapping"]))
  271          if (mapping["status"] != "PASS_CALIBRATION_FIT" or mapping["group_count"] != 12
  272                  or mapping["pair_count"] != 4536 or mapping["positive_count"] != 240
  273                  or mapping["actual_domain"] != point["actual_domain"]
  274                  or mapping["model_state_sha256"] != point["model_state_sha256"]
  275                  or mapping["score_source"] != point["scores"]["calibration"]
  276                  or mapping["calibration_group_ids"] != [r["group_uid"] for r in partition["calibration"]
  277                                                         if r["domain"] == point["actual_domain"]]):
  278              raise ValueError("Current-domain calibration identity differs")
  279          if stage == 1:
  280              first_maps[point["order"]] = dict(zip(("a", "b"), method.calibration.parameters(mapping)))
  281          if point["first_map_parameters"] != first_maps[point["order"]]:
  282              raise ValueError("Fixed first-stage diagnostic map changed")
  283          restored = {"raw": raw}
  284          for role, use_map in (("stage-cal", mapping), ("first-cal", point["first_map_parameters"])):
  285              values = array(root, point["scores"][role], (60, 378), np.float64)
  286              if not np.array_equal(values, method.calibration.transform(raw, use_map)):
  287                  raise ValueError("Saved map/score pairing differs")
  288              method.calibration.preserve_order(raw, values)
  289              restored[role] = values
  290          log_record = manifest["training"][name]
  291          log = data.read_json(data.verify(root / log_record["path"], log_record))
  292          values = array(root, log["update_file"], (288, len(method.STEP_COLUMNS)), np.float64)
  293          expected_arm = "shared" if stage == 1 else name.split("_")[1]
  294          if (log["updates"] != 288 or log["adam_step"] != stage * 288
  295                  or log["order"] != point["order"] or log["stage"] != stage
  296                  or log["method"] != expected_arm or log["actual_domain"] != point["actual_domain"]
  297                  or log["update_columns"] != list(method.STEP_COLUMNS)
  298                  or len(log["current_ids"]) != 288 or len(set(log["current_ids"])) != 48
  299                  or set(log["current_ids"]) != {uid for uid, d in fitting.items() if d == point["actual_domain"]}
  300                  or any(log["current_ids"].count(uid) != 6 for uid in set(log["current_ids"]))):
  301              raise ValueError("Current presentation budget differs")
  302          columns = dict(zip(method.STEP_COLUMNS, values.T))
  303          if (not np.array_equal(columns["encoder_lr"], [method.stage_lr(s) for s in range(1, 289)])
  304                  or not np.all(columns["head_lr"] == .001)
  305                  or not np.allclose(columns["current_total"], columns["current_bce"] + columns["current_rank"]
  306                                     + .5 * columns["current_hard"], atol=2e-6, rtol=1e-6)
  307                  or not np.allclose(columns["history_total"], columns["history_bce"] + columns["history_rank"]
  308                                     + .5 * columns["history_hard"], atol=2e-6, rtol=1e-6)
  309                  or not np.allclose(columns["total"], columns["current_total"] + columns["history_total"]
  310                                     + .5 * columns["logit_mse"], atol=1e-12, rtol=0)):
  311              raise ValueError("Learning-rate or loss decomposition differs")
  312          key = (point["order"], stage)
  313          current_tuple = tuple(log["current_ids"])
  314          if key in schedules and schedules[key] != current_tuple:
  315              raise ValueError("Current sequences are not paired")
  316          schedules[key] = current_tuple
  317          if log["method"] in ("er", "logit"):
  318              if len(log["history_ids"]) != 288 or set(log["history_ids"]) & set(log["current_ids"]):
  319                  raise ValueError("Historical budget or separation differs")
  320              memory_key = (point["order"] + "_" + expected_arm + "_after1" if stage == 2
  321                            else point["order"] + "_" + expected_arm + "_stage2")
  322              memory_record = manifest["memories"][memory_key]
  323              payload = data.verify(root / memory_record["file"]["path"], memory_record["file"]).read_bytes()
  324              memory = method.Memory.from_bytes(payload)
  325              if (memory.first_map != first_maps[point["order"]]
  326                      or memory.with_logits != (expected_arm == "logit")
  327                      or memory.order != point["order"] or memory.seed != p["memory_seed"]
  328                      or any(fitting[g.uid] not in point["order"][:stage - 1] for g in memory.reservoir.groups)
  329                      or any(fitting[uid] != point["order"][origin - 1]
  330                             for uid, origin in memory.reference_origins.items())):
  331                  raise ValueError("Historical origin or reference contract differs")
  332              memory.begin_stage(stage)
  333              replayed = [memory.draw()[0].uid for _ in range(288)]
  334              if replayed != log["history_ids"] or memory.summary() != log["memory_after_training"]:
  335                  raise ValueError("Actual saved memory/draw state differs from training")
  336              if expected_arm == "er" and np.any(columns["logit_mse"] != 0):
  337                  raise ValueError("ER included an unauthorized logit penalty")
  338              value = (log["memory_after_training"]["members"], log["history_ids"])
  339              if key in memories and memories[key] != value:
  340                  raise ValueError("ER and LOGIT history members/draws differ")
  341              memories[key] = value
  342          elif log["history_ids"] or np.any(values[:, 4:9] != 0):
  343              raise ValueError("SEQ/shared first stage used history")
  344          points[name] = restored
  345      return {"initial": initial, "points": points}
  346  
  347  
  348  def collect(root: Path, scores: dict, labelled: list, partition: dict, source_files: list) -> dict:
  349      """Write every matrix/count before bootstrap, comparison or acceptance."""
  350      root.mkdir()
  351      truth = np.asarray([group.labels for group in labelled], dtype=np.uint8)
  352      if [g.uid for g in labelled] != [r["group_uid"] for r in partition["development"]]:
  353          raise ValueError("Valid truth/score group order differs")
  354      result = {"status": "COLLECTING", "points": {}, "source_files": source_files,
  355                "group_ids": [g.uid for g in labelled],
  356                "domains": [r["domain"] for r in partition["development"]],
  357                "metric_columns": list(method.metrics.COLUMNS)}
  358      for name in ["initial", *evaluation.expected_points()]:
  359          entries = {"raw": scores["initial"]} if name == "initial" else scores["points"][name]
  360          stored = {}
  361          for role, values in entries.items():
  362              matrix, counts = method.metrics.group_metrics(truth, values)
  363              matrix_record = save_array(root / f"{name}_{role}.npy", matrix, root)
  364              count_path = root / f"{name}_{role}_counts.json"
  365              data.write_json(count_path, counts)
  366              stored[role] = {"matrix": matrix_record, "counts": data.record(count_path, root)}
  367          if name == "initial":
  368              result["initial"] = stored["raw"]
  369          else:
  370              result["points"][name] = stored
  371      result["status"] = "ALL_PILOT_MATRICES_SAVED_BEFORE_COMPARISONS"
  372      data.write_json(root / "collected.json", result)
  373      return result
  374  
  375  
  376  def train(job: Path, p: dict, budget: Any) -> tuple[dict, dict, list]:
  377      import torch
  378  
  379      root = job / "run"
  380      root.mkdir()
  381      for name in ("models", "branches", "work", "scores", "maps", "points", "memory", "updates"):
  382          (root / name).mkdir()
  383      c = method.config(p)
  384      groups, metadata, checked = base.public.public_inputs(c)
  385      selected, partition = base.partition(groups, metadata, c)
  386      archive = core.model_files(base.model_config(c, "split_rank"))
  387      if any(archive[k] != c["models"]["split_rank"][k]
  388             for k in ("file_count", "total_size_bytes", "content_sha256")):
  389          raise ValueError("Actual local pretrained BGE archive differs")
  390      data.write_json(root / "partition.json", partition)
  391      manifest = {"status": "RUNNING", "source_files": sources(), "points": {}, "training": {},
  392                  "partition": data.record(root / "partition.json", root), "public_inputs": checked,
  393                  "pretrained_archive": archive, "physical_updates": 0, "gradient_group_presentations": 0,
  394                  "memories": {}, "initial": {}, "policy_sha256": method.POLICY_SHA256}
  395      data.write_json(root / "startup.json", manifest)
  396      model = base.load_model(c, "split_rank")
  397      initial_digest = core.state_digest(model.state_dict())
  398      # Public preflight checks lengths only; no truncation or feedback to training.
  399      maximum = 0
  400      for split in ("train", "development"):
  401          for group in groups[split]:
  402              budget.check()
  403              lengths = [len(row) for row in model.encoder.tokenizer(
  404                  base.record_texts(group, "separate_moments"), padding=False, truncation=False)["input_ids"]]
  405              maximum = max(maximum, max(lengths))
  406              if maximum > c["input"]["token_budget"]:
  407                  raise ValueError("Formal input exceeds frozen token budget")
  408      initial_scores = method.ranking.score(model, groups["development"], c, budget.check)
  409      manifest["initial"] = {"model_state_sha256": initial_digest, "maximum_tokens": maximum,
  410                             "scores": save_array(root / "scores" / "initial_raw.npy", initial_scores, root)}
  411      del model
  412      gc.collect()
  413      torch.cuda.empty_cache()
  414      groups["train"] = parse_once(job, groups["train"], c, "train")
  415      selected, labelled_partition = base.partition(groups, metadata, c)
  416      if labelled_partition != partition:
  417          raise ValueError("Partition changed after train alignment")
  418      supply = method.Supply(selected, partition)
  419      for order in method.ORDERS:
  420          model = base.load_model(c, "split_rank")
  421          if core.state_digest(model.state_dict()) != initial_digest:
  422              raise ValueError("Order initialization differs")
  423          optimizer = core.make_optimizer(model, c)
  424          shared_path = order + "_shared"
  425          current, current_cal = supply.current(shared_path, order, 1)
  426          train_stage(model, optimizer, current, None, c, p, order, 1, "seq", root, budget)
  427          shared = checkpoint(root, shared_path, model, optimizer, c, order, 1,
  428                              current_cal, groups["development"], None, budget)
  429          first_map = shared["first_map_parameters"]
  430          initial_memory = {}
  431          for arm in ("er", "logit"):
  432              memory = method.Memory(order, p["memory_seed"], arm == "logit", first_map)
  433              memory.auxiliary = shared["learner_auxiliary"]
  434              tick = time.monotonic()
  435              memory.retain(current, 1, lambda rows: method.ranking.score(model, rows, c, budget.check))
  436              retention_seconds = time.monotonic() - tick
  437              initial_memory[arm] = memory.to_bytes()
  438              manifest["memories"][order + "_" + arm + "_after1"] = save_memory(root, order + "_" + arm + "_after1", memory)
  439              manifest["memories"][order + "_" + arm + "_after1"]["retention_and_reference_seconds"] = retention_seconds
  440          if method.Memory.from_bytes(initial_memory["er"]).summary()["members"] != method.Memory.from_bytes(initial_memory["logit"]).summary()["members"]:
  441              raise ValueError("Shared first memory selection differs")
  442          del memory, current, current_cal, optimizer, model
  443          gc.collect()
  444          torch.cuda.empty_cache()
  445          for arm in method.UPDATED:
  446              path = order + "_" + arm
  447              supply.branch_after_first(path, shared_path)
  448              model, optimizer = restore_branch(root, shared, c)
  449              memory = method.Memory.from_bytes(initial_memory[arm]) if arm != "seq" else None
  450              for stage in (2, 3):
  451                  current, current_cal = supply.current(path, order, stage)
  452                  train_stage(model, optimizer, current, memory, c, p, order, stage, arm, root, budget)
  453                  name = f"{order}_{arm}_stage{stage}"
  454                  point = checkpoint(root, name, model, optimizer, c, order, stage, current_cal,
  455                                     groups["development"], first_map, budget)
  456                  if memory is not None:
  457                      memory.auxiliary = point["learner_auxiliary"]
  458                      data.write_json(root / "memory" / (name + "_budget.json"), memory.summary())
  459                  if memory is not None and stage == 2:
  460                      tick = time.monotonic()
  461                      memory.retain(current, 2, lambda rows: method.ranking.score(model, rows, c, budget.check))
  462                      retention_seconds = time.monotonic() - tick
  463                      manifest["memories"][name] = save_memory(root, name, memory)
  464                      manifest["memories"][name]["retention_and_reference_seconds"] = retention_seconds
  465                  del current, current_cal
  466              del model, optimizer, memory
  467              gc.collect()
  468              torch.cuda.empty_cache()
  469          for name in [n for n in evaluation.expected_points() if n.startswith(order)]:
  470              manifest["points"][name] = data.record(root / "points" / (name + ".json"), root)
  471              manifest["training"][name] = data.record(root / "updates" / (name + ".json"), root)
  472          manifest["physical_updates"] = len(manifest["points"]) * 288
  473          manifest["gradient_group_presentations"] = manifest["physical_updates"] + len(manifest["points"]) // 7 * 1152
  474          data.write_json(root / "progress.json", manifest)
  475      if sources() != manifest["source_files"]:
  476          raise ValueError("Scientific sources changed during formal run")
  477      manifest.update(status=COMPLETE, budget=budget.state())
  478      data.write_json(root / "manifest.json", manifest)
  479      return manifest, partition, groups["development"]
  480  
  481  
  482  def execute(job: Path, audit_path: Path, authorization_path: Path) -> dict:
  483      import torch
  484  
  485      if platform.system() != "Linux":
  486          raise RuntimeError("Use existing Linux py310 only")
  487      p = method.contract()
  488      authorization = data.read_json(authorization_path)
  489      if (authorization.get("status") != "AUTHORIZED_BGE_CONTINUAL_TRAIN_VALID"
  490              or authorization.get("policy_sha256") != method.POLICY_SHA256
  491              or authorization.get("source_files") != sources()
  492              or authorization.get("job") != job.relative_to(data.ROOT).as_posix()
  493              or authorization.get("physical_updates") != 6048
  494              or authorization.get("train_parse_attempts") != 1 or authorization.get("valid_parse_attempts") != 1
  495              or authorization.get("test_access") is not False or authorization.get("owners_access") is not False
  496              or authorization.get("runtime") != p["runtime"]):
  497          raise ValueError("A separate matching formal-stage authorization is required")
  498      audit = data.read_json(audit_path)
  499      if (audit.get("status") != "PASS_BGE_CONTINUAL_HANDMADE_CPU"
  500              or audit.get("source_files") != sources() or audit.get("contracts", {}).get("failed") != 0
  501              or audit.get("contracts", {}).get("skipped") != 0
  502              or audit.get("formal_inputs") is not False or audit.get("formal_labels") is not False
  503              or set(audit.get("native", {})) != set(method.UPDATED)):
  504          raise ValueError("Current handmade CPU/native verification is required")
  505      for record in audit["native"].values():
  506          data.verify(audit_path.parent / record["path"], record)
  507      if not job.is_relative_to((data.ROOT / "reports").resolve()) or any((job / n).exists() for n in ("run", "evaluation", "access.json")):
  508          raise ValueError("Use a new reports job; no automatic resumption")
  509      if not torch.cuda.is_available() or torch.cuda.device_count() != 1 or not torch.cuda.is_bf16_supported():
  510          raise RuntimeError("One eligible visible GPU required")
  511      available = next(int(row.split()[1]) * 1024 for row in Path("/proc/meminfo").read_text().splitlines()
  512                       if row.startswith("MemAvailable:"))
  513      if (torch.cuda.mem_get_info()[0] < p["runtime"]["minimum_free_gpu_bytes"]
  514              or available < p["runtime"]["minimum_free_host_bytes"]
  515              or shutil.disk_usage(data.ROOT).free < p["runtime"]["maximum_output_bytes"]):
  516          raise RuntimeError("Insufficient shared resources; wait without affecting other users")
  517      job.mkdir(parents=True, exist_ok=True)
  518      torch.set_num_threads(1)
  519      os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})
  520      torch.use_deterministic_algorithms(True)
  521      torch.backends.cuda.matmul.allow_tf32 = False
  522      torch.backends.cudnn.allow_tf32 = False
  523      torch.backends.cudnn.benchmark = False
  524      budget = base.persistence.Budget(job, {"runtime": p["runtime"]})
  525      data.write_json(job / "access.json", {"train": 0, "valid": 0, "heldout": 0, "owners": 0})
  526      data.write_json(job / "execution.json", {"authorization": data.record(authorization_path, data.ROOT),
  527                                              "audit": data.record(audit_path, data.ROOT),
  528                                              "source_files": sources(), "python": platform.python_version(),
  529                                              "torch": torch.__version__, "cuda": torch.version.cuda,
  530                                              "gpu": torch.cuda.get_device_name(), "cpu_affinity": sorted(os.sched_getaffinity(0))})
  531      try:
  532          manifest, partition, valid_groups = train(job, p, budget)
  533          scores = blind_gate(job / "run", manifest, p)
  534          data.write_json(job / "before_valid.json", {"status": "PASS_COMPLETE_BLIND_GATE",
  535                                                      "manifest": data.record(job / "run/manifest.json", job),
  536                                                      "points": 21, "supervision": data.read_json(job / "access.json")})
  537          labelled = parse_once(job, valid_groups, method.config(p), "development")
  538          collect(job / "evaluation", scores, labelled, partition, manifest["source_files"])
  539          del labelled, scores, valid_groups
  540          result = evaluation.finalize(job / "evaluation")
  541          budget.check(1)
  542          if sources() != manifest["source_files"]:
  543              raise ValueError("Scientific sources changed during evaluation")
  544          completion = {"status": result["status"], "physical_updates": 6048,
  545                        "label_parses": data.read_json(job / "access.json"), "budget": budget.state(),
  546                        "evaluation": data.record(job / "evaluation/evaluation.json", job)}
  547          data.write_json(job / "completion.json", completion)
  548          return completion
  549      except Exception as error:
  550          data.write_json(job / "failure.json", {"status": "FAILED_NO_AUTOMATIC_RETRY", "error": str(error),
  551                                                 "label_parses": data.read_json(job / "access.json"),
  552                                                 "recovery": "After complete collection, finalize reads saved evidence only"})
  553          raise
  554  
  555  
  556  def main() -> None:
  557      parser = argparse.ArgumentParser(description=__doc__)
  558      parser.add_argument("action", choices=("execute", "finalize"))
  559      parser.add_argument("--out", type=Path, required=True)
  560      parser.add_argument("--audit", type=Path)
  561      parser.add_argument("--authorization", type=Path)
  562      args = parser.parse_args()
  563      if platform.system() != "Linux":
  564          parser.error("Research scripts run only on Linux py310")
  565      if args.action == "execute":
  566          if args.audit is None or args.authorization is None:
  567              parser.error("execute requires --audit and --authorization")
  568          result = execute(args.out.resolve(), args.audit.resolve(), args.authorization.resolve())
  569      else:
  570          result = evaluation.finalize(args.out.resolve())
  571      print(data.json_bytes({"status": result["status"], "out": str(args.out)}).decode())
  572  
  573  
  574  if __name__ == "__main__":
  575      main()
