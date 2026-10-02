    1  """Linux single-GPU execution of the approved new-population comparison.
    2  
    3  Run only after the user resumes this concrete Linux stage. The production path
    4  first checks the changed real LaBSE semantics, then executes the frozen schedule.
    5  It never opens valid/test labels and does not resume a failed run automatically.
    6  """
    7  from __future__ import annotations
    8  
    9  import argparse
   10  import gc
   11  import hashlib
   12  import platform
   13  import random
   14  import shutil
   15  import sys
   16  import time
   17  import unittest
   18  from datetime import datetime, timezone
   19  from pathlib import Path
   20  from typing import Any
   21  
   22  import numpy as np
   23  
   24  import step28_continual_population as core
   25  import step28_continual_population_data as data
   26  
   27  ARMS = ("sequential", "er", "cumulative")
   28  COMPLETE = "COMPLETE_BLIND_SCORES_RELOADED_VALID_AND_TEST_LABELS_UNREAD"
   29  
   30  
   31  def sources() -> list[dict]:
   32      paths = sorted((data.ROOT / "scripts").glob("step28_continual_population*.py"))
   33      paths += sorted((data.ROOT / "tests").glob("test_step28_continual_population*.py"))
   34      paths.append(data.POLICY_PATH)
   35      return [data.record(path, data.ROOT) for path in paths]
   36  
   37  
   38  class Budget:
   39      def __init__(self, root: Path, config: dict):
   40          self.root, self.config = root, config["runtime"]
   41          self.started = time.monotonic()
   42          self.last_disk_check = -float("inf")
   43          self.peak_bytes = 0
   44  
   45      def check(self, reserve: int = 0) -> None:
   46          elapsed = time.monotonic() - self.started
   47          if elapsed >= self.config["maximum_gpu_stage_seconds"]:
   48              raise RuntimeError("Approved GPU-stage time budget reached")
   49          if reserve or elapsed - self.last_disk_check > 10:
   50              used = sum(p.stat().st_size for p in self.root.rglob("*") if p.is_file())
   51              self.peak_bytes = max(self.peak_bytes, used)
   52              self.last_disk_check = elapsed
   53              if used + reserve > self.config["maximum_output_bytes"]:
   54                  raise RuntimeError("Approved output budget would be exceeded")
   55              if shutil.disk_usage(self.root).free < reserve:
   56                  raise RuntimeError("Insufficient free space; do not affect other users")
   57  
   58      def state(self) -> dict:
   59          self.check(reserve=1)
   60          return {"elapsed_seconds": time.monotonic() - self.started, "peak_observed_bytes": self.peak_bytes}
   61  
   62  
   63  def checkpoint_reserve(model: Any, optimizer: Any | None) -> int:
   64      # Conservative reserve before creating a file, including future torch metadata.
   65      parameters = sum(p.numel() * p.element_size() for p in model.state_dict().values())
   66      return parameters * (3 if optimizer is not None else 1) + 64 * 1024**2
   67  
   68  
   69  def remove_work_file(path: Path, root: Path) -> None:
   70      # Only new, verified intermediate states of THIS run. Never a general cleanup.
   71      resolved, work = path.resolve(), (root / "work").resolve()
   72      if resolved.parent != work or resolved.suffix != ".pt" or resolved.is_symlink():
   73          raise ValueError("Not this run's disposable intermediate checkpoint")
   74      resolved.unlink()
   75  
   76  
   77  def point(model: Any, optimizer: Any | None, name: str, groups: dict,
   78            config: dict, out: Path, budget: Budget, memory: data.Memory | None = None) -> dict:
   79      started = time.monotonic()
   80      before = {split: core.score(model, rows, config, budget.check) for split, rows in groups.items()}
   81      inference_seconds = time.monotonic() - started
   82      checkpoint = out / "work" / f"{name}.pt"
   83      metadata = {"point": name, "config": config}
   84      budget.check(checkpoint_reserve(model, optimizer))
   85      state = core.save_state(checkpoint, model, optimizer, metadata)
   86      restored = core.restore_state(checkpoint, model, optimizer, state["state_sha256"])
   87      if restored != metadata:
   88          raise ValueError("Checkpoint metadata differs")
   89      result = {"checkpoint": {**state, "path": checkpoint.relative_to(out).as_posix()},
   90                "model_state_sha256": core.state_digest(model.state_dict()),
   91                "full_model_and_adam_reloaded": True, "scores": {}}
   92      for split, rows in groups.items():
   93          inference_started = time.monotonic()
   94          after = core.score(model, rows, config, budget.check)
   95          inference_seconds += time.monotonic() - inference_started
   96          if not np.array_equal(before[split], after):
   97              raise ValueError("Complete blind scores differ after actual checkpoint reload")
   98          path = out / "scores" / f"{name}_{split}.npy"
   99          np.save(path, before[split], allow_pickle=False)
  100          result["scores"][split] = data.record(path, out)
  101      if memory is not None:
  102          payload = memory.to_bytes()
  103          path = out / "memory" / f"{name}.json"
  104          path.write_bytes(payload)
  105          restored_memory = data.Memory.from_bytes(path.read_bytes())
  106          if restored_memory.to_bytes() != payload:
  107              raise ValueError("Actual persisted memory differs")
  108          result["memory"] = {**data.record(path, out), "seen": memory.seen,
  109                              "retained_groups": [g.uid for g in memory.groups],
  110                              "disk_roundtrip_exact": True}
  111      result["timing"] = {"inference_seconds_including_replay": inference_seconds,
  112                          "total_point_seconds": time.monotonic() - started}
  113      print(data.json_bytes({"event": "point_reloaded", "point": name,
  114                            **result["timing"], **budget.state()}).decode(), flush=True)
  115      return result
  116  
  117  
  118  def retain_inference(model: Any, name: str, point_record: dict, config: dict,
  119                       out: Path, budget: Budget) -> dict:
  120      target = out / "models" / f"{name}.pt"
  121      budget.check(checkpoint_reserve(model, None))
  122      saved = core.save_state(target, model, None, {"point": name, "config": config})
  123      core.restore_state(target, model, None, saved["state_sha256"])
  124      if core.state_digest(model.state_dict()) != point_record["model_state_sha256"]:
  125          raise ValueError("Retained inference model differs from the replayed model")
  126      return {**saved, "path": target.relative_to(out).as_posix(),
  127              "actual_loaded_model_equals_replayed_state": True}
  128  
  129  
  130  def train_stage(model: Any, optimizer: Any, current: list[data.Group], memory: data.Memory | None,
  131                  config: dict, order: str, stage: int, budget: Budget, *, cumulative: bool = False) -> dict:
  132      started = time.monotonic()
  133      base_seed = config["initialization_seed"]
  134      stream = data.seed_for(base_seed, order, stage, "cumulative" if cumulative else "current")
  135      rows = data.schedule(current, config["epochs_per_stage"], stream)
  136      replay_rng = random.Random(data.seed_for(base_seed, order, stage, "replay_draw"))
  137      first_memory = memory.to_bytes() if memory is not None else None
  138      current_losses, historical_losses, norms, history_ids = [], [], [], []
  139      for index, group in enumerate(rows):
  140          replay = replay_rng.choice(memory.groups) if memory is not None and memory.groups else None
  141          log = core.update(model, optimizer, group, replay, config,
  142                            data.seed_for(stream, index, "dropout_current"),
  143                            data.seed_for(stream, index, "dropout_replay"), budget.check)
  144          current_losses.append(log["current_bce"])
  145          norms.append(log["gradient_norm_before_clip"])
  146          if replay is not None:
  147              historical_losses.append(log["replay_bce"])
  148              history_ids.append(replay.uid)
  149          if (index + 1) % 30 == 0:
  150              print(data.json_bytes({"event": "updates", "order": order, "stage": stage,
  151                    "arm": "cumulative" if cumulative else "er" if memory is not None else "sequential",
  152                    "completed": index + 1, "total": len(rows),
  153                    "stage_training_seconds": time.monotonic() - started, **budget.state()}).decode(), flush=True)
  154      if memory is not None and memory.to_bytes() != first_memory:
  155          raise ValueError("Historical memory changed during a stage")
  156      return {"updates": len(rows), "training_seconds": time.monotonic() - started,
  157              "current_group_ids": [g.uid for g in rows],
  158              "current_order_sha256": hashlib.sha256(data.json_bytes([g.uid for g in rows])).hexdigest(),
  159              "current_dropout_stream": stream, "replay_group_ids": history_ids,
  160              "current_pair_presentations": sum(len(g.labels) for g in rows),
  161              "replay_pair_presentations": len(history_ids) * 378,
  162              "current_bce_mean": float(np.mean(current_losses)),
  163              "replay_bce_mean": float(np.mean(historical_losses)) if historical_losses else None,
  164              "maximum_gradient_norm_before_clip": max(norms), "memory_frozen_during_stage": True}
  165  
  166  
  167  def runtime_check(model: Any, archive: data.Archive, config: dict, out: Path, budget: Budget) -> dict:
  168      """New actual encoder/head gradients, updates and complete state reload."""
  169      import torch
  170      started = time.monotonic()
  171      sys.path.insert(0, str(data.ROOT / "tests"))
  172      suite = unittest.defaultTestLoader.loadTestsFromName("test_step28_continual_population_contracts.TorchContracts")
  173      tests = unittest.TextTestRunner(stream=sys.stdout, verbosity=2).run(suite)
  174      if not tests.wasSuccessful() or tests.skipped or tests.testsRun != 3:
  175          raise RuntimeError("Necessary actual PyTorch semantic cases did not all pass")
  176      current, replay = archive.groups("train", "A")[:2]
  177      original = out / "work" / "runtime_original.pt"
  178      budget.check(checkpoint_reserve(model, None))
  179      original_record = core.save_state(original, model, None, {"role": "precheck_original"})
  180      optimizer = core.make_optimizer(model, config)
  181      initial_digest = {name: core.state_digest(module.state_dict())
  182                        for name, module in (("encoder", model.encoder), ("head", model.head))}
  183      branch = {}
  184      for label, group in (("current", current), ("replay", replay)):
  185          model.train()
  186          optimizer.zero_grad(set_to_none=True)
  187          loss = core.backward_group(model, group, config, data.seed_for(20260909, "runtime", label), budget.check)
  188          branch[label] = {"loss": loss}
  189          for name, module in (("encoder", model.encoder), ("head", model.head)):
  190              norms = [float(p.grad.detach().float().norm()) for p in module.parameters() if p.grad is not None]
  191              if not norms or not all(np.isfinite(norms)) or not max(norms) > 0:
  192                  raise ValueError(f"No finite nonzero {label} gradient in {name}")
  193              branch[label][name + "_gradient_norm"] = float(np.linalg.norm(norms))
  194      update_log = core.update(model, optimizer, current, replay, config, 701, 702, budget.check)
  195      for name, module in (("encoder", model.encoder), ("head", model.head)):
  196          if core.state_digest(module.state_dict()) == initial_digest[name]:
  197              raise ValueError(f"Runtime update did not change {name}")
  198      memory = data.Memory(data.seed_for(20260909, "runtime_memory"))
  199      memory.add_stage(archive.groups("train", "A"))
  200      # A complete group's 378 scores in both blind splits; full formal points follow.
  201      groups = {s: archive.groups(s)[:1] for s in ("development", "heldout")}
  202      result = point(model, optimizer, "runtime", groups, config, out, budget, memory)
  203      if not optimizer.state or not all(float(v["step"]) == 1 for v in optimizer.state.values()):
  204          raise ValueError("First actual Adam update/reload differs")
  205      core.restore_state(original, model, None, original_record["state_sha256"])
  206      for name, module in (("encoder", model.encoder), ("head", model.head)):
  207          if core.state_digest(module.state_dict()) != initial_digest[name]:
  208              raise ValueError("Runtime check altered formal initialization")
  209      del optimizer
  210      gc.collect()
  211      torch.cuda.empty_cache()
  212      remove_work_file(out / result["checkpoint"]["path"], out)
  213      remove_work_file(original, out)
  214      return {"status": "PASS_REAL_LABSE_NEW_INPUT_HEAD_MEMORY_ADAM_RELOAD",
  215              "branches": branch, "update": update_log, "point": result,
  216              "formal_initialization_restored": True, "formal_update_count": 0,
  217              "torch_contracts": {"passed": tests.testsRun, "skipped": 0, "failed": 0},
  218              "runtime_check_seconds": time.monotonic() - started}
  219  
  220  
  221  def run(out: Path, *, check_only: bool = False) -> dict:
  222      import torch
  223      config = data.policy()
  224      if platform.system() != "Linux" or not torch.cuda.is_available() or torch.cuda.device_count() != 1:
  225          raise RuntimeError("Requires the authorized Linux stage and exactly one visible GPU")
  226      if not torch.cuda.is_bf16_supported() or torch.cuda.mem_get_info()[0] < config["runtime"]["minimum_free_gpu_bytes"]:
  227          raise RuntimeError("Insufficient eligible idle GPU capacity; wait")
  228      available = next(int(line.split()[1]) * 1024 for line in Path("/proc/meminfo").read_text().splitlines()
  229                       if line.startswith("MemAvailable:"))
  230      if available < config["runtime"]["minimum_free_host_bytes"]:
  231          raise RuntimeError("Insufficient available host RAM; wait")
  232      if shutil.disk_usage(data.ROOT).free < config["runtime"]["maximum_output_bytes"]:
  233          raise RuntimeError("Insufficient project disk reserve; wait")
  234      if out.exists() or not out.resolve().is_relative_to((data.ROOT / "reports").resolve()):
  235          raise ValueError("Use a new run directory inside project reports")
  236      out.mkdir(parents=True)
  237      for name in ("work", "models", "scores", "memory"):
  238          (out / name).mkdir()
  239      budget = Budget(out, config)
  240      startup_sources = sources()
  241      started_utc = datetime.now(timezone.utc).isoformat()
  242      manifest: dict = {"status": "RUNNING", "started_utc": started_utc, "config": config,
  243                        "source_files": startup_sources, "orders": [],
  244                        "environment": {"python": platform.python_version(), "torch": torch.__version__,
  245                                        "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0)}}
  246      data.write_json(out / "startup.json", manifest)
  247      try:
  248          torch.set_num_threads(config["runtime"]["torch_cpu_threads"])
  249          torch.use_deterministic_algorithms(True)
  250          torch.backends.cuda.matmul.allow_tf32 = False
  251          torch.backends.cudnn.allow_tf32 = False
  252          torch.backends.cudnn.benchmark = False
  253          model_record = core.model_files(config)
  254          for key in ("file_count", "total_size_bytes", "content_sha256"):
  255              if model_record[key] != config["model"][key]:
  256                  raise ValueError("Original pretrained model bytes differ")
  257          manifest["pretrained_model"] = model_record
  258          archive = data.Archive(config, train_labels=True)
  259          manifest["verified_input_files"] = archive.checked
  260          manifest["label_reads"] = {"new_train_csv_offline_packaging": archive.train_label_parses,
  261                                     "development": 0, "heldout": 0, "owners": 0, "old_assets": 0}
  262          model = core.load_model(config)
  263          manifest["runtime_check"] = runtime_check(model, archive, config, out, budget)
  264          data.write_json(out / "runtime_check.json", {"source_files": startup_sources, **manifest["runtime_check"]})
  265          if check_only:
  266              manifest["status"] = "REAL_CHECK_ONLY_NO_FORMAL_TRAINING"
  267          else:
  268              eval_groups = {s: archive.groups(s) for s in ("development", "heldout")}
  269              manifest["evaluation_group_ids"] = {s: [g.uid for g in rows] for s, rows in eval_groups.items()}
  270              initial_model_digest = core.state_digest(model.state_dict())
  271              for order in config["orders"]:
  272                  if core.state_digest(model.state_dict()) != initial_model_digest:
  273                      raise ValueError("Order did not start from common original state")
  274                  log: dict = {"order": order, "points": {}, "training": {}, "trajectory": {}, "models": {}}
  275                  points = log["points"]
  276                  points["initial"] = point(model, None, order + "_initial", eval_groups, config, out, budget)
  277                  initial_path = out / points["initial"]["checkpoint"]["path"]
  278                  optimizer = core.make_optimizer(model, config)
  279                  first = archive.groups("train", order[0])
  280                  log["training"]["shared"] = train_stage(model, optimizer, first, None, config, order, 1, budget)
  281                  memory = data.Memory(data.seed_for(config["initialization_seed"], order, "memory"))
  282                  memory.add_stage(first)
  283                  memory_bytes = memory.to_bytes()
  284                  points["shared"] = point(model, optimizer, order + "_shared", eval_groups, config, out, budget, memory)
  285                  shared_path = out / points["shared"]["checkpoint"]["path"]
  286                  log["models"]["frozen"] = retain_inference(model, order + "_frozen", points["shared"], config, out, budget)
  287                  log["trajectory"]["frozen"] = ["initial", "shared", "shared", "shared"]
  288                  del optimizer
  289                  for arm in ARMS:
  290                      log["training"][arm] = []
  291                      log["trajectory"][arm] = ["initial", "shared"]
  292                      optimizer = core.make_optimizer(model, config)
  293                      core.restore_state(shared_path, model, optimizer, points["shared"]["checkpoint"]["state_sha256"])
  294                      memory = data.Memory.from_bytes(memory_bytes) if arm == "er" else None
  295                      for stage in (2, 3):
  296                          if arm == "cumulative":
  297                              core.restore_state(initial_path, model, None, points["initial"]["checkpoint"]["state_sha256"])
  298                              del optimizer
  299                              optimizer = core.make_optimizer(model, config)
  300                          current = archive.groups("train", order[:stage] if arm == "cumulative" else order[stage - 1])
  301                          stage_log = train_stage(model, optimizer, current, memory, config, order, stage, budget,
  302                                                  cumulative=arm == "cumulative")
  303                          log["training"][arm].append(stage_log)
  304                          if memory is not None:
  305                              memory.add_stage(current)
  306                          key = f"{arm}_stage{stage}"
  307                          points[key] = point(model, optimizer, order + "_" + key, eval_groups, config, out, budget, memory)
  308                          log["trajectory"][arm].append(key)
  309                          if stage == 3:
  310                              log["models"][arm] = retain_inference(model, order + "_" + arm, points[key], config, out, budget)
  311                          remove_work_file(out / points[key]["checkpoint"]["path"], out)
  312                      del optimizer
  313                      gc.collect()
  314                      torch.cuda.empty_cache()
  315                  for index in (0, 1):
  316                      seq, er = (log["training"][a][index] for a in ("sequential", "er"))
  317                      if (seq["current_order_sha256"], seq["current_dropout_stream"]) != (er["current_order_sha256"], er["current_dropout_stream"]):
  318                          raise ValueError("Unpaired current presentations/dropout")
  319                  core.restore_state(initial_path, model, None, points["initial"]["checkpoint"]["state_sha256"])
  320                  remove_work_file(shared_path, out)
  321                  remove_work_file(initial_path, out)
  322                  manifest["orders"].append(log)
  323                  data.write_json(out / f"order_{order}.json", log)
  324              manifest["physical_updates"] = sum(o["training"]["shared"]["updates"] +
  325                  sum(r["updates"] for arm in ARMS for r in o["training"][arm]) for o in manifest["orders"])
  326              if manifest["physical_updates"] != config["physical_updates"]:
  327                  raise ValueError("Physical optimizer update count differs")
  328              manifest["formal_training_seconds"] = sum(o["training"]["shared"]["training_seconds"] +
  329                  sum(r["training_seconds"] for arm in ARMS for r in o["training"][arm]) for o in manifest["orders"])
  330              manifest["formal_inference_seconds_including_replay"] = sum(
  331                  p["timing"]["inference_seconds_including_replay"] for o in manifest["orders"] for p in o["points"].values())
  332              manifest["status"] = COMPLETE
  333          if sources() != startup_sources:
  334              raise ValueError("Scientific sources changed during execution")
  335          manifest["budget"] = budget.state()
  336          data.write_json(out / "manifest.json", manifest)
  337          return {"status": manifest["status"], "output": str(out), "budget": manifest["budget"]}
  338      except Exception as error:
  339          data.write_json(out / "failure.json", {"status": "FAILED_DO_NOT_EVALUATE_OR_AUTO_RESTART",
  340                          "error_type": type(error).__name__, "error": str(error),
  341                          "startup_source_files": startup_sources, "failure_source_files": sources(),
  342                          "elapsed_seconds": time.monotonic() - budget.started,
  343                          "completed_orders": manifest["orders"]})
  344          raise
  345  
  346  
  347  def main() -> None:
  348      parser = argparse.ArgumentParser(description=__doc__)
  349      parser.add_argument("--out", type=Path, required=True, help="New reports directory; never overwrite")
  350      parser.add_argument("--check-only", action="store_true", help="Run changed real LaBSE checks only")
  351      args = parser.parse_args()
  352      print(data.json_bytes(run(args.out.resolve(), check_only=args.check_only)).decode())
  353  
  354  
  355  if __name__ == "__main__":
  356      main()
