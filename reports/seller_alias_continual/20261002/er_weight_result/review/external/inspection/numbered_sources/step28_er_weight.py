    1  """Confirmed ER strength intervention; parent scientific sources remain unchanged."""
    2  from __future__ import annotations
    3  
    4  import copy
    5  from pathlib import Path
    6  from typing import Any, Callable
    7  
    8  import numpy as np
    9  
   10  import step28_bge_continual as parent
   11  import step28_bge_continual_run as prior
   12  
   13  base, data, core, metrics = parent.base, parent.data, parent.core, parent.metrics
   14  POLICY = data.ROOT / "schema/step28_er_weight_policy.json"
   15  POLICY_SHA256 = "9eddf92646892d5890491e068ecbd4834463d0751de530a4701b7de8a98e4396"
   16  ARMS = {"half": .5, "quarter": .25}
   17  ORDERS, ROLES = parent.ORDERS, parent.ROLES
   18  STEP_COLUMNS = ("current_bce", "current_rank", "current_hard", "current_total",
   19                  "history_bce", "history_rank", "history_hard", "history_total",
   20                  "weighted_history_total", "total", "encoder_lr", "head_lr",
   21                  "gradient_norm", "history_weight")
   22  
   23  
   24  def contract() -> dict:
   25      if data.sha256(POLICY) != POLICY_SHA256:
   26          raise ValueError("ER weight policy changed")
   27      p = data.read_json(POLICY)
   28      old = parent.contract()
   29      if (p["arms"] != ARMS or p["orders"] != list(ORDERS) or p["stages"] != [2, 3]
   30              or p["physical_updates"] != 3456 or p["gradient_group_presentations"] != 6912
   31              or p["parent_policy_sha256"] != parent.POLICY_SHA256
   32              or old["memory"]["capacity_groups"] != 6
   33              or p["supervision"]["test_access"] or p["supervision"]["owners_access"]):
   34          raise ValueError("Confirmed intervention or boundaries differ")
   35      return p
   36  
   37  
   38  def config(p: dict) -> dict:
   39      result = parent.config(parent.contract())
   40      result["runtime"] = copy.deepcopy(p["runtime"])
   41      return result
   42  
   43  
   44  def sources() -> list[dict]:
   45      additional = ["docs/SELLER_ALIAS_ER_WEIGHT.zh.md", "schema/step28_er_weight_policy.json",
   46                    "scripts/step28_er_weight.py", "scripts/step28_er_weight_run.py",
   47                    "scripts/step28_er_weight_evaluate.py", "scripts/step28_er_weight_check.py",
   48                    "tests/test_step28_er_weight_contracts.py",
   49                    "scripts/run_step28_er_weight_linux_20261001.sh",
   50                    "scripts/run_step28_er_check_linux_20261001.sh",
   51                    "reports/documentation/20261001/er_weight/decision.json"]
   52      rows = prior.sources() + [data.record(data.ROOT / name, data.ROOT) for name in additional]
   53      return sorted(rows, key=lambda row: row["path"])
   54  
   55  
   56  def point_name(order: str, arm: str, stage: int) -> str:
   57      if order not in ORDERS or arm not in ARMS or stage not in (2, 3):
   58          raise ValueError("Unknown new ER endpoint")
   59      return f"{order}_{arm}_stage{stage}"
   60  
   61  
   62  def expected_points() -> list[str]:
   63      return [point_name(order, arm, stage) for order in ORDERS for arm in ARMS for stage in (2, 3)]
   64  
   65  
   66  def baseline(p: dict, job: Path | None = None) -> dict:
   67      """Verify pinned small records; native state and cache are checked when restored."""
   68      job = job if job is not None else data.ROOT / p["baseline"]["linux_job"]
   69      records = {name: data.read_json(data.verify(job / name, rec))
   70                 for name, rec in p["baseline"]["records"].items()}
   71      manifest = records["run/manifest.json"]
   72      collected = records["evaluation/collected.json"]
   73      if (manifest["status"] != prior.COMPLETE or manifest["source_files"] != prior.sources()
   74              or collected["source_files"] != prior.sources()
   75              or collected["status"] != "ALL_PILOT_MATRICES_SAVED_BEFORE_COMPARISONS"
   76              or records["evaluation/evaluation.json"]["status"] != "COMPLETE_BGE_CONTINUAL_DIAGNOSTIC"
   77              or collected["metric_columns"] != list(metrics.COLUMNS)):
   78          raise ValueError("Original completed evidence or frozen scientific sources differ")
   79      partition = records["run/partition.json"]
   80      if (collected["group_ids"] != [row["group_uid"] for row in partition["development"]]
   81              or collected["domains"] != [row["domain"] for row in partition["development"]]):
   82          raise ValueError("Original valid identities do not align")
   83      for key, value in p["baseline"]["environment"].items():
   84          if records["execution.json"][key] != value:
   85              raise ValueError("Original environment differs from pinned baseline")
   86      return {"job": job, "manifest": manifest, "partition": partition,
   87              "collected": collected, "execution": records["execution.json"]}
   88  
   89  
   90  class ContinuationSupply(parent.Supply):
   91      def resume(self, path: str, order: str, shared: dict) -> None:
   92          """Register a hash-verified shared checkpoint without requesting old fit groups."""
   93          if (path in self.next_stage or order not in ORDERS or shared["order"] != order
   94                  or shared["stage"] != 1 or shared["completed_updates"] != 288
   95                  or not shared.get("full_model_adam_and_rng_restore_verified")
   96                  or not shared.get("model_state_sha256")):
   97              raise ValueError("Continuation requires an intact completed shared first stage")
   98          self.next_stage[path], self.orders[path] = 2, order
   99  
  100  
  101  def update(model: Any, optimizer: Any, current: Any, history: Any, c: dict,
  102             weight: float, stage: int, step: int, current_seed: int, history_seed: int,
  103             *, observe: bool = False, check: Callable[[], None] = lambda: None) -> dict:
  104      """Only lambda changes: two supervised forwards, weighted history, one clip/update."""
  105      import torch
  106  
  107      if (type(weight) not in (int, float) or weight not in (1., .5, .25)
  108              or stage not in (2, 3) or current is None or history is None
  109              or current.labels is None or history.labels is None
  110              or history.uid == current.uid or len(optimizer.param_groups) != 2
  111              or parent.adam_step(optimizer) != (stage - 1) * 288 + step - 1):
  112          raise ValueError("Unexpected ER continuation or supervision")
  113      rates = (parent.stage_lr(step), .001)
  114      for group, rate in zip(optimizer.param_groups, rates, strict=True):
  115          group["lr"] = rate
  116      model.train()
  117      optimizer.zero_grad(set_to_none=True)
  118      before = {name: core.state_digest(module.state_dict()) for name, module in
  119                (("encoder", model.encoder), ("head", model.head))} if observe else {}
  120      result = {name: 0. for name in STEP_COLUMNS}
  121      for role, group, seed, coefficient in (("current", current, current_seed, 1.),
  122                                             ("history", history, history_seed, weight)):
  123          torch.manual_seed(seed)
  124          predicted = base.logits(model, group, c, "split_rank", check)
  125          truth = torch.tensor(group.labels, dtype=torch.float32, device=predicted.device)
  126          terms = parent.ranking.objectives(predicted, truth, .5)
  127          # Keep the original current backward operation exactly; lambda applies to ALL old terms.
  128          objective = terms["total"] if role == "current" else coefficient * terms["total"]
  129          objective.backward()
  130          result.update({role + "_" + key: float(value.detach()) for key, value in terms.items()})
  131          del predicted, truth, terms, objective
  132      result["weighted_history_total"] = weight * result["history_total"]
  133      result["total"] = result["current_total"] + result["weighted_history_total"]
  134      evidence = {}
  135      if observe:
  136          for name, module in (("encoder", model.encoder), ("head", model.head)):
  137              norms = [float(p.grad.float().norm()) for p in module.parameters() if p.grad is not None]
  138              if not norms or not np.isfinite(norms).all() or max(norms) <= 0:
  139                  raise ValueError("No finite task gradient in " + name)
  140              evidence[name] = {"finite_nonzero_combined_gradient": True}
  141      result["gradient_norm"] = float(torch.nn.utils.clip_grad_norm_(
  142          model.parameters(), c["optimizer"]["clip_norm"], error_if_nonfinite=True))
  143      check()
  144      optimizer.step()
  145      if parent.adam_step(optimizer) != (stage - 1) * 288 + step:
  146          raise ValueError("Adam did not make exactly one update")
  147      if observe:
  148          for name, module in (("encoder", model.encoder), ("head", model.head)):
  149              changed = before[name] != core.state_digest(module.state_dict())
  150              if changed != (name == "head" or rates[0] > 0):
  151                  raise ValueError("Unexpected parameter update: " + name)
  152              evidence[name]["parameters_changed"] = changed
  153      result.update(history_weight=weight, encoder_lr=rates[0], head_lr=rates[1],
  154                    modules=evidence, stage=stage, step=step, adam_step=parent.adam_step(optimizer))
  155      return result
