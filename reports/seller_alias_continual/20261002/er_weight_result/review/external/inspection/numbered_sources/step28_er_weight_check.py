    1  """Linux CPU handwritten verification; no formal texts, labels or trained states."""
    2  from __future__ import annotations
    3  
    4  import argparse
    5  import gc
    6  import os
    7  from pathlib import Path
    8  import platform
    9  import resource
   10  import sys
   11  import time
   12  import unittest
   13  from unittest import mock
   14  
   15  import numpy as np
   16  
   17  import step28_er_weight as method
   18  
   19  
   20  def native(arm: str, fixtures: object, check: object) -> tuple[dict, dict]:
   21      import torch
   22  
   23      started = time.monotonic()
   24      c = method.config(method.contract())
   25      model = method.base.load_model(c, "split_rank", "cpu")
   26      optimizer = method.core.make_optimizer(model, c)
   27      old, current = fixtures.handmade_group("native_old", 1), fixtures.handmade_group("native_current", 2)
   28      initial = method.core.state_digest(model.state_dict())
   29      covered = [p for group in optimizer.param_groups for p in group["params"]]
   30      if len({id(p) for p in covered}) != len(covered) or {id(p) for p in covered} != {id(p) for p in model.parameters()}:
   31          raise ValueError("Native optimizer coverage differs")
   32      warm = method.parent.update(model, optimizer, old, None, None, c, "seq", 1, 1, 81, 82,
   33                                  observe=True, check=check)
   34      after_warm = method.core.state_digest(model.state_dict())
   35      # An explicit handmade counter fixture, never a claim of 288 native warm updates.
   36      for state in optimizer.state.values():
   37          state["step"].fill_(288)
   38      before_optimizer = method.core.state_digest(optimizer.state_dict())
   39      probes = {name: parameter for name, parameter in model.named_parameters()
   40                if ".attention.self.query.weight" in name or ".embeddings.word_embeddings.weight" in name
   41                or name.startswith("head.")}
   42      layers = model.encoder[0].auto_model.config.num_hidden_layers
   43      if sum(".attention.self.query.weight" in name for name in probes) != layers:
   44          raise ValueError("Native layer probes incomplete")
   45      before = {name: method.core.state_digest(p) for name, p in probes.items()}
   46      component_parameters = {
   47          "encoder_first_query": model.encoder[0].auto_model.encoder.layer[0].attention.self.query.weight,
   48          "head_hidden": model.head[0].weight}
   49      components = {name: [] for name in component_parameters}
   50      hooks = [parameter.register_hook(lambda grad, name=name: components[name].append(grad.detach().clone()))
   51               for name, parameter in component_parameters.items()]
   52      captured = []
   53      hooks.append(model.head.register_forward_hook(lambda _, __, output: captured.append(output.detach().flatten().numpy().copy())))
   54      try:
   55          update = method.update(model, optimizer, current, old, c, method.ARMS[arm], 2, 1, 107, 108,
   56                                 observe=True, check=check)
   57      finally:
   58          for hook in hooks:
   59              hook.remove()
   60      if len(captured) != 2 or any(len(values) != 2 for values in components.values()):
   61          raise ValueError("Need separate current/history native forwards and gradient contributions")
   62      independent = {}
   63      for role, values, group in (("current", captured[0], current), ("history", captured[1], old)):
   64          independent[role] = fixtures.scalar_losses(values, group.labels)
   65          for term, value in independent[role].items():
   66              if abs(update[role + "_" + term] - value) > (4e-6 if term == "total" else 2e-6):
   67                  raise ValueError("Native objective differs from scalar reference")
   68      expected_total = independent["current"]["total"] + method.ARMS[arm] * independent["history"]["total"]
   69      if abs(update["total"] - expected_total) > 6e-6:
   70          raise ValueError("Native history weight does not affect the full objective")
   71      rows = []
   72      for name, parameter in probes.items():
   73          norm = float(parameter.grad.norm()) if parameter.grad is not None else 0.
   74          changed = before[name] != method.core.state_digest(parameter)
   75          if not np.isfinite(norm) or norm <= 0 or not changed:
   76              raise ValueError("Native parameter was not updated: " + name)
   77          rows.append({"name": name, "gradient_norm_after_clip": norm, "parameters_changed": changed})
   78      for name, contributions in components.items():
   79          if not all(torch.isfinite(g).all() and float(g.norm()) > 0 for g in contributions):
   80              raise ValueError("A native loss contribution misses " + name)
   81      unused = [name for name, p in model.named_parameters() if p.grad is None]
   82      if any(".pooler." not in name for name in unused) or method.parent.adam_step(optimizer) != 289:
   83          raise ValueError("Unexpected native optimizer coverage or step")
   84      result = {"arm": arm, "history_weight": method.ARMS[arm], "native_updates_actually_executed": 2,
   85                "counter_fixture": "One real warm update, then Adam counters 1->288; no native full-stage continuity claim",
   86                "initial_model_state_sha256": initial, "after_warm_model_state_sha256": after_warm,
   87                "before_stage2_optimizer_sha256": before_optimizer,
   88                "after_model_state_sha256": method.core.state_digest(model.state_dict()),
   89                "after_optimizer_sha256": method.core.state_digest(optimizer.state_dict()),
   90                "warm_update": warm, "weighted_update": update,
   91                "captured_current_logits": captured[0].tolist(), "captured_history_logits": captured[1].tolist(),
   92                "independent_objectives": independent, "independent_total": expected_total,
   93                "gradient_components": {name: {role: {"norm": float(g.norm()), "sha256": method.core.state_digest(g)}
   94                                                for role, g in zip(("current", "weighted_history"), values, strict=True)}
   95                                        for name, values in components.items()},
   96                "parameter_probes": rows, "unused_parameters": unused, "backbone_layers": layers,
   97                "seconds": time.monotonic() - started}
   98      del model, optimizer, covered, probes, component_parameters
   99      gc.collect()
  100      return result, components
  101  
  102  
  103  def run(destination: Path) -> dict:
  104      import torch
  105  
  106      if platform.system() != "Linux" or os.environ.get("CUDA_VISIBLE_DEVICES") != "" or destination.exists():
  107          raise ValueError("New Linux CPU evidence only; CUDA_VISIBLE_DEVICES must be empty")
  108      p, source_files = method.contract(), method.sources()
  109      torch.set_num_threads(1)
  110      os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})
  111      torch.use_deterministic_algorithms(True)
  112      destination.parent.mkdir(parents=True, exist_ok=True)
  113      started = time.monotonic()
  114  
  115      def check() -> None:
  116          if time.monotonic() - started >= p["cpu_verification"]["maximum_seconds"]:
  117              raise RuntimeError("CPU verification time limit reached")
  118          if sum(f.stat().st_size for f in destination.parent.rglob("*") if f.is_file()) > p["cpu_verification"]["maximum_evidence_bytes"]:
  119              raise RuntimeError("CPU evidence budget exceeded")
  120  
  121      sys.path.insert(0, str(method.data.ROOT / "tests"))
  122      import test_step28_er_weight_contracts as tests
  123      import test_step28_bge_continual_contracts as fixtures
  124      try:
  125          with mock.patch.object(method.base.public, "public_inputs", side_effect=AssertionError("No formal texts")), \
  126               mock.patch.object(method.base.public, "attach_labels", side_effect=AssertionError("No formal labels")), \
  127               mock.patch.object(method.data, "Archive", side_effect=AssertionError("No formal archive")):
  128              tested = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromModule(tests))
  129              contracts = {"passed": tested.testsRun - len(tested.failures) - len(tested.errors) - len(tested.skipped),
  130                           "failed": len(tested.failures) + len(tested.errors), "skipped": len(tested.skipped)}
  131              method.data.write_json(destination.parent / "contracts.json", contracts)
  132              if not tested.wasSuccessful() or tested.skipped:
  133                  raise RuntimeError("Affected contracts must pass without skips")
  134              check()
  135              c = method.config(p)
  136              archive = method.core.model_files(method.base.model_config(c, "split_rank"))
  137              if any(archive[key] != c["models"]["split_rank"][key]
  138                     for key in ("file_count", "total_size_bytes", "content_sha256")):
  139                  raise ValueError("Pretrained archive differs")
  140              reports, components = {}, {}
  141              for arm in method.ARMS:
  142                  print(method.data.json_bytes({"event": "native_start", "arm": arm}).decode(), flush=True)
  143                  reports[arm], components[arm] = native(arm, fixtures, check)
  144                  method.data.write_json(destination.parent / (arm + ".json"), reports[arm])
  145                  print(method.data.json_bytes({"event": "native_end", "arm": arm, "seconds": reports[arm]["seconds"]}).decode(), flush=True)
  146              for field in ("initial_model_state_sha256", "after_warm_model_state_sha256",
  147                            "before_stage2_optimizer_sha256", "captured_current_logits", "captured_history_logits"):
  148                  if reports["half"][field] != reports["quarter"][field]:
  149                      raise ValueError("Native current/history states are not paired: " + field)
  150              scaling = {}
  151              for name in components["half"]:
  152                  half_current, half_history = components["half"][name]
  153                  quarter_current, quarter_history = components["quarter"][name]
  154                  torch.testing.assert_close(half_current, quarter_current, rtol=0, atol=0)
  155                  torch.testing.assert_close(half_history, 2 * quarter_history, rtol=1e-6, atol=1e-8)
  156                  scaling[name] = {"current_exactly_equal": True,
  157                                   "half_history_equals_twice_quarter": True,
  158                                   "maximum_history_difference": float((half_history - 2 * quarter_history).abs().max())}
  159              if method.sources() != source_files:
  160                  raise ValueError("Sources changed during verification")
  161              result = {"status": "PASS_ER_WEIGHT_HANDMADE_CPU", "source_files": source_files,
  162                        "contracts": contracts, "native": {arm: method.data.record(destination.parent / (arm + ".json"), destination.parent)
  163                                                           for arm in method.ARMS},
  164                        "native_updates_actually_executed": 4, "native_history_gradient_scaling_verified": True,
  165                        "native_component_comparison": scaling, "formal_inputs": False, "formal_labels": False,
  166                        "formal_updates": 0, "gpu": False, "retained_native_weights": 0, "archive": archive,
  167                        "seconds": time.monotonic() - started, "max_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
  168                        "environment": {"python": platform.python_version(), "torch": torch.__version__, "numpy": np.__version__,
  169                                        "cpu_affinity": sorted(os.sched_getaffinity(0))},
  170                        "limitations": "Handwritten groups and tiny/native CPU checks; no formal training, CUDA/BF16 or native 288-step continuity claim"}
  171              method.data.write_json(destination, result)
  172              check()
  173              return result
  174      except Exception as error:
  175          method.data.write_json(destination.parent / "failure.json", {
  176              "status": "ER_WEIGHT_CPU_FAILED", "error": str(error), "seconds": time.monotonic() - started,
  177              "formal_inputs": False, "formal_labels": False})
  178          raise
  179  
  180  
  181  if __name__ == "__main__":
  182      parser = argparse.ArgumentParser(description=__doc__)
  183      parser.add_argument("--out", type=Path, required=True)
  184      args = parser.parse_args()
  185      result = run(args.out.resolve())
  186      print(method.data.json_bytes({key: result[key] for key in ("status", "native_updates_actually_executed", "seconds")}).decode())
